#!/usr/bin/env python3
# InstaOSNIT.py

import os, sys, re, json, time, random, argparse, csv, logging
from uuid import uuid4
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import requests

# Optional imports
try:
    from instaloader import Instaloader, Profile
    HAS_INSTALOADER = True
except Exception:
    HAS_INSTALOADER = False

try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except Exception:
    HAS_TEXTBLOB = False

try:
    import phonenumbers
    from phonenumbers.phonenumberutil import region_code_for_country_code
    import pycountry
    HAS_PHONE = True
except Exception:
    HAS_PHONE = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False
    nlp = None

try:
    from timezonefinder import TimezoneFinder
    HAS_TZF = True
except Exception:
    HAS_TZF = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import pytz
    HAS_PYTZ = True
except Exception:
    HAS_PYTZ = False

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
except Exception:
    STOP_WORDS = set()

DEFAULT_CONFIG = {
    "backoff_base": 2,
    "backoff_max": 300,
    "sticky_seconds": 600,
    "ua_pool": [
        "Instagram 289.0.0.18.67 (iPhone13,4; iOS 16_4; en_US)",
        "Instagram 273.0.0.14.68 Android (29/10; 480dpi; 1080x2400; Samsung; SM-G991B)",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
    ],
    "proxies_file": "proxies.txt",
    "instaloader_session_dir": "instaloader_session",
    "persist_dir": "persist",
    "log_file": "instaosnit.log",
}

WEB_PROFILE_INFO = "https://i.instagram.com/api/v1/users/web_profile_info/?username={username}"
USER_INFO = "https://i.instagram.com/api/v1/users/{user_id}/info/"
USER_FEED = "https://i.instagram.com/api/v1/feed/user/{user_id}/?count={count}"
LOOKUP = "https://i.instagram.com/api/v1/users/lookup/"
USERNAME_INFO = "https://i.instagram.com/api/v1/users/{username}/usernameinfo/"
WWW_PROFILE_A1 = "https://www.instagram.com/{username}/?__a=1&__d=dis"
SELF_INFO_URL = "https://i.instagram.com/api/v1/accounts/current_user/?edit=true"

UA_POOL = DEFAULT_CONFIG["ua_pool"]

def setup_logging(path: str) -> None:
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def load_proxies(path: Optional[str]) -> List[Dict[str, str]]:
    out = []
    if not path or not os.path.exists(path): return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    out.append({"http": ln, "https": ln})
    except (OSError, IOError) as e:
        logging.error(f"Proxy file load error: {e}")
    return out

class StickyProxyPool:
    def __init__(self, proxies: Optional[List[Dict[str,str]]], sticky_seconds: int):
        self.proxies = proxies or []
        self.sticky_seconds = sticky_seconds
        self.current = None  # FIX: Corrected indentation on this line
        self.until = 0
        self.health = {id(p): {"bad":0} for p in self.proxies}

    def get(self) -> Optional[Dict[str,str]]:
        now = time.time()
        if self.current and now < self.until:
            return self.current
        if not self.proxies:
            self.current = None
            return None
        self.current = random.choice(self.proxies)
        self.until = now + self.sticky_seconds
        return self.current

    def mark_bad(self, proxy: Optional[Dict[str,str]]) -> None:
        if not proxy: return
        pid = id(proxy)
        self.health.setdefault(pid, {"bad":0})
        self.health[pid]["bad"] += 1
        logging.warning(f"Proxy marked bad ({self.health[pid]['bad']}): {proxy}")
        if self.health[pid]["bad"] > 3:
            try:
                self.proxies.remove(proxy)
                logging.info(f"Proxy removed: {proxy}")
            except ValueError:
                pass

def pick_headers(app_ids: Optional[List[str]] = None) -> Dict[str, str]:
    app_ids = app_ids or ["936619743392459", "124024574287414", "567067343352427"]
    return {
        "User-Agent": random.choice(UA_POOL),
        "X-IG-App-ID": random.choice(app_ids),
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate"
    }

def gen_device_id(seed: Optional[str] = None) -> str:
    import hashlib
    base = seed or str(uuid4())
    return hashlib.md5(base.encode()).hexdigest()

def build_mobile_headers() -> Dict[str,str]:
    h = pick_headers()
    h.update({
        "X-Ig-Device-Id": gen_device_id(),
        "X-Ig-Android-Id": gen_device_id("android"),
        "X-Ig-Connection-Type": "WIFI"
    })
    return h

def human_delay(base: float = 1.5, jitter: float = 0.6) -> None:
    time.sleep(max(0.05, random.uniform(base - jitter, base + jitter)))

def backoff_sleep(attempt: int, base: int = DEFAULT_CONFIG["backoff_base"], max_wait: int = DEFAULT_CONFIG["backoff_max"]) -> None:
    time.sleep(min(max_wait, base ** attempt))

def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except json.JSONDecodeError:
        logging.warning("JSON decode error on response")
        return {}
    except Exception as e:
        logging.error(f"Unexpected JSON parse error: {e}")
        return {}

def detect_checkpoint(resp: requests.Response) -> Tuple[bool, Dict[str, Any]]:
    j = safe_json(resp)
    if j.get("checkpoint_url") or j.get("challenge") or j.get("checkpoint_required") or j.get("challenge_required"):
        return True, j
    if resp.status_code in (400, 403) and ("challenge" in (resp.text or "").lower()):
        return True, j
    return False, j

class Requester:
    def __init__(self, proxy_pool: StickyProxyPool):
        self.pool = proxy_pool

    def request(self, method: str, url: str, headers: Optional[Dict[str,str]] = None, cookies: Optional[Dict[str,str]] = None,
                timeout: int = 20, data: Any = None, stream: bool = False, max_attempts: int = 6,
                checkpoint_sensitive: bool = False) -> Optional[requests.Response]:
        attempt = 0
        while attempt < max_attempts:
            proxy = self.pool.get()
            human_delay()
            try:
                r = requests.request(method, url, headers=headers or {}, cookies=cookies, timeout=timeout, data=data, stream=stream, allow_redirects=True, proxies=proxy)
                if checkpoint_sensitive:
                    flagged, _ = detect_checkpoint(r)
                    if flagged:
                        logging.error("Checkpoint detected; manual verification required.")
                        return r
                if r.status_code in (429, 403):
                    logging.warning(f"HTTP {r.status_code} for {url}")
                    self.pool.mark_bad(proxy)
                    backoff_sleep(attempt); attempt += 1; continue
                return r
            except requests.exceptions.Timeout:
                logging.warning(f"Timeout for {url}")
                self.pool.mark_bad(proxy); backoff_sleep(attempt); attempt += 1
            except requests.exceptions.ConnectionError:
                logging.warning(f"Connection error for {url}")
                self.pool.mark_bad(proxy); backoff_sleep(attempt); attempt += 1
            except Exception as e:
                logging.error(f"Unexpected request error: {e}")
                self.pool.mark_bad(proxy); backoff_sleep(attempt); attempt += 1
        logging.error(f"Max attempts exceeded for {url}")
        return None

def check_session_validity(req: Requester, sessionid: str) -> bool:
    headers = pick_headers()
    r = req.request("GET", SELF_INFO_URL, headers=headers, cookies={'sessionid': sessionid}, timeout=12)
    return bool(r and r.status_code == 200)

def get_user_web_profile(req: Requester, username: str, sessionid: Optional[str]) -> Dict[str, Any]:
    url = WEB_PROFILE_INFO.format(username=username)
    headers = pick_headers()
    cookies = {'sessionid': sessionid} if sessionid else None
    r = req.request("GET", url, headers=headers, cookies=cookies, timeout=15)
    if not r: return {"error": "network"}
    if r.status_code == 404: return {"error": "not_found", "raw": safe_json(r)}
    return {"raw": safe_json(r), "status_code": r.status_code}

def get_user_info_private(req: Requester, user_id: str, sessionid: str) -> Dict[str, Any]:
    url = USER_INFO.format(user_id=user_id)
    headers = pick_headers()
    r = req.request("GET", url, headers=headers, cookies={'sessionid': sessionid}, timeout=15)
    if not r: return {"error": "network"}
    if r.status_code == 429: return {"error": "rate_limit", "raw": safe_json(r)}
    return {"raw": safe_json(r), "status_code": r.status_code}

def get_feed_media(req: Requester, user_id: str, sessionid: str, count: int = 50) -> Dict[str, Any]:
    url = USER_FEED.format(user_id=user_id, count=count)
    headers = pick_headers()
    r = req.request("GET", url, headers=headers, cookies={'sessionid': sessionid}, timeout=20)
    if not r: return {"error": "network"}
    if r.status_code == 429: return {"error": "rate_limit", "raw": safe_json(r)}
    return {"raw": safe_json(r), "status_code": r.status_code}

def do_advanced_lookup(req: Requester, username: str, sessionid: Optional[str]) -> Dict[str, Any]:
    data = "signed_body=SIGNATURE." + json.dumps({"q": username, "skip_recovery": "1"}, separators=(",", ":"))
    headers = pick_headers()
    r = req.request("POST", LOOKUP, headers=headers, data=data, cookies={'sessionid': sessionid} if sessionid else None, timeout=15)
    if not r: return {"error": "network"}
    try:
        return {"raw": r.json(), "status_code": r.status_code}
    except json.JSONDecodeError:
        return {"raw_text": r.text, "status_code": r.status_code}

def get_usernameinfo(req: Requester, username: str, sessionid: Optional[str]) -> Dict[str, Any]:
    url = USERNAME_INFO.format(username=username)
    headers = pick_headers()
    r = req.request("GET", url, headers=headers, cookies={'sessionid': sessionid} if sessionid else None, timeout=15)
    if not r: return {"error": "network"}
    return {"status_code": r.status_code, "raw": safe_json(r), "text": r.text}

def get_www_profile_a1(req: Requester, username: str) -> Dict[str, Any]:
    url = WWW_PROFILE_A1.format(username=username)
    headers = pick_headers(["936619743392459"])
    r = req.request("GET", url, headers=headers, timeout=15)
    if not r: return {"error": "network"}
    return {"status_code": r.status_code, "raw": safe_json(r), "text": r.text}

def resolve_user_id(req: Requester, username: str, sessionid: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    try:
        res = get_user_web_profile(req, username, sessionid)
        raw = res.get("raw")
        if isinstance(raw, dict):
            u = raw.get("data", {}).get("user")
            if isinstance(u, dict):
                pk = u.get("id") or u.get("pk")
                if pk: return str(pk), "web_profile_info", raw
    except requests.exceptions.RequestException as e:
        logging.error(f"resolve_user_id web_profile_info error: {e}")
    try:
        res = do_advanced_lookup(req, username, sessionid)
        raw = res.get("raw") or {}
        if isinstance(raw, dict):
            user_node = raw.get("user")
            if not user_node:
                for k in ("users", "user", "data"):
                    if raw.get(k):
                        maybe = raw.get(k)
                        if isinstance(maybe, list) and maybe: user_node = maybe[0]
                        elif isinstance(maybe, dict): user_node = maybe
            if isinstance(user_node, dict):
                pk = user_node.get("pk") or user_node.get("id")
                if pk: return str(pk), "users/lookup", raw
    except requests.exceptions.RequestException as e:
        logging.error(f"resolve_user_id lookup error: {e}")
    try:
        res = get_usernameinfo(req, username, sessionid)
        raw = res.get("raw")
        if isinstance(raw, dict):
            user_node = raw.get("user") or raw
            if isinstance(user_node, dict):
                pk = user_node.get("pk") or user_node.get("id")
                if pk: return str(pk), "usernameinfo", raw
    except requests.exceptions.RequestException as e:
        logging.error(f"resolve_user_id usernameinfo error: {e}")
    try:
        res = get_www_profile_a1(req, username)
        raw = res.get("raw")
        if isinstance(raw, dict):
            g = raw.get("graphql") or raw.get("data") or raw
            if isinstance(g, dict):
                u = g.get("user") or g.get("profile") or g
                if isinstance(u, dict):
                    pk = u.get("id") or u.get("pk")
                    if pk: return str(pk), "www_profile_a1", raw
    except requests.exceptions.RequestException as e:
        logging.error(f"resolve_user_id www_profile_a1 error: {e}")
    return None, None, None

def extract_contact_info(text: Optional[str]) -> Dict[str, str]:
    info = {'phone_in_text': 'N/A', 'email_in_text': 'N/A'}
    if not text: return info
    try:
        e = re.search(r'[\w\.-]+@[\w\.-]+', text)
        if e: info['email_in_text'] = e.group(0).lower()
        p = re.search(r'(\+?\d{1,3}\s?\d{2,4}[-\s\.]?\d{2,4}[-\s\.]?\d{2,9})', text)
        if p: info['phone_in_text'] = re.sub(r'[\s\.\-\(\)]', '', p.group(0))
    except re.error as e:
        logging.error(f"Regex error in extract_contact_info: {e}")
    return info

def normalize_phone(num: Optional[str]) -> Dict[str, str]:
    if not num: return {"number": num, "country": "Unknown"}
    if not HAS_PHONE: return {"number": num, "country": "Unknown"}
    try:
        pn = phonenumbers.parse(num)
        cc = region_code_for_country_code(pn.country_code)
        return {"number": num, "country": cc}
    except Exception as e:
        logging.warning(f"Phone normalize error: {e}")
        return {"number": num, "country": "Unknown"}

def extract_keywords(texts: List[Optional[str]], top: int = 15) -> List[Tuple[str, int]]:
    words = []
    for t in texts:
        for w in re.findall(r'\b\w+\b', (t or "").lower()):
            if w not in STOP_WORDS and len(w) > 3:
                words.append(w)
    return Counter(words).most_common(top)

def targeted_keyword_search(texts: List[Optional[str]], terms: List[str]) -> Dict[str, int]:
    if not terms: return {}
    hits = {}
    ft = "\n".join(t for t in texts if t)
    for term in terms:
        try:
            m = re.findall(term, ft, re.IGNORECASE)
            if m: hits[term] = len(m)
        except re.error as e:
            logging.error(f"Regex error in targeted_keyword_search for term {term}: {e}")
            continue
    return hits

def exif_from_url_native(req: Requester, url: Optional[str], outfile: str = "tmp_instaosnit.jpg") -> Dict[str, Any]:
    if not HAS_PIL or not url: return {"error": "PIL or URL unavailable"}
    r = req.request("GET", url, timeout=15, stream=True)
    if not r: return {"error": "network"}
    try:
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(1024): f.write(chunk)
        img = Image.open(outfile)
        exif = img.getexif()
        data = {ExifTags.TAGS.get(tag, tag): val for tag, val in exif.items()}
        try: os.remove(outfile)
        except (OSError, IOError):
            pass
        return data
    except Exception as e:
        logging.error(f"EXIF extraction error: {e}")
        return {"error": f"exif: {e}"}

def infer_locations_from_feed(feed_raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(feed_raw, dict): return {"locations": [], "last_post_ts": None}
    items = feed_raw.get("items") or []
    locations = []
    last_post_ts = None
    for it in items:
        try:
            loc = it.get("location")
            if loc:
                locations.append({"name": loc.get("name"), "address": loc.get("address", ""), "pk": loc.get("pk") or loc.get("id"), "lat": loc.get("lat"), "lng": loc.get("lng")})
            taken = it.get("taken_at") or it.get("device_timestamp") or it.get("timestamp")
            if taken:
                ts = int(taken)
                if ts and (not last_post_ts or ts > last_post_ts): last_post_ts = ts
        except Exception as e:
            logging.warning(f"Feed location parse error: {e}")
            continue
    uniq, seen = [], set()
    for l in locations:
        key = (l.get("pk"), l.get("name"))
        if key not in seen:
            seen.add(key); uniq.append(l)
    return {"locations": uniq, "last_post_ts": last_post_ts}

def load_profile_instaloader(username: str, session_dir: str, login_user: Optional[str]=None, password: Optional[str]=None) -> Tuple[Optional[Instaloader], Optional[Profile], Optional[str]]:
    if not HAS_INSTALOADER: return None, None, None
    L = Instaloader(dirname_pattern=session_dir, download_pictures=False, download_videos=False, download_comments=False, save_metadata=False, quiet=True, filename_pattern="{date_utc}__{id}")
    sessionid: Optional[str] = None
    try:
        if login_user and password:
            os.makedirs(session_dir, exist_ok=True)
            session_file = os.path.join(session_dir, f"{login_user}.json")
            if os.path.exists(session_file):
                try:
                    L.load_session_from_file(login_user, session_file)
                    logging.info(f"Instaloader session loaded: {session_file}")
                except Exception as e:
                    logging.warning(f"Load session failed, trying login: {e}")
                    L.context.login(login_user, password)
                    L.save_session_to_file(session_file)
                    logging.info(f"Instaloader session saved: {session_file}")
            else:
                L.context.login(login_user, password)
                L.save_session_to_file(session_file)
                logging.info(f"Instaloader session saved: {session_file}")
            try:
                sessionid = L.context._session.cookies.get('sessionid') or L.context.session.cookies.get('sessionid')
            except Exception:
                sessionid = None
    except Exception as e:
        logging.error(f"Instaloader login error for {login_user}: {e}")
    try:
        p = Profile.from_username(L.context, username)
        return L, p, sessionid
    except Exception as e:
        logging.warning(f"Profile load error: {e}")
        return L, None, sessionid

def analyze_profile_location_from_profile(p: Optional[Profile], n: int = 200) -> Dict[str, Any]:
    res = {'profile_exists': bool(p), 'profile_data': {}, 'most_frequent_location': None, 'all_locations': [], 'all_coords': []}
    if not p: return res
    res['profile_data'] = {
        'bio': p.biography,
        'external_url': p.external_url or "N/A",
        'is_business': p.is_business_account,
        'public_email': p.public_email or "N/A",
        'public_phone': p.public_phone_number or "N/A",
        'business_address': p.business_address_json or "N/A",
        'business_category': p.business_category_name or "N/A"
    }
    c = Counter(); d = {}; mr = None; count = 0
    for post in p.get_posts():
        if count >= n: break
        count += 1
        if post.location:
            l = post.location; ln = (l.name or "").strip()
            try:
                lat = getattr(l, "lat", None); lng = getattr(l, "lng", None)
            except Exception:
                lat = None; lng = None
            if lat is not None and lng is not None:
                res['all_coords'].append((lat, lng, post.date_utc))
            if ln:
                c[ln] += 1; det = {'name': ln, 'lat': lat, 'lng': lng}; d[ln] = det
                if not mr: mr = det; mr['post_date'] = post.date_utc
    res['all_locations'] = c.most_common()
    if c:
        mc, ct = c.most_common(1)[0]; det = d[mc]; det['count'] = ct
        res['most_frequent_location'] = det; res['most_recent_location'] = mr
    res['text_clues'] = extract_contact_info(f"{p.biography or ''} {p.external_url or ''}")
    return res

def cluster_primary_location(coords: List[Tuple[float,float]]) -> Optional[Dict[str, Any]]:
    if not HAS_SKLEARN or not coords: return None
    try:
        X = [(lat, lng) for lat, lng in coords if lat is not None and lng is not None]
        if len(X) < 3: return None
        labels = DBSCAN(eps=0.01, min_samples=2).fit_predict(X)
        clusters = defaultdict(list)
        for (lat, lng), lbl in zip(X, labels):
            if lbl != -1: clusters[lbl].append((lat, lng))
        if not clusters: return None
        largest = max(clusters.items(), key=lambda kv: len(kv[1]))[1]
        clat = sum(l for l, _ in largest) / len(largest)
        clng = sum(g for _, g in largest) / len(largest)
        return {"lat": clat, "lng": clng, "count": len(largest)}
    except Exception as e:
        logging.error(f"DBSCAN clustering error: {e}")
        return None

def infer_timezone_from_coords(lat: Optional[float], lng: Optional[float]) -> Optional[str]:
    if not HAS_TZF or lat is None or lng is None: return None
    tf = TimezoneFinder()
    try:
        return tf.timezone_at(lng=lng, lat=lat)
    except Exception as e:
        logging.warning(f"Timezone inference error: {e}")
        return None

def convert_to_local(ts: datetime, tz_name: Optional[str]) -> datetime:
    if not tz_name or not HAS_PYTZ: return ts
    try:
        tz = pytz.timezone(tz_name)
        return ts.astimezone(tz)
    except Exception as e:
        logging.warning(f"Timezone convert error: {e}")
        return ts

def analyze_post_behavior_from_profile(p: Optional[Profile], n: int = 200) -> Dict[str, Any]:
    hashtags, mentions, captions = Counter(), Counter(), []
    times = []
    if not p: return {"top_hashtags": [], "top_mentions": [], "timestamps": [], "captions": []}
    count = 0
    for post in p.get_posts():
        if count >= n: break
        count += 1
        if post.caption:
            captions.append(post.caption)
            for w in post.caption.split():
                if w.startswith("#"): hashtags[w.lower()] += 1
                if w.startswith("@"): mentions[w.lower()] += 1
        times.append(post.date_utc)
    return {"top_hashtags": hashtags.most_common(10), "top_mentions": mentions.most_common(10), "timestamps": times, "captions": captions}

def analyze_comments_from_profile(p: Optional[Profile], post_limit: int = 60, collect_texts: bool = True) -> Tuple[List[Tuple[str,int]], Dict[str, List[str]]]:
    inter = {}
    texts_by_user = defaultdict(list)
    if not p: return [], {}
    count = 0
    for post in p.get_posts():
        if count >= post_limit: break
        count += 1
        try:
            for c in post.get_comments():
                uname = c.owner.username
                inter[uname] = inter.get(uname, 0) + 1
                if collect_texts and getattr(c, "text", None):
                    texts_by_user[uname].append(c.text)
        except Exception as e:
            logging.warning(f"Comment iteration error: {e}")
            continue
    ranked = sorted(inter.items(), key=lambda x: x[1], reverse=True)
    return ranked, texts_by_user

def extract_entities_spacy(texts: List[Optional[str]]) -> List[Tuple[Tuple[str,str], int]]:
    if not HAS_SPACY or not nlp: return []
    ents = Counter()
    for t in texts:
        if not t: continue
        try:
            doc = nlp(t)
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE"):
                    ents[(ent.label_, ent.text)] += 1
        except Exception as e:
            logging.warning(f"spaCy NER error: {e}")
            continue
    return ents.most_common(50)

def ghost_followers(p: Optional[Profile], recent_posts: int = 30) -> List[str]:
    if not p: return []
    followers = {u.username for u in p.get_followers()}
    engaged = set()
    count = 0
    for post in p.get_posts():
        if count >= recent_posts: break
        count += 1
        try:
            for c in post.get_comments(): engaged.add(c.owner.username)
            try:
                for l in post.get_likes(): engaged.add(l.username)
            except Exception:
                pass
        except Exception as e:
            logging.warning(f"Ghost follower post iteration error: {e}")
            continue
    return list(followers - engaged)

def likes_network(p: Optional[Profile], posts: int = 30) -> List[Tuple[str,int]]:
    likers = Counter()
    if not p: return []
    count = 0
    for post in p.get_posts():
        if count >= posts: break
        count += 1
        try:
            for l in post.get_likes(): likers[l.username] += 1
        except Exception as e:
            logging.warning(f"Likes iteration error: {e}")
            continue
    return likers.most_common(200)

def posting_frequency_analytics(timestamps: List[datetime]) -> Dict[str, Any]:
    if not timestamps: return {"posts_per_month":"N/A","avg_gap_days":"N/A"}
    timestamps = sorted(timestamps)
    months = Counter([(ts.year, ts.month) for ts in timestamps])
    gaps = []
    for a,b in zip(timestamps, timestamps[1:]):
        gaps.append((b - a).days)
    ppm = sum(months.values())/max(1,len(months))
    avg_gap = sum(gaps)/max(1,len(gaps)) if gaps else 0
    return {"posts_per_month": round(ppm,2), "avg_gap_days": round(avg_gap,2)}

def lda_topics(texts: List[str], n_topics: int = 3, n_words: int = 8) -> List[Dict[str, Any]]:
    if not HAS_SKLEARN or not texts: return []
    texts = [t for t in texts if t and len(t.split()) >= 3]
    if not texts: return []
    vec = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    X = vec.fit_transform(texts)
    if X.shape[0] < 2: return []
    lda = LatentDirichletAllocation(n_components=min(n_topics, X.shape[0]), random_state=42)
    lda.fit(X)
    words = vec.get_feature_names_out()
    topics = []
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-n_words:][::-1]
        topics.append({"topic": idx+1, "terms": [words[i] for i in top_idx]})
    return topics

def ris_links(image_url: Optional[str]) -> Dict[str, str]:
    if not image_url: return {}
    return {
        "Google": f"https://images.google.com/searchbyimage?image_url={image_url}",
        "Yandex": f"https://yandex.com/images/search?rpt=imageview&url={image_url}",
        "TinEye": f"https://tineye.com/search?url={image_url}"
    }

def fetch_followers_bulk(L: Instaloader, usernames: List[str], per_user_limit: int = 500) -> Dict[str, List[str]]:
    out = {}
    for u in usernames:
        try:
            p = Profile.from_username(L.context, u)
            out[u] = [f.username for f in p.get_followers()][:per_user_limit]
        except Exception as e:
            logging.warning(f"Follower fetch error for {u}: {e}")
            out[u] = []
    return out

def jaccard_similarity(a: List[str], b: List[str]) -> Tuple[float, int]:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union, inter

def mutual_overlap_pairs(L: Optional[Instaloader], seed_users: List[str], per_user_limit: int = 300, top_pairs: int = 100) -> List[Tuple[str,str,float,int]]:
    if not (L and HAS_INSTALOADER) or not seed_users: return []
    followers_map = fetch_followers_bulk(L, seed_users, per_user_limit=per_user_limit)
    scores = []
    for u1, u2 in combinations(seed_users, 2):
        sim, inter = jaccard_similarity(followers_map.get(u1, []), followers_map.get(u2, []))
        if inter > 0:
            scores.append((u1, u2, sim, inter))
    scores.sort(key=lambda x: (x[3], x[2]), reverse=True)
    return scores[:top_pairs]

def cluster_posts_by_location(coords_with_ts: List[Tuple[float,float,datetime]], eps: float = 0.01, min_samples: int = 3) -> List[Dict[str, Any]]:
    if not HAS_SKLEARN or not coords_with_ts: return []
    try:
        X = [[lat, lng] for (lat, lng, _) in coords_with_ts if lat is not None and lng is not None]
        if len(X) < min_samples: return []
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        clusters = defaultdict(list)
        idx = 0
        for (lat, lng, dt) in coords_with_ts:
            if lat is None or lng is None: idx += 1; continue
            lbl = labels[idx]
            if lbl != -1:
                clusters[lbl].append((lat, lng, dt))
            idx += 1
        out = []
        for lbl, pts in clusters.items():
            clat = sum(p[0] for p in pts)/len(pts)
            clng = sum(p[1] for p in pts)/len(pts)
            out.append({"label": int(lbl), "center": (clat, clng), "points": pts})
        return out
    except Exception as e:
        logging.error(f"Cluster posts by location error: {e}")
        return []

def peak_hours_per_cluster(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    res = []
    for c in clusters:
        tz_name = infer_timezone_from_coords(c["center"][0], c["center"][1])
        hours = Counter([convert_to_local(p[2], tz_name).hour for p in c["points"]])
        res.append({"label": c["label"], "lat": c["center"][0], "lng": c["center"][1], "top_hours": hours.most_common(3), "count": len(c["points"]), "tz": tz_name})
    return res

def persist_post_ids(path: str, post_ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ids": list(post_ids), "saved_at": datetime.now(timezone.utc).isoformat()}, f)

def detect_deleted(prev_path: str, current_post_ids: List[str]) -> Tuple[List[str], List[str]]:
    try:
        with open(prev_path, "r", encoding="utf-8") as f:
            prev = json.load(f).get("ids", [])
    except (OSError, IOError, json.JSONDecodeError):
        prev = []
    prev_set, cur_set = set(prev), set(current_post_ids)
    return list(prev_set - cur_set), list(cur_set - prev_set)

def export_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

def export_csv(data: Dict[str, Any], path: str) -> None:
    flat = {
        'username': data.get('target_username'),
        'profile_exists': data.get('profile_exists'),
        'bio': data.get('profile_data', {}).get('bio', 'N/A'),
        'primary_location': (data.get('most_frequent_location') or {}).get('name', 'N/A'),
        'last_post_ts': data.get('activity_data', {}).get('last_post_ts', 'N/A'),
        'public_email': data.get('profile_data', {}).get('public_email', 'N/A'),
        'contact_email_clue': data.get('text_clues', {}).get('email_in_text', 'N/A'),
        'top_hashtag_1': (data.get('behavior_data', {}).get('top_hashtags') or [('N/A', 0)])[0][0],
        'followers_count': len(data.get('network_data', {}).get('followers', [])),
        'following_count': len(data.get('network_data', {}).get('following', [])),
        'mutuals_count': len(data.get('network_data', {}).get('mutual_followers', [])),
        'peak_hours_top3': ','.join([f"{h[0]}({h[1]})" for h in data.get('temporal_data', {}).get('top_hours', [])[:3]]),
        'peak_days_top3': ','.join([f"{d[0]}({d[1]})" for d in data.get('temporal_data', {}).get('top_days', [])[:3]]),
        'top_keyword': (data.get('keywords') or [('N/A', 0)])[0][0],
        'top_keyword_count': (data.get('keywords') or [('N/A', 0)])[0][1],
        'stories_count': data.get('stories', {}).get('count', 0),
        'entities_top': ';'.join([f"{lbl}:{txt}({ct})" for (lbl, txt), ct in data.get('entities', [])[:10]])
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader(); w.writerow(flat)

def export_gexf(data: Dict[str, Any], path: str) -> None:
    if not HAS_NETWORKX: raise RuntimeError("NetworkX required")
    G = nx.DiGraph()
    target = data.get('target_username')
    pd = data.get('profile_data', {})
    mutuals = data.get('network_data', {}).get('mutual_followers', []) or []
    commenters = data.get('commenters', []) or []
    likes_net = data.get('likes_network', []) or []
    max_c = max([c for _, c in commenters], default=1)
    G.add_node(target, type='Target', label=target, size=10.0, color='#FF0000',
               is_business=str(pd.get('is_business')),
               email_present=str(pd.get('public_email') not in (None, "N/A")))
    for m in mutuals:
        G.add_node(m, type='Mutual Follower', label=m, size=5.0, color='#00CC00')
        G.add_edge(target, m, type='Follows', weight=1.0)
        G.add_edge(m, target, type='Follows', weight=1.0)
    for u, c in commenters[:500]:
        w = float(c) / float(max_c)
        if u not in G: G.add_node(u, type='Commenter', label=u, size=4.0, color='#0066FF', comment_count=float(c))
        G.add_edge(u, target, type='Comment', weight=w)
    for u, c in likes_net[:500]:
        if u not in G: G.add_node(u, type='Liker', label=u, size=3.5, color='#00AACC', like_count=float(c))
        G.add_edge(u, target, type='Like', weight=float(c))
    overlaps = data.get('overlap_pairs', []) or []
    for u1, u2, sim, inter in overlaps[:300]:
        if u1 not in G: G.add_node(u1, type='User', label=u1, size=3.0, color='#888888')
        if u2 not in G: G.add_node(u2, type='User', label=u2, size=3.0, color='#888888')
        G.add_edge(u1, u2, type='FollowerOverlap', weight=float(sim), overlap_count=int(inter))
    ew = data.get('engagement_windows', {}) or {}
    for u, meta in ew.items():
        if u in G.nodes:
            nx.set_node_attributes(G, {u: {"active_first": meta.get("first"), "active_last": meta.get("last"), "cmt": meta.get("comments"), "like": meta.get("likes")}})
    nx.write_gexf(G, path)

def export_pdf(data: Dict[str, Any], path: str) -> None:
    if not HAS_REPORTLAB: raise RuntimeError("ReportLab required")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path)
    elems = []
    title = Paragraph(f"Forensic OSINT Report: {data.get('target_username')}", styles['Title'])
    elems.append(title); elems.append(Spacer(1, 12))
    pd = data.get('profile_data', {})
    elems.append(Paragraph("Forensic Persistence Analysis", styles['Heading2']))
    elems.append(Paragraph(f"<b>Deleted Posts (since last run):</b> {len(data.get('deleted_since_last_run', []))} posts", styles['Normal']))
    elems.append(Paragraph(f"<b>Added Posts (since last run):</b> {len(data.get('added_since_last_run', []))} posts", styles['Normal']))
    elems.append(Spacer(1, 12))
    target_table = Table([
        ["Username", data.get('target_username') or "N/A"],
        ["Business", str(pd.get('is_business'))],
        ["Public email", pd.get('public_email') or "N/A"],
        ["Public phone", pd.get('public_phone') or "N/A"],
        ["External URL", pd.get('external_url') or "N/A"],
        ["Business category", pd.get('business_category') or "N/A"]
    ], hAlign='LEFT')
    target_table.setStyle(TableStyle([('BACKGROUND',(0,0),(1,0),colors.whitesmoke),('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    elems.append(Paragraph("Target information", styles['Heading2'])); elems.append(target_table); elems.append(Spacer(1, 12))
    tc = data.get('text_clues', {})
    elems.append(Paragraph("Contact clues", styles['Heading2']))
    elems.append(Paragraph(f"Email in text: {tc.get('email_in_text','N/A')}", styles['Normal']))
    elems.append(Paragraph(f"Phone in text: {tc.get('phone_in_text','N/A')} (normalized: {tc.get('phone_clue_norm')})", styles['Normal']))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("Top hashtags", styles['Heading2']))
    elems.append(Paragraph(", ".join([h for h,_ in data.get('behavior_data',{}).get('top_hashtags',[])[:10]]) or "N/A", styles['Normal']))
    elems.append(Paragraph("Top keywords", styles['Heading2']))
    elems.append(Paragraph(", ".join([k for k,_ in data.get('keywords',[])[:15]]) or "N/A", styles['Normal']))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("Temporal peaks", styles['Heading2']))
    th = data.get('temporal_data',{}).get('top_hours',[])[:5]
    td = data.get('temporal_data',{}).get('top_days',[])[:5]
    elems.append(Paragraph("Peak hours: " + ", ".join([f"{h}:00({c})" for h,c in th]) if th else "N/A", styles['Normal']))
    elems.append(Paragraph("Peak days: " + ", ".join([f"{d}({c})" for d,c in td]) if td else "N/A", styles['Normal']))
    ctp = data.get('cluster_temporal', [])
    if ctp:
        elems.append(Spacer(1, 12))
        elems.append(Paragraph("Location clusters (local time peaks)", styles['Heading2']))
        rows = [["Label","Lat","Lng","TZ","Top hours","Count"]]
        for cl in ctp[:10]:
            rows.append([str(cl.get('label')), f"{cl.get('lat'):.5f}", f"{cl.get('lng'):.5f}", cl.get('tz') or "N/A",
                         ", ".join([f"{h}:00({c})" for h,c in cl.get('top_hours',[])]), str(cl.get('count'))])
        t = Table(rows, hAlign='LEFT')
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.whitesmoke)]))
        elems.append(t)
    doc.build(elems)

def engagement_windows(p: Optional[Profile], posts: int=60) -> Dict[str,Dict[str,Any]]:
    if not p: return {}
    windows: Dict[str,List[Any]] = defaultdict(lambda: [None, None, 0, 0])
    count = 0
    for post in p.get_posts():
        if count >= posts: break
        count += 1
        ts = post.date_utc
        try:
            for c in post.get_comments():
                u = c.owner.username
                w = windows[u]
                w[0] = ts if w[0] is None or ts < w[0] else w[0]
                w[1] = ts if w[1] is None or ts > w[1] else w[1]
                w[2] += 1
        except Exception:
            pass
        try:
            for l in post.get_likes():
                u = l.username
                w = windows[u]
                w[0] = ts if w[0] is None or ts < w[0] else w[0]
                w[1] = ts if w[1] is None or ts > w[1] else w[1]
                w[3] += 1
        except Exception:
            pass
    return {u: {"first": w[0].isoformat() if w[0] else None, "last": w[1].isoformat() if w[1] else None,
                "comments": w[2], "likes": w[3]} for u, w in windows.items()}

def build_analysis(req: Requester, username: str, sessionid: Optional[str], post_limit: int, persist_dir: str,
                   deep_network: bool, cluster_temporal: bool, login_user: Optional[str], login_pass: Optional[str]) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {"target_username": username}
    L, p, sessionid_iloader = load_profile_instaloader(username, DEFAULT_CONFIG["instaloader_session_dir"], login_user, login_pass)
    if sessionid_iloader:
        sessionid = sessionid_iloader
        logging.info("Using Instaloader-derived sessionid for API calls.")
    uid, source, raw_used = resolve_user_id(req, username, sessionid)
    analysis['resolved_id'] = uid; analysis['id_source'] = source
    user_info = get_user_info_private(req, uid, sessionid) if uid else {}
    user_raw = user_info.get("raw") or {}
    analysis['raw_user'] = user_raw
    if p:
        prof_res = analyze_profile_location_from_profile(p, n=post_limit)
        analysis.update(prof_res)
    else:
        analysis['profile_exists'] = bool(user_raw)
        analysis['profile_data'] = {
            'bio': user_raw.get('biography'),
            'external_url': user_raw.get('external_url') or "N/A",
            'is_business': bool(user_raw.get('is_business')),
            'public_email': user_raw.get('public_email') or "N/A",
            'public_phone': user_raw.get('public_phone_number') or "N/A",
            'business_address': user_raw.get('business_address_json') or "N/A",
            'business_category': user_raw.get('business_category_name') or "N/A"
        }
        analysis['text_clues'] = extract_contact_info(f"{analysis['profile_data']['bio'] or ''} {analysis['profile_data']['external_url'] or ''}")
    analysis['text_clues']['phone_clue_norm'] = normalize_phone(analysis['text_clues'].get('phone_in_text'))
    analysis['profile_data']['public_phone_norm'] = normalize_phone(analysis['profile_data'].get('public_phone'))
    feed = get_feed_media(req, uid, sessionid, count=post_limit).get("raw") if uid else {}
    analysis['feed_raw'] = feed
    fi = infer_locations_from_feed(feed)
    analysis['activity_data'] = {'last_post_ts': fi.get('last_post_ts'), 'feed_locations': fi}
    captions, timestamps, post_ids_for_persist = [], [], []
    items = feed.get("items") or []
    for it in items:
        if isinstance(it.get("caption"), dict):
            cap = it["caption"].get("text")
            if cap: captions.append(cap)
        taken = it.get("taken_at") or it.get("device_timestamp") or it.get("timestamp")
        if taken:
            try:
                timestamps.append(datetime.fromtimestamp(int(taken), tz=timezone.utc))
            except (ValueError, TypeError):
                pass
        pid = it.get("id") or it.get("pk")
        if pid: post_ids_for_persist.append(str(pid))
    if p:
        beh = analyze_post_behavior_from_profile(p, n=post_limit)
        analysis['behavior_data'] = beh
        timestamps = beh.get('timestamps') or timestamps
        captions = beh.get('captions') or captions
    else:
        analysis['behavior_data'] = {'top_hashtags': [], 'top_mentions': [], 'timestamps': timestamps, 'captions': captions}
    tz_name = None
    mfl = analysis.get('most_frequent_location')
    if not mfl and fi.get('locations'):
        counts = Counter([l.get('name') for l in fi['locations'] if l.get('name')])
        if counts:
            top_name, ct = counts.most_common(1)[0]
            for l in fi['locations']:
                if l.get('name') == top_name:
                    mfl = {'name': top_name, 'lat': l.get('lat'), 'lng': l.get('lng'), 'count': ct}
                    break
    analysis['most_frequent_location'] = mfl
    if p:
        coords_only = [(lat, lng) for (lat, lng, _) in analysis.get('all_coords', []) if lat is not None and lng is not None]
        cluster = cluster_primary_location(coords_only)
        if cluster:
            analysis['cluster_primary'] = cluster
            if not analysis['most_frequent_location']:
                analysis['most_frequent_location'] = {'name': 'cluster_primary', 'lat': cluster['lat'], 'lng': cluster['lng'], 'count': cluster['count']}
    if analysis['most_frequent_location'] and HAS_TZF and HAS_PYTZ:
        tz_name = infer_timezone_from_coords(analysis['most_frequent_location'].get('lat'), analysis['most_frequent_location'].get('lng'))
        if tz_name:
            timestamps = [convert_to_local(ts, tz_name) for ts in timestamps]
    if cluster_temporal and p:
        clusters = cluster_posts_by_location(analysis.get('all_coords', []), eps=0.01, min_samples=3)
        analysis['cluster_temporal'] = peak_hours_per_cluster(clusters)
    hours, days = Counter(), Counter()
    for ts in timestamps:
        hours[ts.hour] += 1; days[ts.strftime('%A')] += 1
    analysis['temporal_data'] = {"top_hours": hours.most_common(24), "top_days": days.most_common(7)}
    all_texts = [analysis['profile_data'].get('bio'), analysis['profile_data'].get('external_url')] + captions
    analysis['keywords'] = extract_keywords(all_texts)
    analysis['entities'] = extract_entities_spacy(all_texts)
    analysis['topics'] = lda_topics(captions, n_topics=3, n_words=8)
    analysis['linguistic_analysis'] = []
    if HAS_TEXTBLOB:
        for t in (captions + [analysis['profile_data'].get('bio'), analysis['profile_data'].get('external_url')]):
            if not t: continue
            try:
                b = TextBlob(t)
                try: lang = b.detect_language()
                except Exception: lang = "unknown"
                analysis['linguistic_analysis'].append({"text": t[:50], "lang": lang, "sentiment": b.sentiment.polarity})
            except Exception as e:
                logging.warning(f"TextBlob error: {e}")
    pic_url = None
    if isinstance(user_raw, dict):
        for k in ("profile_pic_url_hd", "profile_pic_url"):
            if user_raw.get(k): pic_url = user_raw.get(k); break
        if not pic_url:
            hd = user_raw.get("hd_profile_pic_url_info") or {}
            pic_url = hd.get("url")
    if not pic_url and p:
        try:
            pic_url = getattr(p, "profile_pic_url", None)
        except Exception:
            pic_url = None
    analysis['exif_data'] = exif_from_url_native(req, pic_url) if pic_url else {"error": "No profile picture URL"}
    analysis['ris'] = ris_links(pic_url)
    if p:
        net = {"followers": [], "following": []}
        try:
            net["followers"] = [f.username for f in p.get_followers()][:1000]
        except Exception as e:
            logging.warning(f"Followers list error: {e}")
        try:
            net["following"] = [f.username for f in p.get_followees()][:1000]
        except Exception as e:
            logging.warning(f"Followees list error: {e}")
        analysis['network_data'] = {'followers': net['followers'], 'following': net['following'], 'mutual_followers': list(set(net['followers']) & set(net['following']))}
        commenters_ranked, comment_texts_by_user = analyze_comments_from_profile(p, post_limit=60, collect_texts=True)
        analysis['commenters'] = commenters_ranked
        analysis['comment_texts_by_user'] = comment_texts_by_user
        analysis['stories'] = {"count": 0, "locations": []}
        try:
            stories = list(p.get_stories())
            analysis['stories']["count"] = len(stories)
        except Exception as e:
            logging.warning(f"Stories error: {e}")
        analysis['ghost_followers'] = ghost_followers(p, recent_posts=30)
        analysis['likes_network'] = likes_network(p, posts=30)
        analysis['posting_frequency'] = posting_frequency_analytics(timestamps)
        analysis['engagement_windows'] = engagement_windows(p, posts=60)
        if deep_network and L:
            top_seed = list({u for u,_ in (analysis['likes_network'][:50] + analysis['commenters'][:50])} | set(analysis['network_data']['mutual_followers'][:50]))
            analysis['overlap_pairs'] = mutual_overlap_pairs(L, top_seed, per_user_limit=300, top_pairs=100)
    else:
        analysis['network_data'] = {'followers': [], 'following': [], 'mutual_followers': []}
        analysis['commenters'] = []
        analysis['stories'] = {"count": 0, "locations": []}
        analysis['ghost_followers'] = []
        analysis['likes_network'] = []
        analysis['posting_frequency'] = {}
        analysis['engagement_windows'] = {}
        analysis['overlap_pairs'] = []
        analysis['cluster_temporal'] = []
    os.makedirs(persist_dir, exist_ok=True)
    persist_path = os.path.join(persist_dir, f"{username}_posts.json")
    deleted, added = detect_deleted(persist_path, post_ids_for_persist)
    analysis['deleted_since_last_run'] = deleted
    analysis['added_since_last_run'] = added
    persist_post_ids(persist_path, post_ids_for_persist)
    return analysis

def parse_args():
    ap = argparse.ArgumentParser(description="InstaOSNIT_forensic_final - Forensic-grade Instagram OSINT")
    ap.add_argument("--target", "-t", required=True, help="Target username")
    ap.add_argument("--sessionid", "-s", required=False, help="Instagram sessionid cookie")
    ap.add_argument("--post-limit", "-n", type=int, default=200, help="Number of posts to analyze")
    ap.add_argument("--output", "-o", default=None, help="Output filename")
    ap.add_argument("--format", "-f", choices=["json","csv","gexf","pdf"], default="json")
    ap.add_argument("--terms", nargs="*", default=[], help="Targeted keyword search terms")
    ap.add_argument("--proxies", default=DEFAULT_CONFIG["proxies_file"], help="Proxies file")
    ap.add_argument("--ua-file", default=None, help="User-Agent file")
    ap.add_argument("--persist-dir", default=DEFAULT_CONFIG["persist_dir"], help="Persistence directory")
    ap.add_argument("--deep-network", action="store_true", help="Follower overlap analysis")
    ap.add_argument("--cluster-temporal", action="store_true", help="Per-location peak hours")
    ap.add_argument("--login-user", default=None, help="Instaloader login username")
    ap.add_argument("--login-pass", default=None, help="Instaloader login password")
    ap.add_argument("--log-file", default=DEFAULT_CONFIG["log_file"], help="Log file path")
    return ap.parse_args()

def main():
    args = parse_args()
    setup_logging(args.log_file)
    if args.ua_file and os.path.exists(args.ua_file):
        try:
            with open(args.ua_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                if lines: UA_POOL[:] = lines
        except (OSError, IOError) as e:
            logging.error(f"UA pool load error: {e}")
    proxies = load_proxies(args.proxies)
    proxy_pool = StickyProxyPool(proxies, DEFAULT_CONFIG["sticky_seconds"])
    req = Requester(proxy_pool)
    sessionid = args.sessionid
    if sessionid and not check_session_validity(req, sessionid):
        logging.warning("Provided sessionid appears invalid or expired.")
    analysis = build_analysis(req, args.target, sessionid=sessionid, post_limit=args.post_limit, persist_dir=args.persist_dir,
                              deep_network=args.deep_network, cluster_temporal=args.cluster_temporal,
                              login_user=args.login_user, login_pass=args.login_pass)
    if args.terms:
        all_texts = [analysis['profile_data'].get('bio'), analysis['profile_data'].get('external_url')] + (analysis.get('behavior_data',{}).get('captions') or [])
        analysis['targeted_search_results'] = targeted_keyword_search(all_texts, args.terms)
    out = args.output or f"{args.target}_report.{args.format}"
    try:
        if args.format == "json":
            export_json(analysis, out)
        elif args.format == "csv":
            export_csv(analysis, out)
        elif args.format == "gexf":
            export_gexf(analysis, out)
        elif args.format == "pdf":
            export_pdf(analysis, out)
        logging.info(f"Exported: {out}")
        print(out)
    except Exception as e:
        logging.error(f"Export failed: {e}")
        try:
            fallback = f"{args.target}_report.json"
            export_json(analysis, fallback)
            logging.info(f"Saved fallback JSON: {fallback}")
            print(fallback)
        except Exception as ee:
            logging.error(f"Fallback export failed: {ee}")

if __name__ == "__main__":
    main()
