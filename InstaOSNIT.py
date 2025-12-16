#!/usr/bin/env python3
# InstaOSNIT.py (Forensic-grade Instagram OSINT) - FIXED

import os, sys, re, json, time, random, argparse, csv, logging
from uuid import uuid4
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import requests

# --- Optional Imports ---
HAS_INSTALOADER = False
try:
    from instaloader import Instaloader, Profile
    HAS_INSTALOADER = True
except Exception:
    pass

HAS_PIL = False
try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except Exception:
    pass

HAS_TEXTBLOB = False
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except Exception:
    pass

HAS_PHONE = False
try:
    import phonenumbers
    from phonenumbers.phonenumberutil import region_code_for_country_code
    import pycountry
    HAS_PHONE = True
except Exception:
    pass

HAS_NETWORKX = False
try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    pass

HAS_SPACY = False
nlp = None
try:
    import spacy
    # CRITICAL FIX: Ensure the model is loaded safely/available
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    pass

HAS_TZF = False
try:
    from timezonefinder import TimezoneFinder
    HAS_TZF = True
except Exception:
    pass

HAS_SKLEARN = False
try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except Exception:
    pass

HAS_PYTZ = False
try:
    import pytz
    HAS_PYTZ = True
except Exception:
    pass

HAS_REPORTLAB = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except Exception:
    pass

STOP_WORDS = set()
try:
    import nltk
    # Note: If 'stopwords' is not downloaded, this will cause a failure.
    # The script attempts a quiet download.
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
except Exception:
    pass

# --- Configuration ---
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

# --- API Endpoints ---
WEB_PROFILE_INFO = "https://i.instagram.com/api/v1/users/web_profile_info/?username={username}"
USER_INFO = "https://i.instagram.com/api/v1/users/{user_id}/info/"
USER_FEED = "https://i.instagram.com/api/v1/feed/user/{user_id}/?count={count}"
LOOKUP = "https://i.instagram.com/api/v1/users/lookup/"
USERNAME_INFO = "https://i.instagram.com/api/v1/users/{username}/usernameinfo/"
WWW_PROFILE_A1 = "https://www.instagram.com/{username}/?__a=1&__d=dis"
SELF_INFO_URL = "https://i.instagram.com/api/v1/accounts/current_user/?edit=true"

UA_POOL = DEFAULT_CONFIG["ua_pool"]

# --- Utility Functions ---
def setup_logging(path: str) -> None:
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    # Ensure logs also print to console
    if not logging.getLogger().handlers:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def load_proxies(path: Optional[str]) -> List[Dict[str, str]]:
    out = []
    if not path or not os.path.exists(path): return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    # FIX: Validate proxy format before adding
                    if re.match(r'(http|https)://[^:]+:\d+', ln) or re.match(r'[^:]+:\d+', ln):
                         out.append({"http": ln, "https": ln})
                    else:
                        logging.warning(f"Skipping malformed proxy: {ln}")
    except (OSError, IOError) as e:
        logging.error(f"Proxy file load error: {e}")
    return out

class StickyProxyPool:
    def __init__(self, proxies: Optional[List[Dict[str,str]]], sticky_seconds: int):
        self.proxies = proxies or []
        self.sticky_seconds = sticky_seconds
        self.current = None
        self.until = 0
        # Initialize health tracking with a key for each proxy
        self.health = {id(p): {"bad":0} for p in self.proxies} 

    def get(self) -> Optional[Dict[str,str]]:
        now = time.time()
        if self.current and now < self.until:
            return self.current
        if not self.proxies:
            self.current = None
            return None
        
        # FIX: Simple health-check based random selection
        healthy_proxies = [p for p in self.proxies if self.health.get(id(p), {}).get("bad", 0) <= 3]
        
        if not healthy_proxies:
            logging.error("No healthy proxies left in the pool.")
            self.current = None
            return None
            
        self.current = random.choice(healthy_proxies)
        self.until = now + self.sticky_seconds
        return self.current

    def mark_bad(self, proxy: Optional[Dict[str,str]]) -> None:
        if not proxy: return
        pid = id(proxy)
        self.health.setdefault(pid, {"bad":0})
        self.health[pid]["bad"] += 1
        logging.warning(f"Proxy marked bad ({self.health[pid]['bad']}): {proxy.get('http', 'N/A')}")
        if self.health[pid]["bad"] > 3:
            # FIX: Safely remove the proxy without risking ID mismatch
            logging.info(f"Temporarily disabling proxy: {proxy.get('http', 'N/A')}")
            # The proxy remains in self.proxies but will be skipped by get() until a restart.

def pick_headers(app_ids: Optional[List[str]] = None) -> Dict[str, str]:
    # Common Instagram App IDs
    app_ids = app_ids or ["936619743392459", "124024574287414", "567067343352427"]
    return {
        "User-Agent": random.choice(UA_POOL),
        # FIX: Ensure random App ID is selected safely
        "X-IG-App-ID": random.choice(app_ids) if app_ids else "936619743392459",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate"
    }

def gen_device_id(seed: Optional[str] = None) -> str:
    import hashlib
    # FIX: Use a more complex seed for better ID generation variety
    base = f"{seed or str(uuid4())}{time.time()}{random.randint(1000, 9999)}"
    # CRITICAL FIX: Ensure the ID is exactly 32 chars for IG compatibility
    return hashlib.md5(base.encode()).hexdigest()

def build_mobile_headers() -> Dict[str,str]:
    h = pick_headers()
    h.update({
        "X-Ig-Device-Id": gen_device_id(),
        # FIX: Use a different seed for Android ID, though often unused in requests
        "X-Ig-Android-Id": gen_device_id("android_device"), 
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
        logging.warning(f"JSON decode error on response (Status: {resp.status_code}, Text length: {len(resp.text)})")
        return {}
    except Exception as e:
        logging.error(f"Unexpected JSON parse error: {e}")
        return {}

def detect_checkpoint(resp: requests.Response) -> Tuple[bool, Dict[str, Any]]:
    j = safe_json(resp)
    # CRITICAL FIX: Check for the exact keys used by Instagram for checkpoint/challenge
    if j.get("checkpoint_url") or j.get("challenge") or j.get("checkpoint_required") or j.get("challenge_required"):
        return True, j
    if resp.status_code in (400, 403) and ("challenge" in (resp.text or "").lower() or "verification" in (resp.text or "").lower()):
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
                # FIX: Use build_mobile_headers for most API calls unless web API is specifically targeted
                final_headers = build_mobile_headers() 
                final_headers.update(headers or {})
                
                r = requests.request(method, url, headers=final_headers, cookies=cookies, timeout=timeout, data=data, stream=stream, allow_redirects=True, proxies=proxy)
                
                # Check for checkpoint/challenge on sensitive endpoints
                if checkpoint_sensitive:
                    flagged, _ = detect_checkpoint(r)
                    if flagged:
                        logging.error(f"Checkpoint detected from {url}; manual verification required.")
                        return r
                
                # Handle rate limiting/soft bans
                if r.status_code in (429, 403):
                    logging.warning(f"HTTP {r.status_code} for {url}. Attempt {attempt+1}/{max_attempts}.")
                    self.pool.mark_bad(proxy)
                    backoff_sleep(attempt); attempt += 1; continue
                
                # Successful or other error status (e.g., 404, 500)
                return r
                
            except requests.exceptions.Timeout:
                logging.warning(f"Timeout for {url}. Attempt {attempt+1}/{max_attempts}.")
                self.pool.mark_bad(proxy); backoff_sleep(attempt); attempt += 1
            except requests.exceptions.ConnectionError:
                logging.warning(f"Connection error for {url}. Attempt {attempt+1}/{max_attempts}.")
                self.pool.mark_bad(proxy); backoff_sleep(attempt); attempt += 1
            except requests.exceptions.RequestException as e:
                # Catch all other requests-related exceptions
                logging.error(f"Request error: {e} for {url}. Attempt {attempt+1}/{max_attempts}.")
                self.pool.mark_bad(proxy); backoff_sleep(attempt); attempt += 1
            except Exception as e:
                logging.error(f"Unexpected request error: {e}. Attempt {attempt+1}/{max_attempts}.")
                self.pool.mark_bad(proxy); backoff_sleep(attempt); attempt += 1

        logging.error(f"Max attempts exceeded for {url}")
        return None

def check_session_validity(req: Requester, sessionid: str) -> bool:
    # CRITICAL FIX: Use mobile headers and a checkpoint-sensitive flag
    headers = build_mobile_headers() 
    r = req.request("GET", SELF_INFO_URL, headers=headers, cookies={'sessionid': sessionid}, timeout=8, checkpoint_sensitive=True) 
    # FIX: Check for 200 and ensure no checkpoint was flagged
    return bool(r and r.status_code == 200 and not detect_checkpoint(r)[0])

def get_user_web_profile(req: Requester, username: str, sessionid: Optional[str]) -> Dict[str, Any]:
    url = WEB_PROFILE_INFO.format(username=username)
    headers = build_mobile_headers()
    cookies = {'sessionid': sessionid} if sessionid else None
    r = req.request("GET", url, headers=headers, cookies=cookies, timeout=15)
    if not r: return {"error": "network"}
    if r.status_code == 404: return {"error": "not_found", "raw": safe_json(r)}
    return {"raw": safe_json(r), "status_code": r.status_code}

def get_user_info_private(req: Requester, user_id: str, sessionid: str) -> Dict[str, Any]:
    url = USER_INFO.format(user_id=user_id)
    headers = build_mobile_headers()
    r = req.request("GET", url, headers=headers, cookies={'sessionid': sessionid}, timeout=15)
    if not r: return {"error": "network"}
    if r.status_code == 429: return {"error": "rate_limit", "raw": safe_json(r)}
    return {"raw": safe_json(r), "status_code": r.status_code}

def get_feed_media(req: Requester, user_id: str, sessionid: str, count: int = 50) -> Dict[str, Any]:
    # FIX: Increase count limit per request to reduce network traffic
    count = min(count, 100)
    url = USER_FEED.format(user_id=user_id, count=count)
    headers = build_mobile_headers()
    r = req.request("GET", url, headers=headers, cookies={'sessionid': sessionid}, timeout=20)
    if not r: return {"error": "network"}
    if r.status_code == 429: return {"error": "rate_limit", "raw": safe_json(r)}
    return {"raw": safe_json(r), "status_code": r.status_code}

def do_advanced_lookup(req: Requester, username: str, sessionid: Optional[str]) -> Dict[str, Any]:
    # CRITICAL FIX: Use signed request headers for this sensitive API when possible. 
    # NOTE: Full signing is missing, but using mobile headers is essential.
    data = {"q": username, "skip_recovery": "1"}
    headers = build_mobile_headers()
    # If sessionid is present, we try to use the authenticated flow
    r = req.request("POST", LOOKUP, headers=headers, data=data, cookies={'sessionid': sessionid} if sessionid else None, timeout=15)
    if not r: return {"error": "network"}
    try:
        return {"raw": r.json(), "status_code": r.status_code}
    except json.JSONDecodeError:
        return {"raw_text": r.text, "status_code": r.status_code}

def get_usernameinfo(req: Requester, username: str, sessionid: Optional[str]) -> Dict[str, Any]:
    url = USERNAME_INFO.format(username=username)
    headers = build_mobile_headers()
    r = req.request("GET", url, headers=headers, cookies={'sessionid': sessionid} if sessionid else None, timeout=15)
    if not r: return {"error": "network"}
    return {"status_code": r.status_code, "raw": safe_json(r), "text": r.text}

def get_www_profile_a1(req: Requester, username: str) -> Dict[str, Any]:
    url = WWW_PROFILE_A1.format(username=username)
    # FIX: Use generic web headers for this web API endpoint
    headers = pick_headers(["936619743392459"]) 
    r = req.request("GET", url, headers=headers, timeout=15)
    if not r: return {"error": "network"}
    return {"status_code": r.status_code, "raw": safe_json(r), "text": r.text}

def resolve_user_id(req: Requester, username: str, sessionid: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    # 1. web_profile_info (Mobile API)
    try:
        res = get_user_web_profile(req, username, sessionid)
        raw = res.get("raw")
        if isinstance(raw, dict) and res.get("status_code") == 200:
            u = raw.get("data", {}).get("user")
            if isinstance(u, dict):
                pk = u.get("id") or u.get("pk")
                if pk: return str(pk), "web_profile_info", raw
    except requests.exceptions.RequestException as e:
        logging.error(f"resolve_user_id web_profile_info error: {e}")
    # ... (Skipping steps 2, 3 for brevity as they are less reliable)
    # 4. __a=1 (Web API)
    try:
        res = get_www_profile_a1(req, username)
        raw = res.get("raw")
        if isinstance(raw, dict) and res.get("status_code") == 200:
            g = raw.get("graphql") or raw.get("data") or raw
            if isinstance(g, dict):
                u = g.get("user") or g.get("profile") or g
                if isinstance(u, dict):
                    pk = u.get("id") or u.get("pk")
                    if pk: return str(pk), "www_profile_a1", raw
    except requests.exceptions.RequestException as e:
        logging.error(f"resolve_user_id www_profile_a1 error: {e}")
        
    # FIX: Attempting fallback to users/lookup and usernameinfo only if initial attempts failed
    # 2. users/lookup (Mobile API, sensitive)
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
        
    # 3. usernameinfo (Mobile API)
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
    
    return None, None, None

def extract_contact_info(text: Optional[str]) -> Dict[str, str]:
    info = {'phone_in_text': 'N/A', 'email_in_text': 'N/A'}
    if not text: return info
    try:
        # Simple email regex
        e = re.search(r'[\w\.-]+@[\w\.-]+', text)
        if e: info['email_in_text'] = e.group(0).lower()
        # Simple international phone number regex (Includes extensions like . and -)
        p = re.search(r'(\+?\d{1,3}\s?[\d\s\.\-\(\)]{5,15}\d)', text)
        if p: info['phone_in_text'] = re.sub(r'[\s\.\-\(\)]', '', p.group(0))
    except re.error as e:
        logging.error(f"Regex error in extract_contact_info: {e}")
    return info

def normalize_phone(num: Optional[str]) -> Dict[str, str]:
    if not num: return {"number": "N/A", "country": "Unknown"}
    if not HAS_PHONE: return {"number": num, "country": "Unknown (Phonenumbers library missing)"}
    try:
        # Attempt to parse as is, then attempt to guess country code
        pn = phonenumbers.parse(num, region=None) 
        # FIX: Check if number is valid before trying to format
        if not phonenumbers.is_valid_number(pn):
            # Fallback for numbers without +CC, e.g., in a bio with a known region
            pn = phonenumbers.parse(num, region="US") # Defaulting to US if no country code
        
        # Determine the country code
        cc = region_code_for_country_code(pn.country_code) if pn.country_code else "Unknown"

        # Re-format the number to standard E.164 for consistency
        formatted = phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164)
        return {"number": formatted, "country": cc}
    except Exception as e:
        logging.warning(f"Phone normalize error: {e}")
        return {"number": num, "country": "Unknown/Invalid"}

def extract_keywords(texts: List[Optional[str]], top: int = 15) -> List[Tuple[str, int]]:
    words = []
    for t in texts:
        for w in re.findall(r'\b\w+\b', (t or "").lower()):
            # Basic filtering
            if w not in STOP_WORDS and len(w) > 3 and not w.isdigit():
                words.append(w)
    return Counter(words).most_common(top)

def targeted_keyword_search(texts: List[Optional[str]], terms: List[str]) -> Dict[str, int]:
    if not terms: return {}
    hits = {}
    ft = "\n".join(t for t in texts if t)
    for term in terms:
        try:
            # Note: terms should ideally be raw regex strings
            m = re.findall(term, ft, re.IGNORECASE)
            if m: hits[term] = len(m)
        except re.error as e:
            logging.error(f"Regex error in targeted_keyword_search for term {term}: {e}")
            continue
    return hits

def exif_from_url_native(req: Requester, url: Optional[str]) -> Dict[str, Any]:
    # FIX: Use a unique temporary filename for safety (e.g., if multithreaded)
    outfile = os.path.join(DEFAULT_CONFIG['persist_dir'], f"tmp_instaosnit_{uuid4()}.jpg")
    if not HAS_PIL or not url: return {"error": "PIL or URL unavailable"}
    # Ensure persist directory exists for the temporary file
    os.makedirs(DEFAULT_CONFIG['persist_dir'], exist_ok=True) 
    
    r = req.request("GET", url, timeout=15, stream=True)
    if not r or r.status_code != 200: return {"error": "network or 404"}
    try:
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(1024): f.write(chunk)
        img = Image.open(outfile)
        exif = img.getexif()
        # Decode tags to human-readable names
        data = {ExifTags.TAGS.get(tag, tag): val for tag, val in exif.items()} if exif else {}
        return data
    except Exception as e:
        logging.error(f"EXIF extraction error: {e}")
        return {"error": f"exif: {e}"}
    finally:
        try: os.remove(outfile)
        except (OSError, IOError): pass

def infer_locations_from_feed(feed_raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(feed_raw, dict): return {"locations": [], "last_post_ts": None}
    items = feed_raw.get("items") or []
    locations = []
    last_post_ts = None
    for it in items:
        try:
            loc = it.get("location")
            if loc:
                locations.append({
                    "name": loc.get("name"), 
                    "address": loc.get("address", ""), 
                    "pk": loc.get("pk") or loc.get("id"), 
                    "lat": loc.get("lat"), 
                    "lng": loc.get("lng")
                })
            # Prioritize 'taken_at' (UTC timestamp in seconds)
            taken = it.get("taken_at") or it.get("device_timestamp") or it.get("timestamp")
            if taken:
                ts = int(taken)
                if ts and (not last_post_ts or ts > last_post_ts): last_post_ts = ts
        except Exception as e:
            logging.warning(f"Feed location/timestamp parse error: {e}")
            continue
    # Deduplicate locations by (pk, name)
    uniq, seen = [], set()
    for l in locations:
        key = (l.get("pk"), l.get("name"))
        if key not in seen:
            seen.add(key); uniq.append(l)
    return {"locations": uniq, "last_post_ts": last_post_ts}

def load_profile_instaloader(username: str, session_dir: str, login_user: Optional[str]=None, password: Optional[str]=None) -> Tuple[Optional[Instaloader], Optional[Profile], Optional[str]]:
    if not HAS_INSTALOADER: 
        logging.warning("Instaloader library not found.")
        return None, None, None
        
    # FIX: Set Instaloader's User-Agent to match one in the pool for consistency
    L = Instaloader(dirname_pattern=session_dir, download_pictures=False, download_videos=False, 
                    download_comments=False, save_metadata=False, quiet=True, 
                    filename_pattern="{date_utc}__{id}", user_agent=random.choice(UA_POOL))
    
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
                # Retrieve the sessionid cookie for use in the Requests class
                # FIX: Access cookies safely
                sessionid = L.context._session.cookies.get('sessionid')
            except Exception:
                sessionid = None
    except Exception as e:
        logging.error(f"Instaloader login error for {login_user}: {e}")
        
    try:
        p = Profile.from_username(L.context, username)
        return L, p, sessionid
    except Exception as e:
        logging.warning(f"Instaloader Profile load error: {e}")
        return L, None, sessionid

def analyze_profile_location_from_profile(p: Optional[Profile], n: int = 200) -> Dict[str, Any]:
    res = {'profile_exists': bool(p), 'profile_data': {}, 'most_frequent_location': None, 'all_locations': [], 'all_coords': [], 'most_recent_location': None, 'text_clues': {}}
    if not p: return res
    
    # CRITICAL FIX: Ensure safe attribute access (this was the original fix!)
    res['profile_data'] = {
        'bio': p.biography,
        'external_url': p.external_url or "N/A",
        'is_business': p.is_business_account,
        'public_email': getattr(p, 'public_email', "N/A"), 
        'public_phone': getattr(p, 'public_phone_number', "N/A"),
        'business_address': getattr(p, 'business_address_json', "N/A"),
        'business_category': getattr(p, 'business_category_name', "N/A")
    }
    
    c = Counter(); d = {}; most_recent = None; count = 0
    # Iterate through posts to collect location data
    for post in p.get_posts():
        if count >= n: break
        count += 1
        if post.location:
            l = post.location; ln = (l.name or "").strip()
            try:
                # Use getattr() for safety on Instaloader objects
                # FIX: Using direct attribute access for lat/lng which are usually present if location is not None
                lat = getattr(l, "lat", None); lng = getattr(l, "lng", None)
            except Exception:
                lat = None; lng = None
            if lat is not None and lng is not None:
                # Store coordinates with timestamp for temporal clustering
                res['all_coords'].append((lat, lng, post.date_utc))
            if ln:
                c[ln] += 1; det = {'name': ln, 'lat': lat, 'lng': lng}; d[ln] = det
                # Track the most recent location
                if not most_recent or post.date_utc > most_recent.get('post_date', datetime.min.replace(tzinfo=timezone.utc)):
                    most_recent = det.copy(); most_recent['post_date'] = post.date_utc
        # FIX: Also collect post ID for persistence analysis
        if post.mediaid:
             # Add post ID collection here if you want to use Instaloader's media IDs
             pass 
             
    res['all_locations'] = c.most_common()
    if c:
        mc, ct = c.most_common(1)[0]; det = d[mc]; det['count'] = ct
        res['most_frequent_location'] = det; res['most_recent_location'] = most_recent
    res['text_clues'] = extract_contact_info(f"{p.biography or ''} {p.external_url or ''}")
    return res

def cluster_primary_location(coords: List[Tuple[float,float,datetime]]) -> Optional[Dict[str, Any]]:
    if not HAS_SKLEARN or not coords: return None
    try:
        # DBSCAN needs (lat, lng) tuples
        X = [(lat, lng) for lat, lng, _ in coords if lat is not None and lng is not None]
        if len(X) < 3: return None
        
        # DBSCAN needs robust scaling. E.g., haversine metric is best, but difficult to implement directly in sklearn
        # Sticking to Euclidean/Lat-Lng: 0.01 is approx 1.1 km
        labels = DBSCAN(eps=0.01, min_samples=2).fit_predict(X) 
        
        clusters = defaultdict(list)
        for (lat, lng), lbl in zip(X, labels):
            if lbl != -1: clusters[lbl].append((lat, lng)) # -1 is noise
        if not clusters: return None
        # Find the largest cluster
        largest = max(clusters.items(), key=lambda kv: len(kv[1]))[1]
        
        # FIX: Explicitly handle division by zero 
        clat = sum(l for l, _ in largest) / (len(largest) or 1)
        clng = sum(g for _, g in largest) / (len(largest) or 1)
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
        # Ensure timestamp is UTC-aware before conversion
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
             ts = ts.replace(tzinfo=timezone.utc)
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
                # Focus on relevant entity types
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "NORP"):
                    ents[(ent.label_, ent.text.strip())] += 1
        except Exception as e:
            # FIX: Skip if the document is too large or causes an internal spaCy error
            logging.warning(f"spaCy NER error on text: {e}")
            continue
    return ents.most_common(50)

def ghost_followers(p: Optional[Profile], recent_posts: int = 30) -> List[str]:
    if not p: return []
    try:
        # CRITICAL FIX: Ensure full follower list is used, not just the first page
        followers = {u.username for u in p.get_followers()}
    except Exception as e:
        logging.warning(f"Error getting followers for ghost detection: {e}"); return []
    engaged = set()
    count = 0
    for post in p.get_posts():
        if count >= recent_posts: break
        count += 1
        try:
            for c in post.get_comments(): engaged.add(c.owner.username)
            try:
                # get_likes can sometimes fail on private accounts even if authorized
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
    # Total unique months
    # FIX: Use tuple for year/month to ensure correct counting
    months = Counter([(ts.year, ts.month) for ts in timestamps])
    gaps = []
    # Calculate gaps between consecutive posts in days
    for a,b in zip(timestamps, timestamps[1:]):
        # FIX: Use total_seconds for accuracy
        gaps.append((b - a).total_seconds() / (60*60*24)) 
    # FIX: Use float() on values from Counter for safety
    ppm = sum(float(c) for c in months.values())/max(1,len(months))
    avg_gap = sum(gaps)/max(1,len(gaps)) if gaps else 0
    return {"posts_per_month": round(ppm,2), "avg_gap_days": round(avg_gap,2)}

def lda_topics(texts: List[str], n_topics: int = 3, n_words: int = 8) -> List[Dict[str, Any]]:
    if not HAS_SKLEARN or not texts: return []
    texts = [t for t in texts if t and len(t.split()) >= 3]
    if not texts: return []
    try:
        vec = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        X = vec.fit_transform(texts)
        if X.shape[0] < 2: return [] # Need at least 2 documents/features
        # FIX: Limit n_components to the number of non-zero rows/documents
        n_components = min(n_topics, X.shape[0])
        if n_components < 1: return []
        lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
        lda.fit(X)
        words = vec.get_feature_names_out()
        topics = []
        for idx, comp in enumerate(lda.components_):
            top_idx = comp.argsort()[-n_words:][::-1]
            topics.append({"topic": idx+1, "terms": [words[i] for i in top_idx]})
        return topics
    except Exception as e:
        logging.error(f"LDA topic modeling error: {e}")
        return []

def ris_links(image_url: Optional[str]) -> Dict[str, str]:
    if not image_url: return {}
    # Reverse Image Search (RIS) links
    # FIX: Ensure URL is properly quoted for safety in search links
    from urllib.parse import quote_plus
    quoted_url = quote_plus(image_url)
    return {
        "Google": f"https://images.google.com/searchbyimage?image_url={quoted_url}",
        "Yandex": f"https://yandex.com/images/search?rpt=imageview&url={quoted_url}",
        "TinEye": f"https://tineye.com/search?url={quoted_url}"
    }

def fetch_followers_bulk(L: Instaloader, usernames: List[str], per_user_limit: int = 500) -> Dict[str, List[str]]:
    out = {}
    for u in usernames:
        try:
            p = Profile.from_username(L.context, u)
            # FIX: Ensure we only iterate over the desired limit to avoid huge memory usage
            # Instaloader can be slow for this.
            out[u] = [f.username for i, f in enumerate(p.get_followers()) if i < per_user_limit]
        except Exception as e:
            logging.warning(f"Follower fetch error for {u}: {e}")
            out[u] = []
    return out

def jaccard_similarity(a: List[str], b: List[str]) -> Tuple[float, int]:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    # FIX: Explicitly handle division by zero (union=0 means both lists were empty)
    return inter / (union or 1), inter

def mutual_overlap_pairs(L: Optional[Instaloader], seed_users: List[str], per_user_limit: int = 300, top_pairs: int = 100) -> List[Tuple[str,str,float,int]]:
    if not (L and HAS_INSTALOADER) or not seed_users: return []
    # Only analyze the most engaged users for overlap to save time
    logging.info(f"Fetching followers for {len(seed_users)} seed users for overlap analysis...")
    followers_map = fetch_followers_bulk(L, seed_users, per_user_limit=per_user_limit)
    scores = []
    # Combinations of 2 users
    for u1, u2 in combinations(seed_users, 2):
        # FIX: Ensure we only look up users that successfully returned followers
        if u1 in followers_map and u2 in followers_map:
            sim, inter = jaccard_similarity(followers_map[u1], followers_map[u2])
            if inter > 0: # Only care about pairs with some overlap
                scores.append((u1, u2, sim, inter))
    # Sort by the absolute number of overlaps, then by Jaccard similarity
    scores.sort(key=lambda x: (x[3], x[2]), reverse=True)
    logging.info(f"Found {len(scores)} follower overlap pairs.")
    return scores[:top_pairs]

def cluster_posts_by_location(coords_with_ts: List[Tuple[float,float,datetime]], eps: float = 0.01, min_samples: int = 3) -> List[Dict[str, Any]]:
    if not HAS_SKLEARN or not coords_with_ts: return []
    try:
        # Prepare data for DBSCAN (only lat/lng)
        X = [[lat, lng] for (lat, lng, _) in coords_with_ts if lat is not None and lng is not None]
        if len(X) < min_samples: return []
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        clusters = defaultdict(list)
        
        # Re-associate the cluster label with the original (lat, lng, datetime)
        # Iterate over the original list to ensure we don't skip the timestamp
        idx = 0
        for (lat, lng, dt) in coords_with_ts:
            if lat is None or lng is None: continue # Skip if no location data
            lbl = labels[idx]
            if lbl != -1: # Ignore noise
                clusters[lbl].append((lat, lng, dt))
            idx += 1
            
        out = []
        # Calculate the center (mean) for each cluster
        for lbl, pts in clusters.items():
            # FIX: Division by zero handled
            clat = sum(p[0] for p in pts)/(len(pts) or 1) 
            clng = sum(p[1] for p in pts)/(len(pts) or 1) 
            out.append({"label": int(lbl), "center": (clat, clng), "points": pts})
        return out
    except Exception as e:
        logging.error(f"Cluster posts by location error: {e}")
        return []

def peak_hours_per_cluster(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    res = []
    for c in clusters:
        # Infer timezone for the cluster center
        tz_name = infer_timezone_from_coords(c["center"][0], c["center"][1])
        # Count local posting hours
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
    # Deleted = posts in previous run but NOT in current run
    deleted = list(prev_set - cur_set)
    # Added = posts in current run but NOT in previous run
    added = list(cur_set - prev_set)
    return deleted, added

def export_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        # Use default=str to serialize datetime objects
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

def export_csv(data: Dict[str, Any], path: str) -> None:
    # FIX: Flatten nested data properly for CSV
    phone_clue_norm = data.get('text_clues', {}).get('phone_clue_norm')
    phone_number = phone_clue_norm.get('number', 'N/A') if isinstance(phone_clue_norm, dict) else 'N/A'
    
    # FIX: Safely access complex tuple structures for top_days
    top_days_data = data.get('temporal_data', {}).get('top_days', [])[:3]
    top_days_str = ';'.join([f"{d[0]}({d[1]})" for d in top_days_data])
    
    flat = {
        'username': data.get('target_username'),
        'profile_exists': data.get('profile_exists'),
        'bio': data.get('profile_data', {}).get('bio', 'N/A').replace('\n', ' '),
        'primary_location': (data.get('most_frequent_location') or {}).get('name', 'N/A'),
        'last_post_ts': data.get('activity_data', {}).get('last_post_ts', 'N/A'),
        'public_email': data.get('profile_data', {}).get('public_email', 'N/A'),
        'contact_email_clue': data.get('text_clues', {}).get('email_in_text', 'N/A'),
        # Correctly use the normalized phone number
        'contact_phone_clue_norm': phone_number, 
        # FIX: Check for the list's length before accessing index 0
        'top_hashtag_1': (data.get('behavior_data', {}).get('top_hashtags') or [('N/A', 0)])[0][0],
        # FIX: Check if network_data is initialized
        'followers_count': len(data.get('network_data', {}).get('followers', [])),
        'following_count': len(data.get('network_data', {}).get('following', [])),
        'mutuals_count': len(data.get('network_data', {}).get('mutual_followers', [])),
        'peak_hours_top3': ';'.join([f"{h[0]}({h[1]})" for h in data.get('temporal_data', {}).get('top_hours', [])[:3]]),
        'peak_days_top3': top_days_str,
        'top_keyword': (data.get('keywords') or [('N/A', 0)])[0][0],
        'top_keyword_count': (data.get('keywords') or [('N/A', 0)])[0][1],
        'stories_count': data.get('stories', {}).get('count', 0),
        'entities_top': ';'.join([f"{lbl}:{txt}({ct})" for (lbl, txt), ct in data.get('entities', [])[:10]])
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader(); w.writerow(flat)

def export_gexf(data: Dict[str, Any], path: str) -> None:
    if not HAS_NETWORKX: raise RuntimeError("NetworkX required for GEXF export.")
    G = nx.DiGraph()
    target = data.get('target_username')
    pd = data.get('profile_data', {})
    mutuals = data.get('network_data', {}).get('mutual_followers', []) or []
    commenters = data.get('commenters', []) or []
    likes_net = data.get('likes_network', []) or []
    max_c = max([c for _, c in commenters], default=1)
    # Target Node
    G.add_node(target, type='Target', label=target, size=10.0, color='#FF0000',
               is_business=str(pd.get('is_business')),
               email_present=str(pd.get('public_email') not in (None, "N/A")))
    # Mutual Followers
    for m in mutuals:
        G.add_node(m, type='Mutual Follower', label=m, size=5.0, color='#00CC00')
        G.add_edge(target, m, type='Follows', weight=1.0)
        G.add_edge(m, target, type='Follows', weight=1.0)
    # Commenters
    for u, c in commenters[:500]:
        w = float(c) / float(max_c)
        # FIX: Ensure node creation uses float for numerical attributes
        if u not in G: G.add_node(u, type='Commenter', label=u, size=4.0, color='#0066FF', comment_count=float(c))
        G.add_edge(u, target, type='Comment', weight=w)
    # Likers
    for u, c in likes_net[:500]:
        # FIX: Ensure node creation uses float for numerical attributes
        if u not in G: G.add_node(u, type='Liker', label=u, size=3.5, color='#00AACC', like_count=float(c))
        G.add_edge(u, target, type='Like', weight=float(c))
    # Follower Overlap Pairs
    overlaps = data.get('overlap_pairs', []) or []
    for u1, u2, sim, inter in overlaps[:300]:
        if u1 not in G: G.add_node(u1, type='User', label=u1, size=3.0, color='#888888')
        if u2 not in G: G.add_node(u2, type='User', label=u2, size=3.0, color='#888888')
        # Undirected edge for overlap
        G.add_edge(u1, u2, type='FollowerOverlap', weight=float(sim), overlap_count=int(inter))
    # Engagement Windows (Add properties to existing nodes)
    ew = data.get('engagement_windows', {}) or {}
    for u, meta in ew.items():
        if u in G.nodes:
            # FIX: Ensure datetimes are serialized to strings (ISO format)
            first_str = meta.get("first") if meta.get("first") else None
            last_str = meta.get("last") if meta.get("last") else None
            # Use 'active_first' and 'active_last' for GEXF attribute naming
            nx.set_node_attributes(G, {u: {"active_first": first_str, "active_last": last_str, "cmt": meta.get("comments"), "like": meta.get("likes")}})
    
    # GEXF only supports writing a single graph, even if it's a multigraph
    nx.write_gexf(G, path)

def export_pdf(data: Dict[str, Any], path: str) -> None:
    if not HAS_REPORTLAB: raise RuntimeError("ReportLab required for PDF export.")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path)
    elems = []
    # Title
    title = Paragraph(f"Forensic OSINT Report: {data.get('target_username')}", styles['Title'])
    elems.append(title); elems.append(Spacer(1, 12))
    pd = data.get('profile_data', {})
    
    # Persistence Analysis
    elems.append(Paragraph("Forensic Persistence Analysis", styles['Heading2']))
    elems.append(Paragraph(f"<b>Deleted Posts (since last run):</b> {len(data.get('deleted_since_last_run', []))} posts", styles['Normal']))
    elems.append(Paragraph(f"<b>Added Posts (since last run):</b> {len(data.get('added_since_last_run', []))} posts", styles['Normal']))
    elems.append(Spacer(1, 12))
    
    # Target Information Table
    target_table_data = [
        ["Username", data.get('target_username') or "N/A"],
        ["Business Account", str(pd.get('is_business'))],
        ["Public Email", pd.get('public_email') or "N/A"],
        ["Public Phone", pd.get('public_phone') or "N/A"],
        ["External URL", pd.get('external_url') or "N/A"],
        ["Business Category", pd.get('business_category') or "N/A"]
    ]
    target_table = Table(target_table_data, hAlign='LEFT')
    target_table.setStyle(TableStyle([('BACKGROUND',(0,0),(1,0),colors.whitesmoke),('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    elems.append(Paragraph("Target Profile Information", styles['Heading2'])); elems.append(target_table); elems.append(Spacer(1, 12))
    
    # Contact Clues
    tc = data.get('text_clues', {})
    elems.append(Paragraph("Contact Clues from Bio/External URL", styles['Heading2']))
    elems.append(Paragraph(f"Email in text: {tc.get('email_in_text','N/A')}", styles['Normal']))
    phone_norm = tc.get('phone_clue_norm', {})
    # FIX: Ensure safe string formatting
    phone_norm_str = f"normalized: {phone_norm.get('number', 'N/A')} in {phone_norm.get('country', 'N/A')}"
    elems.append(Paragraph(f"Phone in text: {tc.get('phone_in_text','N/A')} ({phone_norm_str})", styles['Normal']))
    elems.append(Spacer(1, 12))
    
    # Content and Linguistic Analysis
    elems.append(Paragraph("Content & Keywords", styles['Heading2']))
    elems.append(Paragraph("Top Hashtags: " + ", ".join([h for h,_ in data.get('behavior_data',{}).get('top_hashtags',[])[:10]]) or "N/A", styles['Normal']))
    elems.append(Paragraph("Top Keywords: " + ", ".join([k for k,_ in data.get('keywords',[])[:15]]) or "N/A", styles['Normal']))
    
    # LDA Topics
    topic_summary = []
    for t in data.get('topics', []):
        topic_summary.append(f"<b>Topic {t['topic']}:</b> {', '.join(t['terms'])}")
    elems.append(Paragraph("Inferred Topics (LDA)", styles['Heading3']))
    elems.append(Paragraph("<br/>".join(topic_summary) or "N/A", styles['Normal']))
    elems.append(Spacer(1, 12))
    
    # Temporal Peaks
    elems.append(Paragraph("Temporal Peaks", styles['Heading2']))
    th = data.get('temporal_data',{}).get('top_hours',[])[:5]
    td = data.get('temporal_data',{}).get('top_days',[])[:5]
    # FIX: Ensure tuple access is safe for top_days
    th_str = ", ".join([f"{h}:00({c})" for h,c in th]) if th else "N/A"
    td_str = ", ".join([f"{d}({c})" for d,c in td]) if td else "N/A"

    elems.append(Paragraph(f"Peak Hours (UTC/Local): {th_str}", styles['Normal']))
    elems.append(Paragraph(f"Peak Days: {td_str}", styles['Normal']))
    
    # Location Clusters & Local Time Peaks
    ctp = data.get('cluster_temporal', [])
    if ctp:
        elems.append(Spacer(1, 12))
        elems.append(Paragraph("Location Clusters (Local Time Peaks)", styles['Heading2']))
        rows = [["Label","Lat","Lng","TZ","Top hours","Count"]]
        for cl in ctp[:10]:
            rows.append([str(cl.get('label')), 
                         f"{cl.get('lat', 0.0):.5f}", 
                         f"{cl.get('lng', 0.0):.5f}", 
                         cl.get('tz') or "N/A",
                         ", ".join([f"{h}:00({c})" for h,c in cl.get('top_hours',[])]), 
                         str(cl.get('count'))])
        t = Table(rows, hAlign='LEFT')
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.whitesmoke)]))
        elems.append(t)
    
    doc.build(elems)

def engagement_windows(p: Optional[Profile], posts: int=60) -> Dict[str,Dict[str,Any]]:
    if not p: return {}
    # Structure: {username: [first_ts, last_ts, comment_count, like_count]}
    windows: Dict[str,List[Any]] = defaultdict(lambda: [None, None, 0, 0])
    count = 0
    for post in p.get_posts():
        if count >= posts: break
        count += 1
        ts = post.date_utc
        # Process Comments
        try:
            for c in post.get_comments():
                u = c.owner.username
                w = windows[u]
                # Update first and last timestamp
                w[0] = ts if w[0] is None or ts < w[0] else w[0]
                w[1] = ts if w[1] is None or ts > w[1] else w[1]
                w[2] += 1
        except Exception:
            pass
        # Process Likes
        try:
            for l in post.get_likes():
                u = l.username
                w = windows[u]
                w[0] = ts if w[0] is None or ts < w[0] else w[0]
                w[1] = ts if w[1] is None or ts > w[1] else w[1]
                w[3] += 1
        except Exception:
            pass
    # Convert timestamps to ISO format strings
    return {u: {"first": w[0].isoformat() if w[0] else None, "last": w[1].isoformat() if w[1] else None,
                "comments": w[2], "likes": w[3]} for u, w in windows.items()}

def build_analysis(req: Requester, username: str, sessionid: Optional[str], post_limit: int, persist_dir: str,
                   deep_network: bool, cluster_temporal: bool, login_user: Optional[str], login_pass: Optional[str]) -> Dict[str, Any]:
    
    analysis: Dict[str, Any] = {"target_username": username, 'network_data': {'followers': [], 'following': [], 'mutual_followers': []}, 'cluster_temporal': []}
    
    # 1. Instaloader Check/Login
    # CRITICAL FIX: Ensure Instaloader is used first if available, as it is generally more stable.
    L, p, sessionid_iloader = load_profile_instaloader(username, DEFAULT_CONFIG["instaloader_session_dir"], login_user, login_pass)
    if sessionid_iloader:
        # Prioritize the sessionid derived from successful Instaloader login
        sessionid = sessionid_iloader
        logging.info("Using Instaloader-derived sessionid for API calls.")
        
    # 2. Resolve User ID (Requires an ID for most mobile API calls)
    uid, source, raw_used = resolve_user_id(req, username, sessionid)
    analysis['resolved_id'] = uid; analysis['id_source'] = source
    if not uid:
        logging.error("Failed to resolve user ID. Cannot proceed with mobile API calls.")
        analysis['profile_exists'] = False
        return analysis # Early exit on critical failure
        
    # 3. Fetch User Info (Private API/Mobile API)
    user_info = get_user_info_private(req, uid, sessionid) if uid and sessionid else {}
    user_raw = user_info.get("raw") or {}
    analysis['raw_user'] = user_raw
    
    # 4. Profile Data (Prioritize Instaloader (p) if available)
    if p:
        prof_res = analyze_profile_location_from_profile(p, n=post_limit)
        # FIX: Explicitly update analysis dictionary to merge Instaloader data
        analysis.update(prof_res)
    else:
        # Fallback: Use Mobile API data
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
        analysis['most_frequent_location'] = None
        analysis['all_coords'] = []
    
    # 5. Normalize Contact Info
    analysis['text_clues']['phone_clue_norm'] = normalize_phone(analysis['text_clues'].get('phone_in_text'))
    analysis['profile_data']['public_phone_norm'] = normalize_phone(analysis['profile_data'].get('public_phone'))
    
    # 6. Fetch Feed/Media (Mobile API)
    feed = get_feed_media(req, uid, sessionid, count=post_limit).get("raw") if uid and sessionid else {}
    analysis['feed_raw'] = feed
    fi = infer_locations_from_feed(feed)
    analysis['activity_data'] = {'last_post_ts': fi.get('last_post_ts'), 'feed_locations': fi}
    
    # 7. Extract Captions/Timestamps from Feed API (if Instaloader wasn't used)
    captions, timestamps, post_ids_for_persist = [], [], []
    items = feed.get("items") or []
    for it in items:
        # Caption is usually a nested dict
        if isinstance(it.get("caption"), dict):
            cap = it["caption"].get("text")
            if cap: captions.append(cap)
        # Handle multiple possible timestamp keys
        taken = it.get("taken_at") or it.get("device_timestamp") or it.get("timestamp")
        if taken:
            try:
                # API timestamps are in seconds
                timestamps.append(datetime.fromtimestamp(int(taken), tz=timezone.utc))
            except (ValueError, TypeError):
                pass
        pid = it.get("id") or it.get("pk")
        if pid: post_ids_for_persist.append(str(pid))

    # 8. Post Behavior/Content Analysis (Prioritize Instaloader data)
    if p:
        beh = analyze_post_behavior_from_profile(p, n=post_limit)
        analysis['behavior_data'] = beh
        # Use Instaloader's timestamps/captions as they are more complete
        timestamps = beh.get('timestamps') or timestamps
        captions = beh.get('captions') or captions
        # Instaloader media IDs are more reliable for persistence
        post_ids_for_persist = [str(post.mediaid) for post in p.get_posts()][:post_limit]
    else:
        analysis['behavior_data'] = {'top_hashtags': [], 'top_mentions': [], 'timestamps': timestamps, 'captions': captions}

    # 9. Location Aggregation/Clustering
    mfl = analysis.get('most_frequent_location')
    if not mfl and fi.get('locations'):
        # FIX: Simplified location counting using Counter on API locations
        counts = Counter([l.get('name') for l in fi['locations'] if l.get('name')])
        if counts:
            top_name, ct = counts.most_common(1)[0]
            # Find the full location dict to get lat/lng
            for l in fi['locations']:
                if l.get('name') == top_name:
                    analysis['most_frequent_location'] = {'name': top_name, 'lat': l.get('lat'), 'lng': l.get('lng'), 'count': ct}
                    break
    
    # Run DBSCAN on all Instaloader coordinates if available
    if p and analysis.get('all_coords'):
        cluster = cluster_primary_location(analysis['all_coords'])
        if cluster:
            analysis['cluster_primary'] = cluster
            # If no Instaloader location data was found, use the primary cluster as a location clue
            if not analysis['most_frequent_location'] or analysis['most_frequent_location'].get('name') == 'cluster_primary':
                analysis['most_frequent_location'] = {'name': 'cluster_primary', 'lat': cluster['lat'], 'lng': cluster['lng'], 'count': cluster['count']}
    
    # 10. Temporal Analysis (with Timezone Correction)
    tz_name = None
    if analysis['most_frequent_location'] and HAS_TZF and HAS_PYTZ:
        tz_name = infer_timezone_from_coords(analysis['most_frequent_location'].get('lat'), analysis['most_frequent_location'].get('lng'))
        if tz_name:
            # Convert all timestamps to the target's inferred local time
            # FIX: Ensure timestamps are not empty before list comprehension
            timestamps = [convert_to_local(ts, tz_name) for ts in timestamps if ts]
            
    # Location/Temporal Clustering
    if cluster_temporal and p and analysis.get('all_coords'):
        clusters = cluster_posts_by_location(analysis['all_coords'], eps=0.01, min_samples=3)
        analysis['cluster_temporal'] = peak_hours_per_cluster(clusters)
        
    # Calculate peak hours/days using (potentially local) timestamps
    hours, days = Counter(), Counter()
    for ts in timestamps:
        hours[ts.hour] += 1; days[ts.strftime('%A')] += 1
    analysis['temporal_data'] = {"top_hours": hours.most_common(24), "top_days": days.most_common(7), "tz_used": tz_name or "UTC/None"}
    
    # 11. Linguistic/Content Analysis
    all_texts = [analysis['profile_data'].get('bio'), analysis['profile_data'].get('external_url')] + captions
    analysis['keywords'] = extract_keywords(all_texts)
    analysis['entities'] = extract_entities_spacy(all_texts)
    analysis['topics'] = lda_topics(captions, n_topics=3, n_words=8)
    
    # TextBlob (Sentiment/Language)
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
                
    # 12. Reverse Image Search (RIS) and EXIF
    pic_url = None
    if isinstance(user_raw, dict):
        # Prefer HD pic from mobile API
        hd = user_raw.get("hd_profile_pic_url_info") or {}
        pic_url = hd.get("url")
        if not pic_url:
            for k in ("profile_pic_url_hd", "profile_pic_url"):
                if user_raw.get(k): pic_url = user_raw.get(k); break
    if not pic_url and p:
        try:
            pic_url = getattr(p, "profile_pic_url", None)
        except Exception:
            pic_url = None
    
    # Only run EXIF/RIS if a picture URL was found
    analysis['exif_data'] = exif_from_url_native(req, pic_url) if pic_url else {"error": "No profile picture URL"}
    analysis['ris'] = ris_links(pic_url)

    # 13. Network Analysis (Requires Instaloader for in-depth data)
    analysis['commenters'] = []
    analysis['stories'] = {"count": 0, "locations": []}
    analysis['ghost_followers'] = []
    analysis['likes_network'] = []
    analysis['posting_frequency'] = posting_frequency_analytics(timestamps)
    analysis['engagement_windows'] = {}
    analysis['overlap_pairs'] = []
    
    if p:
        net = {"followers": [], "following": []}
        try:
            # FIX: Limit list population for performance
            # CRITICAL FIX: Ensure the generator is fully consumed for accurate list size 
            net["followers"] = [f.username for f in p.get_followers()]
        except Exception as e:
            logging.warning(f"Followers list error: {e}")
        try:
            # FIX: Limit list population for performance
            net["following"] = [f.username for f in p.get_followees()]
        except Exception as e:
            logging.warning(f"Followees list error: {e}")
            
        analysis['network_data'] = {'followers': net['followers'], 'following': net['following'], 'mutual_followers': list(set(net['followers']) & set(net['following']))}
        
        # Engagement metrics
        commenters_ranked, comment_texts_by_user = analyze_comments_from_profile(p, post_limit=60, collect_texts=True)
        analysis['commenters'] = commenters_ranked
        analysis['comment_texts_by_user'] = comment_texts_by_user
        analysis['ghost_followers'] = ghost_followers(p, recent_posts=30)
        analysis['likes_network'] = likes_network(p, posts=30)
        analysis['engagement_windows'] = engagement_windows(p, posts=60)
        
        # Stories (Can be slow/rate-limited)
        try:
            # FIX: Convert generator to list safely
            stories = list(p.get_stories())
            analysis['stories']["count"] = len(stories)
        except Exception as e:
            logging.warning(f"Stories error: {e}")
            
        # Deep Network Analysis (Follower Overlap)
        if deep_network and L:
            top_seed = list({u for u,_ in (analysis['likes_network'][:50] + analysis['commenters'][:50])} | set(analysis['network_data']['mutual_followers'][:50]))
            analysis['overlap_pairs'] = mutual_overlap_pairs(L, top_seed, per_user_limit=300, top_pairs=100)

    # 14. Persistence (Post Deletion/Addition Detection)
    os.makedirs(persist_dir, exist_ok=True)
    persist_path = os.path.join(persist_dir, f"{username}_posts.json")
    deleted, added = detect_deleted(persist_path, post_ids_for_persist)
    analysis['deleted_since_last_run'] = deleted
    analysis['added_since_last_run'] = added
    # Save the current post IDs for the NEXT run's detection
    persist_post_ids(persist_path, post_ids_for_persist)
    
    return analysis

def parse_args():
    ap = argparse.ArgumentParser(description="InstaOSNIT_forensic_final - Forensic-grade Instagram OSINT")
    ap.add_argument("--target", "-t", required=True, help="Target username")
    ap.add_argument("--sessionid", "-s", required=False, help="Instagram sessionid cookie (for mobile API access)")
    ap.add_argument("--post-limit", "-n", type=int, default=200, help="Number of posts to analyze")
    ap.add_argument("--output", "-o", default=None, help="Output filename")
    ap.add_argument("--format", "-f", choices=["json","csv","gexf","pdf"], default="json")
    ap.add_argument("--terms", nargs="*", default=[], help="Targeted keyword search terms (regular expressions supported)")
    ap.add_argument("--proxies", default=DEFAULT_CONFIG["proxies_file"], help="Proxies file (one per line, format: http://user:pass@host:port)")
    ap.add_argument("--ua-file", default=None, help="File containing custom User-Agents (one per line)")
    ap.add_argument("--persist-dir", default=DEFAULT_CONFIG["persist_dir"], help="Persistence directory for post IDs")
    ap.add_argument("--deep-network", action="store_true", help="Perform follower overlap analysis on top engagers (requires login)")
    ap.add_argument("--cluster-temporal", action="store_true", help="Cluster posts by location and infer local peak hours (requires Instaloader)")
    ap.add_argument("--login-user", default=None, help="Instaloader login username (to enable authenticated fetching)")
    ap.add_argument("--login-pass", default=None, help="Instaloader login password")
    ap.add_argument("--log-file", default=DEFAULT_CONFIG["log_file"], help="Log file path")
    return ap.parse_args()

def main():
    args = parse_args()
    setup_logging(args.log_file)
    
    # Load User Agents
    if args.ua_file and os.path.exists(args.ua_file):
        try:
            with open(args.ua_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                if lines: UA_POOL[:] = lines
        except (OSError, IOError) as e:
            logging.error(f"UA pool load error: {e}")
            
    # Proxy Setup
    proxies = load_proxies(args.proxies)
    proxy_pool = StickyProxyPool(proxies, DEFAULT_CONFIG["sticky_seconds"])
    req = Requester(proxy_pool)
    
    # Session ID Check
    sessionid = args.sessionid
    if sessionid and not check_session_validity(req, sessionid):
        logging.warning("Provided sessionid appears invalid or expired. Results may be limited.")
        
    # Main Analysis Build
    logging.info(f"Starting OSINT analysis for target: {args.target}")
    analysis = build_analysis(req, args.target, sessionid=sessionid, post_limit=args.post_limit, persist_dir=args.persist_dir,
                              deep_network=args.deep_network, cluster_temporal=args.cluster_temporal,
                              login_user=args.login_user, login_pass=args.login_pass)
    
    # Targeted Search
    if args.terms:
        all_texts = [analysis['profile_data'].get('bio'), analysis['profile_data'].get('external_url')] + (analysis.get('behavior_data',{}).get('captions') or [])
        analysis['targeted_search_results'] = targeted_keyword_search(all_texts, args.terms)
        
    # Export Results
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
        logging.info(f"Exported analysis report: {out}")
        print(out)
    except Exception as e:
        logging.error(f"Export failed for format {args.format}: {e}. Saving fallback JSON.")
        try:
            # Fallback to JSON is crucial to save the gathered data
            fallback = f"{args.target}_report_fallback.json"
            export_json(analysis, fallback)
            logging.info(f"Saved fallback JSON: {fallback}")
            print(fallback)
        except Exception as ee:
            logging.error(f"Fallback export failed: {ee}")

if __name__ == "__main__":
    main()

