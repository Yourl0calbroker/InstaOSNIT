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
        logging.error(f"resolve_
