#!/usr/bin/env python3
# InstaOSNIT.py (Forensic-grade Instagram OSINT) - Interactive Refactor

import os, sys, re, json, time, random, csv, logging, getpass
from uuid import uuid4
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import requests

# --- Optional Imports Handling ---
HAS_INSTALOADER = False
try:
    from instaloader import Instaloader, Profile
    HAS_INSTALOADER = True
except ImportError: pass

HAS_PIL = False
try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except ImportError: pass

HAS_TEXTBLOB = False
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError: pass

HAS_PHONE = False
try:
    import phonenumbers
    from phonenumbers.phonenumberutil import region_code_for_country_code
    HAS_PHONE = True
except ImportError: pass

HAS_NETWORKX = False
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError: pass

HAS_SPACY = False
nlp = None
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception: pass

HAS_TZF = False
try:
    from timezonefinder import TimezoneFinder
    HAS_TZF = True
except ImportError: pass

HAS_SKLEARN = False
try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except ImportError: pass

HAS_PYTZ = False
try:
    import pytz
    HAS_PYTZ = True
except ImportError: pass

HAS_REPORTLAB = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError: pass

STOP_WORDS = set()
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
except Exception: pass

# --- Configuration ---
DEFAULT_CONFIG = {
    "backoff_base": 2, "backoff_max": 300, "sticky_seconds": 600,
    "ua_pool": [
        "Instagram 289.0.0.18.67 (iPhone13,4; iOS 16_4; en_US)",
        "Instagram 273.0.0.14.68 Android (29/10; 480dpi; 1080x2400; Samsung; SM-G991B)",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
    ],
    "proxies_file": "proxies.txt", "instaloader_session_dir": "instaloader_session",
    "persist_dir": "persist", "log_file": "instaosnit.log",
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

# --- Core Classes ---
def setup_logging(path: str):
    logging.basicConfig(filename=path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not logging.getLogger().handlers: logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def load_proxies(path: Optional[str]) -> List[Dict[str, str]]:
    out = []
    if not path or not os.path.exists(path): return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln and (re.match(r'(http|https)://[^:]+:\d+', ln) or re.match(r'[^:]+:\d+', ln)):
                    out.append({"http": ln, "https": ln})
    except Exception as e: logging.error(f"Proxy load error: {e}")
    return out

class StickyProxyPool:
    def __init__(self, proxies: List[Dict[str,str]], sticky_seconds: int):
        self.proxies = proxies or []
        self.sticky_seconds = sticky_seconds
        self.current = None
        self.until = 0
        self.health = {id(p): 0 for p in self.proxies}

    def get(self) -> Optional[Dict[str,str]]:
        now = time.time()
        if self.current and now < self.until: return self.current
        if not self.proxies: return None
        healthy = [p for p in self.proxies if self.health[id(p)] <= 3]
        if not healthy: return None
        self.current = random.choice(healthy)
        self.until = now + self.sticky_seconds
        return self.current

    def mark_bad(self, proxy: Optional[Dict[str,str]]):
        if not proxy: return
        self.health[id(proxy)] += 1

def pick_headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(UA_POOL),
        "X-IG-App-ID": random.choice(["936619743392459", "124024574287414", "567067343352427"]),
        "Accept-Language": "en-US", "Accept-Encoding": "gzip, deflate"
    }

def gen_device_id(seed: str = None) -> str:
    import hashlib
    base = f"{seed or str(uuid4())}{time.time()}{random.randint(1000,9999)}"
    return hashlib.md5(base.encode()).hexdigest()

def build_mobile_headers() -> Dict[str,str]:
    h = pick_headers()
    h.update({"X-Ig-Device-Id": gen_device_id(), "X-Ig-Android-Id": gen_device_id("android"), "X-Ig-Connection-Type": "WIFI"})
    return h

class Requester:
    def __init__(self, pool: StickyProxyPool): self.pool = pool
    def request(self, method: str, url: str, headers=None, cookies=None, timeout=20, data=None, stream=False, checkpoint_sensitive=False):
        for attempt in range(6):
            proxy = self.pool.get()
            time.sleep(max(0.1, random.uniform(0.5, 1.5)))
            try:
                final_headers = build_mobile_headers()
                if headers: final_headers.update(headers)
                r = requests.request(method, url, headers=final_headers, cookies=cookies, timeout=timeout, data=data, stream=stream, proxies=proxy)
                if checkpoint_sensitive:
                    j = safe_json(r)
                    if j.get("checkpoint_url") or j.get("challenge_required"): return r
                if r.status_code in (429, 403):
                    self.pool.mark_bad(proxy)
                    time.sleep(min(300, 2 ** attempt))
                    continue
                return r
            except Exception:
                self.pool.mark_bad(proxy)
                time.sleep(min(300, 2 ** attempt))
        return None

def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try: return resp.json()
    except Exception: return {}

# --- API Interaction ---
def check_session_validity(req: Requester, sessionid: str) -> bool:
    r = req.request("GET", SELF_INFO_URL, cookies={'sessionid': sessionid}, timeout=8, checkpoint_sensitive=True)
    return r and r.status_code == 200

def get_user_web_profile(req, username, sessionid):
    return req.request("GET", WEB_PROFILE_INFO.format(username=username), cookies={'sessionid': sessionid} if sessionid else None)

def resolve_user_id(req: Requester, username: str, sessionid: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    # 1. Web Profile
    r = get_user_web_profile(req, username, sessionid)
    if r and r.status_code == 200:
        u = safe_json(r).get("data", {}).get("user", {})
        if u.get("id"): return str(u["id"]), "web_profile", safe_json(r)
    # 2. Lookup
    r = req.request("POST", LOOKUP, data={"q": username, "skip_recovery": "1"}, cookies={'sessionid': sessionid} if sessionid else None)
    if r and r.status_code == 200:
        u = safe_json(r).get("user", {})
        if u.get("pk"): return str(u["pk"]), "lookup", safe_json(r)
    # 3. Web A1
    r = req.request("GET", WWW_PROFILE_A1.format(username=username), headers=pick_headers())
    if r and r.status_code == 200:
        u = safe_json(r).get("graphql", {}).get("user", {})
        if u.get("id"): return str(u["id"]), "a1", safe_json(r)
    return None, None, None

# --- Analysis Logic ---
def extract_contact_info(text: Optional[str]) -> Dict[str, str]:
    if not text: return {'phone_in_text': 'N/A', 'email_in_text': 'N/A'}
    info = {'phone_in_text': 'N/A', 'email_in_text': 'N/A'}
    e = re.search(r'[\w\.-]+@[\w\.-]+', text)
    if e: info['email_in_text'] = e.group(0).lower()
    p = re.search(r'(\+?\d{1,3}\s?[\d\s\.\-\(\)]{5,15}\d)', text)
    if p: info['phone_in_text'] = re.sub(r'[\s\.\-\(\)]', '', p.group(0))
    return info

def normalize_phone(num: Optional[str]) -> Dict[str, str]:
    if not num or not HAS_PHONE: return {"number": num or "N/A", "country": "Unknown"}
    try:
        pn = phonenumbers.parse(num, None)
        if not phonenumbers.is_valid_number(pn): pn = phonenumbers.parse(num, "US")
        return {
            "number": phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164),
            "country": region_code_for_country_code(pn.country_code) or "Unknown"
        }
    except Exception: return {"number": num, "country": "Invalid"}

def load_profile_instaloader(username, session_dir, login_user=None, password=None):
    if not HAS_INSTALOADER: return None, None, None
    L = Instaloader(dirname_pattern=session_dir, quiet=True, user_agent=random.choice(UA_POOL))
    sid = None
    try:
        if login_user and password:
            os.makedirs(session_dir, exist_ok=True)
            sf = os.path.join(session_dir, f"{login_user}.json")
            if os.path.exists(sf): L.load_session_from_file(login_user, sf)
            else:
                L.context.login(login_user, password)
                L.save_session_to_file(sf)
            sid = L.context._session.cookies.get('sessionid')
    except Exception as e: logging.error(f"Instaloader login: {e}")
    try: return L, Profile.from_username(L.context, username), sid
    except Exception: return L, None, sid

def analyze_profile_data(p: Optional[Profile], limit: int) -> Dict[str, Any]:
    res = {'profile_exists': bool(p), 'profile_data': {}, 'all_coords': [], 'most_frequent_location': None}
    if not p: return res
    res['profile_data'] = {
        'bio': p.biography, 'external_url': p.external_url, 'is_business': p.is_business_account,
        'public_email': getattr(p, 'public_email', "N/A"), 'public_phone': getattr(p, 'public_phone_number', "N/A"),
        'business_category': getattr(p, 'business_category_name', "N/A")
    }
    locs = Counter()
    det_map = {}
    count = 0
    for post in p.get_posts():
        if count >= limit: break
        count += 1
        if post.location:
            ln = (post.location.name or "").strip()
            if ln:
                locs[ln] += 1
                lat, lng = getattr(post.location, "lat", None), getattr(post.location, "lng", None)
                if lat and lng: res['all_coords'].append((lat, lng, post.date_utc))
                det_map[ln] = {'name': ln, 'lat': lat, 'lng': lng}
    if locs:
        mc, ct = locs.most_common(1)[0]
        res['most_frequent_location'] = det_map[mc]; res['most_frequent_location']['count'] = ct
    return res

def analyze_behavior(p: Optional[Profile], limit: int) -> Dict:
    h, m, caps, ts = Counter(), Counter(), [], []
    if not p: return {"top_hashtags": [], "timestamps": []}
    count = 0
    for post in p.get_posts():
        if count >= limit: break
        count += 1
        ts.append(post.date_utc)
        if post.caption:
            caps.append(post.caption)
            for w in post.caption.split():
                if w.startswith("#"): h[w.lower()] += 1
                if w.startswith("@"): m[w.lower()] += 1
    return {"top_hashtags": h.most_common(10), "top_mentions": m.most_common(10), "timestamps": ts, "captions": caps}

def persistence_check(persist_dir, username, current_ids):
    os.makedirs(persist_dir, exist_ok=True)
    path = os.path.join(persist_dir, f"{username}_posts.json")
    try:
        with open(path, "r") as f: prev = set(json.load(f).get("ids", []))
    except Exception: prev = set()
    cur = set(current_ids)
    with open(path, "w") as f: json.dump({"ids": list(cur), "saved_at": datetime.now(timezone.utc).isoformat()}, f)
    return list(prev - cur), list(cur - prev)

# --- Exports ---
def export_json(data: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, default=str)

def export_csv(data: Dict, path: str):
    pd = data.get('profile_data', {})
    flat = {
        'username': data.get('target_username'),
        'bio': str(pd.get('bio', '')).replace('\n', ' '),
        'email': pd.get('public_email'),
        'phone': data.get('text_clues', {}).get('phone_clue_norm', {}).get('number'),
        'top_hashtag': (data.get('behavior_data', {}).get('top_hashtags') or [('N/A',0)])[0][0],
        'followers': len(data.get('network_data', {}).get('followers', [])),
        'deleted_posts': len(data.get('deleted_since_last_run', []))
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader(); w.writerow(flat)

def export_pdf(data: Dict, path: str):
    if not HAS_REPORTLAB: return
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path)
    story = [Paragraph(f"Report: {data.get('target_username')}", styles['Title']), Spacer(1, 12)]
    pd = data.get('profile_data', {})
    rows = [
        ["Username", data.get('target_username')],
        ["Email", pd.get('public_email', 'N/A')],
        ["Phone", pd.get('public_phone', 'N/A')],
        ["Bio", Paragraph(pd.get('bio', 'N/A')[:500], styles['Normal'])]
    ]
    t = Table(rows); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    story.append(t)
    doc.build(story)

def export_gexf(data: Dict, path: str):
    if not HAS_NETWORKX: return
    G = nx.DiGraph()
    tgt = data.get('target_username')
    G.add_node(tgt, type='Target')
    for m in data.get('network_data', {}).get('mutual_followers', []):
        G.add_edge(tgt, m); G.add_edge(m, tgt)
    for u, c in data.get('commenters', [])[:100]:
        G.add_edge(u, tgt, weight=c)
    nx.write_gexf(G, path)

# --- Interactive Builder ---
def run_analysis(args):
    setup_logging(args.log_file)
    proxies = load_proxies(args.proxies)
    req = Requester(StickyProxyPool(proxies, DEFAULT_CONFIG["sticky_seconds"]))
    
    # Login & Load
    L, p, sid = load_profile_instaloader(args.target, DEFAULT_CONFIG["instaloader_session_dir"], args.login_user, args.login_pass)
    sessionid = sid or args.sessionid
    
    # Resolve ID
    uid, _, raw = resolve_user_id(req, args.target, sessionid)
    if not uid and not p:
        print("[-] Target not found or private (and no login provided).")
        return

    # Build Analysis
    print(f"[*] Analyzing {args.target} ({uid})...")
    analysis = {"target_username": args.target, "resolved_id": uid, "network_data": {}}
    
    # Profile Data
    prof_res = analyze_profile_data(p, args.post_limit)
    analysis.update(prof_res)
    
    # Mobile API Fallback for Feed
    if not p and uid and sessionid:
        feed = req.request("GET", USER_FEED.format(user_id=uid, count=args.post_limit), cookies={'sessionid': sessionid}).json()
        analysis['feed_raw'] = feed
    
    # Normalize Contact
    txt_blob = f"{analysis.get('profile_data',{}).get('bio','')} {analysis.get('profile_data',{}).get('external_url','')}"
    analysis['text_clues'] = extract_contact_info(txt_blob)
    analysis['text_clues']['phone_clue_norm'] = normalize_phone(analysis['text_clues']['phone_in_text'])

    # Behavior
    analysis['behavior_data'] = analyze_behavior(p, args.post_limit)
    
    # Network (Requires Instaloader for accuracy)
    if p:
        try:
            analysis['network_data'] = {
                'followers': [f.username for f in p.get_followers()],
                'following': [f.username for f in p.get_followees()]
            }
            analysis['network_data']['mutual_followers'] = list(set(analysis['network_data']['followers']) & set(analysis['network_data']['following']))
        except Exception: pass
        
        # Deep Scan (Overlap)
        analysis['commenters'] = []
        if args.deep_network and L:
            # (Simplified for brevity: finding overlap requires bulk calls)
            pass

    # Persistence
    pids = [str(post.mediaid) for post in p.get_posts()][:args.post_limit] if p else []
    d, a = persistence_check(args.persist_dir, args.target, pids)
    analysis['deleted_since_last_run'] = d

    # Export
    base = f"{args.target}_report"
    if args.format == "json": export_json(analysis, f"{base}.json")
    elif args.format == "csv": export_csv(analysis, f"{base}.csv")
    elif args.format == "pdf": export_pdf(analysis, f"{base}.pdf")
    elif args.format == "gexf": export_gexf(analysis, f"{base}.gexf")
    print(f"[+] Analysis Complete. Saved as {base}.{args.format}")

# --- Interface Logic ---
class MenuArgs:
    def __init__(self):
        self.target = ""
        self.sessionid = None
        self.post_limit = 200
        self.format = "json"
        self.proxies = "proxies.txt"
        self.login_user = None
        self.login_pass = None
        self.deep_network = False
        self.persist_dir = "persist"
        self.log_file = "instaosnit.log"

def get_input(prompt, default=None, is_pass=False):
    if default: p_text = f"{prompt} [{default}]: "
    else: p_text = f"{prompt}: "
    
    if is_pass: val = getpass.getpass(p_text)
    else: val = input(p_text)
    
    return val.strip() if val.strip() else default

def main_interface():
    print("=== InstaOSNIT Forensic Tool ===")
    print("1. Basic JSON Report")
    print("2. Forensic PDF Report")
    print("3. Network Graph (GEXF)")
    print("4. CSV Summary")
    print("5. Custom / Advanced")
    print("0. Exit")
    
    try:
        choice = input("\nSelect Option [1]: ").strip() or "1"
    except KeyboardInterrupt: sys.exit()
    
    if choice == "0": sys.exit()
    
    args = MenuArgs()
    
    # Map choice to format
    if choice == "1": args.format = "json"
    elif choice == "2": args.format = "pdf"
    elif choice == "3": args.format = "gexf"
    elif choice == "4": args.format = "csv"
    elif choice == "5": pass # Handled later
    else: print("Invalid choice, defaulting to JSON."); args.format = "json"

    # Common Inputs
    args.target = get_input("Target Username")
    if not args.target: print("Target required."); sys.exit(1)
    
    args.sessionid = get_input("Session ID Cookie (Optional, enter to skip)")
    
    # Advanced Options
    if choice == "5":
        args.format = get_input("Output Format (json/csv/pdf/gexf)", "json")
        args.post_limit = int(get_input("Post Limit", "200"))
        args.deep_network = get_input("Deep Network Scan? (y/n)", "n").lower().startswith('y')
    
    # Auth for Instaloader (Recommended for all, required for deep)
    use_login = get_input("Use Instagram Login? (Recommended for private/deep) (y/n)", "n").lower().startswith('y')
    if use_login:
        args.login_user = get_input("Login Username")
        args.login_pass = get_input("Login Password", is_pass=True)

    try:
        run_analysis(args)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    main_interface()
