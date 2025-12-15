#!/usr/bin/env python3
# InstaOSNIT.py
# Combined BrokerOSNIT (login + interactive mobile API) + advanced OSINT (linguistics, EXIF, keywords, correlation, exports)

from uuid import uuid4
import os
import sys
import re
import time
import queue
import threading
import json
from collections import Counter
from datetime import datetime, timezone

import requests
import stdiomask

# Optional libs
try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import phonenumbers
    from phonenumbers.phonenumberutil import region_code_for_country_code
    import pycountry
    HAS_PHONE = True
except Exception:
    HAS_PHONE = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except Exception:
    HAS_TEXTBLOB = False

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    HAS_NLTK = True
except Exception:
    stop_words = set()
    HAS_NLTK = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False

# ------------ Config ------------
OTP_TIMEOUT = 60
ANIM_BAR_LEN = 30
ANIM_STEP_DELAY = 0.12
MESSAGE = "Your_l0cal_broker"
LOGIN_URL = "https://i.instagram.com/api/v1/accounts/login/"
TWO_FACTOR_URL = "https://i.instagram.com/api/v1/accounts/login_two_factor/"
WEB_PROFILE_INFO = "https://i.instagram.com/api/v1/users/web_profile_info/?username={username}"
USER_INFO = "https://i.instagram.com/api/v1/users/{user_id}/info/"
USER_FEED = "https://i.instagram.com/api/v1/feed/user/{user_id}/?count={count}"
LOOKUP = "https://i.instagram.com/api/v1/users/lookup/"
USERNAME_INFO = "https://i.instagram.com/api/v1/users/{username}/usernameinfo/"
WWW_PROFILE_A1 = "https://www.instagram.com/{username}/?__a=1&__d=dis"
# ---------------------------------

GREEN = "\033[92m"
RESET = "\033[0m"
CLEAR_LINE = "\033[K"

# ---- utilities ----

def label_cookie_name(name):
    n = (name or "").lower()
    if any(k in n for k in ("session", "sess", "sid", "sessionid")):
        return "session"
    if any(k in n for k in ("auth", "token", "jwt", "access")):
        return "auth"
    if any(k in n for k in ("csrf", "csrftoken")):
        return "csrf"
    if any(k in n for k in ("mid", "mid_")):
        return "mid"
    if any(k in n for k in ("ig_", "ds_user")):
        return "instagram"
    if any(k in n for k in ("lang", "locale")):
        return "locale"
    return "other"

def animate_loading(stop_event, message=MESSAGE, bar_len=ANIM_BAR_LEN, step_delay=ANIM_STEP_DELAY):
    pos = 0
    width = max(3, bar_len // 4)
    revealed = 0
    try:
        while not stop_event.is_set():
            bar = [" "] * bar_len
            for i in range(width):
                idx = (pos + i) % bar_len
                bar[idx] = "="
            pos = (pos + 1) % bar_len
            bar_str = "[" + "".join(bar) + "]"
            line1 = f"{GREEN}{bar_str}{RESET}"
            if revealed < len(message):
                revealed += 1
            line2 = message[:revealed]
            sys.stdout.write("\r" + CLEAR_LINE + line1 + "\n" + CLEAR_LINE + line2 + "\n")
            sys.stdout.flush()
            sys.stdout.write("\033[2A")
            time.sleep(step_delay)
        sys.stdout.write("\r" + CLEAR_LINE + " " * (bar_len + 2) + "\n" + CLEAR_LINE + message + "\n")
        sys.stdout.flush()
    except Exception:
        stop_event.set()

def input_with_timeout(prompt, timeout, stop_event):
    q = queue.Queue()
    def reader(q):
        try:
            s = input(prompt)
            q.put(s)
        except Exception:
            q.put(None)
    th = threading.Thread(target=reader, args=(q,), daemon=True)
    th.start()
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        stop_event.set()
        return None

def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}

def pretty_print_json(obj, label=None):
    if label:
        print(f"\n--- {label} ---")
    try:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(str(obj))

def epoch_to_iso(ts):
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return str(ts)

def head_request_headers(url, cookies=None, headers=None, timeout=15):
    try:
        r = requests.head(url, cookies=cookies, headers=headers or {}, timeout=timeout, allow_redirects=True)
        return {
            "status_code": r.status_code,
            "content_type": r.headers.get("Content-Type"),
            "content_length": r.headers.get("Content-Length"),
            "last_modified": r.headers.get("Last-Modified"),
            "etag": r.headers.get("ETag"),
            "date": r.headers.get("Date"),
            "server": r.headers.get("Server"),
            "final_url": r.url
        }
    except Exception:
        return None

# ---- Instagram data collectors ----

def get_user_web_profile(username, sessionid=None):
    url = WEB_PROFILE_INFO.format(username=username)
    headers = {"User-Agent": "Mozilla/5.0 (compatible)", "x-ig-app-id": "936619743392459"}
    cookies = {'sessionid': sessionid} if sessionid else None
    try:
        r = requests.get(url, headers=headers, cookies=cookies, timeout=15)
    except Exception as e:
        return {"error": f"network: {e}"}
    if r.status_code == 404:
        return {"error": "not_found", "raw": safe_json(r)}
    return {"raw": safe_json(r), "status_code": r.status_code}

def get_user_info_private(user_id, sessionid):
    url = USER_INFO.format(user_id=user_id)
    headers = {"User-Agent": "Instagram 64.0.0.14.96"}
    try:
        r = requests.get(url, headers=headers, cookies={'sessionid': sessionid}, timeout=15)
    except Exception as e:
        return {"error": f"network: {e}"}
    if r.status_code == 429:
        return {"error": "rate_limit", "raw": safe_json(r)}
    return {"raw": safe_json(r), "status_code": r.status_code}

def get_feed_media(user_id, sessionid, count=12):
    url = USER_FEED.format(user_id=user_id, count=count)
    headers = {"User-Agent": "Instagram 64.0.0.14.96"}
    try:
        r = requests.get(url, headers=headers, cookies={'sessionid': sessionid}, timeout=20)
    except Exception as e:
        return {"error": f"network: {e}"}
    if r.status_code == 429:
        return {"error": "rate_limit", "raw": safe_json(r)}
    return {"raw": safe_json(r), "status_code": r.status_code}

def do_advanced_lookup(username, sessionid=None):
    data = "signed_body=SIGNATURE." + json.dumps({"q": username, "skip_recovery": "1"}, separators=(",", ":"))
    headers = {
        "Accept-Language": "en-US",
        "User-Agent": "Instagram 101.0.0.15.120",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-IG-App-ID": "124024574287414",
    }
    cookies = {'sessionid': sessionid} if sessionid else None
    try:
        r = requests.post(LOOKUP, headers=headers, data=data, cookies=cookies, timeout=15)
    except Exception as e:
        return {"error": f"network: {e}"}
    try:
        return {"raw": r.json(), "status_code": r.status_code}
    except Exception:
        return {"raw_text": r.text, "status_code": r.status_code}

def get_usernameinfo(username, sessionid=None):
    url = USERNAME_INFO.format(username=username)
    headers = {"User-Agent": "Instagram 64.0.0.14.96", "x-ig-app-id": "936619743392459"}
    try:
        r = requests.get(url, headers=headers, cookies={'sessionid': sessionid} if sessionid else None, timeout=15)
    except Exception as e:
        return {"error": f"network: {e}"}
    return {"status_code": r.status_code, "raw": safe_json(r), "text": r.text}

def get_www_profile_a1(username):
    url = WWW_PROFILE_A1.format(username=username)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
    except Exception as e:
        return {"error": f"network: {e}"}
    return {"status_code": r.status_code, "raw": safe_json(r), "text": r.text}

# ---- helpers for inference ----

def extract_profile_picture_info(user_raw, sessionid):
    url_candidates = []
    try:
        if isinstance(user_raw, dict):
            for path in (
                ("profile_pic_url_hd",),
                ("profile_pic_url",),
                ("hd_profile_pic_url_info", "url"),
            ):
                cur = user_raw
                for p in path:
                    if cur is None:
                        break
                    cur = cur.get(p) if isinstance(cur, dict) else None
                if isinstance(cur, str) and cur:
                    url_candidates.append(cur)
                elif isinstance(cur, dict) and cur.get("url"):
                    url_candidates.append(cur.get("url"))
    except Exception:
        pass
    url_candidates = [u for i, u in enumerate(url_candidates) if u and u not in url_candidates[:i]]
    results = []
    for u in url_candidates:
        hdrs = head_request_headers(u, cookies={'sessionid': sessionid} if sessionid else None)
        results.append({"url": u, "head": hdrs})
    return results

def infer_locations_from_feed(feed_raw):
    if not isinstance(feed_raw, dict):
        return {"locations": [], "last_post_ts": None}
    items = feed_raw.get("items") or []
    locations = []
    last_post_ts = None
    for it in items or []:
        try:
            loc = it.get("location")
            if loc:
                place = {
                    "name": loc.get("name"),
                    "address": loc.get("address", ""),
                    "pk": loc.get("pk") or loc.get("id"),
                    "lat": loc.get("lat"),
                    "lng": loc.get("lng")
                }
                locations.append(place)
            taken = it.get("taken_at") or it.get("device_timestamp") or it.get("timestamp")
            if taken:
                try:
                    ts = int(taken)
                except Exception:
                    ts = None
                if ts and (not last_post_ts or ts > last_post_ts):
                    last_post_ts = ts
        except Exception:
            continue
    uniq = []
    seen = set()
    for l in locations:
        key = (l.get("pk"), l.get("name"))
        if key not in seen:
            seen.add(key)
            uniq.append(l)
    return {"locations": uniq, "last_post_ts": last_post_ts}

def try_get_last_active(user_raw, feed_info):
    if isinstance(user_raw, dict):
        for k in ("last_activity_at", "last_online_time", "last_seen", "last_activity"):
            v = user_raw.get(k)
            if v:
                return v
    if feed_info and feed_info.get("last_post_ts"):
        return epoch_to_iso(feed_info["last_post_ts"])
    return None

def gather_linked_accounts(user_raw, lookup_raw):
    linked = {"facebook_pages": [], "connected_accounts": [], "external_urls": [], "emails": [], "phones": []}
    def scan_for_keys(obj):
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            lk = str(k).lower()
            try:
                if any(x in lk for x in ("fb", "facebook", "connected", "connected_accounts", "connected_instagram")):
                    linked["connected_accounts"].append({k: v})
                if "external_url" == lk or "external_urls" in lk or "external" in lk:
                    if isinstance(v, str) and v:
                        linked["external_urls"].append(v)
                    elif isinstance(v, list):
                        linked["external_urls"].extend([x for x in v if isinstance(x, str)])
                if "public_email" in lk or "email" in lk:
                    if isinstance(v, str) and v:
                        linked["emails"].append(v)
                if "public_phone" in lk or "phone" in lk:
                    if isinstance(v, str) and v:
                        linked["phones"].append(v)
                if isinstance(v, dict):
                    scan_for_keys(v)
                if isinstance(v, list):
                    for e in v:
                        if isinstance(e, dict):
                            scan_for_keys(e)
            except Exception:
                continue
    scan_for_keys(user_raw or {})
    if isinstance(lookup_raw, dict):
        for k, v in lookup_raw.items():
            lk = str(k).lower()
            if "obfus" in lk or "masked" in lk or "hidden" in lk:
                linked["connected_accounts"].append({k: v})
            if "email" in lk and isinstance(v, str):
                linked["emails"].append(v)
            if "phone" in lk and isinstance(v, str):
                linked["phones"].append(v)
            if "facebook" in lk or "fb" in lk:
                linked["facebook_pages"].append({k: v})
    for k in linked:
        seen = set()
        out = []
        for item in linked[k]:
            if isinstance(item, dict):
                tup = tuple(sorted(item.items()))
            else:
                tup = item
            if tup not in seen:
                seen.add(tup)
                out.append(item)
        linked[k] = out
    return linked

# ---- Robust ID resolution ----

def resolve_user_id(username, sessionid=None):
    try:
        res = get_user_web_profile(username, sessionid)
        raw = res.get("raw")
        if isinstance(raw, dict):
            u = raw.get("data", {}).get("user")
            if isinstance(u, dict):
                pk = u.get("id") or u.get("pk")
                if pk:
                    return str(pk), "web_profile_info", raw
    except Exception:
        pass
    try:
        res = do_advanced_lookup(username, sessionid)
        raw = res.get("raw") or {}
        if isinstance(raw, dict):
            user_node = None
            if raw.get("user"):
                user_node = raw.get("user")
            if not user_node:
                for k in ("users", "user", "data"):
                    if raw.get(k):
                        maybe = raw.get(k)
                        if isinstance(maybe, list) and maybe:
                            user_node = maybe[0]
                        elif isinstance(maybe, dict):
                            user_node = maybe
            if isinstance(user_node, dict):
                pk = user_node.get("pk") or user_node.get("id")
                if pk:
                    return str(pk), "users/lookup", raw
    except Exception:
        pass
    try:
        res = get_usernameinfo(username, sessionid)
        raw = res.get("raw")
        if isinstance(raw, dict):
            user_node = raw.get("user") or raw
            if isinstance(user_node, dict):
                pk = user_node.get("pk") or user_node.get("id")
                if pk:
                    return str(pk), "usernameinfo", raw
    except Exception:
        pass
    try:
        res = get_www_profile_a1(username)
        raw = res.get("raw")
        if isinstance(raw, dict):
            g = raw.get("graphql") or raw.get("data") or raw
            if isinstance(g, dict):
                u = g.get("user") or g.get("profile") or g
                if isinstance(u, dict):
                    pk = u.get("id") or u.get("pk")
                    if pk:
                        return str(pk), "www_profile_a1", raw
    except Exception:
        pass
    return None, None, None

# ---- Advanced analysis (from InstaOSNIT) ----

def extract_contact_info(text):
    info = {'phone_in_text': 'N/A', 'email_in_text': 'N/A'}
    try:
        e = re.search(r'[\w\.-]+@[\w\.-]+', text or "")
        if e:
            info['email_in_text'] = e.group(0).lower()
        p = re.search(r'(\+?\d{1,3}\s?\d{2,4}[-\s\.]?\d{2,4}[-\s\.]?\d{2,9})', text or "")
        if p:
            info['phone_in_text'] = re.sub(r'[\s\.\-\(\)]', '', p.group(0))
    except Exception:
        pass
    return info

def normalize_phone(num):
    if not num:
        return {"number": num, "country": "Unknown"}
    try:
        pn = phonenumbers.parse(num)
        cc = region_code_for_country_code(pn.country_code)
        return {"number": num, "country": cc}
    except Exception:
        return {"number": num, "country": "Unknown"}

def analyze_text_linguistics(texts):
    if not HAS_TEXTBLOB:
        return []
    res = []
    for t in texts:
        t = t or ""
        if not t.strip():
            continue
        try:
            b = TextBlob(t)
            try:
                lang = b.detect_language()
            except Exception:
                lang = "unknown"
            res.append({"text": t[:50], "lang": lang, "sentiment": b.sentiment.polarity})
        except Exception:
            continue
    return res

def extract_keywords(texts):
    words = []
    for t in texts:
        for w in re.findall(r'\b\w+\b', (t or "").lower()):
            if w not in stop_words and len(w) > 3:
                words.append(w)
    return Counter(words).most_common(10)

def targeted_keyword_search(texts, terms):
    if not terms:
        return {}
    hits = {}
    ft = "\n".join(t for t in texts if t)
    for term in terms:
        try:
            m = re.findall(term, ft, re.IGNORECASE)
            if m:
                hits[term] = len(m)
        except Exception:
            continue
    return hits

def exif_from_url_native(url, outfile="tmp_instaosnit.jpg"):
    if not HAS_PIL:
        return {"error": "PIL not available"}
    if not url:
        return {"error": "No URL"}
    try:
        r = requests.get(url, stream=True, timeout=15)
    except Exception as e:
        return {"error": f"network: {e}"}
    try:
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        img = Image.open(outfile)
        exif = img.getexif()
        data = {}
        for tag, val in exif.items():
            data[ExifTags.TAGS.get(tag, tag)] = val
        try:
            os.remove(outfile)
        except Exception:
            pass
        return data
    except Exception as e:
        return {"error": f"exif: {e}"}

def correlate_data(pd, tc, fl):
    corr = []
    if pd.get('public_email') not in (None, "N/A") and tc.get('email_in_text') == pd.get('public_email'):
        corr.append(("Email Match", "High"))
    if pd.get('public_phone') not in (None, "N/A") and tc.get('phone_in_text') == pd.get('public_phone'):
        corr.append(("Phone Match", "High"))
    phone_clue_country = tc.get('phone_clue_norm', {}).get('country')
    public_phone_country = pd.get('public_phone_norm', {}).get('country')
    if phone_clue_country != 'Unknown' and phone_clue_country == public_phone_country:
        corr.append(("Phone Country Match", "Medium"))
    if fl and pd.get('business_address') not in (None, "N/A"):
        corr.append(("Location vs Business Address", "High"))
    return corr

def resolve_external_url(url):
    if not url:
        return "N/A"
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        return r.url
    except Exception:
        return url

def check_email_leakage(email):
    if not email or email == "N/A":
        return {"status": "N/A"}
    return {"google_search": f"https://www.google.com/search?q=%22{email}%22", "status": "Manual Check Recommended"}

def export_results(data, filename, fmt="json"):
    if fmt == "json":
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[EXPORT] Data saved to {filename}")
    elif fmt == "csv":
        import csv
        flat = {
            'username': data.get('target_username'),
            'profile_exists': data.get('profile_exists'),
            'bio': data.get('profile_data', {}).get('bio', 'N/A'),
            'primary_location': (data.get('most_frequent_location') or {}).get('name', 'N/A'),
            'last_post_ts': data.get('activity_data', {}).get('last_post_ts', 'N/A'),
            'public_email': data.get('profile_data', {}).get('public_email', 'N/A'),
            'contact_email_clue': data.get('text_clues', {}).get('email_in_text', 'N/A'),
            'top_hashtag_1': (data.get('behavior_data', {}).get('top_hashtags') or [('N/A', 0)])[0][0]
        }
        flat.update({
            'followers_count': len(data.get('network_data', {}).get('followers', [])),
            'following_count': len(data.get('network_data', {}).get('following', [])),
            'mutuals_count': len(data.get('network_data', {}).get('mutual_followers', [])),
            'peak_hours_top3': ','.join([f"{h[0]}({h[1]})" for h in data.get('temporal_data', {}).get('top_hours', [])]),
            'peak_days_top3': ','.join([f"{d[0]}({d[1]})" for d in data.get('temporal_data', {}).get('top_days', [])]),
            'top_keyword': (data.get('keywords') or [('N/A', 0)])[0][0],
            'top_keyword_count': (data.get('keywords') or [('N/A', 0)])[0][1]
        })
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(flat.keys()))
                w.writeheader(); w.writerow(flat)
            print(f"\n[EXPORT] CSV saved to {filename}")
        except Exception as e:
            print(f"\n[ERROR] CSV export failed: {e}")

def export_gexf(data, filename):
    if not HAS_NETWORKX:
        print("\n[ERROR] NetworkX required for GEXF export.")
        return
    G = nx.DiGraph()
    target = data.get('target_username')
    mutuals = data.get('network_data', {}).get('mutual_followers', []) or []
    commenters = data.get('commenters', []) or []
    G.add_node(target, type='Target', label=target, size=10.0, color='#FF0000')
    for m in mutuals:
        G.add_node(m, type='Mutual Follower', label=m, size=5.0, color='#00CC00')
        G.add_edge(target, m, type='Follows', weight=2.0)
        G.add_edge(m, target, type='Follows', weight=2.0)
    for u, c in commenters[:50]:
        if u not in G: G.add_node(u, type='Commenter', label=u, size=4.0, color='#0066FF')
        G.add_edge(u, target, type='Comment', weight=float(c))
    base = filename if filename else f"{target}_report.json"
    out = base.replace('.json', '_network.gexf').replace('.csv', '_network.gexf')
    try:
        nx.write_gexf(G, out)
        print(f"\n[EXPORT] GEXF saved to {out}")
    except Exception as e:
        print(f"\n[ERROR] GEXF export failed: {e}")

# ---- UI / aggregation ----

def print_account_summary(user):
    print("\n===== Account Summary =====")
    def p(k, label=None):
        if k in user and user.get(k) not in (None, "", [], {}):
            lbl = label or k
            print(f"{lbl:22}: {user.get(k)}")
    p("username", "Username")
    p("full_name", "Full name")
    p("userID", "User ID")
    p("is_private", "Private")
    p("is_verified", "Verified")
    p("is_business", "Business")
    p("follower_count", "Followers")
    p("following_count", "Following")
    p("media_count", "Posts")
    if user.get("external_url"):
        print(f"{'External URL':22}: {user.get('external_url')}")
    bio = user.get("biography")
    if bio:
        print(f"{'Biography':22}:")
        for line in bio.splitlines():
            print(f"  {line}")
    hd = user.get("hd_profile_pic_url_info", {}) if isinstance(user.get("hd_profile_pic_url_info"), dict) else {}
    if hd.get("url"):
        print(f"{'Profile picture':22}: {hd.get('url')}")
    if user.get("public_email"):
        print(f"{'Public email':22}: {user.get('public_email')}")
    if user.get("public_phone_number") or user.get("public_phone_country_code"):
        ph = format_phone(user.get("public_phone_country_code", ""), user.get("public_phone_number", ""))
        if ph:
            print(f"{'Public phone':22}: {ph}")
    print("=" * 28)

def format_phone(public_phone_country_code, public_phone_number):
    if not public_phone_number:
        return None
    try:
        phonenr = f"+{public_phone_country_code} {public_phone_number}" if public_phone_country_code else str(public_phone_number)
        if HAS_PHONE:
            pn = phonenumbers.parse(phonenr)
            countrycode = region_code_for_country_code(pn.country_code)
            country = pycountry.countries.get(alpha_2=countrycode)
            return f"{phonenr} ({country.name})"
        return phonenr
    except Exception:
        return phonenr

def interactive_user_menu_full(user_raw, sessionid, cookie_dict, web_raw=None, lookup_raw=None, feed_raw=None, feed_media_meta=None, analysis=None):
    feed_info = infer_locations_from_feed(feed_raw) if feed_raw else {"locations": [], "last_post_ts": None}
    linked = gather_linked_accounts(user_raw, lookup_raw)
    pic_infos = extract_profile_picture_info(user_raw or web_raw or lookup_raw, sessionid)

    while True:
        print("\nSelect an option:")
        print("  1) Account Summary")
        print("  2) View Full Raw JSON (private API / user object)")
        print("  3) View Web Profile JSON (web_profile_info)")
        print("  4) Advanced Lookup (raw)")
        print("  5) Recent Media & Locations")
        print("  6) Extended Investigator")
        print("  7) View Cookies")
        print("  8) Export (JSON/CSV/GEXF)")
        print("  9) Exit")
        choice = input("Enter choice [1-9]: ").strip()
        if choice == "1":
            print_account_summary(user_raw)
        elif choice == "2":
            pretty_print_json(user_raw, label="User JSON (private API)")
        elif choice == "3":
            pretty_print_json(web_raw, label="Web profile JSON")
        elif choice == "4":
            pretty_print_json(lookup_raw, label="Advanced lookup JSON")
        elif choice == "5":
            if not feed_raw or not isinstance(feed_raw, dict):
                print("[!] No feed data available.")
                continue
            items = feed_raw.get("items") or []
            print(f"\nRecent media count shown: {len(items)} (first {min(12, len(items))})")
            for idx, it in enumerate(items[:12]):
                taken = it.get("taken_at") or it.get("device_timestamp") or it.get("timestamp")
                t_iso = epoch_to_iso(taken) if taken else "unknown"
                caption = None
                if isinstance(it.get("caption"), dict):
                    caption = it["caption"].get("text")
                media_id = it.get("id") or it.get("pk")
                print(f"\n[{idx+1}] id: {media_id}")
                print(f"    taken_at: {t_iso}")
                if it.get("location"):
                    loc = it.get("location")
                    print(f"    location: {loc.get('name')}  (lat:{loc.get('lat')}, lng:{loc.get('lng')})")
                display_url = None
                if isinstance(it.get("image_versions2"), dict):
                    candidates = it["image_versions2"].get("candidates", [])
                    if candidates:
                        display_url = candidates[0].get("url")
                if not display_url and isinstance(it.get("carousel_media"), list) and it["carousel_media"]:
                    cm = it["carousel_media"][0]
                    if isinstance(cm.get("image_versions2"), dict):
                        display_url = cm["image_versions2"].get("candidates", [{}])[0].get("url")
                if display_url:
                    print(f"    media_url: {display_url}")
                    if feed_media_meta and media_id and media_id in feed_media_meta:
                        print(f"    media_head: {feed_media_meta[media_id]}")
                if caption:
                    print(f"    caption: {caption[:160] + ('...' if len(caption) > 160 else '')}")
        elif choice == "6":
            print("\n=== Extended Investigator ===")
            print("\n- Linked / Connected Accounts & Contacts -")
            if any(linked.values()):
                if linked.get("connected_accounts"):
                    print("Connected accounts:")
                    pretty_print_json(linked.get("connected_accounts"))
                if linked.get("facebook_pages"):
                    print("Facebook pages / FB data:")
                    pretty_print_json(linked.get("facebook_pages"))
                if linked.get("external_urls"):
                    print("External URLs:")
                    for u in linked.get("external_urls"):
                        print("  - " + u)
                if linked.get("emails"):
                    print("Emails:")
                    for e in linked.get("emails"):
                        print("  - " + str(e))
                if linked.get("phones"):
                    print("Phones:")
                    for p in linked.get("phones"):
                        print("  - " + str(p))
            else:
                print("No linked accounts / contacts detected.")
            print("\n- Activity / Last-seen heuristics -")
            last_active = try_get_last_active(user_raw, feed_info)
            print(f"Last activity (heuristic): {last_active or 'N/A'}")
            print("\n- Recent media locations (inferred) -")
            if feed_info.get("locations"):
                for loc in feed_info["locations"]:
                    print(f"  - {loc.get('name')} (pk={loc.get('pk')}) lat={loc.get('lat')} lng={loc.get('lng')}")
            else:
                print("  No locations found.")
            print("\n- Profile picture & CDN metadata -")
            if pic_infos:
                for p in pic_infos:
                    print(f"  url: {p.get('url')}")
                    if p.get("head"):
                        print(f"    final_url: {p['head'].get('final_url')}")
                        print(f"    status: {p['head'].get('status_code')}  content-type: {p['head'].get('content_type')}  size: {p['head'].get('content_length')}")
                        print(f"    last_modified: {p['head'].get('last_modified')}  etag: {p['head'].get('etag')}")
                    else:
                        print("    (HEAD failed)")
            else:
                print("  No profile picture URL discovered.")
            if analysis:
                print("\n- Advanced OSINT -")
                pd = analysis.get('profile_data', {})
                tc = analysis.get('text_clues', {})
                td = analysis.get('temporal_data', {})
                print(f"  Public email: {pd.get('public_email')}")
                print(f"  Public phone: {pd.get('public_phone')}")
                print(f"  Phone normalized: {pd.get('public_phone_norm')}")
                print(f"  Text phone clue normalized: {tc.get('phone_clue_norm')}")
                print(f"  External URL resolved: {analysis.get('resolved_url')}")
                if analysis.get('exif_data') and not analysis['exif_data'].get('error'):
                    ex = analysis['exif_data']
                    for k in ['GPSInfo', 'DateTimeOriginal', 'Make', 'Model']:
                        if ex.get(k):
                            print(f"  EXIF {k}: {ex.get(k)}")
                elif analysis.get('exif_data', {}).get('error'):
                    print(f"  EXIF status: {analysis['exif_data']['error']}")
                print("  Keywords:", ', '.join([w[0] for w in analysis.get('keywords', [])]) or 'N/A')
                if td.get('top_hours'):
                    print("  Peak hours UTC:", ', '.join([f"{h[0]}:00({h[1]})" for h in td['top_hours']]))
                if td.get('top_days'):
                    print("  Peak days:", ', '.join([f"{d[0]}({d[1]})" for d in td['top_days']]))
                if analysis.get('linguistic_analysis'):
                    linguistics = analysis['linguistic_analysis']
                    bio_url = linguistics[:2]
                    caption_analysis = linguistics[2:]
                    print("  Linguistics (profile):")
                    for a in bio_url:
                        if a.get('text'):
                            print(f"    - '{a['text']}...' {a['lang'].upper()} Sentiment {a['sentiment']:.2f}")
                    if caption_analysis:
                        lang_summary = Counter()
                        sentiment_map = {}
                        for a in caption_analysis:
                            lang = a['lang'].upper()
                            lang_summary[lang] += 1
                            sentiment_map.setdefault(lang, []).append(a['sentiment'])
                        print("  Linguistics (captions):")
                        for lang, count in lang_summary.items():
                            avg_sentiment = sum(sentiment_map[lang]) / count
                            print(f"    - {lang}: {count} posts, Avg. Sentiment {avg_sentiment:.2f}")
                sr = analysis.get('targeted_search_results', {})
                print("  Targeted keyword hits:")
                if sr:
                    for term, count in sr.items():
                        print(f"    - {term}: {count}")
                else:
                    print("    - None")
                corrs = correlate_data(pd, tc, analysis.get('most_frequent_location'))
                print("  Correlation:")
                if corrs:
                    for c in corrs:
                        print(f"    - {c[0]} [{c[1]}]")
                else:
                    print("    - None")
            print("=== End Investigator ===")
        elif choice == "7":
            print("\n=== Cookies ===")
            if not cookie_dict:
                print("No cookies available.")
            else:
                for name, value in cookie_dict.items():
                    label = label_cookie_name(name)
                    print(f"{name:20} ({label}) = {value}")
        elif choice == "8":
            if not analysis:
                print("[!] No analysis data to export.")
                continue
            path = input("Output filename (e.g., report.json or report.csv): ").strip()
            if not path:
                print("[!] No filename provided.")
                continue
            fmt = "json" if path.lower().endswith(".json") else ("csv" if path.lower().endswith(".csv") else "json")
            export_results(analysis, path, fmt)
            gexf_yn = input("Export GEXF network graph? (y/N): ").strip().lower()
            if gexf_yn == "y":
                export_gexf(analysis, path)
        elif choice == "9":
            break
        else:
            print("[!] Invalid choice. Enter 1-9.")

# ---- Login & main flows ----

def do_login_interactive():
    print(f"[*] Session ID Grabber with 2FA\n")
    username = input(f"[+] Enter Username: ").strip()
    password = stdiomask.getpass(f"[+] Enter Password: ").strip()

    s = requests.Session()
    headers = {
        "Host": "i.instagram.com",
        "X-Ig-Connection-Type": "WiFi",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Ig-Capabilities": "36r/Fx8=",
        "User-Agent": "Instagram 159.0.0.28.123 (iPhone8,1; iOS 14_1)",
        "X-Ig-App-Locale": "en",
        "Accept-Encoding": "gzip, deflate",
    }

    device_id = str(uuid4())
    phone_id = str(uuid4())

    data = {
        "username": username,
        "reg_login": "0",
        "enc_password": f"#PWD_INSTAGRAM:0:&:{password}",
        "device_id": device_id,
        "login_attempt_count": "0",
        "phone_id": phone_id,
    }

    stop_event = threading.Event()
    anim_thread = threading.Thread(target=animate_loading, args=(stop_event,), daemon=True)

    try:
        stop_event.clear()
        anim_thread.start()
        try:
            r = s.post(LOGIN_URL, headers=headers, data=data, timeout=30)
        except Exception as e:
            stop_event.set()
            print("\n\n[!] Network error during login request:", e)
            return None, None
        stop_event.set()
        anim_thread.join(timeout=0.5)

        text = r.text or ""
        j = safe_json(r)

        if 'The password you entered is incorrect' in text or 'bad_password' in text.lower() or \
           (isinstance(j, dict) and j.get('status') == 'fail' and 'password' in j.get('message', '').lower()):
            print("\n\n[!] Wrong password.")
            input("[+] Press Enter to exit...")
            return None, None

        two_factor_required = False
        two_factor_identifier = None
        if isinstance(j, dict) and (j.get("two_factor_required") or j.get("two_factor_info")):
            two_factor_required = True
            info = j.get("two_factor_info") or {}
            two_factor_identifier = info.get("two_factor_identifier") or j.get("two_factor_identifier")
        elif "two_factor_required" in text or "two_factor" in text:
            m = re.search(r'"two_factor_identifier"\s*:\s*"([^"]+)"', text)
            if m:
                two_factor_required = True
                two_factor_identifier = m.group(1)

        if two_factor_required:
            print("\n\n[+] Two-factor authentication required.")
            if two_factor_identifier:
                print(f"    two_factor_identifier: {two_factor_identifier}")
            else:
                print("    (no two_factor_identifier found; we'll still attempt submission)")

            stop_event.clear()
            anim_thread = threading.Thread(target=animate_loading, args=(stop_event,), daemon=True)
            anim_thread.start()
            code = input_with_timeout(f"\nEnter 2FA code (you have {OTP_TIMEOUT} seconds): ", OTP_TIMEOUT, stop_event)
            stop_event.set()
            anim_thread.join(timeout=0.5)

            if not code:
                print("\n\n[!] No 2FA code entered (timed out). Exiting.")
                input("[+] Press Enter to exit...")
                return None, None

            two_data = {
                "username": username,
                "verification_code": code,
                "verificationCode": code,
                "device_id": device_id,
                "phone_id": phone_id,
                "trust_this_device": "1",
            }
            if two_factor_identifier:
                two_data["two_factor_identifier"] = two_factor_identifier

            stop_event.clear()
            anim_thread = threading.Thread(target=animate_loading, args=(stop_event,), daemon=True)
            anim_thread.start()
            try:
                r2 = s.post(TWO_FACTOR_URL, headers=headers, data=two_data, timeout=OTP_TIMEOUT)
            except Exception as e:
                stop_event.set()
                print("\n\n[!] Network error during 2FA request:", e)
                return None, None
            stop_event.set()
            anim_thread.join(timeout=0.5)

            j2 = safe_json(r2)
            text2 = r2.text or ""
            success = False
            if 'logged_in_user' in text2:
                success = True
            elif isinstance(j2, dict) and (j2.get("status") == "ok" and j2.get("logged_in_user")):
                success = True
            cookie_dict_after = s.cookies.get_dict()
            if not success and any(label_cookie_name(n) == "session" for n in cookie_dict_after.keys()):
                success = True

            if success:
                print("\n\n[+] Logged In Success (2FA).")
            else:
                print("\n\n[!] 2FA response did not indicate success. Full response below:")
                pretty_print_json(j2, label="2FA response JSON")
                displayed = text2[:1500] + ("..." if len(text2) > 1500 else "")
                print("\n(truncated raw text):")
                print(displayed)
                input("[+] Press Enter to exit...")
                return None, None

        else:
            if 'logged_in_user' in text or (isinstance(j, dict) and j.get("status") == "ok" and j.get("logged_in_user")):
                print("\n\n[+] Logged In Success.")
            else:
                print("\n\n[!] Login response did not clearly indicate success. Full response JSON below:")
                pretty_print_json(j, label="Login response JSON")
                displayed = text[:1500] + ("..." if len(text) > 1500 else "")
                print("\n(truncated raw text):")
                print(displayed)
                input("[+] Press Enter to exit...")
                return None, None

        print("\n\nCookies found:")
        cookie_dict = s.cookies.get_dict()
        if not cookie_dict:
            print("  No cookies found in session.")
        else:
            for name, value in cookie_dict.items():
                label = label_cookie_name(name)
                print(f"  - {name} ({label}) = {value}")

        return s, cookie_dict

    finally:
        try:
            stop_event.set()
        except Exception:
            pass

def prompt_and_scrape(session_cookies):
    yn = input("\n[?] Do you want to scrape an account using the session id? (y/N): ").strip().lower()
    if yn != "y":
        print("[*] Exiting.")
        return

    sessionid = session_cookies.get("sessionid")
    if not sessionid:
        for k, v in session_cookies.items():
            if label_cookie_name(k) in ("session", "instagram"):
                sessionid = v
                break

    if not sessionid:
        sessionid = input("[!] No sessionid cookie found automatically. Enter sessionid manually: ").strip()
        if not sessionid:
            print("[!] No sessionid provided. Cannot proceed.")
            return

    target = input("[+] Enter target username or numeric id: ").strip()
    if not target:
        print("[!] No target provided. Exiting.")
        return

    if target.isdigit():
        search_type = "id"
        user_id = target
        username = None
    else:
        search_type = "username"
        username = target
        user_id = None

    web_raw = None
    user_raw = None
    lookup_raw = None
    feed_raw = None
    feed_media_meta = {}

    # Resolve user_id
    if username and not user_id:
        user_id, source, raw_used = resolve_user_id(username, sessionid)
        if user_id:
            print(f"[+] Resolved username '{username}' -> user_id {user_id} (source: {source})")
            if source == "web_profile_info":
                web_raw = raw_used
            elif source == "users/lookup":
                lookup_raw = raw_used
            elif source in ("usernameinfo", "www_profile_a1"):
                web_raw = raw_used
        else:
            print(f"[!] Could not resolve user id for '{username}'. Diagnostics:")
            try:
                diagnostics = {}
                d = get_user_web_profile(username, sessionid)
                diagnostics['web_profile_info'] = d.get("raw") or d.get("error")
                e = do_advanced_lookup(username, sessionid)
                diagnostics['lookup'] = e.get("raw") or e.get("raw_text") or e.get("error")
                f = get_usernameinfo(username, sessionid)
                diagnostics['usernameinfo'] = f.get("raw") or f.get("error")
                g = get_www_profile_a1(username)
                diagnostics['www_profile_a1'] = g.get("raw") or g.get("text") or g.get("error")
                pretty_print_json(diagnostics, label="diagnostics")
            except Exception:
                pass
            return

    # Fetch data
    if user_id:
        stop = threading.Event(); t = threading.Thread(target=animate_loading, args=(stop,), daemon=True)
        stop.clear(); t.start()
        user_info_res = get_user_info_private(user_id, sessionid)
        stop.set(); t.join(timeout=0.5)
        user_raw = user_info_res.get("raw") or user_info_res.get("error")

        stop = threading.Event(); t = threading.Thread(target=animate_loading, args=(stop,), daemon=True)
        stop.clear(); t.start()
        lookup_res = do_advanced_lookup(username or (isinstance(user_raw, dict) and user_raw.get("username", "")) or "", sessionid)
        stop.set(); t.join(timeout=0.5)
        lookup_raw = lookup_res.get("raw") or lookup_res.get("raw_text")

        stop = threading.Event(); t = threading.Thread(target=animate_loading, args=(stop,), daemon=True)
        stop.clear(); t.start()
        feed_res = get_feed_media(user_id, sessionid, count=12)
        stop.set(); t.join(timeout=0.5)
        feed_raw = feed_res.get("raw") or {}

        # Media HEAD headers for first items
        try:
            items = feed_raw.get("items") or []
            for it in items[:12]:
                media_id = it.get("id") or it.get("pk")
                display_url = None
                if isinstance(it.get("image_versions2"), dict):
                    candidates = it["image_versions2"].get("candidates", [])
                    if candidates:
                        display_url = candidates[0].get("url")
                if not display_url and isinstance(it.get("carousel_media"), list) and it["carousel_media"]:
                    cm = it["carousel_media"][0]
                    if isinstance(cm.get("image_versions2"), dict):
                        display_url = cm["image_versions2"].get("candidates", [{}])[0].get("url")
                if display_url and media_id:
                    hdr = head_request_headers(display_url, cookies={'sessionid': sessionid})
                    feed_media_meta[media_id] = hdr
        except Exception:
            pass

        # Normalize user object
        if isinstance(user_raw, dict) and user_raw.get("user"):
            if isinstance(user_raw.get("user"), dict):
                user_raw = user_raw["user"]
        if not isinstance(user_raw, dict):
            print("[!] Unexpected user object; showing raw JSON for inspection.")
            pretty_print_json(user_raw, label="User raw")
            return
        if "username" not in user_raw and username:
            user_raw["username"] = username
        if "userID" not in user_raw and user_id:
            user_raw["userID"] = str(user_id)

        # Advanced OSINT aggregate
        analysis = {}
        analysis['target_username'] = user_raw.get('username')
        analysis['profile_exists'] = True
        # Profile data
        analysis['profile_data'] = {
            'bio': user_raw.get('biography'),
            'external_url': user_raw.get('external_url') or "N/A",
            'is_business': bool(user_raw.get('is_business')),
            'public_email': user_raw.get('public_email') or "N/A",
            'public_phone': user_raw.get('public_phone_number') or "N/A",
            'business_address': user_raw.get('business_address_json') or "N/A",
            'business_category': user_raw.get('business_category_name') or "N/A"
        }
        # Contact clues
        text_clues = extract_contact_info(f"{user_raw.get('biography') or ''} {user_raw.get('external_url') or ''}")
        text_clues['phone_clue_norm'] = normalize_phone(text_clues.get('phone_in_text'))
        analysis['text_clues'] = text_clues
        analysis['profile_data']['public_phone_norm'] = normalize_phone(analysis['profile_data'].get('public_phone'))

        # Feed analysis
        fi = infer_locations_from_feed(feed_raw or {})
        analysis['activity_data'] = {'last_post_ts': fi.get('last_post_ts'), 'feed_locations': fi}

        # Behavior & linguistics
        captions = []
        timestamps = []
        try:
            for it in (feed_raw.get("items") or []):
                if isinstance(it.get("caption"), dict):
                    cap = it["caption"].get("text")
                    if cap:
                        captions.append(cap)
                taken = it.get("taken_at") or it.get("device_timestamp") or it.get("timestamp")
                if taken:
                    try:
                        dt = datetime.fromtimestamp(int(taken), tz=timezone.utc)
                        timestamps.append(dt)
                    except Exception:
                        pass
        except Exception:
            pass

        hashtags = Counter()
        mentions = Counter()
        for c in captions:
            for w in c.split():
                if w.startswith("#"): hashtags[w.lower()] += 1
                if w.startswith("@"): mentions[w.lower()] += 1
        analysis['behavior_data'] = {
            'top_hashtags': hashtags.most_common(10),
            'top_mentions': mentions.most_common(10),
            'timestamps': timestamps,
            'captions': captions
        }

        # Temporal patterns
        hours, days = Counter(), Counter()
        for ts in timestamps:
            hours[ts.hour] += 1
            days[ts.strftime('%A')] += 1
        analysis['temporal_data'] = {'top_hours': hours.most_common(3), 'top_days': days.most_common(3)}

        # Keywords, linguistics
        all_texts = [analysis['profile_data']['bio'], analysis['profile_data']['external_url']] + captions
        analysis['keywords'] = extract_keywords(all_texts)
        analysis['linguistic_analysis'] = analyze_text_linguistics(captions + [analysis['profile_data']['bio'], analysis['profile_data']['external_url']])

        # External URL resolve
        analysis['resolved_url'] = resolve_external_url(analysis['profile_data']['external_url'])

        # EXIF from profile picture candidate
        pic_infos = extract_profile_picture_info(user_raw, sessionid)
        exif_done = False
        for p in pic_infos:
            u = p.get('url')
            if u and not exif_done:
                analysis['exif_data'] = exif_from_url_native(u)
                exif_done = True
        if not exif_done:
            analysis['exif_data'] = {"error": "No profile picture URL or EXIF unavailable"}

        # Targeted keyword search
        try:
            terms_line = input("Enter targeted search terms (space-separated) or leave blank: ").strip()
            terms = terms_line.split() if terms_line else []
        except Exception:
            terms = []
        analysis['targeted_search_results'] = targeted_keyword_search(all_texts, terms)

        # Email leakage check
        analysis['email_leakage_check'] = check_email_leakage(analysis['profile_data'].get('public_email'))

        # Network data (followers/following not available via this mobile flow without pagination; placeholder)
        analysis['network_data'] = {'followers': [], 'following': [], 'mutual_followers': []}
        analysis['commenters'] = []  # commenters require extra endpoints; placeholder

        # Most frequent location
        mfl = None
        if fi.get('locations'):
            counts = Counter([l.get('name') for l in fi['locations'] if l.get('name')])
            if counts:
                top_name, ct = counts.most_common(1)[0]
                for l in fi['locations']:
                    if l.get('name') == top_name:
                        mfl = {'name': top_name, 'lat': l.get('lat'), 'lng': l.get('lng'), 'count': ct}
                        break
        analysis['most_frequent_location'] = mfl

        # Launch interactive menu with analysis object
        interactive_user_menu_full(user_raw, sessionid, session_cookies, web_raw=web_raw, lookup_raw=lookup_raw, feed_raw=feed_raw, feed_media_meta=feed_media_meta, analysis=analysis)

def main():
    s, cookies = do_login_interactive()
    if s is None:
        return
    prompt_and_scrape(cookies)
    input("\n[+] Done. Press Enter to exit...")

if __name__ == "__main__":
    main()
