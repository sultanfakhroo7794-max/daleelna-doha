import os, json, time
import requests
import streamlit as st
from tqdm import tqdm
from openai import OpenAI

st.set_page_config(page_title="دليلنا – الدوحة", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OUTSCRAPER_KEY = st.secrets.get("OUTSCRAPER_KEY", os.getenv("OUTSCRAPER_KEY", ""))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

DATA_FILE = "places_doha.json"

AREAS = [
    {"name":"West Bay", "lat":25.324, "lng":51.531},
    {"name":"Pearl",   "lat":25.371, "lng":51.551},
    {"name":"Msheireb","lat":25.286, "lng":51.534},
    {"name":"Old Doha","lat":25.286, "lng":51.537},
]

OSM_TAGS = [
    ("amenity","restaurant"),
    ("amenity","cafe"),
    ("tourism","attraction"),
    ("leisure","park"),
    ("tourism","museum"),
]

def osm_search(lat, lng, radius_km=2.5):
    url = "https://overpass-api.de/api/interpreter"
    radius_m = int(radius_km * 1000)
    queries = []
    for k,v in OSM_TAGS:
        queries.append(f'node(around:{radius_m},{lat},{lng})[{k}="{v}"];')
        queries.append(f'way(around:{radius_m},{lat},{lng})[{k}="{v}"];')
        queries.append(f'relation(around:{radius_m},{lat},{lng})[{k}="{v}"];')
    q = "[out:json];(" + "".join(queries) + ");out center;"
    r = requests.post(url, data=q, timeout=60).json()
    return r.get("elements", [])

def fetch_tripadvisor_reviews(query_name, limit=40):
    if not OUTSCRAPER_KEY:
        return []
    url = "https://api.app.outscraper.com/tripadvisor/reviews"
    params = {"query": query_name, "limit": limit, "key": OUTSCRAPER_KEY}
    try:
        r = requests.get(url, params=params, timeout=60).json()
        return [d.get("review_text","") for d in r.get("data", []) if d.get("review_text")]
    except:
        return []

def consensus_from_reviews(reviews):
    if not client or len(reviews) < 5:
        return {"pros": [], "cons": [], "signature_items": [],
                "family_friendly_score": 0.0, "late_night_score": 0.0, "popular_times": []}

    prompt = """
أنت محلل إجماع لمراجعات أماكن في الدوحة.
المطلوب تلخيص ما اتفق عليه أغلب الناس فقط.
تجاهل الآراء الشاذة/المكررة/المزيفة.
لا تذكر أي معلومات شخصية.
أرجع JSON بهذه المفاتيح:
pros: [..]
cons: [..]
signature_items: [..]
family_friendly_score: رقم بين 0 و1
late_night_score: رقم بين 0 و1
popular_times: ["morning"|"afternoon"|"evening"|"late_night"]
"""
    text = "\n---\n".join(reviews[:80])

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role":"system","content":prompt},
            {"role":"user","content":text}
        ]
    ).choices[0].message.content

    try:
        return json.loads(res)
    except:
        return {"pros": [], "cons": [], "signature_items": [],
                "family_friendly_score": 0.0, "late_night_score": 0.0, "popular_times": []}

def compute_confidence(p):
    official_score = 1.0 if (p.get("opening_hours_text") or p.get("phone")) else 0.0
    consensus_score = 1.0 if (p.get("consensus_pros") or p.get("signature_items")) else 0.0
    return 0.6*official_score + 0.4*consensus_score

def update_all_places():
    existing = {}
    if os.path.exists(DATA_FILE):
        try:
            existing = {p["id"]: p for p in json.load(open(DATA_FILE,"r",encoding="utf-8"))}
        except:
            existing = {}

    new_count = 0

    for area in AREAS:
        elements = osm_search(area["lat"], area["lng"])
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name:
                continue

            pid = str(el.get("id"))
            if pid in existing:
                continue

            venue_type = "attraction"
            if tags.get("amenity") == "restaurant": venue_type = "restaurant"
            if tags.get("amenity") == "cafe": venue_type = "cafe"
            if tags.get("tourism") == "museum": venue_type = "museum"
            if tags.get("leisure") == "park": venue_type = "park"

            existing[pid] = {
                "id": pid,
                "name_ar": name,
                "name_en": name,
                "venue_type": venue_type,
                "area": area["name"],
                "geo_lat": el.get("lat") or el.get("center", {}).get("lat"),
                "geo_lng": el.get("lon") or el.get("center", {}).get("lon"),
                "opening_hours_text": tags.get("opening_hours"),
                "phone": tags.get("phone"),
                "consensus_pros": [],
                "consensus_cons": [],
                "signature_items": [],
                "popular_times": [],
                "family_friendly_score": 0.0,
                "late_night_score": 0.0,
                "data_confidence": 0.0
            }
            new_count += 1

    places = list(existing.values())

    if OUTSCRAPER_KEY:
        st.info("سحب إجماع الناس من TripAdvisor...")
        for p in tqdm(places):
            reviews = fetch_tripadvisor_reviews(p["name_en"])
            cons = consensus_from_reviews(reviews)
            p["consensus_pros"] = cons.get("pros",[])
            p["consensus_cons"] = cons.get("cons",[])
            p["signature_items"] = cons.get("signature_items",[])

    for p in places:
        p["data_confidence"] = compute_confidence(p)

    json.dump(places, open(DATA_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    st.success(f"تم التحديث. أماكن جديدة: {new_count}. الإجمالي: {len(places)}")
    return places

st.markdown("## دليلنا – الدوحة")
st.caption("نسخة خفيفة للتشغيل السريع")

with st.sidebar:
    st.markdown("### تفضيلاتك")
    who = st.selectbox("مع من؟", ["عائلة","شباب","زوجين","لوحدك"])
    kids = st.checkbox("معي أطفال")
    budget = st.selectbox("الميزانية", ["اقتصادي","متوسط","فاخر"])
    mood = st.selectbox("الجو", ["هادي","حيوي","أي شي"])
    st.divider()
    if st.button("🔄 تحديث البيانات الآن"):
        places = update_all_places()
    else:
        places = json.load(open(DATA_FILE,"r",encoding="utf-8")) if os.path.exists(DATA_FILE) else []

query = st.text_input("وين تبي تروح اليوم؟", placeholder="مثال: hidden gems للشباب")

if st.button("سوّي لي اقتراحات"):
    if not places:
        st.warning("اضغط تحديث البيانات أول.")
    else:
        good = [p for p in places if p.get("data_confidence",0) >= 0.5][:8]
        for p in good:
            st.markdown(
                f"""
**{p['name_ar']}** — ثقة {round(p['data_confidence']*100)}%  
النوع: {p['venue_type']} | المنطقة: {p['area']}  
أهم ما اتفق عليه الناس: {", ".join(p.get("consensus_pros",[])[:2]) or "لا توجد مراجعات كافية بعد"}  
"""
            )
