import os, json, time
import requests
import streamlit as st
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

st.set_page_config(page_title="دليلنا – الدوحة", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OUTSCRAPER_KEY = st.secrets.get("OUTSCRAPER_KEY", os.getenv("OUTSCRAPER_KEY", ""))

DATA_FILE = "places_doha.json"

# مناطق الدوحة (تقدر توسّعها لاحقاً)
AREAS = [
    {"name":"West Bay", "lat":25.324, "lng":51.531},
    {"name":"Pearl",   "lat":25.371, "lng":51.551},
    {"name":"Msheireb","lat":25.286, "lng":51.534},
    {"name":"Old Doha","lat":25.286, "lng":51.537},
]

# فئات أماكن
OSM_TAGS = [
    ("amenity","restaurant"),
    ("amenity","cafe"),
    ("tourism","attraction"),
    ("leisure","park"),
    ("tourism","museum"),
]

def osm_search(lat, lng, radius_km=2.5):
    """يجلب أماكن من OpenStreetMap (Overpass API)"""
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
    if not OPENAI_API_KEY or len(reviews) < 5:
        return {"pros": [], "cons": [], "signature_items": [],
                "family_friendly_score": 0.0, "late_night_score": 0.0, "popular_times": []}
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
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
    res = llm.invoke(prompt + "\nREVIEWS:\n" + text).content
    try: 
        return json.loads(res)
    except:
        return {"pros": [], "cons": [], "signature_items": [],
                "family_friendly_score": 0.0, "late_night_score": 0.0, "popular_times": []}

def compute_confidence(p):
    w_official, w_behavior, w_consensus = 0.45, 0.35, 0.20
    official_score = 1.0 if (p.get("official_source_urls") or p.get("opening_hours_text") or p.get("phone")) else 0.0
    behavior_score = 1.0 if (p.get("popular_times") or p.get("family_friendly_score",0)>0 or p.get("late_night_score",0)>0) else 0.0
    consensus_score = 1.0 if (p.get("consensus_pros") or p.get("signature_items")) else 0.0
    return w_official*official_score + w_behavior*behavior_score + w_consensus*consensus_score

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

            # تحديد نوع المكان
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
                "official_source_urls": [],
                "phone": tags.get("phone"),
                "opening_hours_text": tags.get("opening_hours"),
                "popular_times": [],
                "family_friendly_score": 0.0,
                "late_night_score": 0.0,
                "consensus_pros": [],
                "consensus_cons": [],
                "signature_items": [],
                "tags": [],
                "data_confidence": 0.0
            }
            new_count += 1

    places = list(existing.values())

    if OUTSCRAPER_KEY:
        st.info("جاري سحب إجماع الناس من TripAdvisor...")
        for p in tqdm(places):
            reviews = fetch_tripadvisor_reviews(p["name_en"])
            cons = consensus_from_reviews(reviews)
            p["consensus_pros"] = cons.get("pros",[])
            p["consensus_cons"] = cons.get("cons",[])
            p["signature_items"] = cons.get("signature_items",[])
            p["family_friendly_score"] = float(cons.get("family_friendly_score", 0))
            p["late_night_score"] = float(cons.get("late_night_score", 0))
            p["popular_times"] = cons.get("popular_times",[])

    for p in places:
        p["data_confidence"] = compute_confidence(p)

    json.dump(places, open(DATA_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    st.success(f"تم التحديث (OSM). أماكن جديدة: {new_count}. الإجمالي: {len(places)}")
    return places

@st.cache_resource(show_spinner=False)
def build_vector_db(places):
    if not OPENAI_API_KEY: return None
    docs = [Document(page_content=f"{p['name_ar']} | {p['venue_type']} | {p['area']} | pros={p.get('consensus_pros',[])}")
            for p in places]
    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return Chroma.from_documents(docs, emb)

def recommend_places(db, places, query, min_conf=0.7, k=6):
    good = [p for p in places if p.get("data_confidence",0) >= min_conf]
    if not db or not good: return good[:k]
    retriever = db.as_retriever(search_kwargs={"k": k})
    ctx_docs = retriever.get_relevant_documents(query)
    names = [d.page_content.split("|")[0].strip() for d in ctx_docs]
    ranked = [p for p in good if p["name_ar"] in names]
    ranked.sort(key=lambda x: x["data_confidence"], reverse=True)
    return ranked[:k]

st.markdown("## دليلنا – الدوحة")
st.caption("نسخة تجريبية مجانية بدون Google Billing")

with st.sidebar:
    st.markdown("### تفضيلاتك")
    who = st.selectbox("مع من؟", ["عائلة","شباب","زوجين","لوحدك"])
    kids = st.checkbox("معي أطفال")
    kids_age = st.slider("عمر الطفل", 0, 14, 5) if kids else None
    budget = st.selectbox("الميزانية", ["اقتصادي","متوسط","فاخر"])
    time_pref = st.selectbox("وقت الخروج", ["صباح","عصر","ليل","آخر الليل"])
    mood = st.selectbox("الجو", ["هادي","حيوي","أي شي"])
    st.divider()
    if st.button("🔄 تحديث البيانات الآن"):
        places = update_all_places()
        st.cache_resource.clear()
    else:
        places = json.load(open(DATA_FILE,"r",encoding="utf-8")) if os.path.exists(DATA_FILE) else []

query = st.text_input("وين تبي تروح اليوم؟", placeholder="مثال: hidden gems للشباب 3 أيام")

if st.button("سوّي لي اقتراحات"):
    if not places:
        st.warning("اضغط تحديث البيانات أول.")
    else:
        db = build_vector_db(places)
        recs = recommend_places(db, places, query)
        if not recs:
            st.warning("ما عندي بيانات كافية أضمن لك توصية دقيقة.")
        else:
            for p in recs:
                st.markdown(
                    f"""
**{p['name_ar']}** — ثقة {round(p['data_confidence']*100)}%  
النوع: {p['venue_type']} | المنطقة: {p['area']}  
أهم ما اتفق عليه الناس: {", ".join(p.get("consensus_pros",[])[:2]) or "لا توجد مراجعات كافية بعد"}  
"""
                )
