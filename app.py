import json, re
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from rapidfuzz import process, fuzz

MODEL = "gemini-3.5-flash"   # error aaye to "gemini-2.5-flash" try karo

st.set_page_config(page_title="Prescription Scanner", page_icon="💊", layout="wide")
st.title("💊 Prescription Scanner")
st.caption("Educational project, clinical use ke liye nahi hai. Final verification doctor/pharmacist se karwao.")

# Cloud pe key secrets se aayegi, laptop pe sidebar se
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = st.sidebar.text_input("Gemini API key", type="password")

# ---------- Database ----------
drugs = pd.read_csv("drugs.csv")
lookup = {}
for _, r in drugs.iterrows():
    lookup[r["name"].lower()] = r["name"]
    for b in str(r["brand_names_india"]).split("/"):
        b = re.sub(r"\(.*?\)", "", b).strip().lower()
        if b:
            lookup[b] = r["name"]

FIELDS = [
    ("Class", "drug_class"),
    ("Uses", "uses"),
    ("Usual adult dose", "usual_adult_dose"),
    ("Side effects", "side_effects"),
    ("Contraindications", "contraindications"),
    ("Key interactions", "key_interactions"),
    ("Counselling points", "counselling"),
]

# ---------- Helpers ----------
def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    return json.loads(text)

def gemini_call(client, contents):
    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return parse_json(resp.text)

EXTRACT_PROMPT = """You are reading an Indian medical prescription image.
Extract every medicine written. Return ONLY a JSON list:
[{"drug": "name as written", "strength": "e.g. 500 mg or empty", "frequency": "e.g. 1-0-1 / BD / TDS or empty", "duration": "e.g. 5 days or empty", "legible": true}]
If a name is hard to read, give your best guess and set "legible" to false.
Do not invent medicines that are not on the prescription."""

@st.cache_data(show_spinner=False)
def ai_drug_info(names, key):
    client = genai.Client(api_key=key)
    prompt = f"""For each medicine below (Indian brand or generic name), give standard reference information.
Medicines: {list(names)}
Return ONLY a JSON list, same order, each item:
{{"drug": "name as given", "generic_name": "generic name or empty if unknown",
"drug_class": "", "uses": "", "usual_adult_dose": "", "side_effects": "",
"contraindications": "", "key_interactions": "", "counselling": ""}}
Keep each field short (semicolon separated). If you do not recognise the medicine, put "Unknown" in every field.
Do not guess."""
    return gemini_call(client, [prompt])

def show_info(info):
    for label, col in FIELDS:
        st.markdown(f"**{label}:** {info.get(col) or 'Not available'}")

# ---------- Main ----------
file = st.file_uploader("Prescription upload karo", type=["jpg", "jpeg", "png"])

if file and not api_key:
    st.warning("Sidebar mein Gemini API key daalo.")

if file and api_key:
    left, right = st.columns([1, 2])
    left.image(file, use_container_width=True)

    try:
        client = genai.Client(api_key=api_key)
        with st.spinner("Prescription padh raha hoon..."):
            items = gemini_call(client, [
                types.Part.from_bytes(data=file.getvalue(), mime_type=file.type),
                EXTRACT_PROMPT,
            ])
    except Exception as e:
        st.error(f"Prescription padhne mein error: {e}")
        st.stop()

    if not items:
        st.info("Koi medicine nahi mili. Saaf aur seedhi photo try karo.")
        st.stop()

    # Database matching
    matches, unmatched = {}, []
    for i, it in enumerate(items):
        name = (it.get("drug") or "").strip()
        m = process.extractOne(name.lower(), list(lookup.keys()), scorer=fuzz.WRatio, score_cutoff=80)
        if m:
            matches[i] = drugs[drugs["name"] == lookup[m[0]]].iloc[0].to_dict()
        elif name:
            unmatched.append(name)

    # AI fallback (ek hi call mein sab unmatched drugs)
    ai_info = {}
    if unmatched:
        try:
            with st.spinner("Database mein na milne wali medicines ki info la raha hoon..."):
                result = ai_drug_info(tuple(unmatched), api_key)
            for nm, info in zip(unmatched, result):
                ai_info[nm] = info
        except Exception as e:
            st.warning(f"AI info nahi mil paayi: {e}")

    # Summary table
    right.subheader("Summary")
    summary = pd.DataFrame([{
        "Drug": it.get("drug"),
        "Strength": it.get("strength") or "-",
        "Frequency": it.get("frequency") or "-",
        "Duration": it.get("duration") or "-",
        "Source": "Database ✅" if i in matches else "AI ⚠️",
    } for i, it in enumerate(items)])
    right.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Detailed information")

    for i, it in enumerate(items):
        name = it.get("drug") or "Unknown"
        with st.expander(f"💊 {name}  |  {it.get('strength') or ''}  {it.get('frequency') or ''}", expanded=True):
            if not it.get("legible", True):
                st.warning("Handwriting unclear hai, naam verify karo.")
            if i in matches:
                row = matches[i]
                st.success(f"Verified database match: {row['name']}")
                st.markdown(f"**Brands:** {row['brand_names_india']}")
                show_info(row)
            elif name in ai_info:
                info = ai_info[name]
                st.warning("AI generated information. Database mein ye drug nahi hai, isliye textbook se verify karo.")
                if info.get("generic_name"):
                    st.markdown(f"**Generic name:** {info['generic_name']}")
                show_info(info)
            else:
                st.info("Is drug ki information nahi mili.")