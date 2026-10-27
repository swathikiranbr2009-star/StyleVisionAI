import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Style Vision AI", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
.stApp{background:#0c0c0e;color:#f4f0e8;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#141416;border-right:1px solid #29282a}
[data-testid="stSidebar"] *{color:#f4f0e8}
h1,h2,h3{color:#f4f0e8!important}
.brand{color:#d9b878;font-size:14px;font-weight:700;letter-spacing:4px;margin-bottom:8px}
.hero{padding:42px 36px;border:1px solid #39332a;border-radius:20px;background:linear-gradient(90deg,rgba(12,12,14,.96),rgba(12,12,14,.78),rgba(12,12,14,.18)),url('https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&w=1800&q=85');background-position:center;background-size:cover;min-height:280px;margin-bottom:26px}
.hero h1{font-family:'Playfair Display',serif;font-size:clamp(34px,5vw,58px);line-height:1.08;max-width:600px;margin:8px 0 14px}
.hero p{color:#d2ccc1;max-width:550px;font-size:16px}
.gold,.price{color:#d9b878}
.section-label{color:#d9b878;font-size:12px;letter-spacing:3px;font-weight:700;margin-bottom:4px}
.product-card{background:#171719;border:1px solid #2b2a2c;border-radius:14px;padding:12px;min-height:100%}
.muted{color:#aaa5a0;font-size:13px}
div.stButton>button{border-radius:10px;border:1px solid #d9b878;background:#d9b878;color:#141416;font-weight:700;min-height:42px}
div.stButton>button:hover{background:#e8cb8e;border-color:#e8cb8e;color:#141416}
hr{border-color:#2b2a2c}
</style>
""", unsafe_allow_html=True)

# Put clothes.csv OR products.csv in the same folder as app.py.
# If both exist, clothes.csv is used first.
def load_products():
    csv_file = "clothes.csv"
    if not oath(csv_file).exists():
        st.error("clothes,csv not found. Put it in the same folder as app.py")
        st.stops()
    if csv_file is None:
        st.error("CSV not found. Put clothes.csv or products.csv in the same folder as app.py.")
        st.stop()
    try:
        data = pd.read_csv(csv_file)
    except Exception as exc:
        st.error(f"Could not read {csv_file}: {exc}")
        st.stop()

    data.columns = [str(c).strip().lower() for c in data.columns]
    required = ["name", "category", "color", "style", "occasion", "price", "description", "image"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        st.error(f"Missing CSV columns: {', '.join(missing)}. Required: {', '.join(required)}")
        st.stop()

    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price", "name"]).copy()
    for c in ["name", "category", "color", "style", "occasion", "description", "image"]:
        data[c] = data[c].fillna("").astype(str).str.strip()
    for c in ["quality", "quantity"]:
        if c not in data.columns:
            data[c] = ""
    return data, csv_file

df, csv_filename = load_products()

st.markdown("""
<div class="brand">✦ STYLE VISION AI</div>
<div class="hero">
<div class="section-label">YOUR PERSONAL AI STYLE ASSISTANT</div>
<h1>Discover a look<br>that feels like <span class="gold">you.</span></h1>
<p>Describe your vibe, explore styles, and get outfit recommendations tailored to your preferences.</p>
</div>
""", unsafe_allow_html=True)

a, b, c = st.columns(3)
a.metric("Styles in catalogue", len(df))
b.metric("Categories", df["category"].replace("", pd.NA).nunique())
c.metric("Lowest listed price", f"₹{df['price'].min():,.0f}" if not df.empty else "—")
st.markdown("---")

st.sidebar.markdown("## Refine your style")
st.sidebar.caption("Adjust preferences to personalise recommendations.")
max_price = int(df["price"].max()) if not df.empty else 5000
budget = st.sidebar.slider("Maximum budget (₹)", 0, max(100, max_price), max(100, max_price), step=100)

def options_for(column):
    vals = sorted(x for x in df[column].unique().tolist() if x)
    return ["All"] + vals

category = st.sidebar.selectbox("Category", options_for("category"))
color = st.sidebar.selectbox("Colour", options_for("color"))
style = st.sidebar.selectbox("Style", options_for("style"))
occasion = st.sidebar.selectbox("Occasion", options_for("occasion"))

st.markdown('<div class="section-label">01 — TELL US YOUR VIBE</div>', unsafe_allow_html=True)
st.subheader("What are you looking for?")
query = st.text_input("Describe your style", placeholder="e.g. casual black outfit for college, minimalist summer look...")
recommend = st.button("✦ Find My Style", use_container_width=True)

results = df[df["price"] <= budget].copy()
for col, choice in [("category", category), ("color", color), ("style", style), ("occasion", occasion)]:
    if choice != "All":
        results = results[results[col] == choice]

def make_text(frame):
    return (frame["name"] + " " + frame["category"] + " " + frame["color"] + " " +
            frame["style"] + " " + frame["occasion"] + " " + frame["description"])

if recommend and query.strip() and not results.empty:
    try:
        corpus = make_text(results).tolist()
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus + [query.strip()])
        results["match_score"] = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        results = results.sort_values("match_score", ascending=False)
    except ValueError:
        results["match_score"] = 0.0
else:
    results["match_score"] = 0.0

if recommend and not query.strip():
    st.info("Describe the style you're looking for, then press **Find My Style**.")
elif results.empty:
    st.warning("No styles match these filters. Increase your budget or change a filter.")
else:
    if recommend and query.strip():
        st.markdown('<div class="section-label">02 — PERSONAL RECOMMENDATIONS</div>', unsafe_allow_html=True)
        st.subheader("Recommendations for you")
    else:
        st.markdown('<div class="section-label">02 — EXPLORE THE COLLECTION</div>', unsafe_allow_html=True)
        st.subheader("Explore styles")
    st.caption(f"Showing {len(results)} style(s) loaded from {csv_filename}.")
    for start in range(0, len(results), 4):
        cols = st.columns(4, gap="medium")
        for idx, (_, item) in enumerate(results.iloc[start:start + 4].iterrows()):
            with cols[idx]:
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                if item["image"]:
                    st.image(item["image"], use_container_width=True)
                else:
                    st.caption("No image provided")
                st.markdown(f"#### {item['name']}")
                st.markdown(f'<div class="price">₹{item["price"]:,.0f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="muted">{item["category"]} · {item["color"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="muted">{item["style"]} · {item["occasion"]}</div>', unsafe_allow_html=True)
                if item["description"]:
                    st.write(item["description"])
                if str(item.get("quality", "")).strip():
                    st.caption(f"Quality: {item['quality']}")
                if str(item.get("quantity", "")).strip():
                    st.caption(f"Quantity: {item['quantity']}")
                if recommend and query.strip():
                    st.caption(f"Text match: {item['match_score']:.0%}")
                st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-label">03 — STYLE INSPIRATION</div>', unsafe_allow_html=True)
st.subheader("Outfit ideas")
st.write("Get simple outfit combination ideas from your catalogue. This is a recommendation assistant, not a shopping app.")

tops = df[df["category"].str.lower().isin(["top", "tops", "shirt", "t-shirt", "tshirt"])]
bottoms = df[df["category"].str.lower().isin(["bottom", "bottoms", "trousers", "jeans", "pants", "skirt"])]
shoes = df[df["category"].str.lower().isin(["shoe", "shoes", "footwear", "sneakers"])]
if not tops.empty and not bottoms.empty:
    x, y, z = st.columns(3)
    x.markdown("**Top**")
    x.write(tops.iloc[0]["name"])
    y.markdown("**Bottom**")
    y.write(bottoms.iloc[0]["name"])
    z.markdown("**Shoes**")
    z.write(shoes.iloc[0]["name"] if not shoes.empty else "Choose shoes that complement the outfit.")
else:
    st.info("Add product categories like Tops, Bottoms, and Shoes to your CSV to enable outfit ideas.")

st.markdown("---")
st.caption("Style Vision AI is a fashion recommendation prototype. It uses text
