import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import html
import re

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Style Vision AI",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- PREMIUM DARK DESIGN ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --bg:#090b0e; --panel:#121519; --panel2:#191d22; --line:#2a2f36; --text:#f5f5f2; --muted:#a5abb4; --gold:#e9c994; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(180deg,#0b0d10 0%,#090b0e 100%); color:var(--text); }
    [data-testid="stHeader"] { background:rgba(9,11,14,.96); }
    [data-testid="stSidebar"] { background:#101317; border-right:1px solid #252a30; }
    [data-testid="stSidebar"] * { color:#f2f2ef; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
    h1 { font-weight:800 !important; }
    p, label, .stMarkdown { color:#e8e9e8; }
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"], div[data-testid="stNumberInput"] input {
      background:#171b20 !important; color:#fff !important; border-color:#343a42 !important; border-radius:10px !important;
    }
    .stButton > button { border-radius:10px; border:1px solid #444b54; background:#171b20; color:#fff; transition:all .18s ease; }
    .stButton > button:hover { border-color:var(--gold); color:var(--gold); }
    button[kind="primary"] { background:var(--gold) !important; color:#17130d !important; border:0 !important; font-weight:700 !important; }
    div[data-testid="stSlider"] [data-baseweb="slider"] { color:var(--gold); }
    .brandbar { display:flex; justify-content:space-between; align-items:center; gap:16px; padding:8px 0 18px; border-bottom:1px solid #252a30; margin-bottom:20px; }
    .brand { font-family:'Manrope',sans-serif; font-size:1.08rem; font-weight:800; letter-spacing:.13em; color:#f6e4c5; }
    .brand small { display:block; font-family:'DM Sans',sans-serif; color:#a5abb4; font-size:.72rem; letter-spacing:.02em; font-weight:400; margin-top:3px; }
    .navnote { color:#b9bdc4; font-size:.86rem; }
    .hero { min-height:235px; display:flex; align-items:center; padding:32px; border-radius:18px; overflow:hidden; border:1px solid #343536;
      background:linear-gradient(90deg,rgba(7,8,10,.97) 0%,rgba(7,8,10,.86) 42%,rgba(7,8,10,.18) 100%),url('https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&w=1800&q=85') center 42%/cover; margin-bottom:20px; }
    .hero-kicker { text-transform:uppercase; letter-spacing:.22em; font-size:.72rem; color:var(--gold); font-weight:700; margin-bottom:10px; }
    .hero-title { font-family:'Manrope',sans-serif; font-size:clamp(2rem,4vw,3.2rem); line-height:1.04; font-weight:800; letter-spacing:-.05em; max-width:520px; color:#fff; }
    .hero-copy { max-width:430px; color:#d3d4d5; font-size:.98rem; margin-top:12px; }
    .section-label { color:var(--gold); text-transform:uppercase; letter-spacing:.16em; font-size:.7rem; font-weight:700; margin:8px 0 5px; }
    .section-sub { color:#a8adb5; font-size:.9rem; margin-top:-8px; margin-bottom:16px; }
    .stat { background:#121519; border:1px solid #272c32; padding:13px 16px; border-radius:12px; }
    .stat-number { font-family:'Manrope',sans-serif; font-size:1.35rem; font-weight:800; color:#f5dfb8; }
    .stat-label { font-size:.76rem; color:#a5abb4; }
    .product-card { background:linear-gradient(180deg,#171a1f,#111418); border:1px solid #2a2f36; border-radius:14px; padding:10px 10px 12px; margin-bottom:12px; }
    .product-name { font-family:'Manrope',sans-serif; font-weight:700; color:#f6f5f1; font-size:1rem; margin:7px 0 3px; }
    .product-meta { color:#a5abb4; font-size:.78rem; min-height:18px; }
    .product-price { color:#f4d9a9; font-weight:700; font-size:1rem; margin-top:6px; }
    .pill { display:inline-block; border:1px solid #3b4149; border-radius:99px; color:#c5c9ce; padding:3px 8px; font-size:.68rem; margin:5px 4px 0 0; }
    .empty { background:#13171b; border:1px dashed #3a4149; padding:28px; border-radius:14px; text-align:center; color:#c2c6cc; }
    .footer { border-top:1px solid #272c32; margin-top:34px; padding:18px 0 6px; color:#8f969f; font-size:.78rem; display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; }
    [data-testid="stMetric"] { background:#121519; border:1px solid #272c32; padding:12px; border-radius:12px; }
    [data-testid="stExpander"] { border-color:#30363d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HELPERS / DATA ----------------
REQUIRED_COLUMNS = ["name", "category", "color", "style", "occasion", "price", "description", "image"]

def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

@st.cache_data
def load_products():
    # Support either filename so local and cloud copies both work.
    candidates = [Path("clothes.csv"), Path("products.csv")]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "I couldn't find clothes.csv or products.csv. Put one of these files beside app.py."
        )
    data = pd.read_csv(csv_path)
    data.columns = [str(c).strip().lower() for c in data.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError("CSV is missing these required columns: " + ", ".join(missing))
    data["price"] = pd.to_numeric(
        data["price"].astype(str).str.replace(r"[₹,\s]", "", regex=True),
        errors="coerce",
    )
    data = data.dropna(subset=["price"]).copy()
    for column in data.columns:
        if column != "price":
            data[column] = data[column].fillna("").astype(str).str.strip()
    data = data.reset_index(drop=True)
    return data, csv_path.name

try:
    df, csv_filename = load_products()
except Exception as error:
    st.error(f"Couldn't load your product catalogue: {error}")
    st.info("Check that your CSV is in the same GitHub repository folder as app.py.")
    st.stop()

if df.empty:
    st.warning("Your CSV has no valid product rows yet.")
    st.stop()

# Session state
if "cart" not in st.session_state:
    st.session_state.cart = {}
if "favorites" not in st.session_state:
    st.session_state.favorites = set()
if "search_submitted" not in st.session_state:
    st.session_state.search_submitted = False

# Build model only from the real product catalogue
df["search_text"] = (
    df["name"] + " " + df["category"] + " " + df["color"] + " " +
    df["style"] + " " + df["occasion"] + " " + df["description"]
).str.lower()
try:
    vectorizer = TfidfVectorizer(stop_words="english")
    product_vectors = vectorizer.fit_transform(df["search_text"])
except ValueError:
    vectorizer = None
    product_vectors = None

# ---------------- TOP NAV / HERO ----------------
cart_count = sum(st.session_state.cart.values())
st.markdown(
    f"""
    <div class="brandbar">
      <div class="brand">◇ STYLE VISION AI <small>Smarter choices. Better style.</small></div>
      <div class="navnote">HOME&nbsp;&nbsp;&nbsp;&nbsp; SHOP&nbsp;&nbsp;&nbsp;&nbsp; CATEGORIES&nbsp;&nbsp;&nbsp;&nbsp; ABOUT
      &nbsp;&nbsp;&nbsp;&nbsp; 🛍️ {cart_count} in cart</div>
    </div>
    <div class="hero">
      <div>
        <div class="hero-kicker">Your style. Our vision.</div>
        <div class="hero-title">Discover Your<br>Perfect Look</div>
        <div class="hero-copy">Fashion finds tailored to your style, preferences and budget. Shop smarter, dress better.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Summary stats
stat_cols = st.columns(3)
with stat_cols[0]:
    st.markdown(f'<div class="stat"><div class="stat-number">{len(df)}</div><div class="stat-label">PRODUCTS IN CATALOGUE</div></div>', unsafe_allow_html=True)
with stat_cols[1]:
    st.markdown(f'<div class="stat"><div class="stat-number">{df["category"].nunique()}</div><div class="stat-label">CATEGORIES</div></div>', unsafe_allow_html=True)
with stat_cols[2]:
    st.markdown(f'<div class="stat"><div class="stat-number">₹{df["price"].min():,.0f}–₹{df["price"].max():,.0f}</div><div class="stat-label">PRICE RANGE</div></div>', unsafe_allow_html=True)

st.write("")

# ---------------- FILTERS ----------------
st.sidebar.markdown("## ✦ Find your fit")
st.sidebar.caption("Refine the catalogue, then search for your style.")
max_price = max(1000, int(df["price"].max()))
min_price = max(0, int(df["price"].min()))
budget = st.sidebar.slider("Maximum budget (₹)", min_value=0, max_value=max_price, value=max_price, step=max(50, int(max_price / 50) or 50))

def options_for(column):
    values = sorted({v for v in df[column].astype(str).tolist() if v.strip()})
    return ["Any"] + values

category_options = options_for("category")
color_options = options_for("color")
style_options = options_for("style")
occasion_options = options_for("occasion")
category = st.sidebar.selectbox("Category", category_options)
color = st.sidebar.selectbox("Colour", color_options)
style = st.sidebar.selectbox("Style", style_options)
occasion = st.sidebar.selectbox("Occasion", occasion_options)

quality_col = next((c for c in ["quality", "grade"] if c in df.columns), None)
if quality_col:
    quality_options = options_for(quality_col)
    quality = st.sidebar.selectbox("Quality", quality_options)
else:
    quality = "Any"

if "quantity" in df.columns:
    in_stock_only = st.sidebar.checkbox("In-stock products only", value=False)
else:
    in_stock_only = False

st.sidebar.markdown("---")
if st.sidebar.button("Clear filters", use_container_width=True):
    st.rerun()

# ---------------- SEARCH ----------------
st.markdown('<div class="section-label">Personalised discovery</div>', unsafe_allow_html=True)
st.markdown("## Find your style")
st.markdown('<div class="section-sub">Describe an item, colour, vibe or occasion to find relevant products.</div>', unsafe_allow_html=True)

search_col, button_col = st.columns([5, 1])
with search_col:
    preference = st.text_input(
        "Search products",
        placeholder="Try: black oversized T-shirt for college",
        label_visibility="collapsed",
        key="style_search",
    )
with button_col:
    search_clicked = st.button("Find my style ↗", type="primary", use_container_width=True)
if search_clicked:
    st.session_state.search_submitted = True

# ---------------- APPLY FILTERS / RANK ----------------
filtered = df[df["price"] <= budget].copy()
if category != "Any":
    filtered = filtered[filtered["category"].str.casefold() == category.casefold()]
if color != "Any":
    filtered = filtered[filtered["color"].str.casefold() == color.casefold()]
if style != "Any":
    filtered = filtered[filtered["style"].str.casefold() == style.casefold()]
if occasion != "Any":
    filtered = filtered[filtered["occasion"].str.casefold() == occasion.casefold()]
if quality_col and quality != "Any":
    filtered = filtered[filtered[quality_col].str.casefold() == quality.casefold()]
if in_stock_only and "quantity" in filtered.columns:
    qty = pd.to_numeric(filtered["quantity"], errors="coerce").fillna(0)
    filtered = filtered[qty > 0]

query = safe_text(preference).lower()
if not filtered.empty:
    filtered["AI Match Score"] = 0.0
    if query and vectorizer is not None:
        query_vector = vectorizer.transform([query])
        # Score only the filtered products, avoiding index-alignment errors.
        scores = cosine_similarity(query_vector, product_vectors[filtered.index]).flatten()
        filtered["AI Match Score"] = scores
        # Give explicit colour/category matches a modest boost.
        q_words = set(re.findall(r"[a-z0-9]+", query))
        def boost(row):
            bonus = 0.0
            for field in ["name", "category", "color", "style", "occasion"]:
                field_words = set(re.findall(r"[a-z0-9]+", safe_text(row[field]).lower()))
                if q_words & field_words:
                    bonus += 0.025
            return bonus
        filtered["AI Match Score"] = filtered["AI Match Score"] + filtered.apply(boost, axis=1)
        filtered = filtered.sort_values(["AI Match Score", "price"], ascending=[False, True])
    else:
        filtered = filtered.sort_values("price", ascending=True)

sort_by = st.selectbox("Sort products", ["Best match", "Price: low to high", "Price: high to low", "Name: A–Z"], label_visibility="collapsed")
if sort_by == "Price: low to high":
    filtered = filtered.sort_values("price", ascending=True)
elif sort_by == "Price: high to low":
    filtered = filtered.sort_values("price", ascending=False)
elif sort_by == "Name: A–Z":
    filtered = filtered.sort_values("name", ascending=True)

st.markdown("## Recommended for you")
st.caption(f"{len(filtered)} products match your current filters · Catalogue: {csv_filename}")

# ---------------- PRODUCT GRID ----------------
if filtered.empty:
    st.markdown('<div class="empty"><h3>No products found</h3><p>Try increasing your budget or clearing one of the filters.</p></div>', unsafe_allow_html=True)
else:
    display_count = min(len(filtered), 60)
    cards_per_row = 4
    for start in range(0, display_count, cards_per_row):
        row = filtered.iloc[start:start + cards_per_row]
        cols = st.columns(cards_per_row, gap="medium")
        for col, (_, product) in zip(cols, row.iterrows()):
            product_name = safe_text(product["name"]) or "Untitled product"
            product_id = f"{product_name}|{safe_text(product['category'])}|{safe_text(product['color'])}|{product['price']}"
            with col:
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                image_url = safe_text(product.get("image", ""))
                if image_url.startswith(("https://", "http://")):
                    try:
                        st.image(image_url, use_container_width=True)
                    except Exception:
                        st.markdown("🧥")
                else:
                    st.markdown("🧥 *No product image available*")
                st.markdown(f'<div class="product-name">{html.escape(product_name)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="product-meta">{html.escape(safe_text(product["category"]))} · {html.escape(safe_text(product["color"]))}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="product-price">₹{float(product["price"]):,.0f}</div>', unsafe_allow_html=True)
                if quality_col and safe_text(product.get(quality_col, "")):
                    st.markdown(f'<span class="pill">{html.escape(safe_text(product[quality_col]))}</span>', unsafe_allow_html=True)
                if "quantity" in df.columns:
                    quantity = pd.to_numeric(pd.Series([product.get("quantity", 0)]), errors="coerce").fillna(0).iloc[0]
                    stock_label = "In stock" if quantity > 0 else "Out of stock"
                    st.markdown(f'<span class="pill">{stock_label}</span>', unsafe_allow_html=True)
                if safe_text(product["description"]):
                    st.caption(safe_text(product["description"]))
                score = float(product.get("AI Match Score", 0.0))
                if query:
                    st.progress(min(1.0, max(0.0, score)), text=f"Match {score*100:.0f}%")
                btn1, btn2 = st.columns([3, 1])
                with btn1:
                    if st.button("Add to cart", key=f"cart_{start}_{product_id}", use_container_width=True):
                        if "quantity" not in df.columns or pd.to_numeric(pd.Series([product.get("quantity", 1)]), errors="coerce").fillna(0).iloc[0] > 0:
                            st.session_state.cart[product_id] = st.session_state.cart.get(product_id, 0) + 1
                            st.toast(f"Added {product_name} to cart")
                            st.rerun()
                        else:
                            st.warning("This product is out of stock.")
                with btn2:
                    is_fav = product_id in st.session_state.favorites
                    if st.button("♥" if is_fav else "♡", key=f"fav_{start}_{product_id}", use_container_width=True):
                        if is_fav:
                            st.session_state.favorites.remove(product_id)
                        else:
                            st.session_state.favorites.add(product_id)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CART ----------------
with st.expander(f"🛍️ Shopping bag ({sum(st.session_state.cart.values())} items)"):
    if not st.session_state.cart:
        st.write("Your shopping bag is empty.")
    else:
        cart_total = 0.0
        for product_id, qty in list(st.session_state.cart.items()):
            parts = product_id.split("|")
            name = parts[0]
            match = df[df["name"] == name]
            if match.empty:
                continue
            item = match.iloc[0]
            price = float(item["price"])
            line_total = price * qty
            cart_total += line_total
            c1, c2, c3 = st.columns([4, 1, 2])
            c1.write(f"**{name}**")
            c2.write(f"× {qty}")
            c3.write(f"₹{line_total:,.0f}")
            if c1.button(f"Remove {name}", key=f"remove_{product_id}"):
                del st.session_state.cart[product_id]
                st.rerun()
        st.markdown(f"### Total: ₹{cart_total:,.0f}")
        if st.button("Clear shopping bag"):
            st.session_state.cart.clear()
            st.rerun()

with st.expander(f"♡ Favourites ({len(st.session_state.favorites)})"):
    if not st.session_state.favorites:
        st.write("Tap the heart on a product to save it here.")
    else:
        for fav_id in st.session_state.favorites:
            st.write("♡ " + fav_id.split("|")[0])

# ---------------- OUTFIT SUGGESTIONS / ABOUT ----------------
with st.expander("✦ Outfit suggestions"):
    tops = filtered[filtered["category"].str.lower().str.contains(r"top|shirt|tee", regex=True)]
    bottoms = filtered[filtered["category"].str.lower().str.contains(r"bottom|pant|jean|trouser", regex=True)]
    shoes = filtered[filtered["category"].str.lower().str.contains(r"shoe|footwear|sneaker", regex=True)]
    if not tops.empty and not bottoms.empty:
        top_product, bottom_product = tops.iloc[0], bottoms.iloc[0]
        total = float(top_product["price"]) + float(bottom_product["price"])
        st.write(f"**Suggested combination:** {top_product['name']} + {bottom_product['name']}")
        st.write(f"Combined price: ₹{total:,.0f}")
        if not shoes.empty:
            shoe_product = shoes.iloc[0]
            st.write(f"Add {shoe_product['name']} for a full outfit total of ₹{total + float(shoe_product['price']):,.0f}.")
    else:
        st.info("Add tops/shirts and bottoms/jeans to your CSV to generate outfit combinations.")

with st.expander("About Style Vision AI"):
    st.write("Style Vision AI is a fashion recommendation prototype using TF-IDF and cosine similarity to rank products against a text search. Budget, category, colour, style and occasion filters narrow the catalogue.")
    st.caption("Recommendations depend on the accuracy and variety of the product data and image URLs in your CSV.")

st.markdown('<div class="footer"><span>◇ STYLE VISION AI · Smarter choices. Better style.</span><span>Fashion discovery, tailored to you.</span></div>', unsafe_allow_html=True)
