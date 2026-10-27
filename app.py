import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="Style Vision AI",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Style Vision AI")
st.subheader("Your Personal AI Fashion Assistant")
st.write(
    "Discover fashion that matches your style, preferences and budget."
)

# ---------------- LOAD PRODUCTS ----------------

@st.cache_data
def load_products():
    data = pd.read_csv("E:\\Style Vision\\clothes.csv")
    required = [
        "name", "category", "color", "style",
        "occasion", "price", "description", "image"
    ]

    for column in required:
        if column not in data.columns:
            raise ValueError(
                f"Missing column in clothes.csv: {column}"
            )

    data["price"] = pd.to_numeric(
        data["price"], errors="coerce"
    )
    data = data.dropna(subset=["price"]).copy()

    for column in required:
        data[column] = data[column].fillna("").astype(str)

    return data


try:
    df = load_products()
except Exception as error:
    st.error(f"Could not load products.csv: {error}")
    st.info(
        "Make sure products.csv is in the same folder as app.py "
        "and contains the required column headings."
    )
    st.stop()

if df.empty:
    st.warning("Add some products to products.csv first.")
    st.stop()

# ---------------- PREPARE AI MODEL ----------------

df["search_text"] = (
    df["name"] + " " +
    df["category"] + " " +
    df["color"] + " " +
    df["style"] + " " +
    df["occasion"] + " " +
    df["description"]
)

vectorizer = TfidfVectorizer(stop_words="english")
product_vectors = vectorizer.fit_transform(df["search_text"])

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("🎯 Your Preferences")

max_price = int(df["price"].max())
min_price = int(df["price"].min())

budget = st.sidebar.slider(
    "Maximum budget (₹)",
    min_value=0,
    max_value=max(1000, max_price),
    value=max(1000, min_price),
    step=100
)

def filter_options(column):
    values = sorted(
        value for value in df[column].unique()
        if value.strip()
    )
    return ["Any"] + values

category = st.sidebar.selectbox(
    "Category", filter_options("category")
)

color = st.sidebar.selectbox(
    "Preferred colour", filter_options("color")
)

style = st.sidebar.selectbox(
    "Fashion style", filter_options("style")
)

occasion = st.sidebar.selectbox(
    "Occasion", filter_options("occasion")
)

# ---------------- FASHION SEARCH ----------------

st.header("✨ Find Your Style")

preference = st.text_input(
    "Describe what you're looking for",
    placeholder="e.g. black casual clothes for college"
)

if st.button("Find My Style", type="primary"):
    st.session_state["search_clicked"] = True

if "search_clicked" not in st.session_state:
    st.session_state["search_clicked"] = False

# ---------------- FILTER PRODUCTS ----------------
df["price"] = pd.to_numeric(
    df["price"].astype(str).str.replace(r"[₹,\s]", "", regex=True),
    errors="coerce"
)
df = df.dropna(subset=["price"]).copy()

filtered = df[df["price"] <= budget].copy()

if category != "Any":
    filtered = filtered[
        filtered["category"] == category
    ]

if color != "Any":
    filtered = filtered[
        filtered["color"] == color
    ]

if style != "Any":
    filtered = filtered[
        filtered["style"] == style
    ]

if occasion != "Any":
    filtered = filtered[
        filtered["occasion"] == occasion
    ]

# ---------------- AI RECOMMENDATIONS ----------------

if st.session_state["search_clicked"]:
    query_parts = [
        preference,
        "" if category == "Any" else category,
        "" if color == "Any" else color,
        "" if style == "Any" else style,
        "" if occasion == "Any" else occasion
    ]

    query = " ".join(query_parts).strip()

    if filtered.empty:
        st.warning(
            "No products match these filters. "
            "Try increasing your budget or changing your preferences."
        )

    else:
        if query:
            query_vector = vectorizer.transform([query])
            scores = cosine_similarity(
                query_vector,
                product_vectors
            ).flatten()

            filtered["AI Match Score"] = scores[filtered.index]
            filtered = filtered.sort_values(
                "AI Match Score", ascending=False
            )
        else:
            filtered["AI Match Score"] = 0.0
            filtered = filtered.sort_values("price")

        st.success(
            f"Found {len(filtered)} products within your preferences!"
        )

        st.subheader("🛍️ Recommended for You")

        columns = st.columns(3)

        for position, (_, product) in enumerate(
            filtered.iterrows()
        ):
            with columns[position % 3]:
                st.markdown("---")

                if product["image"].startswith(("https://", "http://")):
                    st.image(
                        product["image"],
                        use_container_width=True
                    )

                st.markdown(f"### {product['name']}")
                st.write(f"**Price:** ₹{product['price']:.0f}")
                st.write(f"**Category:** {product['category']}")
                st.write(f"**Colour:** {product['color']}")
                st.write(f"**Style:** {product['style']}")
                st.write(f"**Occasion:** {product['occasion']}")
                st.caption(product["description"])

                score = product["AI Match Score"] * 100
                st.progress(
                    min(1.0, max(0.0, float(score))),
                    text=f"Text similarity: {score:.0f}%"
                )

                feedback_key = f"feedback_{product.name}"
                feedback = st.selectbox(
                    "Your feedback",
                    ["Choose feedback", "👍 Like", "👎 Dislike"],
                    key=feedback_key
                )

                if feedback in ["👍 Like", "👎 Dislike"]:
                    st.caption(
                        "Feedback selected for this demo. "
                        "It does not retrain the model."
                    )

        # ---------------- OUTFIT SUGGESTIONS ----------------

        st.subheader("👕 Outfit Suggestions")

        tops = filtered[
            filtered["category"].str.lower().isin(
                ["tops", "top", "shirt", "t-shirt"]
            )
        ]
        bottoms = filtered[
            filtered["category"].str.lower().isin(
                ["bottoms", "bottom", "pants", "jeans"]
            )
        ]
        shoes = filtered[
            filtered["category"].str.lower().isin(
                ["shoes", "footwear"]
            )
        ]

        if not tops.empty and not bottoms.empty:
            top_product = tops.iloc[0]
            bottom_product = bottoms.iloc[0]

            outfit_total = (
                float(top_product["price"]) +
                float(bottom_product["price"])
            )

            st.write(
                f"**Suggested combination:** "
                f"{top_product['name']} + {bottom_product['name']}"
            )
            st.write(f"Combined price: ₹{outfit_total:.0f}")

            if not shoes.empty:
                shoe_product = shoes.iloc[0]
                full_total = outfit_total + float(shoe_product["price"])

                st.write(
                    f"Add {shoe_product['name']} "
                    f"for a total of ₹{full_total:.0f}."
                )
        else:
            st.info(
                "Add products from different categories to your "
                "catalogue to generate more outfit combinations."
            )

else:
    st.info(
        "Choose your filters, describe your style, "
        "and click **Find My Style** to see recommendations."
    )

# ---------------- PROJECT INFORMATION ----------------

with st.expander("ℹ️ About this project"):
    st.write(
        "Style Vision AI is a fashion recommendation prototype. "
        "It uses TF-IDF and cosine similarity to compare a user's "
        "text preferences with product descriptions. Filters help "
        "limit results by budget, colour, style, category and occasion."
    )
    st.write(
        "This is a content-based recommendation demo, not a trained "
        "commercial AI system. Its recommendations depend on the "
        "products available in products.csv."
    )

st.caption("Style Vision AI | AI Project Prototype")

