"""
app.py — Full Analytics Dashboard for Restaurant Review Sentiment Analysis
Run: streamlit run app.py
"""
import os, re, warnings, pickle
from collections import Counter
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

st.set_page_config(page_title="Restaurant Sentiment Dashboard",
                   page_icon="🍽️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp { background-color: #0F1117; color: #FAFAFA; }
  section[data-testid="stSidebar"] { background-color: #1E2130; }
  .metric-box { background:#1E2130; border-radius:12px; padding:20px;
                border:1px solid #2D3748; text-align:center; margin-bottom:10px; }
  .metric-value { font-size:2.2rem; font-weight:800; }
  .metric-label { color:#94A3B8; font-size:0.85rem; }
  .badge-pos { background:#064e3b; color:#34d399; border-radius:50px;
               padding:5px 16px; font-weight:700; display:inline-block; }
  .badge-neg { background:#7f1d1d; color:#fca5a5; border-radius:50px;
               padding:5px 16px; font-weight:700; display:inline-block; }
  .badge-neu { background:#1e3a5f; color:#93c5fd; border-radius:50px;
               padding:5px 16px; font-weight:700; display:inline-block; }
  h1, h2, h3 { color: #FFFFFF !important; }
  .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight:600; }
  .stTabs [aria-selected="true"] { color:#FFFFFF; border-bottom:2px solid #38BDF8; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/restaurant_reviews.csv", parse_dates=["date"])
    df["review"] = df["review"].str.strip()
    df["rating"] = df["rating"].astype(int)
    df["label"]  = df["true_sentiment"]
    df["rating_str"] = df["rating"].astype(str)

    STOPWORDS = set("a an the is was are were be been have has had do does did will would could should may might i me my we our you your he him his she her it its they them their this that these those and or but not at by for in of on to with as from up out if so all one any each just also then than too very".split())
    def clean(text):
        if not isinstance(text, str): return ""
        text = re.sub(r"[^a-z\s]", " ", text.lower())
        return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)
    df["clean"]      = df["review"].apply(clean)
    df["review_len"] = df["review"].apply(lambda x: len(str(x).split()))
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df

@st.cache_resource
def load_model():
    path = "models/sentiment_pipeline.pkl"
    if os.path.exists(path):
        with open(path,"rb") as f: return pickle.load(f)
    return None

def predict_sentiment(text, model):
    STOPWORDS = set("a an the is was are were be been have has had do does did will would could should may might i me my we our you your he him his she her it its they them their this that these those and or but not at by for in of on to with as from up out if so all".split())
    def clean(t):
        t = re.sub(r"[^a-z\s]", " ", t.lower())
        return " ".join(w for w in t.split() if w not in STOPWORDS and len(w)>2)
    cleaned = clean(text)
    pred = model.predict([cleaned])[0] if model else "Unknown"
    proba = model.predict_proba([cleaned])[0] if model else [0.33,0.33,0.34]
    classes = model.classes_ if model else ["Negative","Neutral","Positive"]
    return pred, dict(zip(classes, proba))

def top_words(series, n=12):
    return pd.DataFrame(Counter(" ".join(series.dropna()).split()).most_common(n), columns=["word","count"])

# ── Load ──────────────────────────────────────────────────────────────────────
df    = load_data()
model = load_model()
COLORS = {"Positive":"#2ECC71","Negative":"#E74C3C","Neutral":"#3498DB"}
plt.rcParams.update({"figure.facecolor":"#1E2130","axes.facecolor":"#1E2130",
                     "axes.edgecolor":"#2D3748","text.color":"white",
                     "xtick.color":"#94A3B8","ytick.color":"#94A3B8",
                     "axes.labelcolor":"#94A3B8","grid.color":"#2D3748"})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Restaurant Sentiment Dashboard")
    st.markdown("---")
    st.markdown("### 🔍 Filters")
    selected_restaurants = st.multiselect(
        "Restaurant", options=sorted(df["restaurant_name"].unique()),
        default=sorted(df["restaurant_name"].unique()))
    selected_sentiment = st.multiselect(
        "Sentiment", options=["Positive","Negative","Neutral"],
        default=["Positive","Negative","Neutral"])
    rating_range = st.slider("Star Rating", 1, 5, (1,5))
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Reviews", f"{len(df):,}")
    st.metric("Restaurants",   df["restaurant_name"].nunique())
    st.metric("Avg Rating",    f"{df['rating'].mean():.2f} ★")

# Apply filters
mask = (df["restaurant_name"].isin(selected_restaurants) &
        df["label"].isin(selected_sentiment) &
        df["rating"].between(*rating_range))
dff = df[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🍽️ Restaurant Review Sentiment Dashboard")
st.markdown(f"*Showing **{len(dff):,}** reviews · {dff['restaurant_name'].nunique()} restaurants · Filters active*")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "☁️ Word Analysis", "🏠 By Restaurant", "📅 Trends", "🔍 Predict"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("Total Reviews", f"{len(dff):,}", "#38BDF8"),
        ("Positive",  f"{(dff['label']=='Positive').sum():,} ({(dff['label']=='Positive').mean()*100:.0f}%)", "#34D399"),
        ("Negative",  f"{(dff['label']=='Negative').sum():,} ({(dff['label']=='Negative').mean()*100:.0f}%)", "#F87171"),
        ("Neutral",   f"{(dff['label']=='Neutral').sum():,} ({(dff['label']=='Neutral').mean()*100:.0f}%)",  "#93C5FD"),
        ("Avg Rating", f"{dff['rating'].mean():.2f} ★", "#FBB724"),
    ]
    for col, (label, value, color) in zip([col1,col2,col3,col4,col5], kpis):
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### ⭐ Rating Distribution")
        fig, ax = plt.subplots(figsize=(7,4))
        rc = dff["rating"].value_counts().sort_index()
        bar_colors = ["#E74C3C","#E67E22","#F1C40F","#2ECC71","#27AE60"]
        bars = ax.bar(rc.index, rc.values, color=bar_colors, edgecolor="#1E2130", linewidth=1.5, width=0.6)
        ax.set_xlabel("Star Rating"); ax.set_ylabel("Count")
        ax.set_title("Rating Distribution", fontsize=13, fontweight="bold")
        for x,y in zip(rc.index, rc.values):
            ax.text(x, y+2, str(y), ha="center", fontsize=10, fontweight="bold", color="white")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("#### 🎭 Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(7,4))
        counts = dff["label"].value_counts()
        ax.bar(counts.index, counts.values,
               color=[COLORS.get(l,"grey") for l in counts.index],
               edgecolor="#1E2130", linewidth=1.5, width=0.5)
        ax.set_xlabel("Sentiment"); ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution", fontsize=13, fontweight="bold")
        for i,(label,v) in enumerate(counts.items()):
            ax.text(i, v+2, str(v), ha="center", fontsize=11, fontweight="bold", color="white")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 📝 Review Length by Sentiment")
        fig, ax = plt.subplots(figsize=(7,4))
        for lbl, col in COLORS.items():
            subset = dff[dff["label"]==lbl]["review_len"]
            if len(subset): ax.hist(subset, bins=15, color=col, alpha=0.72, label=lbl, edgecolor="#1E2130")
        ax.axvline(dff["review_len"].mean(), color="white", linestyle="--", linewidth=2,
                   label=f"Mean={dff['review_len'].mean():.0f}")
        ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency"); ax.legend()
        ax.set_title("Review Length by Sentiment", fontsize=12, fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_d:
        st.markdown("#### 🔥 Rating × Sentiment Heatmap")
        fig, ax = plt.subplots(figsize=(7,4))
        cross = pd.crosstab(dff["rating"], dff["label"])
        sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.5, linecolor="#1E2130", ax=ax,
                    cbar_kws={"label":"Count"})
        ax.set_xlabel("Sentiment"); ax.set_ylabel("Star Rating")
        ax.set_title("Rating vs Sentiment Alignment", fontsize=12, fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Stacked bar
    st.markdown("#### 📊 Sentiment Mix by Star Rating (%)")
    fig, ax = plt.subplots(figsize=(12,4))
    cross2 = pd.crosstab(dff["rating"], dff["label"], normalize="index") * 100
    for col in ["Positive","Negative","Neutral"]:
        if col not in cross2.columns: cross2[col] = 0
    cross2[["Positive","Negative","Neutral"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#2ECC71","#E74C3C","#3498DB"], edgecolor="#1E2130", linewidth=0.8)
    ax.set_xlabel("Star Rating"); ax.set_ylabel("Percentage (%)")
    ax.set_title("Sentiment Mix per Star Rating", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WORD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### ☁️ Word Frequency Analysis")
    col_a, col_b, col_c = st.columns(3)
    subsets = [
        (dff["clean"], "All Reviews", "#38BDF8"),
        (dff[dff["label"]=="Positive"]["clean"], "✅ Positive Reviews", "#34D399"),
        (dff[dff["label"]=="Negative"]["clean"], "❌ Negative Reviews", "#F87171"),
    ]
    for col, (series, title, color) in zip([col_a,col_b,col_c], subsets):
        with col:
            st.markdown(f"#### {title}")
            data = top_words(series, 12)
            fig, ax = plt.subplots(figsize=(5,5))
            ax.barh(data["word"][::-1], data["count"][::-1], color=color, edgecolor="#1E2130", alpha=0.85)
            ax.set_xlabel("Frequency")
            for i,c in enumerate(data["count"][::-1]):
                ax.text(c+0.2, i, str(c), va="center", fontsize=9, color="white")
            ax.set_title(f"Top 12 Words", fontsize=11, fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### 💡 Sentiment Keyword Comparison")
    pos_words = top_words(dff[dff["label"]=="Positive"]["clean"], 15)
    neg_words = top_words(dff[dff["label"]=="Negative"]["clean"], 15)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ What customers LOVE**")
        st.dataframe(pos_words.style.background_gradient(cmap="Greens", subset=["count"]),
                     use_container_width=True, height=420)
    with col2:
        st.markdown("**❌ Most common COMPLAINTS**")
        st.dataframe(neg_words.style.background_gradient(cmap="Reds", subset=["count"]),
                     use_container_width=True, height=420)

    st.markdown("---")
    # N-gram analysis
    st.markdown("### 🔗 Top Bigrams (2-word phrases)")
    from sklearn.feature_extraction.text import CountVectorizer
    col_e, col_f = st.columns(2)
    for col, (subset, title, color) in zip([col_e, col_f], [
        (dff[dff["label"]=="Positive"]["clean"], "Positive Bigrams", "#34D399"),
        (dff[dff["label"]=="Negative"]["clean"], "Negative Bigrams", "#F87171"),
    ]):
        with col:
            text_data = subset.dropna().tolist()
            if len(text_data) > 5:
                cv = CountVectorizer(ngram_range=(2,2), max_features=10)
                cv.fit(text_data)
                X = cv.transform(text_data)
                bigram_counts = pd.DataFrame({
                    "bigram": cv.get_feature_names_out(),
                    "count": X.toarray().sum(axis=0)
                }).sort_values("count", ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.barh(bigram_counts["bigram"][::-1], bigram_counts["count"][::-1],
                        color=color, edgecolor="#1E2130", alpha=0.85)
                ax.set_title(f"Top Bigrams — {title}", fontsize=12, fontweight="bold")
                ax.set_xlabel("Frequency")
                fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BY RESTAURANT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🏠 Restaurant Performance Dashboard")

    rest = dff.groupby("restaurant_name").agg(
        Reviews=("review","count"),
        Avg_Rating=("rating","mean"),
        Pct_Positive=("label",lambda x:(x=="Positive").mean()*100),
        Pct_Negative=("label",lambda x:(x=="Negative").mean()*100),
        Pct_Neutral=("label",lambda x:(x=="Neutral").mean()*100),
    ).round(1).sort_values("Avg_Rating", ascending=False).reset_index()
    rest.columns = ["Restaurant","Reviews","Avg Rating","% Positive","% Negative","% Neutral"]

    # Scoreboard
    st.dataframe(
        rest.style
            .background_gradient(subset=["Avg Rating"], cmap="RdYlGn", vmin=1, vmax=5)
            .background_gradient(subset=["% Positive"], cmap="Greens")
            .background_gradient(subset=["% Negative"], cmap="Reds"),
        use_container_width=True, height=450)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Average Rating per Restaurant")
        fig, ax = plt.subplots(figsize=(7,6))
        colors_rest = plt.cm.RdYlGn(np.linspace(0.15,0.9,len(rest)))
        ax.barh(rest["Restaurant"][::-1], rest["Avg Rating"][::-1],
                color=colors_rest, edgecolor="#1E2130")
        ax.axvline(dff["rating"].mean(), color="white", linestyle="--", linewidth=2,
                   label=f"Overall avg {dff['rating'].mean():.2f}★")
        ax.set_xlabel("Average Star Rating"); ax.legend(); ax.set_xlim(0,5)
        for i,v in enumerate(rest["Avg Rating"][::-1]):
            ax.text(v+0.05, i, f"{v:.2f}★", va="center", fontsize=9, color="white")
        ax.set_title("Avg Rating per Restaurant", fontsize=12, fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("#### Sentiment Breakdown per Restaurant")
        fig, ax = plt.subplots(figsize=(7,6))
        rest_sorted = rest.sort_values("% Positive", ascending=True)
        y = np.arange(len(rest_sorted))
        ax.barh(y, rest_sorted["% Positive"], color="#2ECC71", alpha=0.85, label="Positive")
        ax.barh(y, rest_sorted["% Negative"], left=rest_sorted["% Positive"], color="#E74C3C", alpha=0.85, label="Negative")
        ax.barh(y, rest_sorted["% Neutral"], left=rest_sorted["% Positive"]+rest_sorted["% Negative"], color="#3498DB", alpha=0.85, label="Neutral")
        ax.set_yticks(y); ax.set_yticklabels(rest_sorted["Restaurant"], fontsize=9)
        ax.set_xlabel("Percentage (%)"); ax.legend(loc="lower right")
        ax.set_title("Sentiment Breakdown per Restaurant", fontsize=12, fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Drill-down
    st.markdown("---")
    st.markdown("#### 🔎 Restaurant Deep Dive")
    selected_rest = st.selectbox("Select a restaurant", options=sorted(dff["restaurant_name"].unique()))
    rest_df = dff[dff["restaurant_name"]==selected_rest]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Reviews", len(rest_df))
    c2.metric("Avg Rating", f"{rest_df['rating'].mean():.2f}★")
    c3.metric("% Positive", f"{(rest_df['label']=='Positive').mean()*100:.0f}%")
    c4.metric("% Negative", f"{(rest_df['label']=='Negative').mean()*100:.0f}%")
    st.markdown("**Sample Positive Reviews:**")
    for r in rest_df[rest_df["label"]=="Positive"]["review"].head(2):
        st.success(r)
    st.markdown("**Sample Negative Reviews:**")
    for r in rest_df[rest_df["label"]=="Negative"]["review"].head(2):
        st.error(r)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📅 Time-Series Analysis")

    dff2 = dff.copy()
    dff2["ym"] = dff2["date"].dt.to_period("M")
    monthly = dff2.groupby("ym").agg(
        reviews=("review","count"), avg_rating=("rating","mean")).reset_index()
    monthly["ym_str"] = monthly["ym"].astype(str)

    # Line chart
    st.markdown("#### Monthly Review Volume & Average Rating")
    fig, ax1 = plt.subplots(figsize=(14,5)); ax2 = ax1.twinx()
    ax1.bar(range(len(monthly)), monthly["reviews"], color="#3498DB", alpha=0.6, label="Review Count")
    ax2.plot(range(len(monthly)), monthly["avg_rating"], color="#E74C3C", linewidth=2.5,
             marker="o", markersize=4, label="Avg Rating")
    step = max(1,len(monthly)//10)
    ax1.set_xticks(range(0,len(monthly),step))
    ax1.set_xticklabels(monthly["ym_str"].iloc[::step], rotation=45, ha="right")
    ax1.set_xlabel("Month"); ax1.set_ylabel("Review Count", color="#3498DB")
    ax2.set_ylabel("Average Rating", color="#E74C3C"); ax2.set_ylim(1,5)
    ax1.set_title("Monthly Review Volume & Rating Trend", fontsize=13, fontweight="bold")
    l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc="upper left")
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # Sentiment over time
    st.markdown("#### Sentiment Trend Over Time")
    monthly_sent = dff2.groupby(["ym","label"]).size().unstack(fill_value=0)
    monthly_sent.index = monthly_sent.index.astype(str)
    fig, ax = plt.subplots(figsize=(14,5))
    for lbl, color in COLORS.items():
        if lbl in monthly_sent.columns:
            ax.plot(monthly_sent.index, monthly_sent[lbl], color=color,
                    linewidth=2, label=lbl, marker="o", markersize=3)
    step2 = max(1,len(monthly_sent)//10)
    ax.set_xticks(range(0,len(monthly_sent),step2))
    ax.set_xticklabels(monthly_sent.index[::step2], rotation=45, ha="right")
    ax.set_xlabel("Month"); ax.set_ylabel("Review Count"); ax.legend()
    ax.set_title("Monthly Sentiment Trend", fontsize=13, fontweight="bold")
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🔍 Real-time Sentiment Prediction")
    st.markdown("Enter any restaurant review below to get an instant sentiment classification.")

    if "history" not in st.session_state:
        st.session_state.history = []

    col_inp, col_out = st.columns([1.1, 0.9], gap="large")
    with col_inp:
        review_input = st.text_area("Your Review", placeholder="e.g. The pasta was amazing! Staff was so friendly and the ambiance was perfect.", height=150)
        ex1, ex2, ex3 = st.columns(3)
        if ex1.button("😊 Positive"): review_input = "Absolutely fantastic food and the staff made us feel so welcome. Will definitely return!"
        if ex2.button("😞 Negative"): review_input = "Terrible experience. Waited over an hour, food was cold and the manager was completely rude."
        if ex3.button("😐 Neutral"):  review_input = "Decent place for a quick lunch. The food was okay and nothing particularly stood out."
        predict_btn = st.button("🔍 Analyse Sentiment", use_container_width=True)

    with col_out:
        if predict_btn and review_input.strip() and model:
            pred, proba = predict_sentiment(review_input, model)
            badge = {"Positive":"badge-pos","Negative":"badge-neg","Neutral":"badge-neu"}[pred]
            emoji = {"Positive":"✅","Negative":"❌","Neutral":"💬"}[pred]
            st.markdown(f'<span class="{badge}">{emoji} {pred}</span>', unsafe_allow_html=True)
            st.markdown("")

            # Probability bar chart
            fig, ax = plt.subplots(figsize=(6,3))
            classes = list(proba.keys()); scores = list(proba.values())
            colors_p = [COLORS.get(c,"grey") for c in classes]
            ax.barh(classes, scores, color=colors_p, edgecolor="#1E2130")
            ax.set_xlim(0,1); ax.set_xlabel("Confidence Score")
            ax.set_title("Prediction Confidence", fontsize=12, fontweight="bold")
            for i,(c,s) in enumerate(zip(classes,scores)):
                ax.text(s+0.01, i, f"{s*100:.1f}%", va="center", fontsize=11, fontweight="bold", color="white")
            fig.tight_layout(); st.pyplot(fig); plt.close()

            st.session_state.history.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "review": review_input[:70]+"…" if len(review_input)>70 else review_input,
                "sentiment": pred, "confidence": f"{max(scores)*100:.1f}%"
            })
        elif predict_btn and not model:
            st.warning("Model not found. Run the notebook first to generate models/sentiment_pipeline.pkl")
        elif predict_btn:
            st.warning("Please enter a review.")
        else:
            st.info("Enter a review and click **Analyse Sentiment**.")

    # History
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕐 Prediction History")
        hist_df = pd.DataFrame(st.session_state.history[::-1])
        col_hist1, col_hist2 = st.columns([2,1])
        with col_hist1:
            st.dataframe(hist_df, use_container_width=True, height=300)
        with col_hist2:
            hcounts = hist_df["sentiment"].value_counts()
            fig, ax = plt.subplots(figsize=(4,4))
            ax.pie(hcounts.values, labels=hcounts.index,
                   colors=[COLORS.get(l,"grey") for l in hcounts.index],
                   autopct="%1.0f%%", startangle=90,
                   wedgeprops={"edgecolor":"#1E2130","linewidth":2})
            ax.set_title("Session Mix", fontsize=11, fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close()
        csv = hist_df.to_csv(index=False)
        st.download_button("⬇️ Download History CSV", data=csv,
                           file_name="prediction_history.csv", mime="text/csv")
        if st.button("🗑️ Clear History"): st.session_state.history=[]; st.rerun()

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#64748B;font-size:0.82rem;'>🍽️ Restaurant Sentiment Dashboard · Built with Python, Scikit-learn & Streamlit · 2025</div>", unsafe_allow_html=True)
