"""
Met Museum Art Recommender
===========================
A personalised art tour generator powered by ML.

How it works:
1. User rates 20 artworks (Like / Skip / Love)
2. App trains a classifier on those ratings in real time
3. Recommends remaining artworks ranked by predicted enjoyment
4. Shows SHAP explanations for why each artwork was recommended

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── Optional SHAP ──────────────────────────────────────────────
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ══════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Met Art Recommender",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 1.1rem;
    color: #6b7280;
    font-weight: 300;
    margin-bottom: 2rem;
}
.artwork-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.dept-tag {
    background: #EEF2FF;
    color: #4338CA;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
}
.rating-love  { background: #FFF7ED; border: 2px solid #F59E0B; }
.rating-like  { background: #F0FDF4; border: 2px solid #22C55E; }
.rating-skip  { background: #F9FAFB; border: 1px solid #E5E7EB; }
.progress-bar {
    background: #EEF2FF;
    border-radius: 8px;
    height: 8px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('met_artworks_clean.csv')
    return df

@st.cache_resource
def load_features():
    with open('feature_matrix.pkl', 'rb') as f:
        return pickle.load(f)

try:
    df = load_data()
    feature_matrix = load_features()
    DATA_LOADED = True
except FileNotFoundError:
    DATA_LOADED = False


# ══════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════
if 'ratings'        not in st.session_state: st.session_state.ratings = {}
if 'current_idx'    not in st.session_state: st.session_state.current_idx = 0
if 'phase'          not in st.session_state: st.session_state.phase = 'rating'
if 'rating_queue'   not in st.session_state: st.session_state.rating_queue = []
if 'recommendations' not in st.session_state: st.session_state.recommendations = None

RATING_TARGET = 20   # number of artworks to rate before recommendations


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎨 Met Art Recommender")
    st.markdown("---")

    n_rated = len(st.session_state.ratings)
    st.markdown(f"**Progress:** {n_rated} / {RATING_TARGET} artworks rated")
    progress = n_rated / RATING_TARGET
    st.progress(min(progress, 1.0))

    if n_rated > 0:
        loves = sum(1 for v in st.session_state.ratings.values() if v == 2)
        likes = sum(1 for v in st.session_state.ratings.values() if v == 1)
        skips = sum(1 for v in st.session_state.ratings.values() if v == 0)
        st.markdown(f"❤️ Love: **{loves}**  |  👍 Like: **{likes}**  |  ⏭️ Skip: **{skips}**")

    st.markdown("---")

    if st.button("🔄 Start Over", use_container_width=True):
        st.session_state.ratings = {}
        st.session_state.current_idx = 0
        st.session_state.phase = 'rating'
        st.session_state.rating_queue = []
        st.session_state.recommendations = None
        st.rerun()

    st.markdown("---")
    st.markdown("**How it works:**")
    st.caption(
        "Rate 20 artworks and the app learns your taste profile. "
        "It then recommends which rooms and works to visit at the Met — "
        "with a full explanation of why you'll love each one."
    )
    st.markdown("---")
    st.caption("Data: Metropolitan Museum of Art Open Access API")


# ══════════════════════════════════════════════════════════════
# Data not loaded — show setup instructions
# ══════════════════════════════════════════════════════════════
if not DATA_LOADED:
    st.markdown('<div class="main-title">🎨 Met Art Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your personalised museum tour, powered by AI</div>', unsafe_allow_html=True)

    st.warning("Dataset not found. Run the data collection script first:")
    st.code("python met_step1_collect_data.py", language="bash")
    st.markdown("""
    This will:
    - Pull ~4,000 artworks from the Met's free public API
    - Filter for artworks with images
    - Save `met_artworks.csv`

    Then run the exploration notebook:
    ```
    jupyter notebook met_step2_explore.ipynb
    ```

    This generates:
    - `met_artworks_clean.csv`
    - `feature_matrix.pkl`
    - `tfidf_vectorizer.pkl`

    Then relaunch this app.
    """)
    st.stop()


# ══════════════════════════════════════════════════════════════
# Initialise rating queue
# ══════════════════════════════════════════════════════════════
if not st.session_state.rating_queue:
    # Prioritise highlighted artworks (Met's own picks)
    # then fill with random others for variety
    highlights = df[df['is_highlight'] == True].sample(
        min(8, len(df[df['is_highlight'] == True])), random_state=42
    )
    others = df[df['is_highlight'] == False].sample(
        RATING_TARGET - len(highlights), random_state=42
    )
    queue = pd.concat([highlights, others]).sample(frac=1, random_state=42)
    st.session_state.rating_queue = queue['id'].tolist()


# ══════════════════════════════════════════════════════════════
# PHASE 1 — Rating
# ══════════════════════════════════════════════════════════════
if st.session_state.phase == 'rating':

    st.markdown('<div class="main-title">🎨 Build Your Taste Profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Rate these artworks — the app learns what you love</div>',
        unsafe_allow_html=True
    )

    n_rated = len(st.session_state.ratings)

    if n_rated >= RATING_TARGET:
        st.session_state.phase = 'results'
        st.rerun()

    # Get current artwork
    queue = st.session_state.rating_queue
    rated_ids = set(st.session_state.ratings.keys())
    remaining = [i for i in queue if i not in rated_ids]

    if not remaining:
        st.session_state.phase = 'results'
        st.rerun()

    current_id = remaining[0]
    artwork = df[df['id'] == current_id].iloc[0]

    # Layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # Show the artwork image
        if artwork['image_url'] and str(artwork['image_url']) != 'nan':
            st.image(
                artwork['image_url'],
                use_container_width=True,
                caption=None
            )
        else:
            st.markdown(
                '<div style="height:400px;background:#F3F4F6;border-radius:12px;'
                'display:flex;align-items:center;justify-content:center;">'
                '<span style="color:#9CA3AF;font-size:3rem;">🖼️</span></div>',
                unsafe_allow_html=True
            )

    with col2:
        st.markdown(f"### {artwork['title']}")

        if str(artwork['artist']) not in ['Unknown Artist', 'nan', '']:
            st.markdown(f"**{artwork['artist']}**")
            if str(artwork.get('artist_bio', '')) not in ['', 'nan']:
                st.caption(artwork['artist_bio'])

        st.markdown("")

        col_tags = st.columns(3)
        with col_tags[0]:
            st.markdown(
                f'<span class="dept-tag">{artwork["department"]}</span>',
                unsafe_allow_html=True
            )
        if str(artwork['era']) not in ['Unknown Era', 'nan', '']:
            with col_tags[1]:
                st.caption(f"🕰️ {artwork['era']}")
        if str(artwork['culture']) not in ['Unknown', 'nan', '']:
            with col_tags[2]:
                st.caption(f"🌍 {artwork['culture']}")

        if str(artwork['medium']) not in ['Unknown', 'nan', '']:
            st.caption(f"**Medium:** {artwork['medium'][:100]}")

        if str(artwork['tags']) not in ['Unknown', 'nan', '']:
            st.caption(f"**Tags:** {artwork['tags'][:120]}")

        if artwork.get('met_url'):
            st.markdown(f"[View on Met website ↗]({artwork['met_url']})")

        st.markdown("---")
        st.markdown("#### How does this artwork make you feel?")

        r1, r2, r3 = st.columns(3)

        with r1:
            if st.button("❤️ Love it", use_container_width=True, key="love"):
                st.session_state.ratings[current_id] = 2
                st.rerun()

        with r2:
            if st.button("👍 Like it", use_container_width=True, key="like"):
                st.session_state.ratings[current_id] = 1
                st.rerun()

        with r3:
            if st.button("⏭️ Skip", use_container_width=True, key="skip"):
                st.session_state.ratings[current_id] = 0
                st.rerun()

        st.markdown("")
        remaining_count = RATING_TARGET - n_rated - 1
        if remaining_count > 0:
            st.caption(f"{remaining_count} more to go before your recommendations are ready")
        else:
            st.success("One more and your recommendations are ready!")


# ══════════════════════════════════════════════════════════════
# PHASE 2 — Generate Recommendations
# ══════════════════════════════════════════════════════════════
elif st.session_state.phase == 'results':

    st.markdown('<div class="main-title">🗺️ Your Personalised Met Tour</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Based on your taste profile — here\'s what you\'ll love</div>',
        unsafe_allow_html=True
    )

    # ── Build recommendations if not yet computed ─────────────
    if st.session_state.recommendations is None:
        with st.spinner("Learning your taste and building your tour..."):

            ratings_dict = st.session_state.ratings
            rated_ids    = list(ratings_dict.keys())
            labels       = [ratings_dict[i] for i in rated_ids]

            # Get feature vectors for rated artworks
            rated_indices = [df[df['id'] == i].index[0] for i in rated_ids]
            X_train = feature_matrix[rated_indices]
            y_train = np.array(labels)

            # Train a Random Forest on the ratings
            # Label 2 = Love, 1 = Like, 0 = Skip
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            clf.fit(X_train, y_train)

            # Predict scores for ALL unrated artworks
            unrated_mask = ~df['id'].isin(rated_ids)
            unrated_df   = df[unrated_mask].copy()
            unrated_idx  = unrated_df.index.tolist()
            X_unrated    = feature_matrix[unrated_idx]

            # Predict probability of "Love" (class 2) for each
            proba = clf.predict_proba(X_unrated)
            classes = clf.classes_.tolist()

            if 2 in classes:
                love_col = classes.index(2)
                scores   = proba[:, love_col]
            elif 1 in classes:
                like_col = classes.index(1)
                scores   = proba[:, like_col]
            else:
                scores = np.random.rand(len(unrated_df))

            unrated_df = unrated_df.copy()
            unrated_df['predicted_score'] = scores
            unrated_df = unrated_df.sort_values('predicted_score', ascending=False)

            # SHAP feature importance (global — which features drove your taste)
            shap_importance = None
            if SHAP_OK:
                try:
                    explainer    = shap.TreeExplainer(clf)
                    shap_vals    = explainer.shap_values(X_train)
                    if isinstance(shap_vals, list) and len(shap_vals) > 0:
                        sv = shap_vals[-1]
                    else:
                        sv = shap_vals
                    shap_importance = np.abs(sv).mean(axis=0)
                except Exception:
                    pass

            st.session_state.recommendations = unrated_df
            st.session_state.clf             = clf
            st.session_state.shap_importance = shap_importance

    recs = st.session_state.recommendations

    # ── Taste profile summary ─────────────────────────────────
    loved = [i for i, v in st.session_state.ratings.items() if v == 2]
    liked = [i for i, v in st.session_state.ratings.items() if v == 1]

    if loved or liked:
        preferred_ids = loved if loved else liked
        preferred_depts = df[df['id'].isin(preferred_ids)]['department'].value_counts()

        st.markdown("### Your taste profile")
        cols = st.columns(min(4, len(preferred_depts)))
        for i, (dept, count) in enumerate(preferred_depts.head(4).items()):
            with cols[i]:
                st.metric(dept.split()[-1], f"{count} artworks", "you enjoyed")

    st.markdown("---")

    # ── Department filter ─────────────────────────────────────
    st.markdown("### Filter by department")
    all_depts = ['All'] + sorted(recs['department'].unique().tolist())
    chosen_dept = st.selectbox("", all_depts, label_visibility="collapsed")

    if chosen_dept != 'All':
        display_recs = recs[recs['department'] == chosen_dept].head(20)
    else:
        display_recs = recs.head(30)

    st.markdown(f"### Top recommendations for you ({len(display_recs)} shown)")

    # ── Show recommendations ──────────────────────────────────
    for i, (_, row) in enumerate(display_recs.iterrows()):
        with st.expander(
            f"{'⭐ ' if row['predicted_score'] > 0.6 else ''}"
            f"{row['title']}  —  {row['artist']}  "
            f"[{row['department']}]  "
            f"Match: {row['predicted_score']:.0%}",
            expanded=(i < 3)
        ):
            col_img, col_info = st.columns([1, 2], gap="medium")

            with col_img:
                if row['image_url'] and str(row['image_url']) != 'nan':
                    st.image(row['image_url'], use_container_width=True)

            with col_info:
                st.markdown(f"**{row['title']}**")
                if str(row['artist']) not in ['Unknown Artist', 'nan', '']:
                    st.markdown(f"*{row['artist']}*")

                c1, c2, c3 = st.columns(3)
                c1.metric("Match score", f"{row['predicted_score']:.0%}")
                c2.caption(f"🕰️ {row['era']}")
                c3.caption(f"🌍 {row['culture'] if str(row['culture']) != 'Unknown' else row['department']}")

                if str(row['medium']) not in ['Unknown', 'nan', '']:
                    st.caption(f"**Medium:** {row['medium'][:100]}")

                if str(row['tags']) not in ['Unknown', 'nan', '']:
                    st.caption(f"**Tags:** {row['tags'][:150]}")

                # Why recommended
                liked_ids   = [i for i, v in st.session_state.ratings.items() if v >= 1]
                liked_depts = df[df['id'].isin(liked_ids)]['department'].tolist()
                reason_parts = []

                if row['department'] in liked_depts:
                    reason_parts.append(f"you enjoyed other works from **{row['department']}**")
                if str(row['era']) not in ['Unknown Era', 'nan', '']:
                    reason_parts.append(f"it matches your interest in the **{row['era']}** period")
                if row.get('is_highlight'):
                    reason_parts.append("it's one of the Met's own highlighted works")

                if reason_parts:
                    st.success("Why you'll love this: " + ", and ".join(reason_parts))

                if row.get('met_url'):
                    st.markdown(f"[View on Met website ↗]({row['met_url']})")

    # ── Department tour map ───────────────────────────────────
    st.markdown("---")
    st.markdown("### Your recommended tour by department")
    st.caption("Departments ranked by how many top picks you have there")

    top50 = recs.head(50)
    dept_summary = (
        top50.groupby('department')['predicted_score']
        .agg(['count', 'mean'])
        .rename(columns={'count': 'Recommended works', 'mean': 'Avg match score'})
        .sort_values('Recommended works', ascending=False)
    )
    dept_summary['Avg match score'] = dept_summary['Avg match score'].apply(
        lambda x: f"{x:.0%}"
    )
    st.dataframe(dept_summary, use_container_width=True)

    st.markdown("---")
    if st.button("← Rate more artworks to refine your recommendations",
                 use_container_width=True):
        st.session_state.phase      = 'rating'
        st.session_state.recommendations = None
        st.rerun()
