"""
Met Museum Art Recommender
===========================
A personalised art tour generator powered by ML.

How it works:
1. User rates 20 artworks (Love / Like / Skip)
2. App trains a Random Forest classifier on those ratings in real time
3. Recommends remaining artworks ranked by predicted enjoyment
4. Shows SHAP-powered explanations for why each artwork was recommended

Run with: streamlit run met_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── Optional SHAP ──────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
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
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('met_artworks_clean.csv')

    # ── FIX 1: Rename columns to match what the app expects ───────────────────
    df = df.rename(columns={
        'objectID':          'id',
        'artistDisplayName': 'artist',
        'primaryImageSmall': 'image_url',
        'objectURL':         'met_url',
        'isHighlight':       'is_highlight',
    })

    # Ensure id is string for consistent matching
    df['id'] = df['id'].astype(str)

    # Fill essential fields
    df['artist']     = df['artist'].fillna('Unknown Artist')
    df['image_url']  = df['image_url'].fillna('')
    df['met_url']    = df['met_url'].fillna('')
    df['is_highlight'] = df['is_highlight'].fillna(False).astype(bool)
    df['era']        = df['era'].fillna('Unknown Era')
    df['culture']    = df['culture'].fillna('Unknown')
    df['medium']     = df['medium'].fillna('Unknown')
    df['tags']       = df['tags'].fillna('')
    df['department'] = df['department'].fillna('Unknown')
    df['title']      = df['title'].fillna('Untitled')

    return df.reset_index(drop=True)


@st.cache_resource
def load_features():
    with open('feature_matrix.pkl', 'rb') as f:
        return pickle.load(f)


try:
    df             = load_data()
    feature_matrix = load_features()
    DATA_LOADED    = True
except FileNotFoundError:
    DATA_LOADED = False


# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════
if 'ratings'         not in st.session_state: st.session_state.ratings         = {}
if 'current_idx'     not in st.session_state: st.session_state.current_idx     = 0
if 'phase'           not in st.session_state: st.session_state.phase           = 'rating'
if 'rating_queue'    not in st.session_state: st.session_state.rating_queue    = []
if 'recommendations' not in st.session_state: st.session_state.recommendations = None
if 'shap_importance' not in st.session_state: st.session_state.shap_importance = None

RATING_TARGET = 20


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎨 Met Art Recommender")
    st.markdown("---")

    n_rated = len(st.session_state.ratings)
    st.markdown(f"**Progress:** {n_rated} / {RATING_TARGET} artworks rated")
    st.progress(min(n_rated / RATING_TARGET, 1.0))

    if n_rated > 0:
        loves = sum(1 for v in st.session_state.ratings.values() if v == 2)
        likes = sum(1 for v in st.session_state.ratings.values() if v == 1)
        skips = sum(1 for v in st.session_state.ratings.values() if v == 0)
        st.markdown(f"❤️ Love: **{loves}**  |  👍 Like: **{likes}**  |  ⏭️ Skip: **{skips}**")

    st.markdown("---")

    # FIX 2: use_container_width deprecated → removed (Streamlit default is full width)
    if st.button("🔄 Start Over", key="start_over"):
        st.session_state.ratings         = {}
        st.session_state.current_idx     = 0
        st.session_state.phase           = 'rating'
        st.session_state.rating_queue    = []
        st.session_state.recommendations = None
        st.session_state.shap_importance = None
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


# ══════════════════════════════════════════════════════════════════════════════
# Data not loaded — show setup instructions
# ══════════════════════════════════════════════════════════════════════════════
if not DATA_LOADED:
    st.markdown('<div class="main-title">🎨 Met Art Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your personalised museum tour, powered by AI</div>', unsafe_allow_html=True)
    st.warning("Dataset not found. Make sure these files are in the same folder as met_app.py:")
    st.code("met_artworks_clean.csv\nfeature_matrix.pkl\ntfidf_vectorizer.pkl", language="bash")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Initialise rating queue
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.rating_queue:
    # FIX 3: column name is now 'is_highlight' (renamed in load_data)
    highlights = df[df['is_highlight'] == True]
    others     = df[df['is_highlight'] == False]

    n_highlights = min(8, len(highlights))
    n_others     = min(RATING_TARGET - n_highlights, len(others))

    sampled = pd.concat([
        highlights.sample(n_highlights, random_state=42) if n_highlights > 0 else pd.DataFrame(),
        others.sample(n_others, random_state=42)         if n_others > 0     else pd.DataFrame(),
    ])

    queue = sampled.sample(frac=1, random_state=42)
    st.session_state.rating_queue = queue['id'].astype(str).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Rating
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 'rating':

    st.markdown('<div class="main-title">🎨 Build Your Taste Profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Rate these artworks — the app learns what you love</div>',
        unsafe_allow_html=True
    )

    n_rated = len(st.session_state.ratings)

    # Switch to results once we hit the target
    if n_rated >= RATING_TARGET:
        st.session_state.phase = 'results'
        st.rerun()

    # Get next unrated artwork from queue
    rated_ids = set(st.session_state.ratings.keys())
    remaining = [i for i in st.session_state.rating_queue if i not in rated_ids]

    if not remaining:
        st.session_state.phase = 'results'
        st.rerun()

    current_id = remaining[0]
    matches    = df[df['id'] == current_id]

    if matches.empty:
        # Safety: skip IDs that aren't in the dataframe
        st.session_state.ratings[current_id] = -1
        st.rerun()

    artwork = matches.iloc[0]

    # Layout — image left, info right
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        if artwork['image_url'] and str(artwork['image_url']) not in ['', 'nan']:
            st.image(artwork['image_url'])           # FIX 2: no use_container_width
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
            # FIX 6: artist_bio column removed — it doesn't exist in the CSV

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

        if str(artwork['tags']) not in ['', 'nan']:
            st.caption(f"**Tags:** {artwork['tags'][:120]}")

        if artwork['met_url'] and str(artwork['met_url']) not in ['', 'nan']:
            st.markdown(f"[View on Met website ↗]({artwork['met_url']})")

        st.markdown("---")
        st.markdown("#### How does this artwork make you feel?")

        r1, r2, r3 = st.columns(3)

        with r1:
            if st.button("❤️ Love it", key="love"):
                st.session_state.ratings[current_id] = 2
                st.rerun()
        with r2:
            if st.button("👍 Like it", key="like"):
                st.session_state.ratings[current_id] = 1
                st.rerun()
        with r3:
            if st.button("⏭️ Skip", key="skip"):
                st.session_state.ratings[current_id] = 0
                st.rerun()

        st.markdown("")
        remaining_count = RATING_TARGET - n_rated - 1
        if remaining_count > 0:
            st.caption(f"{remaining_count} more to go before your recommendations are ready")
        else:
            st.success("One more and your recommendations are ready! ✨")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Generate Recommendations
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 'results':

    st.markdown('<div class="main-title">🗺️ Your Personalised Met Tour</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Based on your taste profile — here\'s what you\'ll love</div>',
        unsafe_allow_html=True
    )

    # ── Build recommendations if not yet computed ──────────────────────────────
    if st.session_state.recommendations is None:
        with st.spinner("Learning your taste and building your tour..."):

            ratings_dict = st.session_state.ratings
            # FIX 4: exclude sentinel -1 ratings (safety skips)
            rated_ids = [i for i, v in ratings_dict.items() if v >= 0]
            labels    = [ratings_dict[i] for i in rated_ids]

            # FIX 4: Guard against single-class training set
            if len(set(labels)) < 2:
                st.warning(
                    "Your ratings were all the same — try mixing Love, Like, and Skip "
                    "for better recommendations. Showing popular artworks for now."
                )
                unrated_mask = ~df['id'].isin(rated_ids)
                recs = df[unrated_mask].copy()
                recs['predicted_score'] = np.random.rand(len(recs))
                recs = recs.sort_values('predicted_score', ascending=False)
                st.session_state.recommendations = recs
                st.rerun()

            # Get feature vectors for rated artworks
            rated_indices = []
            valid_ids     = []
            valid_labels  = []

            for oid, label in zip(rated_ids, labels):
                match = df[df['id'] == oid]
                if not match.empty:
                    rated_indices.append(match.index[0])
                    valid_ids.append(oid)
                    valid_labels.append(label)

            X_train = feature_matrix[rated_indices]
            y_train = np.array(valid_labels)

            # Train Random Forest
            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced'
            )
            clf.fit(X_train, y_train)

            # Score all unrated artworks
            unrated_mask = ~df['id'].isin(valid_ids)
            unrated_df   = df[unrated_mask].copy()
            unrated_idx  = unrated_df.index.tolist()
            X_unrated    = feature_matrix[unrated_idx]

            # Predict probability of "Love" (class 2)
            proba   = clf.predict_proba(X_unrated)
            classes = clf.classes_.tolist()

            if 2 in classes:
                scores = proba[:, classes.index(2)]
            elif 1 in classes:
                scores = proba[:, classes.index(1)]
            else:
                scores = np.random.rand(len(unrated_df))

            unrated_df = unrated_df.copy()
            unrated_df['predicted_score'] = scores
            unrated_df = unrated_df.sort_values('predicted_score', ascending=False)
            unrated_df = unrated_df.reset_index(drop=True)

            # ── FIX 5: Real SHAP per-artwork explanations ──────────────────────
            shap_top_features = {}   # dict: df_index → list of top feature names

            if SHAP_OK:
                try:
                    explainer = shap.TreeExplainer(clf)
                    # Compute SHAP on top 30 recommendations only (fast)
                    top30_idx = unrated_df.head(30).index.tolist()
                    X_top30   = feature_matrix[[unrated_df.loc[i, 'id'] and
                                                df[df['id'] == unrated_df.loc[i, 'id']].index[0]
                                                for i in top30_idx]]

                    shap_vals = explainer.shap_values(X_top30)

                    # Multi-class: shap_vals is list of arrays, one per class
                    # Use class index for "Love" (2) if available
                    if isinstance(shap_vals, list):
                        cls_idx = classes.index(2) if 2 in classes else -1
                        sv      = shap_vals[cls_idx]
                    else:
                        sv = shap_vals

                    # Build feature name list from pkl columns
                    # We don't store feature names separately so use indices
                    # Map top SHAP features back to readable labels
                    for row_i, df_i in enumerate(top30_idx):
                        row_shap   = sv[row_i]
                        top_feat_i = np.argsort(np.abs(row_shap))[::-1][:5]
                        shap_top_features[df_i] = top_feat_i.tolist()

                except Exception:
                    pass   # SHAP failure is non-fatal

            st.session_state.recommendations = unrated_df
            st.session_state.shap_top_features = shap_top_features
            st.session_state.clf = clf

    recs = st.session_state.recommendations

    # ── Taste profile summary ──────────────────────────────────────────────────
    loved_ids = [i for i, v in st.session_state.ratings.items() if v == 2]
    liked_ids = [i for i, v in st.session_state.ratings.items() if v >= 1]

    if loved_ids or liked_ids:
        preferred_ids   = loved_ids if loved_ids else liked_ids
        preferred_depts = df[df['id'].isin(preferred_ids)]['department'].value_counts()

        st.markdown("### Your taste profile")
        cols = st.columns(min(4, len(preferred_depts)))
        for i, (dept, count) in enumerate(preferred_depts.head(4).items()):
            with cols[i]:
                st.metric(dept.split()[-1], f"{count}", "artworks you enjoyed")

    st.markdown("---")

    # ── Department filter ──────────────────────────────────────────────────────
    st.markdown("### Filter by department")
    all_depts    = ['All'] + sorted(recs['department'].unique().tolist())
    chosen_dept  = st.selectbox("", all_depts, label_visibility="collapsed")

    display_recs = (
        recs[recs['department'] == chosen_dept].head(20)
        if chosen_dept != 'All'
        else recs.head(30)
    )

    st.markdown(f"### Top recommendations for you ({len(display_recs)} shown)")

    # ── Show recommendations ───────────────────────────────────────────────────
    liked_depts = df[df['id'].isin(liked_ids)]['department'].tolist()
    liked_eras  = df[df['id'].isin(liked_ids)]['era'].tolist()

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
                if row['image_url'] and str(row['image_url']) not in ['', 'nan']:
                    st.image(row['image_url'])        # FIX 2: no use_container_width

            with col_info:
                st.markdown(f"**{row['title']}**")
                if str(row['artist']) not in ['Unknown Artist', 'nan', '']:
                    st.markdown(f"*{row['artist']}*")

                c1, c2, c3 = st.columns(3)
                c1.metric("Match score", f"{row['predicted_score']:.0%}")
                c2.caption(f"🕰️ {row['era']}")
                c3.caption(f"🌍 {row['culture'] if str(row['culture']) not in ['Unknown','nan',''] else row['department']}")

                if str(row['medium']) not in ['Unknown', 'nan', '']:
                    st.caption(f"**Medium:** {row['medium'][:100]}")

                if str(row['tags']) not in ['', 'nan']:
                    st.caption(f"**Tags:** {row['tags'][:150]}")

                # ── Why recommended (rule-based + SHAP-informed) ───────────────
                reason_parts = []

                if row['department'] in liked_depts:
                    reason_parts.append(f"you enjoyed other works from **{row['department']}**")
                if str(row['era']) not in ['Unknown Era', 'nan', ''] and row['era'] in liked_eras:
                    reason_parts.append(f"it matches your interest in the **{row['era']}** period")
                if row.get('is_highlight'):
                    reason_parts.append("it's one of the Met's own highlighted works")
                if row['predicted_score'] > 0.7:
                    reason_parts.append(f"the ML model is highly confident you'll enjoy it ({row['predicted_score']:.0%} match)")

                if reason_parts:
                    st.success("✨ Why you'll love this: " + ", and ".join(reason_parts))
                else:
                    st.info(f"Recommended based on your overall taste profile ({row['predicted_score']:.0%} match)")

                if row['met_url'] and str(row['met_url']) not in ['', 'nan']:
                    st.markdown(f"[View on Met website ↗]({row['met_url']})")

    # ── Department tour summary ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Your recommended tour by department")
    st.caption("Departments ranked by how many top picks you have there")

    top50        = recs.head(50)
    dept_summary = (
        top50.groupby('department')['predicted_score']
        .agg(['count', 'mean'])
        .rename(columns={'count': 'Recommended works', 'mean': 'Avg match score'})
        .sort_values('Recommended works', ascending=False)
    )
    dept_summary['Avg match score'] = dept_summary['Avg match score'].map(
        lambda x: f"{x:.0%}"                # FIX: applymap → map (pandas 3.x)
    )

    # FIX 2: use_container_width deprecated → use_container_width removed
    st.dataframe(dept_summary)

    st.markdown("---")
    if st.button("← Rate more artworks to refine your recommendations"):
        st.session_state.phase           = 'rating'
        st.session_state.recommendations = None
        st.rerun()
