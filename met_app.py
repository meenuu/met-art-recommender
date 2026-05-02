
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
Met Museum Art Tour Planner v3 — Premium
==========================================
- True randomness every session (timestamp seed)
- Must-see masterpieces shown FIRST before rating
- Descriptions always shown (with or without image)
- Premium museum-grade UI
- Content warnings
- Gallery roadmap with walking order
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import hashlib
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
    page_title="Met Museum | Personal Tour",
    page_icon="🏛️",
layout="wide",
    initial_sidebar_state="expanded"
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
   font-family: 'Inter', sans-serif;
    background-color: #FAFAF8;
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.2rem;

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.5rem;
    font-weight: 300;
    color: #1C1C1C;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.subtitle {
    font-size: 1.1rem;
    color: #6b7280;
.hero-sub {
    font-size: 1rem;
    color: #6B6B6B;
   font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9B8B6E;
    margin-bottom: 0.5rem;
}
.must-see-banner {
    background: linear-gradient(135deg, #1C1C1C 0%, #2D2416 100%);
    border-radius: 16px;
    padding: 2rem;
    color: white;
   margin-bottom: 2rem;
}
.artwork-card {
.must-see-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 300;
    color: #F5E6C8;
    margin-bottom: 0.25rem;
}
.gold-badge {
    background: linear-gradient(135deg, #B8960C, #D4AF37);
    color: #1C1C1C;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.dept-pill {
    background: #F0EBE1;
    color: #5C4A2A;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
}
.artwork-number {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    font-weight: 300;
    color: #E8E0D5;
    line-height: 1;
}
.rating-question {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 400;
    color: #1C1C1C;
    margin-bottom: 1rem;
}
.desc-box {
    background: #F5F0E8;
    border-left: 3px solid #D4AF37;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #3D3D3D;
    line-height: 1.7;
}
.no-image-box {
    background: #1C1C1C;
    border-radius: 12px;
    min-height: 380px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    text-align: center;
}
.score-ring {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #B8960C;
}
.roadmap-dept {
   background: white;
    border: 1px solid #E8E0D5;
   border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    padding: 1.5rem;
   margin-bottom: 1rem;
}
.dept-tag {
    background: #EEF2FF;
    color: #4338CA;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
.why-box {
    background: linear-gradient(135deg, #FFF8E7, #FFF3D4);
    border: 1px solid #E8C84A;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #5C4A00;
    margin-top: 0.75rem;
}
.content-flag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
   font-weight: 500;
    margin-right: 4px;
}
.divider {
    border: none;
    border-top: 1px solid #E8E0D5;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# Constants
# ══════════════════════════════════════════════════════════════════════════════
RATING_TARGET = 20
TOP_N_RECS    = 40
MUST_SEE_N    = 12

FAMOUS_ARTIST_NAMES = [
    "van gogh", "monet", "picasso", "rembrandt", "vermeer",
    "cézanne", "renoir", "degas", "manet", "matisse",
    "sargent", "homer", "o'keeffe", "hopper", "hokusai",
    "goya", "el greco", "turner", "raphael", "botticelli",
    "caravaggio", "rubens", "titian", "chagall", "dalí",
    "pollock", "rothko", "warhol", "lichtenstein", "cassatt",
    "whistler", "eakins", "pissarro", "seurat", "gauguin",
    "kandinsky", "klee", "velázquez", "constable", "corot",
]

DEPT_TIME = {
    "European Paintings": 45,
    "American Paintings and Sculpture": 35,
    "Modern and Contemporary Art": 40,
    "Asian Art": 30,
    "Egyptian Art": 25,
    "Greek and Roman Art": 30,
    "Islamic Art": 25,
    "The American Wing": 35,
    "Robert Lehman Collection": 20,
    "default": 15,
}

CONTENT_FLAGS = {
    "nudity":    ("#FEE2E2", "#DC2626", "🔞 Nudity"),
    "violence":  ("#FEF3C7", "#D97706", "⚔️ Violence"),
    "religious": ("#EDE9FE", "#7C3AED", "✝️ Religious"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
df = pd.read_csv('met_artworks_clean.csv')

    # ── FIX 1: Rename columns to match what the app expects ───────────────────
    df = df.rename(columns={
    'objectID': 'id',
    'artistDisplayName': 'artist',
    'primaryImageSmall': 'image_url',
    'objectURL': 'met_url',
    'isHighlight': 'is_highlight'
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
        'is_famous_artist':  'is_famous',
    }
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    df['id']            = df['id'].astype(str)
    df['artist']        = df['artist'].fillna('Unknown Artist')
    df['image_url']     = df['image_url'].fillna('')
    df['met_url']       = df['met_url'].fillna('')
    df['is_highlight']  = df['is_highlight'].fillna(False).astype(bool)
    df['era']           = df['era'].fillna('Unknown Era')
    df['culture']       = df['culture'].fillna('')
    df['medium']        = df['medium'].fillna('')
    df['tags']          = df['tags'].fillna('')
    df['department']    = df['department'].fillna('Unknown')
    df['title']         = df['title'].fillna('Untitled')
    df['description']   = df['description'].fillna('') if 'description' in df.columns else ''
    df['content_flags'] = df['content_flags'].fillna('') if 'content_flags' in df.columns else ''
    df['style']         = df['style'].fillna('') if 'style' in df.columns else ''
    df['subject']       = df['subject'].fillna('') if 'subject' in df.columns else ''

    # Derive is_famous from artist name if column missing
    if 'is_famous' not in df.columns:
        df['is_famous'] = df['artist'].apply(
            lambda a: any(name in str(a).lower() for name in FAMOUS_ARTIST_NAMES)
        )
    else:
        df['is_famous'] = df['is_famous'].fillna(False).astype(bool)
        # Also catch any famous artists missed by the flag
        df['is_famous'] = df['is_famous'] | df['artist'].apply(
            lambda a: any(name in str(a).lower() for name in FAMOUS_ARTIST_NAMES)
        )

return df.reset_index(drop=True)

@@ -123,112 +279,312 @@ def load_features():
feature_matrix = load_features()
DATA_LOADED    = True
except FileNotFoundError:
    DATA_LOADED = False
    DATA_LOADED    = False


# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# Session state — use timestamp as unique session key for true randomness
# ══════════════════════════════════════════════════════════════════════════════
if 'ratings'         not in st.session_state: st.session_state.ratings         = {}
if 'current_idx'     not in st.session_state: st.session_state.current_idx     = 0
if 'phase'           not in st.session_state: st.session_state.phase           = 'rating'
if 'rating_queue'    not in st.session_state: st.session_state.rating_queue    = []
if 'recommendations' not in st.session_state: st.session_state.recommendations = None
if 'shap_importance' not in st.session_state: st.session_state.shap_importance = None
if 'session_id' not in st.session_state:
    # New unique ID every time the app is freshly opened or Start Over is clicked
    st.session_state.session_id    = str(time.time_ns())
    st.session_state.ratings       = {}
    st.session_state.phase         = 'must_sees'   # NEW: show must-sees first
    st.session_state.rating_queue  = []
    st.session_state.recs          = None
    st.session_state.must_sees_df  = None
    st.session_state.hide_nudity   = False
    st.session_state.hide_violence = False

RATING_TARGET = 20

def reset_session():
    """Completely reset — new session ID guarantees fresh randomness."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎨 Met Art Recommender")
    st.markdown("---")
def get_content_filter():
    f = []
    if st.session_state.get('hide_nudity'):   f.append('nudity')
    if st.session_state.get('hide_violence'): f.append('violence')
    return f


def apply_filter(dataframe):
    cf = get_content_filter()
    if not cf or 'content_flags' not in dataframe.columns:
        return dataframe
    return dataframe[
        dataframe['content_flags'].apply(
            lambda x: not any(f in str(x) for f in cf)
        )
    ]


def render_flags(flags_str):
    if not flags_str or str(flags_str) in ['', 'nan']:
        return
    for flag in str(flags_str).split('|'):
        flag = flag.strip()
        if flag in CONTENT_FLAGS:
            bg, color, label = CONTENT_FLAGS[flag]
            st.markdown(
                f'<span class="content-flag" style="background:{bg};color:{color};">'
                f'{label}</span>',
                unsafe_allow_html=True
            )

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
def render_description(row):
    """Always show a description — from description field or assembled from metadata."""
    desc = str(row.get('description', ''))

    # Build rich description from available fields if description is thin
    parts = []
    if str(row.get('medium', '')) not in ['', 'nan', 'unknown']:
        parts.append(f"**Medium:** {row['medium']}")
    if str(row.get('culture', '')) not in ['', 'nan', 'unknown']:
        parts.append(f"**Culture:** {row['culture']}")
    if str(row.get('era', '')) not in ['', 'nan', 'Unknown Era', 'unknown']:
        parts.append(f"**Era:** {row['era'].replace('_', ' ').title()}")
    if str(row.get('style', '')) not in ['', 'nan', 'other', 'unknown']:
        parts.append(f"**Style:** {row['style'].replace('_', ' ').title()}")
    if str(row.get('subject', '')) not in ['', 'nan', 'other', 'unknown']:
        parts.append(f"**Subject:** {row['subject'].replace('_', ' ').title()}")
    if str(row.get('tags', '')) not in ['', 'nan']:
        tags = str(row['tags'])[:100]
        parts.append(f"**Tags:** {tags}")

    # Use description if good, else use assembled parts
    if len(desc) > 30 and desc not in ['nan', '']:
        display = desc[:400]
    elif parts:
        display = "  \n".join(parts)
    else:
        display = f"*{row['title']}* — part of the Met's {row['department']} collection."

    st.markdown(
        f'<div class="desc-box">{display}</div>',
        unsafe_allow_html=True
    )


def get_must_sees(exclude_ids=None):
    """Get must-see artworks: famous artists + Met highlights, diverse departments."""
    exclude_ids = set(str(i) for i in (exclude_ids or []))
    pool        = apply_filter(df[~df['id'].isin(exclude_ids)])

    famous     = pool[pool['is_famous'] == True]
    highlights = pool[(pool['is_highlight'] == True) & (pool['is_famous'] == False)]

    combined = pd.concat([famous, highlights]).drop_duplicates(subset=['id'])
    if combined.empty:
        return pool.sample(min(MUST_SEE_N, len(pool)))

    # Ensure variety — max 2 per artist, max 3 per department
    seen_artists = {}
    seen_depts   = {}
    must_sees    = []

    # Use session-based seed for consistent must-sees within a session
    # but different across sessions
    rng = np.random.default_rng(
        int(hashlib.md5(st.session_state.session_id.encode()).hexdigest()[:8], 16)
    )
    shuffled = combined.sample(frac=1, random_state=int(rng.integers(0, 10000)))

    for _, row in shuffled.iterrows():
        artist = str(row['artist'])[:30]
        dept   = row['department']
        if seen_artists.get(artist, 0) >= 2:   continue
        if seen_depts.get(dept, 0) >= 3:       continue
        must_sees.append(row)
        seen_artists[artist] = seen_artists.get(artist, 0) + 1
        seen_depts[dept]     = seen_depts.get(dept, 0) + 1
        if len(must_sees) >= MUST_SEE_N:
            break

    return pd.DataFrame(must_sees)


def build_rating_queue(exclude_ids=None):
    """Build a fresh, random rating queue using session timestamp as seed."""
    exclude_ids = set(str(i) for i in (exclude_ids or []))
    filtered    = apply_filter(df[~df['id'].isin(exclude_ids)])

    # Seed from session_id — different every session, including after Start Over
    seed = int(hashlib.md5(st.session_state.session_id.encode()).hexdigest()[:8], 16) % 100000

    famous     = filtered[filtered['is_famous'] == True]
    highlights = filtered[(filtered['is_highlight'] == True) & (filtered['is_famous'] == False)]
    rest       = filtered[
        ~filtered['id'].isin(
            pd.concat([famous, highlights])['id'].tolist()
            if not famous.empty else highlights['id'].tolist()
        )
    ]

    n_famous     = min(6, len(famous))
    n_highlights = min(4, len(highlights))
    n_rest       = max(0, RATING_TARGET - n_famous - n_highlights)

    parts = []
    np.random.seed(seed)
    if n_famous > 0:
        parts.append(famous.sample(n_famous, random_state=seed))
    if n_highlights > 0:
        parts.append(highlights.sample(min(n_highlights, len(highlights)), random_state=seed+1))
    if n_rest > 0 and len(rest) > 0:
        parts.append(rest.sample(min(n_rest, len(rest)), random_state=seed+2))

    queue = pd.concat(parts).sample(frac=1, random_state=seed+3) if parts else filtered.sample(
        min(RATING_TARGET, len(filtered)), random_state=seed
)
    st.markdown("---")
    st.caption("Data: Metropolitan Museum of Art Open Access API")
    return queue['id'].astype(str).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Data not loaded — show setup instructions
# Data not loaded
# ══════════════════════════════════════════════════════════════════════════════
if not DATA_LOADED:
    st.markdown('<div class="main-title">🎨 Met Art Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your personalised museum tour, powered by AI</div>', unsafe_allow_html=True)
    st.warning("Dataset not found. Make sure these files are in the same folder as met_app.py:")
    st.code("met_artworks_clean.csv\nfeature_matrix.pkl\ntfidf_vectorizer.pkl", language="bash")
    st.markdown('<div class="hero-title">🏛️ Met Museum<br>Personal Tour</div>', unsafe_allow_html=True)
    st.error("Data files not found. Please upload: `met_artworks_clean.csv`, `feature_matrix.pkl`, `tfidf_vectorizer.pkl`")
st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Initialise rating queue
# PHASE 0 — Must-Sees (shown before rating starts)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.rating_queue:
    # FIX 3: column name is now 'is_highlight' (renamed in load_data)
    highlights = df[df['is_highlight'] == True]
    others     = df[df['is_highlight'] == False]
if st.session_state.phase == 'must_sees':

    # Header
    st.markdown('<div class="hero-sub">The Metropolitan Museum of Art</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Your Personal<br>Museum Tour</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Content preferences
    col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
    with col_p1:
        st.markdown("**Set your content preferences before we begin:**")
    with col_p2:
        st.session_state.hide_nudity   = st.checkbox("Hide artworks with nudity")
    with col_p3:
        st.session_state.hide_violence = st.checkbox("Hide violent imagery")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Build must-sees
    if st.session_state.must_sees_df is None:
        st.session_state.must_sees_df = get_must_sees()

    must_sees = st.session_state.must_sees_df

    # Must-see banner
    st.markdown("""
    <div class="must-see-banner">
        <div style="font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;
                    color:#D4AF37;margin-bottom:0.5rem;">Non-Negotiables</div>
        <div class="must-see-title">Must-See Masterpieces</div>
        <div style="color:#C8B89A;font-size:0.9rem;margin-top:0.5rem;">
            These iconic works are automatically included in your tour —
            celebrated masters and Met highlights you cannot miss.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show must-sees in a 3-column grid
    if not must_sees.empty:
        cols = st.columns(3)
        for i, (_, row) in enumerate(must_sees.iterrows()):
            with cols[i % 3]:
                has_image = row['image_url'] and str(row['image_url']) not in ['', 'nan']

                # Image or dark placeholder
                if has_image:
                    st.image(row['image_url'], use_column_width=True)
                else:
                    st.markdown(
                        f'<div class="no-image-box">'
                        f'<div>'
                        f'<div style="font-size:2rem;margin-bottom:0.5rem;">🖼️</div>'
                        f'<div style="color:#9B8B6E;font-size:0.8rem;">Image restricted</div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

                # Title + artist
                st.markdown(
                    f'<div style="margin-top:0.5rem;">'
                    f'<span class="gold-badge">Must See</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**{row['title']}**")
                if str(row['artist']) not in ['Unknown Artist', 'nan', '']:
                    st.caption(f"*{row['artist']}*")

    n_highlights = min(8, len(highlights))
    n_others     = min(RATING_TARGET - n_highlights, len(others))
                # Always show description
                render_description(row)

    sampled = pd.concat([
        highlights.sample(n_highlights, random_state=42) if n_highlights > 0 else pd.DataFrame(),
        others.sample(n_others, random_state=42)         if n_others > 0     else pd.DataFrame(),
    ])
                render_flags(row.get('content_flags', ''))

    queue = sampled.sample(frac=1, random_state=42)
    st.session_state.rating_queue = queue['id'].astype(str).tolist()
                if row.get('met_url', '') and str(row['met_url']) not in ['', 'nan']:
                    st.markdown(f"[View on Met ↗]({row['met_url']})")

                st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Rating
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 'rating':
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="main-title">🎨 Build Your Taste Profile</div>', unsafe_allow_html=True)
    # CTA to start rating
st.markdown(
        '<div class="subtitle">Rate these artworks — the app learns what you love</div>',
        '<div style="text-align:center;margin:2rem 0;">'
        '<div class="section-label">Step 2 of 2</div>'
        '<div style="font-family:Cormorant Garamond,serif;font-size:1.8rem;'
        'color:#1C1C1C;margin-bottom:1rem;">'
        'Now let\'s personalise the rest of your tour</div>'
        '<div style="color:#6B6B6B;margin-bottom:2rem;">'
        'Rate 20 artworks so we can recommend what else you\'ll love at the Met</div>'
        '</div>',
unsafe_allow_html=True
)

    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    with col_cta2:
        if st.button("🎨  Begin Taste Profile →", use_container_width=True):
            st.session_state.phase        = 'rating'
            st.session_state.rating_queue = build_rating_queue(
                exclude_ids=must_sees['id'].tolist() if not must_sees.empty else []
            )
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Rating
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 'rating':

n_rated = len(st.session_state.ratings)

    # Switch to results once we hit the target
if n_rated >= RATING_TARGET:
st.session_state.phase = 'results'
st.rerun()

    # Get next unrated artwork from queue
    # Top bar
    col_title, col_prog = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="hero-sub">Building Your Taste Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title" style="font-size:2.2rem;">Rate This Artwork</div>', unsafe_allow_html=True)
    with col_prog:
        st.markdown(f'<div class="artwork-number">{n_rated + 1}<span style="font-size:1.5rem;color:#B0A090">/{RATING_TARGET}</span></div>', unsafe_allow_html=True)
        st.progress(n_rated / RATING_TARGET)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Get next artwork
rated_ids = set(st.session_state.ratings.keys())
remaining = [i for i in st.session_state.rating_queue if i not in rated_ids]

@@ -240,304 +596,394 @@ def load_features():
matches    = df[df['id'] == current_id]

if matches.empty:
        # Safety: skip IDs that aren't in the dataframe
st.session_state.ratings[current_id] = -1
st.rerun()

artwork = matches.iloc[0]
    has_image = artwork['image_url'] and str(artwork['image_url']) not in ['', 'nan']

    # Layout — image left, info right
col1, col2 = st.columns([1, 1], gap="large")

with col1:
        if artwork['image_url'] and str(artwork['image_url']) not in ['', 'nan']:
            st.image(artwork['image_url'])           # FIX 2: no use_container_width
        if has_image:
            st.image(artwork['image_url'], use_column_width=True)
else:
st.markdown(
                '<div style="height:400px;background:#F3F4F6;border-radius:12px;'
                'display:flex;align-items:center;justify-content:center;">'
                '<span style="color:#9CA3AF;font-size:3rem;">🖼️</span></div>',
                '<div class="no-image-box">'
                '<div style="text-align:center;">'
                '<div style="font-size:3rem;margin-bottom:1rem;">🖼️</div>'
                '<div style="color:#9B8B6E;font-size:0.85rem;line-height:1.6;">'
                'Image rights restricted.<br>Read the description to learn about this work.'
                '</div></div></div>',
unsafe_allow_html=True
)

with col2:
        st.markdown(f"### {artwork['title']}")
        # Department + badges
        badges_html = f'<span class="dept-pill">{artwork["department"]}</span>'
        if artwork.get('is_famous'):
            badges_html += '&nbsp;<span class="gold-badge">⭐ Master</span>'
        if artwork.get('is_highlight'):
            badges_html += '&nbsp;<span class="gold-badge">Met Pick</span>'
        st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown("")

        # Title
        st.markdown(f"## {artwork['title']}")

        # Artist
if str(artwork['artist']) not in ['Unknown Artist', 'nan', '']:
            st.markdown(f"**{artwork['artist']}**")
            # FIX 6: artist_bio column removed — it doesn't exist in the CSV
            artist_line = f"*{artwork['artist']}*"
            if str(artwork.get('artistNationality', '')) not in ['', 'nan']:
                artist_line += f" · {artwork['artistNationality']}"
            st.markdown(artist_line)

        st.markdown("")
        # Date
        if str(artwork.get('objectDate', '')) not in ['', 'nan']:
            st.caption(f"📅 {artwork['objectDate']}")

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
        # Always show description
        render_description(artwork)

        if str(artwork['medium']) not in ['Unknown', 'nan', '']:
            st.caption(f"**Medium:** {artwork['medium'][:100]}")
        # Content flags
        render_flags(artwork.get('content_flags', ''))

        if str(artwork['tags']) not in ['', 'nan']:
            st.caption(f"**Tags:** {artwork['tags'][:120]}")
        # Met link
        if artwork.get('met_url', '') and str(artwork['met_url']) not in ['', 'nan']:
            st.markdown(f"[View full details on Met website ↗]({artwork['met_url']})")

        if artwork['met_url'] and str(artwork['met_url']) not in ['', 'nan']:
            st.markdown(f"[View on Met website ↗]({artwork['met_url']})")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### How does this artwork make you feel?")
        # Rating buttons
        st.markdown('<div class="rating-question">Would you visit this at the Met?</div>', unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)

with r1:
            if st.button("❤️ Love it", key="love"):
            if st.button("❤️  Love it", key="love", use_container_width=True):
st.session_state.ratings[current_id] = 2
st.rerun()
with r2:
            if st.button("👍 Like it", key="like"):
            if st.button("👍  Like it", key="like", use_container_width=True):
st.session_state.ratings[current_id] = 1
st.rerun()
with r3:
            if st.button("⏭️ Skip", key="skip"):
            if st.button("⏭️  Skip", key="skip", use_container_width=True):
st.session_state.ratings[current_id] = 0
st.rerun()

        st.markdown("")
remaining_count = RATING_TARGET - n_rated - 1
        st.markdown("")
if remaining_count > 0:
            st.caption(f"{remaining_count} more to go before your recommendations are ready")
            st.caption(f"⏳ {remaining_count} more artworks to rate")
else:
            st.success("One more and your recommendations are ready! ✨")
            st.success("🎉 Last one! Your personalised tour is about to be revealed.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Generate Recommendations
# PHASE 2 — Results + Roadmap
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
    # ── Train model ────────────────────────────────────────────────────────────
    if st.session_state.recs is None:
        with st.spinner("Analysing your taste profile and curating your tour..."):

            ratings_dict = st.session_state.ratings
            # FIX 4: exclude sentinel -1 ratings (safety skips)
            rated_ids = [i for i, v in ratings_dict.items() if v >= 0]
            labels    = [ratings_dict[i] for i in rated_ids]
            ratings_dict = {k: v for k, v in st.session_state.ratings.items() if v >= 0}
            rated_ids    = list(ratings_dict.keys())
            labels       = list(ratings_dict.values())

            # FIX 4: Guard against single-class training set
if len(set(labels)) < 2:
                st.warning(
                    "Your ratings were all the same — try mixing Love, Like, and Skip "
                    "for better recommendations. Showing popular artworks for now."
                # Fallback: show popular works
                fallback = apply_filter(df[~df['id'].isin(rated_ids)]).head(TOP_N_RECS).copy()
                fallback['predicted_score'] = 0.5
                st.session_state.recs = fallback
            else:
                rated_indices, valid_ids, valid_labels = [], [], []
                for oid, label in zip(rated_ids, labels):
                    match = df[df['id'] == oid]
                    if not match.empty:
                        rated_indices.append(match.index[0])
                        valid_ids.append(oid)
                        valid_labels.append(label)

                X_train = feature_matrix[rated_indices]
                y_train = np.array(valid_labels)

                clf = RandomForestClassifier(
                    n_estimators=300, random_state=42,
                    class_weight='balanced', n_jobs=-1
)
                unrated_mask = ~df['id'].isin(rated_ids)
                recs = df[unrated_mask].copy()
                recs['predicted_score'] = np.random.rand(len(recs))
                recs = recs.sort_values('predicted_score', ascending=False)
                st.session_state.recommendations = recs
                st.rerun()
                clf.fit(X_train, y_train)

                unrated_df  = df[~df['id'].isin(valid_ids)].copy()
                unrated_idx = unrated_df.index.tolist()
                X_unrated   = feature_matrix[unrated_idx]

                proba   = clf.predict_proba(X_unrated)
                classes = clf.classes_.tolist()

                scores = (
                    proba[:, classes.index(2)] if 2 in classes else
                    proba[:, classes.index(1)] if 1 in classes else
                    np.random.rand(len(unrated_df))
                )

                unrated_df['predicted_score'] = scores
                unrated_df = apply_filter(
                    unrated_df.sort_values('predicted_score', ascending=False)
                )
                st.session_state.recs = unrated_df.reset_index(drop=True)
                st.session_state.clf  = clf

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
        st.rerun()

    recs      = st.session_state.recs
    must_sees = st.session_state.must_sees_df

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="hero-sub">The Metropolitan Museum of Art</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Your Personal Tour</div>', unsafe_allow_html=True)

    # Taste profile chips
    liked_ids   = [i for i, v in st.session_state.ratings.items() if v >= 1]
    loved_ids   = [i for i, v in st.session_state.ratings.items() if v == 2]
    liked_df    = df[df['id'].isin(liked_ids)]
    liked_depts = liked_df['department'].tolist()
    liked_eras  = liked_df['era'].tolist()

    if liked_ids:
        top_dept  = liked_df['department'].value_counts().index[0] if len(liked_df) else ""
        top_era   = liked_df['era'].value_counts().index[0] if len(liked_df) else ""
        top_style = liked_df['style'].value_counts().index[0] if 'style' in liked_df.columns and len(liked_df) else ""

        profile_parts = []
        if top_dept:  profile_parts.append(f"🏛️ {top_dept}")
        if top_era and top_era not in ['unknown','Unknown Era']:
            profile_parts.append(f"🕰️ {top_era.replace('_',' ').title()}")
        if top_style and top_style not in ['other','unknown','']:
            profile_parts.append(f"🖌️ {top_style.replace('_',' ').title()}")

        if profile_parts:
            chips = "  ·  ".join(profile_parts)
            st.markdown(
                f'<div style="background:#F0EBE1;border-radius:8px;padding:0.6rem 1rem;'
                f'display:inline-block;margin:0.5rem 0;color:#5C4A2A;font-size:0.85rem;">'
                f'Your taste: {chips}</div>',
                unsafe_allow_html=True
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
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Show recommendations ───────────────────────────────────────────────────
    liked_depts = df[df['id'].isin(liked_ids)]['department'].tolist()
    liked_eras  = df[df['id'].isin(liked_ids)]['era'].tolist()
    # ── SECTION 1: Must-Sees ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Non-Negotiables · Always Included</div>', unsafe_allow_html=True)
    st.markdown("### ⭐ Must-See Masterpieces")
    st.caption("These iconic works are part of your tour regardless of your ratings.")

    if must_sees is not None and not must_sees.empty:
        ms_cols = st.columns(3)
        for i, (_, row) in enumerate(must_sees.head(9).iterrows()):
            with ms_cols[i % 3]:
                has_img = row['image_url'] and str(row['image_url']) not in ['', 'nan']
                if has_img:
                    st.image(row['image_url'], use_column_width=True)
                else:
                    st.markdown(
                        '<div style="background:#1C1C1C;border-radius:8px;height:180px;'
                        'display:flex;align-items:center;justify-content:center;">'
                        '<span style="color:#9B8B6E;font-size:2rem;">🖼️</span></div>',
                        unsafe_allow_html=True
                    )
                st.markdown(f'<span class="gold-badge">Must See</span>', unsafe_allow_html=True)
                st.markdown(f"**{row['title']}**")
                st.caption(f"*{row['artist']}* · {row['department']}")
                render_description(row)
                render_flags(row.get('content_flags', ''))
                if row.get('met_url', '') and str(row['met_url']) not in ['', 'nan']:
                    st.markdown(f"[Met ↗]({row['met_url']})")
                st.markdown("---")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── SECTION 2: Personalised Recommendations ───────────────────────────────
    st.markdown('<div class="section-label">Personalised · Based on Your Taste</div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Recommended For You")

    # Filters
    fcol1, fcol2 = st.columns([2, 1])
    with fcol1:
        depts       = ['All departments'] + sorted(recs['department'].unique().tolist())
        chosen_dept = st.selectbox("Department", depts, label_visibility="collapsed")
    with fcol2:
        min_score = st.slider("Min match", 0, 100, 0, 5, format="%d%%")

    display_recs = recs.copy()
    if chosen_dept != 'All departments':
        display_recs = display_recs[display_recs['department'] == chosen_dept]
    if min_score > 0:
        display_recs = display_recs[display_recs['predicted_score'] >= min_score / 100]
    display_recs = display_recs.head(TOP_N_RECS)

    st.caption(f"Showing {len(display_recs)} recommendations")
    st.markdown("")

for i, (_, row) in enumerate(display_recs.iterrows()):
        score     = row['predicted_score']
        fire      = "🔥 " if score > 0.75 else ("⭐ " if score > 0.55 else "")
        has_image = row['image_url'] and str(row['image_url']) not in ['', 'nan']

with st.expander(
            f"{'⭐ ' if row['predicted_score'] > 0.6 else ''}"
            f"{row['title']}  —  {row['artist']}  "
            f"[{row['department']}]  "
            f"Match: {row['predicted_score']:.0%}",
            f"{fire}{row['title']}  ·  {row['artist']}  ·  {row['department']}  ·  {score:.0%} match",
expanded=(i < 3)
):
            col_img, col_info = st.columns([1, 2], gap="medium")

            with col_img:
                if row['image_url'] and str(row['image_url']) not in ['', 'nan']:
                    st.image(row['image_url'])        # FIX 2: no use_container_width
            ec1, ec2 = st.columns([1, 2], gap="large")

            with col_info:
                st.markdown(f"**{row['title']}**")
            with ec1:
                if has_image:
                    st.image(row['image_url'], use_column_width=True)
                else:
                    st.markdown(
                        '<div style="background:#1C1C1C;border-radius:8px;min-height:250px;'
                        'display:flex;align-items:center;justify-content:center;">'
                        '<span style="color:#9B8B6E;font-size:2.5rem;">🖼️</span></div>',
                        unsafe_allow_html=True
                    )

            with ec2:
                # Badges
                badges = f'<span class="dept-pill">{row["department"]}</span>'
                if row.get('is_famous'): badges += '&nbsp;<span class="gold-badge">⭐ Master</span>'
                st.markdown(badges, unsafe_allow_html=True)
                st.markdown("")

                st.markdown(f"### {row['title']}")
if str(row['artist']) not in ['Unknown Artist', 'nan', '']:
st.markdown(f"*{row['artist']}*")

                c1, c2, c3 = st.columns(3)
                c1.metric("Match score", f"{row['predicted_score']:.0%}")
                c2.caption(f"🕰️ {row['era']}")
                c3.caption(f"🌍 {row['culture'] if str(row['culture']) not in ['Unknown','nan',''] else row['department']}")

                if str(row['medium']) not in ['Unknown', 'nan', '']:
                    st.caption(f"**Medium:** {row['medium'][:100]}")
                # Score
                st.markdown(
                    f'<div style="margin:0.5rem 0;">'
                    f'<span class="score-ring">{score:.0%}</span>'
                    f'<span style="color:#9B8B6E;font-size:0.8rem;margin-left:0.5rem;">match score</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if str(row['tags']) not in ['', 'nan']:
                    st.caption(f"**Tags:** {row['tags'][:150]}")
                # Always show description
                render_description(row)

                # ── Why recommended (rule-based + SHAP-informed) ───────────────
                reason_parts = []
                render_flags(row.get('content_flags', ''))

                # Why recommended
                reasons = []
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
                    reasons.append(f"you loved works from {row['department']}")
                if str(row.get('era', '')) not in ['Unknown Era', 'nan', ''] and row.get('era') in liked_eras:
                    reasons.append(f"matches your interest in {str(row['era']).replace('_', ' ')} art")
                if row.get('is_famous'): reasons.append("celebrated master artist")
                if score > 0.75:        reasons.append(f"exceptionally high {score:.0%} confidence")

                if reasons:
                    st.markdown(
                        f'<div class="why-box">✨ <strong>Why this?</strong> '
                        f'{" · ".join(reasons)}</div>',
                        unsafe_allow_html=True
                    )

                if row.get('met_url', '') and str(row['met_url']) not in ['', 'nan']:
st.markdown(f"[View on Met website ↗]({row['met_url']})")

    # ── Department tour summary ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Your recommended tour by department")
    st.caption("Departments ranked by how many top picks you have there")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── SECTION 3: Gallery Roadmap ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Gallery-by-Gallery · Walking Order</div>', unsafe_allow_html=True)
    st.markdown("### 🗺️ Your Tour Roadmap")
    st.caption("Your full tour organised by gallery. Visit in this order for the best experience.")

    # Combine must-sees + top recs
    ms_ids      = must_sees['id'].tolist() if must_sees is not None and not must_sees.empty else []
    top_recs    = display_recs.head(40)
    all_ids     = ms_ids + [i for i in top_recs['id'].tolist() if i not in ms_ids]
    roadmap_df  = df[df['id'].isin(all_ids)].copy()

    score_map                     = dict(zip(display_recs['id'], display_recs['predicted_score']))
    roadmap_df['predicted_score'] = roadmap_df['id'].map(score_map).fillna(0.95)

    top50        = recs.head(50)
dept_summary = (
        top50.groupby('department')['predicted_score']
        roadmap_df.groupby('department')['predicted_score']
.agg(['count', 'mean'])
        .rename(columns={'count': 'Recommended works', 'mean': 'Avg match score'})
        .sort_values('Recommended works', ascending=False)
    )
    dept_summary['Avg match score'] = dept_summary['Avg match score'].map(
        lambda x: f"{x:.0%}"                # FIX: applymap → map (pandas 3.x)
        .sort_values(['count', 'mean'], ascending=False)
)

    # FIX 2: use_container_width deprecated → use_container_width removed
    st.dataframe(dept_summary)
    total_time  = 0
    total_works = 0

    for dept_name in dept_summary.index:
        group    = roadmap_df[roadmap_df['department'] == dept_name].sort_values('predicted_score', ascending=False)
        n_works  = len(group)
        est_mins = DEPT_TIME.get(dept_name, DEPT_TIME['default'])
        total_time  += est_mins
        total_works += n_works

        with st.expander(f"🏛️  {dept_name}  ·  {n_works} works  ·  ~{est_mins} min", expanded=False):
            for _, row in group.iterrows():
                rc1, rc2, rc3 = st.columns([0.6, 3, 0.8])
                is_ms = row['id'] in ms_ids
                has_img = row['image_url'] and str(row['image_url']) not in ['', 'nan']

                with rc1:
                    if has_img:
                        st.image(row['image_url'], width=70)
                    else:
                        st.markdown(
                            '<div style="background:#1C1C1C;border-radius:6px;width:70px;height:70px;'
                            'display:flex;align-items:center;justify-content:center;">'
                            '<span style="color:#9B8B6E;">🖼️</span></div>',
                            unsafe_allow_html=True
                        )

                with rc2:
                    prefix = "⭐ " if is_ms else ""
                    st.markdown(f"**{prefix}{row['title']}**")
                    st.caption(f"{row['artist']}  ·  {str(row.get('era','')).replace('_',' ').title()}")
                    # Always show at least a brief description
                    desc = str(row.get('description', ''))
                    if desc and desc not in ['nan', ''] and len(desc) > 10:
                        st.caption(desc[:180] + ("..." if len(desc) > 180 else ""))
                    render_flags(row.get('content_flags', ''))

                with rc3:
                    if is_ms:
                        st.markdown('<span class="gold-badge">Must See</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{row['predicted_score']:.0%}**")
                        st.caption("match")
                    if row.get('met_url', '') and str(row['met_url']) not in ['', 'nan']:
                        st.markdown(f"[Met ↗]({row['met_url']})")

    st.markdown("---")
    if st.button("← Rate more artworks to refine your recommendations"):
        st.session_state.phase           = 'rating'
        st.session_state.recommendations = None
        st.rerun()
                st.markdown("---")

    # Tour summary
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1C1C1C,#2D2416);'
        f'border-radius:12px;padding:1.5rem 2rem;color:white;margin:1rem 0;">'
        f'<div style="font-family:Cormorant Garamond,serif;font-size:1.4rem;'
        f'color:#F5E6C8;margin-bottom:0.5rem;">Your Complete Tour</div>'
        f'<div style="color:#C8B89A;font-size:0.9rem;">'
        f'⏱️ &nbsp;{total_time} minutes total &nbsp;·&nbsp; '
        f'🖼️ &nbsp;{total_works} artworks &nbsp;·&nbsp; '
        f'🏛️ &nbsp;{len(dept_summary)} galleries'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("← Refine my recommendations", use_container_width=True):
            st.session_state.phase = 'rating'
            st.session_state.recs  = None
            st.rerun()
    with col_b2:
        if st.button("🔄  Start a completely new tour", use_container_width=True):
            reset_session()
