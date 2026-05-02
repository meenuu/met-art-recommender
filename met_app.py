"""
The Met · Personal Tour — Premium Dark Edition
================================================
Dark editorial design. All bugs fixed.
IndexError fix: iconic artworks excluded from rating queue,
ratings only map to CSV artworks that exist in feature_matrix.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import hashlib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="The Met · Personal Tour",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ══════════════════════════════════
   DARK THEME — VIVID & PROFESSIONAL
══════════════════════════════════ */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"],
.main, .block-container, [class*="css"] {
    background-color: #0F0F13 !important;
    color: #D4CCE0 !important;
    font-family: 'Inter', sans-serif !important;
}
p, span, label, li     { color: #D4CCE0 !important; }
h1,h2,h3,h4,h5,h6      { color: #EDE8F5 !important; }
.stCaption p,
[data-testid="stCaptionContainer"] p {
    color: #6A6278 !important;
    font-size: 0.8rem !important;
}
hr { border-color: #23202E !important; margin: 1.5rem 0 !important; }
.block-container { padding: 2.5rem 3.5rem !important; max-width: 1400px !important; }

/* ── HIDE STREAMLIT UI ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #8B7FD4 !important;
    color: #A89EE8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.74rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    border-radius: 3px !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #8B7FD4 !important;
    color: #0F0F13 !important;
    border-color: #8B7FD4 !important;
}

/* ── EXPANDERS ── */
[data-testid="stExpander"] {
    background: #16141E !important;
    border: 1px solid #23202E !important;
    border-radius: 6px !important;
    margin-bottom: 6px !important;
}
[data-testid="stExpander"] summary {
    color: #C0B8D8 !important;
    font-size: 0.86rem !important;
    font-weight: 400 !important;
}
[data-testid="stExpander"] summary:hover { color: #A89EE8 !important; }

/* ── PROGRESS BAR ── */
[data-testid="stProgress"] > div { background: #1E1C28 !important; border-radius: 4px !important; }
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7B6FD0, #A89EE8) !important;
    border-radius: 4px !important;
}

/* ── SELECT / SLIDER ── */
[data-testid="stSelectbox"] > div > div {
    background: #16141E !important;
    border-color: #2E2A3E !important;
    color: #D4CCE0 !important;
}
[data-testid="stSlider"] p { color: #6A6278 !important; }

/* ── CHECKBOXES ── */
[data-testid="stCheckbox"] label p { color: #6A6278 !important; font-size: 0.82rem !important; }

/* ── INFO/WARNING BOXES ── */
[data-testid="stInfo"] {
    background: #100F1A !important;
    border: 1px solid #2E2050 !important;
    border-radius: 6px !important;
}
[data-testid="stInfo"] p { color: #9B8FD8 !important; font-size: 0.82rem !important; }

/* ══════════════════════════════════
   TYPOGRAPHY
══════════════════════════════════ */
.eyebrow {
    font-size: 0.61rem;
    font-weight: 600;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #8B7FD4 !important;
    display: block;
    margin-bottom: 0.6rem;
}
.display-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    font-weight: 300;
    line-height: 1.06;
    color: #EDE8F5 !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}
.display-title-sm {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 300;
    line-height: 1.1;
    color: #EDE8F5 !important;
    margin-bottom: 0.25rem;
}
.body-muted {
    font-size: 0.84rem;
    font-weight: 300;
    color: #6A6278 !important;
    line-height: 1.7;
}

/* ── SECTION RULE ── */
.section-rule {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2rem 0 1.25rem 0;
}
.section-rule-text {
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #8B7FD4 !important;
    white-space: nowrap;
}
.section-rule-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2E2A3E 0%, transparent 100%);
}

/* ══════════════════════════════════
   MUST-SEE CARDS (text-only)
══════════════════════════════════ */
.ms-card {
    background: #13111C;
    border: 1px solid #23202E;
    border-top: 2px solid #7B6FD0;
    border-radius: 6px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
    min-height: 230px;
    transition: border-color 0.2s;
}
.ms-card:hover { border-color: #3E3860; }
.ms-num {
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #3A3460 !important;
    margin-bottom: 0.75rem;
}
.ms-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #EDE8F5 !important;
    line-height: 1.3;
    margin-bottom: 0.2rem;
}
.ms-artist {
    font-size: 0.76rem;
    font-weight: 500;
    color: #8B7FD4 !important;
    letter-spacing: 0.04em;
    margin-bottom: 0.75rem;
}
.ms-desc {
    font-size: 0.79rem;
    color: #5A5268 !important;
    line-height: 1.65;
    margin-bottom: 0.75rem;
}
.ms-footer {
    font-size: 0.7rem;
    color: #3A3460 !important;
    border-top: 1px solid #1E1C28;
    padding-top: 0.55rem;
}
.ms-footer a {
    color: #7B6FD0 !important;
    text-decoration: none;
    transition: color 0.15s;
}
.ms-footer a:hover { color: #A89EE8 !important; }

/* ══════════════════════════════════
   RATING PHASE
══════════════════════════════════ */
.rating-counter {
    font-family: 'Cormorant Garamond', serif;
    font-size: 5rem;
    font-weight: 300;
    color: #1E1C28 !important;
    line-height: 1;
}
.rating-counter-frac {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    color: #28243A !important;
}
.rating-q {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-style: italic;
    font-weight: 300;
    color: #6A6278 !important;
    margin: 0.5rem 0 1.25rem 0;
}

/* ── ARTWORK INFO ── */
.art-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #EDE8F5 !important;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}
.art-artist {
    font-size: 0.82rem;
    color: #8B7FD4 !important;
    letter-spacing: 0.05em;
    margin-bottom: 1.1rem;
}

/* ── DESCRIPTION BOX ── */
.desc-box {
    background: #0D0C16;
    border-left: 2px solid #5A50A8;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.82rem;
    color: #6A6278 !important;
    line-height: 1.75;
    border-radius: 0 4px 4px 0;
}

/* ── BADGES ── */
.badge-gold {
    display: inline-block;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 3px 9px;
    border: 1px solid #7B6FD0;
    border-radius: 2px;
    color: #A89EE8 !important;
    margin-right: 6px;
}
.badge-dim {
    display: inline-block;
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 9px;
    border: 1px solid #2E2A3E;
    border-radius: 2px;
    color: #4A4460 !important;
    margin-right: 6px;
}

/* ── SCORE ── */
.score-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 300;
    color: #A89EE8 !important;
    line-height: 1;
}
.score-lbl {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3A3460 !important;
}

/* ── WHY BOX ── */
.why-box {
    background: #0E0D18;
    border-left: 2px solid #7B6FD0;
    padding: 0.7rem 1rem;
    font-size: 0.79rem;
    color: #8B80C0 !important;
    line-height: 1.65;
    margin-top: 0.75rem;
    border-radius: 0 3px 3px 0;
}

/* ── TOUR STATS ── */
.stat-block {
    text-align: center;
    padding: 1rem 2rem;
}
.stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 300;
    color: #A89EE8 !important;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3A3460 !important;
    margin-top: 0.25rem;
}

/* ── TASTE CHIPS ── */
.taste-chip {
    display: inline-block;
    background: #13111C;
    border: 1px solid #23202E;
    padding: 4px 12px;
    font-size: 0.74rem;
    color: #6A6278 !important;
    border-radius: 3px;
    margin: 2px 4px 2px 0;
}

/* ── CONTENT FLAG ── */
.cflag {
    display: inline-block;
    padding: 2px 8px;
    font-size: 0.65rem;
    font-weight: 500;
    background: #180F1A;
    border: 1px solid #4A1A3A;
    color: #C06080 !important;
    border-radius: 2px;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
RATING_TARGET = 20
TOP_N_RECS    = 40
MUST_SEE_N    = 12

FAMOUS_ARTIST_NAMES = [
    "van gogh", "rembrandt", "hokusai", "degas", "eakins",
    "whistler", "hiroshige", "pissarro", "seurat", "goya",
    "vermeer", "botticelli", "el greco", "sargent", "rubens",
    "velázquez", "velazquez", "homer", "constable", "cézanne",
    "cezanne", "raphael", "delacroix", "caravaggio", "corot",
    "bruegel", "leutze", "pollock", "david",
    "monet", "picasso", "renoir", "manet", "matisse",
    "rothko", "warhol", "lichtenstein", "cassatt",
    "o'keeffe", "hopper", "chagall", "dali", "dalí",
    "kandinsky", "klee", "titian", "gauguin", "turner",
]

# Hardcoded iconic artworks — shown in must-sees, NOT in rating queue
# IDs intentionally prefixed with "iconic_" to ensure they never match CSV rows
ICONIC_MET_ARTWORKS = [
    {
        "id": "iconic_11417",
        "title": "Washington Crossing the Delaware",
        "artist": "Emanuel Leutze", "year": "1851",
        "department": "The American Wing",
        "description": "One of the most famous images in American history. This enormous canvas (12 × 21 ft) depicts Washington's daring crossing of the icy Delaware River on December 25–26, 1776 — a pivotal moment in the Revolution. Gallery 760.",
        "met_url": "https://www.metmuseum.org/art/collection/search/11417",
        "content_flags": "", "is_famous": True, "era": "nineteenth_century",
        "style": "oil_painting", "culture": "American", "medium": "Oil on canvas",
    },
    {
        "id": "iconic_12127",
        "title": "Madame X (Madame Pierre Gautreau)",
        "artist": "John Singer Sargent", "year": "1883–84",
        "department": "The American Wing",
        "description": "The most scandalous portrait of its era. Sargent's daring portrayal caused a sensation at the 1884 Paris Salon. The contrast of pale skin against the black gown became a defining image of 19th-century portraiture. Gallery 771.",
        "met_url": "https://www.metmuseum.org/art/collection/search/12127",
        "content_flags": "", "is_famous": True, "era": "nineteenth_century",
        "style": "oil_painting", "culture": "American", "medium": "Oil on canvas",
    },
    {
        "id": "iconic_436532",
        "title": "Self-Portrait with a Straw Hat",
        "artist": "Vincent van Gogh", "year": "1887",
        "department": "European Paintings",
        "description": "Painted during Van Gogh's Paris years, this portrait shows his rapid evolution under Impressionist influence. Vivid brushwork and dazzling colour contrasts — blues, oranges, yellows — mark his break from the sombre Dutch palette. Gallery 825.",
        "met_url": "https://www.metmuseum.org/art/collection/search/436532",
        "content_flags": "", "is_famous": True, "era": "nineteenth_century",
        "style": "oil_painting", "culture": "Dutch", "medium": "Oil on canvas",
    },
    {
        "id": "iconic_437394",
        "title": "Aristotle with a Bust of Homer",
        "artist": "Rembrandt van Rijn", "year": "1653",
        "department": "European Paintings",
        "description": "A masterpiece of psychological depth and dramatic light. The Met paid $2.3 million for it in 1961 — then the highest price ever paid for a painting. Rembrandt's genius is at its absolute peak here. Gallery 964.",
        "met_url": "https://www.metmuseum.org/art/collection/search/437394",
        "content_flags": "", "is_famous": True, "era": "baroque_rococo",
        "style": "oil_painting", "culture": "Dutch", "medium": "Oil on canvas",
    },
    {
        "id": "iconic_437870",
        "title": "Young Woman with a Water Pitcher",
        "artist": "Johannes Vermeer", "year": "c. 1662",
        "department": "European Paintings",
        "description": "A serene domestic scene bathed in Vermeer's signature cool northern light. Only 34 Vermeers are known to exist worldwide — making this one of the Met's most precious possessions. Gallery 964.",
        "met_url": "https://www.metmuseum.org/art/collection/search/437870",
        "content_flags": "", "is_famous": True, "era": "baroque_rococo",
        "style": "oil_painting", "culture": "Dutch", "medium": "Oil on canvas",
    },
    {
        "id": "iconic_437130",
        "title": "Bridge over a Pond of Water Lilies",
        "artist": "Claude Monet", "year": "1899",
        "department": "European Paintings",
        "description": "Painted in Monet's garden at Giverny, this canvas captures the Japanese footbridge reflected in the lily pond he designed himself. A cornerstone of Impressionism and a rare Monet at the Met. Gallery 819.",
        "met_url": "https://www.metmuseum.org/art/collection/search/437130",
        "content_flags": "", "is_famous": True, "era": "nineteenth_century",
        "style": "oil_painting", "culture": "French", "medium": "Oil on canvas",
    },
    {
        "id": "iconic_436105",
        "title": "The Death of Socrates",
        "artist": "Jacques-Louis David", "year": "1787",
        "department": "European Paintings",
        "description": "Socrates calmly accepts death, reaching for the hemlock cup while followers grieve. The crisp neoclassical style made this the defining image of Enlightenment idealism. Gallery 614.",
        "met_url": "https://www.metmuseum.org/art/collection/search/436105",
        "content_flags": "", "is_famous": True, "era": "baroque_rococo",
        "style": "oil_painting", "culture": "French", "medium": "Oil on canvas",
    },
    {
        "id": "iconic_488978",
        "title": "Autumn Rhythm (Number 30)",
        "artist": "Jackson Pollock", "year": "1950",
        "department": "Modern and Contemporary Art",
        "description": "Pollock created this by dripping paint onto canvas laid on the floor — his revolutionary drip technique. The gestural sweep of black, white, and brown conveys raw, untamed energy. One of the greatest Abstract Expressionist works. Gallery 919.",
        "met_url": "https://www.metmuseum.org/art/collection/search/488978",
        "content_flags": "", "is_famous": True, "era": "early_modern",
        "style": "oil_painting", "culture": "American", "medium": "Enamel on canvas",
    },
    {
        "id": "iconic_435809",
        "title": "The Harvesters",
        "artist": "Pieter Bruegel the Elder", "year": "1565",
        "department": "European Paintings",
        "description": "Part of a series depicting the months of the year — this August scene shows peasants harvesting wheat under a blazing summer sky. One of the greatest landscape paintings ever made. Gallery 636.",
        "met_url": "https://www.metmuseum.org/art/collection/search/435809",
        "content_flags": "", "is_famous": True, "era": "renaissance",
        "style": "oil_painting", "culture": "Netherlandish", "medium": "Oil on wood",
    },
    {
        "id": "iconic_547802",
        "title": "The Little Fourteen-Year-Old Dancer",
        "artist": "Edgar Degas", "year": "1922 (cast)",
        "department": "European Sculpture and Decorative Arts",
        "description": "The only sculpture Degas ever exhibited publicly — originally shown with real fabric: tutu, hair ribbon, satin shoes. Critics were shocked by its realism. Today's bronze casts preserve his radical vision. Gallery 800.",
        "met_url": "https://www.metmuseum.org/art/collection/search/547802",
        "content_flags": "", "is_famous": True, "era": "nineteenth_century",
        "style": "sculpture", "culture": "French", "medium": "Bronze with fabric tutu",
    },
    {
        "id": "iconic_544039",
        "title": "Sphinx of Hatshepsut",
        "artist": "Ancient Egyptian", "year": "c. 1479–1458 BCE",
        "department": "Egyptian Art",
        "description": "This granite sphinx bears the face of Hatshepsut — one of ancient Egypt's most powerful female pharaohs. The lion body symbolises royal power; the human face, wisdom. One of the finest Egyptian sculptures at the Met. Gallery 115.",
        "met_url": "https://www.metmuseum.org/art/collection/search/544039",
        "content_flags": "", "is_famous": True, "era": "ancient",
        "style": "sculpture", "culture": "Egyptian", "medium": "Granite",
    },
    {
        "id": "iconic_317385",
        "title": "The Temple of Dendur",
        "artist": "Ancient Egyptian", "year": "c. 15 BCE",
        "department": "Egyptian Art",
        "description": "An entire ancient Egyptian temple gifted to the US by Egypt in 1965 — reassembled stone by stone inside the Met. It stands in a vast sun-lit gallery with a reflecting pool. One of the most awe-inspiring rooms in any museum in the world. Gallery 131.",
        "met_url": "https://www.metmuseum.org/art/collection/search/317385",
        "content_flags": "", "is_famous": True, "era": "ancient",
        "style": "sculpture", "culture": "Egyptian", "medium": "Aeolian sandstone",
    },
]

# All iconic IDs — used to exclude from rating queue
ICONIC_IDS = {art["id"] for art in ICONIC_MET_ARTWORKS}

MET_COLLECTION_NOTE = (
    "Van Gogh's *Starry Night* and Monet's *Water Lilies* murals are at MoMA — "
    "but the Met holds Van Gogh's Self-Portrait and Monet's Bridge, both included above. "
    "The Met's true strengths: Rembrandt, Vermeer, Hokusai, Degas, Egyptian antiquities, and American masters."
)

DEPT_TIME = {
    "European Paintings": 45, "American Paintings and Sculpture": 35,
    "Modern and Contemporary Art": 40, "Asian Art": 30,
    "Egyptian Art": 25, "Greek and Roman Art": 30,
    "Islamic Art": 25, "The American Wing": 35,
    "Robert Lehman Collection": 20, "default": 15,
}

CONTENT_FLAGS_DEF = {
    "nudity":    "🔞 Nudity",
    "violence":  "⚔️ Violence",
    "religious": "✝️ Religious",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('met_artworks_clean.csv')
    renames = {
        'objectID': 'id', 'artistDisplayName': 'artist',
        'primaryImageSmall': 'image_url', 'objectURL': 'met_url',
        'isHighlight': 'is_highlight', 'is_famous_artist': 'is_famous',
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

    if 'is_famous' not in df.columns:
        df['is_famous'] = df['artist'].apply(
            lambda a: any(n in str(a).lower() for n in FAMOUS_ARTIST_NAMES)
        )
    else:
        df['is_famous'] = df['is_famous'].fillna(False).astype(bool)
        df['is_famous'] = df['is_famous'] | df['artist'].apply(
            lambda a: any(n in str(a).lower() for n in FAMOUS_ARTIST_NAMES)
        )

    return df.reset_index(drop=True)


@st.cache_resource
def load_features():
    with open('feature_matrix.pkl', 'rb') as f:
        return pickle.load(f)



try:
    df             = load_data()
    feature_matrix = load_features()
    # ── CRITICAL FIX: sync df rows to feature_matrix size ─────────────────
    # The CSV (2788 rows) was regenerated after the feature_matrix (2022 rows)
    # was built. Any df index >= 2022 causes IndexError on feature_matrix lookup.
    # Solution: truncate df to exactly the rows the matrix was built from.
    n_matrix = feature_matrix.shape[0]
    if len(df) > n_matrix:
        df = df.iloc[:n_matrix].reset_index(drop=True)
    DATA_LOADED    = True
except FileNotFoundError:
    DATA_LOADED    = False



# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
if 'session_id' not in st.session_state:
    st.session_state.session_id   = str(time.time_ns())
    st.session_state.ratings      = {}
    st.session_state.phase        = 'must_sees'
    st.session_state.rating_queue = []
    st.session_state.recs         = None
    st.session_state.must_sees_df = None
    st.session_state.hide_nudity  = False
    st.session_state.hide_violence= False


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def get_cf():
    f = []
    if st.session_state.get('hide_nudity'):    f.append('nudity')
    if st.session_state.get('hide_violence'):  f.append('violence')
    return f


def apply_filter(dataframe):
    cf = get_cf()
    if not cf or 'content_flags' not in dataframe.columns:
        return dataframe
    return dataframe[dataframe['content_flags'].apply(
        lambda x: not any(f in str(x) for f in cf)
    )]


def render_flags(flags_str):
    if not flags_str or str(flags_str) in ['', 'nan']:
        return
    for flag in str(flags_str).split('|'):
        flag = flag.strip()
        if flag in CONTENT_FLAGS_DEF:
            st.markdown(
                f'<span class="cflag">{CONTENT_FLAGS_DEF[flag]}</span>',
                unsafe_allow_html=True
            )


def build_description(row):
    """Build rich description from best available fields."""
    desc = str(row.get('description', ''))
    if len(desc) > 30 and desc not in ['nan', '']:
        return desc[:400]
    parts = []
    if str(row.get('medium', '')) not in ['', 'nan', 'unknown']:
        parts.append(f"Medium: {row['medium']}")
    if str(row.get('culture', '')) not in ['', 'nan', 'unknown']:
        parts.append(f"Culture: {row['culture']}")
    if str(row.get('era', '')) not in ['', 'nan', 'Unknown Era', 'unknown']:
        parts.append(f"Era: {str(row['era']).replace('_', ' ').title()}")
    if str(row.get('tags', '')) not in ['', 'nan']:
        parts.append(f"Tags: {str(row['tags'])[:80]}")
    return "  ·  ".join(parts) if parts else f"Part of the Met's {row.get('department','collection')}."


def render_desc_box(row):
    st.markdown(
        f'<div class="desc-box">{build_description(row)}</div>',
        unsafe_allow_html=True
    )


def section_rule(label):
    st.markdown(
        f'<div class="section-rule">'
        f'<span class="section-rule-text">{label}</span>'
        f'<div class="section-rule-line"></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def get_must_sees(exclude_ids=None):
    exclude_ids = set(str(i) for i in (exclude_ids or []))
    cf          = get_cf()

    iconic_rows = []
    for art in ICONIC_MET_ARTWORKS:
        if art['id'] in exclude_ids:
            continue
        if cf and any(f in str(art.get('content_flags', '')) for f in cf):
            continue
        iconic_rows.append(art)

    iconic_df = pd.DataFrame(iconic_rows) if iconic_rows else pd.DataFrame()

    # Fill remaining from CSV famous artists
    already  = set(iconic_df['id'].tolist()) if not iconic_df.empty else set()
    already.update(exclude_ids)
    pool     = apply_filter(df[~df['id'].isin(already)])
    famous   = pool[pool['is_famous'] == True]
    hilights = pool[(pool['is_highlight'] == True) & (~pool['is_famous'])]
    combined = pd.concat([famous, hilights]).drop_duplicates(subset=['id'])

    remaining = max(0, MUST_SEE_N - len(iconic_df))
    csv_rows  = []
    seen_art  = {}

    if not combined.empty and remaining > 0:
        seed = int(hashlib.md5(st.session_state.session_id.encode()).hexdigest()[:8], 16)
        rng  = np.random.default_rng(seed)
        shuf = combined.sample(frac=1, random_state=int(rng.integers(0, 10000)))
        for _, row in shuf.iterrows():
            artist = str(row['artist']).split('(')[0].strip()
            if seen_art.get(artist, 0) >= 1:
                continue
            csv_rows.append(row)
            seen_art[artist] = seen_art.get(artist, 0) + 1
            if len(csv_rows) >= remaining:
                break

    if not iconic_df.empty and csv_rows:
        csv_df   = pd.DataFrame(csv_rows)
        all_cols = list(set(iconic_df.columns.tolist() + csv_df.columns.tolist()))
        for col in all_cols:
            if col not in iconic_df.columns: iconic_df[col] = ''
            if col not in csv_df.columns:    csv_df[col]    = ''
        return pd.concat([iconic_df, csv_df[all_cols]], ignore_index=True)
    elif not iconic_df.empty:
        return iconic_df
    elif csv_rows:
        return pd.DataFrame(csv_rows)
    return pool.head(MUST_SEE_N)


def build_rating_queue(exclude_ids=None):
    """
    Build rating queue from CSV artworks ONLY.
    Iconic artworks are never included — they don't exist in feature_matrix.
    """
    exclude_ids = set(str(i) for i in (exclude_ids or []))
    # Also exclude all iconic IDs (they start with "iconic_")
    exclude_ids.update(ICONIC_IDS)

    filtered = apply_filter(df[~df['id'].isin(exclude_ids)])
    seed     = int(hashlib.md5(st.session_state.session_id.encode()).hexdigest()[:8], 16) % 100000

    famous   = filtered[filtered['is_famous'] == True]
    hilights = filtered[(filtered['is_highlight'] == True) & (~filtered['is_famous'])]
    rest     = filtered[~filtered['id'].isin(
        pd.concat([famous, hilights])['id'].tolist()
        if not famous.empty else hilights['id'].tolist()
    )]

    n_f = min(6, len(famous))
    n_h = min(4, len(hilights))
    n_r = max(0, RATING_TARGET - n_f - n_h)

    parts = []
    if n_f > 0: parts.append(famous.sample(n_f, random_state=seed))
    if n_h > 0: parts.append(hilights.sample(min(n_h, len(hilights)), random_state=seed+1))
    if n_r > 0 and len(rest) > 0: parts.append(rest.sample(min(n_r, len(rest)), random_state=seed+2))

    queue = pd.concat(parts).sample(frac=1, random_state=seed+3) if parts else \
            filtered.sample(min(RATING_TARGET, len(filtered)), random_state=seed)
    return queue['id'].astype(str).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Data not loaded
# ══════════════════════════════════════════════════════════════════════════════
if not DATA_LOADED:
    st.markdown('<div class="display-title">🏛️ The Met<br>Personal Tour</div>', unsafe_allow_html=True)
    st.error("Data files not found. Ensure `met_artworks_clean.csv`, `feature_matrix.pkl`, `tfidf_vectorizer.pkl` are present.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<span class="eyebrow">The Met · Personal Tour</span>', unsafe_allow_html=True)
    st.markdown("---")
    n = len(st.session_state.ratings)
    st.markdown(f'<span class="eyebrow">Progress · {n} / {RATING_TARGET}</span>', unsafe_allow_html=True)
    st.progress(min(n / RATING_TARGET, 1.0))
    if n > 0:
        loves = sum(1 for v in st.session_state.ratings.values() if v == 2)
        likes = sum(1 for v in st.session_state.ratings.values() if v == 1)
        skips = sum(1 for v in st.session_state.ratings.values() if v == 0)
        st.markdown(f'<span class="body-muted">❤️ {loves} &nbsp;·&nbsp; 👍 {likes} &nbsp;·&nbsp; ⏭ {skips}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<span class="eyebrow">Content filters</span>', unsafe_allow_html=True)
    st.session_state.hide_nudity   = st.checkbox("Exclude nudity", value=st.session_state.get('hide_nudity', False))
    st.session_state.hide_violence = st.checkbox("Exclude violence", value=st.session_state.get('hide_violence', False))
    st.markdown("---")
    if st.button("↺  Start Over"):
        reset_session()
    st.markdown("---")
    st.markdown('<span class="body-muted">Rate 20 artworks. Get a personalised tour.</span>', unsafe_allow_html=True)
    st.markdown('<span class="body-muted">Data: Met Museum Open Access API</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Must-Sees
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 'must_sees':

    # Header
    st.markdown('<span class="eyebrow">The Metropolitan Museum of Art · New York</span>', unsafe_allow_html=True)
    st.markdown('<div class="display-title">Your Personal<br>Museum Tour</div>', unsafe_allow_html=True)
    st.markdown('<div class="body-muted">An AI-powered guide built around your taste — curated in real time.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Build must-sees
    if st.session_state.must_sees_df is None:
        st.session_state.must_sees_df = get_must_sees()
    must_sees = st.session_state.must_sees_df

    section_rule("Non-Negotiables · Always Included")
    st.markdown(
        '<div class="body-muted" style="margin-bottom:1rem;">'
        'These iconic works are guaranteed in your tour regardless of your ratings. '
        'We\'ve intentionally withheld images — experience them for the first time in person.</div>',
        unsafe_allow_html=True
    )
    st.info(f"ℹ️ {MET_COLLECTION_NOTE}")
    st.markdown("")

    # 3-column text-only cards
    if not must_sees.empty:
        cols = st.columns(3)
        for i, (_, row) in enumerate(must_sees.iterrows()):
            with cols[i % 3]:
                year = str(row.get('year', row.get('objectDate', ''))).strip()
                yr   = f" · {year}" if year and year not in ['nan', ''] else ""
                dept = str(row.get('department', ''))
                desc = build_description(row)[:260]
                url  = str(row.get('met_url', ''))
                link = f"<a href='{url}' target='_blank'>View on Met ↗</a>" if url not in ['','nan'] else ""
                num  = str(i + 1).zfill(2)

                st.markdown(f"""
<div class="ms-card">
  <div class="ms-num">— {num}</div>
  <div class="ms-title">{row['title']}</div>
  <div class="ms-artist">{row['artist']}{yr}</div>
  <div class="ms-desc">{desc}</div>
  <div class="ms-footer">🏛 {dept}&nbsp;&nbsp;{link}</div>
</div>""", unsafe_allow_html=True)
                render_flags(row.get('content_flags', ''))

    st.markdown("---")

    section_rule("Step 2 of 2 · Personalise Your Tour")
    st.markdown(
        '<div class="body-muted">Rate 20 artworks so we can recommend '
        'everything else you\'ll love at the Met.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("Begin Taste Profile  →", use_container_width=True):
            st.session_state.phase        = 'rating'
            ms_ids = must_sees['id'].tolist() if not must_sees.empty else []
            st.session_state.rating_queue = build_rating_queue(exclude_ids=ms_ids)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Rating
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 'rating':

    n_rated = len(st.session_state.ratings)

    if n_rated >= RATING_TARGET:
        st.session_state.phase = 'results'
        st.rerun()

    # Progress header
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown('<span class="eyebrow">Building Your Taste Profile</span>', unsafe_allow_html=True)
        st.markdown('<div class="display-title-sm">Rate This Artwork</div>', unsafe_allow_html=True)
        st.progress(n_rated / RATING_TARGET)
        st.markdown(f'<span class="body-muted">{n_rated} of {RATING_TARGET} rated</span>', unsafe_allow_html=True)
    with hcol2:
        st.markdown(
            f'<div class="rating-counter">{n_rated + 1}'
            f'<span class="rating-counter-frac">/{RATING_TARGET}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Get next unrated artwork from queue
    rated_ids = set(st.session_state.ratings.keys())
    remaining = [i for i in st.session_state.rating_queue if i not in rated_ids]

    if not remaining:
        st.session_state.phase = 'results'
        st.rerun()

    current_id = remaining[0]
    matches    = df[df['id'] == current_id]

    if matches.empty:
        # Skip any IDs not found in df (safety)
        st.session_state.ratings[current_id] = -1
        st.rerun()

    artwork   = matches.iloc[0]
    has_image = artwork['image_url'] and str(artwork['image_url']) not in ['', 'nan']

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        if has_image:
            st.image(artwork['image_url'], use_column_width=True)
        else:
            st.markdown(
                '<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:4px;'
                'min-height:400px;display:flex;align-items:center;justify-content:center;">'
                '<div style="text-align:center;">'
                '<div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.3;">🖼</div>'
                '<div style="font-size:0.72rem;letter-spacing:0.15em;text-transform:uppercase;'
                'color:#2A2520;">Image restricted</div></div></div>',
                unsafe_allow_html=True
            )

    with col2:
        # Badges
        badges = f'<span class="badge-dim">{artwork["department"]}</span>'
        if artwork.get('is_famous'):
            badges += '<span class="badge-gold">Master</span>'
        if artwork.get('is_highlight'):
            badges += '<span class="badge-gold">Met Pick</span>'
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown("")

        st.markdown(f'<div class="art-title">{artwork["title"]}</div>', unsafe_allow_html=True)

        artist_line = str(artwork['artist'])
        if str(artwork.get('artistNationality', '')) not in ['', 'nan']:
            artist_line += f"  ·  {artwork['artistNationality']}"
        if str(artwork.get('objectDate', '')) not in ['', 'nan']:
            artist_line += f"  ·  {artwork['objectDate']}"
        st.markdown(f'<div class="art-artist">{artist_line}</div>', unsafe_allow_html=True)

        render_desc_box(artwork)
        render_flags(artwork.get('content_flags', ''))

        if artwork.get('met_url', '') and str(artwork['met_url']) not in ['', 'nan']:
            st.markdown(f'<a href="{artwork["met_url"]}" target="_blank" style="color:#BFA14A;font-size:0.75rem;letter-spacing:0.05em;">View full record on Met website ↗</a>', unsafe_allow_html=True)

        st.markdown('<div class="rating-q">Would you stop and look at this at the Met?</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            if st.button("❤  Love it", key="love"):
                st.session_state.ratings[current_id] = 2
                st.rerun()
        with r2:
            if st.button("👍  Like it", key="like"):
                st.session_state.ratings[current_id] = 1
                st.rerun()
        with r3:
            if st.button("⏭  Skip", key="skip"):
                st.session_state.ratings[current_id] = 0
                st.rerun()

        left = RATING_TARGET - n_rated - 1
        st.markdown("")
        if left > 0:
            st.markdown(f'<span class="body-muted">{left} more to go</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#BFA14A;font-size:0.82rem;">Last one — your tour is almost ready.</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Results
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 'results':

    # ── Train model ─────────────────────────────────────────────────────────
    if st.session_state.recs is None:
        with st.spinner("Analysing your taste and curating your tour..."):

            # Only use ratings for artworks that exist in the CSV + feature_matrix
            # This is the IndexError fix — filter out any iconic_ IDs
            valid_ratings = {
                k: v for k, v in st.session_state.ratings.items()
                if v >= 0 and not str(k).startswith('iconic_')
            }
            rated_ids = list(valid_ratings.keys())
            labels    = list(valid_ratings.values())

            if len(set(labels)) < 2:
                st.warning("Try mixing Love, Like, and Skip for better results. Showing popular works for now.")
                fallback = apply_filter(df[~df['id'].isin(rated_ids)]).head(TOP_N_RECS).copy()
                fallback['predicted_score'] = 0.5
                st.session_state.recs = fallback
            else:
                # Map rated IDs to feature_matrix row indices
                rated_indices, valid_ids, valid_labels = [], [], []
                for oid, label in zip(rated_ids, labels):
                    match = df[df['id'] == oid]
                    if not match.empty:
                        rated_indices.append(match.index[0])
                        valid_ids.append(oid)
                        valid_labels.append(label)

                if len(set(valid_labels)) < 2:
                    st.warning("Not enough variety in ratings. Showing popular works.")
                    fallback = apply_filter(df[~df['id'].isin(valid_ids)]).head(TOP_N_RECS).copy()
                    fallback['predicted_score'] = 0.5
                    st.session_state.recs = fallback
                else:
                    X_train = feature_matrix[rated_indices]
                    y_train = np.array(valid_labels)

                    clf = RandomForestClassifier(
                        n_estimators=300, random_state=42,
                        class_weight='balanced', n_jobs=-1
                    )
                    clf.fit(X_train, y_train)

                    unrated_df  = df[~df['id'].isin(valid_ids)].copy()
                    unrated_idx = unrated_df.index.tolist()
                    X_unrated   = feature_matrix[unrated_idx]

                    proba   = clf.predict_proba(X_unrated)
                    classes = clf.classes_.tolist()
                    scores  = (
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

        st.session_state.must_sees_df = get_must_sees(
            exclude_ids=list(st.session_state.ratings.keys())
        )
        st.rerun()

    recs      = st.session_state.recs
    must_sees = st.session_state.must_sees_df

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown('<span class="eyebrow">The Metropolitan Museum of Art · New York</span>', unsafe_allow_html=True)
    st.markdown('<div class="display-title">Your Personal Tour</div>', unsafe_allow_html=True)

    # Taste chips
    liked_ids   = [i for i, v in st.session_state.ratings.items() if v >= 1]
    liked_df    = df[df['id'].isin(liked_ids)]
    liked_depts = liked_df['department'].tolist()
    liked_eras  = liked_df['era'].tolist()

    if not liked_df.empty:
        chips = ""
        for dept, cnt in liked_df['department'].value_counts().head(3).items():
            chips += f'<span class="taste-chip">🏛 {dept} ({cnt})</span>'
        if 'style' in liked_df.columns:
            for s, cnt in liked_df['style'].value_counts().head(2).items():
                if s and s not in ['other', 'unknown', '']:
                    chips += f'<span class="taste-chip">🖌 {str(s).replace("_"," ").title()}</span>'
        for era, cnt in liked_df['era'].value_counts().head(2).items():
            if era and era not in ['unknown', 'Unknown Era']:
                chips += f'<span class="taste-chip">🕰 {str(era).replace("_"," ").title()}</span>'
        if chips:
            st.markdown(f'<div style="margin:0.5rem 0 1.5rem 0;">{chips}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── MUST-SEES ────────────────────────────────────────────────────────────
    section_rule("Non-Negotiables · Must-See Masterpieces")
    st.markdown('<div class="body-muted" style="margin-bottom:1rem;">Iconic works always included in your tour. No images — see them in person.</div>', unsafe_allow_html=True)

    if must_sees is not None and not must_sees.empty:
        ms_c = st.columns(3)
        for i, (_, row) in enumerate(must_sees.head(12).iterrows()):
            with ms_c[i % 3]:
                year = str(row.get('year', row.get('objectDate', ''))).strip()
                yr   = f" · {year}" if year and year not in ['nan',''] else ""
                desc = build_description(row)[:220]
                url  = str(row.get('met_url',''))
                link = f"<a href='{url}' target='_blank'>View on Met ↗</a>" if url not in ['','nan'] else ""
                num  = str(i+1).zfill(2)
                st.markdown(f"""
<div class="ms-card">
  <div class="ms-num">— {num}</div>
  <div class="ms-title">{row['title']}</div>
  <div class="ms-artist">{row['artist']}{yr}</div>
  <div class="ms-desc">{desc}</div>
  <div class="ms-footer">🏛 {row['department']}&nbsp;&nbsp;{link}</div>
</div>""", unsafe_allow_html=True)
                render_flags(row.get('content_flags',''))

    st.markdown("---")

    # ── PERSONALISED RECOMMENDATIONS ─────────────────────────────────────────
    section_rule("Personalised · Based on Your Taste")

    fcol1, fcol2 = st.columns([2, 1])
    with fcol1:
        all_depts   = ['All departments'] + sorted(recs['department'].unique().tolist())
        chosen_dept = st.selectbox("Department", all_depts, label_visibility="collapsed")
    with fcol2:
        min_score = st.slider("Min match", 0, 100, 0, 5, format="%d%%", label_visibility="collapsed")

    disp = recs.copy()
    if chosen_dept != 'All departments':
        disp = disp[disp['department'] == chosen_dept]
    if min_score > 0:
        disp = disp[disp['predicted_score'] >= min_score / 100]
    disp = disp.head(TOP_N_RECS)

    st.markdown(f'<span class="body-muted">{len(disp)} recommendations</span>', unsafe_allow_html=True)
    st.markdown("")

    for i, (_, row) in enumerate(disp.iterrows()):
        score     = row['predicted_score']
        fire      = "🔥 " if score > 0.75 else ""
        has_image = row['image_url'] and str(row['image_url']) not in ['', 'nan']

        with st.expander(
            f"{fire}{row['title']}  ·  {row['artist']}  ·  {row['department']}  ·  {score:.0%}",
            expanded=(i < 2)
        ):
            ec1, ec2 = st.columns([1, 2], gap="large")

            with ec1:
                if has_image:
                    st.image(row['image_url'], use_column_width=True)
                else:
                    st.markdown(
                        '<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:4px;'
                        'min-height:250px;display:flex;align-items:center;justify-content:center;">'
                        '<span style="color:#2A2520;font-size:2rem;">🖼</span></div>',
                        unsafe_allow_html=True
                    )

            with ec2:
                # Badges
                bdg = f'<span class="badge-dim">{row["department"]}</span>'
                if row.get('is_famous'): bdg += '<span class="badge-gold">Master</span>'
                st.markdown(bdg, unsafe_allow_html=True)
                st.markdown("")

                st.markdown(f'<div class="art-title" style="font-size:1.5rem;">{row["title"]}</div>', unsafe_allow_html=True)
                if str(row['artist']) not in ['Unknown Artist','nan','']:
                    st.markdown(f'<div class="art-artist">{row["artist"]}</div>', unsafe_allow_html=True)

                # Score
                st.markdown(
                    f'<div style="margin:0.75rem 0;">'
                    f'<div class="score-num">{score:.0%}</div>'
                    f'<div class="score-lbl">Match Score</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                render_desc_box(row)
                render_flags(row.get('content_flags',''))

                # Why recommended
                reasons = []
                if row['department'] in liked_depts:
                    reasons.append(f"you enjoyed other works from {row['department']}")
                if str(row.get('era','')) not in ['Unknown Era','nan',''] and row.get('era') in liked_eras:
                    reasons.append(f"matches your interest in {str(row['era']).replace('_',' ')} art")
                if row.get('is_famous'): reasons.append("celebrated master artist")
                if score > 0.75:        reasons.append(f"exceptionally high {score:.0%} confidence")

                if reasons:
                    st.markdown(
                        f'<div class="why-box">✦ <strong>Why this?</strong> '
                        f'{" · ".join(reasons)}</div>',
                        unsafe_allow_html=True
                    )

                if row.get('met_url','') and str(row['met_url']) not in ['','nan']:
                    st.markdown(f'<a href="{row["met_url"]}" target="_blank" style="color:#BFA14A;font-size:0.75rem;letter-spacing:0.05em;">View on Met website ↗</a>', unsafe_allow_html=True)

    st.markdown("---")

    # ── GALLERY ROADMAP ───────────────────────────────────────────────────────
    section_rule("Gallery Roadmap · Walking Order")
    st.markdown('<div class="body-muted" style="margin-bottom:1rem;">Your full tour organised by gallery.</div>', unsafe_allow_html=True)

    ms_ids     = must_sees['id'].tolist() if must_sees is not None and not must_sees.empty else []
    all_ids    = ms_ids + [i for i in disp.head(40)['id'].tolist() if i not in ms_ids]

    # Only pull roadmap artworks that exist in the CSV (not iconic_ IDs)
    csv_ids    = [i for i in all_ids if not str(i).startswith('iconic_')]
    roadmap_df = df[df['id'].isin(csv_ids)].copy()

    score_map                     = dict(zip(disp['id'], disp['predicted_score']))
    roadmap_df['predicted_score'] = roadmap_df['id'].map(score_map).fillna(0.9)

    dept_summary = (
        roadmap_df.groupby('department')['predicted_score']
        .agg(['count','mean'])
        .sort_values(['count','mean'], ascending=False)
    )

    total_time  = 0
    total_works = len(all_ids)  # includes iconic

    for dept_name in dept_summary.index:
        group    = roadmap_df[roadmap_df['department'] == dept_name].sort_values('predicted_score', ascending=False)
        n_works  = len(group)
        est_mins = DEPT_TIME.get(dept_name, DEPT_TIME['default'])
        total_time += est_mins

        with st.expander(f"🏛  {dept_name}  ·  {n_works} works  ·  ~{est_mins} min", expanded=False):
            for _, row in group.iterrows():
                rc1, rc2, rc3 = st.columns([0.5, 3, 0.8])
                has_img = row['image_url'] and str(row['image_url']) not in ['','nan']

                with rc1:
                    if has_img:
                        st.image(row['image_url'], width=65)
                    else:
                        st.markdown(
                            '<div style="background:#111;border:1px solid #1E1E1E;border-radius:2px;'
                            'width:65px;height:65px;display:flex;align-items:center;justify-content:center;">'
                            '<span style="color:#2A2520;font-size:1.2rem;">🖼</span></div>',
                            unsafe_allow_html=True
                        )
                with rc2:
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"{row['artist']}  ·  {str(row.get('era','')).replace('_',' ').title()}")
                    desc = str(row.get('description',''))
                    if desc and desc not in ['nan',''] and len(desc) > 10:
                        st.caption(desc[:160] + "...")
                    render_flags(row.get('content_flags',''))
                with rc3:
                    st.markdown(f'<div class="score-num" style="font-size:1.6rem;">{row["predicted_score"]:.0%}</div>', unsafe_allow_html=True)
                    if row.get('met_url','') and str(row['met_url']) not in ['','nan']:
                        st.markdown(f'<a href="{row["met_url"]}" target="_blank" style="color:#BFA14A;font-size:0.72rem;">Met ↗</a>', unsafe_allow_html=True)
                st.markdown("---")

    # Also add iconic artworks to roadmap display (text-only)
    iconic_in_tour = [art for art in ICONIC_MET_ARTWORKS if art['id'] in ms_ids]
    if iconic_in_tour:
        with st.expander(f"⭐  Must-See Masterpieces  ·  {len(iconic_in_tour)} works  ·  Already included", expanded=False):
            for art in iconic_in_tour:
                rc1, rc2, rc3 = st.columns([0.5, 3, 0.8])
                with rc1:
                    st.markdown('<div style="background:#111;border:1px solid #BFA14A;border-radius:2px;width:65px;height:65px;display:flex;align-items:center;justify-content:center;"><span style="color:#BFA14A;font-size:1.2rem;">⭐</span></div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"**{art['title']}**")
                    st.caption(f"{art['artist']}  ·  {art.get('year','')}")
                    st.caption(str(art.get('description',''))[:160] + "...")
                with rc3:
                    st.markdown('<span class="badge-gold">Must See</span>', unsafe_allow_html=True)
                    if art.get('met_url',''):
                        st.markdown(f'<a href="{art["met_url"]}" target="_blank" style="color:#BFA14A;font-size:0.72rem;">Met ↗</a>', unsafe_allow_html=True)
                st.markdown("---")

    # Tour stats
    st.markdown(
        f'<div style="background:#111111;border:1px solid #1E1E1E;border-radius:4px;'
        f'padding:1.5rem 2rem;margin:1.5rem 0;display:flex;gap:3rem;">'
        f'<div class="stat-block"><div class="stat-num">{total_time}</div><div class="stat-lbl">Minutes</div></div>'
        f'<div class="stat-block"><div class="stat-num">{total_works}</div><div class="stat-lbl">Artworks</div></div>'
        f'<div class="stat-block"><div class="stat-num">{len(dept_summary)}</div><div class="stat-lbl">Galleries</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Refine recommendations", use_container_width=True):
            st.session_state.phase = 'rating'
            st.session_state.recs  = None
            st.rerun()
    with c2:
        if st.button("↺  Start a new tour", use_container_width=True):
            reset_session()
