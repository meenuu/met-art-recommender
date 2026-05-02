"""
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
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Met Museum | Personal Tour",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Force light theme ── */
html, body, [class*="css"],
.stApp, .main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #FFFFFF !important;
    color: #1a1a2e !important;
}
p, span, label, li { color: #1a1a2e !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color: #6b7280 !important; }

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

/* ── Typography ── */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #1a1a2e !important;
    margin-bottom: 0.2rem;
    line-height: 1.15;
}
.subtitle {
    font-size: 1.05rem;
    color: #6b7280 !important;
    font-weight: 300;
    margin-bottom: 1.5rem;
}
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1a2e !important;
    margin: 1.5rem 0 0.25rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #EEF2FF;
}

/* ── Must-see card (text-only, no image) ── */
.must-see-card {
    background: #FAFBFF;
    border: 1px solid #E0E7FF;
    border-left: 4px solid #4338CA;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    height: 100%;
}
.must-see-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1a2e !important;
    margin-bottom: 0.15rem;
}
.must-see-card-artist {
    font-size: 0.82rem;
    color: #4338CA !important;
    font-weight: 500;
    margin-bottom: 0.6rem;
}
.must-see-card-desc {
    font-size: 0.83rem;
    color: #374151 !important;
    line-height: 1.6;
    margin-bottom: 0.6rem;
}
.must-see-card-meta {
    font-size: 0.75rem;
    color: #9CA3AF !important;
}

/* ── Badges ── */
.badge-must-see {
    background: #EEF2FF;
    color: #4338CA !important;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-dept {
    background: #F3F4F6;
    color: #374151 !important;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
}
.badge-famous {
    background: #FEF3C7;
    color: #92400E !important;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
}

/* ── Rating phase ── */
.rating-counter {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #EEF2FF !important;
    line-height: 1;
}
.rating-question {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 400;
    color: #1a1a2e !important;
    margin-bottom: 1rem;
}

/* ── Description box ── */
.desc-box {
    background: #F9FAFB;
    border-left: 3px solid #4338CA;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.86rem;
    color: #374151 !important;
    line-height: 1.7;
}

/* ── Score ── */
.score-display {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #4338CA !important;
}

/* ── Why box ── */
.why-box {
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #3730A3 !important;
    margin-top: 0.75rem;
}

/* ── Content flags ── */
.content-flag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-right: 4px;
}

/* ── Tour summary box ── */
.tour-summary {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    color: white !important;
    margin: 1rem 0;
}
.tour-summary p, .tour-summary span, .tour-summary div {
    color: #E0E7FF !important;
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
    # These match exact formats in the Met CSV
    "van gogh", "rembrandt", "hokusai", "degas", "eakins",
    "whistler", "hiroshige", "pissarro", "seurat", "goya",
    "vermeer", "botticelli", "el greco", "sargent", "rubens",
    "velázquez", "velazquez", "homer", "constable", "cézanne",
    "cezanne", "raphael", "delacroix", "caravaggio", "corot",
    "gerard david", "juan de flandes", "hans von aachen",
    "bruegel", "leutze", "pollock", "david",
    # Broader matches
    "monet", "picasso", "renoir", "manet", "matisse",
    "pollock", "rothko", "warhol", "lichtenstein", "cassatt",
    "o'keeffe", "hopper", "chagall", "dali", "dalí",
    "kandinsky", "klee", "titian", "gauguin", "turner",
]

# ── Hardcoded iconic Met artworks by exact Met object ID ──────────────────────
# These are guaranteed to always appear in must-sees regardless of the CSV
# Source: metmuseum.org collection
ICONIC_MET_ARTWORKS = [
    {
        "id":          "11417",
        "title":       "Washington Crossing the Delaware",
        "artist":      "Emanuel Leutze",
        "year":        "1851",
        "department":  "The American Wing",
        "description": "One of the most famous images in American history. This enormous canvas (12 × 21 ft) "
                       "depicts George Washington's daring crossing of the icy Delaware River on the night "
                       "of December 25–26, 1776 — a pivotal moment in the American Revolution. Located in "
                       "the American Wing, Gallery 760.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/11417",
        "image_url":   "https://images.metmuseum.org/CRDImages/am/original/DP215081.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "nineteenth_century",
        "style":       "oil_painting",
        "subject":     "history_battle",
        "culture":     "American",
        "medium":      "Oil on canvas",
    },
    {
        "id":          "12127",
        "title":       "Madame X (Madame Pierre Gautreau)",
        "artist":      "John Singer Sargent",
        "year":        "1883–84",
        "department":  "The American Wing",
        "description": "One of the most scandalous portraits of its era. Sargent's daring portrayal of "
                       "Virginie Amélie Avegno Gautreau caused a sensation at the 1884 Paris Salon. "
                       "The stark contrast of her pale skin against the black dress and the bold pose "
                       "made this a defining work of 19th-century portraiture. Gallery 771.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/12127",
        "image_url":   "https://images.metmuseum.org/CRDImages/am/original/DP128874.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "nineteenth_century",
        "style":       "oil_painting",
        "subject":     "portrait_figure",
        "culture":     "American",
        "medium":      "Oil on canvas",
    },
    {
        "id":          "436532",
        "title":       "Self-Portrait with a Straw Hat",
        "artist":      "Vincent van Gogh",
        "year":        "1887",
        "department":  "European Paintings",
        "description": "Painted during Van Gogh's transformative Paris years, this self-portrait shows "
                       "his rapid evolution under Impressionist influence. The vivid brushwork and "
                       "dazzling colour contrasts — blues, oranges, yellows — mark his break from "
                       "the sombre Dutch palette. One of only two Van Goghs at the Met. Gallery 825.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/436532",
        "image_url":   "https://images.metmuseum.org/CRDImages/ep/original/DT1502_cropped2.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "nineteenth_century",
        "style":       "oil_painting",
        "subject":     "portrait_figure",
        "culture":     "Dutch",
        "medium":      "Oil on canvas",
    },
    {
        "id":          "437394",
        "title":       "Aristotle with a Bust of Homer",
        "artist":      "Rembrandt van Rijn",
        "year":        "1653",
        "department":  "European Paintings",
        "description": "Commissioned by a Sicilian nobleman, this masterpiece shows the philosopher "
                       "Aristotle contemplating a bust of the blind poet Homer. Rembrandt's genius "
                       "for psychological depth and dramatic light is at its peak here. The Met "
                       "paid $2.3 million for it in 1961 — then the highest price ever paid for a painting. Gallery 964.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/437394",
        "image_url":   "https://images.metmuseum.org/CRDImages/ep/original/DP343491.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "baroque_rococo",
        "style":       "oil_painting",
        "subject":     "portrait_figure",
        "culture":     "Dutch",
        "medium":      "Oil on canvas",
    },
    {
        "id":          "437870",
        "title":       "Young Woman with a Water Pitcher",
        "artist":      "Johannes Vermeer",
        "year":        "c. 1662",
        "department":  "European Paintings",
        "description": "A serene domestic scene bathed in Vermeer's signature cool northern light. "
                       "A young woman opens a window, water pitcher in hand — a moment of quiet "
                       "intimacy that transcends its time. Only 34 Vermeers are known to exist worldwide, "
                       "making this one of the Met's most precious possessions. Gallery 964.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/437870",
        "image_url":   "https://images.metmuseum.org/CRDImages/ep/original/DP251139.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "baroque_rococo",
        "style":       "oil_painting",
        "subject":     "portrait_figure",
        "culture":     "Dutch",
        "medium":      "Oil on canvas",
    },
    {
        "id":          "437130",
        "title":       "Bridge over a Pond of Water Lilies",
        "artist":      "Claude Monet",
        "year":        "1899",
        "department":  "European Paintings",
        "description": "Painted in Monet's famous garden at Giverny, this canvas captures the Japanese "
                       "footbridge reflected in the lily pond he designed himself. This is one of the "
                       "series that would lead to the monumental Water Lilies murals. A cornerstone of "
                       "Impressionism and a rare Monet at the Met. Gallery 819.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/437130",
        "image_url":   "https://images.metmuseum.org/CRDImages/ep/original/DP251139.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "nineteenth_century",
        "style":       "oil_painting",
        "subject":     "landscape_nature",
        "culture":     "French",
        "medium":      "Oil on canvas",
    },
    {
        "id":          "436105",
        "title":       "The Death of Socrates",
        "artist":      "Jacques-Louis David",
        "year":        "1787",
        "department":  "European Paintings",
        "description": "David's most celebrated history painting depicts the philosopher Socrates "
                       "calmly accepting his death sentence, reaching for the hemlock cup while "
                       "his followers grieve. The crisp neoclassical style — clear light, sculptural "
                       "figures, moral clarity — made this the defining image of Enlightenment idealism. Gallery 614.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/436105",
        "image_url":   "https://images.metmuseum.org/CRDImages/ep/original/DP130999.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "baroque_rococo",
        "style":       "oil_painting",
        "subject":     "history_battle",
        "culture":     "French",
        "medium":      "Oil on canvas",
    },
    {
        "id":          "488978",
        "title":       "Autumn Rhythm (Number 30)",
        "artist":      "Jackson Pollock",
        "year":        "1950",
        "department":  "Modern and Contemporary Art",
        "description": "One of the largest and most important works of Abstract Expressionism. "
                       "Pollock created this by dripping and pouring paint directly onto canvas "
                       "laid on the floor — his famous 'drip technique'. The gestural sweep of "
                       "black, white, and brown conveys raw energy and autumn's rhythm. Gallery 919.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/488978",
        "image_url":   "https://images.metmuseum.org/CRDImages/ma/original/DP229585.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "early_modern",
        "style":       "oil_painting",
        "subject":     "abstract",
        "culture":     "American",
        "medium":      "Enamel on canvas",
    },
    {
        "id":          "435809",
        "title":       "The Harvesters",
        "artist":      "Pieter Bruegel the Elder",
        "year":        "1565",
        "department":  "European Paintings",
        "description": "One of the greatest landscape paintings ever made. Part of a series depicting "
                       "the months of the year, this August scene shows peasants harvesting wheat "
                       "under a blazing summer sky. Bruegel's panoramic vision — the vast Flemish "
                       "landscape, the tired workers, the shared meal — defines Northern Renaissance painting. Gallery 636.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/435809",
        "image_url":   "https://images.metmuseum.org/CRDImages/ep/original/DP251139.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "renaissance",
        "style":       "oil_painting",
        "subject":     "landscape_nature",
        "culture":     "Netherlandish",
        "medium":      "Oil on wood",
    },
    # ── Famous Sculptures ──────────────────────────────────────────────────────
    {
        "id":          "547802",
        "title":       "The Little Fourteen-Year-Old Dancer",
        "artist":      "Edgar Degas",
        "year":        "1922 (cast)",
        "department":  "European Sculpture and Decorative Arts",
        "description": "Originally exhibited in 1881 with real fabric — tutu, hair ribbon, satin shoes — "
                       "this was the only sculpture Degas ever exhibited publicly. Critics were shocked "
                       "by its unflinching realism. Today's bronze casts preserve his radical vision "
                       "of merging sculpture with everyday materials. Gallery 800.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/547802",
        "image_url":   "https://images.metmuseum.org/CRDImages/es/original/DP251139.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "nineteenth_century",
        "style":       "sculpture",
        "subject":     "portrait_figure",
        "culture":     "French",
        "medium":      "Bronze with fabric tutu, hair ribbon, and base",
    },
    {
        "id":          "544039",
        "title":       "Sphinx of Hatshepsut",
        "artist":      "Ancient Egyptian",
        "year":        "c. 1479–1458 BCE",
        "department":  "Egyptian Art",
        "description": "One of the finest Egyptian sculptures at the Met. This sphinx bears the face "
                       "of Hatshepsut, one of ancient Egypt's most powerful female pharaohs. Carved "
                       "from granite, the piece represents her divine authority — the lion body "
                       "symbolising royal power, the human face her wisdom. Gallery 115.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/544039",
        "image_url":   "https://images.metmuseum.org/CRDImages/eg/original/DP251139.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "ancient",
        "style":       "sculpture",
        "subject":     "portrait_figure",
        "culture":     "Egyptian",
        "medium":      "Granite",
    },
    {
        "id":          "317385",
        "title":       "The Temple of Dendur",
        "artist":      "Ancient Egyptian",
        "year":        "c. 15 BCE",
        "department":  "Egyptian Art",
        "description": "An entire ancient Egyptian temple — gifted to the United States by Egypt in 1965 "
                       "and reassembled stone by stone inside the Met. Built by Emperor Augustus for the "
                       "god Osiris, it stands in a vast sun-lit gallery with a reflecting pool. "
                       "One of the most awe-inspiring rooms in any museum in the world. Gallery 131.",
        "met_url":     "https://www.metmuseum.org/art/collection/search/317385",
        "image_url":   "https://images.metmuseum.org/CRDImages/eg/original/DP251139.jpg",
        "content_flags": "",
        "is_famous":   True,
        "is_highlight": True,
        "era":         "ancient",
        "style":       "sculpture",
        "subject":     "religious_mythological",
        "culture":     "Egyptian",
        "medium":      "Aeolian sandstone, water",
    },
]

# Honest note about what the Met actually owns vs other museums
MET_COLLECTION_NOTE = (
    "💡 **About the Met's collection:** The must-sees above include the Met's "
    "actual iconic works — Washington Crossing the Delaware, Madame X, Van Gogh's "
    "Self-Portrait, Rembrandt's Aristotle, Vermeer's Water Pitcher, Monet's Bridge, "
    "Pollock's Autumn Rhythm, Bruegel's Harvesters, Degas's Dancer, and the Temple of Dendur. "
    "Note: Van Gogh's *Starry Night* is at MoMA; Monet's *Water Lilies* murals are also at MoMA. "
    "The Met's strengths are Old Masters, Japanese prints, Egyptian antiquities, and American paintings."
)

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

    # Normalise column names
    renames = {
        'objectID':          'id',
        'artistDisplayName': 'artist',
        'primaryImageSmall': 'image_url',
        'objectURL':         'met_url',
        'isHighlight':       'is_highlight',
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


@st.cache_resource
def load_features():
    with open('feature_matrix.pkl', 'rb') as f:
        return pickle.load(f)


try:
    df             = load_data()
    feature_matrix = load_features()
    DATA_LOADED    = True
except FileNotFoundError:
    DATA_LOADED    = False


# ══════════════════════════════════════════════════════════════════════════════
# Session state — use timestamp as unique session key for true randomness
# ══════════════════════════════════════════════════════════════════════════════
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


def reset_session():
    """Completely reset — new session ID guarantees fresh randomness."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
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
    """
    Must-sees = hardcoded iconic Met artworks (always first)
              + famous artist works from CSV to fill remaining slots.
    """
    exclude_ids = set(str(i) for i in (exclude_ids or []))
    cf          = get_content_filter()

    # ── Always include hardcoded iconic artworks ──────────────────────────────
    iconic_rows = []
    for art in ICONIC_MET_ARTWORKS:
        if art['id'] in exclude_ids:
            continue
        if cf and any(f in str(art.get('content_flags', '')) for f in cf):
            continue
        iconic_rows.append(art)

    iconic_df = pd.DataFrame(iconic_rows) if iconic_rows else pd.DataFrame()

    # ── Fill remaining slots from CSV famous artists ───────────────────────────
    already_used    = set(iconic_df['id'].tolist()) if not iconic_df.empty else set()
    already_used.update(exclude_ids)
    pool            = apply_filter(df[~df['id'].isin(already_used)])
    famous          = pool[pool['is_famous'] == True]
    highlights      = pool[(pool['is_highlight'] == True) & (pool['is_famous'] == False)]
    combined        = pd.concat([famous, highlights]).drop_duplicates(subset=['id'])
    remaining_slots = max(0, MUST_SEE_N - len(iconic_df))
    csv_rows        = []
    seen_artists    = {}

    if not combined.empty and remaining_slots > 0:
        rng      = np.random.default_rng(
            int(hashlib.md5(st.session_state.session_id.encode()).hexdigest()[:8], 16)
        )
        shuffled = combined.sample(frac=1, random_state=int(rng.integers(0, 10000)))
        for _, row in shuffled.iterrows():
            artist = str(row['artist']).split('(')[0].strip()
            if seen_artists.get(artist, 0) >= 1:
                continue
            csv_rows.append(row)
            seen_artists[artist] = seen_artists.get(artist, 0) + 1
            if len(csv_rows) >= remaining_slots:
                break

    # ── Combine iconic + CSV rows ──────────────────────────────────────────────
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
    else:
        return pool.head(MUST_SEE_N)


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
    return queue['id'].astype(str).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Data not loaded
# ══════════════════════════════════════════════════════════════════════════════
if not DATA_LOADED:
    st.markdown('<div class="hero-title">🏛️ Met Museum<br>Personal Tour</div>', unsafe_allow_html=True)
    st.error("Data files not found. Please upload: `met_artworks_clean.csv`, `feature_matrix.pkl`, `tfidf_vectorizer.pkl`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Must-Sees (shown before rating starts)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 'must_sees':

    # Header — v2 Playfair style
    st.markdown('<div class="main-title">🏛️ Met Museum<br>Personal Tour</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your AI-powered guide to the Metropolitan Museum of Art</div>', unsafe_allow_html=True)

    # Content preferences
    col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
    with col_p1:
        st.markdown("**Set your content preferences before we begin:**")
    with col_p2:
        st.session_state.hide_nudity   = st.checkbox("Hide nudity")
    with col_p3:
        st.session_state.hide_violence = st.checkbox("Hide violence")

    st.markdown("---")

    # Build must-sees
    if st.session_state.must_sees_df is None:
        st.session_state.must_sees_df = get_must_sees()

    must_sees = st.session_state.must_sees_df

    # Section header
    st.markdown('<div class="section-header">⭐ Non-Negotiable Must-Sees</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">These iconic works are automatically included in your tour — '
        'no matter your ratings. We\'ve intentionally left out images so you can experience '
        'them in person for the first time. That\'s the whole point.</div>',
        unsafe_allow_html=True
    )
    st.info(MET_COLLECTION_NOTE)
    st.markdown("")

    # Text-only cards in 3-column grid — NO images
    if not must_sees.empty:
        cols = st.columns(3)
        for i, (_, row) in enumerate(must_sees.iterrows()):
            with cols[i % 3]:
                year  = str(row.get('year', row.get('objectDate', ''))).strip()
                dept  = str(row.get('department', ''))
                desc  = str(row.get('description', ''))
                flags = str(row.get('content_flags', ''))
                url   = str(row.get('met_url', ''))

                # Build description — use the rich hardcoded one if available
                if len(desc) > 30 and desc not in ['nan', '']:
                    display_desc = desc[:280] + ("..." if len(desc) > 280 else "")
                else:
                    parts = []
                    if str(row.get('medium','')) not in ['','nan','unknown']:
                        parts.append(f"Medium: {row['medium']}")
                    if str(row.get('culture','')) not in ['','nan','unknown']:
                        parts.append(f"Culture: {row['culture']}")
                    if str(row.get('era','')) not in ['','nan','Unknown Era','unknown']:
                        parts.append(f"Era: {row['era'].replace('_',' ').title()}")
                    display_desc = " · ".join(parts) if parts else f"Part of the Met's {dept} collection."

                year_str = f" ({year})" if year and year not in ['nan', ''] else ""

                st.markdown(f"""
<div class="must-see-card">
    <div class="must-see-card-title">{row['title']}{year_str}</div>
    <div class="must-see-card-artist">{row['artist']}</div>
    <div class="must-see-card-desc">{display_desc}</div>
    <div class="must-see-card-meta">
        🏛️ {dept}
        {"&nbsp;&nbsp;·&nbsp;&nbsp;<a href='" + url + "' target='_blank' style='color:#4338CA;'>View on Met ↗</a>" if url not in ['','nan'] else ""}
    </div>
</div>
""", unsafe_allow_html=True)

                render_flags(flags)

    st.markdown("---")

    # CTA
    st.markdown("### Now let's personalise the rest of your tour")
    st.markdown('<div class="subtitle">Rate 20 artworks so we can recommend everything else you\'ll love at the Met.</div>', unsafe_allow_html=True)

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

    if n_rated >= RATING_TARGET:
        st.session_state.phase = 'results'
        st.rerun()

    # Top bar
    col_title, col_prog = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="subtitle">Building Your Taste Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="main-title" style="font-size:2rem;">Rate This Artwork</div>', unsafe_allow_html=True)
    with col_prog:
        st.markdown(f'<div class="rating-counter">{n_rated + 1}<span style="font-size:1.2rem;color:#C7D2FE">/{RATING_TARGET}</span></div>', unsafe_allow_html=True)
        st.progress(n_rated / RATING_TARGET)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Get next artwork
    rated_ids = set(st.session_state.ratings.keys())
    remaining = [i for i in st.session_state.rating_queue if i not in rated_ids]

    if not remaining:
        st.session_state.phase = 'results'
        st.rerun()

    current_id = remaining[0]
    matches    = df[df['id'] == current_id]

    if matches.empty:
        st.session_state.ratings[current_id] = -1
        st.rerun()

    artwork = matches.iloc[0]
    has_image = artwork['image_url'] and str(artwork['image_url']) not in ['', 'nan']

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        if has_image:
            st.image(artwork['image_url'], use_column_width=True)
        else:
            st.markdown(
                '<div class="no-image-box">'
                '<div style="text-align:center;">'
                '<div style="font-size:3rem;margin-bottom:1rem;">🖼️</div>'
                '<div style="color:#9B8B6E;font-size:0.85rem;line-height:1.6;">'
                'Image rights restricted.<br>Read the description to learn about this work.'
                '</div></div></div>',
                unsafe_allow_html=True
            )

    with col2:
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
            artist_line = f"*{artwork['artist']}*"
            if str(artwork.get('artistNationality', '')) not in ['', 'nan']:
                artist_line += f" · {artwork['artistNationality']}"
            st.markdown(artist_line)

        # Date
        if str(artwork.get('objectDate', '')) not in ['', 'nan']:
            st.caption(f"📅 {artwork['objectDate']}")

        # Always show description
        render_description(artwork)

        # Content flags
        render_flags(artwork.get('content_flags', ''))

        # Met link
        if artwork.get('met_url', '') and str(artwork['met_url']) not in ['', 'nan']:
            st.markdown(f"[View full details on Met website ↗]({artwork['met_url']})")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Rating buttons
        st.markdown('<div class="rating-question">Would you visit this at the Met?</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            if st.button("❤️  Love it", key="love", use_container_width=True):
                st.session_state.ratings[current_id] = 2
                st.rerun()
        with r2:
            if st.button("👍  Like it", key="like", use_container_width=True):
                st.session_state.ratings[current_id] = 1
                st.rerun()
        with r3:
            if st.button("⏭️  Skip", key="skip", use_container_width=True):
                st.session_state.ratings[current_id] = 0
                st.rerun()

        remaining_count = RATING_TARGET - n_rated - 1
        st.markdown("")
        if remaining_count > 0:
            st.caption(f"⏳ {remaining_count} more artworks to rate")
        else:
            st.success("🎉 Last one! Your personalised tour is about to be revealed.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Results + Roadmap
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 'results':

    # ── Train model ────────────────────────────────────────────────────────────
    if st.session_state.recs is None:
        with st.spinner("Analysing your taste profile and curating your tour..."):

            ratings_dict = {k: v for k, v in st.session_state.ratings.items() if v >= 0}
            rated_ids    = list(ratings_dict.keys())
            labels       = list(ratings_dict.values())

            if len(set(labels)) < 2:
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

        st.rerun()

    recs      = st.session_state.recs
    must_sees = st.session_state.must_sees_df

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.9rem;font-weight:600;letter-spacing:0.05em;'
        'text-transform:uppercase;color:#6b7280 !important;margin-bottom:0.25rem;">'
        'The Metropolitan Museum of Art</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="main-title">🗺️ Your Personal Tour</div>', unsafe_allow_html=True)

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
                f'display:inline-block;margin:0.5rem 0;color:#5C4A2A !important;font-size:0.85rem;">'
                f'Your taste: {chips}</div>',
                unsafe_allow_html=True
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── SECTION 1: Must-Sees ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">⭐ Must-See Masterpieces</div>', unsafe_allow_html=True)
    st.caption("Iconic works automatically included in your tour — curated by the Met and art history.")

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

    st.markdown("---")

    # ── SECTION 2: Personalised Recommendations ───────────────────────────────
    st.markdown('<div class="section-header">🎯 Recommended For You</div>', unsafe_allow_html=True)
    st.caption("Ranked by predicted enjoyment based on your taste profile.")

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
            f"{fire}{row['title']}  ·  {row['artist']}  ·  {row['department']}  ·  {score:.0%} match",
            expanded=(i < 3)
        ):
            ec1, ec2 = st.columns([1, 2], gap="large")

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

                # Score
                st.markdown(
                    f'<div style="margin:0.5rem 0;">'
                    f'<span class="score-ring">{score:.0%}</span>'
                    f'<span style="color:#9B8B6E;font-size:0.8rem;margin-left:0.5rem;">match score</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Always show description
                render_description(row)

                render_flags(row.get('content_flags', ''))

                # Why recommended
                reasons = []
                if row['department'] in liked_depts:
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

    st.markdown("---")

    # ── SECTION 3: Gallery Roadmap ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Your Gallery Roadmap</div>', unsafe_allow_html=True)
    st.caption("Your full tour organised by gallery — visit in this order for the best experience.")

    # Combine must-sees + top recs
    ms_ids      = must_sees['id'].tolist() if must_sees is not None and not must_sees.empty else []
    top_recs    = display_recs.head(40)
    all_ids     = ms_ids + [i for i in top_recs['id'].tolist() if i not in ms_ids]
    roadmap_df  = df[df['id'].isin(all_ids)].copy()

    score_map                     = dict(zip(display_recs['id'], display_recs['predicted_score']))
    roadmap_df['predicted_score'] = roadmap_df['id'].map(score_map).fillna(0.95)

    dept_summary = (
        roadmap_df.groupby('department')['predicted_score']
        .agg(['count', 'mean'])
        .sort_values(['count', 'mean'], ascending=False)
    )

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
                        st.markdown('<span class="badge-must-see">Must See</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="score-display">{row["predicted_score"]:.0%}</span>', unsafe_allow_html=True)
                        st.caption("match")
                    if row.get('met_url', '') and str(row['met_url']) not in ['', 'nan']:
                        st.markdown(f"[Met ↗]({row['met_url']})")

                st.markdown("---")

    # Tour summary — v2 dark navy style
    st.markdown(
        f'<div class="tour-summary">'
        f'<div style="font-family:Playfair Display,serif;font-size:1.3rem;'
        f'color:#E0E7FF !important;margin-bottom:0.5rem;font-weight:700;">Your Complete Tour</div>'
        f'<div style="color:#C7D2FE !important;font-size:0.9rem;">'
        f'⏱️ &nbsp;{total_time} minutes total &nbsp;·&nbsp; '
        f'🖼️ &nbsp;{total_works} artworks &nbsp;·&nbsp; '
        f'🏛️ &nbsp;{len(dept_summary)} galleries'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("← Refine my recommendations", use_container_width=True):
            st.session_state.phase = 'rating'
            st.session_state.recs  = None
            st.rerun()
    with col_b2:
        if st.button("🔄  Start a completely new tour", use_container_width=True):
            reset_session()
