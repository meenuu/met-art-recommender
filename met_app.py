"""
The Met · Personal Tour — Final Version
=========================================
Phase 0 (Landing): Exact warm CSS from original snippet
Phase 1 + 2: Dark premium purple theme
IndexError fixed: df truncated to feature_matrix row count
Iconic artworks excluded from rating queue
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
    page_title="Met Art Recommender",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Phase 0 uses the exact original warm CSS
#       Phases 1 & 2 inject dark overrides on top
# ══════════════════════════════════════════════════════════════════════════════

# ── EXACT original warm CSS (unchanged) ──────────────────────────────────────
WARM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&display=swap');

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

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

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
.must-see-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
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
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 700;
    color: #E8E0D5;
    line-height: 1;
}
.rating-question {
    font-family: 'Playfair Display', serif;
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
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #B8960C;
}
.roadmap-dept {
    background: white;
    border: 1px solid #E8E0D5;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
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

/* Must-see cards — warm cream */
.ms-card {
    background: #FDFAF5;
    border: 1px solid #E8DDD0;
    border-top: 3px solid #C9A55A;
    border-radius: 8px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
    min-height: 230px;
}
.ms-num {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #D4B870;
    margin-bottom: 0.7rem;
}
.ms-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.3;
    margin-bottom: 0.2rem;
}
.ms-artist {
    font-size: 0.76rem;
    font-weight: 500;
    color: #B8860B;
    letter-spacing: 0.04em;
    margin-bottom: 0.75rem;
}
.ms-desc {
    font-size: 0.79rem;
    color: #6b7280;
    line-height: 1.65;
    margin-bottom: 0.75rem;
}
.ms-footer {
    font-size: 0.7rem;
    color: #9B8B6E;
    border-top: 1px solid #EDE3D8;
    padding-top: 0.55rem;
}
.ms-footer a { color: #B8860B; text-decoration: none; }

/* CTA box */
.cta-box {
    background: #FDF6EC;
    border: 1px solid #E8DDD0;
    border-radius: 10px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
}
.cta-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.5rem;
}
.cta-body { font-size: 0.88rem; color: #6b7280; font-weight: 300; }

/* Warm info note */
.warm-note {
    background: #FDF6EC;
    border: 1px solid #E8C870;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    font-size: 0.82rem;
    color: #7A6020;
    line-height: 1.6;
    margin-bottom: 1.25rem;
}
</style>
"""

# ── Dark override injected for phases 1 & 2 ──────────────────────────────────
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap');

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
[data-testid="stCaptionContainer"] p { color: #6A6278 !important; font-size: 0.8rem !important; }
hr { border-color: #23202E !important; }
.block-container { padding: 2.5rem 3.5rem !important; max-width: 1400px !important; }

[data-testid="stExpander"] { background: #16141E !important; border: 1px solid #23202E !important; border-radius: 6px !important; margin-bottom: 6px !important; }
[data-testid="stExpander"] summary { color: #C0B8D8 !important; font-size: 0.86rem !important; }
[data-testid="stExpander"] summary:hover { color: #A89EE8 !important; }
[data-testid="stInfo"] { background: #100F1A !important; border: 1px solid #2E2050 !important; }
[data-testid="stInfo"] p { color: #9B8FD8 !important; }
[data-testid="stProgress"] > div { background: #1E1C28 !important; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #7B6FD0, #A89EE8) !important; }
.stButton > button {
    background: transparent !important; border: 1px solid #8B7FD4 !important;
    color: #A89EE8 !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.74rem !important; font-weight: 500 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important; border-radius: 3px !important; width: 100% !important;
}
.stButton > button:hover { background: #8B7FD4 !important; color: #0F0F13 !important; }
[data-testid="stSelectbox"] > div > div { background: #16141E !important; border-color: #2E2A3E !important; color: #D4CCE0 !important; }
[data-testid="stCheckbox"] label p { color: #6A6278 !important; }

/* Dark-phase typography */
.section-label { font-size: 0.61rem; font-weight: 600; letter-spacing: 0.28em; text-transform: uppercase; color: #8B7FD4; display: block; margin-bottom: 0.6rem; }
.dp-title { font-family: 'Cormorant Garamond', serif; font-size: 3.2rem; font-weight: 300; line-height: 1.08; color: #EDE8F5; margin-bottom: 0.5rem; }
.dp-title-sm { font-family: 'Cormorant Garamond', serif; font-size: 2.2rem; font-weight: 300; color: #EDE8F5; margin-bottom: 0.25rem; }
.dp-body { font-size: 0.84rem; font-weight: 300; color: #6A6278; line-height: 1.7; }
.dp-rule { display: flex; align-items: center; gap: 1rem; margin: 2rem 0 1.2rem 0; }
.dp-rule-text { font-size: 0.6rem; font-weight: 600; letter-spacing: 0.28em; text-transform: uppercase; color: #8B7FD4; white-space: nowrap; }
.dp-rule-line { flex: 1; height: 1px; background: linear-gradient(90deg, #2E2A3E 0%, transparent 100%); }

/* Dark must-see cards */
.dp-ms-card { background: #13111C; border: 1px solid #23202E; border-top: 2px solid #7B6FD0; border-radius: 6px; padding: 1.3rem 1.4rem; margin-bottom: 1rem; min-height: 210px; }
.dp-ms-num  { font-size: 0.58rem; font-weight: 600; letter-spacing: 0.25em; text-transform: uppercase; color: #3A3460; margin-bottom: 0.7rem; }
.dp-ms-title { font-family: 'Cormorant Garamond', serif; font-size: 1.08rem; font-weight: 600; color: #EDE8F5; line-height: 1.3; margin-bottom: 0.2rem; }
.dp-ms-artist { font-size: 0.76rem; font-weight: 500; color: #8B7FD4; letter-spacing: 0.04em; margin-bottom: 0.75rem; }
.dp-ms-desc { font-size: 0.79rem; color: #5A5268; line-height: 1.65; margin-bottom: 0.75rem; }
.dp-ms-footer { font-size: 0.7rem; color: #3A3460; border-top: 1px solid #1E1C28; padding-top: 0.55rem; }
.dp-ms-footer a { color: #7B6FD0; text-decoration: none; }

/* Rating */
.rating-num { font-family: 'Cormorant Garamond', serif; font-size: 5rem; font-weight: 300; color: #1E1C28; line-height: 1; }
.rating-num-frac { font-family: 'Cormorant Garamond', serif; font-size: 1.4rem; color: #2A2538; }
.rating-q { font-family: 'Cormorant Garamond', serif; font-size: 1.1rem; font-style: italic; font-weight: 300; color: #6A6278; margin: 0.5rem 0 1.25rem 0; }
.art-title { font-family: 'Cormorant Garamond', serif; font-size: 2rem; font-weight: 400; color: #EDE8F5; line-height: 1.2; margin-bottom: 0.2rem; }
.art-artist { font-size: 0.82rem; color: #8B7FD4; letter-spacing: 0.05em; margin-bottom: 1.1rem; }

/* Dark desc-box override */
.desc-box { background: #0D0C16 !important; border-left: 2px solid #5A50A8 !important; color: #6A6278 !important; }

/* Badges */
.badge-purple { display: inline-block; font-size: 0.58rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; padding: 3px 9px; border: 1px solid #7B6FD0; border-radius: 2px; color: #A89EE8; margin-right: 6px; }
.badge-dim    { display: inline-block; font-size: 0.58rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; padding: 3px 9px; border: 1px solid #2E2A3E; border-radius: 2px; color: #4A4460; margin-right: 6px; }

/* Score */
.score-num { font-family: 'Cormorant Garamond', serif; font-size: 2.8rem; font-weight: 300; color: #A89EE8; line-height: 1; }
.score-lbl { font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: #3A3460; }

/* Why box dark override */
.why-box { background: #0E0D18 !important; border: none !important; border-left: 2px solid #7B6FD0 !important; color: #8B80C0 !important; }

/* Taste chips */
.taste-chip { display: inline-block; background: #13111C; border: 1px solid #23202E; padding: 4px 12px; font-size: 0.74rem; color: #6A6278; border-radius: 3px; margin: 2px 4px 2px 0; }

/* Content flag dark */
.cflag { display: inline-block; padding: 2px 8px; font-size: 0.65rem; font-weight: 500; background: #180F1A; border: 1px solid #4A1A3A; color: #C06080; border-radius: 2px; margin-right: 4px; }

/* Stats */
.stat-num { font-family: 'Cormorant Garamond', serif; font-size: 2.8rem; font-weight: 300; color: #A89EE8; line-height: 1; }
.stat-lbl { font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: #3A3460; margin-top: 0.25rem; }
</style>
"""

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

ICONIC_MET_ARTWORKS = [
    {"id":"iconic_11417","title":"Washington Crossing the Delaware","artist":"Emanuel Leutze","year":"1851","department":"The American Wing","description":"One of the most famous images in American history. This enormous canvas (12 × 21 ft) depicts Washington's daring crossing of the icy Delaware River on December 25–26, 1776. Gallery 760.","met_url":"https://www.metmuseum.org/art/collection/search/11417","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"oil_painting","culture":"American","medium":"Oil on canvas"},
    {"id":"iconic_12127","title":"Madame X (Madame Pierre Gautreau)","artist":"John Singer Sargent","year":"1883–84","department":"The American Wing","description":"The most scandalous portrait of its era. Sargent's daring portrayal caused a sensation at the 1884 Paris Salon. The contrast of pale skin against the black gown became a defining image of 19th-century portraiture. Gallery 771.","met_url":"https://www.metmuseum.org/art/collection/search/12127","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"oil_painting","culture":"American","medium":"Oil on canvas"},
    {"id":"iconic_436532","title":"Self-Portrait with a Straw Hat","artist":"Vincent van Gogh","year":"1887","department":"European Paintings","description":"Painted during Van Gogh's Paris years, this portrait shows his rapid evolution under Impressionist influence. Vivid brushwork and dazzling colour contrasts — blues, oranges, yellows — mark his break from the sombre Dutch palette. Gallery 825.","met_url":"https://www.metmuseum.org/art/collection/search/436532","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"oil_painting","culture":"Dutch","medium":"Oil on canvas"},
    {"id":"iconic_437394","title":"Aristotle with a Bust of Homer","artist":"Rembrandt van Rijn","year":"1653","department":"European Paintings","description":"A masterpiece of psychological depth and dramatic light. The Met paid $2.3 million for it in 1961 — then the highest price ever paid for a painting. Rembrandt's genius is at its peak here. Gallery 964.","met_url":"https://www.metmuseum.org/art/collection/search/437394","content_flags":"","is_famous":True,"era":"baroque_rococo","style":"oil_painting","culture":"Dutch","medium":"Oil on canvas"},
    {"id":"iconic_437870","title":"Young Woman with a Water Pitcher","artist":"Johannes Vermeer","year":"c. 1662","department":"European Paintings","description":"A serene domestic scene bathed in Vermeer's signature cool northern light. Only 34 Vermeers are known to exist worldwide — making this one of the Met's most precious possessions. Gallery 964.","met_url":"https://www.metmuseum.org/art/collection/search/437870","content_flags":"","is_famous":True,"era":"baroque_rococo","style":"oil_painting","culture":"Dutch","medium":"Oil on canvas"},
    {"id":"iconic_437130","title":"Bridge over a Pond of Water Lilies","artist":"Claude Monet","year":"1899","department":"European Paintings","description":"Painted in Monet's garden at Giverny, this canvas captures the Japanese footbridge reflected in the lily pond he designed himself. A cornerstone of Impressionism and a rare Monet at the Met. Gallery 819.","met_url":"https://www.metmuseum.org/art/collection/search/437130","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"oil_painting","culture":"French","medium":"Oil on canvas"},
    {"id":"iconic_436105","title":"The Death of Socrates","artist":"Jacques-Louis David","year":"1787","department":"European Paintings","description":"Socrates calmly accepts death, reaching for the hemlock cup while followers grieve. The defining image of Enlightenment idealism. Gallery 614.","met_url":"https://www.metmuseum.org/art/collection/search/436105","content_flags":"","is_famous":True,"era":"baroque_rococo","style":"oil_painting","culture":"French","medium":"Oil on canvas"},
    {"id":"iconic_488978","title":"Autumn Rhythm (Number 30)","artist":"Jackson Pollock","year":"1950","department":"Modern and Contemporary Art","description":"Pollock created this by dripping paint onto canvas laid on the floor — his revolutionary drip technique. The gestural sweep of black, white, and brown conveys raw, untamed energy. Gallery 919.","met_url":"https://www.metmuseum.org/art/collection/search/488978","content_flags":"","is_famous":True,"era":"early_modern","style":"oil_painting","culture":"American","medium":"Enamel on canvas"},
    {"id":"iconic_435809","title":"The Harvesters","artist":"Pieter Bruegel the Elder","year":"1565","department":"European Paintings","description":"Part of a series depicting the months of the year — this August scene shows peasants harvesting wheat under a blazing summer sky. One of the greatest landscape paintings ever made. Gallery 636.","met_url":"https://www.metmuseum.org/art/collection/search/435809","content_flags":"","is_famous":True,"era":"renaissance","style":"oil_painting","culture":"Netherlandish","medium":"Oil on wood"},
    {"id":"iconic_547802","title":"The Little Fourteen-Year-Old Dancer","artist":"Edgar Degas","year":"1922 (cast)","department":"European Sculpture and Decorative Arts","description":"The only sculpture Degas ever exhibited publicly — originally shown with real fabric: tutu, hair ribbon, satin shoes. Critics were shocked by its realism. Gallery 800.","met_url":"https://www.metmuseum.org/art/collection/search/547802","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"sculpture","culture":"French","medium":"Bronze with fabric tutu"},
    {"id":"iconic_544039","title":"Sphinx of Hatshepsut","artist":"Ancient Egyptian","year":"c. 1479–1458 BCE","department":"Egyptian Art","description":"This granite sphinx bears the face of Hatshepsut — one of ancient Egypt's most powerful female pharaohs. The lion body symbolises royal power; the human face, wisdom. Gallery 115.","met_url":"https://www.metmuseum.org/art/collection/search/544039","content_flags":"","is_famous":True,"era":"ancient","style":"sculpture","culture":"Egyptian","medium":"Granite"},
    {"id":"iconic_317385","title":"The Temple of Dendur","artist":"Ancient Egyptian","year":"c. 15 BCE","department":"Egyptian Art","description":"An entire ancient Egyptian temple gifted to the US by Egypt in 1965 — reassembled stone by stone inside the Met. It stands in a vast sun-lit gallery with a reflecting pool. One of the most awe-inspiring rooms in any museum. Gallery 131.","met_url":"https://www.metmuseum.org/art/collection/search/317385","content_flags":"","is_famous":True,"era":"ancient","style":"sculpture","culture":"Egyptian","medium":"Aeolian sandstone"},
]

ICONIC_IDS = {art["id"] for art in ICONIC_MET_ARTWORKS}

MET_NOTE = (
    "ℹ️  Van Gogh's Starry Night and Monet's Water Lilies murals are at MoMA — but the Met holds "
    "Van Gogh's Self-Portrait and Monet's Bridge, both included above. "
    "The Met's true strengths: Rembrandt, Vermeer, Hokusai, Degas, Egyptian antiquities, and American masters."
)

DEPT_TIME = {
    "European Paintings":45,"American Paintings and Sculpture":35,
    "Modern and Contemporary Art":40,"Asian Art":30,"Egyptian Art":25,
    "Greek and Roman Art":30,"Islamic Art":25,"The American Wing":35,
    "Robert Lehman Collection":20,"default":15,
}

CONTENT_FLAGS_DEF = {"nudity":"🔞 Nudity","violence":"⚔️ Violence","religious":"✝️ Religious"}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('met_artworks_clean.csv')
    renames = {'objectID':'id','artistDisplayName':'artist',
                'primaryImageSmall':'image_url','objectURL':'met_url',
                'isHighlight':'is_highlight','is_famous_artist':'is_famous'}
    for old,new in renames.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old:new})
    df['id']           = df['id'].astype(str)
    df['artist']       = df['artist'].fillna('Unknown Artist')
    df['image_url']    = df['image_url'].fillna('')
    df['met_url']      = df['met_url'].fillna('')
    df['is_highlight'] = df['is_highlight'].fillna(False).astype(bool)
    df['era']          = df['era'].fillna('Unknown Era')
    df['culture']      = df['culture'].fillna('')
    df['medium']       = df['medium'].fillna('')
    df['tags']         = df['tags'].fillna('')
    df['department']   = df['department'].fillna('Unknown')
    df['title']        = df['title'].fillna('Untitled')
    df['description']  = df['description'].fillna('') if 'description' in df.columns else ''
    df['content_flags']= df['content_flags'].fillna('') if 'content_flags' in df.columns else ''
    df['style']        = df['style'].fillna('') if 'style' in df.columns else ''
    if 'is_famous' not in df.columns:
        df['is_famous'] = df['artist'].apply(lambda a: any(n in str(a).lower() for n in FAMOUS_ARTIST_NAMES))
    else:
        df['is_famous'] = df['is_famous'].fillna(False).astype(bool)
        df['is_famous'] = df['is_famous'] | df['artist'].apply(lambda a: any(n in str(a).lower() for n in FAMOUS_ARTIST_NAMES))
    return df.reset_index(drop=True)


@st.cache_resource
def load_features():
    with open('feature_matrix.pkl','rb') as f:
        return pickle.load(f)


try:
    df             = load_data()
    feature_matrix = load_features()
    # ── CRITICAL: Truncate df to match feature_matrix row count ──────────────
    # Prevents IndexError when CSV has more rows than the matrix (e.g. 2788 vs 2022)
    n_matrix = feature_matrix.shape[0]
    if len(df) > n_matrix:
        df = df.iloc[:n_matrix].reset_index(drop=True)
    DATA_LOADED = True
except FileNotFoundError:
    DATA_LOADED = False


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
if 'session_id' not in st.session_state:
    st.session_state.session_id    = str(time.time_ns())
    st.session_state.ratings       = {}
    st.session_state.phase         = 'must_sees'
    st.session_state.rating_queue  = []
    st.session_state.recs          = None
    st.session_state.must_sees_df  = None
    st.session_state.hide_nudity   = False
    st.session_state.hide_violence = False


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
# Inject phase CSS
# Phase 0: warm CSS only (already injected above as WARM_CSS)
# Phase 1 & 2: dark overrides injected on top
# ══════════════════════════════════════════════════════════════════════════════
phase = st.session_state.get('phase', 'must_sees')
st.markdown(WARM_CSS, unsafe_allow_html=True)
if phase != 'must_sees':
    st.markdown(DARK_CSS, unsafe_allow_html=True)



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
        lambda x: not any(f in str(x) for f in cf))]

def render_flags(flags_str):
    if not flags_str or str(flags_str) in ['','nan']:
        return
    for flag in str(flags_str).split('|'):
        flag = flag.strip()
        if flag in CONTENT_FLAGS_DEF:
            st.markdown(f'<span class="cflag">{CONTENT_FLAGS_DEF[flag]}</span>', unsafe_allow_html=True)

def build_desc(row):
    desc = str(row.get('description',''))
    if len(desc) > 30 and desc not in ['nan','']:
        return desc[:400]
    parts = []
    if str(row.get('medium','')) not in ['','nan','unknown']:   parts.append(f"Medium: {row['medium']}")
    if str(row.get('culture','')) not in ['','nan','unknown']:  parts.append(f"Culture: {row['culture']}")
    if str(row.get('era',''))    not in ['','nan','Unknown Era','unknown']:
        parts.append(f"Era: {str(row['era']).replace('_',' ').title()}")
    if str(row.get('tags',''))   not in ['','nan']:             parts.append(f"Tags: {str(row['tags'])[:80]}")
    return "  ·  ".join(parts) if parts else f"Part of the Met's {row.get('department','collection')}."

def render_desc_box(row):
    st.markdown(f'<div class="desc-box">{build_desc(row)}</div>', unsafe_allow_html=True)

def dp_rule(label):
    st.markdown(
        f'<div class="dp-rule"><span class="dp-rule-text">{label}</span>'
        f'<div class="dp-rule-line"></div></div>',
        unsafe_allow_html=True)

def get_must_sees(exclude_ids=None):
    exclude_ids = set(str(i) for i in (exclude_ids or []))
    cf = get_cf()
    iconic_rows = [a for a in ICONIC_MET_ARTWORKS
                   if a['id'] not in exclude_ids
                   and not (cf and any(f in str(a.get('content_flags','')) for f in cf))]
    iconic_df = pd.DataFrame(iconic_rows) if iconic_rows else pd.DataFrame()

    already  = set(iconic_df['id'].tolist()) if not iconic_df.empty else set()
    already.update(exclude_ids)
    pool     = apply_filter(df[~df['id'].isin(already)])
    famous   = pool[pool['is_famous'] == True]
    hilights = pool[(pool['is_highlight'] == True) & (~pool['is_famous'])]
    combined = pd.concat([famous, hilights]).drop_duplicates(subset=['id'])
    remaining = max(0, MUST_SEE_N - len(iconic_df))
    csv_rows, seen_art = [], {}

    if not combined.empty and remaining > 0:
        seed = int(hashlib.md5(st.session_state.session_id.encode()).hexdigest()[:8], 16)
        rng  = np.random.default_rng(seed)
        shuf = combined.sample(frac=1, random_state=int(rng.integers(0, 10000)))
        for _, row in shuf.iterrows():
            artist = str(row['artist']).split('(')[0].strip()
            if seen_art.get(artist, 0) >= 1: continue
            csv_rows.append(row)
            seen_art[artist] = seen_art.get(artist, 0) + 1
            if len(csv_rows) >= remaining: break

    if not iconic_df.empty and csv_rows:
        csv_df   = pd.DataFrame(csv_rows)
        all_cols = list(set(iconic_df.columns.tolist() + csv_df.columns.tolist()))
        for col in all_cols:
            if col not in iconic_df.columns: iconic_df[col] = ''
            if col not in csv_df.columns:    csv_df[col]    = ''
        return pd.concat([iconic_df, csv_df[all_cols]], ignore_index=True)
    elif not iconic_df.empty: return iconic_df
    elif csv_rows: return pd.DataFrame(csv_rows)
    return pool.head(MUST_SEE_N)

def build_rating_queue(exclude_ids=None):
    exclude_ids = set(str(i) for i in (exclude_ids or []))
    exclude_ids.update(ICONIC_IDS)  # never include iconic artworks in rating queue
    filtered = apply_filter(df[~df['id'].isin(exclude_ids)])
    seed     = int(hashlib.md5(st.session_state.session_id.encode()).hexdigest()[:8], 16) % 100000
    famous   = filtered[filtered['is_famous'] == True]
    hilights = filtered[(filtered['is_highlight'] == True) & (~filtered['is_famous'])]
    rest     = filtered[~filtered['id'].isin(
        pd.concat([famous, hilights])['id'].tolist() if not famous.empty else hilights['id'].tolist())]
    n_f = min(6, len(famous)); n_h = min(4, len(hilights)); n_r = max(0, RATING_TARGET - n_f - n_h)
    parts = []
    if n_f > 0: parts.append(famous.sample(n_f, random_state=seed))
    if n_h > 0: parts.append(hilights.sample(min(n_h, len(hilights)), random_state=seed+1))
    if n_r > 0 and len(rest) > 0: parts.append(rest.sample(min(n_r, len(rest)), random_state=seed+2))
    queue = pd.concat(parts).sample(frac=1, random_state=seed+3) if parts else \
            filtered.sample(min(RATING_TARGET, len(filtered)), random_state=seed)
    return queue['id'].astype(str).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if phase == 'must_sees':
        st.markdown('<span class="section-label">The Met · Personal Tour</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="section-label">The Met · Personal Tour</span>', unsafe_allow_html=True)
    st.markdown("---")
    n = len(st.session_state.ratings)
    st.progress(min(n / RATING_TARGET, 1.0))
    st.caption(f"{n} / {RATING_TARGET} artworks rated")
    if n > 0:
        loves = sum(1 for v in st.session_state.ratings.values() if v == 2)
        likes = sum(1 for v in st.session_state.ratings.values() if v == 1)
        skips = sum(1 for v in st.session_state.ratings.values() if v == 0)
        st.caption(f"❤️ {loves}  ·  👍 {likes}  ·  ⏭ {skips}")
    st.markdown("---")
    st.caption("Content filters")
    st.session_state.hide_nudity   = st.checkbox("Exclude nudity",   value=st.session_state.get('hide_nudity', False))
    st.session_state.hide_violence = st.checkbox("Exclude violence",  value=st.session_state.get('hide_violence', False))
    st.markdown("---")
    if st.button("↺  Start Over"):
        reset_session()


# ══════════════════════════════════════════════════════════════════════════════
# Data not loaded
# ══════════════════════════════════════════════════════════════════════════════
if not DATA_LOADED:
    st.markdown('<div class="main-title">🏛️ The Met · Personal Tour</div>', unsafe_allow_html=True)
    st.error("Data files not found. Ensure `met_artworks_clean.csv`, `feature_matrix.pkl`, `tfidf_vectorizer.pkl` are present.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Landing page · WARM CREAM THEME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 'must_sees':

    # ── Hero ──────────────────────────────────────────────────────────────────
    col_hero, col_stats = st.columns([3, 1], gap="large")

    with col_hero:
        st.markdown('<span class="section-label">The Metropolitan Museum of Art · New York</span>', unsafe_allow_html=True)
        st.markdown('<div class="main-title">Your Personal<br>Museum Tour</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">An AI-powered guide built around your taste. '
            'Rate 20 artworks, and we\'ll generate a personalised tour of the Met — '
            'ranked by predicted enjoyment, with a gallery-by-gallery roadmap.</div>',
            unsafe_allow_html=True)

    with col_stats:
        st.markdown("")
        st.markdown("")
        for num, lbl in [("2,022", "Artworks catalogued"), ("12", "Must-see masterpieces"), ("20", "Ratings to personalise")]:
            st.markdown(
                f'<div style="text-align:right;margin-bottom:1.2rem;">'
                f'<div style="font-family:Playfair Display,serif;font-size:2.4rem;font-weight:700;color:#1a1a2e;line-height:1;">{num}</div>'
                f'<div style="font-size:0.7rem;font-weight:500;letter-spacing:0.12em;text-transform:uppercase;color:#9B8B6E;">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # Content preferences
    cp1, cp2, cp3 = st.columns([3, 1, 1])
    with cp1:
        st.markdown('<span class="section-label">Before we begin</span>', unsafe_allow_html=True)
        st.markdown('<span style="font-size:0.88rem;color:#6b7280;">Set your content preferences:</span>', unsafe_allow_html=True)
    with cp2:
        st.session_state.hide_nudity   = st.checkbox("Exclude nudity",   value=st.session_state.get('hide_nudity', False), key="p0_nudity")
    with cp3:
        st.session_state.hide_violence = st.checkbox("Exclude violence",  value=st.session_state.get('hide_violence', False), key="p0_violence")

    st.markdown("---")

    # Build must-sees
    if st.session_state.must_sees_df is None:
        st.session_state.must_sees_df = get_must_sees()
    must_sees = st.session_state.must_sees_df

    # ── Must-sees section ─────────────────────────────────────────────────────
    st.markdown('<span class="section-label">Non-Negotiables · Always included in your tour</span>', unsafe_allow_html=True)
    st.markdown('<div class="main-title" style="font-size:1.8rem;">⭐ Must-See Masterpieces</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle" style="font-size:0.88rem;" style="margin-bottom:1rem;">These iconic works are guaranteed in your tour regardless of your ratings. '
        'We\'ve intentionally withheld images — experience them for the first time in person.</div>',
        unsafe_allow_html=True)

    st.markdown(f'<div class="warm-note">{MET_NOTE}</div>', unsafe_allow_html=True)

    if not must_sees.empty:
        cols = st.columns(3)
        for i, (_, row) in enumerate(must_sees.iterrows()):
            with cols[i % 3]:
                year = str(row.get('year', row.get('objectDate',''))).strip()
                yr   = f" · {year}" if year and year not in ['nan',''] else ""
                desc = build_desc(row)[:260]
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

    # ── CTA ───────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
<div class="cta-box">
  <div class="cta-title">Now let's personalise your tour</div>
  <div class="cta-body">Rate 20 artworks and we'll recommend everything else<br>you'll love at the Met — with a full gallery roadmap.</div>
</div>""", unsafe_allow_html=True)
        if st.button("Begin Taste Profile  →", use_container_width=True):
            ms_ids = must_sees['id'].tolist() if not must_sees.empty else []
            st.session_state.phase        = 'rating'
            st.session_state.rating_queue = build_rating_queue(exclude_ids=ms_ids)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Rating · DARK PURPLE THEME
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 'rating':

    n_rated = len(st.session_state.ratings)
    if n_rated >= RATING_TARGET:
        st.session_state.phase = 'results'
        st.rerun()

    # Progress header
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown('<span class="section-label">Building Your Taste Profile</span>', unsafe_allow_html=True)
        st.markdown('<div class="dp-title-sm">Rate This Artwork</div>', unsafe_allow_html=True)
        st.progress(n_rated / RATING_TARGET)
        st.markdown(f'<span class="dp-body">{n_rated} of {RATING_TARGET} rated</span>', unsafe_allow_html=True)
    with hc2:
        st.markdown(
            f'<div class="rating-num">{n_rated+1}'
            f'<span class="rating-num-frac">/{RATING_TARGET}</span></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # Get next artwork
    rated_ids = set(st.session_state.ratings.keys())
    remaining = [i for i in st.session_state.rating_queue if i not in rated_ids]
    if not remaining:
        st.session_state.phase = 'results'; st.rerun()

    current_id = remaining[0]
    matches    = df[df['id'] == current_id]
    if matches.empty:
        st.session_state.ratings[current_id] = -1; st.rerun()

    artwork   = matches.iloc[0]
    has_image = artwork['image_url'] and str(artwork['image_url']) not in ['','nan']

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        if has_image:
            st.image(artwork['image_url'], use_column_width=True)
        else:
            st.markdown(
                '<div style="background:#0F0F13;border:1px solid #1E1C28;border-radius:4px;'
                'min-height:400px;display:flex;align-items:center;justify-content:center;">'
                '<div style="text-align:center;"><div style="font-size:2.5rem;opacity:0.2;margin-bottom:0.75rem;">🖼</div>'
                '<div style="font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;color:#2A2538;">Image restricted</div>'
                '</div></div>', unsafe_allow_html=True)

    with col2:
        bdg = f'<span class="badge-dim">{artwork["department"]}</span>'
        if artwork.get('is_famous'): bdg += '<span class="badge-purple">Master</span>'
        if artwork.get('is_highlight'): bdg += '<span class="badge-purple">Met Pick</span>'
        st.markdown(bdg, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f'<div class="art-title">{artwork["title"]}</div>', unsafe_allow_html=True)
        al = str(artwork['artist'])
        if str(artwork.get('artistNationality','')) not in ['','nan']: al += f"  ·  {artwork['artistNationality']}"
        if str(artwork.get('objectDate',''))        not in ['','nan']: al += f"  ·  {artwork['objectDate']}"
        st.markdown(f'<div class="art-artist">{al}</div>', unsafe_allow_html=True)
        render_desc_box(artwork)
        render_flags(artwork.get('content_flags',''))
        if artwork.get('met_url','') and str(artwork['met_url']) not in ['','nan']:
            st.markdown(f'<a href="{artwork["met_url"]}" target="_blank" style="color:#8B7FD4;font-size:0.75rem;letter-spacing:0.05em;">View on Met website ↗</a>', unsafe_allow_html=True)
        st.markdown('<div class="rating-q">Would you stop and look at this at the Met?</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            if st.button("❤  Love it", key="love"):
                st.session_state.ratings[current_id] = 2; st.rerun()
        with r2:
            if st.button("👍  Like it", key="like"):
                st.session_state.ratings[current_id] = 1; st.rerun()
        with r3:
            if st.button("⏭  Skip", key="skip"):
                st.session_state.ratings[current_id] = 0; st.rerun()
        left = RATING_TARGET - n_rated - 1
        st.markdown("")
        if left > 0:
            st.markdown(f'<span class="dp-body">{left} more to go</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#A89EE8;font-size:0.82rem;">Last one — your tour is almost ready.</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Results · DARK PURPLE THEME
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 'results':

    # ── Train model ──────────────────────────────────────────────────────────
    if st.session_state.recs is None:
        with st.spinner("Analysing your taste and curating your tour..."):
            # Filter out iconic_ IDs and invalid ratings before touching feature_matrix
            valid = {k: v for k, v in st.session_state.ratings.items()
                     if v >= 0 and not str(k).startswith('iconic_')}
            rated_ids = list(valid.keys())
            labels    = list(valid.values())

            if len(set(labels)) < 2:
                st.warning("Try mixing Love, Like, and Skip for better results.")
                fb = apply_filter(df[~df['id'].isin(rated_ids)]).head(TOP_N_RECS).copy()
                fb['predicted_score'] = 0.5
                st.session_state.recs = fb
            else:
                rated_indices, valid_ids, valid_labels = [], [], []
                for oid, label in zip(rated_ids, labels):
                    match = df[df['id'] == oid]
                    if not match.empty:
                        idx = match.index[0]
                        if idx < feature_matrix.shape[0]:   # safety guard
                            rated_indices.append(idx)
                            valid_ids.append(oid)
                            valid_labels.append(label)

                if len(set(valid_labels)) < 2:
                    fb = apply_filter(df[~df['id'].isin(valid_ids)]).head(TOP_N_RECS).copy()
                    fb['predicted_score'] = 0.5
                    st.session_state.recs = fb
                else:
                    X_train = feature_matrix[rated_indices]
                    y_train = np.array(valid_labels)
                    clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                                 class_weight='balanced', n_jobs=-1)
                    clf.fit(X_train, y_train)

                    unrated_df  = df[~df['id'].isin(valid_ids)].copy()
                    # Only keep rows within feature_matrix bounds
                    unrated_df  = unrated_df[unrated_df.index < feature_matrix.shape[0]]
                    unrated_idx = unrated_df.index.tolist()
                    X_unrated   = feature_matrix[unrated_idx]
                    proba       = clf.predict_proba(X_unrated)
                    classes     = clf.classes_.tolist()
                    scores      = (proba[:, classes.index(2)] if 2 in classes else
                                   proba[:, classes.index(1)] if 1 in classes else
                                   np.random.rand(len(unrated_df)))
                    unrated_df['predicted_score'] = scores
                    unrated_df = apply_filter(unrated_df.sort_values('predicted_score', ascending=False))
                    st.session_state.recs = unrated_df.reset_index(drop=True)
                    st.session_state.clf  = clf

        st.session_state.must_sees_df = get_must_sees(exclude_ids=list(st.session_state.ratings.keys()))
        st.rerun()

    recs      = st.session_state.recs
    must_sees = st.session_state.must_sees_df

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown('<span class="section-label">The Metropolitan Museum of Art · New York</span>', unsafe_allow_html=True)
    st.markdown('<div class="dp-title">Your Personal Tour</div>', unsafe_allow_html=True)

    liked_ids   = [i for i, v in st.session_state.ratings.items() if v >= 1]
    liked_df    = df[df['id'].isin(liked_ids)]
    liked_depts = liked_df['department'].tolist()
    liked_eras  = liked_df['era'].tolist()

    if not liked_df.empty:
        chips = ""
        for dept, cnt in liked_df['department'].value_counts().head(3).items():
            chips += f'<span class="taste-chip">🏛 {dept} ({cnt})</span>'
        for era, cnt in liked_df['era'].value_counts().head(2).items():
            if era not in ['unknown','Unknown Era']:
                chips += f'<span class="taste-chip">🕰 {str(era).replace("_"," ").title()}</span>'
        if chips:
            st.markdown(f'<div style="margin:0.5rem 0 1.5rem 0;">{chips}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Must-Sees ─────────────────────────────────────────────────────────────
    dp_rule("Non-Negotiables · Must-See Masterpieces")
    st.markdown('<div class="dp-body" style="margin-bottom:1rem;">Iconic works always included — no images, save the surprise for in person.</div>', unsafe_allow_html=True)

    if must_sees is not None and not must_sees.empty:
        ms_c = st.columns(3)
        for i, (_, row) in enumerate(must_sees.head(12).iterrows()):
            with ms_c[i % 3]:
                year = str(row.get('year', row.get('objectDate',''))).strip()
                yr   = f" · {year}" if year and year not in ['nan',''] else ""
                desc = build_desc(row)[:220]
                url  = str(row.get('met_url',''))
                link = f"<a href='{url}' target='_blank'>View on Met ↗</a>" if url not in ['','nan'] else ""
                num  = str(i+1).zfill(2)
                st.markdown(f"""
<div class="dp-ms-card">
  <div class="dp-ms-num">— {num}</div>
  <div class="dp-ms-title">{row['title']}</div>
  <div class="dp-ms-artist">{row['artist']}{yr}</div>
  <div class="dp-ms-desc">{desc}</div>
  <div class="dp-ms-footer">🏛 {row['department']}&nbsp;&nbsp;{link}</div>
</div>""", unsafe_allow_html=True)
                render_flags(row.get('content_flags',''))

    st.markdown("---")

    # ── Personalised Recommendations ─────────────────────────────────────────
    dp_rule("Personalised · Based on Your Taste")

    fc1, fc2 = st.columns([2, 1])
    with fc1:
        all_depts   = ['All departments'] + sorted(recs['department'].unique().tolist())
        chosen_dept = st.selectbox("Department", all_depts, label_visibility="collapsed")
    with fc2:
        min_score = st.slider("Min match", 0, 100, 0, 5, format="%d%%", label_visibility="collapsed")

    disp = recs.copy()
    if chosen_dept != 'All departments': disp = disp[disp['department'] == chosen_dept]
    if min_score > 0:                    disp = disp[disp['predicted_score'] >= min_score/100]
    disp = disp.head(TOP_N_RECS)
    st.markdown(f'<span class="dp-body">{len(disp)} recommendations</span>', unsafe_allow_html=True)
    st.markdown("")

    for i, (_, row) in enumerate(disp.iterrows()):
        score     = row['predicted_score']
        fire      = "🔥 " if score > 0.75 else ""
        has_image = row['image_url'] and str(row['image_url']) not in ['','nan']
        with st.expander(
            f"{fire}{row['title']}  ·  {row['artist']}  ·  {row['department']}  ·  {score:.0%}",
            expanded=(i < 2)):
            ec1, ec2 = st.columns([1, 2], gap="large")
            with ec1:
                if has_image:
                    st.image(row['image_url'], use_column_width=True)
                else:
                    st.markdown('<div style="background:#0F0F13;border:1px solid #1E1C28;border-radius:4px;min-height:250px;display:flex;align-items:center;justify-content:center;"><span style="color:#2A2538;font-size:2rem;">🖼</span></div>', unsafe_allow_html=True)
            with ec2:
                bdg = f'<span class="badge-dim">{row["department"]}</span>'
                if row.get('is_famous'): bdg += '<span class="badge-purple">Master</span>'
                st.markdown(bdg, unsafe_allow_html=True)
                st.markdown("")
                st.markdown(f'<div class="art-title" style="font-size:1.5rem;">{row["title"]}</div>', unsafe_allow_html=True)
                if str(row['artist']) not in ['Unknown Artist','nan','']:
                    st.markdown(f'<div class="art-artist">{row["artist"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="margin:0.75rem 0;"><div class="score-num">{score:.0%}</div><div class="score-lbl">Match Score</div></div>', unsafe_allow_html=True)
                render_desc_box(row)
                render_flags(row.get('content_flags',''))
                reasons = []
                if row['department'] in liked_depts: reasons.append(f"you enjoyed other works from {row['department']}")
                if str(row.get('era','')) not in ['Unknown Era','nan',''] and row.get('era') in liked_eras:
                    reasons.append(f"matches your interest in {str(row['era']).replace('_',' ')} art")
                if row.get('is_famous'): reasons.append("celebrated master artist")
                if score > 0.75:        reasons.append(f"exceptionally high {score:.0%} confidence")
                if reasons:
                    st.markdown(f'<div class="why-box">✦ <strong>Why this?</strong> {" · ".join(reasons)}</div>', unsafe_allow_html=True)
                if row.get('met_url','') and str(row['met_url']) not in ['','nan']:
                    st.markdown(f'<a href="{row["met_url"]}" target="_blank" style="color:#8B7FD4;font-size:0.75rem;">View on Met website ↗</a>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Gallery Roadmap ───────────────────────────────────────────────────────
    dp_rule("Gallery Roadmap · Walking Order")
    st.markdown('<div class="dp-body" style="margin-bottom:1rem;">Your full tour organised by gallery.</div>', unsafe_allow_html=True)

    ms_ids     = must_sees['id'].tolist() if must_sees is not None and not must_sees.empty else []
    csv_ms_ids = [i for i in ms_ids if not str(i).startswith('iconic_')]
    all_ids    = csv_ms_ids + [i for i in disp.head(40)['id'].tolist() if i not in csv_ms_ids]
    roadmap_df = df[df['id'].isin(all_ids) & (df.index < feature_matrix.shape[0])].copy()
    score_map  = dict(zip(disp['id'], disp['predicted_score']))
    roadmap_df['predicted_score'] = roadmap_df['id'].map(score_map).fillna(0.9)

    dept_summary = (roadmap_df.groupby('department')['predicted_score']
                    .agg(['count','mean']).sort_values(['count','mean'], ascending=False))
    total_time  = 0
    total_works = len(ms_ids) + len(disp.head(40))

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
                    if has_img: st.image(row['image_url'], width=65)
                    else: st.markdown('<div style="background:#111;border:1px solid #1E1C28;border-radius:2px;width:65px;height:65px;display:flex;align-items:center;justify-content:center;"><span style="color:#2A2538;">🖼</span></div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"{row['artist']}  ·  {str(row.get('era','')).replace('_',' ').title()}")
                    d = str(row.get('description',''))
                    if d and d not in ['nan',''] and len(d) > 10: st.caption(d[:160]+"...")
                    render_flags(row.get('content_flags',''))
                with rc3:
                    st.markdown(f'<div class="score-num" style="font-size:1.5rem;">{row["predicted_score"]:.0%}</div>', unsafe_allow_html=True)
                    if row.get('met_url','') and str(row['met_url']) not in ['','nan']:
                        st.markdown(f'<a href="{row["met_url"]}" target="_blank" style="color:#8B7FD4;font-size:0.72rem;">Met ↗</a>', unsafe_allow_html=True)
                st.markdown("---")

    # Iconic artworks in roadmap
    iconic_in_tour = [a for a in ICONIC_MET_ARTWORKS if a['id'] in ms_ids]
    if iconic_in_tour:
        with st.expander(f"⭐  Must-See Masterpieces  ·  {len(iconic_in_tour)} works  ·  Always included", expanded=False):
            for art in iconic_in_tour:
                rc1, rc2, rc3 = st.columns([0.5, 3, 0.8])
                with rc1:
                    st.markdown('<div style="background:#13111C;border:1px solid #7B6FD0;border-radius:2px;width:65px;height:65px;display:flex;align-items:center;justify-content:center;"><span style="color:#8B7FD4;">⭐</span></div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"**{art['title']}**")
                    st.caption(f"{art['artist']}  ·  {art.get('year','')}")
                    st.caption(str(art.get('description',''))[:160]+"...")
                with rc3:
                    st.markdown('<span class="badge-purple">Must See</span>', unsafe_allow_html=True)
                    if art.get('met_url',''):
                        st.markdown(f'<a href="{art["met_url"]}" target="_blank" style="color:#8B7FD4;font-size:0.72rem;">Met ↗</a>', unsafe_allow_html=True)
                st.markdown("---")

    # Tour stats
    st.markdown(
        f'<div style="background:#13111C;border:1px solid #23202E;border-radius:6px;'
        f'padding:1.5rem 2rem;margin:1.5rem 0;display:flex;gap:3rem;align-items:center;">'
        f'<div style="text-align:center;"><div class="stat-num">{total_time}</div><div class="stat-lbl">Minutes</div></div>'
        f'<div style="text-align:center;"><div class="stat-num">{total_works}</div><div class="stat-lbl">Artworks</div></div>'
        f'<div style="text-align:center;"><div class="stat-num">{len(dept_summary)}</div><div class="stat-lbl">Galleries</div></div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Refine recommendations", use_container_width=True):
            st.session_state.phase = 'rating'; st.session_state.recs = None; st.rerun()
    with c2:
        if st.button("↺  Start a new tour", use_container_width=True):
            reset_session()
