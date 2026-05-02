"""
🏛️ Met Museum | Personal Tour
Award-winning dark museum UI.
Design inspired by: The Met, MoMA, Tate Modern editorial standards.
Colour palette: Deep charcoal #0D0D0D, warm ivory #F0EAD6, gold #C9A84C
Typography: DM Serif Display (editorial titles) + Inter (clean body)
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
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Met Museum | Personal Tour",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# MASTER CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════
   1. GLOBAL RESET & BASE
═══════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"],
.main, .block-container,
section[data-testid="stSidebar"],
[class*="css"] {
    background-color: #0D0D0D !important;
    color: #E8E0D0 !important;
    font-family: 'Inter', sans-serif !important;
}

.block-container {
    padding: 0 3rem 4rem 3rem !important;
    max-width: 1380px !important;
}

/* All text defaults */
p, span, div, li, label { color: #E8E0D0 !important; }
h1, h2, h3, h4, h5, h6  { color: #F5F0E8 !important; }
a { color: #C9A84C !important; text-decoration: none; }
a:hover { color: #E8C870 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0D0D0D; }
::-webkit-scrollbar-thumb { background: #2A2520; border-radius: 2px; }

/* ═══════════════════════════════════════════
   2. HIDE STREAMLIT CHROME
═══════════════════════════════════════════ */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden !important; height: 0 !important; }

/* ═══════════════════════════════════════════
   3. STREAMLIT COMPONENT OVERRIDES
═══════════════════════════════════════════ */

/* Progress bar */
[data-testid="stProgress"] > div {
    background: #1E1A14 !important;
    border-radius: 1px !important;
    height: 2px !important;
}
[data-testid="stProgress"] > div > div {
    background: #C9A84C !important;
    border-radius: 1px !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #0D0D0D !important;
    background: #C9A84C !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #E8C870 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Secondary / ghost buttons (skip) */
button[data-testid="baseButton-secondary"] {
    color: #E8E0D0 !important;
    background: transparent !important;
    border: 1px solid #2A2520 !important;
}
button[data-testid="baseButton-secondary"]:hover {
    border-color: #C9A84C !important;
    color: #C9A84C !important;
    background: transparent !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #111111 !important;
    border: 1px solid #1E1A14 !important;
    border-radius: 0 !important;
    margin-bottom: 2px !important;
}
[data-testid="stExpander"] summary {
    color: #B8B0A0 !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: #C9A84C !important;
}
[data-testid="stExpander"] > div > div {
    padding: 0 1.25rem 1.25rem 1.25rem !important;
}

/* Selectbox */
[data-testid="stSelectbox"] label { color: #6B6358 !important; font-size: 0.75rem !important; }
[data-testid="stSelectbox"] > div > div {
    background: #111111 !important;
    border: 1px solid #2A2520 !important;
    border-radius: 0 !important;
    color: #E8E0D0 !important;
}

/* Slider */
[data-testid="stSlider"] label { color: #6B6358 !important; font-size: 0.75rem !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #C9A84C !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label p { color: #8A8070 !important; font-size: 0.8rem !important; }

/* Captions */
.stCaption p,
[data-testid="stCaptionContainer"] p {
    color: #6B6358 !important;
    font-size: 0.78rem !important;
}

/* Info box */
[data-testid="stInfo"] {
    background: #130F08 !important;
    border: 1px solid #3A2E18 !important;
    border-radius: 0 !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stInfo"] p { color: #A89060 !important; font-size: 0.82rem !important; }

/* Warning */
[data-testid="stWarning"] {
    background: #130F08 !important;
    border: 1px solid #5A4020 !important;
    color: #C8A060 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] > div {
    background: #0A0A0A !important;
    border-right: 1px solid #1E1A14 !important;
    padding: 2rem 1.5rem !important;
}

/* Horizontal rules */
hr {
    border: none !important;
    border-top: 1px solid #1E1A14 !important;
    margin: 2rem 0 !important;
}

/* ═══════════════════════════════════════════
   4. TYPOGRAPHY SYSTEM
═══════════════════════════════════════════ */

/* Overline / eyebrow label */
.t-overline {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #C9A84C !important;
    display: block;
    margin-bottom: 1rem;
}

/* Hero display title */
.t-display {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.8rem, 5vw, 5rem);
    font-weight: 400;
    line-height: 1.05;
    color: #F5F0E8 !important;
    letter-spacing: -0.02em;
    margin-bottom: 1.5rem;
}

/* Section title */
.t-section {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    line-height: 1.15;
    color: #F5F0E8 !important;
    margin-bottom: 0.5rem;
    letter-spacing: -0.01em;
}

/* Card title */
.t-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.35;
    color: #F0EAD6 !important;
    margin-bottom: 0.3rem;
    letter-spacing: 0;
}

/* Body large */
.t-body-lg {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    line-height: 1.75;
    color: #A89880 !important;
}

/* Body standard */
.t-body {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    line-height: 1.7;
    color: #8A7A68 !important;
}

/* Mono label */
.t-mono {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #5A5040 !important;
    text-transform: uppercase;
}

/* ═══════════════════════════════════════════
   5. LAYOUT PRIMITIVES
═══════════════════════════════════════════ */

/* Full-width top navigation bar */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.5rem 0;
    border-bottom: 1px solid #1E1A14;
    margin-bottom: 0;
}
.nav-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    font-weight: 400;
    color: #F5F0E8 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.nav-meta {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3A3028 !important;
}

/* Hero section container */
.hero-container {
    padding: 5rem 0 4rem 0;
    border-bottom: 1px solid #1E1A14;
    margin-bottom: 4rem;
}

/* Thin gold rule */
.gold-rule {
    height: 1px;
    background: linear-gradient(90deg, #C9A84C 0%, transparent 60%);
    margin: 2.5rem 0;
    border: none;
}

/* Section divider with label */
.section-divider {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin: 3rem 0 2rem 0;
}
.section-divider-text {
    font-family: 'Inter', sans-serif;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #C9A84C !important;
    white-space: nowrap;
    flex-shrink: 0;
}
.section-divider-line {
    flex: 1;
    height: 1px;
    background: #1E1A14;
}

/* ═══════════════════════════════════════════
   6. MUST-SEE CARDS — PHASE 0
═══════════════════════════════════════════ */
.ms-grid-card {
    background: #111111;
    border-top: 1px solid #C9A84C;
    border-left: 1px solid #1A1714;
    border-right: 1px solid #1A1714;
    border-bottom: 1px solid #1A1714;
    padding: 1.5rem;
    height: 100%;
    min-height: 240px;
    transition: background 0.2s ease;
    position: relative;
}
.ms-grid-card:hover {
    background: #161410;
}
.ms-card-index {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    color: #3A3028 !important;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.ms-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: #F0EAD6 !important;
    line-height: 1.35;
    margin-bottom: 0.35rem;
    letter-spacing: -0.01em;
}
.ms-card-artist {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: #C9A84C !important;
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
}
.ms-card-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 300;
    color: #6B6050 !important;
    line-height: 1.65;
    margin-bottom: 1rem;
}
.ms-card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top: 1px solid #1E1A14;
    padding-top: 0.75rem;
    margin-top: auto;
}
.ms-card-dept {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3A3028 !important;
}
.ms-card-link {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    color: #C9A84C !important;
    letter-spacing: 0.08em;
    text-decoration: none;
}

/* ═══════════════════════════════════════════
   7. STATS ROW — PHASE 0
═══════════════════════════════════════════ */
.stats-row {
    display: flex;
    gap: 0;
    border-top: 1px solid #1E1A14;
    border-bottom: 1px solid #1E1A14;
    margin: 3rem 0;
}
.stat-item {
    flex: 1;
    padding: 1.75rem 2rem;
    border-right: 1px solid #1E1A14;
    text-align: center;
}
.stat-item:last-child { border-right: none; }
.stat-number {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #C9A84C !important;
    line-height: 1;
    display: block;
    margin-bottom: 0.4rem;
}
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3A3028 !important;
}

/* ═══════════════════════════════════════════
   8. CTA BOX — PHASE 0
═══════════════════════════════════════════ */
.cta-block {
    border: 1px solid #1E1A14;
    padding: 3rem;
    text-align: center;
    margin: 3rem 0;
    background: #0A0A0A;
}
.cta-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    font-weight: 400;
    color: #F5F0E8 !important;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
}
.cta-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 300;
    color: #6B6050 !important;
    margin-bottom: 0;
    line-height: 1.6;
}

/* ═══════════════════════════════════════════
   9. RATING PHASE — PHASE 1
═══════════════════════════════════════════ */
.rating-progress-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3A3028 !important;
    margin-bottom: 0.5rem;
}
.rating-counter {
    font-family: 'DM Serif Display', serif;
    font-size: 6rem;
    font-weight: 400;
    color: #1A1714 !important;
    line-height: 1;
    letter-spacing: -0.03em;
}
.rating-counter-of {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    font-weight: 400;
    color: #252018 !important;
}
.artwork-title-display {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    color: #F5F0E8 !important;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}
.artwork-artist-display {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: #C9A84C !important;
    letter-spacing: 0.08em;
    margin-bottom: 1.5rem;
}
.artwork-meta-pill {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4A4030 !important;
    border: 1px solid #2A2218;
    padding: 3px 10px;
    margin-right: 6px;
    margin-bottom: 6px;
    border-radius: 0;
}
.rating-prompt {
    font-family: 'DM Serif Display', serif;
    font-style: italic;
    font-size: 1rem;
    color: #5A5040 !important;
    margin: 1.5rem 0 1.25rem 0;
    display: block;
}

/* Image container */
.artwork-image-wrap {
    position: relative;
    background: #0A0A0A;
    border: 1px solid #1E1A14;
}
.no-image-placeholder {
    background: #0A0A0A;
    border: 1px solid #1E1A14;
    min-height: 420px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
}
.no-image-icon {
    font-size: 2.5rem;
    opacity: 0.15;
}
.no-image-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2A2218 !important;
}

/* Description box */
.desc-box {
    background: #0A0A0A;
    border-left: 2px solid #C9A84C;
    padding: 1rem 1.25rem;
    margin: 1.25rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 300;
    color: #7A6A58 !important;
    line-height: 1.75;
}

/* ═══════════════════════════════════════════
   10. RESULTS PHASE — PHASE 2
═══════════════════════════════════════════ */

/* Taste profile chips */
.taste-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 1rem 0 2rem 0; }
.taste-chip {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8A7A68 !important;
    border: 1px solid #2A2218;
    padding: 5px 14px;
    background: #0D0D0D;
}

/* Recommendation card */
.rec-card {
    background: #111111;
    border: 1px solid #1A1714;
    padding: 1.5rem;
    margin-bottom: 2px;
    transition: background 0.15s;
}
.rec-card:hover { background: #141210; }
.rec-score {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    color: #C9A84C !important;
    line-height: 1;
}
.rec-score-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3A3028 !important;
    margin-top: 0.2rem;
}

/* Why box */
.why-box {
    background: #0D0C08;
    border-left: 2px solid #C9A84C;
    padding: 0.85rem 1.1rem;
    margin-top: 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 300;
    color: #8A7A50 !important;
    line-height: 1.65;
}

/* Must-see badge */
.badge-gold {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #C9A84C !important;
    border: 1px solid #C9A84C;
    padding: 3px 10px;
    margin-right: 8px;
}
.badge-dim {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    font-weight: 400;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3A3028 !important;
    border: 1px solid #2A2218;
    padding: 3px 10px;
    margin-right: 8px;
}

/* Tour stats footer */
.tour-stats-bar {
    border-top: 1px solid #1E1A14;
    border-bottom: 1px solid #1E1A14;
    display: flex;
    gap: 0;
    margin: 2.5rem 0;
}
.tour-stat {
    flex: 1;
    padding: 1.5rem 2rem;
    border-right: 1px solid #1E1A14;
    text-align: center;
}
.tour-stat:last-child { border-right: none; }
.tour-stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    font-weight: 400;
    color: #C9A84C !important;
    line-height: 1;
    display: block;
}
.tour-stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3A3028 !important;
    margin-top: 0.3rem;
    display: block;
}

/* Content flags */
.cflag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    padding: 2px 8px;
    background: #180A0A;
    border: 1px solid #4A1818;
    color: #B05050 !important;
    margin-right: 4px;
    text-transform: uppercase;
}

/* Dark must-see cards (results page) */
.dp-ms-card {
    background: #111111;
    border-top: 1px solid #C9A84C;
    border-left: 1px solid #1A1714;
    border-right: 1px solid #1A1714;
    border-bottom: 1px solid #1A1714;
    padding: 1.5rem;
    min-height: 210px;
    margin-bottom: 2px;
}
.dp-ms-idx  { font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:#3A3028 !important; margin-bottom:0.75rem; }
.dp-ms-title { font-family:'DM Serif Display',serif; font-size:1rem; font-weight:400; color:#F0EAD6 !important; line-height:1.35; margin-bottom:0.3rem; }
.dp-ms-artist { font-family:'Inter',sans-serif; font-size:0.72rem; font-weight:500; color:#C9A84C !important; letter-spacing:0.06em; margin-bottom:0.75rem; }
.dp-ms-desc { font-family:'Inter',sans-serif; font-size:0.77rem; font-weight:300; color:#5A5040 !important; line-height:1.65; margin-bottom:0.75rem; }
.dp-ms-footer { font-size:0.65rem; color:#3A3028 !important; border-top:1px solid #1A1714; padding-top:0.6rem; }
.dp-ms-footer a { color:#C9A84C !important; }

/* Roadmap image cell */
.roadmap-thumb {
    width: 64px;
    height: 64px;
    object-fit: cover;
    border: 1px solid #1E1A14;
}
.roadmap-thumb-empty {
    width: 64px;
    height: 64px;
    background: #0A0A0A;
    border: 1px solid #1E1A14;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #2A2218;
    font-size: 1rem;
}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
RATING_TARGET = 20
TOP_N_RECS    = 40
MUST_SEE_N    = 12

FAMOUS_ARTIST_NAMES = [
    "van gogh","rembrandt","hokusai","degas","eakins","whistler","hiroshige",
    "pissarro","seurat","goya","vermeer","botticelli","el greco","sargent",
    "rubens","velázquez","velazquez","homer","constable","cézanne","cezanne",
    "raphael","delacroix","caravaggio","corot","bruegel","leutze","pollock","david",
    "monet","picasso","renoir","manet","matisse","rothko","warhol","lichtenstein",
    "cassatt","o'keeffe","hopper","chagall","dali","dalí","kandinsky","klee",
    "titian","gauguin","turner",
]

ICONIC_MET_ARTWORKS = [
    {"id":"iconic_11417","title":"Washington Crossing the Delaware","artist":"Emanuel Leutze","year":"1851","department":"The American Wing","description":"One of the most iconic images in American history. This enormous canvas (12 × 21 ft) depicts Washington's daring crossing of the icy Delaware on December 25–26, 1776 — a pivotal moment in the Revolution. Gallery 760.","met_url":"https://www.metmuseum.org/art/collection/search/11417","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"oil_painting","culture":"American","medium":"Oil on canvas"},
    {"id":"iconic_12127","title":"Madame X","artist":"John Singer Sargent","year":"1883–84","department":"The American Wing","description":"The most scandalous portrait of its era. Sargent's daring portrayal caused a sensation at the 1884 Paris Salon. The contrast of pale skin against the black gown became a defining image of 19th-century portraiture. Gallery 771.","met_url":"https://www.metmuseum.org/art/collection/search/12127","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"oil_painting","culture":"American","medium":"Oil on canvas"},
    {"id":"iconic_436532","title":"Self-Portrait with a Straw Hat","artist":"Vincent van Gogh","year":"1887","department":"European Paintings","description":"Painted during Van Gogh's transformative Paris years. Vivid brushwork and dazzling colour contrasts — blues, oranges, yellows — mark his break from the sombre Dutch palette. One of only two Van Goghs at the Met. Gallery 825.","met_url":"https://www.metmuseum.org/art/collection/search/436532","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"oil_painting","culture":"Dutch","medium":"Oil on canvas"},
    {"id":"iconic_437394","title":"Aristotle with a Bust of Homer","artist":"Rembrandt van Rijn","year":"1653","department":"European Paintings","description":"A masterpiece of psychological depth and dramatic light. The Met paid $2.3 million for it in 1961 — then the highest price ever paid for a painting. Rembrandt at his absolute peak. Gallery 964.","met_url":"https://www.metmuseum.org/art/collection/search/437394","content_flags":"","is_famous":True,"era":"baroque_rococo","style":"oil_painting","culture":"Dutch","medium":"Oil on canvas"},
    {"id":"iconic_437870","title":"Young Woman with a Water Pitcher","artist":"Johannes Vermeer","year":"c. 1662","department":"European Paintings","description":"A serene domestic scene bathed in Vermeer's signature cool northern light. Only 34 Vermeers are known to exist worldwide — making this one of the Met's most precious possessions. Gallery 964.","met_url":"https://www.metmuseum.org/art/collection/search/437870","content_flags":"","is_famous":True,"era":"baroque_rococo","style":"oil_painting","culture":"Dutch","medium":"Oil on canvas"},
    {"id":"iconic_437130","title":"Bridge over a Pond of Water Lilies","artist":"Claude Monet","year":"1899","department":"European Paintings","description":"Painted in Monet's garden at Giverny, capturing the Japanese footbridge reflected in the lily pond he designed himself. A cornerstone of Impressionism and a rare Monet at the Met. Gallery 819.","met_url":"https://www.metmuseum.org/art/collection/search/437130","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"oil_painting","culture":"French","medium":"Oil on canvas"},
    {"id":"iconic_436105","title":"The Death of Socrates","artist":"Jacques-Louis David","year":"1787","department":"European Paintings","description":"Socrates calmly accepts death, reaching for the hemlock cup while followers grieve. The defining image of Enlightenment idealism. Crisp neoclassical style at its most powerful. Gallery 614.","met_url":"https://www.metmuseum.org/art/collection/search/436105","content_flags":"","is_famous":True,"era":"baroque_rococo","style":"oil_painting","culture":"French","medium":"Oil on canvas"},
    {"id":"iconic_488978","title":"Autumn Rhythm (Number 30)","artist":"Jackson Pollock","year":"1950","department":"Modern and Contemporary Art","description":"Pollock created this by dripping paint onto canvas on the floor — his revolutionary drip technique. The gestural sweep of black, white, and brown conveys raw, untamed energy. A defining work of Abstract Expressionism. Gallery 919.","met_url":"https://www.metmuseum.org/art/collection/search/488978","content_flags":"","is_famous":True,"era":"early_modern","style":"oil_painting","culture":"American","medium":"Enamel on canvas"},
    {"id":"iconic_435809","title":"The Harvesters","artist":"Pieter Bruegel the Elder","year":"1565","department":"European Paintings","description":"Part of a series on the months of the year — this August scene shows peasants harvesting wheat under a blazing summer sky. One of the greatest landscape paintings ever made. Gallery 636.","met_url":"https://www.metmuseum.org/art/collection/search/435809","content_flags":"","is_famous":True,"era":"renaissance","style":"oil_painting","culture":"Netherlandish","medium":"Oil on wood"},
    {"id":"iconic_547802","title":"The Little Fourteen-Year-Old Dancer","artist":"Edgar Degas","year":"1922 (cast)","department":"European Sculpture","description":"The only sculpture Degas ever exhibited publicly — originally shown with real fabric: tutu, hair ribbon, satin shoes. Critics were shocked by its unflinching realism. Today's bronze casts preserve his radical vision. Gallery 800.","met_url":"https://www.metmuseum.org/art/collection/search/547802","content_flags":"","is_famous":True,"era":"nineteenth_century","style":"sculpture","culture":"French","medium":"Bronze, fabric, hair"},
    {"id":"iconic_544039","title":"Sphinx of Hatshepsut","artist":"Ancient Egyptian","year":"c. 1479–1458 BCE","department":"Egyptian Art","description":"This granite sphinx bears the face of Hatshepsut — one of ancient Egypt's most powerful female pharaohs. The lion body symbolises royal power; the human face, divine wisdom. Gallery 115.","met_url":"https://www.metmuseum.org/art/collection/search/544039","content_flags":"","is_famous":True,"era":"ancient","style":"sculpture","culture":"Egyptian","medium":"Granite"},
    {"id":"iconic_317385","title":"The Temple of Dendur","artist":"Ancient Egyptian","year":"c. 15 BCE","department":"Egyptian Art","description":"An entire ancient Egyptian temple — gifted to the US by Egypt in 1965 and reassembled stone by stone inside the Met. It stands in a vast sun-lit gallery with a reflecting pool. One of the most awe-inspiring rooms in any museum in the world. Gallery 131.","met_url":"https://www.metmuseum.org/art/collection/search/317385","content_flags":"","is_famous":True,"era":"ancient","style":"sculpture","culture":"Egyptian","medium":"Aeolian sandstone"},
]

ICONIC_IDS = {art["id"] for art in ICONIC_MET_ARTWORKS}

MET_NOTE = (
    "Van Gogh's Starry Night and Monet's Water Lilies series are at MoMA. "
    "The Met holds Van Gogh's Self-Portrait and Monet's Bridge — both above. "
    "The Met's strengths: Rembrandt, Vermeer, Hokusai, Degas, Egyptian antiquities, American masters."
)

DEPT_TIME = {
    "European Paintings":45,"American Paintings and Sculpture":35,
    "Modern and Contemporary Art":40,"Asian Art":30,"Egyptian Art":25,
    "Greek and Roman Art":30,"Islamic Art":25,"The American Wing":35,
    "Robert Lehman Collection":20,"default":15,
}

CONTENT_FLAGS_DEF = {"nudity":"🔞 Nudity","violence":"⚔️ Violence","religious":"✝️ Religious"}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('met_artworks_clean.csv')
    renames = {'objectID':'id','artistDisplayName':'artist',
                'primaryImageSmall':'image_url','objectURL':'met_url',
                'isHighlight':'is_highlight','is_famous_artist':'is_famous'}
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
        df['is_famous'] = df['artist'].apply(lambda a: any(n in str(a).lower() for n in FAMOUS_ARTIST_NAMES))
    else:
        df['is_famous'] = df['is_famous'].fillna(False).astype(bool)
        df['is_famous'] = df['is_famous'] | df['artist'].apply(lambda a: any(n in str(a).lower() for n in FAMOUS_ARTIST_NAMES))
    return df.reset_index(drop=True)


@st.cache_resource
def load_features():
    with open('feature_matrix.pkl', 'rb') as f:
        return pickle.load(f)


try:
    df             = load_data()
    feature_matrix = load_features()
    # CRITICAL: truncate df to feature_matrix size (fixes IndexError)
    if len(df) > feature_matrix.shape[0]:
        df = df.iloc[:feature_matrix.shape[0]].reset_index(drop=True)
    DATA_LOADED = True
except FileNotFoundError:
    DATA_LOADED = False


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
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
# HELPERS
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
    if not flags_str or str(flags_str) in ['', 'nan']:
        return
    for flag in str(flags_str).split('|'):
        flag = flag.strip()
        if flag in CONTENT_FLAGS_DEF:
            st.markdown(f'<span class="cflag">{CONTENT_FLAGS_DEF[flag]}</span>', unsafe_allow_html=True)

def build_desc(row):
    desc = str(row.get('description', ''))
    if len(desc) > 30 and desc not in ['nan', '']:
        return desc[:420]
    parts = []
    if str(row.get('medium',   '')) not in ['','nan','unknown']: parts.append(f"Medium: {row['medium']}")
    if str(row.get('culture',  '')) not in ['','nan','unknown']: parts.append(f"Culture: {row['culture']}")
    if str(row.get('era',      '')) not in ['','nan','Unknown Era','unknown']:
        parts.append(f"Era: {str(row['era']).replace('_',' ').title()}")
    if str(row.get('tags',     '')) not in ['','nan']:           parts.append(f"Tags: {str(row['tags'])[:80]}")
    return "  ·  ".join(parts) if parts else f"Part of the Met's {row.get('department','collection')}."

def render_desc_box(row):
    st.markdown(f'<div class="desc-box">{build_desc(row)}</div>', unsafe_allow_html=True)

def section_div(label):
    st.markdown(
        f'<div class="section-divider">'
        f'<span class="section-divider-text">{label}</span>'
        f'<div class="section-divider-line"></div>'
        f'</div>', unsafe_allow_html=True)

def get_must_sees(exclude_ids=None):
    exclude_ids = set(str(i) for i in (exclude_ids or []))
    cf          = get_cf()
    iconic_rows = [a for a in ICONIC_MET_ARTWORKS
                   if a['id'] not in exclude_ids
                   and not (cf and any(f in str(a.get('content_flags','')) for f in cf))]
    iconic_df   = pd.DataFrame(iconic_rows) if iconic_rows else pd.DataFrame()
    already     = set(iconic_df['id'].tolist()) if not iconic_df.empty else set()
    already.update(exclude_ids)
    pool        = apply_filter(df[~df['id'].isin(already)])
    famous      = pool[pool['is_famous'] == True]
    hilights    = pool[(pool['is_highlight'] == True) & (~pool['is_famous'])]
    combined    = pd.concat([famous, hilights]).drop_duplicates(subset=['id'])
    remaining   = max(0, MUST_SEE_N - len(iconic_df))
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
    elif csv_rows:            return pd.DataFrame(csv_rows)
    return pool.head(MUST_SEE_N)

def build_rating_queue(exclude_ids=None):
    exclude_ids = set(str(i) for i in (exclude_ids or []))
    exclude_ids.update(ICONIC_IDS)
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
# NAVIGATION BAR (all phases)
# ══════════════════════════════════════════════════════════════════════════════
phase = st.session_state.get('phase', 'must_sees')
n_rated = len(st.session_state.ratings)

st.markdown(
    f'<div class="nav-bar">'
    f'<span class="nav-logo">🏛️ &nbsp; The Met &nbsp;·&nbsp; Personal Tour</span>'
    f'<span class="nav-meta">{n_rated} / {RATING_TARGET} rated</span>'
    f'</div>',
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<span class="t-overline">🏛️ Met · Personal Tour</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.progress(min(n_rated / RATING_TARGET, 1.0))
    st.caption(f"{n_rated} of {RATING_TARGET} artworks rated")
    if n_rated > 0:
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
    st.markdown("---")
    st.caption("Powered by ML · Met Museum Open Access API")


# ══════════════════════════════════════════════════════════════════════════════
# DATA NOT LOADED
# ══════════════════════════════════════════════════════════════════════════════
if not DATA_LOADED:
    st.markdown('<div class="t-display" style="margin-top:3rem;">Data files not found.</div>', unsafe_allow_html=True)
    st.error("Ensure `met_artworks_clean.csv`, `feature_matrix.pkl`, and `tfidf_vectorizer.pkl` are in the same folder.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if phase == 'must_sees':

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1], gap="large")
    with col_left:
        st.markdown('<span class="t-overline">The Metropolitan Museum of Art · New York</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="t-display">Your Personal<br>Museum Tour</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="t-body-lg">An AI that learns your taste in 20 artworks, then builds you '
            'a ranked, personalised tour of the Met — with a gallery-by-gallery roadmap '
            'and a list of iconic works you cannot leave without seeing.</div>',
            unsafe_allow_html=True)

    with col_right:
        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="stats-row" style="flex-direction:column;border:none;margin:0;">'
            '<div class="stat-item" style="border-right:none;border-bottom:1px solid #1E1A14;text-align:right;padding:1.2rem 0;">'
            '<span class="stat-number">2,022</span>'
            '<span class="stat-label">Artworks catalogued</span>'
            '</div>'
            '<div class="stat-item" style="border-right:none;border-bottom:1px solid #1E1A14;text-align:right;padding:1.2rem 0;">'
            '<span class="stat-number">12</span>'
            '<span class="stat-label">Must-see masterpieces</span>'
            '</div>'
            '<div class="stat-item" style="border-right:none;text-align:right;padding:1.2rem 0;">'
            '<span class="stat-number">20</span>'
            '<span class="stat-label">Ratings to personalise</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True)

    # Gold rule
    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # Content preferences
    pref_c1, pref_c2, pref_c3 = st.columns([3, 1, 1])
    with pref_c1:
        st.markdown('<span class="t-mono">Set your content preferences</span>', unsafe_allow_html=True)
    with pref_c2:
        st.session_state.hide_nudity   = st.checkbox("Exclude nudity",   value=st.session_state.get('hide_nudity', False), key="p0n")
    with pref_c3:
        st.session_state.hide_violence = st.checkbox("Exclude violence",  value=st.session_state.get('hide_violence', False), key="p0v")

    # ── Must-Sees ─────────────────────────────────────────────────────────────
    section_div("Non-Negotiables · Must-See Masterpieces")

    st.markdown(
        '<div class="t-body" style="margin-bottom:1.5rem;">'
        'These iconic works are guaranteed in your tour — curated from the Met\'s own highlights '
        'and art history\'s most celebrated masters. We\'ve withheld images intentionally: '
        'experience them in person for the first time.</div>',
        unsafe_allow_html=True)

    # Amber note
    st.markdown(
        f'<div style="background:#0D0900;border:1px solid #3A2E10;padding:0.9rem 1.2rem;'
        f'margin-bottom:1.5rem;font-family:Inter,sans-serif;font-size:0.8rem;'
        f'color:#8A7040;line-height:1.6;">'
        f'<strong style="color:#C9A84C;">Note —</strong> {MET_NOTE}</div>',
        unsafe_allow_html=True)

    if st.session_state.must_sees_df is None:
        st.session_state.must_sees_df = get_must_sees()
    must_sees = st.session_state.must_sees_df

    if not must_sees.empty:
        cols = st.columns(3, gap="small")
        for i, (_, row) in enumerate(must_sees.iterrows()):
            with cols[i % 3]:
                year = str(row.get('year', row.get('objectDate', ''))).strip()
                yr   = f", {year}" if year and year not in ['nan', ''] else ""
                desc = build_desc(row)[:280]
                url  = str(row.get('met_url', ''))
                link = f'<a class="ms-card-link" href="{url}" target="_blank">View on Met ↗</a>' if url not in ['', 'nan'] else ''
                idx  = str(i + 1).zfill(2)
                st.markdown(f"""
<div class="ms-grid-card">
  <div class="ms-card-index">No. {idx}</div>
  <div class="ms-card-title">{row['title']}</div>
  <div class="ms-card-artist">{row['artist']}{yr}</div>
  <div class="ms-card-desc">{desc}</div>
  <div class="ms-card-footer">
    <span class="ms-card-dept">{row['department']}</span>
    {link}
  </div>
</div>""", unsafe_allow_html=True)
                render_flags(row.get('content_flags', ''))

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
<div class="cta-block">
  <div class="cta-title">Now, let us learn your taste</div>
  <div class="cta-subtitle">Rate 20 artworks — Love, Like, or Skip.<br>
  We'll train a model on your choices and build a personalised tour in seconds.</div>
</div>""", unsafe_allow_html=True)
        if st.button("Begin Taste Profile  →", use_container_width=True):
            ms_ids = must_sees['id'].tolist() if not must_sees.empty else []
            st.session_state.phase        = 'rating'
            st.session_state.rating_queue = build_rating_queue(exclude_ids=ms_ids)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — RATING
# ══════════════════════════════════════════════════════════════════════════════
elif phase == 'rating':

    if n_rated >= RATING_TARGET:
        st.session_state.phase = 'results'
        st.rerun()

    # Progress
    st.markdown(f'<div class="rating-progress-text">Rating {n_rated + 1} of {RATING_TARGET}</div>', unsafe_allow_html=True)
    st.progress(n_rated / RATING_TARGET)
    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    # Get artwork
    rated_ids = set(st.session_state.ratings.keys())
    remaining = [i for i in st.session_state.rating_queue if i not in rated_ids]
    if not remaining:
        st.session_state.phase = 'results'; st.rerun()

    current_id = remaining[0]
    matches    = df[df['id'] == current_id]
    if matches.empty:
        st.session_state.ratings[current_id] = -1; st.rerun()

    artwork   = matches.iloc[0]
    has_image = artwork['image_url'] and str(artwork['image_url']) not in ['', 'nan']

    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        if has_image:
            st.markdown('<div class="artwork-image-wrap">', unsafe_allow_html=True)
            st.image(artwork['image_url'], use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="no-image-placeholder">'
                '<div class="no-image-icon">🖼</div>'
                '<div class="no-image-text">Image rights restricted</div>'
                '</div>', unsafe_allow_html=True)

    with col_info:
        # Counter watermark
        st.markdown(
            f'<div class="rating-counter">{n_rated + 1}'
            f'<span class="rating-counter-of">/{RATING_TARGET}</span></div>',
            unsafe_allow_html=True)

        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

        # Department + badges
        bdg = f'<span class="badge-dim">{artwork["department"]}</span>'
        if artwork.get('is_famous'):    bdg += '<span class="badge-gold">Master</span>'
        if artwork.get('is_highlight'): bdg += '<span class="badge-dim">Met Pick</span>'
        st.markdown(bdg, unsafe_allow_html=True)
        st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

        # Title + artist
        st.markdown(f'<div class="artwork-title-display">{artwork["title"]}</div>', unsafe_allow_html=True)
        al = str(artwork['artist'])
        if str(artwork.get('artistNationality', '')) not in ['', 'nan']: al += f"  ·  {artwork['artistNationality']}"
        if str(artwork.get('objectDate',        '')) not in ['', 'nan']: al += f"  ·  {artwork['objectDate']}"
        st.markdown(f'<div class="artwork-artist-display">{al}</div>', unsafe_allow_html=True)

        # Meta pills
        pills = ''
        if str(artwork.get('era',    '')) not in ['', 'nan', 'Unknown Era', 'unknown']:
            pills += f'<span class="artwork-meta-pill">{str(artwork["era"]).replace("_"," ").title()}</span>'
        if str(artwork.get('style',  '')) not in ['', 'nan', 'other', 'unknown']:
            pills += f'<span class="artwork-meta-pill">{str(artwork["style"]).replace("_"," ").title()}</span>'
        if str(artwork.get('culture','')) not in ['', 'nan', 'unknown']:
            pills += f'<span class="artwork-meta-pill">{artwork["culture"]}</span>'
        if pills:
            st.markdown(f'<div style="margin-bottom:1rem;">{pills}</div>', unsafe_allow_html=True)

        # Description
        render_desc_box(artwork)
        render_flags(artwork.get('content_flags', ''))

        if artwork.get('met_url', '') and str(artwork['met_url']) not in ['', 'nan']:
            st.markdown(
                f'<a href="{artwork["met_url"]}" target="_blank" '
                f'style="font-size:0.72rem;letter-spacing:0.08em;color:#C9A84C;">'
                f'View full record on Met website ↗</a>',
                unsafe_allow_html=True)

        # Rating prompt
        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown('<span class="rating-prompt">Would you stop and look at this?</span>', unsafe_allow_html=True)

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
        st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
        if left > 0:
            st.markdown(f'<span class="t-mono">{left} more to go</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#C9A84C;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;font-weight:600;">Last one — your tour is almost ready</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif phase == 'results':

    # ── Train model ──────────────────────────────────────────────────────────
    if st.session_state.recs is None:
        with st.spinner("Analysing your taste and curating your tour..."):
            valid = {k: v for k, v in st.session_state.ratings.items()
                     if v >= 0 and not str(k).startswith('iconic_')}
            rated_ids = list(valid.keys())
            labels    = list(valid.values())

            if len(set(labels)) < 2:
                fb = apply_filter(df[~df['id'].isin(rated_ids)]).head(TOP_N_RECS).copy()
                fb['predicted_score'] = 0.5
                st.session_state.recs = fb
                st.warning("Try mixing Love, Like, and Skip for better recommendations.")
            else:
                rated_indices, valid_ids, valid_labels = [], [], []
                for oid, label in zip(rated_ids, labels):
                    match = df[df['id'] == oid]
                    if not match.empty:
                        idx = match.index[0]
                        if idx < feature_matrix.shape[0]:
                            rated_indices.append(idx)
                            valid_ids.append(oid)
                            valid_labels.append(label)

                if len(set(valid_labels)) < 2:
                    fb = apply_filter(df[~df['id'].isin(valid_ids)]).head(TOP_N_RECS).copy()
                    fb['predicted_score'] = 0.5
                    st.session_state.recs = fb
                else:
                    clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                                  class_weight='balanced', n_jobs=-1)
                    clf.fit(feature_matrix[rated_indices], np.array(valid_labels))

                    unrated_df  = df[~df['id'].isin(valid_ids)].copy()
                    unrated_df  = unrated_df[unrated_df.index < feature_matrix.shape[0]]
                    unrated_idx = unrated_df.index.tolist()
                    proba       = clf.predict_proba(feature_matrix[unrated_idx])
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
    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<span class="t-overline">Your Personalised Tour</span>', unsafe_allow_html=True)
    st.markdown('<div class="t-display">The Met — Curated for You</div>', unsafe_allow_html=True)

    # Taste chips
    liked_ids   = [i for i, v in st.session_state.ratings.items() if v >= 1]
    liked_df    = df[df['id'].isin(liked_ids)]
    liked_depts = liked_df['department'].tolist()
    liked_eras  = liked_df['era'].tolist()

    if not liked_df.empty:
        chips = '<div class="taste-row">'
        for dept, cnt in liked_df['department'].value_counts().head(3).items():
            chips += f'<span class="taste-chip">🏛 {dept}</span>'
        for era, cnt in liked_df['era'].value_counts().head(2).items():
            if era not in ['unknown', 'Unknown Era']:
                chips += f'<span class="taste-chip">🕰 {str(era).replace("_"," ").title()}</span>'
        if 'style' in liked_df.columns:
            for s, cnt in liked_df['style'].value_counts().head(1).items():
                if s and s not in ['other', 'unknown', '']:
                    chips += f'<span class="taste-chip">🖌 {str(s).replace("_"," ").title()}</span>'
        chips += '</div>'
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # ── Must-Sees ─────────────────────────────────────────────────────────────
    section_div("Non-Negotiables · Always Included")
    st.markdown('<div class="t-body" style="margin-bottom:1.5rem;">Iconic works guaranteed in your tour. No images — discover them in person.</div>', unsafe_allow_html=True)

    if must_sees is not None and not must_sees.empty:
        ms_c = st.columns(3, gap="small")
        for i, (_, row) in enumerate(must_sees.head(12).iterrows()):
            with ms_c[i % 3]:
                year = str(row.get('year', row.get('objectDate', ''))).strip()
                yr   = f", {year}" if year and year not in ['nan', ''] else ""
                desc = build_desc(row)[:240]
                url  = str(row.get('met_url', ''))
                link = f'<a class="dp-ms-footer" style="color:#C9A84C;" href="{url}" target="_blank">Met ↗</a>' if url not in ['', 'nan'] else ''
                idx  = str(i + 1).zfill(2)
                st.markdown(f"""
<div class="dp-ms-card">
  <div class="dp-ms-idx">No. {idx}</div>
  <div class="dp-ms-title">{row['title']}</div>
  <div class="dp-ms-artist">{row['artist']}{yr}</div>
  <div class="dp-ms-desc">{desc}</div>
  <div class="dp-ms-footer">{row['department']}&nbsp;&nbsp;{link}</div>
</div>""", unsafe_allow_html=True)
                render_flags(row.get('content_flags', ''))

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    section_div("Personalised Recommendations · Ranked by Predicted Enjoyment")

    fc1, fc2 = st.columns([2, 1])
    with fc1:
        all_depts   = ['All departments'] + sorted(recs['department'].unique().tolist())
        chosen_dept = st.selectbox("Filter by department", all_depts, label_visibility="collapsed")
    with fc2:
        min_score = st.slider("Min match score", 0, 100, 0, 5, format="%d%%", label_visibility="collapsed")

    disp = recs.copy()
    if chosen_dept != 'All departments': disp = disp[disp['department'] == chosen_dept]
    if min_score > 0:                    disp = disp[disp['predicted_score'] >= min_score / 100]
    disp = disp.head(TOP_N_RECS)

    st.markdown(f'<span class="t-mono" style="margin-bottom:1rem;display:block;">{len(disp)} works recommended</span>', unsafe_allow_html=True)

    for i, (_, row) in enumerate(disp.iterrows()):
        score     = row['predicted_score']
        fire      = "🔥 " if score > 0.75 else ""
        has_image = row['image_url'] and str(row['image_url']) not in ['', 'nan']

        with st.expander(
            f"{fire}{row['title']}  ·  {row['artist']}  ·  {row['department']}  ·  {score:.0%} match",
            expanded=(i < 2)):

            ec1, ec2 = st.columns([1, 2], gap="large")

            with ec1:
                if has_image:
                    st.image(row['image_url'], use_column_width=True)
                else:
                    st.markdown(
                        '<div class="no-image-placeholder" style="min-height:220px;">'
                        '<div class="no-image-icon">🖼</div>'
                        '<div class="no-image-text">Image restricted</div></div>',
                        unsafe_allow_html=True)

            with ec2:
                bdg = f'<span class="badge-dim">{row["department"]}</span>'
                if row.get('is_famous'): bdg += '<span class="badge-gold">Master</span>'
                st.markdown(bdg, unsafe_allow_html=True)
                st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

                st.markdown(f'<div class="artwork-title-display" style="font-size:1.5rem;">{row["title"]}</div>', unsafe_allow_html=True)
                if str(row['artist']) not in ['Unknown Artist', 'nan', '']:
                    st.markdown(f'<div class="artwork-artist-display">{row["artist"]}</div>', unsafe_allow_html=True)

                # Score
                st.markdown(
                    f'<div style="margin:1rem 0;">'
                    f'<span class="rec-score">{score:.0%}</span>'
                    f'<div class="rec-score-label">Match Score</div>'
                    f'</div>',
                    unsafe_allow_html=True)

                render_desc_box(row)
                render_flags(row.get('content_flags', ''))

                # Why
                reasons = []
                if row['department'] in liked_depts: reasons.append(f"you enjoyed other works from {row['department']}")
                if str(row.get('era','')) not in ['Unknown Era','nan',''] and row.get('era') in liked_eras:
                    reasons.append(f"matches your interest in {str(row['era']).replace('_',' ')} art")
                if row.get('is_famous'): reasons.append("created by a celebrated master")
                if score > 0.75:        reasons.append(f"exceptionally high model confidence")
                if reasons:
                    st.markdown(
                        f'<div class="why-box">✦ <strong style="color:#C9A84C;">Why this work?</strong>  '
                        f'{" · ".join(reasons)}</div>',
                        unsafe_allow_html=True)

                if row.get('met_url', '') and str(row['met_url']) not in ['', 'nan']:
                    st.markdown(
                        f'<a href="{row["met_url"]}" target="_blank" '
                        f'style="font-size:0.72rem;color:#C9A84C;letter-spacing:0.08em;">'
                        f'View on Met website ↗</a>',
                        unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # ── Gallery Roadmap ───────────────────────────────────────────────────────
    section_div("Gallery Roadmap · Visit in This Order")
    st.markdown('<div class="t-body" style="margin-bottom:1.5rem;">Your complete tour, gallery by gallery.</div>', unsafe_allow_html=True)

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

        with st.expander(
            f"🏛  {dept_name}  ·  {n_works} works  ·  ~{est_mins} min",
            expanded=False):
            for _, row in group.iterrows():
                rc1, rc2, rc3 = st.columns([0.5, 3.5, 0.8])
                has_img = row['image_url'] and str(row['image_url']) not in ['', 'nan']
                with rc1:
                    if has_img:
                        st.image(row['image_url'], width=64)
                    else:
                        st.markdown(
                            '<div class="roadmap-thumb-empty">🖼</div>',
                            unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"{row['artist']}  ·  {str(row.get('era','')).replace('_',' ').title()}")
                    d = str(row.get('description', ''))
                    if d and d not in ['nan', ''] and len(d) > 10:
                        st.caption(d[:180] + "...")
                    render_flags(row.get('content_flags', ''))
                with rc3:
                    st.markdown(
                        f'<span class="rec-score" style="font-size:1.4rem;">{row["predicted_score"]:.0%}</span>',
                        unsafe_allow_html=True)
                    if row.get('met_url', '') and str(row['met_url']) not in ['', 'nan']:
                        st.markdown(
                            f'<a href="{row["met_url"]}" target="_blank" '
                            f'style="color:#C9A84C;font-size:0.72rem;">Met ↗</a>',
                            unsafe_allow_html=True)
                st.markdown("---")

    # Iconic works in roadmap
    iconic_in_tour = [a for a in ICONIC_MET_ARTWORKS if a['id'] in ms_ids]
    if iconic_in_tour:
        with st.expander(
            f"⭐  Must-See Masterpieces  ·  {len(iconic_in_tour)} works  ·  Always included",
            expanded=False):
            for art in iconic_in_tour:
                rc1, rc2, rc3 = st.columns([0.5, 3.5, 0.8])
                with rc1:
                    st.markdown(
                        '<div class="roadmap-thumb-empty" style="border-color:#C9A84C;color:#C9A84C;">⭐</div>',
                        unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"**{art['title']}**")
                    st.caption(f"{art['artist']}  ·  {art.get('year','')}")
                    st.caption(str(art.get('description',''))[:180] + "...")
                with rc3:
                    st.markdown('<span class="badge-gold" style="font-size:0.55rem;">Must See</span>', unsafe_allow_html=True)
                    if art.get('met_url', ''):
                        st.markdown(
                            f'<a href="{art["met_url"]}" target="_blank" '
                            f'style="color:#C9A84C;font-size:0.72rem;">Met ↗</a>',
                            unsafe_allow_html=True)
                st.markdown("---")

    # Tour stats
    st.markdown(
        f'<div class="tour-stats-bar">'
        f'<div class="tour-stat"><span class="tour-stat-num">{total_time}</span><span class="tour-stat-lbl">Minutes estimated</span></div>'
        f'<div class="tour-stat"><span class="tour-stat-num">{total_works}</span><span class="tour-stat-lbl">Artworks in tour</span></div>'
        f'<div class="tour-stat"><span class="tour-stat-num">{len(dept_summary)}</span><span class="tour-stat-lbl">Galleries to visit</span></div>'
        f'</div>',
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Refine my recommendations", use_container_width=True):
            st.session_state.phase = 'rating'
            st.session_state.recs  = None
            st.rerun()
    with c2:
        if st.button("↺  Start a completely new tour", use_container_width=True):
            reset_session()
