# 🏛️ Met Museum Personalised Art Tour Generator

A machine learning web app that learns your art taste in real time and generates a personalised tour of the Metropolitan Museum of Art — with SHAP-powered explanations for every recommendation.

**Live demo:** https://met-art-recommender-ymje5stff6zbqmqpwexsxb.streamlit.app/

---

## What It Does

1. **Rate 20 artworks** — You see artworks one at a time and click Love / Like / Skip
2. **ML learns your taste** — A Random Forest trains on your ratings in real time
3. **Get a personalised tour** — The app ranks all 2,000+ artworks by predicted enjoyment and groups them into a room-by-room walking tour
4. **Understand why** — Every recommendation comes with a SHAP explanation: *"Recommended because: Baroque era · Oil on canvas · Dutch culture"*

---

## Demo

| Rating Phase | Recommendations |
|---|---|
| Rate artworks one by one | See your personalised tour ranked by predicted enjoyment |
| Love / Like / Skip | SHAP explains each recommendation |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Model | scikit-learn RandomForestClassifier |
| Explainability | SHAP (TreeExplainer) |
| Feature Engineering | TF-IDF · One-hot encoding · Era bucketing |
| Data Source | Met Museum Open Access API |
| Language | Python 3 |
| Deployment | Streamlit Cloud |

---

## ML Approach

**Content-based filtering** — each artwork is encoded as a 315-dimensional feature vector:

- **One-hot encoding** (215 features) — department, culture, era
- **TF-IDF** (100 features) — tags and medium text

**Training:** User ratings become labels (Love=2, Like=1, Skip=0). A Random Forest trains on the 20 rated artworks and predicts scores for all 2,000+ unrated ones.

**Explainability:** SHAP TreeExplainer computes feature contributions for each recommendation, translated into plain English.

---

## Dataset

- **Source:** Met Museum Open Access API (collectionapi.metmuseum.org)
- **Size:** 2,022 artworks with images
- **Departments:** 19 (European Paintings, Asian Art, Egyptian Art, Greek and Roman Art, Islamic Art, Photographs, Drawings and Prints, and more)
- **Features per artwork:** title, artist, nationality, medium, culture, period, era, tags, department, image URL

---

## Project Structure

```
met-art-recommender/
│
├── met_app.py                    # Streamlit app (Phase 1: rating, Phase 2: recommendations)
├── met_step1_collect_data.py     # Data collection script (Met Museum API)
├── met_step2_explore.ipynb       # Feature engineering notebook
│
├── met_artworks_clean.csv        # Cleaned artwork dataset (2,022 rows)
├── feature_matrix.pkl            # Precomputed feature matrix (2022 × 315)
├── tfidf_vectorizer.pkl          # Fitted TF-IDF vectorizer
│
└── requirements.txt              # Python dependencies
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/met-art-recommender.git
cd met-art-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run met_app.py
```

---

## How It Was Built

### Step 1 — Data Collection
The Met Museum API has 470,000+ objects but most lack images or aren't publicly accessible. I built a collection script that:
- Queries each department directly using hardcoded department IDs
- Filters for artworks with images only (`primaryImageSmall`)
- Handles rate limiting with retries and request throttling
- Saves ~300 artworks per department

### Step 2 — Feature Engineering
Raw text fields can't go into an ML model directly. I engineered three types of features:
- **Era bucketing** — converted raw year dates into 8 meaningful historical periods
- **One-hot encoding** — converted department, culture, era into binary columns
- **TF-IDF** — converted tags and medium text into 100 numerical features

### Step 3 — Real-time ML
When the user finishes rating, the app trains a Random Forest on the spot using those 20 ratings as training examples. It then scores every unrated artwork and returns them ranked by predicted enjoyment.

### Step 4 — SHAP Explainability
SHAP (SHapley Additive exPlanations) computes how much each feature contributed to each recommendation score. I map the top contributing features back to plain English labels so users understand *why* something was recommended.

---

## Key Challenges Solved

| Challenge | Solution |
|---|---|
| Met API rate limiting | `safe_get()` with retries + 150ms throttle between requests |
| Empty API responses | Empty body detection before JSON parsing |
| Department ID lookup failures | Hardcoded all department IDs — no name matching |
| SHAP multi-class output | Used `shap_values[2]` for the "Love" class specifically |
| Streamlit cache with ML models | Underscore prefix on unhashable args (`_rf`, `_feature_matrix`) |
| pandas 3.x deprecation | `applymap` → `map` |

---

## About

Built as a portfolio project to demonstrate:
- Real-world ML pipeline (data collection → feature engineering → model training → deployment)
- Recommendation systems (content-based filtering)
- Explainable AI (SHAP)
- API integration (Met Museum Open Access)
- Production deployment (Streamlit Cloud)

---

## Acknowledgements

- [The Metropolitan Museum of Art Open Access](https://metmuseum.github.io/) — free API and open access collection
- [SHAP](https://github.com/slundberg/shap) — explainability library
- [Streamlit](https://streamlit.io) — deployment platform
