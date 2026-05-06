# Engineering Notes — What I Actually Had to Fix

> Behind the Met Museum Personal Art Tour Generator  
> Live app: https://met-art-recommender-ymje5stff6zbqmqpwexsxb.streamlit.app/

Building this looked straightforward on paper. It wasn't. Here are the real problems I hit — and exactly how I solved them.

---

## Problem 1 — The API was silently failing

The Met Museum's API would return HTTP 200 (success) but with a completely empty response body. No error. No message. My JSON parser crashed every time with:

```
Expecting value: line 1 column 1 (char 0)
```

The API was rate-limiting me without telling me.

**Fix:** Built a `safe_get()` function that detects empty response bodies before attempting to parse, retries up to 3 times with a 2-second backoff, and adds a 150ms throttle between every request.

```python
def safe_get(url):
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=15)
            if not r.text.strip():
                time.sleep(RETRY_DELAY)
                continue
            return r.json()
        except (json.JSONDecodeError, requests.exceptions.Timeout):
            time.sleep(RETRY_DELAY)
    return None
```

---

## Problem 2 — Entire departments were silently skipped

My original script looked up department IDs by matching names from the API. "European Paintings" wasn't matching exactly — so the whole department was skipped without any error. I didn't notice until I saw the dataset had almost no paintings.

**Fix:** Hardcoded all department IDs directly. No name matching. No ambiguity.

```python
DEPARTMENTS = {
    "European Paintings":          11,
    "Modern and Contemporary Art": 21,
    "Asian Art":                    6,
    "Egyptian Art":                10,
    "Greek and Roman Art":         13,
    "Islamic Art":                 14,
    "The American Wing":            1,
}
```

---

## Problem 3 — Van Gogh and Monet weren't in my dataset

I searched for them. The API returned results. But when I checked the final CSV — barely anything. Turns out the Met doesn't own Van Gogh's *Starry Night* or Monet's *Water Lilies*. Those are at MoMA.

**Fix:** Directly searched the API by artist name for every famous artist and hardcoded the exact Met object IDs for 12 iconic works — guaranteeing they always appear regardless of how the dataset is sampled. Added an honest note in the app explaining what the Met actually owns vs other museums.

---

## Problem 4 — IndexError crashing the app in production

After rerunning data collection, my CSV had 2,788 rows — but the feature matrix I had built earlier only had 2,022 rows. When the model tried to look up artwork index 2,359 in a matrix of size 2,022:

```
IndexError: index 2359 is out of bounds for axis 0 with size 2022
```

This only appeared in production. Locally everything worked fine.

**Fix:** Two guards — truncate the dataframe to match the feature matrix size at load time, and add a per-index bounds check inside the training loop.

```python
# At load time — sync df rows to feature matrix
if len(df) > feature_matrix.shape[0]:
    df = df.iloc[:feature_matrix.shape[0]].reset_index(drop=True)

# In training loop — check every index before use
if idx < feature_matrix.shape[0]:
    rated_indices.append(idx)
```

---

## Problem 5 — Every user saw the exact same 20 artworks

I had `random_state=42` everywhere for reproducibility during development and forgot to remove it before deploying. Every session, every user — identical queue. Completely defeated the point of personalisation.

**Fix:** Generate a unique session ID from `time.time_ns()` on first load. Every session gets a genuinely different random seed — including after pressing Start Over.

```python
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(time.time_ns())
```

---

## Problem 6 — Hardcoded artworks crashing the ML model

I hardcoded 12 iconic artworks (Van Gogh, Rembrandt, Vermeer etc.) that are always shown as must-sees. But their IDs didn't exist in the feature matrix — so when the model tried to train on ratings that included them, it crashed.

**Fix:** Prefixed all hardcoded artwork IDs with `iconic_` so they can never accidentally match a real CSV row. The rating queue builder and model training loop both explicitly filter them out.

```python

---

## The Lesson

Real ML engineering is 20% modelling and 80% everything else.

Data quality. API reliability. Production edge cases. State management. The model itself was the easy part.

---

🔗 [Live App](https://met-art-recommender-ymje5stff6zbqmqpwexsxb.streamlit.app/) · [README](./README.md)
