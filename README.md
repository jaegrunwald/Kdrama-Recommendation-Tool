# K-Drama Recommendation Tool

A web-based recommendation system for Korean dramas. Get suggestions by naming a drama you like, or by choosing genres, tags, rating, and year range.

## Prerequisites

- **Python 3.8+**  
  [Download Python](https://www.python.org/downloads/) if needed. During install, check **"Add Python to PATH"**.

## Download & Run

### Option A: Download as ZIP

1. Download this project as a ZIP (e.g. from GitHub: **Code → Download ZIP**), then extract it.
2. Open a terminal in the extracted folder (e.g. right‑click the folder → **Open in Terminal**, or `cd` into it).
3. Run the app:
   - **Windows:** double‑click `run.bat`, or in terminal: `run.bat`
   - **Mac/Linux:** in terminal: `chmod +x run.sh` then `./run.sh`
4. Open **http://127.0.0.1:5000** in your browser.

### Option B: Using the terminal only

```bash
# 1. Go into the project folder
cd "Kdrama Recommendation Tool"

# 2. Install dependencies (one time)
pip install -r requirements.txt

# 3. Start the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## What’s included

- **Dataset:** `kdrama_DATASET.csv` is included so the app works right after install. No extra download needed.
- **Web UI:** Search by drama name (with autocomplete) or filter by genres, tags, min rating, and year range.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `python` not found | Try `py app.py` (Windows) or install Python and add it to PATH. |
| `pip` not found | Use `python -m pip install -r requirements.txt`. |
| Port 5000 in use | Edit `app.py` and change `port=5000` to another port (e.g. `5001`). |

## Project structure

- `app.py` — Flask web server and API
- `kdrama_recommender.py` — Recommendation logic (TF-IDF, cosine similarity)
- `static/index.html` — Web interface
- `kdrama_DATASET.csv` — Drama data (included)
