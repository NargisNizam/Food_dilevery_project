# Food Delivery Analytics Challenge — Hackathon Task-A (Batch-12)

Python/Pandas EDA on a 38,964-row food-delivery dataset to find what drives
delivery time, with an AI (LLM) step that turns the calculated results into
a plain-language business explanation. **No Machine Learning is used** —
all analysis is done with Pandas; the AI only explains numbers Python
already computed.

## Folder structure

```
food_delivery_project/
├── main.py                     # console version — all analysis code (Tasks A-G)
├── app.py                      # BONUS: interactive Streamlit browser dashboard
├── food_delivery_dataset.csv   # provided dataset
├── requirements.txt
├── README.md                   # this file
└── outputs/                    # generated on run of main.py
    ├── chart1_avg_time_by_traffic.png
    ├── chart2_distance_vs_time.png
    ├── chart3_avg_time_by_weather.png
    ├── chart4_weather_traffic_heatmap.png
    ├── cleaning_log.txt
    ├── ai_explanation.txt      # only if an API key was set
    └── insights_report.txt     # full console output, saved
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

### (Optional) enable Task G — AI-powered explanation

The API key is **never hard-coded** in `main.py`. Set it as an environment
variable before running:

**Windows PowerShell**
```powershell
$env:ANTHROPIC_API_KEY="your-key-here"
python main.py
```

**macOS/Linux**
```bash
export ANTHROPIC_API_KEY="your-key-here"
python main.py
```

**Or with a `.env` file** (don't commit this file):
```
ANTHROPIC_API_KEY=your-key-here
```
and load it by adding `from dotenv import load_dotenv; load_dotenv()` near
the top of `main.py`.

If no key is set, the script still runs completely — it just prints the
data that *would* have been sent to the AI and skips the live call, so the
project always "runs without manual changes to the source code" as required.

## Bonus: browser dashboard (`app.py`)

A second, optional entry point renders the same analysis as an interactive
web dashboard in the browser, with filters for City/Weather/Traffic and
live-updating charts and metrics.

```bash
streamlit run app.py
```

This opens automatically at `http://localhost:8501`. The AI-explanation
step here takes the API key from a password-style sidebar box at runtime
(or the `ANTHROPIC_API_KEY` environment variable) — still never hard-coded.
`main.py` (console) and `app.py` (browser) are independent — either can be
run on its own; `app.py` is the bonus feature, `main.py` covers the core
required deliverables including the saved chart/report files.

## Data cleaning decisions (Task B)

1. Stripped stray whitespace from all text columns; normalised `"NaN"`/empty
   strings to real `NaN`.
2. Converted `Time_taken (min)`, `Delivery_person_Age`,
   `Delivery_person_Ratings`, and `multiple_deliveries` to numeric types, and
   `Order_Date` to datetime — these were loaded as text/object by default.
3. Removed exact duplicate rows (0 found in this dataset).
4. Flagged ages outside 15–65 and ratings outside 0–5 as invalid → set to
   missing before imputation (0 found in this dataset, but the check is
   kept for robustness against future data).
5. **Missing `Delivery_person_Age` (1,019 rows) and `Delivery_person_Ratings`
   (1,055 rows)** → filled with the **median**, not the mean, since these are
   only ~2.6–2.7% of rows and the median is less sensitive to outliers than
   the mean.
6. **Missing `multiple_deliveries`** → filled with `0`, assuming no bundled
   order when the field wasn't recorded.
7. **Missing `Time_Orderd` (835 rows)** → left as `NaN`. These rows are kept
   for every analysis except the hour-of-day breakdown, since a clock-time
   value can't be meaningfully guessed.
8. Rows with missing `Time_taken (min)` (the core metric) are dropped, since
   this cannot be reliably imputed (none were missing in this dataset).

Full log is also written to `outputs/cleaning_log.txt` on each run.

## Basic analysis (Task C) — results from this dataset

| Metric | Value |
|---|---|
| Total deliveries | 38,964 |
| Average delivery time | 26.58 min |
| Min / Max delivery time | 10 / 54 min |
| Average distance | 9.77 km |
| Average speed | 23.59 km/h |
| Average delivery-person rating | 4.63 / 5 |
| Average delivery-person age | 29.6 years |

## Answers to the 3 competition questions (Task D)

**Q1 — Traffic Impact:** `Jam` traffic has the highest average delivery time
(31.44 min), vs. 21.50 min for `Low` traffic.

**Q2 — Distance Impact:** Correlation between distance and delivery time is
**0.32** (moderate positive). Average time rises from ~22.4 min for 0–5 km
trips to ~30.1 min for 15–20 km trips — delivery time increases with
distance, though not perfectly linearly (traffic/weather add noise).

**Q3 — Combined Conditions:** `Fog` weather combined with `Jam` traffic has
the highest average delivery time (36.89 min) — worse than either factor
alone.

*(All three answers are computed programmatically in `main.py` via
`groupby()`/`.corr()` — nothing is hard-coded.)*

## Business insights (Task F)

1. **Traffic is the single biggest delay driver** — Jam vs. Low traffic is a
   ~10-minute gap. → Add a traffic-aware buffer to promised ETAs and
   consider surge staffing in heavy traffic.
2. **Distance moderately increases delivery time** (corr = 0.32) → ETAs
   shown to customers should scale with distance instead of a flat number.
3. **Fog + Jam is the worst combination** (36.9 min) → could trigger an
   automatic ETA adjustment and proactive customer notification.
4. **Higher-rated delivery agents tend to be faster** (corr ≈ -0.36 between
   rating and time) → ratings are a usable signal for assigning
   time-sensitive orders.

## Notes

- `distance_km` and `delivery_speed` were already present in the provided
  CSV; `delivery_speed` is a categorical label (Slow/Average/Fast), so a
  separate numeric `speed_kmph` (`distance_km / (time/60)`) was calculated
  for the "average speed" figure required in Task C.
- Screenshots of the running project should be added by the student when
  submitting, per the checklist.
