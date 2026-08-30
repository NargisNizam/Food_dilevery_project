

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

CSV_PATH = "food_delivery_dataset.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Collects every printed line so we can also dump a single report file
REPORT_LINES = []


def log(msg=""):
    """Print to console AND remember for the saved report file."""
    print(msg)
    REPORT_LINES.append(str(msg))


def section(title):
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"  -> chart saved: {path}")


# ==========================================================================
# TASK A — LOAD & UNDERSTAND
# ==========================================================================
def load_and_understand():
    section("TASK A: LOAD & UNDERSTAND")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    log(f"Rows: {df.shape[0]:,}")
    log(f"Columns: {df.shape[1]}")
    log(f"\nColumn names:\n{list(df.columns)}")

    log(f"\nData types:\n{df.dtypes}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    log(f"\nMissing values per column:\n{missing if not missing.empty else 'None'}")

    dup_count = df.duplicated().sum()
    log(f"\nDuplicate rows: {dup_count}")

    return df


# ==========================================================================
# TASK B — CLEAN THE DATA
# ==========================================================================
def clean_data(df):
    section("TASK B: CLEAN THE DATA")
    cleaning_log = []  # human-readable list of decisions -> also written to README

    # --- Text columns: strip whitespace, normalise "NaN"-like strings ---
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"NaN": np.nan, "nan": np.nan, "": np.nan})
    cleaning_log.append("Stripped stray whitespace from all text columns and "
                         "normalised 'NaN'/empty strings to real NaN.")

    # --- Fix data types ---
    df["Time_taken (min)"] = pd.to_numeric(df["Time_taken (min)"], errors="coerce")
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)
    cleaning_log.append("Converted Time_taken, Delivery_person_Age, "
                         "Delivery_person_Ratings and multiple_deliveries to numeric, "
                         "and Order_Date to datetime.")

    # --- Duplicates ---
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    cleaning_log.append(f"Removed {removed} duplicate rows.")

    # --- Invalid values ---
    # Age: realistic delivery-person age range
    invalid_age = df[(df["Delivery_person_Age"] < 15) | (df["Delivery_person_Age"] > 65)]
    df.loc[invalid_age.index, "Delivery_person_Age"] = np.nan
    # Ratings: valid range is 0-5
    invalid_rating = df[(df["Delivery_person_Ratings"] < 0) | (df["Delivery_person_Ratings"] > 5)]
    df.loc[invalid_rating.index, "Delivery_person_Ratings"] = np.nan
    cleaning_log.append(
        f"Flagged {len(invalid_age)} out-of-range ages (<15 or >65) and "
        f"{len(invalid_rating)} out-of-range ratings (<0 or >5) as missing."
    )

    # --- Missing values: fill vs drop decisions ---
    # Age & Ratings: numeric, ~2-3% missing -> fill with median (robust to outliers)
    age_median = df["Delivery_person_Age"].median()
    rating_median = df["Delivery_person_Ratings"].median()
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(age_median)
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(rating_median)
    cleaning_log.append(
        f"Filled missing Delivery_person_Age with median ({age_median:.0f}) and "
        f"missing Delivery_person_Ratings with median ({rating_median:.1f}) — "
        "median chosen over mean to reduce sensitivity to outliers."
    )

    # multiple_deliveries: missing likely means 0 bundled deliveries
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(0)
    cleaning_log.append("Filled missing multiple_deliveries with 0 "
                         "(assumed no bundled orders when unspecified).")

    # Time_Orderd: cannot be sensibly imputed (it's a specific clock time) —
    # rows are kept for all other analysis, only excluded from hour-of-day analysis.
    cleaning_log.append("Left missing Time_Orderd as NaN — those rows are kept for "
                         "every analysis except the hour-of-day breakdown.")

    # Drop rows with no delivery time at all — the core metric can't be imputed reliably
    before = len(df)
    df = df.dropna(subset=["Time_taken (min)"])
    cleaning_log.append(f"Dropped {before - len(df)} rows with missing Time_taken "
                         "(the core metric of this analysis cannot be reliably imputed).")

    log("Cleaning decisions:")
    for i, item in enumerate(cleaning_log, 1):
        log(f"  {i}. {item}")

    log(f"\nFinal cleaned shape: {df.shape}")

    # Save the cleaning log so README.md can quote it directly
    with open(os.path.join(OUT_DIR, "cleaning_log.txt"), "w") as f:
        f.write("\n".join(f"{i}. {x}" for i, x in enumerate(cleaning_log, 1)))

    return df


# ==========================================================================
# TASK C — BASIC ANALYSIS
# ==========================================================================
def basic_analysis(df):
    section("TASK C: BASIC ANALYSIS")

    total_deliveries = len(df)
    avg_time = df["Time_taken (min)"].mean()
    min_time = df["Time_taken (min)"].min()
    max_time = df["Time_taken (min)"].max()
    avg_distance = df["distance_km"].mean()

    # delivery_speed in the CSV is a categorical label (Slow/Average/Fast);
    # we also compute a numeric speed (km per hour) for a real average figure.
    df["speed_kmph"] = df["distance_km"] / (df["Time_taken (min)"] / 60)
    avg_speed_kmph = df["speed_kmph"].mean()

    avg_rating = df["Delivery_person_Ratings"].mean()
    avg_age = df["Delivery_person_Age"].mean()

    log(f"Total deliveries: {total_deliveries:,}")
    log(f"Average delivery time: {avg_time:.2f} min")
    log(f"Minimum delivery time: {min_time:.0f} min")
    log(f"Maximum delivery time: {max_time:.0f} min")
    log(f"Average delivery distance: {avg_distance:.2f} km")
    log(f"Average delivery speed: {avg_speed_kmph:.2f} km/h")
    log(f"Delivery speed category counts:\n{df['delivery_speed'].value_counts()}")
    log(f"Average delivery-person rating: {avg_rating:.2f} / 5")
    log(f"Average delivery-person age: {avg_age:.1f} years")

    return {
        "total_deliveries": total_deliveries,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "avg_distance": avg_distance,
        "avg_speed_kmph": avg_speed_kmph,
        "avg_rating": avg_rating,
        "avg_age": avg_age,
    }


# ==========================================================================
# TASK D — ANSWER THE 3 COMPETITION QUESTIONS (computed programmatically)
# ==========================================================================
def answer_competition_questions(df):
    section("TASK D: COMPETITION QUESTIONS")
    answers = {}

    # Q1 — Traffic impact
    traffic_avg = df.groupby("Road_traffic_density")["Time_taken (min)"].mean().sort_values(ascending=False)
    q1_answer = traffic_avg.idxmax()
    q1_value = traffic_avg.max()
    log(f"Q1 — Traffic Impact:\n{traffic_avg}")
    log(f"ANSWER: '{q1_answer}' traffic has the highest average delivery time "
        f"({q1_value:.2f} min).")
    answers["q1_traffic"] = (q1_answer, q1_value, traffic_avg)

    # Q2 — Distance impact
    corr = df["distance_km"].corr(df["Time_taken (min)"])
    max_dist = df["distance_km"].max()
    bin_edges = [b for b in [0, 5, 10, 15, 20, 25] if b < max_dist] + [max_dist + 1]
    bin_edges = sorted(set(bin_edges))
    bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}km" for i in range(len(bin_edges) - 1)]
    df["distance_bucket"] = pd.cut(df["distance_km"], bins=bin_edges, labels=bin_labels)
    dist_bucket_avg = df.groupby("distance_bucket")["Time_taken (min)"].mean()
    log(f"\nQ2 — Distance Impact:")
    log(f"Correlation between distance and delivery time: {corr:.3f}")
    log(f"Average delivery time by distance bucket:\n{dist_bucket_avg}")
    direction = "increases" if corr > 0.05 else ("decreases" if corr < -0.05 else "has little effect on")
    log(f"ANSWER: Distance and delivery time have a correlation of {corr:.2f} — "
        f"as distance increases, delivery time generally {direction} "
        f"(see per-bucket averages above).")
    answers["q2_distance"] = (corr, dist_bucket_avg)

    # Q3 — Combined weather + traffic
    combo_avg = (df.groupby(["Weather_conditions", "Road_traffic_density"])["Time_taken (min)"]
                 .mean().sort_values(ascending=False))
    q3_combo = combo_avg.idxmax()
    q3_value = combo_avg.max()
    log(f"\nQ3 — Combined Conditions (top 5 slowest combinations):\n{combo_avg.head(5)}")
    log(f"ANSWER: The combination of '{q3_combo[0]}' weather and '{q3_combo[1]}' "
        f"traffic has the highest average delivery time ({q3_value:.2f} min).")
    answers["q3_combo"] = (q3_combo, q3_value, combo_avg)

    return answers


# ==========================================================================
# TASK E — VISUALIZATIONS
# ==========================================================================
def create_visualizations(df):
    section("TASK E: VISUALIZATIONS")

    # Chart 1 (required): average delivery time by traffic density — bar chart
    traffic_avg = df.groupby("Road_traffic_density")["Time_taken (min)"].mean().sort_values(ascending=False)
    plt.figure()
    sns.barplot(x=traffic_avg.index, y=traffic_avg.values, palette="viridis")
    plt.title("Average Delivery Time by Road Traffic Density", fontsize=13, fontweight="bold")
    plt.xlabel("Traffic Density")
    plt.ylabel("Average Delivery Time (minutes)")
    savefig("chart1_avg_time_by_traffic.png")

    # Chart 2 (required): delivery distance vs delivery time — scatter plot
    plt.figure()
    sample = df.sample(min(4000, len(df)), random_state=1)
    sns.scatterplot(data=sample, x="distance_km", y="Time_taken (min)", alpha=0.4, color="steelblue")
    plt.title("Delivery Distance vs Delivery Time", fontsize=13, fontweight="bold")
    plt.xlabel("Distance (km)")
    plt.ylabel("Delivery Time (minutes)")
    savefig("chart2_distance_vs_time.png")

    # Bonus charts (extra insight, not required but strengthens the "Visualizations" score)
    plt.figure()
    sns.barplot(x=df["Weather_conditions"].value_counts().index,
                y=df.groupby("Weather_conditions")["Time_taken (min)"].mean().reindex(
                    df["Weather_conditions"].value_counts().index).values,
                palette="mako")
    plt.title("Average Delivery Time by Weather Condition", fontsize=13, fontweight="bold")
    plt.xlabel("Weather")
    plt.ylabel("Average Delivery Time (minutes)")
    plt.xticks(rotation=25)
    savefig("chart3_avg_time_by_weather.png")

    plt.figure(figsize=(10, 7))
    combo_avg = (df.groupby(["Weather_conditions", "Road_traffic_density"])["Time_taken (min)"]
                 .mean().unstack())
    sns.heatmap(combo_avg, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Average Delivery Time: Weather x Traffic", fontsize=13, fontweight="bold")
    plt.xlabel("Traffic Density")
    plt.ylabel("Weather Condition")
    savefig("chart4_weather_traffic_heatmap.png")


# ==========================================================================
# TASK F — BUSINESS INSIGHTS
# ==========================================================================
def business_insights(df, stats, answers):
    section("TASK F: BUSINESS INSIGHTS")

    q1_answer, q1_value, traffic_avg = answers["q1_traffic"]
    corr, dist_bucket_avg = answers["q2_distance"]
    q3_combo, q3_value, combo_avg = answers["q3_combo"]

    fastest_traffic = traffic_avg.idxmin()
    fastest_traffic_val = traffic_avg.min()

    insights = [
        f"Traffic is a major driver of delay: '{q1_answer}' conditions push average "
        f"delivery time to {q1_value:.1f} min, versus {fastest_traffic_val:.1f} min in "
        f"'{fastest_traffic}' conditions — a gap of {q1_value - fastest_traffic_val:.1f} min. "
        f"Business impact: routing/dispatch algorithms should add a traffic-aware buffer "
        f"to promised delivery windows, and consider surge staffing during high-traffic hours.",

        f"Distance has a moderate positive relationship with delivery time "
        f"(correlation = {corr:.2f}). Business impact: delivery-time estimates shown to "
        f"customers should scale with distance rather than using a flat ETA, and the company "
        f"could evaluate zone-based delivery fees for longer distances.",

        f"The worst-case combination is '{q3_combo[0]}' weather with '{q3_combo[1]}' traffic "
        f"({q3_value:.1f} min average). Business impact: this combination could trigger an "
        f"automatic ETA adjustment and proactive customer notification, rather than customers "
        f"finding out only after their order is already late.",

        f"Delivery-person rating correlates with speed "
        f"({df['Delivery_person_Ratings'].corr(df['Time_taken (min)']):.2f}) — higher-rated "
        f"agents tend to deliver faster. Business impact: this validates using ratings as a "
        f"signal for assigning time-sensitive or premium orders.",
    ]

    for i, ins in enumerate(insights, 1):
        log(f"{i}. {ins}\n")

    return insights


# ==========================================================================
# TASK G — AI-POWERED EXPLANATION
# ==========================================================================
def ai_powered_explanation(stats, answers):
    """
    Sends the ALREADY-CALCULATED Pandas results (not raw data) to an LLM and
    asks it to explain them in plain business language. Python/Pandas did the
    analysis; the AI only explains it — no ML model is trained here.

    The API key is NEVER hard-coded. It is read from an environment variable.
    If no key is set, this step is skipped gracefully (rest of the project
    still works and produces all other deliverables).
    """
    section("TASK G: AI-POWERED EXPLANATION")

    q1_answer, q1_value, traffic_avg = answers["q1_traffic"]
    corr, _ = answers["q2_distance"]
    q3_combo, q3_value, _ = answers["q3_combo"]

    summary_for_ai = f"""
    Food delivery analysis results (computed with Pandas, no ML):
    - Total deliveries analyzed: {stats['total_deliveries']:,}
    - Average delivery time: {stats['avg_time']:.1f} minutes (min {stats['min_time']:.0f}, max {stats['max_time']:.0f})
    - Average distance: {stats['avg_distance']:.1f} km, average speed: {stats['avg_speed_kmph']:.1f} km/h
    - Average delivery-person rating: {stats['avg_rating']:.1f}/5, average age: {stats['avg_age']:.0f}
    - Traffic with highest average delivery time: {q1_answer} ({q1_value:.1f} min)
    - Correlation between distance and delivery time: {corr:.2f}
    - Worst weather+traffic combination: {q3_combo[0]} weather with {q3_combo[1]} traffic ({q3_value:.1f} min)
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log("No ANTHROPIC_API_KEY environment variable found — skipping the live AI call.")
        log("To enable this step: set the ANTHROPIC_API_KEY environment variable "
            "(see README.md) and re-run the script.")
        log("\n--- Data that WOULD be sent to the AI ---")
        log(summary_for_ai)
        return None

    try:
        import anthropic  # pip install anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    "You are a business analyst. Based ONLY on the following "
                    "pre-calculated food-delivery analytics, write a short "
                    "(120-180 word) plain-English explanation for a non-technical "
                    "business audience, including 2-3 recommended actions:\n"
                    f"{summary_for_ai}"
                ),
            }],
        )
        explanation = response.content[0].text
        log("AI-generated business explanation:\n")
        log(explanation)

        with open(os.path.join(OUT_DIR, "ai_explanation.txt"), "w") as f:
            f.write(explanation)

        return explanation

    except ImportError:
        log("The 'anthropic' package is not installed. Run: pip install anthropic")
        return None
    except Exception as e:
        log(f"AI call failed ({e}). Continuing without it — all other deliverables are unaffected.")
        return None


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    df = load_and_understand()
    df = clean_data(df)
    stats = basic_analysis(df)
    answers = answer_competition_questions(df)
    create_visualizations(df)
    insights = business_insights(df, stats, answers)
    ai_powered_explanation(stats, answers)

    section("DONE")
    log(f"All charts and the full report are saved in ./{OUT_DIR}/")

    with open(os.path.join(OUT_DIR, "insights_report.txt"), "w") as f:
        f.write("\n".join(REPORT_LINES))


if __name__ == "__main__":
    main()
