

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery Analytics",
    page_icon="🛵",
    layout="wide",
)

CSV_PATH = "food_delivery_dataset.csv"


# --------------------------------------------------------------------------
# DATA LOADING + CLEANING (cached so it only runs once per session)
# --------------------------------------------------------------------------
@st.cache_data
def load_and_clean():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"NaN": np.nan, "nan": np.nan, "": np.nan})

    df["Time_taken (min)"] = pd.to_numeric(df["Time_taken (min)"], errors="coerce")
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)

    df = df.drop_duplicates()

    invalid_age = (df["Delivery_person_Age"] < 15) | (df["Delivery_person_Age"] > 65)
    df.loc[invalid_age, "Delivery_person_Age"] = np.nan
    invalid_rating = (df["Delivery_person_Ratings"] < 0) | (df["Delivery_person_Ratings"] > 5)
    df.loc[invalid_rating, "Delivery_person_Ratings"] = np.nan

    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(df["Delivery_person_Age"].median())
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(df["Delivery_person_Ratings"].median())
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(0)

    df = df.dropna(subset=["Time_taken (min)"])
    df["speed_kmph"] = df["distance_km"] / (df["Time_taken (min)"] / 60)

    return df


df_full = load_and_clean()

# --------------------------------------------------------------------------
# SIDEBAR — FILTERS
# --------------------------------------------------------------------------
st.sidebar.header("Filters")

cities = sorted(df_full["City"].dropna().unique())
weathers = sorted(df_full["Weather_conditions"].dropna().unique())
traffics = sorted(df_full["Road_traffic_density"].dropna().unique())

sel_city = st.sidebar.multiselect("City", cities, default=cities)
sel_weather = st.sidebar.multiselect("Weather", weathers, default=weathers)
sel_traffic = st.sidebar.multiselect("Traffic density", traffics, default=traffics)

df = df_full[
    df_full["City"].isin(sel_city)
    & df_full["Weather_conditions"].isin(sel_weather)
    & df_full["Road_traffic_density"].isin(sel_traffic)
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** of {len(df_full):,} deliveries.")

st.sidebar.markdown("---")
st.sidebar.subheader("AI explanation (optional)")
api_key_input = st.sidebar.text_input(
    "Anthropic API key", type="password",
    value=os.environ.get("ANTHROPIC_API_KEY", ""),
    help="Never hard-coded — read from this input or the ANTHROPIC_API_KEY env var.",
)

# --------------------------------------------------------------------------
# HEADER + KPIs
# --------------------------------------------------------------------------
st.title("🛵 Food Delivery Analytics Dashboard")
st.caption("Hackathon Task-A (Batch-12) — Pandas-driven EDA, no Machine Learning.")

if df.empty:
    st.warning("No data matches the current filters. Adjust the sidebar filters.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total deliveries", f"{len(df):,}")
k2.metric("Avg delivery time", f"{df['Time_taken (min)'].mean():.1f} min")
k3.metric("Avg distance", f"{df['distance_km'].mean():.1f} km")
k4.metric("Avg speed", f"{df['speed_kmph'].mean():.1f} km/h")
k5.metric("Avg rating", f"{df['Delivery_person_Ratings'].mean():.2f} / 5")

st.markdown("---")

# --------------------------------------------------------------------------
# TASK D — COMPETITION QUESTIONS (computed live from the filtered data)
# --------------------------------------------------------------------------
st.header("Competition Questions")

traffic_avg = df.groupby("Road_traffic_density")["Time_taken (min)"].mean().sort_values(ascending=False)
corr = df["distance_km"].corr(df["Time_taken (min)"])
combo_avg = (df.groupby(["Weather_conditions", "Road_traffic_density"])["Time_taken (min)"]
             .mean().sort_values(ascending=False))

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Q1 — Traffic Impact")
    st.write(f"**{traffic_avg.idxmax()}** traffic has the highest average "
             f"delivery time: **{traffic_avg.max():.1f} min**")
with c2:
    st.subheader("Q2 — Distance Impact")
    st.write(f"Correlation between distance and time: **{corr:.2f}**")
    st.write("Positive correlation → longer distance generally means longer delivery time.")
with c3:
    st.subheader("Q3 — Combined Conditions")
    top_combo = combo_avg.index[0]
    st.write(f"**{top_combo[0]}** weather + **{top_combo[1]}** traffic is worst: "
             f"**{combo_avg.iloc[0]:.1f} min**")

st.markdown("---")

# --------------------------------------------------------------------------
# TASK E — VISUALIZATIONS (interactive Plotly)
# --------------------------------------------------------------------------
st.header("Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(
        traffic_avg.reset_index(), x="Road_traffic_density", y="Time_taken (min)",
        title="Average Delivery Time by Traffic Density",
        labels={"Road_traffic_density": "Traffic Density", "Time_taken (min)": "Avg Time (min)"},
        color="Time_taken (min)", color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    sample = df.sample(min(4000, len(df)), random_state=1)
    fig2 = px.scatter(
        sample, x="distance_km", y="Time_taken (min)",
        title="Delivery Distance vs Delivery Time",
        labels={"distance_km": "Distance (km)", "Time_taken (min)": "Time (min)"},
        opacity=0.4,
    )
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    weather_avg = df.groupby("Weather_conditions")["Time_taken (min)"].mean().sort_values(ascending=False)
    fig3 = px.bar(
        weather_avg.reset_index(), x="Weather_conditions", y="Time_taken (min)",
        title="Average Delivery Time by Weather",
        labels={"Weather_conditions": "Weather", "Time_taken (min)": "Avg Time (min)"},
        color="Time_taken (min)", color_continuous_scale="Sunset",
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    heat_data = df.groupby(["Weather_conditions", "Road_traffic_density"])["Time_taken (min)"].mean().unstack()
    fig4 = px.imshow(
        heat_data, text_auto=".1f", aspect="auto",
        title="Avg Delivery Time: Weather × Traffic",
        labels=dict(x="Traffic Density", y="Weather", color="Avg Time (min)"),
        color_continuous_scale="YlOrRd",
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------
# TASK F — BUSINESS INSIGHTS
# --------------------------------------------------------------------------
st.header("Business Insights")

fastest_traffic = traffic_avg.idxmin()
rating_corr = df["Delivery_person_Ratings"].corr(df["Time_taken (min)"])

st.markdown(f"""
1. **Traffic is the biggest delay driver.** `{traffic_avg.idxmax()}` conditions average
   **{traffic_avg.max():.1f} min** vs **{traffic_avg.min():.1f} min** in `{fastest_traffic}`
   conditions — a **{traffic_avg.max() - traffic_avg.min():.1f}-minute gap**.
   *Action: add a traffic-aware buffer to ETAs; consider surge staffing during heavy traffic.*

2. **Distance moderately increases delivery time** (correlation = **{corr:.2f}**).
   *Action: scale customer-facing ETAs with distance instead of a flat estimate.*

3. **Worst combination:** `{top_combo[0]}` weather + `{top_combo[1]}` traffic
   (**{combo_avg.iloc[0]:.1f} min**).
   *Action: auto-adjust ETA and proactively notify customers when this combination occurs.*

4. **Higher-rated delivery agents tend to be faster** (correlation = **{rating_corr:.2f}**).
   *Action: use ratings as a signal when assigning time-sensitive orders.*
""")

st.markdown("---")

# --------------------------------------------------------------------------
# TASK G — AI-POWERED EXPLANATION
# --------------------------------------------------------------------------
st.header("AI-Powered Explanation")

summary_for_ai = f"""
Food delivery analysis results (computed with Pandas, no ML):
- Deliveries analyzed: {len(df):,}
- Average delivery time: {df['Time_taken (min)'].mean():.1f} min
- Average distance: {df['distance_km'].mean():.1f} km, average speed: {df['speed_kmph'].mean():.1f} km/h
- Traffic with highest average delivery time: {traffic_avg.idxmax()} ({traffic_avg.max():.1f} min)
- Correlation between distance and delivery time: {corr:.2f}
- Worst weather+traffic combination: {top_combo[0]} weather with {top_combo[1]} traffic ({combo_avg.iloc[0]:.1f} min)
"""

if st.button("Generate AI explanation"):
    if not api_key_input:
        st.info("Enter an API key in the sidebar (or set ANTHROPIC_API_KEY) to enable this.")
        st.code(summary_for_ai)
    else:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key_input)
            with st.spinner("Asking the AI to explain the results..."):
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
            st.success(response.content[0].text)
        except ImportError:
            st.error("Run: pip install anthropic")
        except Exception as e:
            st.error(f"AI call failed: {e}")

st.markdown("---")
st.caption("Data source: provided food_delivery_dataset.csv · No ML models trained · "
           "Analysis performed entirely in Pandas.")
