import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

REGION_FILES = {
    "Region 0": DATA_DIR / "geo_data_0.csv",
    "Region 1": DATA_DIR / "geo_data_1.csv",
    "Region 2": DATA_DIR / "geo_data_2.csv",
}

TARGET_COL = "product"        # Adjust if your target column has a different name
N_WELLS = 200                  # Number of wells selected per region
BUDGET = 10_000_000_000        # Total budget for drilling
REVENUE_PER_BBL = 4_500        # Revenue per thousand barrels produced
FEATURE_COLS = [col for col in range(10)]  # f0 through f9


def prepare_features(df):
    # If columns are named f0,f1,...f9
    feature_cols = [col for col in df.columns if col.startswith("f")]
    df = df.copy()
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df[TARGET_COL].astype(float)
    return X, y

# -------------------------------------------------------------------
# Data and modeling helpers
# -------------------------------------------------------------------
@st.cache_data
def load_region_data() -> dict:
    """Load data for each region into a dictionary of DataFrames."""
    data_dict = {}
    for region, path in REGION_FILES.items():
        df = pd.read_csv(path)
        data_dict[region] = df
    return data_dict


def train_linear_model(df: pd.DataFrame):

    X, y = prepare_features(df)

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)

    return model, rmse, preds


def calculate_profit_from_predictions(preds: np.ndarray) -> float:
    """Select top N wells by prediction and calculate profit."""
    top_preds = np.sort(preds)[-N_WELLS:]
    total_product = top_preds.sum()

    revenue = total_product * REVENUE_PER_BBL
    profit = revenue - BUDGET

    return profit


def bootstrap_profit(
    preds: np.ndarray,
    n_iterations: int = 1000,
    sample_size: int = N_WELLS,
    random_state: int = 42,
):
    """Run bootstrapping simulation on predictions to estimate profit distribution."""
    rng = np.random.default_rng(random_state)
    profits = []

    for _ in range(n_iterations):
        sample = rng.choice(preds, size=sample_size, replace=True)
        total_product = sample.sum()
        revenue = total_product * REVENUE_PER_BBL
        profit = revenue - BUDGET
        profits.append(profit)

    profits = np.array(profits)
    mean_profit = profits.mean()
    lower_ci = np.percentile(profits, 2.5)
    upper_ci = np.percentile(profits, 97.5)
    loss_prob = (profits < 0).mean()

    return profits, mean_profit, lower_ci, upper_ci, loss_prob


@st.cache_data
def compute_region_metrics():
    """Train models and compute metrics for all regions once."""
    data_dict = load_region_data()

    results = {}
    for region, df in data_dict.items():
        model, rmse, preds = train_linear_model(df)
        base_profit = calculate_profit_from_predictions(preds)
        (
            profits,
            mean_profit,
            lower_ci,
            upper_ci,
            loss_prob,
        ) = bootstrap_profit(preds)

        results[region] = {
            "df": df,
            "model": model,
            "rmse": rmse,
            "preds": preds,
            "base_profit": base_profit,
            "profits": profits,
            "mean_profit": mean_profit,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "loss_prob": loss_prob,
        }

    return results


# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Oil Well Profitability - Regional Analysis",
        layout="wide",
    )

    st.title("🛢️ Oil Well Profitability Prediction")
    st.markdown(
        """
This dashboard summarizes a machine learning project that evaluates which region
is most profitable and least risky for new oil well development.

Data comes from three regions. A Linear Regression model predicts oil reserves, and 
bootstrapping is used to estimate the distribution of possible profits and the risk of loss.
        """
    )

    results = compute_region_metrics()

    # Identify best region based on mean profit and loss probability
    best_region = max(results.keys(), key=lambda r: results[r]["mean_profit"])

    # Sidebar
    st.sidebar.header("Controls")
    region_choice = st.sidebar.selectbox(
        "Select a region to explore",
        list(results.keys()),
        index=list(results.keys()).index(best_region),
    )

    n_iter = st.sidebar.slider(
        "Profit distribution percentile range",
        min_value=80,
        max_value=99,
        value=95,
        step=1,
        help="Controls the width of the confidence interval displayed.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Suggested region for drilling:** `{best_region}`"
    )

    # Overall comparison section
    st.subheader("Region comparison overview")

    comparison_rows = []
    for region, res in results.items():
        comparison_rows.append(
            {
                "Region": region,
                "RMSE": round(res["rmse"], 2),
                "Mean profit": round(res["mean_profit"], 2),
                "Lower CI": round(res["lower_ci"], 2),
                "Upper CI": round(res["upper_ci"], 2),
                "Loss probability": round(res["loss_prob"], 3),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).set_index("Region")
    st.dataframe(comparison_df)

    st.markdown(
        f"**Recommended region based on this analysis: `{best_region}` "
        f"(highest mean profit and low loss probability).**"
    )

    st.markdown("---")

    # Detailed view for selected region
    st.subheader(f"Detailed view - {region_choice}")
    region_res = results[region_choice]

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{region_res['rmse']:.2f}")
    col2.metric("Base profit (top 200 wells)", f"{region_res['base_profit'] / 1e9:.2f} B$")
    col3.metric(
        "Loss probability",
        f"{region_res['loss_prob'] * 100:.1f} %",
    )

    # Confidence interval based on user selected percentile range
    lower_p = (100 - n_iter) / 2
    upper_p = 100 - lower_p
    profits = region_res["profits"]
    lower_custom = np.percentile(profits, lower_p)
    upper_custom = np.percentile(profits, upper_p)

    st.markdown(
        f"**Bootstrapped profit interval ({n_iter} percent range):** "
        f"{lower_custom / 1e9:.2f} B$ to {upper_custom / 1e9:.2f} B$"
    )

    # Profit distribution plot
    st.markdown("#### Profit distribution from bootstrapping")

    fig, ax = plt.subplots()
    ax.hist(profits / 1e9, bins=40)
    ax.set_xlabel("Profit (billion $)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bootstrapped profit distribution - {region_choice}")
    st.pyplot(fig)

    # Raw data preview
    with st.expander("Show raw data for this region"):
        st.write(results[region_choice]["df"].head())


if __name__ == "__main__":
    main()

