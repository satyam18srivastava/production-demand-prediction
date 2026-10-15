
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Production Demand Intelligence",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_demand_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"
DATA_PATH = BASE_DIR / "production_demand_dataset.csv"

# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554, #1d4ed8);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
    }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. LOAD MODEL, FEATURES AND DATASET
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "Demand"]).copy()

    for column in ["Production", "Sales", "Inventory",
                   "Price", "Promotion", "Holiday", "Demand"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna(
        subset=["Production", "Sales", "Inventory", "Price",
                "Promotion", "Holiday", "Demand"]
    ).copy()

if not MODEL_PATH.exists():
    st.error("Model file missing: best_demand_model.pkl")
    st.stop()

if not FEATURES_PATH.exists():
    st.error("Feature file missing: model_features.pkl")
    st.stop()

if not DATA_PATH.exists():
    st.error("Dataset missing: production_demand_dataset.csv")
    st.stop()

try:
    model = load_model()
    features = joblib.load(FEATURES_PATH)
    df = load_dataset()
except Exception as error:
    st.error(f"Could not load project files: {error}")
    st.stop()

if df.empty:
    st.error("Dataset is empty or contains invalid values.")
    st.stop()

# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

missing_features = [f for f in features if f not in df.columns]

if missing_features:
    st.error(f"Missing model features: {missing_features}")
    st.stop()

X = df[features]
y = df["Demand"]

# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("📦 Demand Intelligence")
st.sidebar.caption("Machine Learning Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard Overview",
        "Demand Prediction",
        "Demand Analytics",
        "Historical Data",
        "Model Evaluation"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Prototype built using Python, Pandas, "
    "Scikit-learn and Streamlit."
)

# --------------------------------------------------
# 6. COMMON HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Production Demand Intelligence</h1>
    <p>Understand demand patterns and explore ML-based predictions.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Academic prototype using synthetic practice data. "
    "Predictions are not validated for real-world production."
)

# --------------------------------------------------
# 7. DASHBOARD OVERVIEW
# --------------------------------------------------
if page == "Dashboard Overview":

    st.subheader("Business Overview")

    total_records = len(df)
    avg_demand = df["Demand"].mean()
    total_demand = df["Demand"].sum()
    total_products = (
        df["Product_ID"].nunique()
        if "Product_ID" in df.columns else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Records", f"{total_records:,}")
    c2.metric("Products", f"{total_products}")
    c3.metric("Average Demand", f"{avg_demand:,.2f}")
    c4.metric("Total Recorded Demand", f"{total_demand:,.0f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Monthly Demand Trend")
        monthly = (
            df.set_index("Date")["Demand"]
            .resample("MS")
            .sum()
            .rename("Total Demand")
        )
        st.line_chart(monthly)

    with right:
        st.subheader("Demand by Product")

        if "Product_ID" in df.columns:
            product_demand = (
                df.groupby("Product_ID")["Demand"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(product_demand)

    st.subheader("Latest Dataset Records")
    st.dataframe(
        df.sort_values("Date", ascending=False).head(10),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 8. INTERACTIVE DEMAND PREDICTION
# --------------------------------------------------
elif page == "Demand Prediction":

    st.subheader("Predict Product Demand")
    st.write("Enter the product and operational details.")

    col1, col2 = st.columns(2)

    with col1:
        production = st.number_input(
            "Production Quantity", min_value=0, value=500
        )
        sales = st.number_input(
            "Sales Quantity", min_value=0, value=450
        )
        inventory = st.number_input(
            "Inventory", min_value=0, value=200
        )

    with col2:
        price = st.number_input(
            "Product Price", min_value=0.0, value=100.0
        )
        promotion = st.selectbox("Promotion Active?", [0, 1])
        holiday = st.selectbox("Holiday?", [0, 1])

    prediction_date = st.date_input("Date")

    if st.button("Predict Demand", type="primary",
                 use_container_width=True):

        input_values = {
            "Production": production,
            "Sales": sales,
            "Inventory": inventory,
            "Price": price,
            "Promotion": promotion,
            "Holiday": holiday,
            "Year": prediction_date.year,
            "Month": prediction_date.month,
            "Day": prediction_date.day,
            "DayOfWeek": prediction_date.weekday()
        }

        try:
            input_df = pd.DataFrame(
                [[input_values[f] for f in features]],
                columns=features
            )

            prediction = float(model.predict(input_df)[0])

            st.success("Prediction completed successfully!")
            st.metric(
                "Estimated Demand",
                f"{max(0, prediction):,.2f} units"
            )

            st.write("**Inputs used by the model**")
            st.dataframe(input_df, use_container_width=True)

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# 9. DEMAND ANALYTICS
# --------------------------------------------------
elif page == "Demand Analytics":

    st.subheader("Explore Demand Patterns")

    if "Product_ID" in df.columns:
        products = sorted(df["Product_ID"].dropna().unique())
        selected_products = st.multiselect(
            "Select products",
            products,
            default=products
        )
        filtered = df[df["Product_ID"].isin(selected_products)]
    else:
        filtered = df.copy()

    if filtered.empty:
        st.warning("Select at least one product.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Average Demand by Product**")
            if "Product_ID" in filtered.columns:
                avg_by_product = (
                    filtered.groupby("Product_ID")["Demand"].mean()
                )
                st.bar_chart(avg_by_product)

        with col2:
            st.write("**Monthly Average Demand**")
            monthly_avg = (
                filtered.set_index("Date")["Demand"]
                .resample("MS")
                .mean()
                .dropna()
            )
            st.line_chart(monthly_avg)

        st.write("**Production vs Demand**")
        comparison = filtered[["Production", "Demand"]].copy()
        st.line_chart(comparison)

# --------------------------------------------------
# 10. HISTORICAL DATA
# --------------------------------------------------
elif page == "Historical Data":

    st.subheader("Historical Dataset Explorer")

    filtered = df.copy()

    if "Product_ID" in filtered.columns:
        products = ["All"] + sorted(
            filtered["Product_ID"].dropna().unique().tolist()
        )
        chosen = st.selectbox("Filter by product", products)

        if chosen != "All":
            filtered = filtered[filtered["Product_ID"] == chosen]

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    date_range = st.date_input(
        "Filter date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["Date"].dt.date >= start_date) &
            (filtered["Date"].dt.date <= end_date)
        ]

    st.write(f"**Records displayed: {len(filtered)}**")

    st.dataframe(
        filtered.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="filtered_demand_data.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# 11. MODEL EVALUATION
# --------------------------------------------------
elif page == "Model Evaluation":

    st.subheader("Saved Model Evaluation")
    st.write(
        "The saved model is evaluated on the same 80/20 split "
        "configuration used in the notebook."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    try:
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("R² Score", f"{r2:.4f}")

        st.caption(
            "MAE and RMSE measure prediction error. "
            "R² measures variance explained, not classification accuracy."
        )

        chart_data = pd.DataFrame({
            "Actual Demand": y_test.to_numpy(),
            "Predicted Demand": predictions
        })

        st.subheader("Actual vs Predicted Demand")
        st.scatter_chart(
            chart_data,
            x="Actual Demand",
            y="Predicted Demand"
        )

        st.subheader("Sample Predictions")
        st.dataframe(chart_data.head(20), use_container_width=True)

    except Exception as error:
        st.error(f"Evaluation failed: {error}")

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Production Demand Prediction System | "
    "Educational ML prototype"
)
