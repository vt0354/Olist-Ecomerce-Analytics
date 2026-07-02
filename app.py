import os
import gdown
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Olist E-commerce Analytics",
    layout="wide"
)

# =========================
# DOWNLOAD LARGE DATASET
# =========================

os.makedirs("data", exist_ok=True)

FILE_ID = "1m-1cjSB4OW5bR383tPpK6Dsjq9XGuPLU"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
OUTPUT = "data/olist_master_dashboard.csv"

if not os.path.exists(OUTPUT):
    with st.spinner("Downloading dashboard dataset..."):
        gdown.download(URL, OUTPUT, quiet=False)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    master = pd.read_csv("data/olist_master_dashboard.csv")
    rfm = pd.read_csv("data/rfm_customer_segments.csv")
    clv = pd.read_csv("data/customer_lifetime_value.csv")
    propensity = pd.read_csv("data/customer_propensity_scores_improved.csv")
    model_comparison = pd.read_csv("data/model_comparison_improved.csv")
    feature_importance = pd.read_csv("data/propensity_feature_importance_improved.csv")

    master["order_purchase_timestamp"] = pd.to_datetime(
        master["order_purchase_timestamp"],
        errors="coerce"
    )

    return master, rfm, clv, propensity, model_comparison, feature_importance


master, rfm, clv, propensity, model_comparison, feature_importance = load_data()

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Executive Overview",
        "Sales Analysis",
        "Customer Segmentation",
        "CLV Analysis",
        "Propensity Model"
    ]
)

st.title("🛒 Olist E-commerce Analytics Dashboard")

# =========================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================

if page == "Executive Overview":
    st.header("Executive Overview")

    total_revenue = master["payment_value"].sum()
    total_orders = master["order_id"].nunique()
    total_customers = master["customer_unique_id"].nunique()
    avg_order_value = total_revenue / total_orders

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Revenue", f"${total_revenue:,.2f}")
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Total Customers", f"{total_customers:,}")
    c4.metric("Average Order Value", f"${avg_order_value:,.2f}")

    monthly = (
        master.groupby(master["order_purchase_timestamp"].dt.to_period("M"))["payment_value"]
        .sum()
        .reset_index()
    )

    monthly["order_purchase_timestamp"] = monthly["order_purchase_timestamp"].astype(str)

    fig = px.line(
        monthly,
        x="order_purchase_timestamp",
        y="payment_value",
        markers=True,
        title="Monthly Revenue Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

    state_rev = (
        master.groupby("customer_state")["payment_value"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig2 = px.bar(
        state_rev.head(15),
        x="customer_state",
        y="payment_value",
        title="Top States by Revenue"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# PAGE 2: SALES ANALYSIS
# =========================

elif page == "Sales Analysis":
    st.header("Sales & Product Analysis")

    category_rev = (
        master.groupby("product_category_name_english")["payment_value"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig = px.bar(
        category_rev,
        x="payment_value",
        y="product_category_name_english",
        orientation="h",
        title="Top Product Categories by Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

    payment_rev = (
        master.groupby("payment_type")["payment_value"]
        .sum()
        .reset_index()
    )

    fig2 = px.pie(
        payment_rev,
        names="payment_type",
        values="payment_value",
        title="Revenue by Payment Type"
    )
    st.plotly_chart(fig2, use_container_width=True)

    city_orders = (
        master.groupby("customer_city")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    fig3 = px.bar(
        city_orders,
        x="customer_city",
        y="order_id",
        title="Top Cities by Orders"
    )
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# PAGE 3: CUSTOMER SEGMENTATION
# =========================

elif page == "Customer Segmentation":
    st.header("RFM Customer Segmentation")

    segment_count = (
        rfm.groupby("Segment")["customer_unique_id"]
        .count()
        .reset_index()
    )

    fig = px.pie(
        segment_count,
        names="Segment",
        values="customer_unique_id",
        title="Customer Segment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    segment_revenue = (
        rfm.groupby("Segment")["Monetary"]
        .sum()
        .reset_index()
    )

    fig2 = px.bar(
        segment_revenue,
        x="Segment",
        y="Monetary",
        title="Revenue by RFM Segment"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        color="Segment",
        title="Recency vs Monetary"
    )
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# PAGE 4: CLV ANALYSIS
# =========================

elif page == "CLV Analysis":
    st.header("Customer Lifetime Value Analysis")

    top_clv = clv.sort_values("EstimatedCLV", ascending=False).head(20)

    fig = px.bar(
        top_clv,
        x="customer_unique_id",
        y="EstimatedCLV",
        title="Top 20 Customers by Estimated CLV"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(
        clv,
        x="EstimatedCLV",
        nbins=50,
        title="CLV Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "CLVSegment" in clv.columns:
        clv_segment = (
            clv.groupby("CLVSegment")["customer_unique_id"]
            .count()
            .reset_index()
        )

        fig3 = px.bar(
            clv_segment,
            x="CLVSegment",
            y="customer_unique_id",
            title="Customers by CLV Segment"
        )
        st.plotly_chart(fig3, use_container_width=True)

# =========================
# PAGE 5: PROPENSITY MODEL
# =========================

elif page == "Propensity Model":
    st.header("Propensity-to-Buy Model Insights")

    st.subheader("Model Comparison")
    st.dataframe(model_comparison)

    fig = px.bar(
        model_comparison,
        x="Model",
        y="ROC_AUC",
        title="Model Comparison by ROC-AUC"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Feature Importance")

    top_features = feature_importance.head(20)

    fig2 = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 20 Feature Importance"
    )
    st.plotly_chart(fig2, use_container_width=True)

    segment_count = (
        propensity.groupby("PropensitySegment")["customer_unique_id"]
        .count()
        .reset_index()
    )

    fig3 = px.pie(
        segment_count,
        names="PropensitySegment",
        values="customer_unique_id",
        title="Propensity Segment Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("High Propensity Customers")
    high_customers = propensity[propensity["PropensitySegment"] == "High"]
    st.dataframe(high_customers.head(100))
