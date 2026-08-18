import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Invoice Anomaly Detector",
    page_icon="🧾",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "final_invoice_anomaly_dataset.csv"
    )

    return data


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

model, scaler = load_model()
df = load_data()


# =========================================================
# FEATURE COLUMNS
# These MUST match the features used during training
# =========================================================

feature_columns = [
    "quantity",
    "unit_price",
    "tax_amount",
    "total_amount",
    "vendor_avg_amount",
    "vendor_amount_deviation",
    "vendor_std_amount",
    "customer_avg_amount",
    "customer_amount_deviation",
    "tax_ratio",
    "amount_per_quantity"
]


# =========================================================
# TITLE
# =========================================================

st.title("🧾 Invoice Anomaly Detector")

st.markdown(
    """
    ### AI-powered invoice risk detection

    This application uses **Isolation Forest** to identify
    unusual invoice transactions based on vendor, customer,
    pricing, quantity and tax behavior.
    """
)


# =========================================================
# DASHBOARD METRICS
# =========================================================

total_invoices = len(df)

anomalies = df[
    df["predicted_anomaly"] == 1
]

high_risk = df[
    df["risk_level"] == "HIGH"
]

medium_risk = df[
    df["risk_level"] == "MEDIUM"
]

low_risk = df[
    df["risk_level"] == "LOW"
]


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Invoices",
        f"{total_invoices:,}"
    )


with col2:

    st.metric(
        "Detected Anomalies",
        f"{len(anomalies):,}"
    )


with col3:

    st.metric(
        "High Risk",
        f"{len(high_risk):,}"
    )


with col4:

    st.metric(
        "Medium Risk",
        f"{len(medium_risk):,}"
    )


# =========================================================
# NEW INVOICE ANALYSIS
# =========================================================

st.divider()

st.header("🔍 Analyze a New Invoice")

st.write(
    """
    Enter the details of a new invoice below.
    The trained Isolation Forest model will calculate
    its anomaly score and risk level.
    """
)


# =========================================================
# GET AVAILABLE VENDORS AND CUSTOMERS
# =========================================================

vendor_options = sorted(
    df["vendor_name"]
    .dropna()
    .astype(str)
    .unique()
)

customer_options = sorted(
    df["customer_id"]
    .dropna()
    .unique()
)


# =========================================================
# NEW INVOICE FORM
# =========================================================

with st.form("new_invoice_form"):

    st.subheader("📝 Invoice Details")

    col1, col2 = st.columns(2)


    with col1:

        new_invoice_no = st.text_input(
            "Invoice Number",
            value="NEW-001"
        )

        selected_vendor = st.selectbox(
            "Vendor",
            vendor_options
        )

        selected_customer = st.selectbox(
            "Customer",
            customer_options
        )


    with col2:

        quantity = st.number_input(
            "Quantity",
            min_value=1.0,
            value=10.0,
            step=1.0
        )

        unit_price = st.number_input(
            "Unit Price (₹)",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        tax_amount = st.number_input(
            "Tax Amount (₹)",
            min_value=0.0,
            value=1800.0,
            step=100.0
        )


    analyze_button = st.form_submit_button(
        "🔎 Analyze Invoice"
    )


# =========================================================
# PROCESS NEW INVOICE
# =========================================================

if analyze_button:

    # -----------------------------------------------------
    # Calculate invoice values
    # -----------------------------------------------------

    subtotal_amount = (
        quantity * unit_price
    )

    total_amount = (
        subtotal_amount + tax_amount
    )


    # -----------------------------------------------------
    # Vendor historical information
    # -----------------------------------------------------

    vendor_history = df[
        df["vendor_name"].astype(str)
        == str(selected_vendor)
    ]


    vendor_avg_amount = (
        vendor_history["total_amount"].mean()
    )


    vendor_std_amount = (
        vendor_history["total_amount"].std()
    )


    if pd.isna(vendor_avg_amount):

        vendor_avg_amount = (
            df["total_amount"].mean()
        )


    if pd.isna(vendor_std_amount):

        vendor_std_amount = (
            df["total_amount"].std()
        )


    # -----------------------------------------------------
    # Vendor amount deviation
    # -----------------------------------------------------

    if vendor_avg_amount > 0:

        vendor_amount_deviation = (
            total_amount /
            vendor_avg_amount
        )

    else:

        vendor_amount_deviation = 1.0


    # -----------------------------------------------------
    # Customer historical information
    # -----------------------------------------------------

    customer_history = df[
        df["customer_id"] == selected_customer
    ]


    customer_avg_amount = (
        customer_history["total_amount"].mean()
    )


    if pd.isna(customer_avg_amount):

        customer_avg_amount = (
            df["total_amount"].mean()
        )


    # -----------------------------------------------------
    # Customer amount deviation
    # -----------------------------------------------------

    if customer_avg_amount > 0:

        customer_amount_deviation = (
            total_amount /
            customer_avg_amount
        )

    else:

        customer_amount_deviation = 1.0


    # -----------------------------------------------------
    # Tax ratio
    # -----------------------------------------------------

    if subtotal_amount > 0:

        tax_ratio = (
            tax_amount /
            subtotal_amount
        )

    else:

        tax_ratio = 0.0


    # -----------------------------------------------------
    # Amount per quantity
    # -----------------------------------------------------

    if quantity > 0:

        amount_per_quantity = (
            total_amount /
            quantity
        )

    else:

        amount_per_quantity = 0.0


    # =====================================================
    # CREATE NEW INVOICE FEATURE VECTOR
    # =====================================================

    new_invoice_features = pd.DataFrame({

        "quantity": [quantity],

        "unit_price": [unit_price],

        "tax_amount": [tax_amount],

        "total_amount": [total_amount],

        "vendor_avg_amount": [
            vendor_avg_amount
        ],

        "vendor_amount_deviation": [
            vendor_amount_deviation
        ],

        "vendor_std_amount": [
            vendor_std_amount
        ],

        "customer_avg_amount": [
            customer_avg_amount
        ],

        "customer_amount_deviation": [
            customer_amount_deviation
        ],

        "tax_ratio": [
            tax_ratio
        ],

        "amount_per_quantity": [
            amount_per_quantity
        ]

    })


    # =====================================================
    # CLEAN FEATURES
    # =====================================================

    new_invoice_features = (
        new_invoice_features
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    # Fill missing values using training dataset medians

    training_medians = (
        df[feature_columns]
        .median()
    )


    new_invoice_features = (
        new_invoice_features
        .fillna(training_medians)
    )


    # =====================================================
    # SCALE FEATURES
    # =====================================================

    new_scaled_features = scaler.transform(
        new_invoice_features[
            feature_columns
        ]
    )


    # =====================================================
    # ISOLATION FOREST PREDICTION
    # =====================================================

    prediction = model.predict(
        new_scaled_features
    )[0]


    decision_score = model.decision_function(
        new_scaled_features
    )[0]


    # =====================================================
    # CALCULATE ANOMALY SCORE
    # =====================================================

    score_min = (
        df["decision_score"].min()
    )

    score_max = (
        df["decision_score"].max()
    )


    if score_max != score_min:

        anomaly_score = (
            1 -
            (
                (decision_score - score_min)
                /
                (score_max - score_min)
            )
        ) * 100

    else:

        anomaly_score = 50.0


    anomaly_score = float(
        np.clip(
            anomaly_score,
            0,
            100
        )
    )


    # =====================================================
    # RISK CLASSIFICATION
    # =====================================================

    if anomaly_score >= 80:

        risk_level = "HIGH"

    elif anomaly_score >= 50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # =====================================================
    # GENERATE EXPLANATION
    # =====================================================

    reasons = []


    # Vendor amount

    if vendor_amount_deviation >= 3:

        reasons.append(
            f"Invoice amount is "
            f"{vendor_amount_deviation:.1f}× "
            f"the vendor's historical average."
        )


    elif vendor_amount_deviation >= 2:

        reasons.append(
            f"Invoice amount is "
            f"{vendor_amount_deviation:.1f}× "
            f"the vendor's historical average."
        )


    # Customer amount

    if customer_amount_deviation >= 3:

        reasons.append(
            f"Invoice amount is "
            f"{customer_amount_deviation:.1f}× "
            f"the customer's historical average."
        )


    # Quantity

    if quantity >= 100:

        reasons.append(
            "Invoice contains an unusually high quantity."
        )


    # Unit price

    if unit_price >= 20000:

        reasons.append(
            "Unit price is unusually high."
        )


    # Tax

    if tax_ratio >= 0.30:

        reasons.append(
            f"Tax ratio is unusually high "
            f"({tax_ratio * 100:.1f}%)."
        )


    # Model-based reason

    if len(reasons) == 0:

        if prediction == -1:

            reasons.append(
                "The invoice differs significantly "
                "from learned normal transaction patterns."
            )

        else:

            reasons.append(
                "No major abnormal pattern was detected."
            )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.divider()

    st.subheader(
        "📊 New Invoice Analysis Result"
    )


    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        st.metric(
            "Invoice Amount",
            f"₹{total_amount:,.2f}"
        )


    with result_col2:

        st.metric(
            "Anomaly Score",
            f"{anomaly_score:.2f}%"
        )


    with result_col3:

        if risk_level == "HIGH":

            st.error(
                "🚨 HIGH RISK"
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "⚠️ MEDIUM RISK"
            )

        else:

            st.success(
                "✅ LOW RISK"
            )


    # =====================================================
    # INVOICE SUMMARY
    # =====================================================

    st.subheader(
        "🧾 Invoice Summary"
    )


    summary_col1, summary_col2, summary_col3 = (
        st.columns(3)
    )


    with summary_col1:

        st.write(
            "**Invoice Number**"
        )

        st.write(
            new_invoice_no
        )


    with summary_col2:

        st.write(
            "**Vendor**"
        )

        st.write(
            selected_vendor
        )


    with summary_col3:

        st.write(
            "**Customer**"
        )

        st.write(
            selected_customer
        )


    # =====================================================
    # EXPLANATION
    # =====================================================

    st.subheader(
        "💡 Why?"
    )


    for reason in reasons:

        st.info(
            "• " + reason
        )


    # =====================================================
    # VENDOR COMPARISON
    # =====================================================

    st.subheader(
        "🏢 Vendor Comparison"
    )


    comparison_col1, comparison_col2 = (
        st.columns(2)
    )


    with comparison_col1:

        st.metric(
            "New Invoice",
            f"₹{total_amount:,.2f}"
        )


    with comparison_col2:

        st.metric(
            "Vendor Average",
            f"₹{vendor_avg_amount:,.2f}"
        )


    st.write(
        f"The new invoice is "
        f"**{vendor_amount_deviation:.2f}×** "
        f"the vendor's historical average."
    )


# =========================================================
# EXISTING INVOICE SEARCH
# =========================================================

st.divider()

st.sidebar.header(
    "🔎 Search Existing Invoice"
)


invoice_id = st.sidebar.text_input(
    "Enter Invoice Number"
)


if invoice_id:

    result = df[
        df["invoice_no"].astype(str)
        == str(invoice_id)
    ]


    if len(result) == 0:

        st.sidebar.error(
            "Invoice not found."
        )


    else:

        invoice = result.iloc[0]


        st.subheader(
            "📄 Existing Invoice Analysis"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.write(
                "**Invoice Number**"
            )

            st.write(
                invoice["invoice_no"]
            )


            st.write(
                "**Vendor**"
            )

            st.write(
                invoice["vendor_name"]
            )


            st.write(
                "**Category**"
            )

            st.write(
                invoice["category"]
            )


        with col2:

            st.write(
                "**Invoice Amount**"
            )

            st.write(
                f"₹{invoice['total_amount']:,.2f}"
            )


            st.write(
                "**Quantity**"
            )

            st.write(
                int(invoice["quantity"])
            )


            st.write(
                "**Unit Price**"
            )

            st.write(
                f"₹{invoice['unit_price']:,.2f}"
            )


        with col3:

            st.write(
                "**Anomaly Score**"
            )

            st.write(
                f"{invoice['anomaly_score']:.2f}%"
            )


            st.write(
                "**Risk Level**"
            )


            if invoice["risk_level"] == "HIGH":

                st.error(
                    "🚨 HIGH RISK"
                )

            elif invoice["risk_level"] == "MEDIUM":

                st.warning(
                    "⚠️ MEDIUM RISK"
                )

            else:

                st.success(
                    "✅ LOW RISK"
                )


        st.subheader(
            "💡 Why was this invoice flagged?"
        )


        st.info(
            invoice["anomaly_reason"]
        )


        st.subheader(
            "🏢 Vendor Behavior"
        )


        vendor_col1, vendor_col2 = (
            st.columns(2)
        )


        with vendor_col1:

            st.metric(
                "Current Invoice",
                f"₹{invoice['total_amount']:,.2f}"
            )


        with vendor_col2:

            st.metric(
                "Vendor Average",
                f"₹{invoice['vendor_avg_amount']:,.2f}"
            )


        st.write(
            f"This invoice is "
            f"**{invoice['vendor_amount_deviation']:.2f}×** "
            f"the vendor's historical average."
        )


# =========================================================
# TOP SUSPICIOUS INVOICES
# =========================================================

st.divider()

st.subheader(
    "🚨 Top Suspicious Invoices"
)


top_anomalies = (
    df[
        df["predicted_anomaly"] == 1
    ]
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    .head(20)
)


display_columns = [
    "invoice_no",
    "vendor_name",
    "total_amount",
    "anomaly_score",
    "risk_level",
    "anomaly_reason"
]


st.dataframe(
    top_anomalies[
        display_columns
    ],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# RISK DISTRIBUTION
# =========================================================

st.divider()

st.subheader(
    "📊 Risk Distribution"
)


risk_counts = (
    df["risk_level"]
    .value_counts()
)


st.bar_chart(
    risk_counts
)


# =========================================================
# VENDOR ANALYSIS
# =========================================================

st.divider()

st.subheader(
    "🏢 Vendor Risk Analysis"
)


vendor_analysis = (
    df.groupby("vendor_name")
    .agg(

        total_invoices=(
            "invoice_no",
            "count"
        ),

        average_amount=(
            "total_amount",
            "mean"
        ),

        suspicious_invoices=(
            "predicted_anomaly",
            "sum"
        )

    )
    .sort_values(
        "suspicious_invoices",
        ascending=False
    )
)


st.dataframe(
    vendor_analysis,
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Invoice Anomaly Detector | "
    "Unsupervised Machine Learning using Isolation Forest"
)