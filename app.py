import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import time

# =========================
# SESSION STATE INIT
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

if "model_results" not in st.session_state:
    st.session_state.model_results = None

# =========================
# PAGE CONFIG (UPGRADED)
# =========================
st.set_page_config(
    page_title="MMM Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# PREMIUM CSS
# =========================
st.markdown("""
<style>

/* Global */
body {
    background-color: #0E1117;
    color: #E6EDF3;
}

/* Layout spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #222;
}

/* Cards */
.card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #222;
    margin-bottom: 15px;
}

/* KPI Cards */
.metric-card {
    background-color: #161B22;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #222;
    text-align: center;
}

/* Typography */
h1 {
    font-size: 32px;
    font-weight: 700;
    color: #E6EDF3;
}

h2 {
    font-size: 22px;
    color: #9CA3AF;
}

h3 {
    font-size: 18px;
    color: #9CA3AF;
}

/* Buttons */
button[kind="primary"] {
    background-color: #4CAF50 !important;
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    padding: 10px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER (PROFESSIONAL)
# =========================
col1, col2 = st.columns([1, 6])

with col1:
    st.image("logo2.png", width=150)

with col2:
    st.markdown("<h1>MMM Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Marketing Mix Modeling & Optimization Suite</h3>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Trusted by Data-Driven Marketing Teams")

# =========================
# PASSWORD
# =========================
password = st.text_input("🔒 Enter Password", type="password")
if password != "anshul123":
    st.stop()

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("📊 MMM Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "📈 EDA",
        "📊 Dashboard",
        "🎯 Simulator",
        "💰 Optimizer",
        "📘 Help"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built by Anshul 🚀")

# =========================
# EDA TAB (FINAL CLEAN VERSION)
# =========================
if page == "📈 EDA":

    st.header("📊 Exploratory Data Analysis")

    file = st.file_uploader("Upload CSV", type=["csv"])
    
    if file:
        st.session_state.df = pd.read_csv(file)

    df = st.session_state.df

    # ✅ FIX: Check if df exists
    if df is not None:

        # =========================
        # DATA PREVIEW
        # =========================
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        col1, col2 = st.columns(2)
        sales_col = col1.selectbox("Select Sales Column", numeric_cols, key="eda_sales")
        spend_cols = col2.multiselect("Select Spend Columns", numeric_cols, key="eda_spend")

        if sales_col and spend_cols:

            # =========================
            # TIME SERIES
            # =========================
            st.subheader("📈 Sales & Spend Over Time")

            ts_df = df[[sales_col] + spend_cols].copy()
            st.line_chart(ts_df)

            # =========================
            # CORRELATION MATRIX
            # =========================
            st.subheader("🔥 Correlation Matrix")

            corr = df[[sales_col] + spend_cols].corr()
            st.dataframe(corr, use_container_width=True)

            # =========================
            # CORRELATION WITH SALES
            # =========================
            st.subheader("📊 Correlation with Sales")

            sales_corr = corr[sales_col].drop(sales_col).sort_values(ascending=False)
            st.bar_chart(sales_corr)

            # =========================
            # VIF (MULTICOLLINEARITY)
            # =========================
            st.subheader("⚠️ VIF (Multicollinearity Check)")

            from statsmodels.stats.outliers_influence import variance_inflation_factor

            X = df[spend_cols].dropna()

            vif_data = pd.DataFrame()
            vif_data["Channel"] = spend_cols
            vif_data["VIF"] = [
                variance_inflation_factor(X.values, i)
                for i in range(len(spend_cols))
            ]

            st.dataframe(vif_data, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - VIF < 5 → Good  
            - VIF 5–10 → Moderate multicollinearity  
            - VIF > 10 → High multicollinearity (problematic)
            """)

            # =========================
            # AUTO INSIGHTS
            # =========================
            st.subheader("🧠 Auto Insights")

            insights = []

            for ch, val in sales_corr.head(3).items():
                insights.append(f"📈 {ch} strongly correlates with sales ({val:.2f})")

            for ch, val in sales_corr[sales_corr < 0].items():
                insights.append(f"🚨 {ch} negatively correlates with sales ({val:.2f}) — check lag or tracking issues")

            for _, row in vif_data.iterrows():
                if row["VIF"] > 10:
                    insights.append(f"⚠️ {row['Channel']} has high multicollinearity (VIF={row['VIF']:.1f})")
                elif row["VIF"] > 5:
                    insights.append(f"⚠️ {row['Channel']} has moderate multicollinearity (VIF={row['VIF']:.1f})")

            if insights:
                for ins in insights:
                    st.markdown(f"- {ins}")
            else:
                st.info("No strong insights detected")

    else:
        # ✅ FIX: Correct placement
        st.info("👆 Upload a CSV file to begin EDA")
        
# =========================
# FUNCTIONS
# =========================
def adstock_transform(x, decay):
    result = np.zeros(len(x))
    for t in range(len(x)):
        result[t] = x[t] + (decay * result[t-1] if t > 0 else 0)
    return result

def hill_saturation(x, alpha, gamma):
    return (x**alpha) / (x**alpha + gamma**alpha)

# =========================
# INSTRUCTIONS TAB
# =========================
if page == "📘 Help":
    st.header("📘 How to Use This MMM Tool")

    st.markdown("""

### 🔁 Adstock (Decay Rate)
Adstock models the **carryover effect of advertising over time**.

- **Low decay (0.0 – 0.3)** → Immediate impact, fades quickly  
- **Medium decay (0.3 – 0.6)** → Moderate carryover  
- **High decay (0.6 – 0.9)** → Long-lasting impact  

👉 Use higher values for **brand channels (TV, YouTube)**  
👉 Use lower values for **performance channels (Search, Meta Ads)**  

---

### 📈 Alpha (Response Curve Shape)
Alpha controls the **steepness of the response curve**.

- **Low alpha (~0.5 – 1.0)** → Gradual increase in returns  
- **Medium alpha (~1.0 – 2.0)** → Balanced response  
- **High alpha (>2.0)** → Sharp increase after threshold  

👉 Higher alpha means **strong response after a certain spend level**

---

### 📉 Gamma (Saturation Level)
Gamma determines **when diminishing returns begin**.

- **Low gamma** → Saturation happens early  
- **High gamma** → Saturation happens later  

👉 Low gamma = good for **performance channels**  
👉 High gamma = suitable for **brand-building channels**

---

### 🎯 Practical Guidelines

| Channel Type     | Decay | Alpha | Gamma |
|------------------|------|-------|-------|
| TV / Branding    | High | Medium | High |
| Digital Display  | Medium | Medium | Medium |
| Search / Performance | Low | High | Low |

---

### ⚠️ Model Diagnostics

- If **Train R² is much higher than Test R²** → Model is likely **overfitting**
- If **ROI values look unrealistic** → Check:
  - data quality  
  - extreme parameter values  

---

### 💡 What You Can Do With This Tool

- 📊 Measure channel contribution  
- 📈 Understand diminishing returns  
- 💰 Optimize budget allocation  
- 🎯 Run scenario simulations  

---

Built with ❤️ by Anshul
""")

# =========================
# =========================
# MAIN DASHBOARD
# =========================
if page == "📊 Dashboard":

    df = st.session_state.df

    if df is None:
        st.info("👆 Upload a dataset in EDA section first")
        st.stop()

    st.markdown("### 📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    col1, col2 = st.columns(2)
    sales_col = col1.selectbox("Sales Column", numeric_cols)
    spend_cols = col2.multiselect("Spend Columns", numeric_cols)

    # =========================
    # VALIDATION
    # =========================
    if not sales_col or not spend_cols:
        st.warning("⚠️ Please select sales and spend columns")
        st.stop()

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.header("⚙️ Controls")

    params = {}
    for col in spend_cols:
        st.sidebar.subheader(col)
        params[col] = {
            "decay": st.sidebar.slider(f"{col} Decay", 0.0, 1.0, 0.5, key=col+"d"),
            "alpha": st.sidebar.slider(f"{col} Alpha", 0.1, 3.0, 1.5, key=col+"a"),
            "gamma": st.sidebar.slider(f"{col} Gamma", 1.0, 200.0, 100.0, key=col+"g")
        }

    ridge_alpha = st.sidebar.slider("Ridge Alpha", 0.1, 10.0, 1.0)

    # =========================
    # RUN MODEL
    # =========================
    if st.button("🚀 Run MMM Model"):

        st.markdown("### 🤖 MMM Engine Initializing...")
        try:
            st.video(open("robot.mp4", "rb").read())
        except:
            st.info("Add robot.mp4 for animation")

        status = st.empty()
        progress = st.progress(0)

        steps = [
            "🔧 Applying Adstock...",
            "📉 Applying Saturation...",
            "🧠 Training Model...",
            "📊 Calculating Contributions..."
        ]

        for i, step in enumerate(steps):
            status.markdown(f"<div class='card'>{step}</div>", unsafe_allow_html=True)
            time.sleep(0.8)
            progress.progress((i+1)/len(steps))

        # =========================
        # FEATURE ENGINEERING
        # =========================
        transformed_cols = []

        for col in spend_cols:
            p = params[col]
            ad = adstock_transform(df[col].values, p["decay"])
            fr = hill_saturation(ad, p["alpha"], p["gamma"])
            new_col = col+"_t"
            df[new_col] = fr
            transformed_cols.append(new_col)

        df["trend"] = np.arange(len(df))
        df["sin"] = np.sin(2*np.pi*df.index/12)
        df["cos"] = np.cos(2*np.pi*df.index/12)

        features = transformed_cols + ["trend","sin","cos"]

        X = df[features]
        y = df[sales_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # =========================
        # MULTI MODEL TRAINING
        # =========================
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.metrics import mean_absolute_percentage_error

        models = {
            "Ridge": Ridge(alpha=ridge_alpha),
            "Lasso": Lasso(alpha=0.1),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
        }

        results = []
        trained_models = {}

        for name, m in models.items():
            m.fit(X_train, y_train)

            train_r2 = m.score(X_train, y_train)
            test_r2 = m.score(X_test, y_test)

            preds = m.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, preds)

            results.append({
                "Model": name,
                "Train R²": round(train_r2, 3),
                "Test R²": round(test_r2, 3),
                "MAPE": round(mape, 3)
            })

            trained_models[name] = m

        results_df = pd.DataFrame(results)

        st.subheader("🧠 Model Comparison")
        st.dataframe(results_df, use_container_width=True)

        best_model_name = results_df.sort_values("MAPE").iloc[0]["Model"]
        model = trained_models[best_model_name]

        st.success(f"✅ Best Model Selected: {best_model_name}")

        train_r2 = results_df.loc[results_df["Model"] == best_model_name, "Train R²"].values[0]
        test_r2 = results_df.loc[results_df["Model"] == best_model_name, "Test R²"].values[0]

        # =========================
        # CONTRIBUTION
        # =========================
        coeffs = model.coef_
        contrib = pd.DataFrame(X_scaled * coeffs, columns=features)
        total_contrib = contrib.sum()

        media_contrib = total_contrib[transformed_cols]
        media_pct = (media_contrib / media_contrib.sum()) * 100

        baseline = model.intercept_ + total_contrib["trend"] + total_contrib["sin"] + total_contrib["cos"]

        total_sales = y.sum()
        baseline_pct = baseline / total_sales * 100
        promo_pct = 100 - baseline_pct

        # ROI
        spend_totals = df[spend_cols].sum()
        roi = {}
        for i, col in enumerate(spend_cols):
            roi[col] = media_contrib.iloc[i] / spend_totals[col]
        roi_df = pd.Series(roi)

        # =========================
        # DASHBOARD OUTPUT
        # =========================
        st.success("✅ MMM Analysis Ready")

        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("Train R²", round(train_r2,3)),
            ("Test R²", round(test_r2,3)),
            ("Baseline %", f"{baseline_pct:.1f}%"),
            ("Promo %", f"{promo_pct:.1f}%")
        ]

        for col, (label, value) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                <h3>{label}</h3>
                <h2>{value}</h2>
                </div>
                """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Contribution")
            st.bar_chart(media_pct)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📈 ROI")
            st.bar_chart(roi_df)
            st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # STORE RESULTS
        # =========================
        st.session_state.model_results = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "params": params,
            "spend_cols": spend_cols,
            "roi_df": roi_df,
            "media_pct": media_pct
        }

        # =========================
        # DOWNLOAD
        # =========================
        st.download_button(
            "📥 Download Results",
            data=media_pct.to_csv(),
            file_name="mmm_results.csv"
        )
# =========================
# SIMULATOR (IMPROVED)
# =========================        
if page == "🎯 Simulator":

    if st.session_state.model_results is None:
        st.warning("Run model first from Dashboard")
        st.stop()

    data = st.session_state.model_results
    df = st.session_state.df

    model = data["model"]
    scaler = data["scaler"]
    features = data["features"]
    params = data["params"]
    spend_cols = data["spend_cols"]

    st.header("🎯 Scenario Simulator")

    # =========================
    # USER INPUT
    # =========================
    scenario = {}
    for col in spend_cols:
        change = st.slider(f"{col} Spend Change (%)", -50, 100, 0)
        scenario[col] = df[col].mean() * (1 + change/100)

    scenario_df = pd.DataFrame([scenario])

    # =========================
    # TRANSFORM SCENARIO
    # =========================
    for col in spend_cols:
        p = params[col]
        ad = adstock_transform(scenario_df[col].values, p["decay"])
        scenario_df[col] = hill_saturation(ad, p["alpha"], p["gamma"])

    scenario_X = scaler.transform(
        scenario_df.reindex(columns=features, fill_value=0)
    )

    predicted = model.predict(scenario_X)[0]

    # =========================
    # BASELINE CALCULATION (NEW)
    # =========================
    baseline_input = df[spend_cols].mean().to_frame().T

    for col in spend_cols:
        p = params[col]
        ad = adstock_transform(baseline_input[col].values, p["decay"])
        baseline_input[col] = hill_saturation(ad, p["alpha"], p["gamma"])

    baseline_X = scaler.transform(
        baseline_input.reindex(columns=features, fill_value=0)
    )

    baseline_pred = model.predict(baseline_X)[0]

    # =========================
    # OUTPUT (COMPARISON)
    # =========================
    delta = predicted - baseline_pred

    col1, col2 = st.columns(2)
    col1.metric("Baseline Sales", round(baseline_pred, 2))
    col2.metric("New Sales", round(predicted, 2), delta=round(delta, 2))
# =========================
# =========================
# OPTIMIZER (IMPROVED)
# =========================
if page == "💰 Optimizer":

    if st.session_state.model_results is None:
        st.warning("Run model first from Dashboard")
        st.stop()

    data = st.session_state.model_results
    df = st.session_state.df

    model = data["model"]
    scaler = data["scaler"]
    features = data["features"]
    params = data["params"]
    spend_cols = data["spend_cols"]

    st.header("💰 Budget Optimizer")

    total_budget = st.number_input("Total Budget", value=1000000)

    # =========================
    # BASELINE SPEND DISTRIBUTION
    # =========================
    base_spend = df[spend_cols].mean()

    # =========================
    # INITIAL ALLOCATION (START POINT)
    # =========================
    weights = base_spend / base_spend.sum()
    alloc = weights * total_budget

    # =========================
    # TRANSFORM ALLOCATION THROUGH MODEL
    # =========================
    alloc_df = pd.DataFrame([alloc])

    for col in spend_cols:
        p = params[col]
        ad = adstock_transform(alloc_df[col].values, p["decay"])
        alloc_df[col] = hill_saturation(ad, p["alpha"], p["gamma"])

    alloc_X = scaler.transform(
        alloc_df.reindex(columns=features, fill_value=0)
    )

    predicted_sales = model.predict(alloc_X)[0]

    # =========================
    # OUTPUT
    # =========================
    opt_df = pd.DataFrame({
        "Channel": spend_cols,
        "Budget": alloc.values
    })

    st.metric("📊 Expected Sales from Allocation", round(predicted_sales, 2))

    st.dataframe(opt_df, use_container_width=True)
    st.bar_chart(opt_df.set_index("Channel"))
