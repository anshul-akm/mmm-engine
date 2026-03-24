import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
import time
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Anshul MMM Intelligence", layout="wide")

# =========================
# PASSWORD (IMPROVED)
# =========================
password = st.text_input("🔒 Enter Password", type="password")
if password != os.getenv("APP_PASSWORD", "anshul123"):
    st.stop()

# =========================
# CSS (UPGRADED PREMIUM UI)
# =========================
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
h1, h2, h3 {text-align:center; color:#00FFAA;}

.card {
    padding:20px;
    border-radius:16px;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    box-shadow:0px 0px 20px rgba(0,255,170,0.25);
    margin-bottom:15px;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.02);
}

[data-testid="stMetric"] {
    background-color:#111;
    padding:10px;
    border-radius:10px;
    box-shadow:0px 0px 10px rgba(0,255,170,0.2);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.image("logo.png", width=1500)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h1>Welcome to MMM Engine 🚀</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:gray;'>AI-powered Marketing Mix Modeling by Anshul</h4>", unsafe_allow_html=True)

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

def detect_best_lag(x, y, max_lag=12):
    corrs = []
    for lag in range(max_lag):
        shifted = np.roll(x, lag)
        corr = np.corrcoef(shifted[lag:], y[lag:])[0,1]
        corrs.append(corr)
    return np.argmax(corrs), max(corrs)

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊 MMM Dashboard", "📊 EDA", "❓ Help"])

# =========================
# EDA TAB
# =========================
with tab2:

    st.header("📊 Exploratory Data Analysis")

    file = st.file_uploader("Upload CSV for EDA", type=["csv"], key="eda_upload")

    if file:
        df = pd.read_csv(file)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        col1, col2 = st.columns(2)
        sales_col = col1.selectbox("Select Sales Column", numeric_cols, key="eda_sales")
        spend_cols = col2.multiselect("Select Spend Columns", numeric_cols, key="eda_spend")

        if sales_col and spend_cols:

            # DATA QUALITY
            st.subheader("🧪 Data Quality Check")
            issues = []
            if df.isnull().sum().sum() > 0:
                issues.append("Missing values detected")

            for col in spend_cols:
                if df[col].std() == 0:
                    issues.append(f"{col} has no variation")

            if issues:
                for i in issues:
                    st.warning(i)
            else:
                st.success("Data looks clean")

            # TIME SERIES
            st.subheader("📈 Sales & Spend Over Time")
            st.line_chart(df[[sales_col] + spend_cols])

            # CORRELATION
            st.subheader("🔥 Correlation Matrix")
            corr = df[[sales_col] + spend_cols].corr()
            st.dataframe(corr)

            # LAG DETECTION
            st.subheader("⏳ Lag Detection")
            for col in spend_cols:
                lag, c = detect_best_lag(df[col].values, df[sales_col].values)
                st.markdown(f"- **{col}** → Lag {lag} (corr {c:.2f})")

# =========================
# HELP TAB
# =========================
with tab3:
    st.header("📘 MMM Guide")
    st.write("Use adstock, saturation, and regression to understand marketing impact.")

# =========================
# MAIN DASHBOARD
# =========================
with tab1:

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        col1, col2 = st.columns(2)
        sales_col = col1.selectbox("Sales Column", numeric_cols)
        spend_cols = col2.multiselect("Spend Columns", numeric_cols)

        # SIDEBAR
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

        if st.button("🚀 Run MMM Model"):

            # SAFE ANIMATION
            st.markdown("### 🤖 MMM Engine Initializing...")
            if os.path.exists("robot.mp4"):
                st.video(open("robot.mp4", "rb").read())
            else:
                st.info("Add robot.mp4 for animation")

            # PROGRESS
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
                time.sleep(0.6)
                progress.progress((i+1)/len(steps))

            # TRANSFORM
            transformed_cols = []
            for col in spend_cols:
                p = params[col]
                ad = adstock_transform(df[col].values, p["decay"])
                fr = hill_saturation(ad, p["alpha"], p["gamma"])
                df[col+"_t"] = fr
                transformed_cols.append(col+"_t")

            df["trend"] = np.arange(len(df))
            df["sin"] = np.sin(2*np.pi*df.index/12)
            df["cos"] = np.cos(2*np.pi*df.index/12)

            features = transformed_cols + ["trend","sin","cos"]

            X = df[features]
            y = df[sales_col]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # TRAIN TEST
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model = Ridge(alpha=ridge_alpha)
            model.fit(X_train, y_train)

            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)

            # CROSS VALIDATION
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, test_idx in kf.split(X_scaled):
                model.fit(X_scaled[train_idx], y.iloc[train_idx])
                cv_scores.append(model.score(X_scaled[test_idx], y.iloc[test_idx]))

            cv_r2 = np.mean(cv_scores)

            # ORIGINAL CONTRIBUTION (KEPT)
            coeffs = model.coef_
            contrib = pd.DataFrame(X_scaled * coeffs, columns=features)
            total_contrib = contrib.sum()

            media_contrib = total_contrib[transformed_cols]
            media_pct = (media_contrib / media_contrib.sum()) * 100

            # NEW CONTRIBUTION
            base_pred = model.predict(X_scaled)

            inc_contrib = {}
            for i, col in enumerate(transformed_cols):
                X_temp = X_scaled.copy()
                X_temp[:, i] = 0
                pred_without = model.predict(X_temp)
                inc_contrib[spend_cols[i]] = (base_pred - pred_without).sum()

            inc_contrib = pd.Series(inc_contrib)
            inc_pct = inc_contrib / inc_contrib.sum() * 100

            # ROI
            spend_totals = df[spend_cols].sum()
            roi_df = media_contrib / spend_totals.replace(0, np.nan)

            # METRICS
            st.success("✅ MMM Analysis Ready")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train R²", round(train_r2,3))
            m2.metric("Test R²", round(test_r2,3))
            m3.metric("CV R²", round(cv_r2,3))
            m4.metric("Best Channel", roi_df.idxmax())

            # CHARTS
            st.subheader("📊 Contribution")
            st.bar_chart(media_pct)

            st.subheader("📊 Advanced Contribution")
            st.bar_chart(inc_pct)

            st.subheader("📈 ROI")
            st.bar_chart(roi_df)

            # RESPONSE CURVES (UNCHANGED)
            st.subheader("📈 Response Curves")
            cols = st.columns(2)

            for i, col in enumerate(spend_cols):
                p = params[col]
                x = np.linspace(0, df[col].max()*1.5, 50)
                y_curve = hill_saturation(adstock_transform(x, p["decay"]), p["alpha"], p["gamma"])
                curve = pd.DataFrame({"Spend": x, "Response": y_curve})
                with cols[i % 2]:
                    st.markdown(f"**{col}**")
                    st.line_chart(curve.set_index("Spend"))

            # SMART RECOMMENDATIONS
            st.subheader("🧠 Recommendations")
            for col in spend_cols:
                if roi_df[col] > roi_df.mean():
                    st.markdown(f"- 📈 Increase {col}")
                else:
                    st.markdown(f"- 📉 Reduce {col}")

            # EXECUTIVE SUMMARY
            st.subheader("📌 Executive Summary")
            best = roi_df.idxmax()
            worst = roi_df.idxmin()

            st.markdown(f"""
            <div class='card'>
            🔥 Best Channel: {best} <br>
            ⚠️ Weak Channel: {worst} <br>
            💡 Shift budget from {worst} → {best}
            </div>
            """, unsafe_allow_html=True)

            # AI INSIGHTS
            st.subheader("🤖 AI Insights")
            st.write(f"{best} is driving the highest ROI")
            st.write(f"{worst} is underperforming")

            if train_r2 - test_r2 > 0.1:
                st.warning("Model may be overfitting")

            # SCENARIO
            st.subheader("🎯 Scenario Simulator")
            scenario = {}

            for col in spend_cols:
                change = st.slider(f"{col} Spend Change (%)", -50, 100, 0)
                scenario[col] = df[col].mean() * (1 + change/100)

            scenario_df = pd.DataFrame([scenario])

            for col in spend_cols:
                p = params[col]
                ad = adstock_transform(scenario_df[col].values, p["decay"])
                scenario_df[col] = hill_saturation(ad, p["alpha"], p["gamma"])

            scenario_X = scaler.transform(
                scenario_df.reindex(columns=features, fill_value=0)
            )

            predicted = model.predict(scenario_X)[0]
            st.metric("📊 Predicted Sales", round(predicted,2))

            # DOWNLOAD
            st.download_button("📥 Download Results", data=media_pct.to_csv(), file_name="mmm_results.csv")

    else:
        st.info("Upload dataset to begin")
