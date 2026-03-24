import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Anshul MMM Intelligence", layout="wide")

# =========================
# PASSWORD
# =========================
password = st.text_input("🔒 Enter Password", type="password")
if password != "anshul123":
    st.stop()

# =========================
# CSS (AESTHETIC UI)
# =========================
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
h1, h2, h3 {text-align:center; color:#00FFAA;}
.card {
    padding:20px;
    border-radius:12px;
    background-color:#111;
    box-shadow:0px 0px 10px rgba(0,255,170,0.2);
    margin-bottom:15px;
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
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊 MMM Dashboard", "📊 EDA", "❓ Help"])

# =========================
# EDA TAB (FINAL CLEAN VERSION)
# =========================
with tab2:

    st.header("📊 Exploratory Data Analysis")

    file = st.file_uploader("Upload CSV for EDA", type=["csv"], key="eda_upload")

    if file:
        df = pd.read_csv(file)

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

            # Top correlations
            for ch, val in sales_corr.head(3).items():
                insights.append(f"📈 {ch} strongly correlates with sales ({val:.2f})")

            # Negative correlation
            for ch, val in sales_corr[sales_corr < 0].items():
                insights.append(f"🚨 {ch} negatively correlates with sales ({val:.2f}) — check lag or tracking issues")

            # VIF warnings
            for _, row in vif_data.iterrows():
                if row["VIF"] > 10:
                    insights.append(f"⚠️ {row['Channel']} has high multicollinearity (VIF={row['VIF']:.1f})")
                elif row["VIF"] > 5:
                    insights.append(f"⚠️ {row['Channel']} has moderate multicollinearity (VIF={row['VIF']:.1f})")

            # Display insights
            if insights:
                for ins in insights:
                    st.markdown(f"- {ins}")
            else:
                st.info("No strong insights detected")

    else:
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
with tab3:
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
# MAIN DASHBOARD
# =========================
with tab1:

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        col1, col2 = st.columns(2)
        sales_col = col1.selectbox("Sales Column", numeric_cols)
        spend_cols = col2.multiselect("Spend Columns", numeric_cols)

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

            # 🎬 Animation
            st.markdown("### 🤖 MMM Engine Initializing...")
            try:
                st.video(open("robot.mp4", "rb").read())
            except:
                st.info("Add robot.mp4 for animation")

            # ⏳ Cinematic Progress
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
            # MODEL
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

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model = Ridge(alpha=ridge_alpha)
            model.fit(X_train, y_train)

            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)

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
            # DASHBOARD
            # =========================
            st.success("✅ MMM Analysis Ready")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train R²", round(train_r2,3))
            m2.metric("Test R²", round(test_r2,3))
            m3.metric("Baseline %", f"{baseline_pct:.1f}%")
            m4.metric("Promo %", f"{promo_pct:.1f}%")

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
            # RESPONSE CURVES
            # =========================
            st.subheader("📈 Response Curves")
            cols = st.columns(2)  # show 2 per row

            for i, col in enumerate(spend_cols):
                p = params[col]
                x = np.linspace(0, df[col].max()*1.5, 50)
                y_curve = hill_saturation(adstock_transform(x, p["decay"]), p["alpha"], p["gamma"])
                curve = pd.DataFrame({"Spend": x, "Response": y_curve})
                with cols[i % 2]:
                    st.markdown(f"**{col}**")
                    st.line_chart(curve.set_index("Spend"), height=250)

            # =========================
            # SCENARIO SIMULATOR
            # =========================
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

            
            # =========================
            # BUDGET OPTIMIZER (FIXED)
            # =========================
            st.subheader("💰 Budget Optimizer")

            total_budget = st.number_input("Total Budget", value=1000000)

            roi_clean = roi_df.replace([np.inf, -np.inf], np.nan).fillna(0)

            weights = roi_clean.clip(lower=0)

            if weights.sum() == 0:
                st.warning("⚠️ Cannot optimize budget (all ROI ≤ 0). Showing equal allocation.")
                weights = pd.Series([1/len(spend_cols)]*len(spend_cols), index=spend_cols)
            else:
                weights = weights / weights.sum()

            alloc = weights * total_budget

            opt_df = pd.DataFrame({
                "Channel": spend_cols,
                "Budget": alloc.values
            })

            st.dataframe(opt_df, use_container_width=True)
            st.bar_chart(opt_df.set_index("Channel"))
            # =========================
            # DOWNLOAD
            # =========================
            st.download_button("📥 Download Results", data=media_pct.to_csv(), file_name="mmm_results.csv")

    else:
        st.info("👆 Upload your dataset to begin")
