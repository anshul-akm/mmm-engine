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
st.image("logo.png", width=600)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h1>Anshul MMM Intelligence 🚀</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:gray;'>AI-powered Marketing Mix Modeling</h4>", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📊 Dashboard", "📘 Learn MMM"])

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
with tab2:
    st.header("📘 MMM Guide")

    st.markdown("""
### 🔁 Adstock
Carryover of ads  
Higher = longer impact  

### 📈 Alpha
Curve steepness  

### 📉 Gamma
Diminishing returns  

---

### 🎯 Tips
- TV → high decay  
- Digital → medium  
- Performance → low gamma  

---

### ⚠️ Model Health
Train R² >> Test R² → overfitting  
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

            for col in spend_cols:
                p = params[col]
                x = np.linspace(0, df[col].max()*1.5, 50)
                y_curve = hill_saturation(adstock_transform(x, p["decay"]), p["alpha"], p["gamma"])

                curve = pd.DataFrame({"Spend": x, "Response": y_curve})
                st.markdown(f"**{col}**")
                st.line_chart(curve.set_index("Spend"))

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
            # BUDGET OPTIMIZER
            # =========================
            st.subheader("💰 Budget Optimizer")

            total_budget = st.number_input("Total Budget", value=1000000)

            weights = roi_df.clip(lower=0)
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
