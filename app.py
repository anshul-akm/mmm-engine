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
# PASSWORD PROTECTION
# =========================
password = st.text_input("Enter Password", type="password")
if password != "anshul123":
    st.warning("🔒 Enter password to access app")
    st.stop()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #00FFAA;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER (LOGO + TITLE)
# =========================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=300)

st.markdown("""
<h1 style='text-align:center;'>Anshul MMM Intelligence 🚀</h1>
<h4 style='text-align:center; color:gray;'>AI-powered Marketing Mix Modeling</h4>
""", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📊 MMM Dashboard", "📘 Instructions"])

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
# TAB 2: INSTRUCTIONS
# =========================
with tab2:

    st.header("📘 How to Use This MMM Tool")

    st.markdown("""
### 1. Select Columns
- **Sales column** → Your target variable  
- **Spend columns** → Media channels  

---

### 2. Understanding Parameters

#### 🔁 Adstock Decay
- Controls carryover effect of media
- **0 → No memory**
- **1 → Long memory**

👉 Higher decay = ads impact lasts longer

---

#### 📈 Alpha (Shape)
- Controls curve steepness
- **Low alpha → gradual effect**
- **High alpha → sharp response**

---

#### 📉 Gamma (Saturation point)
- Controls diminishing returns
- **Low gamma → saturates early**
- **High gamma → saturates late**

---

### 🎯 Practical Tips

- TV → High decay (0.7–0.9)  
- Digital → Medium decay (0.3–0.6)  
- Performance → Lower gamma  
- Branding → Higher gamma  

---

### ⚠️ Model Interpretation

- **Train R² >> Test R² → Overfitting**
- Focus on:
  - Contribution %
  - ROI
  - Scenario results  

---

### 💡 What This Tool Helps You Do

✅ Budget allocation  
✅ Channel efficiency analysis  
✅ Forecasting  
✅ Scenario planning  

---

Built with ❤️ by Anshul Analytics
""")

# =========================
# TAB 1: MAIN APP
# =========================
with tab1:

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        sales_col = st.selectbox("Select SALES column", numeric_cols)
        spend_cols = st.multiselect("Select SPEND columns", numeric_cols)

        # =========================
        # SIDEBAR PARAMETERS
        # =========================
        st.sidebar.header("Channel Parameters")

        channel_params = {}
        for col in spend_cols:
            st.sidebar.subheader(col)
            decay = st.sidebar.slider(f"{col} Decay", 0.0, 1.0, 0.5, key=col+"_d")
            alpha = st.sidebar.slider(f"{col} Alpha", 0.1, 3.0, 1.5, key=col+"_a")
            gamma = st.sidebar.slider(f"{col} Gamma", 1.0, 200.0, 100.0, key=col+"_g")

            channel_params[col] = {"decay": decay, "alpha": alpha, "gamma": gamma}

        ridge_alpha = st.sidebar.slider("Ridge Alpha", 0.1, 10.0, 1.0)

        # =========================
        # RUN MODEL
        # =========================
        if st.button("🚀 Run MMM Model"):

            # 🎬 Animation
            st.markdown("### 🤖 MMM Engine Booting Up...")
            try:
                video_file = open("robot.mp4", "rb")
                st.video(video_file.read())
            except:
                st.info("Upload robot.mp4 for animation")

            # ⏳ Progress storytelling
            status = st.empty()
            progress = st.progress(0)

            steps = [
                "🔧 Applying Adstock...",
                "📉 Applying Saturation...",
                "🧠 Training Model...",
                "📊 Calculating Contributions..."
            ]

            for i, step in enumerate(steps):
                status.info(step)
                time.sleep(1)
                progress.progress((i+1)/len(steps))

            # =========================
            # MODEL
            # =========================
            transformed_cols = []

            for col in spend_cols:
                params = channel_params[col]
                ad = adstock_transform(df[col].values, params["decay"])
                fr = hill_saturation(ad, params["alpha"], params["gamma"])

                new_col = col + "_transformed"
                df[new_col] = fr
                transformed_cols.append(new_col)

            df["trend"] = np.arange(len(df))
            df["sin"] = np.sin(2*np.pi*df.index/12)
            df["cos"] = np.cos(2*np.pi*df.index/12)

            features = transformed_cols + ["trend", "sin", "cos"]

            X = df[features]
            y = df[sales_col]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            model = Ridge(alpha=ridge_alpha)
            model.fit(X_train, y_train)

            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)

            # OLS
            X_ols = sm.add_constant(X_scaled)
            ols = sm.OLS(y, X_ols).fit()

            coeffs = model.coef_
            contributions = X_scaled * coeffs
            contrib_df = pd.DataFrame(contributions, columns=features)

            total_contribution = contrib_df.sum()
            media_contribution = total_contribution[transformed_cols]
            media_pct = (media_contribution / media_contribution.sum()) * 100

            roi = media_contribution / df[spend_cols].sum().values
            roi_df = pd.Series(roi, index=spend_cols)

            status.success("✅ Model Completed!")

            # =========================
            # DASHBOARD
            # =========================
            st.success("🚀 MMM Analysis Ready")

            col1, col2, col3 = st.columns(3)
            col1.metric("Train R²", round(train_r2, 3))
            col2.metric("Test R²", round(test_r2, 3))
            col3.metric("Channels", len(spend_cols))

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Contribution")
                st.bar_chart(media_pct)

            with col2:
                st.subheader("📈 ROI")
                st.bar_chart(roi_df)

            st.subheader("📉 P-values")
            st.write(ols.pvalues)

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
                params = channel_params[col]
                ad = adstock_transform(scenario_df[col].values, params["decay"])
                scenario_df[col] = hill_saturation(ad, params["alpha"], params["gamma"])

            scenario_X = scaler.transform(
                scenario_df.reindex(columns=features, fill_value=0)
            )

            predicted = model.predict(scenario_X)[0]

            st.metric("📊 Predicted Sales (Scenario)", round(predicted, 2))

            # =========================
            # DOWNLOAD
            # =========================
            st.download_button(
                "📥 Download Results",
                data=media_pct.to_csv(),
                file_name="mmm_results.csv"
            )

    else:
        st.info("👆 Upload a CSV file to start")
