import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import time
import base64

# =========================
# HEADER (LOGO + TITLE)
# =========================

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=400)

st.markdown(
    "<h2 style='text-align: center;'>Anshul's MMM Regressor Engine 🚀</h2>",
    unsafe_allow_html=True
)
st.set_page_config(
    page_title="Anshul MMM Engine",
    page_icon="🤖",
    layout="wide"
)
# =========================
# FUNCTIONS
# =========================

def adstock_transform(x, decay):
    result = np.zeros(len(x))
    for t in range(len(x)):
        if t == 0:
            result[t] = x[t]
        else:
            result[t] = x[t] + decay * result[t-1]
    return result


def hill_saturation(x, alpha, gamma):
    return (x**alpha) / (x**alpha + gamma**alpha)


# =========================
# FILE UPLOAD
# =========================

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    sales_col = st.selectbox("Select SALES column", df.columns)
    spend_cols = st.multiselect("Select SPEND columns", df.columns)

    # =========================
    # CHANNEL PARAMETERS
    # =========================

    st.sidebar.header("Channel Parameters")
    channel_params = {}

    for col in spend_cols:
        st.sidebar.subheader(col)

        decay = st.sidebar.slider(f"{col} Adstock Decay", 0.0, 1.0, 0.5, key=col+"_decay")
        alpha = st.sidebar.slider(f"{col} Hill Alpha", 0.1, 3.0, 1.5, key=col+"_alpha")
        gamma = st.sidebar.slider(f"{col} Hill Gamma", 1.0, 200.0, 100.0, key=col+"_gamma")

        channel_params[col] = {
            "decay": decay,
            "alpha": alpha,
            "gamma": gamma
        }

    st.sidebar.header("Model Parameter")
    ridge_alpha = st.sidebar.slider("Ridge Alpha", 0.1, 10.0, 1.0)

    # =========================
    # RUN MODEL
    # =========================

    if st.button("Run MMM Model"):

        # 🎬 Autoplay Video
        video_file = open("robot.mp4", "rb").read()

        st.markdown(
            f"""
            <video width="800" controls>
            <source src="data:video/mp4;base64,{base64.b64encode(video_file).decode()}" type="video/mp4">
            </video>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("⚙️ MMM Engine is processing..."):

            progress_bar = st.progress(0)

            # ⏳ Intentional delay
            for i in range(0, 101, 5):
                time.sleep(0.2)
                progress_bar.progress(i)

            st.write("🔧 Applying Adstock...")
            time.sleep(1)

            st.write("📉 Applying Saturation...")
            time.sleep(1)

            st.write("🧠 Training Model...")
            time.sleep(1)

            st.write("📊 Calculating Contributions...")
            time.sleep(1)

            # =========================
            # ACTUAL MODEL
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
            df["sin_season"] = np.sin(2 * np.pi * df.index / 12)
            df["cos_season"] = np.cos(2 * np.pi * df.index / 12)

            features = transformed_cols + ["trend", "sin_season", "cos_season"]

            X = df[features]
            y = df[sales_col]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = Ridge(alpha=ridge_alpha)
            model.fit(X_scaled, y)

            r2 = model.score(X_scaled, y)

            X_ols = sm.add_constant(X_scaled)
            ols = sm.OLS(y, X_ols).fit()

            coeffs = model.coef_
            contributions = X_scaled * coeffs
            contrib_df = pd.DataFrame(contributions, columns=features)

            total_contribution = contrib_df.sum()

            baseline = (
                model.intercept_
                + total_contribution["trend"]
                + total_contribution["sin_season"]
                + total_contribution["cos_season"]
            )

            media_contribution = total_contribution[transformed_cols]
            media_pct = (media_contribution / media_contribution.sum()) * 100

            progress_bar.empty()

        # =========================
        # OUTPUT
        # =========================

        st.success("🚀 MMM Engine Completed Successfully!")

        st.write("### R-squared")
        st.write(round(r2, 4))

        st.write("### Coefficients")
        st.write(dict(zip(features, coeffs)))

        st.write("### P-values")
        st.write(ols.pvalues)

        st.write("### Media Contribution %")
        st.write(media_pct)

        st.write("### Baseline Sales")
        st.write(round(baseline, 2))

else:
    st.write("👆 Please upload a CSV file to start")
