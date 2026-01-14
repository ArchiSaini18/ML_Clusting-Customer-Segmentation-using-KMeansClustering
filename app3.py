# =================================================
# IMPORTS
# =================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="🎯",
    layout="wide"
)

# =================================================
# BASIC CSS
# =================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #E06B80, #ff9a9e); }
.stButton>button { background:#ff1493; color:white; border-radius:12px; }
.prediction-card {
    background:white; padding:25px; border-radius:20px;
    box-shadow:0 10px 30px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# =================================================
# USER DATABASE
# =================================================
USER_DB = "users.csv"
if not os.path.exists(USER_DB):
    pd.DataFrame(columns=["username","password"]).to_csv(USER_DB,index=False)

def load_users():
    return pd.read_csv(USER_DB)

def save_user(u,p):
    df = load_users()
    df.loc[len(df)] = [u,p]
    df.to_csv(USER_DB,index=False)

# =================================================
# SESSION STATE
# =================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# =================================================
# LOGIN PAGE
# =================================================
def login_page():
    st.title("🎯 Customer Segmentation System")
    tab1, tab2 = st.tabs(["🔐 Login","📝 Sign Up"])

    users = load_users()

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in users.username.values and p == users.loc[users.username==u,"password"].values[0]:
                st.session_state.logged_in = True
                st.session_state.current_user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        c = st.text_input("Confirm Password", type="password")
        if st.button("Create Account"):
            if p == c:
                save_user(u,p)
                st.session_state.logged_in = True
                st.session_state.current_user = u
                st.rerun()
            else:
                st.error("Passwords do not match")

# =================================================
# DATA + MODEL
# =================================================
@st.cache_data
def load_data():
    df = pd.read_csv("customer_segmentation.csv")
    df.dropna(inplace=True)
    df["Age"] = 2026 - df["Year_Birth"]
    df["Total_Spending"] = df[
        ["MntWines","MntFruits","MntMeatProducts",
         "MntFishProducts","MntSweetProducts","MntGoldProds"]
    ].sum(axis=1)
    return df

@st.cache_resource
def train_model(df):
    features = ["Age","Income","Total_Spending",
                "NumWebPurchases","NumStorePurchases",
                "NumWebVisitsMonth","Recency"]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    model = KMeans(n_clusters=6, random_state=42)
    clusters = model.fit_predict(X)

    return model, scaler, features, clusters

# =================================================
# MAIN DASHBOARD
# =================================================
def prediction_page():
    st.title("📊 Customer Segmentation Dashboard")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    df = load_data()
    model, scaler, features, clusters = train_model(df)
    df["Cluster"] = clusters
    cluster_summary = df.groupby("Cluster")[features].mean()


    # =================================================
    # 🔮 INPUT SECTION
    # =================================================
    st.markdown("---")
    st.markdown("## 🔮 Customer Cluster Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 100, 35)
        income = st.number_input("Income", 0, 200000, 50000)
        spending = st.number_input("Total Spending", 0, 5000, 500)
    with col2:
        web = st.number_input("Web Purchases", 0, 30, 5)
        store = st.number_input("Store Purchases", 0, 30, 5)
    with col3:
        visits = st.number_input("Web Visits", 0, 30, 5)
        recency = st.number_input("Recency", 0, 365, 30)

# =================================================
# PREDICTION
# =================================================

    if st.button("Predict Cluster"):
        input_df = pd.DataFrame([[age,income,spending,web,store,visits,recency]],
                                columns=features)

        cluster = model.predict(scaler.transform(input_df))[0]

        st.markdown(f"""
        <div class="prediction-card">
            <h2>🎯 Predicted Cluster: {cluster}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📋 Cluster Summary")
        summary = cluster_summary.copy()
        summary["Customers"] = df["Cluster"].value_counts().sort_index().values

        st.dataframe(
            summary.style
            .format({"Income":"${:,.0f}","Total_Spending":"${:,.0f}"})
            .background_gradient(cmap="RdPu"),
            use_container_width=True
        )


    # =================================================
    # 📊 GRAPHS AFTER ALL INPUTS
    # =================================================
    st.markdown("## 📊 Your Profile vs Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], kde=True, ax=ax)
        ax.axvline(age, color="red", linestyle="--", label="Your Age")
        ax.legend()
        ax.set_title("Age Comparison")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df["Income"], kde=True, ax=ax)
        ax.axvline(income, color="red", linestyle="--", label="Your Income")
        ax.legend()
        ax.set_title("Income Comparison")
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots()
        sns.histplot(df["Total_Spending"], kde=True, ax=ax)
        ax.axvline(spending, color="red", linestyle="--", label="Your Spending")
        ax.legend()
        ax.set_title("Spending Comparison")
        st.pyplot(fig)


   
# =================================================
# RUN APP
# =================================================
if st.session_state.logged_in:
    prediction_page()
else:
    login_page()
