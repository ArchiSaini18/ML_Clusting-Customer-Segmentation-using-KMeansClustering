
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# -------------------------------------------------
# CSS Styling with Light Pink Theme
# -------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #ffe4e9; }
html, body, [class*="css"] { color: #2c2c2c !important; }
h1, h2, h3, h4, h5, h6, label, p, span { color: #2c2c2c !important; }

.stTextInput input {
    background-color: white;
    color: black !important;
    border-radius: 8px;
    border: 2px solid #ffb3c1;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: white !important;
    border-radius: 8px;
    border: 2px solid #ffb3c1;
}

.stSelectbox div[data-baseweb="select"] span {
    color: #2c2c2c !important;
    font-weight: 600;
}

div[role="listbox"] {
    background-color: #fff0f3 !important;
    border-radius: 8px;
}

div[role="listbox"] span {
    color: #2c2c2c !important;
}

div[role="listbox"] div:hover {
    background-color: #ffb3c1 !important;
}

.stButton > button {
    background-color: #ff69b4;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #ff1493;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    border: 2px solid #ffb3c1;
}

.stNumberInput input {
    background-color: white;
    color: black !important;
    border-radius: 8px;
    border: 2px solid #ffb3c1;
}

.prediction-box {
    background-color: #fff0f3;
    padding: 20px;
    border-radius: 15px;
    border: 3px solid #ff69b4;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# User Database (CSV)
# -------------------------------------------------
USER_DB = "users.csv"

if not os.path.exists(USER_DB):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_DB, index=False)

def load_users():
    return pd.read_csv(USER_DB)

def save_user(username, password):
    users = load_users()
    users.loc[len(users)] = [username, password]
    users.to_csv(USER_DB, index=False)

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# -------------------------------------------------
# LOGIN / SIGNUP PAGE
# -------------------------------------------------
def login_page():
    st.title("📊 Welcome to Customer Segmentation Dashboard")
    st.write("Analyze customer behavior and discover market segments")
    
    users_df = load_users()

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # -------- LOGIN --------
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if username in users_df.username.values:
                real_pass = users_df.loc[
                    users_df.username == username, "password"
                ].values[0]
                if password == real_pass:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.rerun()
                else:
                    st.error("Wrong password")
            else:
                st.error("User not found")

    # -------- SIGNUP --------
    with tab2:
        new_user = st.text_input("Choose Username", key="signup_user")
        new_pass = st.text_input("Choose Password", type="password", key="signup_pass")
        confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")

        if st.button("Create Account"):
            if new_user in users_df.username.values:
                st.error("Username already exists")
            elif new_pass != confirm:
                st.error("Passwords do not match")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters")
            else:
                save_user(new_user, new_pass)
                st.session_state.logged_in = True
                st.session_state.current_user = new_user
                st.rerun()

# -------------------------------------------------
# DATA PROCESSING
# -------------------------------------------------
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('customer_segmentation.csv')
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # Process date
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    
    # Create new features
    df['Age'] = 2026 - df['Year_Birth']
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']
    
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df["Total_Spending"] = df[spend_cols].sum(axis=1)
    
    df['Customer_Since'] = (pd.Timestamp("today") - df["Dt_Customer"]).dt.days
    
    # Campaign acceptance
    df['AcceptAny'] = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", 
                           "AcceptedCmp4", "AcceptedCmp5", "Response"]].sum(axis=1)
    df['AcceptAny'] = df["AcceptAny"].apply(lambda x: 1 if x > 0 else 0)
    
    # Age groups
    bins = [18, 30, 40, 50, 60, 70, 90]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    
    return df

@st.cache_data
def perform_clustering(_df):
    features = ["Age", "Income", "Total_Spending", "NumWebPurchases", 
                "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
    X = _df[features].copy()
    
    # Scaling
    sc = StandardScaler()
    x_scaled = sc.fit_transform(X)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=6, random_state=42)
    clusters = kmeans.fit_predict(x_scaled)
    
    return clusters, x_scaled, features, kmeans, sc

def get_cluster_description(cluster_num, cluster_data):
    """Generate detailed description for each cluster"""
    descriptions = {
        0: {
            "name": "💎 Premium High-Value Customers",
            "characteristics": [
                "High income and high spending",
                "Frequent purchasers across all channels",
                "Most valuable customer segment",
                "Strong brand loyalty"
            ],
            "color": "#FFD700"
        },
        1: {
            "name": "🛒 Digital Savvy Shoppers",
            "characteristics": [
                "High web purchases",
                "Low store purchases",
                "Tech-savvy and prefer online shopping",
                "Active online engagement"
            ],
            "color": "#FF69B4"
        },
        2: {
            "name": "🏪 Traditional Store Shoppers",
            "characteristics": [
                "Prefer in-store shopping",
                "Lower web purchases",
                "Moderate spending patterns",
                "Value personal shopping experience"
            ],
            "color": "#87CEEB"
        },
        3: {
            "name": "⚡ Active Engaged Customers",
            "characteristics": [
                "Low recency (recently purchased)",
                "Highly active and engaged",
                "Regular purchase patterns",
                "Strong customer relationship"
            ],
            "color": "#90EE90"
        },
        4: {
            "name": "💤 Dormant/At-Risk Customers",
            "characteristics": [
                "High recency (inactive)",
                "Haven't purchased recently",
                "Need re-engagement campaigns",
                "Risk of churn"
            ],
            "color": "#FFA500"
        },
        5: {
            "name": "💰 Budget-Conscious Shoppers",
            "characteristics": [
                "Low income segment",
                "Lowest spending levels",
                "Price-sensitive customers",
                "Seek value and deals"
            ],
            "color": "#DDA0DD"
        }
    }
    
    return descriptions.get(cluster_num, {
        "name": f"Cluster {cluster_num}",
        "characteristics": ["Customer segment"],
        "color": "#CCCCCC"
    })

# -------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------
def prediction_page(df, model, scaler, features):
    st.header("🔮 Customer Cluster Prediction")
    st.write("Enter customer details to predict their cluster segment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        income = st.number_input("Annual Income ($)", min_value=0, max_value=200000, value=50000, step=1000)
        total_spending = st.number_input("Total Spending ($)", min_value=0, max_value=5000, value=500, step=50)
    
    with col2:
        num_web = st.number_input("Web Purchases", min_value=0, max_value=30, value=5, step=1)
        num_store = st.number_input("Store Purchases", min_value=0, max_value=30, value=5, step=1)
    
    with col3:
        num_visits = st.number_input("Monthly Web Visits", min_value=0, max_value=30, value=5, step=1)
        recency = st.number_input("Days Since Last Purchase", min_value=0, max_value=365, value=30, step=1)
    
    if st.button("🎯 Predict Cluster", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'Total_Spending': [total_spending],
            'NumWebPurchases': [num_web],
            'NumStorePurchases': [num_store],
            'NumWebVisitsMonth': [num_visits],
            'Recency': [recency]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Predict cluster
        predicted_cluster = model.predict(input_scaled)[0]
        
        # Get cluster info
        cluster_info = get_cluster_description(predicted_cluster, df[df['Cluster'] == predicted_cluster])
        
        # Display prediction
        st.markdown("---")
        st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
        st.markdown(f"## Predicted Cluster: {predicted_cluster}")
        st.markdown(f"### {cluster_info['name']}")
        
        st.markdown("#### 📋 Cluster Characteristics:")
        for char in cluster_info['characteristics']:
            st.markdown(f"- {char}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show cluster statistics
        st.markdown("---")
        st.subheader("📊 Your Profile vs Cluster Average")
        
        cluster_data = df[df['Cluster'] == predicted_cluster][features].mean()
        
        comparison_df = pd.DataFrame({
            'Your Input': input_data.iloc[0].values,
            'Cluster Average': cluster_data.values
        }, index=features)
        
        st.dataframe(comparison_df.style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input vs Cluster Average")
            fig, ax = plt.subplots(figsize=(8, 6))
            x_pos = np.arange(len(features))
            width = 0.35
            
            ax.bar(x_pos - width/2, input_data.iloc[0].values, width, 
                   label='Your Input', color='#ff69b4', alpha=0.8)
            ax.bar(x_pos + width/2, cluster_data.values, width, 
                   label='Cluster Avg', color='#ffb3c1', alpha=0.8)
            
            ax.set_xlabel('Features')
            ax.set_ylabel('Values (Scaled)')
            ax.set_title('Comparison Chart')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.legend()
            ax.set_facecolor('#fff0f3')
            fig.patch.set_facecolor('#ffe4e9')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Cluster Distribution in Dataset")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#FFD700', '#FF69B4', '#87CEEB', '#90EE90', '#FFA500', '#DDA0DD']
            cluster_counts.plot(kind='bar', color=colors, ax=ax, alpha=0.8)
            
            # Highlight predicted cluster
            ax.patches[predicted_cluster].set_edgecolor('red')
            ax.patches[predicted_cluster].set_linewidth(3)
            
            ax.set_title("All Clusters (Your cluster highlighted)")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Number of Customers")
            ax.set_facecolor('#fff0f3')
            fig.patch.set_facecolor('#ffe4e9')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Show all cluster descriptions
        st.markdown("---")
        st.subheader("📚 All Cluster Descriptions")
        
        for i in range(6):
            cluster_info = get_cluster_description(i, None)
            with st.expander(f"Cluster {i}: {cluster_info['name']}"):
                for char in cluster_info['characteristics']:
                    st.write(f"• {char}")
                
                cluster_stats = df[df['Cluster'] == i][features].mean()
                st.write("**Average Statistics:**")
                st.dataframe(cluster_stats, use_container_width=True)

# -------------------------------------------------
# DASHBOARD PAGE
# -------------------------------------------------
def dashboard_page():
    st.title("📊 Customer Segmentation Dashboard")
    st.write(f"Welcome, **{st.session_state.current_user}**")
    
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.rerun()
    
    # Load data
    try:
        df = load_and_process_data()
    except FileNotFoundError:
        st.error("⚠️ Error: 'customer_segmentation.csv' file not found. Please upload the dataset.")
        return
    
    # Perform clustering and save to session state
    clusters, x_scaled, features, model, scaler = perform_clustering(df)
    df['Cluster'] = clusters
    st.session_state.trained_model = model
    st.session_state.scaler = scaler
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Select Analysis", [
        "Overview",
        "Predict Customer Cluster",
        "Distributions",
        "Correlations",
        "Segmentation Analysis",
        "Campaign Performance"
    ])
    
    # OVERVIEW PAGE
    if page == "Overview":
        st.header("📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Avg Income", f"${df['Income'].mean():,.0f}")
        with col3:
            st.metric("Avg Spending", f"${df['Total_Spending'].mean():,.0f}")
        with col4:
            st.metric("Avg Age", f"{df['Age'].mean():.1f}")
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    # PREDICTION PAGE
    elif page == "Predict Customer Cluster":
        prediction_page(df, model, scaler, features)
    
    # DISTRIBUTIONS PAGE
    elif page == "Distributions":
        st.header("📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['Age'], bins=30, kde=True, color='#ff69b4', ax=ax)
            ax.set_title("Age Distribution")
            ax.set_facecolor('#fff0f3')
            fig.patch.set_facecolor('#ffe4e9')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Income Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['Income'], bins=30, kde=True, color='#ffb3c1', ax=ax)
            ax.set_title("Income Distribution")
            ax.set_facecolor('#fff0f3')
            fig.patch.set_facecolor('#ffe4e9')
            st.pyplot(fig)
        
        st.subheader("Total Spending Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["Total_Spending"], bins=30, kde=True, color='#ff1493', ax=ax)
        ax.set_title("Total Spending Distribution")
        ax.set_facecolor('#fff0f3')
        fig.patch.set_facecolor('#ffe4e9')
        st.pyplot(fig)
    
    # CORRELATIONS PAGE
    elif page == "Correlations":
        st.header("🔗 Correlation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Income by Education Level")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x="Education", y="Income", data=df, palette="PiYG", ax=ax)
            plt.xticks(rotation=45, ha='right')
            ax.set_title("Income by Education Level")
            ax.set_facecolor('#fff0f3')
            fig.patch.set_facecolor('#ffe4e9')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Spending by Marital Status")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x="Marital_Status", y="Total_Spending", data=df, palette="RdPu", ax=ax)
            plt.xticks(rotation=45, ha='right')
            ax.set_title("Spending by Marital Status")
            ax.set_facecolor('#fff0f3')
            fig.patch.set_facecolor('#ffe4e9')
            st.pyplot(fig)
        
        st.subheader("Correlation Matrix")
        corr = df[["Income", "Age", "Recency", "Total_Spending", 
                   "NumWebPurchases", "NumStorePurchases"]].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='RdPu', ax=ax)
        ax.set_title("Correlation Matrix")
        fig.patch.set_facecolor('#ffe4e9')
        st.pyplot(fig)
        
        st.subheader("Income by Education and Marital Status")
        pivot_income = df.pivot_table(values='Income', index='Education', 
                                      columns='Marital_Status', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_income, annot=True, fmt=".0f", cmap="RdPu", ax=ax)
        ax.set_title("Average Income by Education and Marital Status")
        fig.patch.set_facecolor('#ffe4e9')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
    
    # SEGMENTATION ANALYSIS PAGE
    elif page == "Segmentation Analysis":
        st.header("🎯 Customer Segmentation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cluster Distribution")
            cluster_counts = df["Cluster"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#FFD700', '#FF69B4', '#87CEEB', '#90EE90', '#FFA500', '#DDA0DD']
            cluster_counts.plot(kind='bar', color=colors, ax=ax, alpha=0.8)
            ax.set_title("Customers per Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            ax.set_facecolor('#fff0f3')
            fig.patch.set_facecolor('#ffe4e9')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Elbow Method")
            wcss = []
            for i in range(2, 10):
                kmeans = KMeans(n_clusters=i, random_state=42)
                kmeans.fit(x_scaled)
                wcss.append(kmeans.inertia_)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(range(2, 10), wcss, marker="o", color='#ff69b4', linewidth=2, markersize=8)
            ax.set_title("Elbow Method for Optimal K")
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("WCSS")
            ax.set_facecolor('#fff0f3')
            fig.patch.set_facecolor('#ffe4e9')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.subheader("Cluster Characteristics")
        cluster_summary = df.groupby("Cluster")[features].mean()
        st.dataframe(cluster_summary.style.background_gradient(cmap='RdPu'), 
                     use_container_width=True)
        
        # Show cluster descriptions
        st.subheader("📋 Cluster Profiles")
        for i in range(6):
            cluster_info = get_cluster_description(i, None)
            with st.expander(f"{cluster_info['name']}"):
                for char in cluster_info['characteristics']:
                    st.write(f"• {char}")
        
        st.subheader("Average Spending by Education")
        group1 = df.groupby("Education")["Total_Spending"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        group1.plot(kind="bar", color="#ffb3c1", ax=ax)
        ax.set_title("Average Spending by Education")
        ax.set_ylabel("Average Total Spending")
        plt.xticks(rotation=45, ha='right')
        ax.set_facecolor('#fff0f3')
        fig.patch.set_facecolor('#ffe4e9')
        st.pyplot(fig)
        
        st.subheader("Average Income by Age Group")
        group3 = df.groupby("AgeGroup")["Income"].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        group3.plot(kind='barh', color='#ff69b4', ax=ax)
        ax.set_title("Average Income by Age Group")
        ax.set_xlabel("Average Income")
        ax.set_facecolor('#fff0f3')
        fig.patch.set_facecolor('#ffe4e9')
        st.pyplot(fig)
    
    # CAMPAIGN PERFORMANCE PAGE
    elif page == "Campaign Performance":
        st.header("📢 Campaign Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overall Acceptance Rate", 
                     f"{df['AcceptAny'].mean()*100:.1f}%")
        
        with col2:
            st.metric("Total Campaigns Accepted", 
                     f"{df['AcceptAny'].sum():,}")
        
        st.subheader("Campaign Acceptance by Marital Status")
        group2 = df.groupby("Marital_Status")["AcceptAny"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        group2.plot(kind='bar', color='#ffb3c1', ax=ax)
        ax.set_title("Campaign Acceptance Rate by Marital Status")
        ax.set_ylabel("Acceptance Rate")
        plt.xticks(rotation=45, ha='right')
        ax.set_facecolor('#fff0f3')
        fig.patch.set_facecolor('#ffe4e9')
        st.pyplot(fig)
        
        st.subheader("Individual Campaign Performance")
        campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", 
                        "AcceptedCmp4", "AcceptedCmp5", "Response"]
        campaign_acceptance = df[campaign_cols].sum()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        campaign_acceptance.plot(kind='bar', color='#ff69b4', ax=ax)
        ax.set_title("Acceptances by Campaign")
        ax.set_ylabel("Number of Acceptances")
        plt.xticks(rotation=45, ha='right')
        ax.set_facecolor('#fff0f3')
        fig.patch.set_facecolor('#ffe4e9')
        st.pyplot(fig)

# -------------------------------------------------
# ROUTING
# -------------------------------------------------
if st.session_state.logged_in:
    dashboard_page()
else:
    login_page()