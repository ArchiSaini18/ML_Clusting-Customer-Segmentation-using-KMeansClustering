🛍️ **Customer Segmentation using K-Means Clustering**

This project analyzes customer behavior and groups customers into meaningful segments using the K-Means clustering algorithm.
The goal is to help businesses understand customer purchasing patterns, identify high-value and at-risk customers, and design data-driven marketing strategies through an interactive Streamlit dashboard.

📌 **Project Overview**

In this project, we:

-Built a secure Streamlit application with user authentication (login & signup).

- Loaded and preprocessed customer data by:

   - Handling missing values

   - Engineering new features such as Age, Total Spending, Customer Tenure, and Campaign Acceptance

- Performed exploratory data analysis (EDA) to study distributions, correlations, and customer behavior.

- Scaled numerical features using StandardScaler.

- Applied K-Means clustering to segment customers into 6 distinct groups.

- Selected the number of clusters using the Elbow Method (WCSS).

- Created interpretable cluster profiles (e.g., premium customers, budget shoppers, dormant users).

- Developed a real-time customer cluster prediction tool to assign new customers to the right segment.

- Visualized results using interactive charts, heatmaps, and comparison plots.

- Analyzed marketing campaign performance across customer segments.

📂 **Dataset**

Source: Customer marketing and transaction dataset (customer_segmentation.csv)

**Typical Features**:

- Demographics: Age, Income, Education, Marital Status

- Behavioral: Web Purchases, Store Purchases, Web Visits, Recency

- Spending: Wines, Fruits, Meat, Fish, Sweets, Gold Products

- Campaign Responses

- Target: Unsupervised learning (no predefined labels)

🛠️ **Technologies Used**

- Python 3.x

- Streamlit – interactive dashboard & UI

- Pandas, NumPy – data manipulation & feature engineering

- Scikit-learn – scaling, K-Means clustering

- Matplotlib & Seaborn – visual analytics

- CSV-based user authentication for login/signup

📊 **Model Selection & Evaluation**

- Elbow Method (WCSS) used to identify the optimal number of clusters.

**Chosen number of clusters balances:**

  - Statistical compactness

  - Clear behavioral separation

  - Business interpretability

  **Clusters validated through:**

- Feature distributions

- Spending and engagement patterns

- Campaign response behavior

📈 **Visualizations**

- Customer age, income, and spending distributions

- Correlation heatmaps between income, spending, and purchase behavior

- Elbow curve for K selection

- Cluster distribution bar charts

- Cluster comparison charts (user input vs. cluster average)

- Campaign performance dashboards

- Cluster profile summaries with expandable descriptions

🧭 **Workflow**

Customer Data
→ Data Cleaning & Feature Engineering
→ Exploratory Data Analysis (EDA)
→ Feature Scaling
→ Elbow Method (WCSS)
→ K-Means Clustering
→ Cluster Profiling
→ Streamlit Dashboard & Prediction System

💼 **Deliverables**

- Cleaned and feature-engineered dataset

- Trained K-Means clustering model

- Saved StandardScaler for prediction

- Interactive Streamlit dashboard

- Customer cluster prediction tool

- Cluster profile insights for marketing teams

🔮 **Future Improvements**

- Evaluate Silhouette, Calinski-Harabasz, and Davies-Bouldin scores

- Experiment with GMM or DBSCAN for non-spherical clusters

- Add online behavior and seasonal features

- Integrate a database-backed authentication system

- Deploy dashboard to cloud (Streamlit Cloud / AWS)

