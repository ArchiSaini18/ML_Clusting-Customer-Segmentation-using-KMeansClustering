🛍️ **Customer Segmentation using K-Means Clustering**

This project analyzes customer behavior and groups customers into meaningful segments using the K-Means clustering algorithm.
The goal is to help businesses understand customer purchasing patterns, identify high-value and low-value customers, and support data-driven marketing decisions through an interactive Streamlit dashboard.

📌 **Project Overview**

In this project, we:

- Built a secure Streamlit application with login and signup authentication using a CSV-based user database.

- Loaded and preprocessed customer data by:

- Handling missing values

- Engineering new features such as Age and Total Spending

- Applied feature scaling using StandardScaler to normalize customer attributes.

- Implemented K-Means clustering to segment customers into 6 distinct clusters based on purchasing behavior.

- Used the Elbow Method (WCSS) to justify the optimal number of clusters.

- Created cluster summaries to understand average customer behavior per segment.

- Developed a real-time cluster prediction system that assigns new customers to the appropriate segment.

- Visualized customer distributions and comparisons using interactive histograms and charts.

- Enabled profile comparison between user input and the overall dataset.

📂 **Dataset**

- Source: Customer marketing and transaction dataset (customer_segmentation.csv)

**Typical Features**

- Demographics:

- Age

- Income

**Behavioral**:

- Number of Web Purchases

- Number of Store Purchases

- Web Visits per Month

- Recency (days since last purchase)

**Spending**:

- Wines, Fruits, Meat, Fish, Sweets, Gold Products

- Total Spending (engineered feature)

**Target**:

- Unsupervised learning (no predefined labels)

🛠️ **Technologies Used**

- Python 3.x

- Streamlit – interactive dashboard & UI

- Pandas, NumPy – data handling & feature engineering

- Scikit-learn – feature scaling & K-Means clustering

- Matplotlib & Seaborn – data visualization

- CSV-based authentication system – login & signup

📊 **Model Selection & Evaluation**

- Elbow Method (WCSS) used to determine the optimal number of clusters.

- Final number of clusters selected based on:

- Compactness of clusters

- Clear separation of customer behaviors

- Business interpretability of segments

**Clusters validated through**:

- Average spending and income levels

- Purchase frequency patterns

- Recency and engagement behavior

📈 **Visualizations**

- Customer age, income, and spending distributions

- User input vs dataset comparison (highlighted with reference lines)

- Cluster summary tables with gradient styling

- Distribution histograms for behavioral analysis

- Cluster-level statistics for business insight

🧭 **Workflow**

Customer Data
→ Data Cleaning & Feature Engineering
→ Feature Scaling
→ Elbow Method (WCSS)
→ K-Means Clustering
→ Cluster Summary & Profiling
→ Streamlit Dashboard
→ Real-Time Cluster Prediction

💼 **Deliverables**

- Cleaned and feature-engineered dataset

- Trained K-Means clustering model

- Saved StandardScaler for prediction

- Interactive Streamlit dashboard

- Real-time customer cluster prediction system

- Cluster summary insights for marketing strategy

🔮 **Future Improvements**

- Add campaign response and online behavior features

- Replace CSV authentication with a database-backed system

- Deploy dashboard to Streamlit Cloud or AWS

- Implement cluster-based marketing recommendations

**Screenshot**
1.link: https://github.com/ArchiSaini18/ML_Clusting-Customer-Segmentation-using-KMeansClustering/blob/main/Screenshot%202026-01-14%20082835.png
2.link : https://github.com/ArchiSaini18/ML_Clusting-Customer-Segmentation-using-KMeansClustering/blob/main/Screenshot%202026-01-14%20082752.png
3. link: https://github.com/ArchiSaini18/ML_Clusting-Customer-Segmentation-using-KMeansClustering/blob/main/Screenshot%202026-01-14%20082810.png
4. link: https://github.com/ArchiSaini18/ML_Clusting-Customer-Segmentation-using-KMeansClustering/blob/main/Screenshot%202026-01-14%20082822.png
