🛍️ Customer Segmentation using K-Means Clustering
This project groups mall customers into meaningful segments based on their spending patterns and purchase behavior using the K-Means clustering algorithm.
 The goal is to help the mall understand who buys more vs. less, tailor marketing offers, and improve customer experience.

📌 Project Overview
In this project, we:

- Load and preprocess customer data (handle missing values, encode categorical features like Gender, scale numerics like Income/Spending).

- Explore the data to understand distributions and relationships.

- Select the number of clusters using WCSS (Elbow Method) and validate with Silhouette/Calinski-Harabasz/Davies-Bouldin scores.

- Train K-Means to form segments (e.g., high spenders, budget shoppers, occasional visitors).

- Profile clusters to derive actionable business insights (e.g., offers, loyalty campaigns).

- Build an assignment pipeline to place new customers into the right segment.

📂 Dataset

- Source: Mall CRM/transactions (or the classic “Mall Customers” dataset).

- Typical Features: Age, Gender, Annual Income, Spending Score.

- Target: Unsupervised (no labels).

🛠️ Technologies Used

- Python 3.x

- Pandas, NumPy – data handling

- Scikit-learn – preprocessing, K-Means, metrics

- Matplotlib / Seaborn (or Plotly) – visualization

📊 Model Selection & Evaluation

Elbow (WCSS) to narrow down candidate k.

Silhouette, Calinski-Harabasz, and Davies-Bouldin to compare cluster quality.

Final k chosen by both metrics and business interpretability (segments must make sense to marketing/ops).

📈 Visualizations

- Elbow curve (WCSS vs. k) to show the “knee.”

- Silhouette plot for the chosen k.

- 2D/3D scatter (PCA/t-SNE) colored by cluster.

- Cluster profiles: heatmap/boxplots of feature means per cluster.

- RFM/behavior charts to highlight differences between segments.

🧭 Workflow
Customer Data → Data Preprocessing → Data Analysis (EDA) → WCSS (Elbow) → K-Means Clustering → Visualization & Cluster Profiling

💼 Deliverables

- Cleaned dataset and preprocessing pipeline

- Trained K-Means model and saved scaler

- Cluster profile report (who buys more/less, key traits)

- Script/notebook to assign new customers to segments

🔮 Future Improvements
- Test GMM or DBSCAN for non-spherical clusters

- Add features (online behavior, seasonality, coupons)

- Deploy a Streamlit/Gradio dashboard for interactive segmentation

- A/B test targeted campaigns per segment and track uplift
