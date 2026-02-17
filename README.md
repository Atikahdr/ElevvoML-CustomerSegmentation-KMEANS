🛍️ Customer Segmentation Using Machine Learning
---
🚀 Explore the live demo:
https://elevvoml-customersegmentation-kmeans.streamlit.app/

🚀 Machine Learning Project | KMeans Cluster

🌟 Level-1 → Task 2 + Bonus Completed ✅

---
📌 Project Overview
---

This project focuses on customer segmentation using unsupervised machine learning techniques. The goal is to group customers based on purchasing behavior and income patterns to generate actionable business insights.

By leveraging clustering algorithms, this project helps businesses better understand their customer base and implement more targeted marketing strategies.

🎯 Task Description
---
The objective of this project is to:
- Segment customers based on Annual Income and Spending Score
- Identify high-value and low-value customer groups
- Compare clustering algorithms (K-Means vs DBSCAN)
- Evaluate which algorithm produces more meaningful business segmentation
- Build an interactive Streamlit application for real-time prediction
 ---
 
📊 Dataset
---
Dataset Used: Mall Customers Dataset

Features:
- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

Key Variables for Clustering:

- Annual Income (k$)
- Spending Score (1–100)

The dataset contains customers with diverse income levels and spending behaviors, making it suitable for behavioral segmentation.

 ---
 
🛠️ Tools & Libraries
---

Programming Language
- Python

Libraries
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Machine Learning Algorithms
- K-Means Clustering
- DBSCAN
 ---
 
🔄 Project Workflow
---
1️⃣ Data Exploration
- Analyzed distribution of Age, Income, and Spending Score
- Identified patterns and potential clustering structure
- Checked data distribution and variance

2️⃣ Data Preprocessing
- Feature selection (Income & Spending Score)
- Feature scaling using StandardScaler

3️⃣ Model Development

🔹 K-Means Clustering
- Determined optimal K using:
  - Elbow Method
  - Silhouette Score
- Selected K = 5
- Generated clearly separated clusters

🔹 DBSCAN
- Applied density-based clustering
- Tuned eps and min_samples
- Compared performance with K-Means

4️⃣ Model Evaluation
- Compared clustering structure visually
- Used Silhouette Score for evaluation
- Analyzed cluster interpretability for business context

5️⃣ Deployment
- Built interactive Streamlit application
- Real-time customer segment prediction
- Scatter plot visualization
- Prediction history tracking
---

📈 Business Insights
---

🏆 1. Premium Customers (High Income – High Spending)

- Most valuable segment
- Strong purchasing power
- Ideal for loyalty programs & premium campaigns

📊 2. Growth Opportunity Segment (High Income – Low Spending)

- High earning but low engagement
- Potential for upselling and targeted marketing
- Strategic segment for revenue growth

🛍️ 3. Young Big Spenders (Low Income – High Spending)

- Behavior-driven consumers
- Highly responsive to trends & promotions

👥 4. Mass Market (Mid Income – Mid Spending)

- Stable customer base
- Suitable for general marketing campaigns

📉 5. Low Value Segment (Low Income – Low Spending)

- Low contribution to revenue
- Lower marketing priority
 ---

🔍 Algorithm Comparison
---
| Aspect	| K-Means	| DBSCAN |
|---------|---------|--------|
| Cluster Separation	| Clear & well-defined |	Mostly single cluster |
| Business Interpretability	| High	| Low |
| Suitable for Dataset	| Yes	| Less suitable |
| Type	| Centroid-based	| Density-based | 

Conclusion:

K-Means produced more meaningful and actionable customer segmentation compared to DBSCAN for this dataset.

🧠 Concepts Covered
---
 - Data Visualization
 - Unsupervised Learning
 - Clustering Algorithms
 - K-Means Clustering
 - DBSCAN
 - Elbow Method
 - Silhouette Score
 - Feature Scaling
 - Model Comparison
 - Business Interpretation of ML Results
 - Model Deployment using Streamlit
 ---
 
🚀 Streamlit Application Features
---

- Customer segment prediction
- Interactive scatter visualization
- Cluster-based colored output
- Prediction history tracking
 ---
 
👩‍💻 Author
---
Atikah DR
Machine Learning Enthusiast | Data Science Learner | Elevvo ML Internship Project

---
