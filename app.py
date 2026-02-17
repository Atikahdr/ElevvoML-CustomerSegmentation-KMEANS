import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load Model & Scaler
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# Session State for Navigation
if "page" not in st.session_state:
    st.session_state.page = "home"
if "input_data" not in st.session_state:
    st.session_state.input_data = None
if "cluster" not in st.session_state:
    st.session_state.cluster = None
if "history" not in st.session_state:
    st.session_state.history = []

# 🔹 Configuration Page
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
)
# Page Navigation
st.sidebar.title("Customer Segmentation")

page_map = {
    "🏠 Home": "home",
    "📝 Input Data": "input", 
    "📋 Table Data": "table",
    "📊 Scatter Plot": "scatter",
    "🧾 History": "history"
}

current_page = st.session_state.page
if current_page not in page_map.values():
    current_page = "home"
    st.session_state.page = "home"

menu = st.sidebar.radio(
    "Select Page",
    list(page_map.keys()),
    index=list(page_map.values()).index(st.session_state.page)
)

st.session_state.page = page_map[menu]

st.sidebar.markdown("---")
st.sidebar.caption("Created by AtikahDR")

# Prediction Function
def predict_segmnet(model, scaler, income, spending_score):
    try:
        input_data = np.array([[income, spending_score]])
        scaled_data = scaler.transform(input_data)
        cluster = model.predict(scaled_data)[0]
        distances = model.transform(scaled_data)
        min_distance = np.min(distances)

        return cluster, min_distance
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Home Page
if st.session_state.page == "home":
    st.title("🛍️ Customer Segmentation App")
    header_img = "cs.png"
    st.image(header_img, use_container_width=True)

    st.markdown("""
    ## 📊 Welcome to the Customer Segmentation App
                
    This application uses **Machine Learning (K-Means Clustering)** 
    to segment customers based on purchasing behavior.
                
    ### 🎯 Project Objective
    he goal of this system is to help businesses:
    - Optimize resource allocation
    -  Identify high-value customer segments
    - Detect potential growth opportunities
    - Improve marketing strategy personalization

    ## 🧠 Machine Learning Model
    This segmentation model is built using:
    - **StandardScaler** → to normalize income and spending score
    - **K-Means Clustering (k=5)** → to group customers into distinct segments

    The model clusters customers based on:
    - 💰 Annual Income  
    - 🛒 Spending Score  

    Using the optimal number of clusters determined through:
    - Elbow Method
    - Silhouette Score (0.555)

    ---
    ## 📌 Identified Customer Segments

    The model successfully identifies 5 distinct customer groups:

    1. **Premium Customers** (High Income – High Spending)
    2. **Growth Opportunity Segment** (High Income – Low Spending)
    3. **Young Big Spenders** (Low Income – High Spending)
    4. **Mass Market Customers** (Mid Income – Mid Spending)
    5. **Low Value Segment** (Low Income – Low Spending)

    Each segment provides actionable business insights for targeted marketing strategies.

    ## 🚀 How to Use This Dashboard

    Use the navigation menu on the left to:

    - 📝 Predict a new customer's segment  
    - 📊 Explore clustering visualization  

    This tool demonstrates how data-driven segmentation can enhance strategic decision-making in modern businesses.
    """)

    st.info("Use the sidebar navigation to explore the dashboard.")
        
    if st.button("🚀 Start Customer Segmentation"):
        st.session_state.page = "input"
        st.rerun()

# Input Page
elif st.session_state.page == "input":

    st.title("🎯 Customer Segment Prediction")
    st.markdown("Enter Customer Information")

    col1, col2 = st.columns(2)

    with col1:  
        income = st.number_input("Annual Income (k$)", min_value=15, step=1)
        spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, step=1)
    
    with col2:
        age = st.number_input("Age", min_value=18, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Spacer columns untuk center button
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        predict_clicked = st.button("🚀 Predict Segment", use_container_width=True)

    if predict_clicked:

        # Predict the segmment
        cluster, distance = predict_segmnet(model, scaler, income, spending_score)

        if cluster is not None:

            segment_descriptions = {
                0: "Mass Market",
                1: "Premium Customers",
                2: "Young Big Spenders",
                3: "Growth Opportunity",
                4: "Low Value Segment"
            }

            segment_name = segment_descriptions.get(cluster)

            if cluster == 0:
                st.info(f"Predicted Customer Segment: {segment_name} (Mid Income - Mid Spending)")
            elif cluster == 1:
                st.success(f"Predicted Customer Segment: {segment_name} (High Income - High Spending)")
            elif cluster == 2:
                st.warning(f"Predicted Customer Segment: {segment_name} (Low Income - High Spending)")
            elif cluster == 3:
                st.info(f"Predicted Customer Segment: {segment_name} (High Income - Low Spending)")
            else:
                st.error(f"Predicted Customer Segment: {segment_name} (Low Income - Low Spending)")

            # Save data to history
            data_cus = {
                "Income": income,
                "Spending Score": spending_score,
                "Age": age,
                "Gender": gender,
                "Cluster": cluster,
                "Segment": segment_name
            }
            st.session_state.history.append(data_cus)
           
# Table Page
elif st.session_state.page == "table":

    st.title("📋 Customer Data Table")

    if len(st.session_state.history) == 0:
        st.info("No prediction history available. Please predict a customer segment first.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        st.caption("This table shows the history of customer segment predictions made by the user.")

# Scatter Plot Page
elif st.session_state.page == "scatter":

    st.title("📊 Scatter Plot - User Prediction Data")

    if len(st.session_state.history) == 0:
        st.info("No prediction history available. Please predict a customer segment first.")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        fig, ax = plt.subplots()

        scatter = ax.scatter(history_df["Income"], history_df["Spending Score"], c=history_df["Cluster"], cmap='viridis', s=100)
        ax.set_xlabel('Annual Income (k$)')
        ax.set_ylabel('Spending Score (1-100)')
        ax.set_title('Customer Segmentation Scatter Plot')

        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)
        st.caption("Scatter Plot generated from user prediction history.")

# History Page
elif st.session_state.page == "history":
    st.title("📋 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No prediction history available. Please predict a customer segment first.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        st.caption("This table shows the history of customer segment predictions made by the user.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑 Clear History"):
                st.session_state.history = []
                st.rerun()

        with col2:
            st.download_button(
                label="⬇ Download CSV",
                data=history_df.to_csv(index=False),
                file_name="customer_history.csv",
                mime="text/csv"
            )

    #  Footer
st.markdown("---")

st.caption("💡 K-Means Clustering - Customer Segmentation | Machine Learning Prediction Project")

