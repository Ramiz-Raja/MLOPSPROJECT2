# frontend/streamlit_app.py
import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-card {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌸 Iris Species Classifier</h1>', unsafe_allow_html=True)
st.markdown("### A Machine Learning-powered tool to classify Iris flowers based on their measurements")

# Sidebar for model information
with st.sidebar:
    st.header("📊 Model Information")
    
    # Try to get model info from backend
    try:
        model_info_resp = requests.get(f"{BACKEND_URL}/model/info", timeout=5)
        if model_info_resp.status_code == 200:
            model_info = model_info_resp.json()
            st.success("✅ Model Connected")
            
            if model_info.get("metadata"):
                metadata = model_info["metadata"]
                st.metric("Accuracy", f"{metadata.get('test_accuracy', 0):.1%}")
                st.metric("F1 Score", f"{metadata.get('test_f1', 0):.3f}")
                st.metric("CV Mean", f"{metadata.get('cv_mean', 0):.3f}")
                
                if metadata.get("class_names"):
                    st.write("**Species Classes:**")
                    for i, class_name in enumerate(metadata["class_names"]):
                        st.write(f"• {i}: {class_name}")
        else:
            st.error("❌ Model Not Available")
    except:
        st.warning("⚠️ Backend Not Connected")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔬 Input Measurements")
    st.markdown("Enter the measurements of your Iris flower:")
    
    # Input form with better styling and validation
    with st.form("iris_prediction_form"):
        sepal_len = st.number_input(
            "Sepal Length (cm)", 
            min_value=4.0, 
            max_value=8.0, 
            value=5.1, 
            step=0.1,
            help="Length of the sepal in centimeters"
        )
        
        sepal_wid = st.number_input(
            "Sepal Width (cm)", 
            min_value=2.0, 
            max_value=4.5, 
            value=3.5, 
            step=0.1,
            help="Width of the sepal in centimeters"
        )
        
        petal_len = st.number_input(
            "Petal Length (cm)", 
            min_value=1.0, 
            max_value=7.0, 
            value=1.4, 
            step=0.1,
            help="Length of the petal in centimeters"
        )
        
        petal_wid = st.number_input(
            "Petal Width (cm)", 
            min_value=0.1, 
            max_value=2.5, 
            value=0.2, 
            step=0.1,
            help="Width of the petal in centimeters"
        )
        
        submitted = st.form_submit_button("🔍 Classify Iris Species", use_container_width=True)

with col2:
    st.header("📈 Data Visualization")
    
    # Create a sample visualization
    sample_data = {
        'Measurement': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
        'Value': [sepal_len, sepal_wid, petal_len, petal_wid],
        'Type': ['Sepal', 'Sepal', 'Petal', 'Petal']
    }
    
    df_viz = pd.DataFrame(sample_data)
    
    # Create a bar chart
    fig = px.bar(
        df_viz, 
        x='Measurement', 
        y='Value',
        color='Type',
        title="Current Input Measurements",
        color_discrete_map={'Sepal': '#1f77b4', 'Petal': '#ff7f0e'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Prediction section
if submitted:
    st.header("🎯 Prediction Results")
    
    payload = {"features": [sepal_len, sepal_wid, petal_len, petal_wid]}
    
    try:
        with st.spinner("🔍 Analyzing your Iris flower..."):
            resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        
        if "error" in data:
            st.markdown('<div class="error-card">', unsafe_allow_html=True)
            st.error(f"❌ Backend Error: {data['error']}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Display prediction results
            prediction = data.get('prediction', [0])[0]
            predicted_class = data.get('predicted_class', 'Unknown')
            probabilities = data.get('probability', [[]])[0]
            model_info = data.get('model_info', {})
            
            # Create prediction card
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            # Main prediction
            st.markdown(f"### 🌸 **Predicted Species: {predicted_class}**")
            
            # Confidence level
            if probabilities:
                max_prob = max(probabilities)
                confidence = max_prob * 100
                
                if confidence > 80:
                    st.success(f"🎯 **High Confidence: {confidence:.1f}%**")
                elif confidence > 60:
                    st.warning(f"⚠️ **Medium Confidence: {confidence:.1f}%**")
                else:
                    st.error(f"❓ **Low Confidence: {confidence:.1f}%**")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Probability breakdown
            if probabilities:
                st.subheader("📊 Probability Breakdown")
                
                # Get class names from model info or use defaults
                class_names = model_info.get('class_names', ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'])
                
                prob_data = {
                    'Species': class_names,
                    'Probability': [p * 100 for p in probabilities]
                }
                
                prob_df = pd.DataFrame(prob_data)
                
                # Create probability bar chart
                fig_prob = px.bar(
                    prob_df, 
                    x='Species', 
                    y='Probability',
                    title="Prediction Probabilities",
                    color='Probability',
                    color_continuous_scale='Blues'
                )
                fig_prob.update_layout(height=400)
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Show probability table
                st.subheader("📋 Detailed Probabilities")
                prob_df['Probability'] = prob_df['Probability'].round(2)
                st.dataframe(prob_df, use_container_width=True)
            
            # Model performance info
            if model_info:
                st.subheader("🤖 Model Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{model_info.get('accuracy', 'N/A')}")
                with col2:
                    st.metric("F1 Score", f"{model_info.get('f1_score', 'N/A')}")
                with col3:
                    st.metric("CV Mean", f"{model_info.get('cv_mean', 'N/A')}")
    
    except requests.exceptions.RequestException as e:
        st.markdown('<div class="error-card">', unsafe_allow_html=True)
        st.error(f"❌ Connection Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown('<div class="error-card">', unsafe_allow_html=True)
        st.error(f"❌ Unexpected Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer with information
st.markdown("---")
st.markdown("### ℹ️ About This Application")
st.markdown("""
This application uses a machine learning model trained on the famous Iris dataset to classify Iris flowers into three species:
- **Iris Setosa**: Typically has shorter petals and wider sepals
- **Iris Versicolor**: Medium-sized petals and sepals
- **Iris Virginica**: Usually has longer petals and narrower sepals

The model uses logistic regression with cross-validation to provide accurate predictions with confidence scores.
""")

# Health check
with st.expander("🔧 System Status"):
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            if health_data.get("status") == "ok":
                st.success("✅ Backend Service: Healthy")
                if health_data.get("model_validation"):
                    validation = health_data["model_validation"]
                    if validation.get("validation_passed"):
                        st.success("✅ Model Validation: Passed")
                    else:
                        st.warning("⚠️ Model Validation: Issues detected")
            else:
                st.error("❌ Backend Service: Degraded")
        else:
            st.error("❌ Backend Service: Unavailable")
    except:
        st.error("❌ Backend Service: Not reachable")
