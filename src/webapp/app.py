import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from src.hybrid.hybrid_model import HybridModel

# Constants
DATA_DIR = "data"
MODELS_DIR = "models"

def load_models():
    """Load all trained models"""
    models = {
        "Logistic Regression": pickle.load(open(f"{MODELS_DIR}/ml/logistic_regression.pkl", "rb")),
        "Random Forest": pickle.load(open(f"{MODELS_DIR}/ml/random_forest.pkl", "rb")),
        "XGBoost": pickle.load(open(f"{MODELS_DIR}/ml/xgboost.pkl", "rb")),
        "Deep Learning": tf.keras.models.load_model(f"{MODELS_DIR}/dl/best_model.h5"),
        "Hybrid": HybridModel(
            pickle.load(open(f"{MODELS_DIR}/ml/logistic_regression.pkl", "rb")),
            tf.keras.models.load_model(f"{MODELS_DIR}/dl/best_model.h5")
        )
    }
    return models

def safe_feature(func, *args, default=0.0, **kwargs):
    """Safely calculate features with error handling"""
    try:
        return func(*args, **kwargs)
    except:
        return default

def get_ml_features(image):
    """Extract comprehensive features from image for ML models"""
    img_array = np.array(image.convert('L'))  # Convert to grayscale
    
    # Calculate exactly 30 features to match model expectations
    features = {
        # Basic statistics (5)
        'mean_intensity': safe_feature(np.mean, img_array),
        'std_intensity': safe_feature(np.std, img_array),
        'min_intensity': safe_feature(np.min, img_array),
        'max_intensity': safe_feature(np.max, img_array),
        'median_intensity': safe_feature(np.median, img_array),
        
        # Texture features (10)
        'energy': safe_feature(lambda x: np.mean(x**2), img_array),
        'entropy': safe_feature(lambda x: -np.sum(x * np.log(x + 1e-10)), img_array),
        'contrast': safe_feature(np.std, img_array),
        'homogeneity': safe_feature(lambda x: np.mean(1/(1+x**2)), img_array),
        'dissimilarity': safe_feature(lambda x: np.mean(np.abs(x - np.mean(x))), img_array),
        'correlation': safe_feature(lambda x: np.corrcoef(x.ravel(), x.T.ravel())[0,1], img_array),
        'asm': safe_feature(lambda x: np.sum((x/np.sum(x))**2), img_array),
        'idm': safe_feature(lambda x: np.sum(1/(1+x**2)), img_array),
        'cluster_shade': safe_feature(lambda x: np.sum((x - np.mean(x))**3), img_array),
        'cluster_prominence': safe_feature(lambda x: np.sum((x - np.mean(x))**4), img_array),
        
        # Shape features (5)
        'aspect_ratio': img_array.shape[1] / img_array.shape[0],
        'area': img_array.size,
        'perimeter': safe_feature(lambda x: 2*(x.shape[0]+x.shape[1]), img_array),
        'circularity': safe_feature(lambda x: (4*np.pi*x.size)/(2*(x.shape[0]+x.shape[1])**2), img_array),
        'solidity': safe_feature(lambda x: x.size/(x.shape[0]*x.shape[1]), img_array),
        
        # Statistical features (10)
        'iqr': safe_feature(lambda x: np.percentile(x, 75) - np.percentile(x, 25), img_array),
        'skewness': safe_feature(lambda x: np.mean((x - np.mean(x))**3) / (np.std(x)**3), img_array),
        'kurtosis': safe_feature(lambda x: np.mean((x - np.mean(x))**4) / (np.std(x)**4) - 3, img_array),
        'uniformity': safe_feature(lambda x: np.sum((np.histogram(x, bins=10)[0] / x.size)**2), img_array),
        'smoothness': safe_feature(lambda x: 1 - (1 / (1 + np.var(x))), img_array),
        'mad': safe_feature(lambda x: np.mean(np.abs(x - np.mean(x))), img_array),
        'rms': safe_feature(lambda x: np.sqrt(np.mean(x**2)), img_array),
        'var': safe_feature(np.var, img_array),
        'range': safe_feature(lambda x: np.max(x) - np.min(x), img_array),
        'coverage': safe_feature(lambda x: np.mean(x > np.mean(x)), img_array)
    }
    
    # Convert to DataFrame with single row
    df = pd.DataFrame([features])
    return df

# Main app
def main():
    st.set_page_config(layout="wide")
    
    # Sidebar controls
    st.sidebar.title("Controls")
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest", "XGBoost", "Deep Learning", "Hybrid"]
    )
    
    uploaded_file = st.sidebar.file_uploader("Upload Mammogram", type=["png", "jpg"])

    # Main content
    st.title("Breast Cancer Prediction")
    st.markdown("---")

    # Load models
    models = load_models()

    # Get image with proper handling
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')  # Force grayscale
        st.session_state.image_path = uploaded_file.name
    else:
        st.warning("Please upload a mammogram image to analyze")
        st.stop()

    # Display image and predictions
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Mammogram")
        plt.figure(figsize=(8, 6))
        img_array = np.array(image)
        # Normalize and display with proper contrast
        vmin = np.percentile(img_array, 1)
        vmax = np.percentile(img_array, 99)
        plt.imshow(img_array, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title('Mammogram')
        plt.axis('off')
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("Prediction")
        
        # Get prediction
        if selected_model in ["Deep Learning", "Hybrid"]:
            img_array = np.array(image.convert('RGB').resize((224, 224)))
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            if selected_model == "Deep Learning":
                proba = models[selected_model].predict(img_array)[0][0]
            else:
                ml_features = get_ml_features(image)
                proba = models[selected_model].predict_proba(ml_features, img_array)
        else:
            ml_features = get_ml_features(image)
            proba = models[selected_model].predict_proba(ml_features)[0, 1]
        
        # Display results
        st.metric("Prediction", "Malignant" if proba >= 0.5 else "Benign")
        st.metric("Confidence", f"{proba:.1%}")
        st.progress(float(proba if proba >= 0.5 else 1-proba))
        
        if proba >= 0.5:
            st.warning("This mammogram shows characteristics associated with malignancy.")
        else:
            st.success("This mammogram appears benign.")

    # SHAP Explanation Section
    st.subheader("Model Explanation")
    try:
        import shap
        from src.xai.explainer import get_shap_explanation
        
        if selected_model in ["Deep Learning", "Hybrid"]:
            img_array = np.array(image.convert('RGB').resize((224, 224)))
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            try:
                explanation = get_shap_explanation(
                    models[selected_model],
                    img_array
                )
                if explanation is not None:
                    st.pyplot(explanation)
                else:
                    st.warning("Explanation not available for this model")
            except Exception as e:
                st.error(f"Could not generate explanation: {str(e)}")
            
        elif selected_model in ["Random Forest", "XGBoost"]:
            ml_features = get_ml_features(image)
            
            try:
                explainer = shap.TreeExplainer(models[selected_model])
                shap_values = explainer.shap_values(ml_features)
                
                st.subheader("Feature Importance")
                fig, ax = plt.subplots()
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                shap.summary_plot(shap_values, ml_features, plot_type="bar", show=False)
                st.pyplot(fig)
                
                st.subheader("Prediction Explanation")
                fig, ax = plt.subplots()
                if isinstance(explainer.expected_value, list):
                    expected_value = explainer.expected_value[1]
                else:
                    expected_value = explainer.expected_value
                shap.force_plot(
                    expected_value, 
                    shap_values, 
                    ml_features.iloc[0],
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate explanation: {str(e)}")
            
        elif selected_model == "Logistic Regression":
            ml_features = get_ml_features(image)
            
            # Use KernelExplainer for Logistic Regression
            def predict_proba_wrapper(x):
                return models[selected_model].predict_proba(x)
            
            explainer = shap.KernelExplainer(
                predict_proba_wrapper,
                shap.sample(ml_features, 10)
            )
            shap_values = explainer.shap_values(ml_features)
            
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Handle binary classification case properly
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get values for positive class
            shap.summary_plot(shap_values, ml_features, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Could not generate explanation: {str(e)}")

if __name__ == '__main__':
    main()
