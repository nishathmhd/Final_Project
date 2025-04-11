import numpy as np
import matplotlib.pyplot as plt
import shap
import torch
from typing import Union, Tuple
from src.hybrid.hybrid_model import HybridModel

def get_shap_explanation(
    model: Union[torch.nn.Module, HybridModel], 
    input_data: np.ndarray,
    background_samples: int = 20
) -> plt.Figure:
    """Generate comprehensive SHAP explanation for DL or Hybrid models"""
    try:
        # Convert input to tensor
        input_tensor = torch.from_numpy(input_data).float()
        
        # Create background data
        background = input_tensor[np.random.choice(
            input_tensor.shape[0], 
            min(background_samples, input_tensor.shape[0]), 
            replace=False
        )]
        
        # For hybrid models, use the DL component
        if isinstance(model, HybridModel):
            model = model.dl_model
            
        # Create explainer
        explainer = shap.GradientExplainer(
            model,
            background
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_tensor)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.image_plot(
            shap_values,
            -input_data,  # Negate for medical imaging
            show=False
        )
        
        return fig
        
    except Exception as e:
        raise RuntimeError(f"SHAP explanation failed: {str(e)}")

def explain_ml_model(
    model,
    features: np.ndarray,
    feature_names: list = None
) -> Tuple[plt.Figure, plt.Figure]:
    """Generate comprehensive SHAP explanation for ML models"""
    try:
        # Create explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(features)
        
        # Create summary plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            features,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        
        # Create force plot
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_values.values[0],
            features.iloc[0],
            matplotlib=True,
            show=False
        )
        
        return fig1, fig2
        
    except Exception as e:
        raise RuntimeError(f"ML explanation failed: {str(e)}")
