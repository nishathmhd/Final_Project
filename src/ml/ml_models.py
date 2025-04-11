"""
Machine Learning models for breast cancer diagnosis.
"""

import os
import sys
import numpy as np # type: ignore
import pandas as pd # type: ignore
import pickle
import time
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
import xgboost as xgb # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Add project root to path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import ML_MODELS, ML_MODELS_DIR, RESULTS_DIR, RANDOM_STATE
from src.preprocessing.data_processor import load_processed_data, get_feature_names

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train and evaluate a Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        tuple: (best_model, best_params, validation_accuracy)
    """
    print("Training Logistic Regression...")
    grid_search = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        ML_MODELS["logistic_regression"],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Logistic Regression best parameters: {best_params}")
    print(f"Logistic Regression validation accuracy: {val_accuracy:.4f}")
    
    return best_model, best_params, val_accuracy

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train and evaluate a Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        tuple: (best_model, best_params, validation_accuracy)
    """
    print("Training Random Forest...")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        ML_MODELS["random_forest"],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Random Forest best parameters: {best_params}")
    print(f"Random Forest validation accuracy: {val_accuracy:.4f}")
    
    return best_model, best_params, val_accuracy

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train and evaluate an XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        tuple: (best_model, best_params, validation_accuracy)
    """
    print("Training XGBoost...")
    
    # Fix for newer XGBoost versions
    params = ML_MODELS["xgboost"].copy()
    
    grid_search = GridSearchCV(
        xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
        params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"XGBoost best parameters: {best_params}")
    print(f"XGBoost validation accuracy: {val_accuracy:.4f}")
    
    return best_model, best_params, val_accuracy

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model on the test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for reporting
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{model_name} Evaluation on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign (0)', 'Malignant (1)'],
                yticklabels=['Benign (0)', 'Malignant (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    
    # Save metrics to file
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return metrics

def train_and_evaluate_all_models():
    """
    Train and evaluate all ML models.
    
    Returns:
        dict: Dictionary containing the best model for each algorithm
    """
    # Load processed data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Dictionary to store results
    models = {}
    all_metrics = []
    
    # Train and evaluate Logistic Regression
    start_time = time.time()
    lr_model, lr_params, lr_val_acc = train_logistic_regression(X_train, y_train, X_val, y_val)
    lr_training_time = time.time() - start_time
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    lr_metrics['training_time'] = lr_training_time
    lr_metrics['best_params'] = lr_params
    all_metrics.append(lr_metrics)
    models['logistic_regression'] = lr_model
    
    # Train and evaluate Random Forest
    start_time = time.time()
    rf_model, rf_params, rf_val_acc = train_random_forest(X_train, y_train, X_val, y_val)
    rf_training_time = time.time() - start_time
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_metrics['training_time'] = rf_training_time
    rf_metrics['best_params'] = rf_params
    all_metrics.append(rf_metrics)
    models['random_forest'] = rf_model
    
    # Train and evaluate XGBoost
    start_time = time.time()
    xgb_model, xgb_params, xgb_val_acc = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_training_time = time.time() - start_time
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    xgb_metrics['training_time'] = xgb_training_time
    xgb_metrics['best_params'] = xgb_params
    all_metrics.append(xgb_metrics)
    models['xgboost'] = xgb_model
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'ml_models_metrics.csv'), index=False)
    
    # Save models
    for model_name, model in models.items():
        model_path = os.path.join(ML_MODELS_DIR, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {model_name} model to {model_path}")
    
    # Create model comparison plot
    plot_model_comparison(all_metrics)
    
    return models

def plot_model_comparison(metrics_list):
    """
    Create a comparison plot for all trained models.
    
    Args:
        metrics_list: List of dictionaries containing model metrics
    """
    model_names = [metrics['model_name'] for metrics in metrics_list]
    accuracy = [metrics['accuracy'] for metrics in metrics_list]
    precision = [metrics['precision'] for metrics in metrics_list]
    recall = [metrics['recall'] for metrics in metrics_list]
    f1 = [metrics['f1_score'] for metrics in metrics_list]
    roc_auc = [metrics['roc_auc'] for metrics in metrics_list]
    
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })
    
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Value')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='Value', hue='Metric', data=melted_df)
    plt.title('ML Model Performance Comparison')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ml_models_comparison.png'))
    
def load_best_model():
    """
    Load the best ML model based on metrics.
    
    Returns:
        tuple: (best_model, model_name)
    """
    # Load metrics
    metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, 'ml_models_metrics.csv'))
    
    # Find the best model (highest F1 score)
    best_model_row = metrics_df.loc[metrics_df['f1_score'].idxmax()]
    best_model_name = best_model_row['model_name'].lower().replace(" ", "_")
    
    # Load the model
    model_path = os.path.join(ML_MODELS_DIR, f"{best_model_name}.pkl")
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
    
    return best_model, best_model_name

if __name__ == "__main__":
    # Train and evaluate all models
    models = train_and_evaluate_all_models()
    
    # Get the best model
    best_model, best_model_name = load_best_model()
    print(f"\nBest ML model: {best_model_name}")