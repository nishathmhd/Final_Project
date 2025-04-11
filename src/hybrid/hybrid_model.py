import os
import sys
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import ML_MODELS_DIR, DL_MODELS_DIR, RESULTS_DIR, HYBRID_MODELS_DIR

class HybridModel:
    """
    Hybrid model that combines predictions from ML and DL models.
    """
    
    def __init__(self, ml_model, dl_model, ml_weight=0.5):
        """
        Initialize the hybrid model.
        
        Args:
            ml_model: Trained ML model
            dl_model: Trained DL model
            ml_weight: Weight for ML model predictions (0-1)
        """
        self.ml_model = ml_model
        self.dl_model = dl_model
        self.ml_weight = ml_weight
    
    def predict_proba(self, ml_features, dl_image):
        """
        Make probability predictions.
        
        Args:
            ml_features: Features for ML model
            dl_image: Image for DL model
            
        Returns:
            float: Probability of malignancy
        """
        # Get ML prediction
        ml_prob = self.ml_model.predict_proba(ml_features)[0, 1]
        
        # Get DL prediction
        dl_prob = self.dl_model.predict(dl_image)[0, 0]
        
        # Combine predictions
        hybrid_prob = (self.ml_weight * ml_prob) + ((1 - self.ml_weight) * dl_prob)
        
        return hybrid_prob
    
    def predict(self, ml_features, dl_image, threshold=0.5):
        """
        Make class predictions.
        
        Args:
            ml_features: Features for ML model
            dl_image: Image for DL model
            threshold: Classification threshold
            
        Returns:
            int: Predicted class (0=benign, 1=malignant)
        """
        prob = self.predict_proba(ml_features, dl_image)
        return 1 if prob >= threshold else 0
    
    def tune_weight(self, ml_features_val, dl_images_val, y_val):
        """
        Tune the weight parameter to optimize performance.
        
        Args:
            ml_features_val: Validation features for ML model
            dl_images_val: Validation images for DL model
            y_val: True labels
            
        Returns:
            float: Optimal weight
        """
        best_weight = 0.5
        best_accuracy = 0
        
        # Try different weights
        for weight in np.arange(0, 1.1, 0.1):
            self.ml_weight = weight
            
            # Make predictions
            preds = []
            for i in range(len(y_val)):
                ml_feat = ml_features_val[i:i+1]
                dl_img = dl_images_val[i:i+1]
                pred = self.predict(ml_feat, dl_img)
                preds.append(pred)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_val, preds)
            
            print(f"Weight {weight:.1f}: Accuracy {accuracy:.4f}")
            
            # Update best weight
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = weight
        
        self.ml_weight = best_weight
        print(f"Best weight: {best_weight:.1f} with accuracy {best_accuracy:.4f}")
        
        return best_weight
    
    def save(self, path):
        """
        Save the hybrid model.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        params = {
            'ml_weight': self.ml_weight
        }
        
        with open(path, 'wb') as f:
            pickle.dump(params, f)
    
    @classmethod
    def load(cls, path, ml_model, dl_model):
        """
        Load a hybrid model.
        
        Args:
            path: Path to the saved model
            ml_model: ML model
            dl_model: DL model
            
        Returns:
            HybridModel: Loaded model
        """
        with open(path, 'rb') as f:
            params = pickle.load(f)
        
        return cls(ml_model, dl_model, ml_weight=params['ml_weight'])

def create_hybrid_model():
    """
    Create a hybrid model using the best ML and DL models.
    
    Returns:
        HybridModel: Hybrid model
    """
    # Load ML model
    ml_metrics = pd.read_csv(os.path.join(RESULTS_DIR, 'ml_models_metrics.csv'))
    best_ml_model_name = ml_metrics.loc[ml_metrics['f1_score'].idxmax(), 'model_name'].lower().replace(" ", "_")
    
    with open(os.path.join(ML_MODELS_DIR, f"{best_ml_model_name}.pkl"), 'rb') as f:
        ml_model = pickle.load(f)
    
    # Load DL model
    dl_model = tf.keras.models.load_model(os.path.join(DL_MODELS_DIR, 'best_model.h5'))
    
    # Create hybrid model
    hybrid_model = HybridModel(ml_model, dl_model)
    
    return hybrid_model

if __name__ == "__main__":
    # Create hybrid model
    hybrid_model = create_hybrid_model()
    
    # Save hybrid model
    hybrid_model.save(os.path.join(HYBRID_MODELS_DIR, 'hybrid_model.pkl'))
    
    print("Hybrid model created and saved!")
