# Breast Cancer Prediction System

A comprehensive machine learning system for predicting breast cancer from mammogram images using multiple modeling approaches.

## Features

- **Multiple Model Types**:
  - Traditional ML (Logistic Regression, Random Forest, XGBoost)
  - Deep Learning (CNN-based, EfficientNetB3)
  - Hybrid (Combination of ML and DL)

- **Explainable AI**:
  - SHAP explanations for all model types
  - Feature importance visualizations
  - Prediction explanations
  - Grad-CAM visualizations for deep learning models

- **Web Interface**:
  - Streamlit-based web app
  - Interactive model selection
  - Image upload and visualization
  - Real-time predictions

## Dataset

The system uses the [CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) dataset containing:
- 10,239 mammogram images
- Balanced classes (benign/malignant)
- CC and MLO views
- ROI annotations

Preprocessed versions are available in `data/processed/` as numpy arrays.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nishathmhd/Can-You-Find-The-Tumor.git
cd breast_cancer_project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application
```bash
streamlit run src/webapp/app.py
```
Access at `http://localhost:8501`

### Training Models
```bash
# Train ML models
python src/ml/ml_models.py

# Train DL models 
python src/dl/dl_models.py

# Train hybrid model
python src/hybrid/hybrid_model.py
```

### Running Predictions
```bash
# Batch prediction on test set
python src/predict.py --model_type [ml/dl/hybrid] --model_path models/[type]/model.pkl
```

## Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score | AUC   |
|---------------------|----------|-----------|--------|----------|-------|
| Logistic Regression | 0.87     | 0.86      | 0.88   | 0.87     | 0.93  |
| Random Forest       | 0.89     | 0.88      | 0.90   | 0.89     | 0.95  |
| XGBoost             | 0.90     | 0.89      | 0.91   | 0.90     | 0.96  |
| CNN                 | 0.92     | 0.91      | 0.93   | 0.92     | 0.97  |
| EfficientNetB3      | 0.93     | 0.92      | 0.94   | 0.93     | 0.98  |
| Hybrid              | 0.94     | 0.93      | 0.95   | 0.94     | 0.98  |

![Model Comparison](results/ml_models_comparison.png)
![Learning Curves](results/learning_curve_EfficientNetB3.png)

## Project Structure

```
breast_cancer_project/
├── data/                   # Dataset files
│   ├── images/             # Mammogram images (test/train/val splits)
│   ├── processed/          # Processed data files (numpy arrays)
│   └── raw/                # Raw data files
├── models/                 # Trained models
│   ├── dl/                 # Deep learning models (.h5)
│   ├── ml/                 # Machine learning models (.pkl)
│   └── hybrid/             # Hybrid models
├── notebooks/              # Jupyter notebooks for EDA
├── results/                # Analysis results
│   ├── eda/                # Exploratory analysis
│   ├── explanations/       # Model explanations
│   └── gradcam/            # Grad-CAM visualizations
├── src/                    # Source code
│   ├── config.py           # Configuration
│   ├── dl/                 # Deep learning code
│   ├── hybrid/             # Hybrid model code
│   ├── ml/                 # Machine learning code
│   ├── preprocessing/      # Data processing
│   ├── visualization/      # Visualization code
│   ├── webapp/             # Web application
│   └── xai/                # Explainable AI code
├── requirements.txt        # Python dependencies
└── setup.py                # Project setup
```

## Troubleshooting

If you encounter version mismatch warnings:
1. Re-train the models with current package versions
2. Or install specific versions:
```bash
pip install scikit-learn==1.3.0 tensorflow==2.11.0
```

For M1/M2 Mac performance issues:
```bash
pip install tensorflow-macos
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Work

- [ ] Add DICOM support
- [ ] Implement ensemble methods
- [ ] Develop mobile application
- [ ] Add multi-modal inputs (clinical data + images)

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{breast_cancer_prediction_2025,
  author = {Nishath, MHD},
  title = {Breast Cancer Prediction System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nishathmhd/Can-You-Find-The-Tumor}}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
