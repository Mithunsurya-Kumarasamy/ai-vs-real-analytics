# AI vs Real Images — Predictive Analytics Project

A full-stack ML system that classifies AI-generated vs real images using
**Logistic Regression**, **Decision Trees**, and **Random Forest**, with a
React analytics dashboard.

---

## Project Structure

```
ai_vs_real/
├── feature_extraction.py   # Image feature engineering (191 features)
├── ml_pipeline.py          # Full ML pipeline with training + evaluation
├── app.py                  # Flask REST API
├── Dashboard.jsx           # React analytics dashboard
├── requirements.txt
└── dataset/                # ← place Kaggle data here
    ├── train/
    │   ├── AI/
    │   └── Real/
    └── test/
        ├── AI/
        └── Real/
```

---

## Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
From Kaggle: https://www.kaggle.com/datasets/rhythmghai/ai-vs-real-images-dataset  
Place the folders as shown above under `dataset/`.

### 3. Run the ML pipeline
```bash
python ml_pipeline.py
```
This will:
- Extract 191 features per image (cached to `output/features.npz`)
- Run EDA
- Train + tune Logistic Regression, Decision Tree, Random Forest
- Cross-validate with StratifiedKFold (5 folds)
- Compute confusion matrices, ROC curves, learning curves, calibration
- Save all results to `output/results.json`
- Save models to `models/*.pkl`

### 4. Start the Flask API
```bash
python app.py
```
API runs on `http://localhost:5000`

Endpoints:
- `GET  /api/health`            — check status
- `GET  /api/results`           — full pipeline results JSON
- `GET  /api/eda`               — EDA statistics
- `GET  /api/models/comparison` — model comparison table
- `GET  /api/models/<name>`     — per-model results
- `GET  /api/calibration`       — calibration curves data
- `POST /api/predict`           — predict a new image (base64 JSON)

### 5. Run the React dashboard
```bash
# Create a React app
npx create-react-app frontend
cd frontend
npm install recharts
# Copy Dashboard.jsx → src/App.jsx
npm start
```

---

## Predictive Analytics Concepts Covered

| Concept | Implementation |
|---|---|
| Feature Engineering | Color histograms, GLCM texture, LBP, FFT frequency, noise analysis |
| Train/Test Split | 80/20 stratified split |
| Cross-Validation | StratifiedKFold (5 folds) |
| Hyperparameter Tuning | GridSearchCV (LR: C, penalty; DT: max_depth, criterion) |
| Regularization | L1/L2 on Logistic Regression (regularization path) |
| Dimensionality Reduction | PCA (95% variance retention) |
| Logistic Regression | Binary classification with probability outputs |
| Decision Tree | Gini/entropy, depth tuning, pruning analysis |
| Random Forest | Ensemble of 200 trees |
| Feature Importance | Gini importance + permutation importance |
| Confusion Matrix | TP/TN/FP/FN per model |
| ROC / AUC | Multi-model ROC curves |
| Precision / Recall / F1 | Full classification report |
| Learning Curves | Bias-variance tradeoff visualization |
| Calibration Curves | Reliability diagrams + Brier scores |
| Ensemble Prediction | Majority vote across 3 models |

---

## Feature Groups (191 total)

| Group | Count | Description |
|---|---|---|
| Color histogram | 96 | 32-bin histogram per R/G/B channel |
| HSV statistics | 18 | Mean, std, skew, kurtosis, Q25/Q75 per H/S/V |
| GLCM texture | 30 | Contrast, dissimilarity, homogeneity, energy, correlation |
| LBP histogram | 26 | Local binary pattern (radius=3, 24 points) |
| Edge features | 6 | Sobel magnitude, Canny edge density |
| FFT frequency | 5 | Radial frequency band energies + HF ratio |
| Noise analysis | 5 | Residual after Gaussian blur — kurtosis, std |
| Saturation uniformity | 5 | Local variance of saturation — AI images are smoother |
