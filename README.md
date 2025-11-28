# Disease Classification AI

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-black)

Advanced ML-powered disease classification system using Random Forest and XGBoost models. Deployed as a serverless application on Vercel with a modern glassmorphism UI.

## 🚀 Live Demo

Deploy your own instance:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/disease-classification-ai)

## ✨ Features

- **Dual ML Models**: XGBoost (99.96% accuracy) and Random Forest (97% accuracy)
- **46 Input Features**: Age, gender, vitals, and 40 symptom severity ratings (0-3 scale)
- **8 Disease Classes**: Allergy, Asthma, Bronchitis, Common Cold, COVID-19, Influenza, Pneumonia, Tuberculosis
- **Modern Glassmorphism UI**: Responsive design with translucent cards and smooth animations
- **Serverless Architecture**: Deployed on Vercel with Python serverless functions
- **Real-time Predictions**: Instant disease classification with confidence scores

## 📋 Prerequisites

Before deploying, ensure you have:

- Python 3.9+ installed
- Trained ML models (`.joblib` files)
- Git installed
- Vercel account (free tier works)

## 🔧 Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/disease-classification-ai.git
cd disease-classification-ai
```

### 2. Train Models (If Not Done)

```bash
# Install Jupyter and training dependencies
pip install jupyter pandas numpy scikit-learn xgboost matplotlib seaborn

# Run the training notebook
jupyter notebook notebooks/ml_rf_xgb_optuna.ipynb
# Execute all cells to generate models in models/ directory
```

**Required model files in `models/` directory:**

- `xgb_model.joblib` (XGBoost classifier pipeline)
- `rf_model.joblib` (Random Forest classifier pipeline)
- `label_encoder.joblib` (Disease label encoder)

### 3. Test Locally (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Test API locally with Flask
python app.py
```

Visit `http://localhost:5000` to test the interface.

## 🌐 Deploy to Vercel

### Quick Deploy (Recommended)

1. **Install Vercel CLI:**

```bash
npm install -g vercel
```

2. **Login to Vercel:**

```bash
vercel login
```

3. **Deploy:**

```bash
vercel
```

Follow the prompts:

- Set up and deploy? **Y**
- Which scope? Select your account
- Link to existing project? **N**
- Project name? `disease-classification-ai` (or your choice)
- Directory? `./` (root)
- Override settings? **N**

4. **Deploy to Production:**

```bash
vercel --prod
```

### Manual Deploy via GitHub

1. **Push to GitHub:**

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/disease-classification-ai.git
git push -u origin main
```

2. **Import to Vercel:**

- Go to [Vercel Dashboard](https://vercel.com/dashboard)
- Click "Add New..." → "Project"
- Import your GitHub repository
- Vercel will auto-detect the configuration from `vercel.json`
- Click "Deploy"

### Environment Variables (Optional)

If you need custom configuration:

```bash
# Set via CLI
vercel env add PYTHON_VERSION production

# Or in Vercel Dashboard:
# Settings → Environment Variables
```

## 📁 Project Structure

```
disease-classification-ai/
├── api/
│   └── index.py              # Vercel serverless function
├── models/                    # ML models (not in git)
│   ├── xgb_model.joblib
│   ├── rf_model.joblib
│   └── label_encoder.joblib
├── public/
│   └── index.html            # Frontend UI
├── notebooks/                 # Training notebooks (not deployed)
│   └── ml_rf_xgb_optuna.ipynb
├── data/                      # Training data (not deployed)
│   └── synthetic_dataset_25k_40symptoms.csv
├── .gitignore                # Git exclusions
├── .vercelignore             # Vercel exclusions
├── vercel.json               # Vercel configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔑 API Endpoints

### POST `/api/predict`

Predict disease from patient data.

**Request:**

```json
{
  "model": "xgboost",
  "age": 35,
  "gender": "female",
  "smoker": 0,
  "heart_rate": 125,
  "blood_pressure": 117,
  "cholesterol_level": 132,
  "fever": 2,
  "cough": 3,
  ... (40 symptoms total, each 0-3)
}
```

**Response:**

```json
{
  "success": true,
  "prediction": "influenza",
  "confidence": 87.45,
  "model_used": "xgboost",
  "all_probabilities": {
    "influenza": 87.45,
    "common_cold": 8.23,
    "covid19": 2.11,
    ...
  }
}
```

### GET `/api/health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "models_available": true
}
```

### GET `/api/features`

Get required features list.

**Response:**

```json
{
  "features": ["age", "gender", "smoker", ...],
  "total_features": 46
}
```

## 🧪 Testing

```bash
# Test health endpoint
curl https://your-app.vercel.app/api/health

# Test prediction
curl -X POST https://your-app.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model":"xgboost","age":35,"gender":"male",...}'
```

## 📊 Model Performance

| Model         | ROC AUC | Accuracy |
| ------------- | ------- | -------- |
| XGBoost       | 99.96%  | 99.8%    |
| Random Forest | 97.00%  | 96.5%    |

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
- **Backend**: Python 3.9, Flask 3.0
- **ML Libraries**: scikit-learn 1.3, XGBoost 2.0
- **Deployment**: Vercel Serverless Functions
- **Models**: Joblib serialization

## 🔒 Important Notes

### Model Files

⚠️ **Model files are NOT included in git** due to size (10-50MB each). You must:

1. Train models locally using the notebook
2. Commit them to your repository, OR
3. Upload them directly to Vercel via CLI:

```bash
# Include models in deployment
vercel --prod --force
```

### Cold Starts

Vercel serverless functions may experience ~1-2s cold start on first request. Subsequent requests are instant due to model caching.

### File Size Limits

Vercel free tier:

- **Deployment size**: Max 100MB (uncompressed)
- **Function size**: Max 50MB per function
- **Request timeout**: 10 seconds

If models exceed limits, consider:

- Using model compression
- Hosting models on external storage (AWS S3, Google Cloud Storage)
- Upgrading to Vercel Pro

## 🚧 Troubleshooting

### Models not loading

```bash
# Check models exist
ls -lh models/

# Verify they're not in .gitignore
cat .gitignore

# Force include in deployment
git add -f models/*.joblib
git commit -m "Add model files"
git push
```

### Build errors on Vercel

Check Vercel build logs:

- Ensure `requirements.txt` has correct versions
- Verify `api/index.py` imports work
- Check Python version matches (3.9)

### Prediction errors

- Ensure all 46 features are provided in request
- Check feature names match exactly
- Verify data types (age, smoker as integers)

## 📝 License

MIT License - feel free to use for educational or commercial projects.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## ⚠️ Medical Disclaimer

This is an AI prediction tool for **educational purposes only**. Always consult qualified healthcare professionals for medical diagnosis and treatment. Do not use for actual medical decisions.

## 📧 Contact

Questions or issues? Open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).

---

**Built with ❤️ using Python, XGBoost, and Vercel**
