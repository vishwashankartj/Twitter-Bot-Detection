# Twitter Bot Detection

A machine learning system to detect Twitter bots using network analysis with **NetworkX** and **Graph2Vec** libraries. Features both high-precision (100% precision, 22% recall) and high-recall (77% recall, 24% precision) models for different use cases.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🎯 Overview

This project implements a sophisticated bot detection system that analyzes Twitter network graphs to identify automated accounts. By leveraging custom Graph2Vec embeddings (using gensim + Weisfeiler-Lehman kernels) and ensemble machine learning models, the system provides flexible bot detection for different use cases.

### Key Features

- 🔍 **Network Analysis**: Constructs and analyzes follower/following networks using NetworkX
- 🧠 **Custom Graph2Vec**: Weisfeiler-Lehman graph kernels + Word2Vec embeddings (Python 3.13 compatible)
- 🤖 **Ensemble Learning**: Combines Random Forest, SVM, and Neural Network classifiers
- 🎯 **Dual Models**: High-precision model (100% precision) OR high-recall model (77% recall)
- 📊 **Comprehensive Evaluation**: Detailed metrics, visualizations, and performance reports
- 📦 **Automated Pipeline**: End-to-end training pipeline from data generation to predictions
- 🔮 **Prediction Demo**: Test bot detection on new accounts with confidence scores

## 🏗️ Architecture

```
Data Collection → Graph Construction → Graph2Vec Embeddings → Feature Extraction → Classification → Evaluation
```

### Components

1. **Data Processing**: Loads and preprocesses Twitter user data
2. **Graph Builder**: Constructs network graphs from user interactions
3. **Graph2Vec**: Generates graph-level embeddings using Weisfeiler-Lehman kernel
4. **Feature Extractor**: Combines embeddings with network metrics and profile features
5. **Classifiers**: Ensemble of RF, SVM, and Neural Network models
6. **Evaluator**: Comprehensive performance metrics and visualizations

## 📋 Requirements

- Python 3.9+
- NetworkX
- gensim (for Graph2Vec implementation)
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Kaggle API (for dataset download)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Twitter-Bot-Detection.git
cd Twitter-Bot-Detection
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Sample Data (Quick Start)

For immediate testing without Kaggle setup:

```bash
python generate_sample_data.py
```

### 5. Setup Kaggle API (Optional - For Real Data)

For automatic dataset download:

```bash
# Download kaggle.json from https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
python download_data.py
```

## 📊 Dataset

The project uses the **Twitter Bot Detection Dataset** from Kaggle, which includes:

- User profiles (followers, following, tweets count)
- Tweet content and metadata
- Binary labels (bot/human)
- Network interaction data

### Download Dataset

```bash
python download_data.py
```

Or manually download from [Kaggle](https://www.kaggle.com/datasets/search?q=twitter+bot+detection) and place in `data/raw/`.

## 🎓 Usage

### Quick Start (With Sample Data)

```bash
# 1. Generate sample dataset (5,000 accounts)
python generate_sample_data.py

# 2. Train initial model (High Precision)
python train.py

# 3. Train improved model (High Recall)
python train.py --mode recall

# 4. Test predictions on demo accounts
python predict.py --demo
```

### With Real Kaggle Data

```bash
# 1. Download real dataset
python download_data.py

# 2. Train model (runs complete pipeline)
python train.py

# 3. Make predictions on new data
python predict.py --input data/new_accounts.csv --output results/predictions.csv
```

### Step-by-Step Pipeline

```bash
# Load and preprocess data
python src/data/data_loader.py

# Build network graphs
python src/data/graph_builder.py

# Generate Graph2Vec embeddings
python src/models/graph2vec_embeddings.py

# Extract features
python src/data/feature_extractor.py

# Train classifiers
python src/models/classifiers.py
```

### Using the Jupyter Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

## 📈 Performance

### Model Comparison

#### High-Precision Model (Default)
*Best for: Avoiding false positives, user-facing applications*

| Metric | Score |
|--------|-------|
| **Accuracy** | **90.79%** |
| **Precision** | **100%** |
| Recall | 22.22% |
| F1-Score | 36.36% |
| ROC-AUC | 0.72 |

**Use Case**: When banning real users is very costly (e.g., account suspension)

#### High-Recall Model (Improved)
*Best for: Catching more bots, spam prevention*

| Metric | Score |
|--------|-------|
| Accuracy | 68.42% |
| Precision | 24.14% |
| **Recall** | **77.78%** |
| F1-Score | 36.84% |
| ROC-AUC | 0.72 |
| **Optimal Threshold** | **0.090** |

**Use Case**: When catching bots is priority (e.g., spam filtering, content moderation)

### Classifier Components

| Model | Weight | Role |
|-------|--------|------|
| Random Forest | 3 | Primary classifier, feature importance |
| SVM | 2 | Decision boundary optimization |
| Neural Network | 1 | Complex pattern recognition |

## 🗂️ Project Structure

```
Twitter-Bot-Detection/
├── config/
│   └── config.yaml                  # Configuration parameters
├── data/
│   ├── raw/                         # Raw datasets
│   ├── processed/                   # Processed data and features
│   └── external/                    # External data sources
├── src/
│   ├── data/
│   │   ├── data_loader.py           # Data loading and preprocessing
│   │   ├── graph_builder.py         # Network graph construction
│   │   └── feature_extractor.py     # Feature extraction
│   ├── models/
│   │   ├── graph2vec_embeddings.py  # Custom Graph2Vec (gensim + WL)
│   │   └── classifiers.py           # ML classifiers
│   └── evaluation/
│       └── metrics.py               # Evaluation metrics
├── models/                          # Trained models
├── results/                         # Results and visualizations
├── results_improved/                # Improved model results
├── notebooks/
│   └── demo.md                      # Interactive demo guide
├── generate_sample_data.py          # Sample dataset generator
├── download_data.py                 # Kaggle dataset downloader
├── train.py                         # Main training script (Precision/Recall modes)
├── predict.py                       # Prediction script (File/Demo modes)
├── verify_setup.py                  # Setup verification
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🔬 Methodology

### 1. Graph Construction

- Build follower/following networks
- Create interaction graphs (retweets, mentions)
- Generate ego graphs for each user

### 2. Graph2Vec Embeddings

- Apply Weisfeiler-Lehman graph kernel
- Generate fixed-size graph embeddings (128 dimensions)
- Capture structural patterns in networks

### 3. Feature Engineering

- **Graph2Vec embeddings**: 128 dimensions
- **Network metrics**: degree centrality, clustering coefficient, PageRank
- **Profile features**: followers/following ratio, tweet frequency

### 4. Classification

- Ensemble of Random Forest, SVM, and Neural Network
- Soft voting for final predictions
- Hyperparameter tuning via cross-validation

## 📊 Visualizations

The system generates:

- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Network graph visualizations

All visualizations are saved to `results/plots/`.

## 🛠️ Configuration

Edit `config/config.yaml` to customize:

- Graph2Vec parameters (dimensions, iterations)
- Classifier hyperparameters
- Data paths
- Training settings

## 🌐 REST API Deployment

### Start the FastAPI Server

```bash
# Option 1: Using the startup script
./start_api.sh

# Option 2: Direct uvicorn command
source venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Access Swagger Documentation

Once the server is running, open your browser to:
- **Swagger UI**: http://localhost:8000/ (redirects to /docs)
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict?model_type=recall" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_123",
    "username": "example_user",
    "followers_count": 500,
    "friends_count": 400,
    "statuses_count": 2000,
    "verified": 0,
    "default_profile": 0,
    "default_profile_image": 0,
    "favourites_count": 1500,
    "listed_count": 10,
    "account_age_days": 1825
  }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "precision",
    "accounts": [
      {
        "user_id": "user1",
        "username": "bot_account",
        "followers_count": 50,
        "friends_count": 5000,
        "statuses_count": 10000,
        "verified": 0,
        "default_profile": 1,
        "default_profile_image": 1,
        "favourites_count": 100,
        "listed_count": 2,
        "account_age_days": 30
      }
    ]
  }'
```

#### List Available Models
```bash
curl http://localhost:8000/models
```

### Response Format

```json
{
  "user_id": "test_123",
  "username": "example_user",
  "prediction": "Human",
  "bot_probability": 0.067,
  "confidence": 0.933,
  "model_used": "recall"
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Graph2Vec: Learning Distributed Representations of Graphs](https://arxiv.org/abs/1707.05005)
- [NetworkX Documentation](https://networkx.org/)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/)
- [Weisfeiler-Lehman Graph Kernels](https://www.jmlr.org/papers/v12/shervashidze11a.html)

## 👤 Author

**Vishwashankar Janakiraman**

## 🙏 Acknowledgments

- Twitter Bot Detection Dataset from Kaggle
- NetworkX and Gensim libraries
- scikit-learn community

---

**Note**: This project is for educational and research purposes. Always respect Twitter's Terms of Service and user privacy when collecting and analyzing social media data.