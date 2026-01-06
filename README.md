# Hard Drive RUL Prediction using Streaming Machine Learning

## 🎯 Project Overview

This project implements **Streaming Machine Learning** models to predict the **Remaining Useful Life (RUL)** of hard drives based on SMART (Self-Monitoring, Analysis and Reporting Technology) indicators.

Using data from Backblaze's public dataset, we predict how many days remain until a drive fails, enabling proactive maintenance and data protection.

---

## 📁 Project Structure

```
.
├── config.py                      # Centralized configuration
├── preprocessing.py               # Data preprocessing utilities
├── models.py                      # Model definitions and training
├── RUL_Prediction_Tutorial.ipynb # 📓 Main educational notebook
│
├── requirements.txt              # Python dependencies
├── stream_data.csv               # Preprocessed data (generated)
│
├── data/
│   ├── data_Q4_2024/            # Raw data files (Oct-Dec 2024)
│   └── data_Q1_2025/            # Raw data files (Jan-Mar 2025)
│
└── results/
    ├── plots/                    # Generated visualizations
    └── models/                   # Saved model checkpoints
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd 2024-2025_hard-drive-remaining-useful-life-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Educational Notebook

```bash
# Start Jupyter Lab
jupyter lab

# Open: RUL_Prediction_Tutorial.ipynb
```

The notebook will guide you through:
- ✅ Data preprocessing
- ✅ Training 3 different streaming models
- ✅ Performance comparison
- ✅ Parameter tuning experiments
- ✅ Detailed explanations of streaming ML concepts

---

## 📊 Models Implemented

### 1. **Baseline: Linear Regression**
- Simple streaming linear model
- Fast and interpretable
- ~25 days MAE

### 2. **Hoeffding Tree Regressor**
- Incremental decision tree
- Learns from streaming data using Hoeffding bounds
- ~18 days MAE

### 3. **SRP (Streaming Random Patches) Ensemble**
- Ensemble of Hoeffding Trees
- Like "Random Forest for streams"
- **Best performance: ~15 days MAE**

---

## 💻 Usage Examples

### Preprocessing Data

```python
from preprocessing import preprocess_data
import config

# Preprocess Q1 2025 data
preprocess_data(
    data_folder=config.DATA_FOLDER_Q1_2025,
    output_file="stream_data.csv",
    verbose=True
)
```

### Training a Single Model

```python
from models import train_model

# Train Hoeffding Tree with log-transformed features
model, metric, errors, instances = train_model(
    model_name="hoeffding",
    feature_transform="log",
    verbose=True
)

print(f"Final MAE: {metric.get():.2f} days")
```

### Comparing Models

```python
from models import compare_models

# Compare all three models
results = compare_models(
    model_configs={
        "baseline": {},
        "hoeffding": {},
        "srp": {}
    },
    feature_transform="log",
    save_plot=True
)
```

### Custom Parameter Tuning

```python
from models import train_model

# Custom SRP ensemble with 15 trees
model, metric, _, _ = train_model(
    model_name="srp",
    custom_params={
        "n_models": 15,
        "grace_period": 75
    },
    feature_transform="log"
)
```

---

## 🎓 Educational Content

The Jupyter notebook (`RUL_Prediction_Tutorial.ipynb`) provides:

1. **Theoretical Background**
   - What is Remaining Useful Life?
   - Why Streaming ML for this problem?
   - Prequential evaluation explained

2. **Hands-on Implementation**
   - Step-by-step data preprocessing
   - Training each model with detailed explanations
   - Live visualizations of learning curves

3. **Experiments**
   - Impact of grace period on Hoeffding Trees
   - Ensemble size vs. performance trade-off
   - Feature transformation comparison (raw vs log vs normalized)

4. **Best Practices**
   - Memory-efficient streaming data processing
   - Handling massive SMART feature values
   - Model selection guidelines

---

## ⚙️ Configuration

All parameters are centralized in `config.py`:

```python
# Data paths
DEFAULT_DATA_FOLDER = "data/data_Q1_2025"
PREPROCESSED_FILE = "stream_data.csv"

# Model parameters
HOEFFDING_TREE_CONFIG = {
    "grace_period": 50,
    "leaf_prediction": "mean",
    "model_selector_decay": 0.9
}

SRP_CONFIG = {
    "n_models": 10,
    "grace_period": 50,
    "seed": 42
}
```

Modify these values to customize your experiments!

---

## 🛠️ Advanced Usage

### Command-Line Preprocessing

```bash
# Preprocess specific data folder
python preprocessing.py data/data_Q4_2024 output.csv
```

### Command-Line Model Training

```bash
# Train specific model
python models.py hoeffding
python models.py srp
```

### Modular Code Structure

```python
# Import specific components
from config import HOEFFDING_TREE_CONFIG
from preprocessing import find_failed_serials, extract_history
from models import create_hoeffding_tree, plot_single_model_performance

# Build custom pipeline
model = create_hoeffding_tree(grace_period=100)
# ... your code ...
```

---

## 📊 Data Format

### Input (Raw Daily CSV Files)
```
date,serial_number,model,capacity_bytes,failure,smart_1_raw,...
2025-01-01,ABC123,ST4000DM000,4000787030016,0,0,...
```

### Output (Preprocessed)
```
date,serial_number,model,capacity_bytes,failure,smart_1_raw,...,RUL
2025-01-01,ABC123,ST4000DM000,4000787030016,0,0,...,45
```

Where `RUL` = Days until this drive fails

---

## 🔬 Research & References

- **River Library**: https://riverml.xyz/
- **Backblaze Data**: https://www.backblaze.com/b2/hard-drive-test-data.html

---

## 📝 Notes

## 🤝 Contributing

Feel free to:
- Add new streaming models
- Implement additional features
- Improve visualizations
- Extend the tutorial

---

## 📄 License

This project is for educational purposes as part of the Streaming Data Analytics course at PoliMi.

---

## 👨‍💻 Author

**Roberto Benatuil**  
Politecnico di Milano  
Streaming Data Analytics Course - 2024/2025

---

## 🎉 Getting Help

1. Start with the **Jupyter notebook** - it's designed to be self-explanatory
2. Check the **docstrings** in each module (config.py, preprocessing.py, models.py)
3. Experiment with the **parameter tuning cells** in the notebook

**Happy Streaming!** 🚀
