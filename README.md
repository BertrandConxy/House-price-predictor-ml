# House Price Predictor ML

A machine learning project that predicts house prices using a PyTorch neural network with embedding layers for categorical features and continuous features processing.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## 🏠 Overview

This project implements a deep learning model to predict house prices based on various features including both categorical and continuous variables. The model uses PyTorch's embedding layers to handle categorical features effectively and combines them with continuous features through a feed-forward neural network.

## ✨ Features

- **Hybrid Neural Network**: Combines embedding layers for categorical features with batch normalization for continuous features
- **Data Preprocessing**: Automated label encoding for categorical variables and feature engineering
- **Model Persistence**: Save and load trained models for future predictions
- **Performance Visualization**: Training loss visualization and model evaluation metrics
- **Robust Architecture**: Dropout layers and batch normalization for better generalization

## 📊 Dataset

The model uses a house prices dataset (`dataset/houseprice.csv`) with the following features:

### Target Variable
- **SalePrice**: The target variable representing the house sale price

### Categorical Features
- **MSSubClass**: The building class
- **MSZoning**: General zoning classification
- **Street**: Type of road access
- **LotShape**: General shape of property

### Continuous Features
- **LotFrontage**: Linear feet of street connected to property
- **LotArea**: Lot size in square feet
- **1stFlrSF**: First floor square feet
- **2ndFlrSF**: Second floor square feet
- **TotalYears**: Calculated feature (Current Year - Year Built)

## 🏗️ Model Architecture

The model implements a custom `FeedForwardNN` class with the following architecture:

```
Input Layer:
├── Categorical Features → Embedding Layers → Dropout
└── Continuous Features → Batch Normalization

Hidden Layers:
├── Linear(n_features, 100) → ReLU → BatchNorm → Dropout
└── Linear(100, 50) → ReLU → BatchNorm → Dropout

Output Layer:
└── Linear(50, 1) → House Price Prediction
```

### Key Components:
- **Embedding Layers**: Handle categorical features with learned representations
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout**: Prevents overfitting (p=0.4)
- **ReLU Activation**: Non-linear activation function
- **MSE Loss**: Mean Squared Error with RMSE monitoring

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd house-price-predictor-ml
```

2. **Install required dependencies**:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

3. **Ensure dataset is in place**:
```
dataset/
└── houseprice.csv
```

## 💻 Usage

### Training the Model

1. **Open the Jupyter notebook**:
```bash
jupyter notebook notebook.ipynb
```

2. **Run all cells** to:
   - Load and preprocess the data
   - Train the neural network
   - Evaluate model performance
   - Save the trained model

### Using Pre-trained Model

```python
import torch
from model import FeedForwardNN

# Load the saved model
embs_size = [(15, 8), (5, 3), (2, 1), (4, 2)]
model = FeedForwardNN(embs_size, 5, 1, [100, 50], p=0.4)
model.load_state_dict(torch.load('HouseWeights.pt'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(categorical_features, continuous_features)
```

## 📁 Project Structure

```
house-price-predictor-ml/
├── dataset/
│   └── houseprice.csv          # House prices dataset
├── notebook.ipynb             # Main implementation notebook
├── HousePrice.pt              # Complete saved model
├── HouseWeights.pt            # Model state dictionary
└── README.md                  # Project documentation
```

## 📈 Model Performance

### Training Configuration
- **Epochs**: 5,000
- **Batch Size**: 1,200 samples
- **Train/Test Split**: 85%/15%
- **Learning Rate**: 0.01
- **Optimizer**: Adam
- **Loss Function**: RMSE (Root Mean Square Error)

### Embedding Dimensions
- **MSSubClass**: 15 → 8
- **MSZoning**: 5 → 3  
- **Street**: 2 → 1
- **LotShape**: 4 → 2

### Model Evaluation
The model performance is evaluated using:
- **RMSE Loss**: Root Mean Square Error for regression
- **Prediction vs Actual**: Direct comparison of predicted and actual house prices
- **Training Loss Visualization**: Loss curve over training epochs

## 🔧 Technical Details

### Data Preprocessing
1. **Missing Value Handling**: Rows with missing values are dropped
2. **Feature Engineering**: TotalYears calculated from YearBuilt
3. **Label Encoding**: Categorical features encoded using sklearn's LabelEncoder
4. **Tensor Conversion**: All features converted to PyTorch tensors

### Model Features
- **Embedding Strategy**: Categorical features use learned embeddings
- **Regularization**: Dropout (0.4) and Batch Normalization
- **Architecture**: Two hidden layers (100, 50 neurons)
- **Activation**: ReLU activation functions

### Key Hyperparameters
```python
embedding_dim = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
hidden_layers = [100, 50]
dropout_rate = 0.4
learning_rate = 0.01
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with grid search
- [ ] Cross-validation for better model evaluation
- [ ] Feature importance analysis
- [ ] Model ensemble techniques
- [ ] Web interface for predictions
- [ ] Additional feature engineering
- [ ] Model interpretability tools

---

**Note**: This project is designed for educational purposes and demonstrates the implementation of neural networks for regression tasks using PyTorch.
