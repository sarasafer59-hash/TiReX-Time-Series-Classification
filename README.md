# 🦖 TiReX Time Series Classification

A powerful and easy-to-use implementation of **TiReX (Time Series Representation Extraction)** for time series classification tasks. This repository demonstrates how to leverage state-of-the-art embeddings combined with traditional ML classifiers to achieve excellent results on time series data.

## ✨ Features

- **Pre-trained Embeddings**: Utilize TiReX's powerful time series embeddings
- **Multiple Classifiers**: Choose from Random Forest, Linear, or Gradient Boosting classifiers
- **Simple API**: sklearn-style interface for easy integration
- **High Performance**: Achieve competitive results with minimal code
- **Flexible**: Works with univariate and multivariate time series data

## 🚀 Quick Start

### Installation
```bash
pip install 'tirex-ts[notebooks,gluonts,hfdataset,classification]'
pip install aeon
```

For the latest development version:
```bash
pip install 'git+https://github.com/NX-AI/tirex.git#egg=tirex-ts[classification]'
```

### Basic Usage
```python
from tirex.models.classification import TirexRFClassifier
from aeon.datasets import load_italy_power_demand

# Load data
train_X, train_y = load_italy_power_demand(split="train")
test_X, test_y = load_italy_power_demand(split="test")

# Initialize and train classifier
classifier = TirexRFClassifier(n_estimators=50, max_depth=10)
classifier.fit((train_X, train_y))

# Make predictions
predictions = classifier.predict(test_X)
```

## 📊 Available Classifiers

### 1. Random Forest Classifier

Best for robust performance and handling noisy data.
```python
from tirex.models.classification import TirexRFClassifier

classifier = TirexRFClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    batch_size=256
)
```

### 2. Linear Classifier

Fast and efficient for linearly separable data.
```python
from tirex.models.classification import TirexLinearClassifier

classifier = TirexLinearClassifier(
    data_augmentation=False,
    compile=False
)
```

### 3. Gradient Boosting Classifier

High accuracy through ensemble learning.
```python
from tirex.models.classification import TirexGBMClassifier

classifier = TirexGBMClassifier(
    data_augmentation=False,
    compile=False
)
```

## 📓 Example Notebook

Check out the [quick_start_tirex_classification.ipynb](quick_start_tirex_classification.ipynb) notebook for a complete walkthrough that includes:

- Data loading and preprocessing
- Label encoding
- Training multiple classifiers
- Performance evaluation with accuracy and F1 scores
- Comparison between different classifier types

## 🎯 Example Results

On the Italy Power Demand dataset, you can expect:

| Classifier | Accuracy | F1 Score |
|------------|----------|----------|
| Random Forest | ~96% | ~0.96 |
| Linear | ~89% | ~0.90 |
| Gradient Boosting | ~95% | ~0.966 |

*Results may vary based on dataset and hyperparameters*

## 🔧 Workflow

1. **Install Dependencies**: Set up TiReX and required packages
2. **Load Data**: Import your time series dataset
3. **Preprocess**: Encode labels and convert to PyTorch tensors
4. **Train**: Fit your chosen classifier on the training data
5. **Evaluate**: Generate predictions and calculate metrics

## 📚 Use Cases

- **Energy Demand Forecasting**: Classify power consumption patterns
- **Medical Signal Analysis**: Categorize ECG, EEG signals
- **Industrial Monitoring**: Detect anomalies in sensor data
- **Finance**: Classify stock market trends
- **IoT Applications**: Pattern recognition in device data

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📖 Citation

If you use TiReX in your research, please cite the original paper:
```bibtex
@article{tirex2024,
  title={TiReX: Time Series Representation Extraction},
  author={NX-AI Team},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [TiReX GitHub Repository](https://github.com/NX-AI/tirex)
- [Documentation](https://github.com/NX-AI/tirex/tree/main/docs)
- [Example Notebooks](https://github.com/NX-AI/tirex/tree/main/examples)

## 💡 Tips

- Start with Random Forest for baseline performance
- Use data augmentation for small datasets
- Adjust batch_size based on your GPU memory
- Experiment with different classifier parameters for optimal results

---

Made with ❤️ using TiReX

**Questions?** Open an issue or reach out to the community!
