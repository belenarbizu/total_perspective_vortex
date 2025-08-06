# Total Perspective Vortex
This project implements a Brain-Computer Interface (BCI) based on electroencephalographic (EEG) data using machine learning algorithms. The system processes motor imagery data to classify brain signals into different movement intentions (hand vs. feet movements) in real-time.

---

## Project Structure

- `mybci.py`: Main script for training and prediction with command-line interface
- `preprocessing.py`: EEG data parsing, filtering, and feature extraction using MNE
- `csp.py`: Custom implementation of Common Spatial Patterns (CSP) algorithm
- `pca.py`: Custom Principal Component Analysis (PCA) implementation
- `wavelet_transform.py`: Wavelet transform implementation for signal processing
- `pipeline.py`: Complete processing pipeline integrating all components
- `visualization.py`: Utilities for data visualization and analysis

---

## Features

- Custom CSP Implementation: From-scratch Common Spatial Patterns algorithm for spatial filtering
- Custom PCA Implementation: Manual Principal Component Analysis for dimensionality reduction
- Wavelet Transform: Advanced signal processing using wavelet decomposition
- Real-time Classification: Stream processing with <2s latency requirement
- Cross-validation: Robust model evaluation using cross_val_score

---

## Getting Started

1. Clone the repository:

```bash
git clone <repo-url>
cd total-perspective-vortex
```

2. Install dependencies:

```bash
pip install numpy scipy scikit-learn mne matplotlib
```

3. Download the PhysioNet dataset (the dataset is not included in the repository)

---

### Training a Model

Train on a specific subject and experiment:
```bash
python mybci.py 4 14 train
```
This will load and preprocess EEG data for subject 4, experiment 14
Train the model using PCA and Linear Discriminant Analysis
Display cross-validation scores

Example output:
```bash
[1.         1.         1.         0.66666667 0.66666667 0.66666667 1.         0.66666667 0.33333333 0.66666667]

cross_val_score: 0.7666666666666666
```

### Making Predictions
Predict on test data with real-time simulation:
```bash
python mybci.py 4 14 predict
```

Example output:
```bash
epoch 0: 3 2 False
epoch 1: 3 3 True
epoch 2: 3 3 True
Accuracy:  0.6666666666666666
```

### Full Evaluation
Run complete evaluation across all subjects and experiments:
```bash
python mybci.py
```
Example output:
```bash
experiment 0: subject 001: accuracy = 0.6
experiment 0: subject 002: accuracy = 0.8
....
Mean accuracy of the six different experiments for all 109 subjects:
experiment 0: accuracy = 0.5991
experiment 1: accuracy = 0.5718
experiment 2: accuracy = 0.7130
experiment 3: accuracy = 0.6035
experiment 4: accuracy = 0.5937
experiment 5: accuracy = 0.6753
Mean accuracy of 6 experiments: 0.6261
```

---

## Technical Implementation
### Signal Processing Pipeline

- Preprocessing: Raw EEG data filtering and artifact removal using MNE
- Feature Extraction: Power spectral density computation using wavelet transform
- Dimensionality Reduction: Custom PCA algorithm for feature space optimization
- Classification: Machine learning classifier for motor imagery detection

## Principal Component Analysis (PCA)
The PCA implementation is built from scratch using sklearn's BaseEstimator and TransformerMixin classes, making it fully compatible with sklearn pipelines.

### Mathematical Formulation
For EEG signals `{E_n}^N_{n=1} ∈ R^{ch×time}`:

- N: Number of events per class

- ch: Number of electrodes/channels

- time: Length of event recording

The algorithm processes the signal matrix `X ∈ R^{N×d}` where d = ch × time.

### PCA Algorithm Steps:

1. Data Centering:
```bash
X_centered = X - μ
where μ = mean(X, axis=0)
```

2. Covariance Matrix:
```bash
cov_matrix = np.cov(X_standardize, rowvar=False)
```

3. Eigendecomposition:
```bash
eigenvalues, eigenvectors = eigh(cov_matrix)
```

4. Component Selection:

Select top-k eigenvectors based on largest eigenvalues

5. Transformation:
```bash
X_PCA = np.dot(x_centered, self.components)
```

---

### Dataset
The project uses the PhysioNet EEG Motor Movement/Imagery Dataset containing:

- 109 subjects
- 6 different experimental paradigms
- Motor imagery tasks (hand vs. feet movements)
- Multi-channel EEG recordings

---

### Dependencies

- Python 3.x
- NumPy: Numerical computing and linear algebra
- SciPy: Scientific computing and signal processing
- scikit-learn: Machine learning pipeline integration
- MNE: EEG data processing and visualization
- matplotlib: Data visualization and plotting

---

### Architecture
The system implements a complete BCI pipeline following machine learning best practices:

- Data Loading: MNE-based EEG data parsing from PhysioNet
- Preprocessing: Bandpass filtering and artifact removal
- Feature Engineering: Power spectral density and spatial covariance
- Dimensionality Reduction: Custom CSP/PCA implementation
- Classification: Linear Discriminant Analysis from sklearn
- Evaluation: Cross-validation and real-time testing
