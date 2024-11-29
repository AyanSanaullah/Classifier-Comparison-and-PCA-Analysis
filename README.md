# Classifier Comparison and PCA Analysis

This project implements and compares various machine learning classifiers, including a custom-built Bayesian Classifier and scikit-learn models, on a dataset. It evaluates their performance using metrics such as accuracy, precision, recall, and F1-score. Additionally, the impact of dimensionality reduction through Principal Component Analysis (PCA) is assessed.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Insights and Analysis](#insights-and-analysis)
8. [Future Enhancements](#future-enhancements)
9. [License](#license)

---

## Overview
This project explores the performance of several classifiers on a dataset. The pipeline includes data preprocessing, training classifiers, and evaluating them with and without dimensionality reduction using PCA. It also features a custom Bayesian Classifier implemented from scratch, compared against established machine learning models.

---

## Features
- **Custom Bayesian Classifier**: 
  - Implements Naive Bayes-like functionality without using external libraries.
  
- **Scikit-learn Models**:
  - Support Vector Machines (Linear and RBF kernels).
  - K-Nearest Neighbors.
  - Gaussian Naive Bayes.
  - Linear Discriminant Analysis.

- **Dimensionality Reduction**:
  - PCA for feature reduction, retaining 95% of variance.

- **Evaluation Metrics**:
  - Accuracy, precision, recall, F1-score, and confusion matrix for comprehensive analysis.

- **Data Visualization**:
  - PCA scatter plot of the first two components.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `numpy`: Mathematical computations.
  - `pandas`: Data manipulation and analysis.
  - `scikit-learn`: Machine learning models and evaluation metrics.
  - `matplotlib`: Data visualization.

---

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AyanSanaullah/Classifier-Comparison-and-PCA-Analysis
   ```
2. Navigate to the project directory:
   ```bash
   cd classifier-comparison-pca
   ```
3. Replace `#your file path here` in the code with the path to your CSV dataset file.

---

## Usage
1. **Preprocess the data**:
   - Load the dataset, handle missing values, and scale features using `StandardScaler`.

2. **Train and Evaluate Classifiers**:
   - Train a custom Bayesian Classifier.
   - Train scikit-learn models including SVM (Linear and RBF), KNN, Naive Bayes, and LDA.

3. **Apply PCA**:
   - Reduce the feature dimensions while retaining 95% variance.
   - Evaluate classifier performance on the PCA-transformed data.

4. **Visualize Results**:
   - Generate a PCA scatter plot of the first two principal components.

5. **Analyze Results**:
   - Compare performance metrics of classifiers before and after PCA.

---

## Results
### Performance Comparison (Before PCA)
### Without PCA  
| Classifier      | Accuracy  | Precision  | Recall     | F1-Score  |
|-----------------|-----------|------------|------------|-----------|
| Bayesian        | 93.997%   | 93.975%    | 93.997%    | 93.541%   |
| SVM (Linear)    | 91.195%   | 86.906%    | 91.195%    | 88.150%   |
| SVM (RBF)       | 97.265%   | 97.350%    | 97.265%    | 96.979%   |
| KNN             | 98.223%   | 98.195%    | 98.223%    | 98.157%   |
| Naive Bayes     | 93.997%   | 93.975%    | 93.997%    | 93.541%   |
| LDA             | 90.936%   | 91.127%    | 90.936%    | 88.688%   |

---

### With PCA  
| Classifier      | Accuracy  | Precision  | Recall     | F1-Score  |
|-----------------|-----------|------------|------------|-----------|
| Bayesian        | 93.997%   | 93.975%    | 93.997%    | 93.541%   |
| SVM (Linear)    | 91.195%   | 86.906%    | 91.195%    | 88.150%   |
| SVM (RBF)       | 97.265%   | 97.350%    | 97.265%    | 96.979%   |
| KNN             | 98.223%   | 98.195%    | 98.223%    | 98.157%   |
| Naive Bayes     | 93.997%   | 93.975%    | 93.997%    | 93.541%   |
| LDA             | 90.936%   | 91.127%    | 90.936%    | 88.688%   |


---

## Insights and Analysis
- **Top Performer**: K-Nearest Neighbors consistently performed the best across all metrics, both pre- and post-PCA.
- **SVM Kernel Selection**: The RBF kernel outperformed the linear kernel, demonstrating the importance of capturing non-linear patterns in data.
- **Dimensionality Reduction**: PCA retained critical information, as performance metrics remained consistent for most classifiers.
- **Custom Bayesian Classifier**: Performed competitively, validating its implementation.

---

## Future Enhancements
1. Add support for other dimensionality reduction techniques (e.g., t-SNE, LLE).
2. Incorporate additional classifiers such as Random Forest and Gradient Boosting.
3. Automate hyperparameter tuning for each classifier using GridSearchCV or RandomizedSearchCV.
4. Improve visualizations with more detailed plots.

---

## License
This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.
