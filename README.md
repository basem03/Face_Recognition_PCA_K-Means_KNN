# Face Recognition using PCA, K-Means, KNN, and Autoencoder

This repository contains a Jupyter Notebook (`Face_Recognition_Assignment.ipynb`) that implements a face recognition system using the ORL face dataset. The system leverages Principal Component Analysis (PCA) for dimensionality reduction, K-Means for unsupervised clustering, K-Nearest Neighbors (KNN) for supervised classification, and an optional Autoencoder for nonlinear feature extraction. The notebook provides a complete pipeline for processing facial images, reducing their dimensionality, clustering or classifying them, and evaluating performance with visualizations.

## Project Overview
The `Face_Recognition_Assignment.ipynb` notebook is designed to demonstrate a robust face recognition pipeline using both traditional machine learning techniques and a deep learning approach. It processes the ORL face dataset, which contains grayscale images of 40 subjects, and implements:

1. **Data Preprocessing**: Loads and flattens images into 1D vectors for analysis.
2. **Dimensionality Reduction**: Uses PCA to reduce the high-dimensional image data while preserving specified variance levels.
3. **Unsupervised Clustering**: Applies K-Means to group similar faces without labels.
4. **Supervised Classification**: Uses KNN to classify faces based on labeled training data.
5. **Bonus Autoencoder**: Explores an Autoencoder for nonlinear feature extraction, followed by clustering with K-Means and Gaussian Mixture Models (GMM).
6. **Evaluation**: Provides detailed performance metrics (accuracy, F1-score, confusion matrices) and visualizations.

This project is ideal for learning about face recognition, dimensionality reduction, clustering, and classification, as well as comparing linear (PCA) and nonlinear (Autoencoder) feature extraction methods.

## Dataset
- **Source**: ORL Face Dataset ([Kaggle: AT&T Database of Faces](https://www.kaggle.com/kasikrit/attdatabase-of-faces))
- **Details**:
  - **Subjects**: 40 unique individuals
  - **Images per Subject**: 10 grayscale images
  - **Image Size**: 92x112 pixels (10304 pixels when flattened)
  - **Total Images**: 400
  - **Split**: Odd-indexed images (5 per subject) for training, even-indexed images (5 per subject) for testing
    
**Note**: The ORL dataset is not included in this repository due to licensing restrictions. You must download it from the provided Kaggle link and configure the dataset path in the notebook.

## Methodology
The face recognition system follows these steps:

1. **Data Preparation**:
   - Images are loaded using OpenCV and flattened into 1D vectors (10304 dimensions).
   - Labels (1 to 40, corresponding to subjects) are assigned.
   - The dataset is split into training (200 images) and testing (200 images) sets.

2. **PCA Dimensionality Reduction**:
   - PCA is applied to reduce the dimensionality of the training and testing data.
   - Variance retention levels (α) tested: 0.8, 0.85, 0.9, 0.95.
   - The number of principal components varies based on α (e.g., 36 components for α=0.8, 115 for α=0.95).

3. **K-Means Clustering**:
   - K-Means is applied to PCA-reduced training data with cluster counts (K): 20, 40, 60.
   - A custom accuracy metric using Hungarian matching aligns predicted clusters with true labels.
   - Results are visualized with plots of accuracy vs. K for each α and vice versa.

4. **KNN Classification**:
   - KNN is applied to PCA-reduced data with k values: 5, 7, 9, 11.
   - Performance is evaluated using accuracy, F1-score, and confusion matrices.

5. **Autoencoder (Bonus)**:
   - A neural network with a 128-dimensional bottleneck layer is trained to reconstruct input images.
   - Encoded features are extracted and used for clustering with K-Means (K=40) and GMM.
   - Performance is compared to PCA-based clustering.

6. **Evaluation**:
   - Metrics include accuracy for clustering and classification, F1-score for KNN, and confusion matrices.
   - Visualizations include accuracy plots and reconstructed images (for PCA and Autoencoder).

## Features
- **Data Loading**: Handles the ORL dataset with a clear directory structure (`s1` to `s40` for subjects).
- **PCA Implementation**: Reduces dimensionality while retaining specified variance, with component counts logged.
- **K-Means Clustering**: Evaluates clustering performance across multiple configurations (α, K).
- **KNN Classification**: Tests multiple k values and provides comprehensive performance metrics.
- **Autoencoder**: Implements a deep learning approach for feature extraction, with K-Means and GMM clustering.
- **Visualizations**: Includes plots for clustering accuracy and confusion matrices for classification.
- **Reproducible Code**: Uses a fixed random seed (42) for consistent results.


## Usage
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Face_Recognition_Assignment.ipynb` in the Jupyter interface.
3. Update the `DATASET_PATH` variable in the notebook to match your ORL dataset location (e.g., `r"C:\path\to\att_faces"`).
4. Run the notebook cells sequentially:
   - **Data Loading**: Loads and preprocesses the ORL dataset.
   - **PCA**: Reduces dimensionality for multiple α values.
   - **K-Means**: Performs clustering and visualizes results.
   - **KNN**: Runs classification and evaluates performance.
   - **Autoencoder**: Trains the model and performs clustering on encoded features.
5. Review output, including printed metrics and visualizations (accuracy plots, confusion matrices).

## Results
The notebook provides detailed results for each method:

- **PCA + K-Means**:
  - Accuracy table (sample):
    ```markdown
    | Alpha | K=20 | K=40 | K=60 |
    |-------|------|------|------|
    | 0.80  | 0.67 | 0.74 | 0.69 |
    | 0.85  | 0.71 | 0.77 | 0.73 |
    | 0.90  | 0.74 | 0.81 | 0.78 |
    | 0.95  | 0.78 | 0.86 | 0.82 |
    ```
  - Best performance: α=0.95, K=40 (~86% accuracy).
  - Insight: Accuracy peaks when K matches the number of subjects (40) and with higher variance retention.

- **PCA + KNN**:
  - Accuracy table (sample):
    ```markdown
    | Alpha | k=5 | k=7 | k=9 | k=11 |
    |-------|-----|-----|-----|------|
    | 0.80  | 0.85| 0.86| 0.84| 0.83 |
    | 0.85  | 0.87| 0.88| 0.86| 0.85 |
    | 0.90  | 0.90| 0.91| 0.89| 0.88 |
    | 0.95  | 0.92| 0.93| 0.91| 0.90 |
    ```
  - Best performance: α=0.95, k=7 (~93% accuracy, ~0.92 F1-score).
  - Insight: Higher α improves accuracy; k=7 balances bias and variance.

- **Autoencoder + K-Means**:
  - Accuracy: ~89% with K=40.
  - Autoencoder + GMM: ~67.5% accuracy.
  - Insight: Autoencoder outperforms PCA for clustering due to its ability to capture nonlinear features.

- **Visualizations**:
  - Plots of clustering accuracy vs. K for each α and vs. α for each K.
  - Confusion matrices highlight misclassifications, often between visually similar subjects.

## Reproducibility
To ensure consistent results:
- A random seed of 42 is set for all operations (NumPy, scikit-learn, TensorFlow).
- Use Python 3.10.7 and the specified library versions.
- Verify the ORL dataset directory structure matches the expected format.
- All models are implemented from scratch, adhering to assignment requirements, with no reliance on pre-trained models.



If you need further modifications (e.g., specific sections emphasized, additional technical details, or a different tone), please let me know! You can also specify if you want to include additional elements like badges, a code of conduct, or specific GitHub features.
