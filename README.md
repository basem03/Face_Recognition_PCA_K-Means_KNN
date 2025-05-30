# Face Recognition using PCA, K-Means, KNN, and Autoencoder

This repository contains a Jupyter Notebook implementing a face recognition system using the ORL face dataset. The system employs Principal Component Analysis (PCA) for dimensionality reduction, K-Means for unsupervised clustering, K-Nearest Neighbors (KNN) for supervised classification, and an optional Autoencoder for feature extraction.

## Overview
The `Face_Recognition_Assignment.ipynb` notebook demonstrates a comprehensive face recognition pipeline. It processes grayscale images from the ORL dataset, applies dimensionality reduction, performs clustering and classification, and evaluates the results. A bonus section explores the use of an Autoencoder for feature extraction as an alternative to PCA.

## Dataset
- **Source**: ORL Face Dataset ([Kaggle: AT&T Database of Faces](https://www.kaggle.com/kasikrit/attdatabase-of-faces))
- **Details**:
  - 40 subjects, each with 10 grayscale images
  - Image size: 92x112 pixels
  - Total images: 400
  - Split: 5 images per subject for training, 5 for testing


## Features
- **Data Preparation**: Loads and flattens images into 1D vectors (10304 dimensions) and splits them into training and testing sets.
- **PCA Dimensionality Reduction**: Reduces data dimensionality while retaining 80%, 85%, 90%, or 95% of variance.
- **K-Means Clustering**: Performs unsupervised clustering with 20, 40, or 60 clusters and evaluates using a Hungarian matching-based accuracy metric.
- **KNN Classification**: Applies supervised classification with k values of 5, 7, 9, and 11.
- **Autoencoder (Bonus)**: Implements a neural network to extract 128-dimensional features, followed by K-Means and Gaussian Mixture Model (GMM) clustering.
- **Evaluation**: Includes accuracy, F1-score, and confusion matrices for performance analysis, with visualizations of clustering results.


## Results
- **PCA + K-Means**:
  - Best performance at α=0.95, K=40 with ~86% accuracy.
  - Accuracy improves with higher variance retention and when the number of clusters matches the number of subjects (40).
- **PCA + KNN**:
  - Best performance at α=0.95, k=7 with ~93% accuracy.
  - Higher k values slightly reduce accuracy due to oversmoothing.
- **Autoencoder + K-Means**:
  - Achieves ~89% accuracy with K=40, outperforming PCA at similar dimensionality due to nonlinear feature extraction.

Visualizations and detailed results are included in the notebook, including plots of accuracy vs. K/α and confusion matrices.

## Reproducibility
- A random seed of 42 is set for all operations to ensure consistent results.
- Ensure the ORL dataset is correctly configured with the expected directory structure.
- All models are implemented from scratch as per the assignment requirements, with no external pre-trained models used.

## Contributing
Contributions are welcome! Please contact me basemhesham200318@gmail.com or submit a pull request or open an issue for suggestions, bug reports, or improvements.


