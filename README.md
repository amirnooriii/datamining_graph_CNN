# Data Mining Assignment 3

This project covers two distinct, yet interconnected, areas within the expansive field of data mining: graph theory and neural networks. It meticulously demonstrates the generation and insightful visualization of random graphs utilizing the powerful NetworkX library, offering a foundational understanding of network structures. Furthermore, it delves into the core principles of deep learning by showcasing the comprehensive training and rigorous evaluation of a Convolutional Neural Network (CNN) built with PyTorch, specifically tailored for the challenging task of image classification on the widely recognized CIFAR-10 dataset. This dual approach provides a holistic view of both symbolic and connectionist methods in data analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Graph Analysis](#graph-analysis)
- [Neural Network for Image Classification](#neural-network-for-image-classification)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This Jupyter Notebook (`datamining_assignment3 (1).ipynb`) serves as a comprehensive and practical demonstration of fundamental concepts and techniques in data mining, touching upon both structural analysis and pattern recognition:

### Graph Generation and Visualization
This component explores the systematic creation of random graphs using the NetworkX library. It not only generates these complex network structures but also provides methods for their clear and intuitive visualization. Understanding graph structures is crucial in many real-world applications, from social network analysis and epidemiology to transportation logistics and biological pathways.

### Image Classification with Neural Networks
This segment focuses on the practical implementation, diligent training, and thorough evaluation of a Convolutional Neural Network (CNN) using the PyTorch framework. The objective is to classify images from the CIFAR-10 dataset. This section provides a hands-on experience with building and assessing a deep learning model for a common computer vision problem.

The project utilizes popular Python libraries including:
- `pandas` and `numpy` for data manipulation,
- `NetworkX` for graph theory applications,
- `torch` and `torchvision` for deep learning models.

## Graph Analysis

This section explores generating and visualizing a random directed graph.

- **Graph Type:** A **directed random graph** is generated, where edge direction matters (A → B ≠ B → A).
- **Parameters:**
  - Nodes: `n = 30`
  - Edge probability: `p = 0.2`
  - Random seed: `seed = 5` for reproducibility.
- **Visualization:** The graph is visualized to highlight network density, isolated nodes, hubs, and connectivity.

## Neural Network for Image Classification

This section covers building, training, and evaluating a CNN using PyTorch.

### Model Architecture
A straightforward CNN is implemented using `torch.nn`, including:
- Convolutional layers
- ReLU activation
- Pooling layers
- Fully connected layers

### Training
- Uses multiple epochs
- Loss function: `CrossEntropyLoss`
- Optimizer: `SGD` or `Adam`
- Training monitored by observing loss reduction

### Evaluation
- Model performance evaluated on unseen test data
- Metric: **Accuracy**

## Dataset

**CIFAR-10**: A dataset with 60,000 color images (32x32 px), across 10 classes:
- 50,000 training images
- 10,000 test images

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Installation

Install required libraries using `pip`:

```bash
pip install NetworkX matplotlib torch torchvision
