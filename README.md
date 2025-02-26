# Neural Network from Scratch - Binary Classification

This project implements a **3-layer Neural Network** from scratch using **NumPy** for binary classification. The model predicts whether a person belongs to a specific category based on height and weight data.

---

## Table of Contents
- [Overview](#overview)
- [Network Architecture](#network-architecture)
- [Dataset](#dataset)
- [Activation Function](#activation-function)
- [Cost Function](#cost-function)
- [Backpropagation](#backpropagation)
- [Gradient Descent](#gradient-descent)
- [How to Run](#how-to-run)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contact](#contact)

---

## Overview
This implementation builds a neural network from scratch to learn the mapping between inputs (height and weight) and a binary output (0 or 1). The training is performed using the **Gradient Descent** optimization algorithm with manual backpropagation.

---

## Network Architecture
The neural network consists of:
- **Input Layer**: 2 neurons (Height, Weight)
- **Hidden Layer 1**: 3 neurons
- **Hidden Layer 2**: 3 neurons
- **Output Layer**: 1 neuron (Binary Classification)

---

## Dataset
The dataset consists of 10 training examples with height and weight as inputs:
```python
x = np.array([
    [150, 70],
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
```

---

## Activation Function
The **Sigmoid Activation Function** is used for non-linearity:
\[
g(z) = \frac{1}{1 + e^{-z}}
\]

---

## Cost Function
The **Binary Cross-Entropy Loss** is used to measure the error:
\[
J(w, b) = -\frac{1}{m} \sum \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
\]

---

## Backpropagation
The model uses manual backpropagation to calculate gradients for each layer:
1. **Layer 3 (Output Layer)**:
   - Calculate gradients for weights \( w3 \), biases \( b3 \), and activations from the previous layer.

2. **Layer 2 (Hidden Layer)**:
   - Propagate gradients backward to calculate \( w2 \) and \( b2 \).

3. **Layer 1 (Hidden Layer)**:
   - Calculate gradients for \( w1 \) and \( b1 \).

---

## Gradient Descent
Weights and biases are updated using **Gradient Descent**:
\[
w = w - \alpha \frac{\partial J}{\partial w}
\]
- Learning Rate (α): `0.1`
- Epochs: `100`

---

## How to Run
1. **Open in Google Colab or any Python environment.**
2. **Install required dependencies** (if not already installed):
    ```bash
    pip install numpy matplotlib
    ```
3. **Run the script** to train the model:
    ```python
    costs = train()
    ```
4. **Visualize the cost reduction**:
    ```python
    import matplotlib.pyplot as plt
    plt.plot(costs)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Cost Reduction Over Time")
    plt.show()
    ```

---

## Results
- Initial Cost: `0.8409`
- Final Cost after 100 epochs: `0.6932`
- The cost decreases over epochs, indicating the model is learning.

---

## Dependencies
- **NumPy** for numerical calculations.
- **Matplotlib** for cost visualization.

