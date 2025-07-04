# MakeMore: Character-Level Language Model

_A step-by-step implementation of a character-level language model in a Jupyter Notebook, inspired by Andrej Karpathy's "makemore" series._

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-^2.0-orange.svg)
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-informational)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Concepts Covered](#concepts-covered)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [License](#license)

---

## 📌 Overview

This project is a deep dive into the fundamentals of language modeling. The `MakeMore(1).ipynb` notebook guides you through building and training neural networks that can generate human-like names, one character at a time. It starts with a simple bigram model and progressively builds up to a more complex Multi-Layer Perceptron (MLP).

The primary goal is to provide a clear, hands-on understanding of how modern language models work at their core.

---

## ✨ Concepts Covered

This notebook explores several key concepts in deep learning and natural language processing:

-   **Bigram & N-gram Models**: Understanding character probabilities based on preceding characters.
-   **Multi-Layer Perceptron (MLP)**: Building a neural network with embedding, hidden, and output layers.
-   **Character Embeddings**: Representing characters as learnable vectors.
-   **PyTorch Tensors**: Performing all computations using the PyTorch library.
-   **Forward & Backward Pass**: Implementing the training loop, including loss calculation and gradient descent.
-   **Loss Functions**: Using Negative Log-Likelihood to measure model performance.
-   **Batching**: Processing data in mini-batches for efficient training.
-   **Model Sampling**: Generating new names from the trained probability distribution.

---

## 🗂️ Dataset

The model is trained on a dataset of names, typically a simple text file where each line contains one name. This allows the model to learn the statistical patterns of character sequences in names.

---

## 📦 Prerequisites

Ensure you have Python 3 and the following libraries installed. You can install them using pip:

```bash
pip install torch matplotlib jupyterlab
```

---

## 🚀 Usage

1.  **Clone the repository** (if applicable):
    ```bash
    git clone https://github.com/preethamak/MakeMore.git
    cd <project-directory>
    ```

2.  **Launch Jupyter Notebook or JupyterLab**:
    ```bash
    jupyter notebook "MakeMore(1).ipynb"
    ```

3.  **Run the cells**: Execute the notebook cells sequentially to process the data, build the model, train it, and generate new names.

---

## 🧠 Model Architecture

The final model is a Multi-Layer Perceptron (MLP) with the following structure:

1.  **Embedding Layer**: Converts input characters (context) into dense vectors.
2.  **Hidden Layer**: A non-linear layer (`softmax` activation) that learns complex patterns from the embeddings.
3.  **Output Layer**: Produces logits for each character in the vocabulary, representing the probability distribution for the next character.

---

## 📄 License

This project is licensed under the MIT License.
