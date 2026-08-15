# ⚡ Deep Learning Model Optimization — Pruning, Quantization & Knowledge Distillation

An exploratory project comparing three popular techniques for making deep learning models smaller and faster: **pruning**, **quantization**, and **knowledge distillation**, implemented and visualized independently in PyTorch.

---

## 📌 Project Overview

Deep learning models are often over-parameterized for the task they solve, which makes them slower and heavier than necessary — a real constraint for deployment on mobile or embedded devices.

This project explores three widely used model compression techniques, each implemented as a standalone experiment on MNIST:

- **Knowledge Distillation** — training a small "student" model to imitate a larger "teacher" model.
- **Pruning** — removing low-importance weight connections from a trained network.
- **Quantization** — reducing the numerical precision of weights and activations (e.g. float32 → int8).

The focus of this project is on **understanding and visualizing how each technique affects a network's internal structure** (weight distributions, sparsity patterns, training dynamics) rather than on producing a single production-ready compressed model.

---

## 🧠 Techniques Explored

### 1. Knowledge Distillation (`KD.ipynb`)

A small "student" CNN is trained to reproduce the output distribution of a larger, pre-trained "teacher" CNN on MNIST, combining:

- a standard cross-entropy loss against the true labels, and
- a distillation loss (with temperature scaling) against the teacher's soft predictions.

The notebook tracks and plots the training loss of both the teacher and the student across epochs.

### 2. Pruning (`pruning.ipynb`)

Implements **L1-unstructured pruning** (via `torch.nn.utils.prune`) on a LeNet-style CNN, removing 50% of the weights in the convolutional layers based on their magnitude.

The notebook visualizes the raw vs. pruned weight tensors side by side, making the effect of pruning on the network's connectivity directly visible.

### 3. Quantization (`quantization.ipynb`)

Trains a simple CNN on MNIST and applies **dynamic quantization** in PyTorch, then visualizes the resulting weight distributions before and after quantization via histograms.

---

## 📂 Project Structure

```text
-Deep-Learning-Lab-DL-Models-Optimization/
│
├── KD.ipynb              # Knowledge distillation experiment
├── pruning.ipynb          # L1-unstructured pruning experiment
├── quantization.ipynb     # Dynamic quantization experiment
├── Data/                  # MNIST dataset (downloaded via torchvision)
└── ml-final_Demo.mp4      # Video walkthrough of the project
```

---

## 🛠️ Technologies Used

- **Language:** Python 3.10+
- **Framework:** PyTorch (`torch`, `torchvision`, `torch.nn.utils.prune`)
- **Visualization:** Matplotlib
- **Dataset:** MNIST (60,000 train / 10,000 test grayscale digit images, 28×28)

---

## ⚙️ Installation & Usage

```bash
git clone https://github.com/rihabatbir/-Deep-Learning-Lab-DL-Models-Optimization..git
cd -Deep-Learning-Lab-DL-Models-Optimization.
pip install torch torchvision matplotlib numpy
```

Each notebook is self-contained — open and run `KD.ipynb`, `pruning.ipynb`, or `quantization.ipynb` independently in Jupyter.

---

## 🔍 What Each Notebook Shows

| Notebook | What you'll see |
|---|---|
| `KD.ipynb` | Teacher vs. student training loss curves, illustrating how the student converges while learning from soft labels |
| `pruning.ipynb` | Side-by-side visualization of raw vs. pruned convolutional weights, showing the sparsity introduced by L1 pruning |
| `quantization.ipynb` | Weight distribution histograms before and after dynamic quantization |

---

## ⚠️ Scope & Limitations

This project prioritizes **understanding each technique's mechanics** over producing a rigorous quantitative benchmark. As a result:

- Each notebook currently reports **training loss**, not final test-set accuracy.
- The three techniques are **not yet benchmarked against a shared baseline** (same accuracy/model-size/inference-time comparison across all three).
- Results are illustrative (weight distributions, sparsity patterns) rather than tied to hard performance numbers.

These are natural next steps for extending the project into a full comparative benchmark — see below.

---

## 🚀 Possible Future Improvements

- Add a shared evaluation script measuring **test accuracy**, **model size (MB)**, and **inference time** before/after each technique, to produce a single comparative table.
- Combine pruning and quantization on the same model to study their compounding effect.
- Extend experiments beyond MNIST to a more complex dataset (e.g. CIFAR-10).
- Explore structured pruning (removing whole filters/channels) instead of unstructured weight pruning, for actual inference speed-up on standard hardware.

---

##  Work Group
