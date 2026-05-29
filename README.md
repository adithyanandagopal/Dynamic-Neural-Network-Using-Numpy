# NeuralNetwork from Scratch using numpy only

> A fully dynamic neural network — personal project built to understand what runs behind modern AI.

\---

## Origin

This project started as a simple experiment — training an XOR gate using a multilayer perceptron. That small experiment raised more questions than it answered. How does backpropagation actually work? What is a loss function really doing? Why batch training?

One question led to the next, and eventually led to this — a fully dynamic neural network built from scratch using only NumPy. No PyTorch. No Keras. No TensorFlow. Every operation written by hand.

\---

## Features

* **Fully dynamic** — define any layer architecture in one line
* **Forward pass** — ReLU on hidden layers, Softmax on output
* **Backpropagation** — delta rule implemented manually
* **Mini-batch gradient descent** — configurable batch size
* **He weight initialisation** — proper init for ReLU networks
* **Cross entropy loss** — with epsilon for numerical stability
* **Training curves** — loss and accuracy plotted after training
* **Weight saving/loading** — pickle based persistence
* **Test evaluation** — accuracy on test set after training

\---

## Architecture

```
Input Layer
    ↓
Hidden Layer 1  (ReLU)
    ↓
Hidden Layer 2  (ReLU)
    ↓  
  ...
    ↓
Output Layer    (Softmax or Sigmoid)
```

Fully configurable — you define the number of layers and neurons:

```python
# shallow
nn = model(structure=\[784, 128, 10], base\_address="path/")

# deeper
nn = model(structure=\[784, 256, 128, 64, 10], base\_address="path/")

# your own
nn = model(structure=\[input\_size, ...hidden..., num\_classes], base\_address="path/")
```

\---

## Activations

|Layer|Activation|Why|
|-|-|-|
|Hidden layers|ReLU|Fast, avoids vanishing gradient|
|Output layer|Softmax|Converts raw scores to probabilities|
|Output layer |Sigmoid| for binary-converts single neuron output to probability

\---

## CSV Format

Dataset must be a CSV with this structure:

```
label, pixel1, pixel2, pixel3, ... pixelN
3, 107, 118, 127, ...
5, 201, 198, 175, ...
```

First column = `label`, remaining columns = features.

\---

## Preprocessing

User is responsible for preprocessing before passing data:

```python
# normalize pixel values
x = x / 255.0

# flatten to correct shape
x = x.reshape(1, input\_size)

# labels must be 0-indexed
# 0 to num\_classes - 1
```

\---

## Installation

```bash
pip install numpy pandas matplotlib
```

\---

## Usage

### Train

```python
from model import model

nn = model(
    structure    = \[784, 128, 64, 24],
    base\_address = "path/to/save/"    # include trailing slash
)

nn.fit(
    train\_data = "path/to/train.csv",
    test\_data  = "path/to/test.csv",
    epoch      = 10,
    alpha      = 0.01,
    batch\_size = 64
)
# weights auto-saved to base\_address/model.pkl after training
```

### Predict

```python
# preprocess first
x = image.flatten().astype(float) / 255.0
x = x.reshape(1, 784)

# run
nn.run("path/to/model.pkl", x)
# prediction = 3 with 91.2% confidence
```

\---

## What this taught me

* How forward propagation works mathematically
* Deriving and implementing backpropagation from scratch
* What gradient descent is actually doing to the weights
* Why batch training is used over full dataset updates
* He weight initialisation and why it matters for ReLU networks
* How softmax and cross entropy work together
* Dynamic architecture design in Python
* General coding and software design skills

\---

## For Learners

This project is a good starting point if you want to understand the basics of machine learning from the ground up. Reading through the source code will show you:

* What a forward pass looks like in matrix form
* How gradients flow backwards through layers
* How weights actually get updated
* What batch training looks like in a real loop

\---

## Learning Resources used

* [GeeksforGeeks](https://www.geeksforgeeks.org) — backpropagation fundamentals, loss functions, activations
* Claude (Anthropic) and ChatGPT — used to evaluate progress, explain concepts,identify bugs,and track the journey. No AI generated architecture code.
* Various ML blogs and documentation

\---

## Honest Note

This is a learning project, not a production framework. Weight saving/loading was the one part that was vibe coded with AI assistance. Everything else — architecture, forward pass, backpropagation, training loop, dynamic layer system — was designed and written independently.The lessons I learned from AND ,OR, XOR training followed by mnist data set training made me confident and curious which led me to this project.

\---

*First year EC student | Learning what runs behind modern AI | Built from scratch*

