# MNIST 0/1 Classifier
A simple classification of hand drawn 0s and 1s using the MNIST dataset, implemented by using MLP.

# Prerequisites

```bash
    python 3.12 -m venv nn
    source nn/bin/acivate
    pip install -r [requirements.text]
```

# Configuration
It has 3 different gradient descent algorithms, Stochastic Gradient Descent (SGD), Momentum Based Gradient Descent (momentum) and, Adam.

In the train.py all the three algortihms have been written just remove it as it is commented and you are ready to use it!

# Project structure
```text    
├── gui
│   ├── app.py
│   ├── canvas_utils.py
│   ├── inference.py
│   ├── __init__.py
│   ├── preprocess.py
├── models
│   ├── __init__.py
│   ├── mlp.py
│   └── read.py
├── README.md
├── requirements.txt
├── train.py
└── utils
    ├── dataset.py
    ├── metrics.py
    ├── optimizer.py
    ├── plot.py
    └── train_utils.py
```

# Usage
In order to see the graphical analysis of each algorithms performance just run `train.py`, it will give you the desired comparison.  
In order to see how well does the model classifies a drawn digit simply run the command:
```bash
    python -m gui.app
```
# Author
Phenesin (https://github.com/Phenesin)