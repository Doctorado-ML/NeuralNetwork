# N_Network

Neural Network implementation based on the Andrew Ng courses

Implements Batch GD, Stochastic GD (minibatch_size=1) & Stochastic minibatch GD:

- Cost function: Cross Entropy Loss
- Activation functions: relu, sigmoid, tanh
- Regularization: l2 (lambd), Momentum (beta), Dropout (keep_prob)
- Optimization: Minibatch Gradient Descent, RMS Prop, Adam
- Learning rate decay, computes a factor of the learning rate at each # of epochs
- Fair minibatches: Can create batches with the same proportion of labels 1/0 as in train data

Restriction:

- Multiclass only with onehot label

## Install

```bash
pip install git+https://github.com/doctorado-ml/NeuralNetwork
```

## Example

#### Console

```bash
python main.py
```

#### Jupyter Notebook

[![Test](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doctorado-ML/NeuralNetwork/blob/master/test.ipynb) Test notebook

