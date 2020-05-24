'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
Neural Network implementation based on the Andrew Ng courses
Implements Batch GD, Stochastic GD (minibatch_size=1) & Stochastic minibatch GD:
 -Cost function: Cross Entropy Loss
 -Activation functions: relu, sigmoid, tanh
 -Regularization: l2 (lambd), Momentum (beta), Dropout (keep_prob)
 -Optimization: Minibatch Gradient Descent, RMS Prop, Adam
 -Learning rate decay, computes a factor of the learning rate at each # of epochs
 -Fair minibatches: Can create batches with the same proportion of labels 1/0 as in train data
Restriction:
 -Multiclass only with onehot label
'''

import time
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .Metrics import Metrics


# Cost function (Cross-entropy):
# Compute the cross-entropy cost $J$
# $$ J = -\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1 - y^{(i)})\log\left(1 - a^{[L](i)}\right)) \tag{7}$$


class N_Network:
    
    def __init__(self, hyperparam):
        # NN State
        self._ct = 0  # Time inverted in computation
        self._optim = {}  # Update parameters functions depending on the optimization algorithm
        self._optim_update = None  # update function selected
        self._optim_selected = ''
        self._multiclass = False  # Is it a multiclass classification problem?
        self._epochs_decay = ()  # (decay rate, applied each # epochs)
        self._verbose = False
        # Hyperparams
        self._L = 0  # Number of layers including the input layer
        self._n_units = []  # Number of units in each layer
        self._g = []  # Activation functions of each layer
        self._gprime = []  # Derivative of the activation functions needed in backpropagation
        self._alpha = 0  # Learning rate in gradient descent
        self._beta = 0  # Momentum coefficient / acts as beta1 in adam
        self._beta2 = 0.999  # RMS Prop coefficient
        self._epsilon = 1e-8  # RMS Prop value to prevent division by zero
        self._params = {}  # dict of parameters
        self._epochs = 0  # Number of iterations to train
        self._seed = 2020  # Random seed
        self._lambd = 0  # Regularization coefficient
        self._keep_prob = 1  # dropout regularization
        self._minibatch_size = 0  # Number of samples to take into account to upgrade parameters
        self._fair_minibatches = False  # Wether or not create fair minibatches
        if 'filename' in hyperparam:
            self.load(hyperparam['filename'])
            return
        self._m = hyperparam['m']
        self._n = hyperparam['n']
        self._n_units = hyperparam['n_units']
        self._g = hyperparam['g']
        self._gprime = hyperparam['gprime']
        self._alpha = hyperparam['alpha']
        self._learning_rate = self._alpha
        self._epochs = hyperparam['epochs']
        self._L = len(self._n_units)
        # ensures that at most, only one regularization method is chosen
        if 'lambd' in hyperparam:
            self._lambd = hyperparam['lambd']
        else:
            if 'keep_prob' in hyperparam:
                self._keep_prob = hyperparam['keep_prob']
        if 'minibatch_size' in hyperparam:
            self._minibatch_size = hyperparam['minibatch_size']
        else:
            self._minibatch_size = self._m
        if 'fair_minibatches' in hyperparam:
            self._fair_minibatches = hyperparam['fair_minibatches']
        optim = {
            'adam': self._update_parameters_adam,
            'sgd': self._update_parameters_sgd,
            'rms': self._update_parameters_rms
        }
        self._optim_selected = hyperparam['optim']
        self._optim_update = optim[self._optim_selected]
        if hyperparam['optim'] != 'sgd':
            self._beta = 0.9  # if opt. algorithm is rms or adam set default beta/beta1
        if 'beta' in hyperparam:
            self._beta = hyperparam['beta']
        np.random.seed(self._seed)
        if 'multiclass' in hyperparam:
            self._multiclass = hyperparam['multiclass']
        if 'epochs_decay' in hyperparam:
            self._epochs_decay = hyperparam['epochs_decay']
        self.initialize()

    # Activation functions
    @staticmethod
    def softmax(x):  # stable softmax
        exps = np.exp(x - np.max(x))
        return exps / exps.sum(axis=0, keepdims=True)

    @staticmethod
    def softmax_prime(x):
        return 1

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def sigmoid_prime(x):
        s = N_Network.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu_prime(x):
        return np.greater(x, 0).astype(int)

    @staticmethod
    def tanh_prime(x):
        z = N_Network.tanh(x)
        return 1 - z * z

    def initialize(self):
        # Initialize dictionaries of Parameters
        b = {}
        W = {}
        Z = {}
        A = {}
        dZ = {}
        dW = {}
        db = {}
        vdW = {}
        vdb = {}
        SdW = {}
        Sdb = {}
        for i in range(self._L):
            if self._verbose:
                print("Initializing %d layer..." % i)
            # Help ease the vanishing / Exploding gradient problem
            cte = 0.01
            if self._g[i] == self.relu:
                # Make Var(W) = 2 / n
                cte = np.sqrt(2 / self._n_units[i - 1])
            else:
                # based on Xavier initialization makes var(W) = 1 / n
                if self._g[i] == self.tanh:
                    cte = 1 / np.sqrt(self._n_units[i - 1])
                else:
                    # makes var(W) = 2 / n
                    if self._g[i] == self.sigmoid:
                        prev_layer = (i - 1) if i > 0 else 0
                        cte = np.sqrt(
                            2 / (self._n_units[prev_layer] + self._n_units[i]))
            # Don't need W and b and its optimizers for the input layer
            if i > 0:
                W[i] = np.random.randn(
                    self._n_units[i], self._n_units[i - 1]) * cte
                b[i] = np.zeros((self._n_units[i], 1))
                dW[i] = np.zeros(
                    (self._n_units[i], self._n_units[i - 1] if i > 0 else self._minibatch_size))
                db[i] = np.zeros((self._n_units[i], 1))
                vdW[i] = np.zeros(
                    (self._n_units[i], self._n_units[i - 1] if i > 0 else self._minibatch_size))
                vdb[i] = np.zeros((self._n_units[i], 1))
                SdW[i] = np.zeros(
                    (self._n_units[i], self._n_units[i - 1] if i > 0 else self._minibatch_size))
                Sdb[i] = np.zeros((self._n_units[i], 1))
            A[i] = np.zeros(
                (self._n_units[i], self._minibatch_size if i < self._L else 1))
            Z[i] = np.zeros(
                (self._n_units[i], self._minibatch_size if i < self._L else 1))
            dZ[i] = np.zeros((self._n_units[i], self._minibatch_size))

        self._params = dict(b=b, W=W, Z=Z, A=A, dZ=dZ, dW=dW,
                            db=db, vdW=vdW, vdb=vdb, SdW=SdW, Sdb=Sdb)

    def get_accuracy(self, y, ypred, direct_result=False):
        m = y.shape[0]
        met = Metrics(y, ypred)
        ac = met.accuracy()
        right = met.correct()
        if direct_result:
            return ac
        return "Accuracy: {0:.3f}% ({1} of {2})".format(100 * ac, right, m)

    def get_metrics(self, y, ypred):
        return Metrics(y, ypred)

    def plot_costs(self):
        plt.plot(self._costs)
        plt.ylabel('Cost (cross-entropy)')
        plt.xlabel('Epochs')
        plt.title("Epochs: {0} Learning rate: {1}".format(
            self._epochs, self._learning_rate))
        plt.show()

    def plot_confusion_matrix(self, y, yhat, title='', figsize=(10, 7), scale=1.4):
        cm = Metrics(y, yhat).confusion_matrix()
        plt.figure(figsize=figsize)
        sns.set(font_scale=scale)
        fig = sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
        x = fig.set_title("{0} ({1}) / {2}". format(title,
                                                    self._optim_selected, self.get_accuracy(y, yhat)))
        x = fig.set_xlabel('Predicted')
        x = fig.set_ylabel('Truth')
        # fig.invert_yaxis()

    def check_dimensions(self):
        for i in range(self._L):
            print("i={0}, b({1}, W{2}, A{3}, Z{4}, vdW{5}, vdb{6}, SdW{7}, Sdb{8}, dW{9}, db{10}\n".format(
                i, self._params['b'][i].shape if i > 0 else ' XXX',
                self._params['W'][i].shape if i > 0 else ' XXX',
                self._params['A'][i].shape,
                self._params['Z'][i].shape,
                self._params['vdW'][i].shape if i > 0 else ' XXX',
                self._params['vdb'][i].shape if i > 0 else ' XXX',
                self._params['SdW'][i].shape if i > 0 else ' XXX',
                self._params['Sdb'][i].shape if i > 0 else ' XXX',
                self._params['dW'][i].shape if i > 0 else ' XXX',
                self._params['db'][i].shape if i > 0 else ' XXX'
            ))

    def get_params(self):
        return self._params

    def num_minibatches(self):
        return math.floor(self._m / self._minibatch_size) + (0 if self._m % self._minibatch_size == 0 else 1)

    def create_minibatches(self, X, y):
        return self.create_fair_minibatches(X, y) if self._fair_minibatches else self.create_random_minibatches(X, y)

    def _balance_sets(self, y):
        """
        Returns:
        class0: category 0 indexes
        class1: category 1 indexes 
        num0: number of samples of 0 category to include in the minibatch
        num1: number of samples of 1 category to include in the minibatch
        """
        class_one = np.array(np.where(y == 1))[0]
        class_zero = np.array(np.where(y == 0))[0]
        percent = len(class_one) / len(y)
        num_class0 = math.floor((1 - percent) * self._minibatch_size)
        num_class1 = self._minibatch_size - num_class0
        return num_class0, num_class1, class_zero, class_one

    def create_fair_minibatches(self, X, y):
        """
        Creates a list of random minibatches from (X, y) 

        """
        mini_batches = []
        num_zero, num_one, class_zero, class_one = self._balance_sets(y)
        # Compute categorized shuffled sets
        X0 = X[class_zero]
        X1 = X[class_one]
        y0 = y[class_zero]
        y1 = y[class_one]
        permutation0 = list(np.random.permutation(len(class_zero)))
        permutation1 = list(np.random.permutation(len(class_one)))
        shuffledX0 = X0[permutation0, :]
        shuffledX1 = X1[permutation1, :]
        shuffledY0 = y0[permutation0, :]
        shuffledY1 = y1[permutation1, :]
        size = self._minibatch_size

        num = math.floor(self._m / size)
        for k in range(num):
            # Inserts the category 0 elements to mini batch
            miniX = shuffledX0[k * num_zero:(k + 1) * num_zero, :]
            miniY = shuffledY0[k * num_zero:(k + 1) * num_zero, :]
            # Appends the cateogory 1 elements to mini batch
            miniX = np.vstack((miniX, X1[k * num_one:(k + 1) * num_one, :]))
            miniY = np.vstack((miniY, y1[k * num_one:(k + 1) * num_one, :]))
            mini_batch = (miniX, miniY)
            mini_batches.append(mini_batch)
        if self._m % num != 0:
            miniX = shuffledX0[num * num_zero:y0.shape[0], :]
            miniY = shuffledY0[num * num_zero:y0.shape[0], :]
            miniX = np.vstack((miniX, X1[num * num_one:y1.shape[0], :]))
            miniY = np.vstack((miniY, y1[num * num_one:y1.shape[0], :]))
            mini_batch = (miniX, miniY)
            mini_batches.append(mini_batch)
        return mini_batches

    def create_random_minibatches(self, X, y):
        """
        Creates a list of random minibatches from (X, y) 

        """
        mini_batches = []
        permutation = list(np.random.permutation(self._m))
        shuffledX = X[permutation, :]
        shuffledY = y[permutation, :]
        size = self._minibatch_size
        num = math.floor(self._m / size)
        for k in range(num):
            miniX = shuffledX[k * size:(k + 1) * size, :]
            miniY = shuffledY[k * size:(k + 1) * size, :]
            mini_batch = (miniX, miniY)
            mini_batches.append(mini_batch)
        if self._m % size != 0:
            miniX = shuffledX[num * size:self._m, :]
            miniY = shuffledY[num * size:self._m, :]
            mini_batch = (miniX, miniY)
            mini_batches.append(mini_batch)
        return mini_batches

    def _compute_Sd(self, i):
        self._params['SdW'][i] = self._beta2 * self._params['SdW'][i] + \
            (1 - self._beta2) * np.square(self._params['dW'][i])
        self._params['Sdb'][i] = self._beta2 * self._params['Sdb'][i] + \
            (1 - self._beta2) * np.square(self._params['db'][i])
        return self._params['SdW'][i], self._params['Sdb'][i]

    def _compute_vd(self, i):
        self._params['vdW'][i] = self._beta * self._params['vdW'][i] + \
            (1 - self._beta) * self._params['dW'][i]
        self._params['vdb'][i] = self._beta * self._params['vdb'][i] + \
            (1 - self._beta) * self._params['db'][i]
        return self._params['vdW'][i], self._params['vdb'][i]

    def _update_parameters_rms(self, t):
        for i in range(1, self._L):
            SdW, Sdb = self._compute_Sd(i)
            dW = self._params['dW'][i]
            db = self._params['db'][i]
            self._params['W'][i] -= self._alpha * \
                dW / (np.sqrt(SdW) + self._epsilon)
            self._params['b'][i] -= self._alpha * \
                db / (np.sqrt(Sdb) + self._epsilon)

    def _update_parameters_adam(self, t):
        for i in range(1, self._L):
            vdW, vdb = self._compute_vd(i)
            SdW, Sdb = self._compute_Sd(i)
            vdW_corr = vdW / (1 - math.pow(self._beta, 2))
            vdb_corr = vdb / (1 - math.pow(self._beta, 2))
            SdW_corr = SdW / (1 - math.pow(self._beta2, t))
            Sdb_corr = Sdb / (1 - math.pow(self._beta2, t))
            self._params['W'][i] -= self._alpha * \
                vdW_corr / (np.sqrt(SdW_corr) + self._epsilon)
            self._params['b'][i] -= self._alpha * \
                vdb_corr / (np.sqrt(Sdb_corr) + self._epsilon)

    def _update_parameters_sgd(self, t):
        for i in range(1, self._L):
            vdW, vdb = self._compute_vd(i)
            self._params['W'][i] -= self._alpha * vdW
            self._params['b'][i] -= self._alpha * vdb

    def set_verbose(self, verbose):
        self._verbose = verbose

    def set_seed(self, seed):
        self._seed = seed
        np.random.seed(self._seed)

    def _cost_function(self, yhat, y):
        """
        Compute cost (cross-entropy) of prediction

        yhat: vector of predictions, shape (number of examples, 1)
        Y:  vector of labels, shape (number of examples, 1)

        Returns: cost
        """
        if self._multiclass:
            cost = -np.mean(y * np.log(yhat + self._epsilon))
        else:
            cost = -np.sum(np.nansum(y * np.log(yhat) + (1 - y)
                                     * np.log(1 - yhat))) / self._minibatch_size
        # Add regularization term
        cost += self._lambd / (2 * self._minibatch_size) * \
            np.sum([np.sum(np.square(x)) for x in self._params['W']])
        assert(cost.shape == ())
        return cost

    def _get_prediction(self, transform=False):
        res = self._get_AL().T
        if transform:
            if self._multiclass:
                return np.argmax(res, axis=1)
            else:
                return np.round(res).astype(int)
        return res

    def _get_AL(self):
        return self._params['A'][self._L - 1]

    def _backward_propagation(self, y):
        AL = self._get_AL()
        Y = y.T
        assert(Y.shape == AL.shape)
        if self._multiclass:
            dA = AL - Y
        else:
            # derivative of cost with respect to A[L]
            dA = np.nan_to_num(-(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)))
        for i in reversed(range(1, self._L)):
            dZ = dA * self._gprime[i](self._params['Z'][i])
            dW = dZ.dot(self._params['A'][i - 1].T) / self._minibatch_size + \
                (self._lambd / self._minibatch_size) * self._params['W'][i]
            db = np.sum(dZ, axis=1, keepdims=True) / self._minibatch_size
            dA = self._params['W'][i].T.dot(dZ)
            self._params['dW'][i] = dW
            self._params['db'][i] = db

    def train(self, X, y):
        return self.fit(X, y)

    def fit(self, X, y):
        self._costs = []
        tic = time.time()
        if self._verbose:
            print('Training neural net...{0} epochs with {1} minibatches'.format(
                self._epochs, self.num_minibatches()))
        divider = 1 if self._epochs < 100 else 100
        t = 0
        for e in range(self._epochs):
            minibatches = self.create_minibatches(X, y)
            cost_total = 0
            for minibatch in minibatches:
                Xt, yt = minibatch
                self._forward_propagation(Xt, train=True)
                # Compute gradient descent
                self._backward_propagation(yt)
                t += 1  # Only used in adam
                self._optim_update(t)
                cost_total += self._cost_function(self._get_prediction(), yt)
            cost_avg = cost_total / self.num_minibatches()
            self._costs.append(cost_avg)
            if e % divider == 0 and self._verbose:
                print("Epoch: {0} Cost {1:.8f}".format(e, cost_avg))
            if self._epochs_decay != ():
                (rate, number) = self._epochs_decay
                if e > 0 and e % number == 0:
                    self._alpha *= rate
                    if self._verbose:
                        print(
                            "*Setting learning rate (alpha) to: {0}".format(self._alpha))
        self._ct = time.time() - tic
        self._alpha = self._learning_rate
        if self._verbose:
            self.print_time()
        return self._costs

    def print_time(self):
        print("Elapsed time: {0:.2f} s".format(self._ct))

    def _forward_propagation(self, X, train=False):
        self._params['A'][0] = X.T
        for i in range(1, self._L):
            if train and self._keep_prob != 1:
                d = np.random.rand(*self._params['A'][i].shape)
                d = (d < self._keep_prob).astype(int)
                '''
                 divide by self._keep_prob is done to keep the same behavior of the neuron in training with dropout and in
                 testing without dropout. "This is important because at test time all neurons see all their inputs, 
                 so we want the outputs of neurons at test time to be identical to their expected outputs at training time"
                 (Stanford CS231n Convolutional Neural Networks for Visual Recognition)
                '''
                self._params['A'][i] = (
                    self._params['A'][i] * d) / self._keep_prob  # inverted dropout
            self._params['Z'][i] = self._params['W'][i].dot(
                self._params['A'][i - 1]) + self._params['b'][i]
            self._params['A'][i] = self._g[i](self._params['Z'][i])
        prediction = self._get_AL()

    def predict(self, X):
        self._forward_propagation(X, train=False)
        if self._multiclass:
            yhat = np.argmax(self._get_prediction(False), axis=1)
        else:
            yhat = self._get_prediction(transform=True)
        return yhat

    def predict_proba(self, X):
        self._forward_propagation(X, train=False)
        return self._get_prediction(transform=False)

    def evaluate(self, X, y, transform=True):
        return self.valid(X, y, transform)

    def valid(self, X, y, transform=True, score=False):
        if X.shape[0] != y.shape[0]:
            print('Dimension error X, y', X.shape, y.shape)
        yhat = self.predict(X)
        ypred = self._get_prediction(transform=True)
        if score:
            return self.get_accuracy(y, ypred, direct_result=True)
        print(self.get_accuracy(y, ypred))
        return yhat

    def score(self, X, y):
        return self.valid(X, y, score=True)

    def mislabeled(self, y, ypred, target=1):
        return Metrics(y, ypred).fn_indices(target)

    def save(self, name=''):
        try:
            filename = "{0}.nn".format(name)
            f = open(filename, 'wb')
            pickle.dump(self.__dict__, f, 2)
            f.close()
        except:
            print("I couldn't write the file ", filename)
            return False
        return True

    def load(self, filename):
        try:
            f = open(filename, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
        except:
            print(filename, " doesn't exists or I couldn't open it.")
            return False
        self.__dict__.update(tmp_dict)
        return True

    def compact_state(self):
        return {
            "_m": self._m,
            "_n": self._n
        }
