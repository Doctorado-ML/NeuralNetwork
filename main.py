import numpy as np
import matplotlib.pyplot as plt
import time
from n_network import N_Network, plot_decision_boundary


def load_planar_dataset(random_seed):
    np.random.seed(random_seed)
    m = 400 # number of examples
    N = int(m / 2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2 # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2 # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T
    return X, Y

random_seed = 1
Xtrain, ytrain = load_planar_dataset(random_seed)
X = Xtrain.T
y = ytrain.T
print('X', X.shape, 'y', y.shape)

# Visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y.T[0], s=40, cmap=plt.cm.Spectral);
plt.title('Dataset')
plt.show();

#Define a four layer network
nu = [X.shape[1], 10, 7, 5, 1]
xg = [0, N_Network.relu, N_Network.relu, N_Network.relu, N_Network.sigmoid]
xgprime = [0, N_Network.relu_prime, N_Network.relu_prime, N_Network.relu_prime, N_Network.sigmoid_prime]
init_params = dict(m=X.shape[0], n=X.shape[1], n_units=nu, g=xg, optim='sgd',
                   gprime=xgprime, epochs=10000, alpha=0.075)
nd = N_Network(init_params)
nd.set_seed(random_seed)
costs = nd.train(X, y)
print("First cost: {0:.6f} final cost: {1:.6f}".format(costs[0], costs[-1]))
print("Number of units in each layer: ", nu)
nd.print_time()
nd.plot_costs()
pred = nd.valid(X, y)
indices = nd.mislabeled(y, pred)
# Plot decission boundary
plot_decision_boundary(nd, X, y, True, '4 Layers N_Network')