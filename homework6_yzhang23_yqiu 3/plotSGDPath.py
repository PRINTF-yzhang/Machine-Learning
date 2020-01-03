import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
from sklearn.decomposition import PCA
import math

## Network architecture
NUM_INPUT = 784  # Number of input neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

## Hyperparameters
## best hyper-parameters:
## hidden_50_lr_0.05_batch_16_epoch_100_validation
NUM_HIDDEN = 50
LEARNING_RATE = 0.05
BATCH_SIZE = 16
NUM_EPOCH = 40


# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("data/mnist_{}_images.npy".format(which))
    labels = np.load("data/mnist_{}_labels.npy".format(which))
    return images, labels

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    W1 = np.reshape(w[:NUM_INPUT * NUM_HIDDEN],(NUM_INPUT,NUM_HIDDEN))
    w = w[NUM_INPUT * NUM_HIDDEN:]
    b1 = np.reshape(w[:NUM_HIDDEN], NUM_HIDDEN)
    w = w[NUM_HIDDEN:]
    W2 = np.reshape(w[:NUM_HIDDEN*NUM_OUTPUT], (NUM_HIDDEN,NUM_OUTPUT))
    w = w[NUM_HIDDEN*NUM_OUTPUT:]
    b2 = np.reshape(w,NUM_OUTPUT)
    return W1, b1, W2, b2

def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def relu(x):
    relu_x = [max(i,0) for i in x]
    return np.array(relu_x)

def cross_entropy(y,y_hat):
    loss = [-y_i * math.log(y_hat_i) for y_i, y_hat_i in zip(y,y_hat)]
    return sum(loss)


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).

def fCE (X, Y, w):
    # print(X.shape)
    W1, b1, W2, b2 = unpack(w)

    cost = 0.0
    for x,y in zip(X,Y):
        # print(x.shape)
        z_1 = np.dot(W1.T,x) + b1
        # print(z_1.shape)
        h_1 = relu(z_1)
        # print(h_1.shape)
        z_2 = np.dot(W2.T,h_1) + b2
        y_hat = softmax(z_2)
        cost += cross_entropy(y,y_hat)

    return cost/len(Y)


def plotSGDPath (trainX, trainY, ws):
    def toyFunction (x1, x2):
        return np.sin((2 * x1**2 - x2) / 10.)

    pca = PCA(n_components=2)
    pca.fit(ws)
    print(ws.shape)
    # X = pca.transform(ws)

    fig = plt.figure()
    ax = Axes3D(fig)
    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-np.pi, +np.pi, 0.2)  # Just an example
    axis2 = np.arange(-np.pi, +np.pi, 0.2)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Xaxis = Xaxis.reshape((Xaxis.size,1))
    Yaxis = Yaxis.reshape((Yaxis.size,1))
    print(Xaxis.shape)
    print(Yaxis.shape)
    XY = np.concatenate((Xaxis,Yaxis),axis=1)
    print(XY.shape)
    T = np.matrix(pca.components_)
    print(T.shape)
    ws_transform = XY*T
    Zaxis = []
    print("ws_transform shape:",ws_transform.shape)
    for ws_t in ws_transform:
        ws_t = ws_t[0,:]
        print("ws_t shape:", ws_t.shape)

        loss = fCE(trainX, trainY, ws_t)
        print(ws_t.shape)
        Zaxis.append(loss)
    Zaxis = np.array(Zaxis)
    Zaxis = Zaxis.reshape((len(axis1), len(axis2)))

    # Zaxis = np.zeros((len(axis1), len(axis2)))
    # for i in range(len(axis1)):
    #     for j in range(len(axis2)):
    #         Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Yaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Zaxis = toyFunction(Xaxis, Yaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()
    plt.savefig('img/loss_landscape.png')



start_time = time.time()
if "trainX" not in globals():
    trainX, trainY = loadData("train")
    testX, testY = loadData("validation")

print("len(trainX): ", len(trainX))
print("len(testX): ", len(testX))

# Initialize weights randomly
W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
b1 = 0.01 * np.ones(NUM_HIDDEN)
W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
b2 = 0.01 * np.ones(NUM_OUTPUT)

NUM_SAMPLES = 2500
idxs = np.random.permutation(trainX.shape[0])[0:NUM_SAMPLES]
ws_name = 'result/hidden_50_lr_0.05_batch_16_epoch_40_ws.npy'
ws = np.load(ws_name)
plotSGDPath(trainX[idxs, :], trainY[idxs,:], ws)


