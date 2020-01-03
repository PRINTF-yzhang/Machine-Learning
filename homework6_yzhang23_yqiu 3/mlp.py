import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import math
import sys
import time
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d, Axes3D


## Network architecture
NUM_INPUT = 784  # Number of input neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

## Hyperparameters

train_flag = False

if train_flag:
    NUM_HIDDEN = int(sys.argv[1])  # Number of hidden neurons
    LEARNING_RATE = float(sys.argv[2])
    BATCH_SIZE = int(sys.argv[3])
    NUM_EPOCH = int(sys.argv[4])
else:
    ## best hyper-parameters:
    ## hidden_50_lr_0.05_batch_16_epoch_100_validation
    NUM_HIDDEN = 50
    LEARNING_RATE = 0.05
    BATCH_SIZE = 16
    NUM_EPOCH = 40

print("NUM_HIDDEN: ", NUM_HIDDEN)
print("LEARNING_RATE: ", LEARNING_RATE)
print("BATCH_SIZE: ", BATCH_SIZE)
print("NUM_EPOCH: ", NUM_EPOCH)
out_file_name = 'result/hidden_%s_lr_%s_batch_%s_epoch_%s_' % (str(NUM_HIDDEN), str(LEARNING_RATE), str(BATCH_SIZE),str(NUM_EPOCH))
print(out_file_name)

"""
1. Number of units in the hidden layer (suggestions: {30, 40, 50}) 
2. Learning rate (suggestions: {0.001, 0.005, 0.01, 0.05, 0.1, 0.5})
3. Minibatch size (suggestions: 16, 32, 64, 128, 256)
4. Number of epochs
5. Regularization strength
"""

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

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    W1_ = np.reshape(W1,NUM_INPUT*NUM_HIDDEN)
    # print(W1_.shape)
    W2_ = np.reshape(W2,NUM_HIDDEN*NUM_OUTPUT)
    # print(W2_.shape)
    w = np.concatenate((W1_,b1, W2_, b2))
    # print(w.shape)
    return w

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("data/mnist_{}_images.npy".format(which))
    labels = np.load("data/mnist_{}_labels.npy".format(which))
    return images, labels



def plotSGDPath (trainX, trainY, ws):
    def toyFunction (x1, x2):
        return np.sin((2 * x1**2 - x2) / 10.)

    pca = PCA(n_components=2)
    pca.fit(ws)
    # X = pca.transform(ws)

    a = np.matrix()
    W_ = np.matrix(pca.components_)


    fig = plt.figure()
    ax = Axes3D(fig)
    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    axis2 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)


    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Yaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Zaxis = toyFunction(Xaxis, Yaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()

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

def cal_acc(X, Y, w):
    # print(X.shape)
    W1, b1, W2, b2 = unpack(w)

    hit_target = 0
    for x, y in zip(X, Y):
        # print(x.shape)
        z_1 = np.dot(W1.T, x) + b1
        # print(z_1.shape)
        h_1 = relu(z_1)
        # print(h_1.shape)
        z_2 = np.dot(W2.T, h_1) + b2
        y_hat = softmax(z_2)
        # print("y_hat: ",np.argmax(y_hat))
        # print("y: ",np.argmax(y))
        if np.argmax(y_hat) == np.argmax(y):
            hit_target += 1
    acc = hit_target/float(len(Y))
    return acc

def relu_delta(x):
    relu_delta_x = [1 if i>=0 else 0 for i in x]
    return np.array(relu_delta_x)


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    delta_W_1 = np.zeros((NUM_INPUT, NUM_HIDDEN))
    delta_b_1 = np.zeros(NUM_HIDDEN)
    delta_W_2 = np.zeros((NUM_HIDDEN, NUM_OUTPUT))
    delta_b_2 = np.zeros(NUM_OUTPUT)
    for x,y in zip(X,Y):
        z_1 = np.dot(W1.T, x) + b1
        h_1 = relu(z_1)
        z_2 = np.dot(W2.T, h_1) + b2
        y_hat = softmax(z_2)
        delta_W_2 += np.dot(h_1.reshape(NUM_HIDDEN,1),(y_hat - y).reshape(1,NUM_OUTPUT))
        delta_b_2 += y_hat - y
        g = np.dot((y_hat - y).reshape(1,NUM_OUTPUT), W2.T)*relu_delta(z_1)
        delta_W_1 += np.dot(x.reshape(NUM_INPUT,1),g)
        delta_b_1 += g.reshape(NUM_HIDDEN)
    N = len(Y)
    delta_W_1 /= N
    delta_b_1 /= N
    delta_W_2 /= N
    delta_b_2 /= N

    # print("The shape of delta_W_1:", delta_W_1.shape)
    # print("The shape of delta_b_1:", delta_b_1.shape)
    # print("The shape of delta_W_2:", delta_W_2.shape)
    # print("The shape of bdelta_b_22:", delta_b_2.shape)
    delta = pack(delta_W_1, delta_b_1, delta_W_2, delta_b_2)
    return delta

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train(trainX, trainY, testX, testY, w):
    print('w.shape:', w.shape)
    ws = [w]
    global_step = 0
    N = len(trainX)
    train_loss = []
    validation_loss = []
    test_acc = []
    for epoch in range(NUM_EPOCH):
        tr_loss = 0.0
        idxs = np.random.permutation(trainX.shape[0])
        batch_num = int(N / BATCH_SIZE)
        for index in range(batch_num):
            batch_idxs = idxs[index*BATCH_SIZE:(index+1)*BATCH_SIZE]  ## the current batch
            batch_x = trainX[batch_idxs,:]
            batch_y = trainY[batch_idxs,:]
            batch_loss = fCE(batch_x,batch_y,w)
            tr_loss += batch_loss * BATCH_SIZE
            global_step += 1
            grad = gradCE(batch_x,batch_y,w)
            w = w - LEARNING_RATE * grad
            if global_step % 500 == 0:
                print("[global step: %d] [loss_batch: %f]" % (global_step, batch_loss))
                ws.append(w)
        ### the last batch

        batch_idxs = idxs[batch_num * BATCH_SIZE:]  ## the last batch
        if len(batch_idxs) > 0:
            batch_x = trainX[batch_idxs, :]
            batch_y = trainY[batch_idxs, :]
            batch_loss = fCE(batch_x, batch_y, w)
            if global_step % 50 == 0:
                print("[global step: %d] [loss_batch: %f]" % (global_step, batch_loss))
            global_step += 1
            tr_loss += batch_loss * (N - batch_num * BATCH_SIZE)
            grad = gradCE(batch_x, batch_y, w)
            w = w - LEARNING_RATE * grad
            ws.append(w)

        tr_loss /= N
        print("[epoch: %d] [loss_train: %f]" % (epoch, tr_loss))
        train_loss.append([epoch, global_step, tr_loss])
        loss_val = fCE(testX, testY, w)
        print("[epoch: %d] [loss_validation: %f]\n" % (epoch, loss_val))
        validation_loss.append([epoch,global_step,loss_val])
        acc = cal_acc(testX, testY, w)
        test_acc.append([epoch,global_step,acc])
        print("[epoch: %d] [test_acc: %f]\n" % (epoch, acc))

    train_loss = np.array(train_loss)
    validation_loss = np.array(validation_loss)
    test_acc = np.array(test_acc)
    ws = np.array(ws)
    np.save(out_file_name + 'train_loss.npy',train_loss)
    np.save(out_file_name + 'validation_loss.npy',validation_loss)
    np.save(out_file_name + 'test_acc.npy',test_acc)
    np.save(out_file_name + 'ws.npy',ws)
    return ws

if __name__ == "__main__":
    # Load data
    start_time = time.time()
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        if train_flag:
            testX, testY = loadData("validation")
        else:
            testX, testY = loadData("test")

    print("len(trainX): ", len(trainX))
    print("len(testX): ", len(testX))

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    # print("The shape of W1:", W1.shape)
    # print("The shape of b1:", b1.shape)
    # print("The shape of W2:", W2.shape)
    # print("The shape of b2:", b2.shape)
    w = pack(W1, b1, W2, b2)

    W1_, b1_, W2_, b2_ = unpack(w)
    # print((W1 - W1_).view())
    # print((b1 - b1_).view())
    # print((W2 - W2_).view())
    # print((b2 - b2_).view())

    # Check that the gradient is correct on just a few examples (randomly drawn).

    check_flag = False
    if check_flag:
        idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
        discrepancy = scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                        lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                        w)
        print("The discrepancy between the gradients of the implemented and the approximation: %f" % discrepancy)

        if discrepancy < 0.01:
            print("My implemented cost and gradient functions are correct")


    # # Train the network and obtain the sequence of w's obtained using SGD.
    ws = train(trainX, trainY, testX, testY, w)

    end_time = time.time()

    print("running time: ", end_time - start_time)

    # # Plot the SGD trajectory
    NUM_SAMPLES = 2500
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_SAMPLES]
    plotSGDPath(trainX[idxs,:], trainY[:,idxs], ws)
