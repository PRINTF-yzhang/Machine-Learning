#author Ying Zhang & Yujia Qiu
# CS 453X
# Professor Jacob Whitehill
# Hw3
# pip3 install scikit-image
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import math
import scipy.sparse
import random

#train a softmax regressor to classify images of hand-written digits from the MNIST dataset.

def getPreds(someX,y,w):
    preds = np.zeros_like(someX.dot(w))
    preds[np.arange((someX.dot(w)).shape[0]),(someX.dot(w)).argmax(1)] = 1
    return preds

def fPC(someX, y, w):
    preds = getPreds(someX, y, w)
    return np.mean(preds == y)

#cited from source https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

#cited from source https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
def softmax(someX):
    someX -= np.max(someX)
    sm = (np.exp(someX).T / np.sum(np.exp(someX),axis=1)).T
    return sm

def reshapeAndAppend1s (someX):
    #xtilde=xtilde(:);
    #np.ones()
    #TD = [face.flatten() for face in faces]
    #set_array = np.array(TD)
    ap = np.append(someX, np.ones((someX.shape[0], 1)), axis=1)
    return ap

def gradient_w(someX,y,w):
    #return X.dot(yhat(X, w, b) - y) / X.shape[1]
    return -np.dot(someX.T,(y - (softmax(someX.dot(w))))) / someX.shape[0]

def cross_entropy(someX,y,w):
    #loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w)
    return -np.sum(y*np.log(softmax(someX.dot(w)))) / y.shape[0]
                   
#source https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/
def next_batch(X, y, batchSize):
    # loop over our dataset `X` in mini-batches of size `batchSize`
    for i in np.arange(0, X.shape[0], batchSize):
        # yield a tuple of the current batched data and labels
        yield (X[i:i + batchSize], y[i:i + batchSize])
#source https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/
def sto_gra_dec(someX,y,epochs,batchSize,alpha = 0.1):
    # initializeweight matrix such it has the same number of columns as input features
    W = np.random.randn(someX.shape[1],10)*0.1
    lossHistory = []
    # loop over the desired number of epochs
    for epoch in np.arange(0, epochs):
        # initialize the total loss for the epoch
        epochLoss = []
        # loop over data in batches
        for batchX, batchY in next_batch(someX, y, batchSize):
            loss = cross_entropy(batchX,batchY,W)
            epochLoss.append(loss)
            # the gradient update is the dot product between the transpose of our current batch and the error on the batch
            gradient = gradient_w(batchX,batchY,W)
           # use the gradient computed on the current batch to take a "step" in the correct direction
            W += -alpha * gradient
                   
            # update loss history list by taking the average loss across all batches
        lossHistory.append(np.average(epochLoss))
    return W, lossHistory
#part 2
#translation, rotation, scaling,random noise.
# cited from source https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
def augment_rotation(X, angle):
    #skt.rotate(x.reshape(28, 28), 20)
    #reshape(-1, 784)
    #could not broadcast input array from shape (28,28) into shape (784)
    return skimage.transform.rotate(X.reshape(28, 28), 20).flatten()


def augment_noise(X):
    return X + (0.01*np.random.randn(28**2)+0.01)

#def augment_translate_up(X)

def augment_scale(X):
    rx = np.zeros(X.shape)
    rx[X > 0.001] = 0.3
    X += rx
    X[X>1] =1
    return X

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

def parse(someX, y):
    parses = np.random.permutation(someX.shape[0])
    px = someX[parses]
    py = y[parses]
    return px, py

def AUG_SGD(X, y):
    tdata = X
    tlable = y
    random.seed(1)
    for j in range(3):
        rx = np.zeros(X.shape)
        ry = np.zeros(y.shape)
        for index,pic in enumerate(X):
            l = y[index]
            num = l.nonzero()[0][0]
            if num in (0, 9):
                pic = augment_rotation(pic, 90)
            pic = augment_scale(pic)
            pic = augment_noise(pic)
            img = augment_rotation(pic, 90)
            rx[index, :] = pic
            ry[index, :] = l
        tdata = np.concatenate([tdata, rx])
        tlable = np.concatenate([tlable, ry])

    return tdata, tlable

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("small_mnist_train_images.npy"))
    ytr = np.load("small_mnist_train_labels.npy")
    tdata = np.load("small_mnist_train_images.npy")
    tlable = np.load("small_mnist_train_labels.npy")
    Xtilde_tr, ytr = parse(Xtilde_tr, ytr)
    Xtilde_tr_aug_u, ytr_aug = AUG_SGD(tdata,tlable)
    Xtilde_tr_aug = reshapeAndAppend1s(Xtilde_tr_aug_u)
    Xtilde_tr_aug,ytr_aug = parse(Xtilde_tr_aug,ytr_aug)
    Xtilde_te = reshapeAndAppend1s(np.load("small_mnist_test_images.npy"))
    yte = np.load("small_mnist_test_labels.npy")
    W, lossHistory = sto_gra_dec(Xtilde_tr, ytr,100,100)
    print("-------------------------data of SGD---------------------------")
    print(" Training cost: ")
    print(cross_entropy(Xtilde_tr, ytr, W))
    print(" Training percent-correct accuracy: ")
    print(fPC(Xtilde_tr, ytr, W))
    print(" Testing cost: ")
    print(cross_entropy(Xtilde_te, yte, W))
    print(" Testing percent-correct accuracy: ")
    print(fPC(Xtilde_te, yte, W))
    
    print("-------------------------data of AUG SGD---------------------------")
    print(" Training cost: ")
    print(cross_entropy(Xtilde_tr_aug, ytr_aug, W))
    print(" Training percent-correct accuracy: ")
    print(fPC(Xtilde_tr_aug, ytr_aug, W))
    print(" Testing cost: ")
    print(cross_entropy(Xtilde_te, yte, W))
    print(" Testing percent-correct accuracy: ")
    print(fPC(Xtilde_te, yte, W))

    tdata = tdata
    tlable = tlable
    random.seed(1)
    for j in range(3):
        rx = np.zeros(tdata.shape)
        ry = np.zeros(tlable.shape)
        for index,pic in enumerate(tdata):
            l = tlable[index]
            num = l.nonzero()[0][0]
            if num in (0, 9):
                pic = augment_rotation(pic, 90)
            pic = augment_scale(pic)
            pic = augment_noise(pic)
            img = augment_rotation(pic, 90)
            rx[index, :] = pic
            ry[index, :] = l
        tdata = np.concatenate([tdata, rx])
        tlable = np.concatenate([tlable, ry])
        index = 22
        plt.imshow(tdata[index].reshape((28, 28)))
        plt.show()
        plt.imshow(tdata[index + 5000].reshape((28, 28)))
        plt.show()
        plt.imshow(tdata[index + 10000].reshape((28, 28)))
        plt.show()
