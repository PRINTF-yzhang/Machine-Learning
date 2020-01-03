#author Ying Zhang & Yujia Qiu
# CS 453X
# Professor Jacob Whitehill
# Hw2
import numpy as np
import matplotlib.pyplot as plt

#Visualizing the machine’s behavior
#create a 48 × 48 image representing the learned weights w (without the b term)
#from each of the different training methods
# source https://matplotlib.org/gallery/subplots_axes_and_figures/subplots_demo.html
def vmethod1(w1):
    figure,ele = plt.subplots()
    plot = np.reshape(w1[:-1], (48,48))
    ele.imshow(plot)
    ele.set_title('Method1')
    figure.show()

def vmethod2(w2):
    figure,ele = plt.subplots()
    plot2 = np.reshape(w2[:-1], (48,48))
    ele.imshow(plot2)
    ele.set_title('Method2')
    figure.show()

def vmethod3(w3):
    figure,ele = plt.subplots()
    plot3 = np.reshape(w3[:-1], (48,48))
    ele.imshow(plot3)
    ele.set_title('Method3')
    figure.show()

# Next, using the regressor in part (c), predict the ages of all the
#   images in the test set and report the RMSE (in years). Then, show the top 5 most egregious
#   errors
# numpy.argsort(a, axis=-1, kind='quicksort', order=None)
# Returns the indices that would sort an array.
# Perform an indirect sort along the given axis using the algorithm specified by the kind keyword. It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
def show_five_worse(w, Xtilde, y):
    yhat = np.dot(Xtilde.T, w)
    df = yhat - y
    index = df.argsort(axis = 00)[-5:][::1]
    error = Xtilde[:, index]
    #index = np.argmax(df)
    #np.argsort(x, axis=0)  # sorts along first axis (down)
    exp = df[index]
    show = y[index]
    figure, ele = plt.subplots(1,5)
    for i in range(5):
        err_plot = np.reshape(error[:-1,i],(48,48))
        ele[i].imshow(err_plot)
        ele[i].set_title(" predication: " +str(round(exp[i,0,0],2))+ " show: " + str(show[i,0,0]))
    plt.show()
                    
    

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    #xtilde=xtilde(:);
    #np.ones()
    TD = [face.flatten() for face in faces]
    set_array = np.array(TD)
    set_one = np.ones((len(faces), 1))
    return np.append(set_array, set_one, axis=1).T

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    yhat = np.dot(Xtilde.T, w)
    df = yhat - y
    df_square = np.square(df)
    return np.mean(df_square) / 2

def gradient_w(X,y,w,b):
    return X.dot(yhat(X, w, b) - y) / X.shape[1]
# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    gw = Xtilde.dot(Xtilde.T.dot(w) - y)
    gradientW = gw / np.shape(Xtilde)[1]
    penalty = (alpha / (2*np.shape(Xtilde)[1])) * w.T.dot(w)
    return gradientW + penalty

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
# One-shot (analytical) solution
# np.hstack, np.vstack, np.atleast
# numpy.eye(N, M=None, k=0, dtype=<class 'float'>, order='C')
# Return a 2-D array with ones on the diagonal and zeros elsewhere.
# numpy.linalg.solve(a, b)
# Solve a linear matrix equation, or system of linear scalar equations.
def method1 (Xtilde, y):
    XT= Xtilde.dot(Xtilde.T)
    Y = np.eye(len(Xtilde))
    arg = np.linalg.solve(XT, Y)
    return arg.dot(Xtilde).dot(y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
# T = 5000 with a step
# size (aka learning rate) of  = 0.003
# numpy.random.randn(d0, d1, ..., dn)
#Return a sample (or samples) from the “standard normal” distribution.
def method2 (Xtilde, y):
# : Pick a random starting value for w ∈ R2304 and b ∈ R and
# a small learning rate (e.g.,  = .001)
    m2 = np.random.randn(np.shape(Xtilde)[0], 1) * 0.001
    for t in range(5000):
        gfmse = gradfMSE (m2, Xtilde, y, alpha = 0.)
        m2 = m2 - gfmse * 0.003
    return m2

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    #random pickfrom normal distribution with a small learning rate
    m3 = np.random.randn(np.shape(Xtilde)[0], 1) * 0.001
    for t in range(5000):
        gfmse = gradfMSE (m3, Xtilde, y, alpha = ALPHA)
        m3 = m3 - gfmse * 0.003
    return m3


# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")[np.newaxis].T
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")[np.newaxis].T
    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)
    print("one shot solution")
    print(fMSE(w1, Xtilde_tr, ytr))
    print(fMSE(w1, Xtilde_te, yte))
    print(" Gradient descent solution")
    print(fMSE(w2, Xtilde_tr, ytr))
    print(fMSE(w2, Xtilde_te, yte))
    print(" Gradient descent with regularization solution")
    print(fMSE(w3, Xtilde_tr, ytr))
    print(fMSE(w3, Xtilde_te, yte))
    vmethod1(w1)
    vmethod2(w2)
    vmethod3(w3)
    show_five_worse(w3, Xtilde_te, yte)
