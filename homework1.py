# author Ying Zhang & Yujia Qiu
# CS 453X
# Professor Jacob Whitehill
# Part 1: Python and numpy
import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return np.dot(A, B)-C

def problem3 (A, B, C):
    return A*B + C.T

def problem4 (x, y):
    return x.T*y

#return a matrix with the same dimensions as A but that contains all zeros
# use numpy.zeros_like
# Return an array of zeros with the same shape and type as a given array.
def problem5 (A):
    return np.zero_like(A)

#return a vector with the same number of rows as A but that contains all ones.
def problem6 (A):
    return np.ones(A.shape[0])

def problem7 (A, alpha):
    return A + alpha * np.eye(A.shape[0])

def problem8 (A, i, j):
    return A[i,j]

def problem9 (A, i):
    return np.sum(A[i])

def problem10 (A, c, d):
    return np.mean(A[(A>=c) * (A<=d)])

def problem11 (A, k):
    return np.linalg.eig(A)[1][:, (A.shape - k):]

def problem12 (A, x):
    return np.linalg.solve(A,x)

def problem13 (A, x):
    return np.linalg.linalg.solve(A.T , x.T).T

