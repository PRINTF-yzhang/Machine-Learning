# author Ying Zhang & Yujia Qiu
# CS 453X
# Professor Jacob Whitehill
# Part 2: Step-wise Classification
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

#this takes in a vector of ground-truth labels and corresponding vector of guesses, and
#then computes the accuracy (PC)
def fPC (y, yhat):
    PC = np.float(sum(y == yhat))/len(y)
    return PC
#  this takes in a set of predictors, a set of images
# to run it on, as well as the ground-truth labels of that set.
# returns the accuracy (PC) of the predictions w.r.t. the ground-truth labels.
# X: images
# y: ground truth
# predictors [r1,c1,r2,c2]
def measureAccuracyOfPredictors (predictors, X, y):
    #r1 = predictors[0]
    #c1 = predictors[1]
    #r2 = predictors[2]
    #c2 = predictors[3]

    # setall to zeros
    #init = np.zeros_like(y)

    #for predictor in predictors:
    #    r1,c1,r2,c2 = predictor
    #    isSmile = X[:,r1,c1] - X[:,r2,c2]
    #yhat = np.divide(init, 5)
    features = []
    half = 0.5
    #fomula from slides
    for i in range(0,len(predictors)):
        r1,c1,r2,c2 = predictors[i]
        features.append(X[:,r1,c1] > X[:,r2,c2])
    yhat = np.sum(features,axis =0)/len(predictors) > half

    return fPC(y,yhat)


#smiling (1) or not (0).
#greed algrithm
def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    predictors = []
    # in range of 5
    for i in range(0,5):
        accuracy = 0
        #imgae size 24x24
        for r1 in range(0,24):
            for c1 in range(0,24):
                for r2 in range(0,24):
                    for c2 in range(0,24):
                        if not (r1==r2 and c1==c2):
                            predictors.append([r1, c1, r2, c2])
                            current_accuracy = measureAccuracyOfPredictors (predictors, trainingFaces, trainingLabels)
                            if current_accuracy > accuracy:
                                accuracy = current_accuracy
                                current_predictor = [r1,c1,r2,c2]
                            # Source https://www.quora.com/What-is-the-difference-between-a-pop-and-del-x-y-in-Python
                            predictors.pop()
        predictors.append(current_predictor)
    Testing_accuracy = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)

    print('result: ')
    print('Training accuracy: ', accuracy)
    print('Testing accuracy: ', Testing_accuracy)

    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        #for predictor in predictors:
        for i in range(0,len(predictors)):
            predictor = predictors[i]
            rect = patches.Rectangle((predictor[1],predictor[0]),1,1,linewidth=2,edgecolor='y',facecolor='none')
            ax.add_patch(rect)
        # Show r2,c2
            rect = patches.Rectangle((predictor[3],predictor[2]),1,1,linewidth=2,edgecolor='g',facecolor='none')
            ax.add_patch(rect)
        # Display the merged result
        plt.show()

# analyze how training/testing accuracy changes as a function of number of examples
def trainSize(trainingFaces, trainingLabels, testingFaces, testingLabels):
    examples = [400, 800, 1200, 1600, 2000]
    for example in examples:
        print(" trained classifier on the %d examples" % example)
        stepwiseRegression(trainingFaces[:example], trainingLabels[:example], testingFaces, testingLabels)
        print()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
