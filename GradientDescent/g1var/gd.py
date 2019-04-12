import numpy as np 
np.random.seed(2)

# Generate Data set (X, y)
X = np.random.rand(1000, 1)
y = 4 + 3*X + 0.2*np.random.randn(1000, 1) # noise added

# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Loss function genarationally
def loss(w):
    N = Xbar.shape[0]
    return 0.5/N * np.linalg.norm(y - Xbar.dot(w), 2)**2

# Gradient genarationally of Loss function
def gradient(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

# The Gradient Descent with Learning Rate (lr) and Startting Point (sp)
def gradDesc(lr, sp):
    pointSet = [sp]
    for ite in range(100):
        nextPoint = pointSet[-1] - lr*gradient(pointSet[-1])
        pointSet.append(nextPoint)
        print(pointSet[-1])
        if(np.linalg.norm(gradient(nextPoint)) < 1e-3):
            break

    print("The local minimum is w = ", pointSet[-1].T, " after " + str(ite) + " iterations")

sp = np.array([[2], [1]])
gradDesc(1, sp)
