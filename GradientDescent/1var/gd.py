import math
import numpy as np
import matplotlib.pyplot as plt

# Loss function example: y = f(x) = x^2 + 5sin(x)
def loss(x):
    return x**2 + 5*math.sin(x)

# Gradient of y = f(x) = x^2 + 5sin(x) => y' = f'(x) = 2x + 5cos(x)
def gradient(x):
    return 2*x + 5*math.cos(x)

# The Gradient Descent with Learning Rate (lr) and Starting Point (sp)
def gradDesc(lr, sp):
    pointSet = [sp]
    for ite in range(100):
        nextPoint = pointSet[-1] - lr*gradient(pointSet[-1])
        pointSet.append(nextPoint)
        if (abs(gradient(nextPoint)) < 1e-3):
            break
    
    print("The local minimum is " + str(pointSet[-1]) + " with Loss = " + str(loss(pointSet[-1])) + " throughout " + str(ite) + " iterations")
    return (pointSet, ite)

gradDesc(0.1, 5)
gradDesc(0.1, -5)
gradDesc(0.5, 5)
gradDesc(0.5, -5)
