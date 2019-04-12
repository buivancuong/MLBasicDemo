import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(2)

# Generate Data set (X, y)
X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)

# Display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0
print("w0 = " + str(w_0) + "; w1 = " + str(w_1) + " => y = " + str(w_0) + " + " + str(w_1) + "x")

# Draw the fitting line 
plt.plot(X.T, y.T, 'b.')     # show Data set
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()
