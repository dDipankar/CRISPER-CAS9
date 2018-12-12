
# libraries
import numpy as np
import matplotlib.pyplot as plt

x =np.array([10000,20000,30000,40000,44262])
y = np.array([0.39, 0.43, 0.4492, 0.466, 0.465])
plt.plot(x, y)        # plot x and y using default line style and color
plt.plot(x, y, 'bo')  # plot x and y using blue circle markers
#plt.plot(y)           # plot y using x as index array 0..N-1
#plt.plot(y, 'r+')     # ditto, but with red plusses
plt.title("Pearson vs Sample Size")
plt.show()
