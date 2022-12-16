import numpy as np
import matplotlib.pyplot as plt

a = np.arange(2,100000000)
log_v = 1/np.log(a)
a_1 = np.cumsum(log_v)
b_1 = a*log_v


plt.plot(a_1/b_1)
plt.show()