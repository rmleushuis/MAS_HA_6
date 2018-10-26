import numpy as np
from matplotlib import pyplot as plt

# SUBQUESTION 1
estimate = []
N = list(range(1,10001))
for n in N:
    x = np.random.uniform(-5,5,n)
    p = 1/np.sqrt(2*np.pi)*np.exp(-1/2*x**2)
    q = 1/10

    estimate.append(1/n*(x**2*(p/q)).sum())

plt.plot(N,estimate)
plt.ylabel('estimate of X**2')
plt.xlabel('Number of samples')
plt.show()


# SUBQUESTION 2
N = 10000
number_test = 100
estimate_mat = np.zeros((number_test,N))

# for testnr in list(range(0,number_test)):
estimate = []
for n in list(range(1,N+1)):
    x = np.random.uniform(-1,1,n)
    p = (1+np.cos(np.pi*x))/2
    q = 1/2
    estimate.append(1/n*(x**2*(p/q)).sum())
    # print(testnr)
    # estimate_mat[testnr] = estimate

plt.plot(list(range(N)),estimate)
plt.ylabel('estimate of X**2')
plt.xlabel('Number of samples')
plt.show()
#
