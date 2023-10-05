import numpy as np

"""a = np.zeros((3,2))
print(a)
a[0] = 1,2
print(a)
probs_states = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
print(len(probs_states))"""


V = np.ones((4, 3, 2)) * (-1)
print(V)
V[0][0][0] = 1
print("\n \n", V)