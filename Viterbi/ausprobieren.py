import numpy as np

"""a = np.zeros((3,2))
print(a)
a[0] = 1,2
print(a)
probs_states = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
print(len(probs_states))


V = np.ones((4, 3, 2)) * (-1)
print(V)
V[0][0][0] = 1
print("\n \n", V)

P = np.array([[0.3, 0.7], [0.2, 0.8], [0.5, 0.5]])
print(np.shape(P))
print(((P[np.newaxis, 1]).T))

for i in range(5):
    print(i)

P = np.array([0, 1, 2, 3])
print(P[1])"""

"""A = np.array([[1,2, 3],
     [3,4, 5]])
print(A)
row_sums = A.sum(axis=1)
normalized_A = A / row_sums[:, np.newaxis]
print(normalized_A)
#divided = A / sum
#print(divided)"""
A = [[1,2, 3],
     [3,4, 5]]

A.extend([[6, 7, 8], [9, 10, 11]])
print(A)