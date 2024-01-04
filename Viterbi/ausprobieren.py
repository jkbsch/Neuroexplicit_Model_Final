import numpy as np
#import torch
#import matplotlib.pyplot as plt

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
"""A = [[1,2, 3],
     [3,4, 5]]

A.append([[6, 7, 8], [9, 10, 11]])
#print(A)

A = [1, 2, 3, 4, 5, 6]
print(A[0:2])
print(A[2:4])"""
"""A = np.array([1, 2, 3])
B = np.array([1, 3, 2])
print(np.sum(A==B))"""
"""for i in range(1,10):
    print(i)"""
"""A = torch.tensor([1], dtype = torch.int64)
print(torch.is_tensor(A))"""

"""A = np.array([[1, 2, 3],[4, 5, 6]])
B = torch.from_numpy(A)
print(A.shape[0])
print(A)
print(type(B.shape[0]))
print(len(B))"""

"""A = torch.tensor([[2, 2, 2],[4,4,4]], dtype=torch.float64)
B = torch.tensor([[3, 3, 3], [5, 5, 5]])
print(A)
print(B)
C = [A]
C.append(B)
print(C)"""
"""for i in range(1):
    print(i)"""
"""y_true = [0, 0, 0, 1, 2, 3]
y_pred = [0, 0, 1, 1, 2, 2]
length = len(y_pred)
X = np.arange(length)
fig, ax = plt.subplots()

ax.step(X, y_true)
ax.step(X, y_pred)
plt.show()"""

"""for a in [True, False]:
    print(a)"""
"""C = []
A = [1.0, 2.0]
A.append(3.0)
B = [4.0, 5.0, 6.0]
C.append(A)
C.append(B)
print(C)"""


nr_same_epochs = 7
A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
# B = np.array([8, 9, 10, 11, 12, 13, 14, 15])
i = 0

half = int(nr_same_epochs/2)
arr1 = A[i:half]
arr2 = A[i+half+1:i+nr_same_epochs]
element = A[half]

print(arr1, arr2)


