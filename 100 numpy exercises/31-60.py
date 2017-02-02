import numpy as np

# 31. Create a vector of size 3 with values ranging from 0 to 1, both excluded
Z = np.linspace(0,1,5,endpoint=True)[1:-1]
print(Z)

# 32. Create a random vector of size 10 and sort it
sc = np.random.random(10)
sc[::-1].sort() # descending
print(sc)

# 34. Consider two random array A anb B, check if they are equal
A = np.random.randint(0,2,3)
B = np.random.randint(0,2,3)
equal = np.allclose(A,B)
print(A,B, equal)

# 36. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates
cart = np.random.random((10,2))
x, y = cart[:,0], cart[:,1]
r = np.sqrt(x**2+y**2)
angle = np.arctan2(y, x)
print(r, angle)

# 37. Create random vector of size 10 and replace the maximum value by 0
arr = np.random.random(10)
arr[arr.argmax()] = 0
print(arr.argmax(), arr.max(), arr)

# 38. Create an array with x and y coordinates covering the [0,1]x[0,1] area
Z = np.zeros((4,4), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,4),
                             np.linspace(0,1,4))
print(Z)

# 39. construct the Cauchy matrix C (Cij = 1/(xi - yj))
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

# 42. How to find the closest value in an array?
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(v, Z[index])
