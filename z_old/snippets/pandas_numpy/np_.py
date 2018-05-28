# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

# 3. create 10-zero vector
zr = np.zeros(10)
print(zr)

# 4. How to get the documentation of the pandas_numpy add function from the command line?
# np.info(np.add)

# 5. Create a null vector of size 10 but the fifth value which is 1
s = np.zeros(10)
s[4] = 1
print(s)

# 6. Create a vector with values ranging from 10 to 49
r = np.arange(10,50)
print(r)

# 7. Reverse a vector (first element becomes last)
r = np.arange(10,50)
r = r[::-1]
print(r)

# 8. Create a 3x3 matrix with values ranging from 0 to 8
m = np.arange(9).reshape(3,3)
print(m)

# 9. Find indices of non-zero elements from [1,2,0,0,4,0]
zi = np.nonzero([1,2,0,0,4,0])
print(zi)

# 10. Create a 3x3 identity matrix
e = np.eye(3,3)
print(e)

# 11. Create a 3x3x3 array with random values
rn = np.random.random((3,3,3))
print(rn)

# 12. Create a 10x10 array with random values and find the minimum and maximum values
rn = np.random.random((10,10))
print(rn.max(), rn.min())

# 13. Create a random vector of size 30 and find the mean value
rs = np.random.random(30)
print(rs.mean())

# 14. Create a 2d array with 1 on the border and 0 inside
a2 = np.ones((5,5))
a2[1:-1,1:-1] = 0
print(a2)

# 15. What is the result of the following expression?
print(
0 * np.nan,
np.nan == np.nan,
np.inf > np.nan,
np.nan - np.nan,
0.3 == 3 * 0.1,
3 * 0.1 )

# 16. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
de = np.diag(1 + np.arange(4), -1)
print(de)

# 17. Create a 8x8 matrix and fill it with a checkerboard pattern
m = np.zeros((8,8))
m[::2,::2] = 1
m[1::2,1::2] = 1
# m[1::2,::2] = 1
# m[::2,1::2] = 1
print(m)

# 18. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
m = np.zeros((6,7,8))
print(np.unravel_index(100,m.shape))

# 19. Create a checkerboard 8x8 matrix using the tile function
tile = np.tile(np.array([[0,1],[1,0]]), (4,4))
print(tile, tile.shape)

# 20. Normalize a 5x5 random matrix
Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)

# 22. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
pr = np.dot(np.ones((5,3)), np.ones((3,2)))
print(pr)

# 23. Given a 1D array, negate all elements which are between 3 and 8, in place
sc = np.arange(10)
sc[(sc > 3) & (sc < 8)] *= -1
print(sc)

# 24.
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1)) # -1 -> axis

# 25.
Z = np.arange(5)
print(Z,
Z**Z,
3 << Z,
Z <- Z,
1j*Z,
Z/1/1)

# 26.
print(np.array([3,5,6]) // np.array(2)) # floor divide
# print(np.array(0) // np.array(0)) # zero

# 27. How to round a float array?
Z = np.random.uniform(-10,+10,10)
print (np.trunc(Z + np.copysign(0.5, Z)))

# 28. Extract the integer part of a random array using 5 different methods
arr = np.arange(3,step=0.25)
print(arr)
print(np.trunc(arr))
print (arr - arr%1)
print (np.floor(arr))
print (arr.astype(int))

# 29. Create a 5x5 matrix with row values ranging from 0 to 4
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)

# 30. Consider a generator function that generates 10 integers and use it to build an array
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float)
print(Z)


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

# 43. Create a structured array representing a position (x,y) and a color (r,g,b)
pixel = np.zeros(10,
               [('position',
                   [('x',float), ('y', float)]
                 ),
                ('color',
                   [('r', int), ('g', int), ('b', int)]
                 )
               ])
print(pixel)

# 44 Consider a random vector with shape (100,2) representing coordinates, find point by point distances
import scipy
import scipy.spatial
Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print("-"*25, " 44 " ,"-"*25, "\n",D)

# 45 convert a float array into an integer in place
arr = np.arange(10, dtype=np.float32)
arr = arr.astype(dtype=np.int32, copy=False)
print(arr)

# 46 open csv with header
my_data = np.genfromtxt('file.csv', delimiter=',', names=True)
print(my_data.dtype.names, my_data)

# 47 What is the equivalent of enumerate for pandas_numpy arrays?
arr = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(arr):
    print(index, value)
for index in np.ndindex(arr.shape):
    print(index, arr[index])

# 49 How to randomly place p elements in a 2D array?
n = 5
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False), 2)
print(Z)

# 50 Subtract the mean of each row of a matrix
X = np.random.rand(5,10)
X_mean = X.mean(axis=1, keepdims=True)
print("-"*25, " 50 " ,"-"*25)
print(X, X_mean, X.shape, sep='\n')

# 51 Sort an array by column
Z = np.random.rand(10,2)
Z = Z[Z[:,1].argsort()]
print(Z)

# 52 check zero column
with_zero = np.random.randint(0,3,(3,10))
print(with_zero, with_zero.any(axis=0))

# 55 56 59 bincount
# I = list(set(I))
# Z[I] += 1
Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z)) # bincount count values in I and convert it to indexes
print(Z, np.bincount(I, minlength=len(Z)), I)

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X) # using X value and I indexes to fill new array (1 and 6 in index 1 give sum)
print(F)
print(with_zero[...,1], with_zero[:,1])

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S,D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_sums, D_counts, D_means)

# 58 Considering a four dimensions array, how to get sum over the last two axis at once?
A = np.random.randint(0,10,(3,4,3,2))
Ar = A.reshape(A.shape[:-2] + (-1,))
sum = Ar.sum(axis=-1)
print(sum)

# 60 How to get the diagonal of a dot product N x N matrix
A = np.random.randint(1,10,9).reshape(3,3)
B = np.random.randint(1,10,9).reshape(3,3)
# A = np.array([4,9,3,9,9,8,2,4,6]).reshape(3,3)
# B = np.array([7,1,1,5,5,4]).reshape(3,2)

# Slow version
print(np.diag(np.dot(A, B)))
# Fast version
print(np.sum(A * B.T, axis=1))
# Faster version
print(np.einsum("ij,ji->i", A, B))

