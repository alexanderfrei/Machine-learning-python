# http://www.labri.fr/perso/nrougier/teaching/numpy.100/
# 1. import
import numpy as np

# 2. get version
print(np.__version__)

# 3. create 10-zero vector
zr = np.zeros(10)
print(zr)

# 4. How to get the documentation of the numpy add function from the command line?
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


