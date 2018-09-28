
# Linear Algebra


```python
import numpy as np
import matplotlib as plot
```

## Create a vector or Matrix


```python
#1-dimentional Array = a vector
x = np.array([1, 2, 3, 4])
print(x)
print(type(x))
print(x.shape)
x
```

    [1 2 3 4]
    <class 'numpy.ndarray'>
    (4,)





    array([1, 2, 3, 4])




```python
# The array() function can also create 2-dimensional arrays
#3 * 2 matrix
x2 = np.array([[1, 2], [3, 4], [5, 6]])
print(x2)
print(type(x2))
print(x2.shape)
x2
```

    [[1 2]
     [3 4]
     [5 6]]
    <class 'numpy.ndarray'>
    (3, 2)





    array([[1, 2],
           [3, 4],
           [5, 6]])



## Transpose


```python
x_tanspose = x.T
x_tanspose
```




    array([1, 2, 3, 4])




```python
x2
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
x2_transpose = x2.T
x2_transpose
```




    array([[1, 3, 5],
           [2, 4, 6]])




```python
print(x2.shape)
print(x2_transpose.shape)
```

    (3, 2)
    (2, 3)


## Matrices can be added if they have the same shape


```python
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[2, 5], [7, 4], [4, 3]])

C = A + B
```


```python
A
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
B
```




    array([[2, 5],
           [7, 4],
           [4, 3]])




```python
C
```




    array([[ 3,  7],
           [10,  8],
           [ 9,  9]])



## Add a scalar to a matrix


```python
C = A + 4
```


```python
A
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
C
```




    array([[ 5,  6],
           [ 7,  8],
           [ 9, 10]])



## Add two matrices of different shapes


```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
B = np.array([1,2,3])
B
```




    array([1, 2, 3])




```python
C = A + B
C
```




    array([[ 3,  7],
           [10,  8],
           [ 9,  9]])



## Matrix Mutiplication


```python
A = np.array([[1,2,4],[5,6,5],[8,6,8],[9,87,12]])
A
```




    array([[ 1,  2,  4],
           [ 5,  6,  5],
           [ 8,  6,  8],
           [ 9, 87, 12]])




```python
B = np.array([[2, 7], [1, 2], [3, 6]])
B
```




    array([[2, 7],
           [1, 2],
           [3, 6]])




```python
C = A.dot(B) # dot() product of matrix or vector
```


```python
C
```




    array([[ 16,  35],
           [ 31,  77],
           [ 46, 116],
           [141, 309]])



## Identity


```python
I = np.eye(4)
```


```python
I
```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])



## Inverse


```python
A = np.array([[3, 0, 2], [2, 0, -2], [0, 1, 1]])
A
```




    array([[ 3,  0,  2],
           [ 2,  0, -2],
           [ 0,  1,  1]])




```python
A_inv = np.linalg.inv(A)
A_inv
```




    array([[ 0.2,  0.2,  0. ],
           [-0.2,  0.3,  1. ],
           [ 0.2, -0.3, -0. ]])


