#!/usr/bin/env python
# coding: utf-8

# In[1]:


#linear algebra with Numpy:
#Numpy provides the following functions to perform the different algebraic calculations on the input data.

#Function->Definition
#dot()->It is used to calculate the dot product of two arrays.
#vdot()->It is used to calculate the dot product of two vectors.
#inner()->It is used to calculate the inner product of two arrays.
#matmul()->It is used to calculate the matrix multiplication of two arrays.
#det()->It is used to calculate the determinant of a matrix.
#solve()->It is used to solve the linear matrix equation.
#inv()->It is used to calculate the multiplicative inverse of the matrix.


# In[2]:


import numpy as np  
a = np.array([[100,200],[23,12]])  
b = np.array([[10,20],[12,21]])  
dot = np.dot(a,b)  
print(dot)  


# In[3]:


#The dot product is calculated as:

#[100 * 10 + 200 * 12, 100 * 20 + 200 * 21] [23*10+12*12, 23*20 + 12*21]


# In[4]:


import numpy as np  
a = np.array([[100,200],[23,12]])  
b = np.array([[10,20],[12,21]])  
vdot = np.vdot(a,b)  
print(vdot) 


# In[5]:


#vdot is calculated as
#np.vdot(a,b) = 100 *10 + 200 * 20 + 23 * 12 + 12 * 21 = 5528


# In[6]:


#numpy.inner() function
import numpy as np  
a = np.array([1,2,3,4,5,6])  
b = np.array([23,23,12,2,1,2])  
inner = np.inner(a,b)  
print(inner)  


# In[7]:


#numpy.matmul() function
#It is used to return the multiplication of the two matrices.
#It gives an error if the shape of both matrices is not aligned for multiplication
import numpy as np  
a = np.array([[1,2,3],[4,5,6],[7,8,9]])  
b = np.array([[23,23,12],[2,1,2],[7,8,9]])  
mul = np.matmul(a,b)  
print(mul)  


# In[8]:


#numpy determinant 
#The numpy.linalg.det() function is used to calculate the determinant of the matrix
import numpy as np  
a = np.array([[1,2],[3,4]])  
print(np.linalg.det(a))  


# In[9]:


#numpy.linalg.solve() function
#This function is used to solve a quadratic equation where values can be given in the form of the matrix.
import numpy as np  
a = np.array([[1,2],[3,4]])  
b = np.array([[1,2],[3,4]])  
print(np.linalg.solve(a, b))  


# In[10]:


#Vectorized Operation in Numpy:
#The concept of vectorized operations on NumPy allows the use of more optimal and pre-compiled functions and mathematical operations on NumPy array objects and data sequences.
#The Output and Operations will speed-up when compared to simple non-vectorized operations.


# In[11]:


#Universal function
#Universal functions in Numpy are simple mathematical functions. It is just a term that we gave to mathematical functions in the Numpy library.
#Numpy provides various universal functions that cover a wide variety of operations.

#These functions operates on ndarray (N-dimensional array) i.e Numpy’s array class.
#>It performs fast element-wise array operations.
#>It supports various features like array broadcasting, type casting etc.
#>Numpy, universal functions are objects those belongs to numpy.ufunc class.
import numpy as np

arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.add(arr1, arr2)

print(newarr)


# In[12]:


import numpy as np

arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.subtract(arr1, arr2)

print(newarr)


# In[13]:


import numpy as np

arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.multiply(arr1, arr2)

print(newarr)


# In[14]:


import numpy as np

arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 5, 10, 8, 2, 33])

newarr = np.divide(arr1, arr2)

print(newarr)


# In[15]:


import numpy as np

arr = np.array([-1, -2, 1, 2, 3, -4])

newarr = np.absolute(arr)

print(newarr)


# In[16]:


import numpy as np

arr = np.around(3.1666, 2)

print(arr)


# In[17]:


import numpy as np

arr = np.floor([-3.1666, 3.6667])

print(arr)


# In[18]:


import numpy as np

num1 = 4
num2 = 6

x = np.lcm(num1, num2)

print(x)


# In[1]:


#Broadcasting and shape manipulation:
#The term broadcasting refers to how numpy treats arrays with different Dimension during arithmetic operations which lead to certain constraints, 
#the smaller array is broadcast across the larger array so that they have compatible shapes. 
import numpy as np
a = np.array([17, 11, 19]) 
print(a)
b = 3 
print(b)
 
# Broadcasting happened because of
# miss match in array Dimension.
c = a + b
print(c)


# In[2]:


#NumPy arrays have an attribute called shape that returns a tuple with each index having the number of corresponding elements.
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(2, 3, 2)

print(newarr)#1-D array with 12 elements into a 3-D array.


# In[4]:


import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

print(newarr)


# In[5]:


#boolean mask:
#Masking comes up when you want to extract, modify, count, or otherwise manipulate values in an array based on some criterion
x = np.array([1, 2, 3, 4, 5])
x < 3


# In[6]:


x > 3


# In[7]:


x <= 3


# In[8]:


x >= 3 


# In[9]:


x != 3


# In[10]:


x == 3 


# In[16]:


#Dates and time in numpy:
import numpy as np
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print("Yestraday: ",yesterday)
today     = np.datetime64('today', 'D')
print("Today: ",today)
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print("Tomorrow: ",tomorrow)
  


# In[15]:


# to find the first Monday in May 2017.
import numpy as np
print("First Monday in May 2017:")
print(np.busday_offset('2017-05', 0, roll='forward', weekmask='Mon'))
  


# In[ ]:




