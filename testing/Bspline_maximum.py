# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:25:54 2023

@author: s146407
"""

## Find maximum parametric value of B-spline function
import splipy as sp
import numpy as np
import matplotlib.pyplot as plt
_ = np.newaxis
## Construct basis
#  create a set of cubic (order=4) B-spline basis functions
basis = sp.BSplineBasis(order=4, knots=[0,0,0,0,1,2,4,4,4,4])
# create a list of 6 controlpoints (we have 6 basis functions in our basis)
controlpoints = np.array([[  0,  0],
                          [1.8,0.2],
                          [  1,  1],
                          [0.2,0.2],
                          [1.5, 0.1],
                          [  2,  0]])

# basis = sp.BSplineBasis(order=3, knots=[0,0,0,1,1,1])
# # create a list of 6 controlpoints (we have 6 basis functions in our basis)
# controlpoints = np.array([[  0,  0],
#                           [1.8,0.8],
#                           [  1,  1]])

#basis = sp.BSplineBasis(order=3, knots=[0,0,0,1,2,3,3,3])
basis.reparam(0,1)

# 150 uniformly spaced evaluation points on the domain (0,4)
t = np.linspace(0,1,150)

# evaluate *all* basis functions on *all* points t. The returned variable B is a matrix
B = basis.evaluate(t) # B.shape = (150,6), 150 visualization points, 6 basis functions


def gss(f, a, b, bfunc, tol=1e-9):
    """Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678

    """
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if f(c)[0][bfunc] > f(d)[0][bfunc]: # to find the maximum
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2



maxima = [0]
for Bfunc in range(1,basis.num_functions()-1):
    maxima += [gss(basis, 0, 1, bfunc=Bfunc)]
maxima += [1]
maxima = basis.greville()

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(t, B, linestyle='-')
for maximum in maxima:
    ax.axvline(maximum, linestyle='--', color='black')
plt.show()





curve = sp.Curve(basis, controlpoints)

x = curve.evaluate(t)    # compute (x,y)-coordinates of the curve evaluation

# maxima = [0]
# for Bfunc in range(1,basis.num_functions()-1):
#     maxima += [t[np.argmin( np.linalg.norm(x - curve.controlpoints[Bfunc], axis=1) )]]
# maxima += [1]


tangent = curve.tangent(maxima)
normal  = tangent[...,::-1]*np.array([-1,1])
normal /= np.linalg.norm(normal)

x_normal = curve.evaluate(maxima)
n_lines  = np.concatenate([x_normal[...,_], controlpoints[...,_]],axis=2)

 
fig, ax = plt.subplots(figsize=(8, 8))
plt.plot(x[:,0], x[:,1])
plt.plot(curve.controlpoints[:,0], curve.controlpoints[:,1], 'rs-')
for nline in n_lines:
    plt.plot(nline.T[:,0], nline.T[:,1], color='grey')
plt.quiver(x_normal[:,0], x_normal[:,1], normal[:,0], normal[:,1])
plt.gca().set_aspect('equal')
plt.legend(('curve', 'control points'))
plt.show()


