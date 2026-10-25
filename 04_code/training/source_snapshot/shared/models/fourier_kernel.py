"""Normalized tensor-product Fourier kernel for bounded RY encoding frequencies."""
import numpy as np

def fourier_kernel(a,b,order=1):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape[1]!=b.shape[1] or order<1:raise ValueError('Invalid feature dimension/order')
    result=np.ones((len(a),len(b)),dtype=np.float64)
    for j in range(a.shape[1]):
        delta=a[:,j,None]-b[None,:,j]
        factor=np.ones_like(delta)
        for r in range(1,order+1):factor+=np.cos(r*delta)
        result*=factor/(order+1)
    return result

def explicit_features(a,order=1):
    if (2*order+1)**a.shape[1]>100000:raise ValueError('Use the implicit kernel; explicit expansion too large')
    result=np.ones((len(a),1))
    for j in range(a.shape[1]):
        factors=np.column_stack([np.ones(len(a))]+[fun(r*a[:,j]) for r in range(1,order+1) for fun in [np.sin,np.cos]])/np.sqrt(order+1)
        result=(result[:,:,None]*factors[:,None,:]).reshape(len(a),-1)
    return result
