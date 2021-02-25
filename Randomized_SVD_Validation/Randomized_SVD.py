import numpy as np
import matplotlib.pyplot
from time import time


M = 500
N = 400

A = np.random.normal(size=(M,N))

start_time = time()
u,s,v = np.linalg.svd(A)
end_time = time()

print('Time for regular SVD:',end_time-start_time)

# Want to get the first k singular vectors
k = 100

start_time = time()
omega = np.random.normal(size=(N,2*k))
omega_pm = np.matmul(A,np.transpose(A))
Y = np.matmul(omega_pm,np.matmul(A,omega))

Q, R = np.linalg.qr(Y)

B = np.matmul(np.transpose(Q),A)
u_r, s_r, v_r = np.linalg.svd(B,full_matrices=False)

u_r = np.matmul(Q,u_r)

end_time = time()
print('Time for randomized SVD:',end_time-start_time)


# Print shapes
print(u.shape,s.shape,v.shape)
print(u_r.shape,s_r.shape,v_r.shape)

# Reconstruction
cS = s_r*np.identity(2*k)
print(cS.shape)
Ahat = np.matmul(u_r,np.matmul(cS,v_r))

# Find error between A and Ahat
print(np.linalg.norm(A-Ahat))

