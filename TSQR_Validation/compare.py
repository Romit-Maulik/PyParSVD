import numpy as np
import matplotlib.pyplot as plt

qs = np.load('Q_Serial.npy')
rs = np.load('R_Serial.npy')

qp = np.load('Q_Parallel.npy')
rp = np.load('R_Parallel.npy')

plt.figure()
plt.plot(qs[:240,0],label='Q_Serial')
plt.plot(-qp[:240,0],label='Q_Parallel')
plt.legend()

plt.figure()
plt.plot(rs[1:,10],label='R_Serial')
plt.plot(-rp[1:,10],label='R_Parallel')
plt.legend()
plt.show()