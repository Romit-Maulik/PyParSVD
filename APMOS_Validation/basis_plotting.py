import numpy as np
import matplotlib.pyplot as plt

# For visualization
mode_num = 0
modes_serial_mos = np.load('Serial_Modes_MOS.npy')
modes_serial_svd = np.load('Serial_Modes_SVD.npy')
modes_mos = np.load('APMOS_Basis_MOS.npy')
modes_svd = np.load('APMOS_Basis_SVD.npy')

plt.figure()
plt.plot(modes_serial_mos[:,mode_num],label='Serial MOS')
plt.plot(modes_mos[:,mode_num],'r--',label='APMOS MOS',markersize=2)
plt.title('Comparison')
plt.legend()
plt.show()

plt.figure()
plt.plot(modes_serial_svd[:,mode_num],label='Serial SVD')
plt.plot(modes_svd[:,mode_num],'r--',label='APMOS SVD',markersize=2)
plt.title('Comparison')
plt.legend()
plt.show()
