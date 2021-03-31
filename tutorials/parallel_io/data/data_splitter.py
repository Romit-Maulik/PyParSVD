import numpy as np
import matplotlib.pyplot as plt
import h5py

Rnum = 1000.0
x = np.linspace(0.0, 1.0, num=2*8192)
dx = 1.0 / np.shape(x)[0]

ntsteps = 800
tsteps = np.linspace(0.0, 2.0, num=ntsteps)
dt = 2.0 / np.shape(tsteps)[0]

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# This is the Burgers problem definition
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def exact_solution(t):
    t0 = np.exp(Rnum / 8.0)
    return (x / (t+1)) / (1.0 + np.sqrt((t + 1) / t0) * np.exp(Rnum * (x * x) / (4.0 * t + 4)))

def collect_snapshots():
    snapshot_matrix = np.zeros(shape=(np.shape(x)[0], np.shape(tsteps)[0]))
    trange = np.arange(np.shape(tsteps)[0])
    for t in trange:
        snapshot_matrix[:,t] = exact_solution(tsteps[t])[:]
    return snapshot_matrix

# Method of snapshots to accelerate
def method_of_snapshots(Y): #Mean removed
    '''
    Y - Snapshot matrix - shape: NxS
    '''
    new_mat = np.matmul(np.transpose(Y), Y)
    w, v = np.linalg.eig(new_mat)

    # Bases
    phi = np.matmul(Y, np.real(v))
    trange = np.arange(np.shape(Y)[1])
    phi[:,trange] = phi[:,trange] / np.sqrt(np.abs(w)[:])
    return phi, np.sqrt(np.abs(w)[:]) # POD modes


# Collect data
total_data = collect_snapshots()
num_snapshots = total_data.shape[1]
num_dof = total_data.shape[0]

# Generate serial POD with MOS
from time import time
# Initialize timer
start_time = time()
modes, svals = method_of_snapshots(total_data)
np.save('Serial_Modes_MOS.npy', modes)
np.save('Serial_SingularValues.npy', svals)
end_time = time()
print('Time required for serial SVD using method of snapshots', end_time-start_time)
total_ranks = 6
npr = int(num_dof/total_ranks)

# Save data
niters = 4
batches = int(ntsteps / niters)
for iteration in range(niters):
    batch_data = total_data[:,iteration*batches:(iteration+1)*batches]

    h5f = h5py.File('Batch_'+str(iteration)+'_data.h5', 'w')
    h5f.create_dataset('dataset', data=batch_data)
    h5f.close()