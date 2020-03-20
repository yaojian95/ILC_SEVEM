'''
ILC separator class.

'''

import numpy as np


## noise_bin, total_bin, Nf, 

W = np.matrix(np.zeros((Q, Nf)))

for l in range(Q):
    norm = e*np.linalg.pinv((total_bin[l]))*e.T
   
    W[l,:] = e*np.linalg.pinv((total_bin[l]))/norm   

for i in range(Q):
    
    ''' whether include noise_bin or not;  WNW = 1/(eN{-1}e.T)'''
    
    noise_ilc[i] = W[i,:]*(noise_bin[i])*np.transpose(W[i,:])  
    Cl_ilc[n, i] = W[i,:]*(total_bin[i])*np.transpose(W[i,:]) - noise_ilc[i]
    
for i in range(Nf):
    plt.plot(Ell, W[:, i], label = '%s'%fre[i])