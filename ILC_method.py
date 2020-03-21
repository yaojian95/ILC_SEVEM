'''
ILC method to do the component separation.
Both ILC in harmonic space and pixel space are defined.

'''
import logging as log
import numpy as np
import matplotlib.pyplot as plt

### ILC in l-space

class ILC_L(object):
    
    def __init__(self, signal, noise = None, Nf, bins = None, lmax = None):
        
        '''
        ILC in spherical space to do the foreground removal.
        
        Parameters:
        
        singal : numpy.ndarray
        The total CROSS power-sepctrum matrix,
        with global size (N_modes, N_freq, N_freq).
        * N_freq: number of frequency bands
        * N_modes: number of angular modes
        '''
        
        self.signal = signal; self.noise = noise  
        
        
    # calculate the weight for each multipole bin
    
    def __call__(self, psbin, absbin):
#        log.debug('@ abspipe::__call__')
        return self.run(psbin, absbin)
    
    def run(self, psbin, absbin):
        
        '''
                ILC pipeline class call function.  
        '''
        
    
        W = np.matrix(np.zeros((Q, Nf)))  # Q = lmax/bins
        
        for l in range(Q):
            
            norm = e*np.linalg.pinv((total_bin[l]))*e.T
           
            W[l,:] = e*np.linalg.pinv((total_bin[l]))/norm   
        
        for i in range(Q):
                    
            noise_ilc[i] = W[i,:]*(noise_bin[i])*np.transpose(W[i,:])  
            Cl_ilc[n, i] = W[i,:]*(total_bin[i])*np.transpose(W[i,:]) - noise_ilc[i]
            
    def ILC_maps(self, signal):
        
        '''
        Apply the ILC weights in harmonic space to the alms to get the cleaned maps' alms,
        which are then transformed back to pixels.
        '''
        
        m_num = int((1 + lmax)*(lmax+1 -1)/2) # the number of alm of lmax=l for m >= 0; m = 0,1 for l = 1; m = 0,1,2 for l = 2.
        alm_Q = np.zeros((Nf, m_num), dtype = 'complex128'); alm_U = np.zeros((Nf, m_num), dtype = 'complex128')
        
        for i in range(Nf):
            alm_Q[i] = hp.map2alm(total_mask[i], lmax = lmax-1)[1]; # 95GHz, 150GHz, 353GHz
            alm_U[i] = hp.map2alm(total_mask[i], lmax = lmax-1)[2];   
            
        alm_Q_clean = np.zeros(m_num,dtype = 'complex128'); alm_U_clean = np.zeros(m_num, dtype = 'complex128')
        
        for l in np.arange(lmax):
            alm_Q_clean[m_l(lmax - 1, l)] = np.dot(np.array(W[l, :]),alm_Q[:,m_l(lmax - 1, l)])[0] #np.dot(np.array(weight[l, :]),alm_Q[:,m_l(lmax, l)])[0]#
            alm_U_clean[m_l(lmax - 1, l)] = np.dot(np.array(W[l, :]),alm_U[:,m_l(lmax - 1, l)])[0]
            
        alm_Q_clean[m_l(lmax, 0)] = 0; alm_Q_clean[m_l(lmax, 1)] = 0
        alm_U_clean[m_l(lmax, 0)] = 0; alm_U_clean[m_l(lmax, 1)] = 0    
        
        almT = np.zeros_like(alm_Q_clean)
        cmb_clean = hp.alm2map(np.row_stack((almT, alm_Q_clean, alm_U_clean)), nside = nside, lmax = lmax - 1)   
        
        return cmb_clean
            
    def plot_weight():
        Ell = binell()
        for i in range(Nf):
            plt.plot(Ell, W[:, i], label = '%s'%fre[i])
   
### ILC in pixel space         
class ILC_P(object):
    
    '''
    ILC in pixel space to do the foreground removal.
    '''
    
    def __init__(self, signal):
        self.signal = signal
        
    def __call__(self, psbin, absbin):
#        log.debug('@ abspipe::__call__')
        return self.run(psbin, absbin)
    
    def run(self, psbin, absbin): 
        
        total_Q = np.zeros((Nf, 12*nside**2)) ; total_U = np.zeros((Nf, 12*nside**2))## two frequencies
        noise_Q = np.zeros((Nf, 12*nside**2)); noise_U= np.zeros((Nf, 12*nside**2))
    
        for i in range(Nf):
            total_Q[i][mask_index] = total_mask[i][1][mask_index]
            total_U[i][mask_index] = total_mask[i][2][mask_index]
    
            # noise_ILC
            noise_Q[i][mask_index] = noise_mask[i][1][mask_index]
            noise_U[i][mask_index] = noise_mask[i][2][mask_index]
    
        Cov_Q = np.zeros((Nf, Nf)); w_Q = np.zeros(Nf)
        Cov_U = np.zeros((Nf, Nf)); w_U = np.zeros(Nf)
    
        for i in range(Nf):
            for j in range(Nf):
                Cov_Q[i, j] = np.dot(total_Q[i][mask_index] - np.mean(total_Q[i][mask_index]), total_Q[j][mask_index] - np.mean(total_Q[j][mask_index]))/1.0/len(mask_index)
                Cov_U[i, j] = np.dot(total_U[i][mask_index] - np.mean(total_U[i][mask_index]), total_U[j][mask_index] - np.mean(total_U[j][mask_index]))/1.0/len(mask_index)
    
        Cov_Q_inv = np.linalg.pinv(Cov_Q)
        Cov_U_inv = np.linalg.pinv(Cov_U)
    
    
        for i in range(Nf):
            w_Q[i] = np.sum(Cov_Q_inv[i,:])/np.sum(Cov_Q_inv)
            w_U[i] = np.sum(Cov_U_inv[i,:])/np.sum(Cov_U_inv)
    
    
        cmb_Q = np.dot(w_Q, total_Q); cmb_U = np.dot(w_U, total_U)
    
        cmb_I = np.zeros_like(cmb_Q);
        cmb_ILC_pix = np.row_stack((cmb_I, cmb_Q, cmb_U))
    
        noise_ilc_q = np.dot(w_Q, noise_Q); noise_ilc_u = np.dot(w_U, noise_U)
        noise_I = np.zeros_like(noise_ilc_q)
        noise_ilc_pix = np.row_stack((noise_I, noise_ilc_q, noise_ilc_u))
        
        cls_ILC_pix = hp.anafast(cmb_ILC_pix, lmax = lmax, nspec = 3)
        nl_ilc_pix = hp.anafast(noise_ilc_pix, lmax = lmax, nspec = 3)
        
        Cl_ilc_rs[0,n] = bin_l(cls_ILC_pix[1], lmax, Q); Cl_ilc_rs[1,n]  = bin_l(cls_ILC_pix[2], lmax, Q)
        Nl_ilc_rs[0,n] = bin_l(nl_ilc_pix[1], lmax, Q); Nl_ilc_rs[1,n] = bin_l(nl_ilc_pix[2], lmax, Q)        
        

