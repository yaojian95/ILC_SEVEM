import healpy as hp
import numpy as np
from numpy import linalg as LA
import pymaster as nmt

class ABS(object):
    
    def __init__(self, nside, lmax, bin_w, beam = None):
        
        self.nside = nside; self.lmax = lmax;

        self.b = nmt.NmtBin(self.nside, nlb=bin_w, lmax=self.lmax)
    
        self.ell_n = self.b.get_effective_ells(); self.lbin = len(self.ell_n); 
        
    
    def Select_fre(self, ps_in, sel):
        
        '''
        Take some part of the cross power spectrum matrix
        
        '''
        # sel = np.array((1,2,3))
        n_fre = len(sel)
        ps_out = np.ones((self.lbin, n_fre, n_fre)); ### selected power spectra
        
        for q in range(self.lbin):
            x = 0; 
            for i in (sel):
                y = 0;
                for j in (sel):
                    ps_out[q][x,y] = ps_in[q][i, j];
                    y += 1;   
                x += 1;
                
        return ps_out
    
    def run(self, signal, noise, noise_sigma, sel):
        
        
        _nf = len(sel); Q = self.lbin
        s = 5.00827749e-5; Delta = 10*s; f = []; Evals = np.ones((Q,_nf)); E_cut = 1

        D_B_n = np.zeros(Q);
        
        #select part of the cross power spectrum 
        D = self.Select_fre(signal, sel); 
        nl_bin_mean = self.Select_fre(noise, sel); nl_bin_std = self.Select_fre(noise_sigma, sel)
        sigmaD = np.zeros(Q)
        
        for i in range(Q):
            f_q = np.ones(_nf)
            for j in range(_nf):
                f_q[j] = f_q[j]/np.sqrt(nl_bin_std[i][j, j])   ##nl_std_all.shape = (2, Q, Nf, Nf)
            f.append(f_q) 
    
        for l in range(Q):
            D[l] = D[l] - nl_bin_mean[l] 
            for i in range(_nf): 
                for j in range(_nf):
                    D[l][i,j] = D[l][i,j]/np.sqrt(nl_bin_std[l][i, i]*nl_bin_std[l][j, j]) + Delta*f[l][i]*f[l][j] 
    
        for l in range(0,Q): 
            e_vals,E = LA.eig(D[l])
            Evals[l,:] = e_vals        
    
            for i in range(_nf):
                E[:,i]=E[:,i]/LA.norm(E[:,i])**2  
    
            D_B_l = 0; sigmaD_l = 0; G = np.ones(_nf)
            for i in range(_nf):
                if e_vals[i]>=E_cut:
                    G_i = np.dot(f[l],E[:,i])
                    D_B_l += (G_i**2/e_vals[i])
    
            D_B_l = 1.0/ D_B_l - Delta
            D_B_n[l] = D_B_l
    
    #         ### Calculate the theoretical error of ABS method using perturbation theory...
    #         for i in range(Nf):
    #             G[i]= np.dot(f[l],E[:,i])
    #             sigmaD[l] += (G[i]**2/e_vals[i]**2)*(clbb_clean[l] + Delta)**2
            
    #         ## for C_alpha
    #         if l == Q-11:
    #             evals_test = e_vals
    #             con_i = np.ones(Nf)
    #             for i in range(Nf):
    #                 G_test= np.dot(f[l],E[:,i])
    #                 con_i[i] = (G_test**2/e_vals[i])
        
    #             for i in range(Nf):
    #                 con_i[i] = con_i[i]/np.sum(con_i)
            
        
        return D_B_n