# Importing necessary packages
import sys
sys.path.append('/home/yja6qa/arepo_package/')

import arepo_package
import h5py
import os
import numpy as np
from brahma_analysis_clean import *
from sklearn.linear_model import LinearRegression

np.seterr(all="ignore") # Lots of 'RuntimeWarning: Mean of empty slice.' warnings due to NaN's I think...

h = 0.6774

path = '/home/yja6qa/arepo_package/Brahma_Data/'
# Loading in Brahma data
print('Loading Brahma data ...',flush=True)

bFOF_decomp_z0 = ReadBrahmaData(path+'bFOF_z0_decomp')
bFOF_decomp_z1 = ReadBrahmaData(path+'bFOF_z1_decomp')
bFOF_decomp_z2 = ReadBrahmaData(path+'bFOF_z2_decomp')
bFOF_decomp_z3 = ReadBrahmaData(path+'bFOF_z3_decomp')
bFOF_decomp_z4 = ReadBrahmaData(path+'bFOF_z4_decomp')
bFOF_decomp_z5 = ReadBrahmaData(path+'bFOF_z5_decomp')
bFOF_decomp_z6 = ReadBrahmaData(path+'bFOF_z6_decomp')
bFOF_decomp_z7 = ReadBrahmaData(path+'bFOF_z7_decomp')

LW10_decomp_z0 = ReadBrahmaData(path+'bFOF_LW10_z0_decomp')
LW10_decomp_z1 = ReadBrahmaData(path+'bFOF_LW10_z1_decomp')
LW10_decomp_z2 = ReadBrahmaData(path+'bFOF_LW10_z2_decomp')
LW10_decomp_z3 = ReadBrahmaData(path+'bFOF_LW10_z3_decomp')
LW10_decomp_z4 = ReadBrahmaData(path+'bFOF_LW10_z4_decomp')
LW10_decomp_z5 = ReadBrahmaData(path+'bFOF_LW10_z5_decomp')
LW10_decomp_z6 = ReadBrahmaData(path+'bFOF_LW10_z6_decomp')
LW10_decomp_z7 = ReadBrahmaData(path+'bFOF_LW10_z7_decomp')

spin_decomp_z0 = ReadBrahmaData(path+'bFOF_LW10_spin_z0_decomp')
spin_decomp_z1 = ReadBrahmaData(path+'bFOF_LW10_spin_z1_decomp')
spin_decomp_z2 = ReadBrahmaData(path+'bFOF_LW10_spin_z2_decomp')
spin_decomp_z3 = ReadBrahmaData(path+'bFOF_LW10_spin_z3_decomp')
spin_decomp_z4 = ReadBrahmaData(path+'bFOF_LW10_spin_z4_decomp')
spin_decomp_z5 = ReadBrahmaData(path+'bFOF_LW10_spin_z5_decomp')
spin_decomp_z6 = ReadBrahmaData(path+'bFOF_LW10_spin_z6_decomp')
spin_decomp_z7 = ReadBrahmaData(path+'bFOF_LW10_spin_z7_decomp')

rich_decomp_z0 = ReadBrahmaData(path+'bFOF_LW10_spin_rich_z0_decomp')
rich_decomp_z1 = ReadBrahmaData(path+'bFOF_LW10_spin_rich_z1_decomp')
rich_decomp_z2 = ReadBrahmaData(path+'bFOF_LW10_spin_rich_z2_decomp')
rich_decomp_z3 = ReadBrahmaData(path+'bFOF_LW10_spin_rich_z3_decomp')
rich_decomp_z4 = ReadBrahmaData(path+'bFOF_LW10_spin_rich_z4_decomp')
rich_decomp_z5 = ReadBrahmaData(path+'bFOF_LW10_spin_rich_z5_decomp')
rich_decomp_z6 = ReadBrahmaData(path+'bFOF_LW10_spin_rich_z6_decomp')
rich_decomp_z7 = ReadBrahmaData(path+'bFOF_LW10_spin_rich_z7_decomp')

# Storing the BH masses
print('Storing Brahma BH masses, stellar masses, and sigmas ...',flush=True)
MBH_decomp_bFOFz0 = np.array(bFOF_decomp_z0[5])
MBH_decomp_bFOFz1 = np.array(bFOF_decomp_z1[5])
MBH_decomp_bFOFz2 = np.array(bFOF_decomp_z2[5])
MBH_decomp_bFOFz3 = np.array(bFOF_decomp_z3[5])
MBH_decomp_bFOFz4 = np.array(bFOF_decomp_z4[5])
MBH_decomp_bFOFz5 = np.array(bFOF_decomp_z5[5])
MBH_decomp_bFOFz6 = np.array(bFOF_decomp_z6[5])
MBH_decomp_bFOFz7 = np.array(bFOF_decomp_z7[5])

MBH_decomp_LW10z0 = np.array(LW10_decomp_z0[5])
MBH_decomp_LW10z1 = np.array(LW10_decomp_z1[5])
MBH_decomp_LW10z2 = np.array(LW10_decomp_z2[5])
MBH_decomp_LW10z3 = np.array(LW10_decomp_z3[5])
MBH_decomp_LW10z4 = np.array(LW10_decomp_z4[5])
MBH_decomp_LW10z5 = np.array(LW10_decomp_z5[5])
MBH_decomp_LW10z6 = np.array(LW10_decomp_z6[5])
MBH_decomp_LW10z7 = np.array(LW10_decomp_z7[5])

MBH_decomp_spinz0 = np.array(spin_decomp_z0[5])
MBH_decomp_spinz1 = np.array(spin_decomp_z1[5])
MBH_decomp_spinz2 = np.array(spin_decomp_z2[5])
MBH_decomp_spinz3 = np.array(spin_decomp_z3[5])
MBH_decomp_spinz4 = np.array(spin_decomp_z4[5])
MBH_decomp_spinz5 = np.array(spin_decomp_z5[5])
MBH_decomp_spinz6 = np.array(spin_decomp_z6[5])
MBH_decomp_spinz7 = np.array(spin_decomp_z7[5])

MBH_decomp_richz0 = np.array(rich_decomp_z0[5])
MBH_decomp_richz1 = np.array(rich_decomp_z1[5])
MBH_decomp_richz2 = np.array(rich_decomp_z2[5])
MBH_decomp_richz3 = np.array(rich_decomp_z3[5])
MBH_decomp_richz4 = np.array(rich_decomp_z4[5])
MBH_decomp_richz5 = np.array(rich_decomp_z5[5])
MBH_decomp_richz6 = np.array(rich_decomp_z6[5])
MBH_decomp_richz7 = np.array(rich_decomp_z7[5])

# Storing the stellar masses
Mstar_decomp_bFOFz0 = np.array(bFOF_decomp_z0[6])
Mstar_decomp_bFOFz1 = np.array(bFOF_decomp_z1[6])
Mstar_decomp_bFOFz2 = np.array(bFOF_decomp_z2[6])
Mstar_decomp_bFOFz3 = np.array(bFOF_decomp_z3[6])
Mstar_decomp_bFOFz4 = np.array(bFOF_decomp_z4[6])
Mstar_decomp_bFOFz5 = np.array(bFOF_decomp_z5[6])
Mstar_decomp_bFOFz6 = np.array(bFOF_decomp_z6[6])
Mstar_decomp_bFOFz7 = np.array(bFOF_decomp_z7[6])

Mstar_decomp_LW10z0 = np.array(LW10_decomp_z0[6])
Mstar_decomp_LW10z1 = np.array(LW10_decomp_z1[6])
Mstar_decomp_LW10z2 = np.array(LW10_decomp_z2[6])
Mstar_decomp_LW10z3 = np.array(LW10_decomp_z3[6])
Mstar_decomp_LW10z4 = np.array(LW10_decomp_z4[6])
Mstar_decomp_LW10z5 = np.array(LW10_decomp_z5[6])
Mstar_decomp_LW10z6 = np.array(LW10_decomp_z6[6])
Mstar_decomp_LW10z7 = np.array(LW10_decomp_z7[6])

Mstar_decomp_spinz0 = np.array(spin_decomp_z0[6])
Mstar_decomp_spinz1 = np.array(spin_decomp_z1[6])
Mstar_decomp_spinz2 = np.array(spin_decomp_z2[6])
Mstar_decomp_spinz3 = np.array(spin_decomp_z3[6])
Mstar_decomp_spinz4 = np.array(spin_decomp_z4[6])
Mstar_decomp_spinz5 = np.array(spin_decomp_z5[6])
Mstar_decomp_spinz6 = np.array(spin_decomp_z6[6])
Mstar_decomp_spinz7 = np.array(spin_decomp_z7[6])

Mstar_decomp_richz0 = np.array(rich_decomp_z0[6])
Mstar_decomp_richz1 = np.array(rich_decomp_z1[6])
Mstar_decomp_richz2 = np.array(rich_decomp_z2[6])
Mstar_decomp_richz3 = np.array(rich_decomp_z3[6])
Mstar_decomp_richz4 = np.array(rich_decomp_z4[6])
Mstar_decomp_richz5 = np.array(rich_decomp_z5[6])
Mstar_decomp_richz6 = np.array(rich_decomp_z6[6])
Mstar_decomp_richz7 = np.array(rich_decomp_z7[6])

# Storing the bulge sigma's
Sigma_bulge_bFOFz0 = np.linalg.norm(bFOF_decomp_z0[1],axis=1)
Sigma_bulge_bFOFz1 = np.linalg.norm(bFOF_decomp_z1[1],axis=1)
Sigma_bulge_bFOFz2 = np.linalg.norm(bFOF_decomp_z2[1],axis=1)
Sigma_bulge_bFOFz3 = np.linalg.norm(bFOF_decomp_z3[1],axis=1)
Sigma_bulge_bFOFz4 = np.linalg.norm(bFOF_decomp_z4[1],axis=1)
Sigma_bulge_bFOFz5 = np.linalg.norm(bFOF_decomp_z5[1],axis=1)
Sigma_bulge_bFOFz6 = np.linalg.norm(bFOF_decomp_z6[1],axis=1)
Sigma_bulge_bFOFz7 = np.linalg.norm(bFOF_decomp_z7[1],axis=1)

Sigma_bulge_LW10z0 = np.linalg.norm(LW10_decomp_z0[1],axis=1)
Sigma_bulge_LW10z1 = np.linalg.norm(LW10_decomp_z1[1],axis=1)
Sigma_bulge_LW10z2 = np.linalg.norm(LW10_decomp_z2[1],axis=1)
Sigma_bulge_LW10z3 = np.linalg.norm(LW10_decomp_z3[1],axis=1)
Sigma_bulge_LW10z4 = np.linalg.norm(LW10_decomp_z4[1],axis=1)
Sigma_bulge_LW10z5 = np.linalg.norm(LW10_decomp_z5[1],axis=1)
Sigma_bulge_LW10z6 = np.linalg.norm(LW10_decomp_z6[1],axis=1)
Sigma_bulge_LW10z7 = np.linalg.norm(LW10_decomp_z7[1],axis=1)

Sigma_bulge_spinz0 = np.linalg.norm(spin_decomp_z0[1],axis=1)
Sigma_bulge_spinz1 = np.linalg.norm(spin_decomp_z1[1],axis=1)
Sigma_bulge_spinz2 = np.linalg.norm(spin_decomp_z2[1],axis=1)
Sigma_bulge_spinz3 = np.linalg.norm(spin_decomp_z3[1],axis=1)
Sigma_bulge_spinz4 = np.linalg.norm(spin_decomp_z4[1],axis=1)
Sigma_bulge_spinz5 = np.linalg.norm(spin_decomp_z5[1],axis=1)
Sigma_bulge_spinz6 = np.linalg.norm(spin_decomp_z6[1],axis=1)
Sigma_bulge_spinz7 = np.linalg.norm(spin_decomp_z7[1],axis=1)

Sigma_bulge_richz0 = np.linalg.norm(rich_decomp_z0[1],axis=1)
Sigma_bulge_richz1 = np.linalg.norm(rich_decomp_z1[1],axis=1)
Sigma_bulge_richz2 = np.linalg.norm(rich_decomp_z2[1],axis=1)
Sigma_bulge_richz3 = np.linalg.norm(rich_decomp_z3[1],axis=1)
Sigma_bulge_richz4 = np.linalg.norm(rich_decomp_z4[1],axis=1)
Sigma_bulge_richz5 = np.linalg.norm(rich_decomp_z5[1],axis=1)
Sigma_bulge_richz6 = np.linalg.norm(rich_decomp_z6[1],axis=1)
Sigma_bulge_richz7 = np.linalg.norm(rich_decomp_z7[1],axis=1)


bFOF_sigmas = [Sigma_bulge_bFOFz0,Sigma_bulge_bFOFz1,Sigma_bulge_bFOFz2,Sigma_bulge_bFOFz3,Sigma_bulge_bFOFz4,
               Sigma_bulge_bFOFz5,Sigma_bulge_bFOFz6,Sigma_bulge_bFOFz7]
bFOF_sigmas=[np.log10(i) for i in bFOF_sigmas]
bFOF_masses = [MBH_decomp_bFOFz0,MBH_decomp_bFOFz1,MBH_decomp_bFOFz2,MBH_decomp_bFOFz3,MBH_decomp_bFOFz4,MBH_decomp_bFOFz5,
              MBH_decomp_bFOFz6,MBH_decomp_bFOFz7]
bFOF_masses=[np.log10(i) for i in bFOF_masses]

LW10_sigmas = [Sigma_bulge_LW10z0,Sigma_bulge_LW10z1,Sigma_bulge_LW10z2,Sigma_bulge_LW10z3,Sigma_bulge_LW10z4,
               Sigma_bulge_LW10z5,Sigma_bulge_LW10z6,Sigma_bulge_LW10z7]
LW10_sigmas=[np.log10(i) for i in LW10_sigmas]
LW10_masses = [MBH_decomp_LW10z0,MBH_decomp_LW10z1,MBH_decomp_LW10z2,MBH_decomp_LW10z3,MBH_decomp_LW10z4,MBH_decomp_LW10z5,
              MBH_decomp_LW10z6,MBH_decomp_LW10z7]
LW10_masses=[np.log10(i) for i in LW10_masses]

spin_sigmas = [Sigma_bulge_spinz0,Sigma_bulge_spinz1,Sigma_bulge_spinz2,Sigma_bulge_spinz3,Sigma_bulge_spinz4,
               Sigma_bulge_spinz5,Sigma_bulge_spinz6,Sigma_bulge_spinz7]
spin_sigmas=[np.log10(i) for i in spin_sigmas]
spin_masses = [MBH_decomp_spinz0,MBH_decomp_spinz1,MBH_decomp_spinz2,MBH_decomp_spinz3,MBH_decomp_spinz4,MBH_decomp_spinz5,
              MBH_decomp_spinz6,MBH_decomp_spinz7]
spin_masses=[np.log10(i) for i in spin_masses]

rich_sigmas = [Sigma_bulge_richz0,Sigma_bulge_richz1,Sigma_bulge_richz2,Sigma_bulge_richz3,Sigma_bulge_richz4,
               Sigma_bulge_richz5,Sigma_bulge_richz6,Sigma_bulge_richz7]
rich_sigmas=[np.log10(i) for i in rich_sigmas]
rich_masses = [MBH_decomp_richz0,MBH_decomp_richz1,MBH_decomp_richz2,MBH_decomp_richz3,MBH_decomp_richz4,MBH_decomp_richz5,
              MBH_decomp_richz6,MBH_decomp_richz7]
rich_masses=[np.log10(i) for i in rich_masses]

bFOF_mstars = [Mstar_decomp_bFOFz0,Mstar_decomp_bFOFz1,Mstar_decomp_bFOFz2,Mstar_decomp_bFOFz3,Mstar_decomp_bFOFz4,
               Mstar_decomp_bFOFz5,Mstar_decomp_bFOFz6,Mstar_decomp_bFOFz7]
bFOF_mstars=[np.log10(i) for i in bFOF_mstars]

LW10_mstars = [Mstar_decomp_LW10z0,Mstar_decomp_LW10z1,Mstar_decomp_LW10z2,Mstar_decomp_LW10z3,Mstar_decomp_LW10z4,
              Mstar_decomp_LW10z5,Mstar_decomp_LW10z6,Mstar_decomp_LW10z7]
LW10_mstars=[np.log10(i) for i in LW10_mstars]

spin_mstars = [Mstar_decomp_spinz0,Mstar_decomp_spinz1,Mstar_decomp_spinz2,Mstar_decomp_spinz3,Mstar_decomp_spinz4,
              Mstar_decomp_spinz5,Mstar_decomp_spinz6,Mstar_decomp_spinz7]
spin_mstars=[np.log10(i) for i in spin_mstars]

rich_mstars = [Mstar_decomp_richz0,Mstar_decomp_richz1,Mstar_decomp_richz2,Mstar_decomp_richz3,Mstar_decomp_richz4,
              Mstar_decomp_richz5,Mstar_decomp_richz6,Mstar_decomp_richz7]
rich_mstars=[np.log10(i) for i in rich_mstars]

# Doing bootstrap of LHS and RHS for confidence interval error bars

BH_masses = [bFOF_masses,LW10_masses,spin_masses,rich_masses]
sigmas = [bFOF_sigmas,LW10_sigmas,spin_sigmas,rich_sigmas]
mstars = [bFOF_mstars,LW10_mstars,spin_mstars,rich_mstars]
const_sigmas = [1.5,1.75,2]
bin_width = 0.05

# Bootstrap parameters
n_resamples = 10000
rng = np.random.default_rng(42)

LHS = []
RHS = []
RHS_comp3 = []

print('Looping through bootstrapping ...',flush=True)

for n in range(len(BH_masses)):

    print('Starting new box!',flush=True)

    n_samples = [len(BH_masses[n][i]) for i in range(len(BH_masses[n]))]
    
    # Perform bootstrap
    LHS_box = []
    RHS_box = []
    RHS_comp3_box = []
    
    for _ in range(n_resamples):

        if _ == int(n_resamples/4): 
            print('1/4 of the way through this box ...',flush=True)
        elif _ == int(n_resamples/2): 
            print('1/2 of the way through this box ...',flush=True)
        elif _ == int(3*n_resamples/4): 
            print('3/4 of the way through this box ...',flush=True)

        resample_indices = [rng.integers(n, n_samples[i], n_samples[i]) for i in range(len(n_samples))]
        resample_masses = [BH_masses[n][i][resample_indices[i]] for i in range(len(resample_indices))]
        resample_sigmas = [sigmas[n][i][resample_indices[i]] for i in range(len(resample_indices))]
        resample_mstars = [mstars[n][i][resample_indices[i]] for i in range(len(resample_indices))]

        
        dmdz = calc_LHS(resample_masses,resample_sigmas,const_sigmas,bin_width)
        RHS_resampled = calc_RHS(resample_masses,resample_sigmas,resample_mstars,const_sigmas,bin_width)
        RHS_comp3_resampled = calc_RHS(resample_masses,resample_sigmas,resample_mstars,const_sigmas,bin_width,comp = 3)
        
        LHS_box.append(dmdz)
        RHS_box.append(RHS_resampled)
        RHS_comp3_box.append(RHS_comp3_resampled)

    LHS.append(LHS_box)
    RHS.append(RHS_box)
    RHS_comp3.append(RHS_comp3_box)

LHS = np.array(LHS)
RHS = np.array(RHS)
RHS_comp3 = np.array(RHS_comp3)


# Compute confidence interval
print('Now computing confidence intervals ...',flush=True)

alpha = 0.05
lowers_LHS = []
uppers_LHS = []
lowers_RHS = []
uppers_RHS = []
lowers_RHS_comp3 = []
uppers_RHS_comp3 = []

for i in range(4):

    lowers_i_LHS = []
    uppers_i_LHS = []
    lowers_i_RHS = []
    uppers_i_RHS = []
    lowers_i_RHS_comp3 = []
    uppers_i_RHS_comp3 = []
    
    for ii in range(3):

        lowers_ii_LHS = []
        uppers_ii_LHS = []
        lowers_ii_RHS = []
        uppers_ii_RHS = []
        lowers_ii_RHS_comp3 = []
        uppers_ii_RHS_comp3 = []
    
        for iii in range(LHS.shape[-1]):
        
            if len(LHS[i,:,ii,iii][~np.isnan(LHS[i,:,ii,iii])]) == 0:
                lowers_ii_LHS.append(np.nan)
                uppers_ii_LHS.append(np.nan)
            else:
                lower_LHS = np.abs(dmbh_dz_sigma_all[i,ii,iii] - np.array([np.percentile(LHS[i,:,ii,iii][~np.isnan(LHS[i,:,ii,iii])],
                                                                                     100 * (alpha / 2))]))
                upper_LHS = np.abs(dmbh_dz_sigma_all[i,ii,iii] - np.array([np.percentile(LHS[i,:,ii,iii][~np.isnan(LHS[i,:,ii,iii])],
                                                                                     100 * (1 - alpha / 2))]))
                lowers_ii_LHS.append(lower_LHS[0])
                uppers_ii_LHS.append(upper_LHS[0])
                lowers_ii_RHS.append(lower_RHS[0])
                uppers_ii_RHS.append(upper_RHS[0])
                lowers_ii_RHS_comp3.append(lower_RHS_comp3[0])
                uppers_ii_RHS_comp3.append(upper_RHS_comp3[0])
    
        lowers_i_LHS.append(lowers_ii_LHS)
        uppers_i_LHS.append(uppers_ii_LHS)
        lowers_i_RHS.append(lowers_ii_RHS)
        uppers_i_RHS.append(uppers_ii_RHS)
        lowers_i_RHS_comp3.append(lowers_ii_RHS_comp3)
        uppers_i_RHS_comp3.append(uppers_ii_RHS_comp3)

    lowers_LHS.append(lowers_i_LHS)
    uppers_LHS.append(uppers_i_LHS)
    lowers_RHS.append(lowers_i_RHS)
    uppers_RHS.append(uppers_i_RHS)
    lowers_RHS_comp3.append(lowers_i_RHS_comp3)
    uppers_RHS_comp3.append(uppers_i_RHS_comp3)

lowers_LHS = np.array(lowers_LHS)
uppers_LHS = np.array(uppers_LHS)
lowers_RHS = np.array(lowers_RHS)
uppers_RHS = np.array(uppers_RHS)
lowers_RHS_comp3 = np.array(lowers_RHS_comp3)
uppers_RHS_comp3 = np.array(uppers_RHS_comp3)

LHS, RHS, RHS_comp3 = [lowers_LHS,uppers_LHS],[lowers_RHS,uppers_RHS],[lowers_RHS_comp3,uppers_RHS_comp3]

Write2File(LHS, RHS, RHS_comp3,fname=path+'Eqn_bootstraps')
