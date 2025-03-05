import sys
from brahma_analysis import *
sys.path.append('/home/yja6qa/arepo_package/')

import arepo_package
import h5py
import os
import numpy as np

radiative_efficiency=0.2

TNGpath='/standard/torrey-group/IllustrisTNG/Runs/L75n1820TNG'
basePath = TNGpath
snap_num=99 # z=0
hdr  = il.groupcat.loadHeader(basePath, snap_num)
h = hdr['HubbleParam'] ## load in h from the header
box_size = hdr["BoxSize"]
redshift = hdr['Redshift']

# Determining size of subhalos
fields=['SubhaloLenType']
Subhalo_lengths = il.groupcat.loadSubhalos(basePath,snap_num,fields=fields)

SubhaloBHLen = Subhalo_lengths[:,5] # Number of BHs in each subhalo
SubhaloStarsLen = Subhalo_lengths[:,4] # Number of Stars in each subhalo
SubhaloIndices = np.arange(0,len(SubhaloBHLen)) # Indices of subhalos
mask1 = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>1000)  # Only subhalos with a BH and with 1000 stars
Desired_subhalos = SubhaloIndices[mask1] # Only indices of subhalos we want

# Load in all subhalo positions and velocities 
Subhaloprops = il.groupcat.loadSubhalos(basePath,snap_num,fields=['SubhaloPos','SubhaloVel'])
Centrals = il.groupcat.loadHalos(basePath=basePath,snapNum=snap_num,fields='GroupFirstSub')
Central_subhalos = Centrals[Centrals!=-1]

# Initialize lists to append to
Ratios1 = []
Ratios2 = []
Ratios3 = []
Sigmas1 = []
Sigmas2 = []
BH_Masses = []
Star_Masses = []
Coords = []
e_bind_norms = []
Pot_radii = []
Pot_grads = []
Central_ids = []
Subhalo_vels = []


# Now looping through all subhalos with BHs and 1000 stars
for index in Desired_subhalos:
    
    # Skipping halos that might be broken 
    # try: 
        
    fields = ['BH_Mass']
    Subhalo_BH_Masses = il.snapshot.loadSubhalo(basePath, snapNum=snap_num, id=index, partType=5, fields=fields)

    # Load in star properties of current halo
    fields = ['Masses','Coordinates','Velocities','Potential']
    Star_Props = il.snapshot.loadSubhalo(basePath, snap_num, id=index, partType=4, fields=fields)
    Star_Mass=Star_Props['Masses']*1e10/h # Units: Msun

    # Center coord and vel and correct units
    Coordinates,Velocities,Potentials = Center_subhalo(Star_Props,Subhaloprops,box_size,redshift,h,subhalo_id=index)

    # Calculating specific binding energy
    e_bind = 0.5*np.linalg.norm(np.array(Velocities)**2,axis=1) + Potentials

    # Normalizing to max binding energy
    e_bind_norm = e_bind/np.abs(np.min(e_bind))

    # Calculate id's of stars in the bulge
    pos,grad,ratio1,ratio2 = kinematic_decomp(Coordinates,Velocities,Potentials)

    # 3rd circularity metric; not sure how helpful this will be...
    ratio3 = np.cross(Coordinates,Velocities)[:,2]/(np.linalg.norm(Coordinates,axis=1)*np.linalg.norm(Velocities,axis=1))

    bulge1 = ratio1 < 0.5
    bulge2 = ratio2 < 0.5

    Bulge_vel1 = Velocities[bulge1]
    Bulge_mass1 = Star_Mass[bulge1]

    Bulge_vel2 = Velocities[bulge2]
    Bulge_mass2 = Star_Mass[bulge2]

    # Calculate the velocity dispersion

    Mstars_total1 = np.sum(Bulge_mass1) # Total stellar mass
    Mstars_total2 = np.sum(Bulge_mass2) 

    # Here we weight the sigma calculation by stellar mass
    mu_vel1 = np.mean(Bulge_vel1,axis=0) # Average 3D stellar velocity for this subhalo
    DiffSquared1=Bulge_mass1[:, np.newaxis]*np.array((Bulge_vel1 - mu_vel1)** 2)
    mu_vel2 = np.mean(Bulge_vel2,axis=0) 
    DiffSquared2=Bulge_mass2[:, np.newaxis]*np.array((Bulge_vel2 - mu_vel2)** 2)

    Sigma_halo1 = np.sqrt(np.sum(DiffSquared1,axis=0) / Mstars_total1)  # Calculate sigma from subhalo velocity
    Sigma_halo2 = np.sqrt(np.sum(DiffSquared2,axis=0) / Mstars_total2)  # Calculate sigma from subhalo velocity

    Sigmas1.append(Sigma_halo1)
    Sigmas2.append(Sigma_halo2)
    BH_Masses.append(np.max(Subhalo_BH_Masses)) # Add most massive BH mass in subhalo to list
    Ratios1.append(ratio1) # Append the ratio of jz/jcirc for stars in the subhalo
    Ratios2.append(ratio2)
    Ratios3.append(ratio3)
    Coords.append(Coordinates)
    Star_Masses.append(Star_Mass)
    e_bind_norms.append(e_bind_norm)
    Pot_radii.append(pos)
    Pot_grads.append(grad)
    Subhalo_vels.append(Velocities)
    
    if index in Central_subhalos:
        Central_ids.append(index)
    else:
        Central_ids.append(-1) # A value of -1 indicates the the subhalo is not a central

    print('Halo: {},'.format(index),'Sigma: {},'.format(np.linalg.norm(Sigma_halo1)),'BH mass: {},'.format(np.max(Subhalo_BH_Masses)),
         'Ratio max/min: {},'.format((np.max(ratio1),np.min(ratio1)) ) )
        
    # except Exception:
    #     print('Skipping halo {},'.format(index))
    

Write2File(Ratios1,Ratios2,Ratios3,Sigmas1,Sigmas2,BH_Masses,Coords,Star_Masses,Pot_radii,Pot_grads,e_bind_norms,
           Central_ids,Subhalo_vels,fname='Brahma_Data/Kin_Decomp_TNG_z0_r0.5')
    

