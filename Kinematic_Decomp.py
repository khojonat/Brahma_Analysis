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
Ratios = []
Sigmas = []
BH_Masses = []
Star_Masses = []
Coords = []
Pot_E_norms = []


# Now looping through all subhalos with BHs and 1000 stars
for index in Desired_subhalos:
    
    # Skipping halos that might be broken 
    try: 
        
        fields = ['BH_Mass']
        Subhalo_BH_Masses = il.snapshot.loadSubhalo(basePath, snapNum=snap_num, id=index, partType=5, fields=fields)
        
        # Load in star properties of current halo
        fields = ['Masses','Coordinates','Velocities','Potential']
        Star_Props = il.snapshot.loadSubhalo(basePath, snap_num, id=index, partType=4, fields=fields)
        Star_Mass=Star_Props['Masses']*1e10/h # Units: Msun
        
        # Center coord and vel and correct units
        Coordinates,Velocities,Potentials = Center_subhalo(Star_Props,Subhaloprops,box_size,redshift,h,subhalo_id=index)
        
        # Calculating the binding energies of the stars
        Pot_E = Star_Mass * Potentials
        
        # Normalizing to max binding energy
        Pot_E_norm = Pot_E/np.abs(np.min(Pot_E))
        
        # Calculate id's of stars in the disk
        pos,grad,ratio = kinematic_decomp(Coordinates,Velocities,Potentials)
        
        bulge = ratio < 0.5
        
        Bulge_vel = Velocities[bulge]
        Bulge_mass = Star_Mass[bulge]
        
        # Calculate the velocity dispersion
        
        Mstars_total = np.sum(Bulge_mass) # Total stellar mass

        # Here we weight the sigma calculation by stellar mass
        mu_vel = np.mean(Bulge_vel,axis=0) # Average 3D stellar velocity for this subhalo
        DiffSquared=Bulge_mass[:, np.newaxis]*np.array((Bulge_vel - mu_vel)** 2)

        Sigma_halo = np.sqrt(np.sum(DiffSquared,axis=0) / Mstars_total)  # Calculate sigma from subhalo velocity
        
        Sigmas.append(Sigma_halo)
        BH_Masses.append(np.max(Subhalo_BH_Masses)) # Add most massive BH mass in subhalo to list
        Ratios.append(ratio) # Append the ratio of jz/jcirc for stars in the subhalo
        Coords.append(Coordinates)
        Star_Masses.append(Star_Mass)
        Pot_E_norms.append(Pot_E_norm)
        
        print('Halo: {},'.format(index),'Sigma: {},'.format(np.linalg.norm(Sigma_halo)),'BH mass: {},'.format(np.max(Subhalo_BH_Masses)),
             'Ratio max/min: {},'.format((np.max(ratio),np.min(ratio)) ) )
        
    except Exception:
        print('Skipping halo {},'.format(index))
    

Write2File(Ratios,Sigmas,BH_Masses,Coords,Star_Masses,pos,grad,Pot_E_norms,Central_subhalos,
           fname='Brahma_Data/Kin_Decomp_TNG_z0_r0.5')
    

