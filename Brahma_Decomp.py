'''
Now recalculating M-sigma for Brahma sims with the bulge-disk decomposition
'''

from brahma_analysis import *
import sys
import os
sys.path.append('/home/yja6qa/arepo_package')

import numpy as np
import arepo_package
import math
import matplotlib.pyplot as plt
import illustris_python as il
import pickle

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Setting path and constants for this job
path_to_output='/standard/torrey-group/BRAHMA/L12p5n512' # this is the folder containing the simulation run
run='/AREPO/' # name of the simulation runs
output='output_ratio10_SFMFGM5_seed5.00_' # Base name included in every box

# Change these!
box = 'bFOF_LW10_spin_rich' # Name of the box we want to load data from
desired_redshift=6 # Redshift of box that I want
nstars_min = 30

basePath = path_to_output+run+output+box # Combining paths to read data in 
file_format='fof_subfind'
kpc2km=3.0857e16 # Conversion from kpc to km for units

print('Making Subhalo masks...', flush=True)
SubhaloLenType,o = arepo_package.get_subhalo_property(basePath,'SubhaloLenType',desired_redshift,postprocessed=1)
SubhaloBHLen = SubhaloLenType[:,5]
SubhaloStarsLen = SubhaloLenType[:,4]
SubhaloIndices = np.arange(0,len(SubhaloBHLen))
mask = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>nstars_min)  # Only subhalos with a BH and with 1000 stars

SubhaloIndicesWithBH = SubhaloIndices[mask] # Return these so we can cross-reference which subhalos to plot

print('Loading header info...', flush=True)
output_redshift,output_snapshot=arepo_package.desired_redshift_to_output_redshift(basePath,
                                                                    desired_redshift,list_all=False,file_format=file_format)
hdr  = il.groupcat.loadHeader(basePath, output_snapshot)
h = hdr['HubbleParam'] ## load in h from the header
box_size = hdr['BoxSize']
redshift = hdr['Redshift']

print('Loading star and BH properties', flush=True)

Stellar_vel=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Velocities')
Stellar_pos=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Coordinates')
Stellar_pot=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Potential')
Stellar_mass=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Masses')
requested_property=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=5,fields='BH_Mass')

# Grabbing stellar age; negative ages are wind particles that should not be included in sigma calculation
Age=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='GFM_StellarFormationTime')

# Subhalo properties
Subhaloprops,o =  arepo_package.get_subhalo_property(basePath,['SubhaloVel','SubhaloPos','SubhaloHalfmassRad'],desired_redshift,postprocessed=1)

# Scale factor calculation for unit corrections
a = 1/(1+output_redshift)

# Initialize lists to append to:
Ratios = []
Bulge_sigmas = []
Disk_sigmas = []
Total_sigmas = []
BH_Masses = []
Star_Masses = []
Coords = []
Subhalo_vels = []
failed_subhalos = 0


for index in SubhaloIndicesWithBH:
    Vel_subhalo,Vel_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Velocities',4,output_redshift,index,Stellar_vel,store_all_offsets=1,group_type='subhalo')
    pos_subhalo,pos_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Coordinates',4,output_redshift,index,Stellar_pos,store_all_offsets=1,group_type='subhalo')
    pot_subhalo,pot_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Potential',4,output_redshift,index,Stellar_pot,store_all_offsets=1,group_type='subhalo')
    mstar_subhalo,mstar_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Masses',4,output_redshift,index,Stellar_mass,store_all_offsets=1,group_type='subhalo')
    Age_subhalo,Age_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'GFM_StellarFormationTime',4,output_redshift,index,Age,store_all_offsets=1,group_type='subhalo')
    BHMasses_subhalo,BHMasses_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'BH_Mass',5,desired_redshift,index,requested_property,store_all_offsets=1,group_type='subhalo')
    
    # Removing wind particles
    mask = Age_subhalo > 0
    Vel_subhalo = Vel_subhalo[mask]
    mstar_subhalo = mstar_subhalo[mask]
    pot_subhalo = pot_subhalo[mask]
    pos_subhalo = pos_subhalo[mask]

    Star_Props = {'Masses':mstar_subhalo,'Coordinates':pos_subhalo,'Velocities':Vel_subhalo,'Potential':pot_subhalo}
    HMR = Subhaloprops['SubhaloHalfmassRad'][index]
    
    Coordinates,Velocities,Potentials = Center_subhalo(Star_Props,Subhaloprops,box_size,redshift,h,subhalo_id=index)

    # If there are at least 1000 stars, proceed normally
    if len(Coordinates) > 1000:
        Vals = kinematic_decomp_e2(Coordinates,Velocities,Potentials,HMR,nstars_min=nstars_min)
    # Otherwise, set the number of stars per bin to be ~1/20 the total number of stars, to make ~20 bins
    else:
        Vals = kinematic_decomp_e2(Coordinates,Velocities,Potentials,HMR,nstars_min=nstars_min,nstars=int(len(Coordinates)/20))

    if Vals == np.nan: # Some subhalos still have less stars than nstars_min apparently...
        continue
    else:
        ratio,negids,rcs = Vals[0],Vals[1],Vals[2]
        
    Velocities[negids] = np.nan

    # Take the 3D velocity before doing standard deviation calculation
    Vel3d = np.linalg.norm(Velocities,axis=1)

    bulge = ratio < 0.5
    disk = (ratio > 0.5) & (ratio < 1)

    Bulge_vel = Vel3d[bulge]
    Bulge_mass = mstar_subhalo[bulge]

    Disk_vel = Vel3d[disk]
    Disk_mass = mstar_subhalo[disk]
    
    # Calculate the velocity dispersion

    Mbulge_total = np.sum(Bulge_mass) # Total stellar mass
    Mdisk_total = np.sum(Disk_mass)
    Mstars_total = np.sum(mstar_subhalo)

    # Here we weight the sigma calculation by stellar mass
    mu_vel_bulge = np.mean(Bulge_vel,axis=0) # Average 3D stellar velocity for this subhalo
    mu_vel_disk = np.mean(Disk_vel,axis=0) 
    mu_vel_total = np.mean(Vel3d,axis=0) 
    
    BulgeDiffSquared=Bulge_mass*np.array((Bulge_vel - mu_vel_bulge)** 2)
    DiskDiffSquared=Disk_mass*np.array((Disk_vel - mu_vel_disk)** 2)
    TotalDiffSquared=mstar_subhalo*np.array((Vel3d - mu_vel_total)** 2)

    Sigma_bulge = np.sqrt(np.sum(BulgeDiffSquared,axis=0) / Mbulge_total)  # Calculate sigma from subhalo velocity
    Sigma_disk = np.sqrt(np.sum(DiskDiffSquared,axis=0) / Mdisk_total)
    Sigma_total = np.sqrt(np.sum(TotalDiffSquared,axis=0) / Mstars_total)

    if len(ratio[~np.isnan(ratio)]) == 0:
        failed_subhalos += 1
        print("Subhalo {} failed".format(index), flush=True)
    else:
        print('Subhalo: {},'.format(index),'Bulge sigma: {},'.format(Sigma_bulge),'Total sigma: {},'.format(Sigma_total),
              'BH mass: {},'.format(np.max(np.max(BHMasses_subhalo)*1e10*h)),
              'Ratio max/min: {},'.format((np.max(ratio[~np.isnan(ratio)]),np.min(ratio[~np.isnan(ratio)])) ) , flush=True)

    Ratios.append(ratio)
    Bulge_sigmas.append(Sigma_bulge)
    Disk_sigmas.append(Sigma_disk)
    Total_sigmas.append(Sigma_total)
    BH_Masses.append(np.max(BHMasses_subhalo)*1e10*h)
    Star_Masses.append(np.sum(mstar_subhalo)*1e10*h)
    Coords.append(Coordinates)
    Subhalo_vels.append(Velocities)

Write2File(Ratios,Bulge_sigmas,Disk_sigmas,Total_sigmas,BH_Masses,Star_Masses,Coords,Subhalo_vels,
           fname=f'Brahma_Data/{box}_z{desired_redshift}_decomp')

    
