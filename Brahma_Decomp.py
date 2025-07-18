'''
This script was written to apply the kinematic decomposition to the BRAHMA simulations
and store the BH masses and bulge stellar masses and stellar velocity dispersions
'''

from brahma_analysis_clean import *
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
box = 'bFOF' # Name of the box we want to load data from
# desired_redshift=0 
desired_redshift=int(sys.argv[1]) # Redshift of box that I want
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
Subhaloprops,o =  arepo_package.get_subhalo_property(basePath,['SubhaloVel','SubhaloPos','SubhaloHalfmassRadType','SubhaloHalfmassRad'],desired_redshift,postprocessed=1)

# Scale factor calculation for unit corrections
a = 1/(1+output_redshift)

# Initialize lists to append to:
Ratios = []
Bulge_sigmas = []
Disk_sigmas = []
Total_sigmas = []
HMR_sigmas = []
BH_Masses = []
Star_Masses = []
Coords = []
Subhalo_vels = []
StellarHMRs = []
HMRs = []
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
    Stellar_HMR = Subhaloprops['SubhaloHalfmassRadType'][index][4] * a/h # Units: kpc
    HMR = Subhaloprops['SubhaloHalfmassRad'][index] * a/h # Units: kpc
    
    Coordinates,Velocities,Potentials = Center_subhalo(Star_Props,Subhaloprops,box_size,redshift,h,subhalo_id=index)
    radii = np.linalg.norm(Coordinates,axis=1)
    HMR_mask = radii < Stellar_HMR * kpc2km # For computing sigma in HMR

    # If there are at least 1000 stars, proceed normally
    if len(Coordinates) > 1000:
        Vals = kinematic_decomp_e(Coordinates,Velocities,Potentials,nstars_min=nstars_min)
    # Otherwise, set the number of stars per bin to be ~1/20 the total number of stars, to make ~20 bins
    else:
        Vals = kinematic_decomp_e(Coordinates,Velocities,Potentials,nstars_min=nstars_min,nstars=int(len(Coordinates)/20))

    if Vals == np.nan: # Some subhalos still have less stars than nstars_min apparently...
        continue
    else:
        ratio,negids,rcs = Vals[0],Vals[1],Vals[2]
        
    Velocities[negids] = np.nan

    bulge = ratio < 0.5
    disk = (ratio > 0.5) & (ratio < 1)

    Bulge_vel = Velocities[bulge]
    Bulge_mass = mstar_subhalo[bulge]
    Bulge_mass = Bulge_mass.reshape(len(Bulge_mass),1)

    Disk_vel = Velocities[disk]
    Disk_mass = mstar_subhalo[disk]
    Disk_mass = Disk_mass.reshape(len(Disk_mass),1)

    HMR_vel = Velocities[HMR_mask]
    HMR_mass = mstar_subhalo[HMR_mask]
    HMR_mass = HMR_mass.reshape(len(HMR_mass),1)
    
    # Calculate the velocity dispersion

    Mbulge_total = np.sum(Bulge_mass) # Total stellar mass
    Mdisk_total = np.sum(Disk_mass)
    Mstars_total = np.sum(mstar_subhalo)
    MHMR_total = np.sum(HMR_mass)

    Total_mass = mstar_subhalo.reshape(len(mstar_subhalo),1)

    # Here we weight the sigma calculation by stellar mass
    mu_vel_bulge = np.sum(Bulge_mass * Bulge_vel,axis=0) / Mbulge_total
    mu_vel_disk = np.sum(Disk_mass * Disk_vel,axis=0) / Mdisk_total
    mu_vel_total = np.sum(Total_mass * Velocities,axis=0) / Mstars_total
    mu_vel_HMR = np.sum(HMR_mass * HMR_vel,axis=0) / MHMR_total
    
    BulgeDiffSquared=Bulge_mass*np.array((Bulge_vel - mu_vel_bulge)** 2)
    DiskDiffSquared=Disk_mass*np.array((Disk_vel - mu_vel_disk)** 2)
    TotalDiffSquared=Total_mass*np.array((Velocities - mu_vel_total)** 2)
    HMRDiffSquared=HMR_mass*np.array((HMR_vel - mu_vel_HMR)** 2)

    Sigma_bulge = np.sqrt(np.sum(BulgeDiffSquared,axis=0) / Mbulge_total)  # Calculate sigma from subhalo velocity
    Sigma_disk = np.sqrt(np.sum(DiskDiffSquared,axis=0) / Mdisk_total)
    Sigma_total = np.sqrt(np.sum(TotalDiffSquared,axis=0) / Mstars_total)
    Sigma_HMR = np.sqrt(np.sum(HMRDiffSquared,axis=0) / MHMR_total)

    if len(ratio[~np.isnan(ratio)]) == 0:
        failed_subhalos += 1
        print("Subhalo {} failed".format(index), flush=True)
    else:
        print('Subhalo: {},'.format(index),'Bulge sigma: {},'.format(Sigma_bulge),'Total sigma: {},'.format(Sigma_total),
              'BH mass: {},'.format(np.max(np.max(BHMasses_subhalo)*1e10/h)),
              'Ratio max/min: {},'.format((np.max(ratio[~np.isnan(ratio)]),np.min(ratio[~np.isnan(ratio)])) ) , flush=True)
    
    Ratios.append(ratio)
    Bulge_sigmas.append(Sigma_bulge)
    Disk_sigmas.append(Sigma_disk)
    Total_sigmas.append(Sigma_total)
    HMR_sigmas.append(Sigma_HMR)
    BH_Masses.append(np.max(BHMasses_subhalo)*1e10/h)
    Star_Masses.append(np.sum(mstar_subhalo)*1e10/h)
    Coords.append(Coordinates)
    Subhalo_vels.append(Velocities)
    StellarHMRs.append(Stellar_HMR)
    HMRs.append(HMR)

Write2File(Ratios,Bulge_sigmas,Disk_sigmas,Total_sigmas,HMR_sigmas,BH_Masses,Star_Masses,Coords,Subhalo_vels,StellarHMRs,HMRs,
           fname=f'Brahma_Data/{box}_z{desired_redshift}_decomp')