'''
Now recalculating M-sigma for Brahma sims with the bulge-disk decomposition
'''

import brahma_analysis
import sys
import os
sys.path.append('/home/yja6qa/arepo_package')

import numpy as np
import arepo_package
import math
import matplotlib.pyplot as plt
import illustris_python as il
import pickle

path_to_output='/standard/torrey-group/BRAHMA/L12p5n512' # this is the folder containing the simulation run
run='/AREPO/' # name of the simulation runs
output='output_ratio10_SFMFGM5_seed5.00_bFOF/' # Name of the box we want to load data from
basePath = path_to_output+run+output # Combining paths to read data in 
file_format='fof_subfind'
desired_redshift=0
kpc2km=3.0857e16

print('Making Subhalo masks...')
SubhaloLenType,o = arepo_package.get_subhalo_property(basePath,'SubhaloLenType',desired_redshift,postprocessed=1)
SubhaloBHLen = SubhaloLenType[:,5]
SubhaloStarsLen = SubhaloLenType[:,4]
SubhaloIndices = np.arange(0,len(SubhaloBHLen))
mask1 = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>1000)  # Only subhalos with a BH and with 1000 stars
mask2 = SubhaloStarsLen>10                # Only subhalos with stars; want another array of values for a Mstar-msigma plot

SubhaloIndicesWithBH = SubhaloIndices[mask1] # Return these so we can cross-reference which subhalos to plot
SubhaloIndicesWithStars = SubhaloIndices[mask2]

print('Loading header info...')

snap_num=32 # z=0
hdr  = il.groupcat.loadHeader(basePath, snap_num)
h = hdr['HubbleParam'] ## load in h from the header
box_size = hdr['BoxSize']
redshift = hdr['Redshift']

print('Loading star and BH properties')

output_redshift,output_snapshot=arepo_package.desired_redshift_to_output_redshift(basePath,
                                                                    desired_redshift,list_all=False,file_format=file_format)
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
Sigmas = []
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
    Age_subhalo,Age_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'GFM_StellarFormationTime',4,output_redshift,ActualSubhaloIndex,Age,store_all_offsets=1,group_type='subhalo')
    BHMasses_subhalo,BHMasses_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'BH_Mass',5,desired_redshift,ActualSubhaloIndex,requested_property3,store_all_offsets=1,group_type='subhalo')
    
    # Removing wind particles
    mask = Age_subhalo > 0
    Vel_subhalo = Vel_subhalo[mask]
    mstar_subhalo = mstar_subhalo[mask]
    pot_subhalo = pot_subhalo[mask]

    Star_Props = {'Masses':mstar_subhalo,'Coordinates':pos_subhalo,'Velocities':Vel_subhalo,'Potential':pot_subhalo}
    HMR = Subhaloprops['SubhaloHalfmassRad'][index]
    
    Coordinates,Velocities,Potentials = Center_subhalo(Star_Props,Subhaloprops,box_size,redshift,h,subhalo_id=index)
    
    ratio,negids,rcs = kinematic_decomp_e2(Coordinates,Velocities,Potentials,HMR)

    Velocities[negids] = np.nan

    bulge = ratio < 0.5

    Bulge_vel = Velocities[bulge]
    Bulge_mass = Star_Mass[bulge]

    # Calculate the velocity dispersion

    Mstars_total = np.sum(Bulge_mass) # Total stellar mass

    # Here we weight the sigma calculation by stellar mass
    mu_vel = np.mean(Bulge_vel,axis=0) # Average 3D stellar velocity for this subhalo
    DiffSquared=Bulge_mass[:, np.newaxis]*np.array((Bulge_vel - mu_vel)** 2)

    Sigma_halo = np.sqrt(np.sum(DiffSquared,axis=0) / Mstars_total)  # Calculate sigma from subhalo velocity

    if len(ratio[~np.isnan(ratio)]) == 0:
        failed_subhalos += 1
        print("Subhalo {} failed".format(index))
    else:
        print('Subhalo: {},'.format(index),'Sigma: {},'.format(np.linalg.norm(Sigma_halo)),'BH mass: {},'.format(np.max(np.max(BHMasses_subhalo)*1e10*h)),
         'Ratio max/min: {},'.format((np.max(ratio[~np.isnan(ratio)]),np.min(ratio[~np.isnan(ratio)])) ) )

    Ratios.append(ratio)
    Sigmas.append(Sigma_halo)
    BH_Masses.append(np.max(BHMasses_subhalo)*1e10*h)
    Star_Masses.append(np.sum(mstar_subhalo)*1e10*h)
    Coords.append(Coordinates)
    Subhalo_vels.append(Velocities)

brahma_analysis.Write2File(Ratios,Sigmas,BH_Masses,Star_Masses,Coords,Subhalo_vels,fname='Brahma_SM5_z0_decomp')

    
