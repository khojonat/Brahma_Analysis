import brahma_analysis
import sys
sys.path.append('/home/yja6qa/arepo_package/')

import arepo_package
import scipy.interpolate
import h5py
import os
import numpy as np
import plotting
from brahma_analysis import *
from sklearn.linear_model import LinearRegression

h = 0.6774
radiative_efficiency=0.2

path_to_output='/standard/torrey-group/BRAHMA/L12p5n512' # this is the folder containing the simulation run
run='/AREPO/' # name of the simulation runs
output='output_ratio10_SFMFGM5_seed5.00_bFOF/' # Name of the box we want to load data from
basePath = path_to_output+run+output # Combining paths to read data in 

file_format='fof_subfind'

desired_redshift=0
h = 0.6774 # hubble constant 

output_redshift,output_snapshot=arepo_package.desired_redshift_to_output_redshift(basePath,
                                                                    desired_redshift,list_all=False,file_format=file_format)

Pos = arepo_package.get_subhalo_property(basePath,'SubhaloPos',desired_redshift,postprocessed=1)
Vel = arepo_package.get_subhalo_property(basePath,'SubhaloVel',desired_redshift,postprocessed=1)

requested_property1=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Potential')
requested_property2=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Velocities')
requested_property3=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Coordinates')
requested_property4=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Masses')

SubhaloLenType,o = arepo_package.get_subhalo_property(basePath,'SubhaloLenType',desired_redshift,postprocessed=1)
SubhaloBHLen = SubhaloLenType[:,5]
SubhaloStarsLen = SubhaloLenType[:,4]
SubhaloIndices = np.arange(0,len(SubhaloBHLen))
mask1 = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>10)  # Only subhalos with a BH and with stars
mask2 = SubhaloStarsLen>10                # Only subhalos with stars; want another array of values for a Mstar-msigma plot

SubhaloIndicesWithBH = SubhaloIndices[mask1] # Return these so we can cross-reference which subhalos to plot
SubhaloIndicesWithStars = SubhaloIndices[mask2]
desired_indices = range(len(SubhaloIndicesWithStars))

N_disk = []

for i in desired_indices:
    ActualSubhaloIndex = SubhaloIndicesWithStars[i]

    Pot_subhalo,Pot_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Potential',4,output_redshift,ActualSubhaloIndex,requested_property1,store_all_offsets=1,group_type='subhalo')
    Vel_subhalo,Vel_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Velocities',4,output_redshift,ActualSubhaloIndex,requested_property2,store_all_offsets=1,group_type='subhalo')
    Pos_subhalo,Pos_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Coordinates',4,output_redshift,ActualSubhaloIndex,requested_property3,store_all_offsets=1,group_type='subhalo')
    Mstar_subhalo,Mstar_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Masses',4,output_redshift,ActualSubhaloIndex,requested_property4,store_all_offsets=1,group_type='subhalo')
    
    # From assuming a circular orbit for a star with the given grav. potential energy
    maxv = np.sqrt(-Pot_subhalo/Mstar_subhalo)

    # Subtract off subhalo velocity and position
    Vel_subhalo -= Vel[0][ActualSubhaloIndex]
    Pos_subhalo -= Pos[0][ActualSubhaloIndex]
    
    # Calculate magnitude of specific angular momentum for stars
    j = np.linalg.norm(np.cross(Vel_subhalo,Pos_subhalo))
    
    # Calculate magnitude of max specific angular momentum
    jmax = np.linalg.norm(Pos_subhalo)*maxv
    ratio = j/jmax
    
    # Stars with ratio > 0.7 are considered part of the disk
    N_disk.append(len(ratio[ratio>0.7]))
    
brahma_analysis.Write2File(N_disk,fname = 'Brahma_Data/SM5_z0_N_diskstars')