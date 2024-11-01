'''
This script is intended to read in data from the Brahma simulations
and produce a pickle file that can be loaded in for data analysis.

In particular, we seek to load in data related to the M-Sigma relation.
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

# Path to simulation data
path_to_output='/standard/torrey-group/BRAHMA/L12p5n512' # this is the folder containing the simulation run
run='/AREPO/' # name of the simulation runs
output='output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_spin_rich/'
basePath = path_to_output+run+output

file_format='fof_subfind'

desired_redshift=5
h = 0.6774 # hubble constant 

M = []
MStars=[]
ZStars=[]
SFR=[]
Sigmas = []
# SigmaCOM = []
VelsMagBHs = []
VelsMagCOMs = []
VelBHs = []
VelCOMs = []
NStars = []
BH_Progs=[]
BH_Mdots=[]
BHIDs=[]
ZGas=[]
BHArrayIndex=[]
IgnoredBhs = 0
    
SubhaloLenType,o = arepo_package.get_subhalo_property(basePath,'SubhaloLenType',desired_redshift,postprocessed=1)
SubhaloBHLen = SubhaloLenType[:,5]
SubhaloStarsLen = SubhaloLenType[:,4]
SubhaloIndices = np.arange(0,len(SubhaloBHLen))
mask = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>0)  # Only subhalos with a BH and with stars
SubhaloIndicesWithStars = SubhaloIndices[mask]

desired_indices = range(len(SubhaloIndicesWithStars))

SubhaloArrayIndex=SubhaloIndicesWithStars

AllBHIDs = arepo_package.get_particle_property(basePath,'ParticleIDs',5,desired_redshift)


# From get_particle_property_within_postprocessed_groups
output_redshift,output_snapshot=arepo_package.desired_redshift_to_output_redshift(basePath,
                                                                    desired_redshift,list_all=False,file_format=file_format)
requested_property1=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Velocities')
requested_property2=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Masses')
requested_property3=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=5,fields='Masses')
requested_property4=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=5,fields='BH_Progs')
requested_property5=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=5,fields='ParticleIDs')
requested_property6=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=5,fields='BH_Mdot')
requested_property7=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=5,fields='Velocities')

# Subhalo properties
SubhaloSFR,o = arepo_package.get_subhalo_property(basePath,'SubhaloSFR',desired_redshift,postprocessed=1)
SubhaloZStars,o = arepo_package.get_subhalo_property(basePath,'SubhaloStarMetallicity',desired_redshift,postprocessed=1)
SubhaloZGas,o = arepo_package.get_subhalo_property(basePath,'SubhaloGasMetallicity',desired_redshift,postprocessed=1)
SubhaloCMvel,o = arepo_package.get_subhalo_property(basePath,'SubhaloVel',desired_redshift,postprocessed=1)

    
# Scale factor calculation
a = 1/(1+output_redshift)
    
for index in desired_indices:
        
    ActualSubhaloIndex = SubhaloIndicesWithStars[index]
    Vel_subhalo,Vel_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'Velocities',4,output_redshift,ActualSubhaloIndex,requested_property1,store_all_offsets=1,group_type='subhalo')
    MStars_subhalo,Mstars_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'Masses',4,output_redshift,ActualSubhaloIndex,requested_property2,store_all_offsets=1,group_type='subhalo')
    BHMasses_subhalo,BHMasses_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'Masses',5,desired_redshift,ActualSubhaloIndex,requested_property3,store_all_offsets=1,group_type='subhalo')
    BHProgs_subhalo,BHProgs_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'BH_Progs',5,desired_redshift,ActualSubhaloIndex,requested_property4,store_all_offsets=1,group_type='subhalo')
    BHID_subhalo,BHID_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'ParticleIDs',5,desired_redshift,ActualSubhaloIndex,requested_property5,store_all_offsets=1,group_type='subhalo')
    BHMdot_subhalo,BHMdot_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'BH_Mdot',5,desired_redshift,ActualSubhaloIndex,requested_property6,store_all_offsets=1,group_type='subhalo')
    BH_vel_subhalo,BH_vel_subhalo_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'Velocities',5,desired_redshift,ActualSubhaloIndex,requested_property7,store_all_offsets=1,group_type='subhalo')
    
        
    # Velocity calculations
    N = len(Vel_subhalo) # number of stars
    Vel_BH = np.array(Vel_subhalo)/np.sqrt(a) - BH_vel_subhalo[0]/np.sqrt(a) # New units: km/s
    Vel_CM = np.array(Vel_subhalo)/np.sqrt(a) - SubhaloCMvel[ActualSubhaloIndex]   # New units: km/s
        
    VelocityMagBH = np.linalg.norm(Vel_BH, axis=1) # Calculate velocity magnitudes
    VelocityMagCM = np.linalg.norm(Vel_CM, axis=1)
    mu_BH = np.mean(Vel_BH,axis=0) # Average 3D stellar velocity for this subhalo
    Mstars_total = np.sum(MStars_subhalo)
    
    # Here we weight the sigma calculation by stellar mass
    BHDiffSquared=MStars_subhalo[:, np.newaxis]*np.array((Vel_BH - mu_BH)** 2)

    Sigma = np.sqrt(np.sum(BHDiffSquared,axis=0) / Mstars_total)  # Calculate sigma from subhalo velocity
    
    M.append(np.max(BHMasses_subhalo)*1e10*h)
    MStars.append(MStars_subhalo*1e10*h)
    ZStars.append(SubhaloZStars[index])
    SFR.append(SubhaloSFR[index])
    Sigmas.append(Sigma) 
    VelsMagBHs.append(VelocityMagBH)
    VelsMagCOMs.append(VelocityMagCM)
    VelBHs.append(Vel_BH)
    VelCOMs.append(Vel_CM)
    
    
    NStars.append(N)
    BH_Progs.append(BHProgs_subhalo[0])
    BH_Mdots.append(BHMdot_subhalo[0])
    BHIDs.append(BHID_subhalo[0])
    BHArrayIndex.append(np.where(AllBHIDs[0] == BHID_subhalo[0])[0][0])
    ZGas.append(SubhaloZGas[index])
    IgnoredBhs += len(BHMasses_subhalo)-1
    
    
brahma_analysis.Write2File(M,MStars,ZStars,SFR,Sigmas,VelsMagBHs,VelsMagCOMs,VelBHs,VelCOMs,NStars,BH_Progs,BH_Mdots,BHIDs,BHArrayIndex,ZGas,IgnoredBhs,SubhaloArrayIndex,fname='output_ratio10_SFMFGM5_seed5.00_bFOF_LW10_spin_rich_z5')