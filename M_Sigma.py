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
output='output_ratio10_SFMFGM5_seed5.00_bFOF_LW10/' # Name of the box we want to load data from
basePath = path_to_output+run+output # Combining paths to read data in 

file_format='fof_subfind'

desired_redshift=0
h = 0.6774 # hubble constant 

# Initialize all of our data as empty lists
M = []
MStarsBH=[]
MStars=[]
ZStars=[]
SFR=[]
SigmasBH = [] # Sigma for subhalos with a BH and with > 10 stars
SigmaStars = [] # Sigma for subhalos with > 10 stars
VelsMagBHs = []
VelsMagCOMs = []
VelBHs = []
VelCOMs = [] # Stellar velocities with subhalo COM subtracted
NStars = []
BH_Progs=[]
BH_Mdots=[]
BHIDs=[]
ZGas=[]
BHArrayIndex=[]
IgnoredBhs = 0 # Number of black holes in subhalos that our code ignores
    
SubhaloLenType,o = arepo_package.get_subhalo_property(basePath,'SubhaloLenType',desired_redshift,postprocessed=1)
SubhaloBHLen = SubhaloLenType[:,5]
SubhaloStarsLen = SubhaloLenType[:,4]
SubhaloIndices = np.arange(0,len(SubhaloBHLen))
mask1 = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>10)  # Only subhalos with a BH and with stars
mask2 = SubhaloStarsLen>10                # Only subhalos with stars; want another array of values for a Mstar-msigma plot

SubhaloIndicesWithBH = SubhaloIndices[mask1] # Return these so we can cross-reference which subhalos to plot
SubhaloIndicesWithStars = SubhaloIndices[mask2]

desired_indices = range(len(SubhaloIndicesWithStars))

# Get BH IDs so we can store BH array index later, see line 140
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

# Grabbing stellar age; negative ages are wind particles that should not be included in sigma calculation
Age=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='GFM_StellarFormationTime')

# Subhalo properties
SubhaloSFR,o = arepo_package.get_subhalo_property(basePath,'SubhaloSFR',desired_redshift,postprocessed=1)
SubhaloZStars,o = arepo_package.get_subhalo_property(basePath,'SubhaloStarMetallicity',desired_redshift,postprocessed=1)
SubhaloZGas,o = arepo_package.get_subhalo_property(basePath,'SubhaloGasMetallicity',desired_redshift,postprocessed=1)
SubhaloCMvel,o = arepo_package.get_subhalo_property(basePath,'SubhaloVel',desired_redshift,postprocessed=1)

    
# Scale factor calculation for unit corrections
a = 1/(1+output_redshift)
    
for index in desired_indices:
        
    ActualSubhaloIndex = SubhaloIndicesWithStars[index]
    Vel_subhalo,Vel_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'Velocities',4,output_redshift,ActualSubhaloIndex,requested_property1,store_all_offsets=1,group_type='subhalo')
    MStars_subhalo,Mstars_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'Masses',4,output_redshift,ActualSubhaloIndex,requested_property2,store_all_offsets=1,group_type='subhalo')
    Age_subhalo,Age_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'GFM_StellarFormationTime',4,output_redshift,ActualSubhaloIndex,Age,store_all_offsets=1,group_type='subhalo')
    
    # Removing wind particles
    mask = Age_subhalo > 0
    Vel_subhalo = Vel_subhalo[mask]
    MStars_subhalo = MStars_subhalo[mask]
    
    Mstars_total = np.sum(MStars_subhalo)
    
    # Velocity calculations
    N = len(Vel_subhalo) # number of stars

    Vel_CM = np.array(Vel_subhalo)/np.sqrt(a) - SubhaloCMvel[ActualSubhaloIndex]   # New units: km/s
    VelocityMagCM = np.linalg.norm(Vel_CM, axis=1)
    mu_CM = np.mean(Vel_CM,axis=0) # Average 3D stellar velocity for this subhalo
    
    # Here we weight the sigma calculation by stellar mass
    CMDiffSquared=MStars_subhalo[:, np.newaxis]*np.array((Vel_CM - mu_CM)** 2)
    
    Sigma = np.sqrt(np.sum(CMDiffSquared,axis=0) / Mstars_total)    # Calculate sigma from subhalo velocity
    
    # Add all of our current subhalo values to our lists
    MStars.append(MStars_subhalo*1e10*h)
    ZStars.append(SubhaloZStars[ActualSubhaloIndex])
    SFR.append(SubhaloSFR[ActualSubhaloIndex])
    SigmaStars.append(Sigma) 
    VelsMagCOMs.append(VelocityMagCM)
    VelCOMs.append(Vel_CM)
    NStars.append(N)
    ZGas.append(SubhaloZGas[ActualSubhaloIndex])
    
    
    # If the subhalo more than 10 stars but also has a BH
    if ActualSubhaloIndex in SubhaloIndicesWithBH:
        BHMasses_subhalo,BHMasses_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'Masses',5,desired_redshift,ActualSubhaloIndex,requested_property3,store_all_offsets=1,group_type='subhalo')
        BHProgs_subhalo,BHProgs_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'BH_Progs',5,desired_redshift,ActualSubhaloIndex,requested_property4,store_all_offsets=1,group_type='subhalo')
        BHID_subhalo,BHID_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'ParticleIDs',5,desired_redshift,ActualSubhaloIndex,requested_property5,store_all_offsets=1,group_type='subhalo')
        BHMdot_subhalo,BHMdot_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'BH_Mdot',5,desired_redshift,ActualSubhaloIndex,requested_property6,store_all_offsets=1,group_type='subhalo')
        BH_vel_subhalo,BH_vel_subhalo_group,output_redshift=brahma_analysis.get_particle_property_within_postprocessed_groups_adj(basePath,'Velocities',5,desired_redshift,ActualSubhaloIndex,requested_property7,store_all_offsets=1,group_type='subhalo')
        
        Vel_BH = np.array(Vel_subhalo)/np.sqrt(a) - BH_vel_subhalo[0]/np.sqrt(a) # New units: km/s
        VelocityMagBH = np.linalg.norm(Vel_BH, axis=1) # Calculate velocity magnitudes
        mu_BH = np.mean(Vel_BH,axis=0) # Average 3D stellar velocity for this subhalo
        
        BHDiffSquared=MStars_subhalo[:, np.newaxis]*np.array((Vel_BH - mu_BH)** 2)
        SigmaBH = np.sqrt(np.sum(BHDiffSquared,axis=0) / Mstars_total)
        
        # Add out BH properties to our lists
        M.append(np.max(BHMasses_subhalo)*1e10*h)
        SigmasBH.append(SigmaBH)
        MStarsBH.append(MStars_subhalo*1e10*h)
        VelsMagBHs.append(VelocityMagBH)
        VelBHs.append(Vel_BH)
        BH_Progs.append(BHProgs_subhalo[0])
        BH_Mdots.append(BHMdot_subhalo[0])
        BHIDs.append(BHID_subhalo[0])
        BHArrayIndex.append(np.where(AllBHIDs[0] == BHID_subhalo[0])[0][0]) # Store BH array index
        IgnoredBhs += len(BHMasses_subhalo)-1
    
brahma_analysis.Write2File(M,MStarsBH,MStars,ZStars,SFR,SigmasBH,SigmaStars,VelsMagBHs,VelsMagCOMs,VelBHs,VelCOMs,NStars,BH_Progs,BH_Mdots,BHIDs,BHArrayIndex,ZGas,IgnoredBhs,SubhaloIndicesWithStars,SubhaloIndicesWithBH,fname='Brahma_Data/output_ratio10_SFMFGM5_seed5.00_bFOF_nowind_z0')
