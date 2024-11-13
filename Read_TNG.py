'''
This script is intended to read in data from the TNG simulations
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
# Particle data: snapdir_099
# subfind data: groups_099
# offsets: /standard/torrey-group/IllustrisTNG/Runs/L75n1820TNG/postprocessing/offsets/

TNGpath='/standard/torrey-group/IllustrisTNG/Runs/L75n1820TNG'
Illustrispath='/standard/torrey-group/Illustris/Runs/L75n1820FP'

group=135 # z=0 in Illustris
fields=['SubhaloLenType']
subhalos = il.groupcat.loadSubhalos(Illustrispath,group,fields=fields)

SubhaloBHLen = subhalos[:,5]
SubhaloStarsLen = subhalos[:,4]
SubhaloIndices = np.arange(0,len(SubhaloBHLen))
mask1 = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>10)  # Only subhalos with a BH and with 10 stars
mask2 = SubhaloStarsLen>10
SubhaloIndicesWithStars = SubhaloIndices[mask2]
SubhaloIndicesWithBH = SubhaloIndices[mask1]

SubhaloCM_vel = il.groupcat.loadSubhalos(Illustrispath,group,fields='SubhaloVel')

h = 0.6774 # hubble constant 

output_redshift=0
a = 1/(1+output_redshift)

SigmasCM=[]
SigmasBH=[]
BHMass=[]
MStarsBH=[] # Mass of stars with a BH
MStars=[]
NStars=[]
VelStarsBH=[]
VelStars=[]

bad_indices=np.arange(16940,16944,1) # For some reason these Illustris subhalos won't load in

mask = np.isin(SubhaloIndicesWithStars, bad_indices, invert=True) # Removing these troublesome subhalos from our list manually
SubhaloIndicesWithStars=SubhaloIndicesWithStars[mask]

for index in SubhaloIndicesWithStars:
        
    # We want to read in stellar and BH masses and velocities
    fields = ['Masses','Velocities']
    Star_Props = il.snapshot.loadSubhalo(Illustrispath, snapNum=group, id=index, partType=4, fields=fields)
    
    Star_Mass=Star_Props['Masses']
    Star_Vel=Star_Props['Velocities']
    
    N = len(Star_Vel) # number of stars
    Mstars_total = np.sum(Star_Mass) # Total stellar mass
    
    # Subtract subhalo COM velocity to get local velocities
    local_vel_CM = np.array(Star_Vel)/np.sqrt(a) - SubhaloCM_vel[index]/np.sqrt(a) # New units: km/s 
    
    # Here we weight the sigma calculation by stellar mass
    mu_CM = np.mean(local_vel_CM,axis=0) # Average 3D stellar velocity for this subhalo
    CMDiffSquared=Star_Mass[:, np.newaxis]*np.array((local_vel_CM - mu_CM)** 2)
    
    SigmaCM = np.sqrt(np.sum(CMDiffSquared,axis=0) / Mstars_total)  # Calculate sigma from subhalo velocity
    
    # If our subhalo has a BH, we store its properties as well
    if index in SubhaloIndicesWithBH:
        
        BH_Props = il.snapshot.loadSubhalo(Illustrispath, snapNum=group, id=index, partType=5, fields=fields)

        BH_Mass=BH_Props['Masses']
        BH_Vel=BH_Props['Velocities']
        
        Massive_BH_vel = BH_Vel[np.argmax(BH_Mass)] # Take the most massive BH's vel to subtract stellar vel from
        local_vel_BH = np.array(Star_Vel)/np.sqrt(a) - Massive_BH_vel/np.sqrt(a) # New units: km/s   
    
        mu_BH = np.mean(local_vel_BH,axis=0) # Average 3D stellar velocity for this subhalo
        BHDiffSquared=Star_Mass[:, np.newaxis]*np.array((local_vel_BH - mu_BH)** 2)
    
        SigmaBH = np.sqrt(np.sum(BHDiffSquared,axis=0) / Mstars_total)  # Calculate sigma from subhalo velocity
    
        SigmasBH.append(SigmaBH)
        BHMass.append(np.max(BH_Mass)*1e10*h) # Add most massive BH mass to our list
        MStarsBH.append(Star_Mass*1e10*h)
        VelStarsBH.append(local_vel_BH)
    
    SigmasCM.append(SigmaCM)
    MStars.append(Star_Mass*1e10*h)
    NStars.append(N)
    VelStars.append(local_vel_CM)
    
brahma_analysis.Write2File(BHMass,MStars,MStarsBH,SigmasCM,SigmasBH,NStars,VelStars,VelStarsBH,fname='Brahma_Data/Illustris_z0')
