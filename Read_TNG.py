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

group=99
fields=['SubhaloLenType']
subhalos = il.groupcat.loadSubhalos(TNGpath,group,fields=fields)

SubhaloBHLen = subhalos[:,5]
SubhaloStarsLen = subhalos[:,4]
SubhaloIndices = np.arange(0,len(SubhaloBHLen))
mask = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>0)  # Only subhalos with a BH and with stars
SubhaloIndicesWithStars = SubhaloIndices[mask]
desired_indices = range(len(SubhaloIndicesWithStars))

h = 0.6774 # hubble constant 

# Load in TNG BH properties
fields=['Masses','Velocities']
BHProp = il.snapshot.loadSubset(TNGpath,99,5,fields=fields)

BHMass=BHProp['Masses']
BHVel=BHProp['Velocities']

# Load in TNG Star velocities
fields=['Masses','Velocities']
StarsProp=il.snapshot.loadSubset(TNGpath,99,4,fields=fields)

StarMass=StarsProp['Masses']
StarVel=StarsProp['Velocities']

Sigmas=[]
BHMass=[]
Mstars=[]

output_redshift=0
a = 1/(1+output_redshift)

for index in desired_indices:
    
    ActualSubhaloIndex = SubhaloIndicesWithStars[index]
    
    Vel_Star_subhalo = StarVel[ActualSubhaloIndex]
    M_Stars_subhalo = StarMass[ActualSubhaloIndex]
    
    Vel_BH_subhalo = BHVel[ActualSubhaloIndex]
    M_BH_subhalo = BHMass[ActualSubhaloIndex] # Got an index out of range error
    
    N = len(Vel_Star_subhalo) # number of stars
    local_vel = np.array(Vel_Star_subhalo)/np.sqrt(a) - Vel_BH_subhalo[0]/np.sqrt(a) # New units: km/s    
    
    # Velocity_mag_local = np.linalg.norm(local_vel, axis=1) # Calculate velocity magnitudes
    mu_BH = np.mean(local_vel,axis=0) # Average 3D stellar velocity for this subhalo
    Mstars_total = np.sum(M_Stars_subhalo)
    
    # Here we weight the sigma calculation by stellar mass
    BHDiffSquared=M_Stars_subhalo[:, np.newaxis]*np.array((local_vel - mu_BH)** 2)
    
    Sigma = np.sqrt(np.sum(BHDiffSquared,axis=0) / Mstars_total)  # Calculate sigma from subhalo velocity
    
    Sigmas.append(Sigma)
    BHMass.append(np.max(M_BH_subhalo)*1e10*h)
    Mstars.append(M_Stars_subhalo*1e10*h)
    
brahma_analysis.Write2File(BHMass,Mstars,Sigmas,fname='Brahma_Data/TNG_z0')
