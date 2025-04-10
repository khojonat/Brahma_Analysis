'''
This script was written to apply the kinematic decomposition to Illustris and IllustrisTNG
and store the BH masses and bulge stellar masses and stellar velocity dispersions
'''

import sys
from brahma_analysis_clean import *
sys.path.append('/home/yja6qa/arepo_package/')

import arepo_package
import h5py
import os
import numpy as np

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

TNGpath='/standard/torrey-group/IllustrisTNG/Runs/L75n1820TNG'
Illustrispath='/standard/torrey-group/Illustris/Runs/L75n1820FP'

basePath = Illustrispath
snap_num=135 # 135 for z=0 and 49 for z=5 in Illustris, 99 for z=0 and 17 for z=5 in TNG
hdr  = il.groupcat.loadHeader(basePath, snap_num)
box_size = hdr["BoxSize"]
redshift = hdr['Redshift']

if basePath==TNGpath:
    h = hdr['HubbleParam']
else:
    h = 0.704 # Hubble param for Illustris, not in header for some reason

print("Making subhalo mask ...", flush=True)
# Determining size of subhalos
fields=['SubhaloLenType']
Subhalo_lengths = il.groupcat.loadSubhalos(basePath,snap_num,fields=fields)

SubhaloBHLen = Subhalo_lengths[:,5] # Number of BHs in each subhalo
SubhaloStarsLen = Subhalo_lengths[:,4] # Number of Stars in each subhalo
SubhaloIndices = np.arange(0,len(SubhaloBHLen)) # Indices of subhalos
mask1 = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>1000)  # Only subhalos with a BH and with 1000 stars
Desired_subhalos = SubhaloIndices[mask1] # Only indices of subhalos we want

print(f'Looping through {len(Desired_subhalos)} subhalos',flush=True)

print('Loading in subhalo data ...', flush=True)
# Load in all subhalo positions and velocities 
Subhaloprops = il.groupcat.loadSubhalos(basePath,snap_num,fields=['SubhaloPos','SubhaloVel','SubhaloHalfmassRad'])
Centrals = il.groupcat.loadHalos(basePath=basePath,snapNum=snap_num,fields='GroupFirstSub')
Central_subhalos = Centrals[Centrals!=-1]

# Initialize lists to append to
Ratios = []
Sigmas = []
BH_Masses = []
Star_Masses = []
Coords = []
Central_ids = []
Subhalo_vels = []
failed_subhalos = 0
skipped_subhalos = 0
halosskipped = 0

print('Looping through desired subhalos ...', flush=True)

# Now looping through all subhalos with BHs and 1000 stars
for index in Desired_subhalos:

    try: 
    
        print(f'Subhalo {index}',flush=True)
        
        HMR = Subhaloprops['SubhaloHalfmassRad'][index]
            
        fields = ['BH_Mass']
        Subhalo_BH_Masses = il.snapshot.loadSubhalo(basePath, snapNum=snap_num, id=index, partType=5, fields=fields)
    
        # Load in star properties of current halo
        fields = ['Masses','Coordinates','Velocities','Potential','GFM_StellarFormationTime']
        Star_Props = il.snapshot.loadSubhalo(basePath, snap_num, id=index, partType=4, fields=fields)
        Star_Mass=Star_Props['Masses']*1e10/h # Units: Msun
    
        # Mask out wind particles
        mask = Star_Props['GFM_StellarFormationTime'] > 0
    
        # Center coord and vel and correct units
        Coordinates,Velocities,Potentials = Center_subhalo(Star_Props,Subhaloprops,box_size,redshift,h,subhalo_id=index)
        Coordinates = Coordinates[mask]
        Velocities = Velocities[mask]
        Potentials = Potentials[mask]
        Star_Mass = Star_Mass[mask]
        
        # Calculate id's of stars in the bulge
        if len(Coordinates) > 1000:
            Vals = kinematic_decomp_e(Coordinates,Velocities,Potentials)
        # Otherwise, set the number of stars per bin to be ~1/20 the total number of stars, to make ~20 bins
        else:
            Vals = kinematic_decomp_e(Coordinates,Velocities,Potentials,nstars=int(len(Coordinates)/20))
    
        if type(Vals)==float: # Some subhalos still have less stars than nstars_min apparently...
            skipped_subhalos+=1
            print('Nan vals',flush=True)
            continue
        else:
            ratio,negids,rcs = Vals[0],Vals[1],Vals[2]
            
        Velocities[negids] = np.nan
    
        # Define bulge stars as those stars with jz/jcirc < 0.5
        bulge = ratio < 0.5
    
        Bulge_vel = Velocities[bulge]
        Bulge_mass = Star_Mass[bulge]
        Bulge_mass = Bulge_mass.reshape(len(Bulge_mass),1)
    
        # Calculate the velocity dispersion
        Mbulge_total = np.sum(Bulge_mass) # Total stellar mass
    
        # Here we weight the sigma calculation by stellar mass
        mu_vel = np.sum(Bulge_mass * Bulge_vel,axis=0) / Mbulge_total
        DiffSquared=Bulge_mass*np.array((Bulge_vel - mu_vel)** 2)
    
        Sigma_bulge = np.sqrt(np.sum(DiffSquared,axis=0) / Mbulge_total)  # Calculate sigma from subhalo velocity
    
        Sigmas.append(Sigma_bulge)
        BH_Masses.append(np.max(Subhalo_BH_Masses)*1e10*h) # Add most massive BH mass in subhalo to list
        Ratios.append(ratio) # Append the ratio of jz/jcirc for stars in the subhalo
        Coords.append(Coordinates)
        Star_Masses.append(np.sum(Star_Mass)*1e10*h)
        Subhalo_vels.append(Velocities)
        
        if index in Central_subhalos:
            Central_ids.append(index)
        else:
            Central_ids.append(-1) # A value of -1 indicates the the subhalo is not a central
    
        if len(ratio[~np.isnan(ratio)]) == 0:
            failed_subhalos += 1
            print("Subhalo {} failed".format(index), flush=True)
        else:
            print('Sigma: {},'.format(np.linalg.norm(Sigma_bulge)),'BH mass: {},'.format(np.max(Subhalo_BH_Masses)),
                  'Ratio max/min: {},'.format((np.max(ratio[~np.isnan(ratio)]),np.min(ratio[~np.isnan(ratio)])) ) , flush=True)

    except Exception:
        halosskipped+=1

failure_rate = failed_subhalos/len(Desired_subhalos)
print("Failure rate: {}".format(failure_rate),flush=True)
print(f"Nan values: {skipped_subhalos}",flush=True)
print(f"(Potentially) Missing files: {halosskipped}",flush=True)

Write2File(Ratios,Sigmas,BH_Masses,Coords,Star_Masses,Central_ids,Subhalo_vels,fname='Brahma_Data/Kin_Decomp_Ill_z0')


