import sys
import os
sys.path.append('/home/yja6qa/arepo_package')

import numpy as np
import arepo_package
import math
import matplotlib.pyplot as plt
import illustris_python as il
import cloudpickle
from rotate import *
import h5py
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def mean_trends(Prop1list,Prop2list,redshifts,limits,bins:int):

    '''
    mean_trends is a function designed to take the above data and bin it, returning the mean 
    and standard deviation values for each bin at each redshift in each box
    
    Inputs:
    
    Prop1list,Prop2list: Lists of lists of properties you specified to pull.
                         Dimensions are expected to be (Different boxes, Different redshifts)
    redshifts: Redshifts entered in Prop1list,Prop2list
    limits: limits output from load_data, log10 of x axis min and max values
    bins: Number of bins you want along your x axis
    
    Outputs:
    
    AllBoxMeans: A numpy array of the mean binned y values for each redshift, for each box
    AllBoxStdDevs: A numoy array of the std dev of y values for each redshift, for each box
    XPoints: Points along the x axis to plot your average y values
    '''

    AllBoxMeans = []
    AllBoxStdDevs = []
    Xpoints = []
    Allids = []

    # I have no idea why this is necessary, but when I don't include this the loop breaks because of the num argument
    numbins=bins

    # Want to find mean and std dev for each simulation
    for i in range(len(Prop1list)):

        BoxMeans = []
        BoxStdDevs = []
        Box_ids = [] 
        
        # low = math.floor(limits[0])
        # high = math.ceil(limits[1])
        
        low = limits[0]
        high = limits[1]
        # Add a manual shift of i in log scale to the bins to prevent overlap
        bins = np.log10(np.logspace(low,high,num=numbins))+i*0.025
        
        # Add the bin average to serve as x values when plotting; same for all z's
        Xpoints.append(np.array([np.mean([bins[n],bins[n+1]]) for n in range(0,len(bins)-1)]))

        # For each redshift
        for ii in range(len(redshifts)):

            ZMeans = []
            ZStdDevs = []
            Z_ids = []

            # For each bin we make
            for iii in range(len(bins)-1):

                # Store the ids of the Property1 to calculate the mean and std.dev of Property2
                ids = np.where(np.logical_and(np.log10(Prop1list[i][ii])>=bins[iii],
                                          np.log10(Prop1list[i][ii])<=bins[iii+1]))[0]
                
                Vals = Prop2list[i][ii][ids][np.nonzero(Prop2list[i][ii][ids])]
                
                # If we have at least 5 points in the bin
                if len(Vals) > 5:

                    ZMeans.append(np.mean(np.log10(Vals)))
                    ZStdDevs.append(np.std(np.log10(Vals)))
                    Z_ids.append(ids)

                # Otherwise, skip this bin
                else: 
                    
                    ZMeans.append(np.nan)
                    ZStdDevs.append(np.nan)
                    Z_ids.append(ids)
                    
            BoxMeans.append(ZMeans)
            BoxStdDevs.append(ZStdDevs)
            Box_ids.append(Z_ids)

        AllBoxMeans.append(BoxMeans)
        AllBoxStdDevs.append(BoxStdDevs)
        Allids.append(Box_ids)

    AllBoxMeans = np.array(AllBoxMeans)
    AllBoxStdDevs = np.array(AllBoxStdDevs)

    return(AllBoxMeans,AllBoxStdDevs,Xpoints)


def mean_trends_adj(Prop1list,Prop2list,redshifts,limits,bins:int):

    '''
    Adjusted version of mean_trends to produce bins for each redshift in a box as opposed
    to for each box at a given redshift
    '''
    
    AllBoxMeans = []
    AllBoxStdDevs = []
    Xpoints = []
    Allids = []

    # I have no idea why this is necessary, but when I don't include this the loop breaks because of the num argument
    numbins=bins

    # Want to find mean and std dev for each box
    for i in range(len(Prop1list)):

        BoxMeans = []
        BoxStdDevs = []
        Box_ids = []
        
        # low = math.floor(limits[0])
        # high = math.ceil(limits[1])
        
        low = limits[0]
        high = limits[1]

        # For each redshift
        for ii in range(len(redshifts)):
            
            # Add a manual shift of ii+0.5 in log scale to the bins to prevent overlap
            bins = np.log10(np.logspace(low,high,num=numbins))+ii*0.025 # 0.2 for relations, 0.25 for HMR
            # print(bins)
        
            # Add the bin average to serve as x values when plotting; same for all z's
            Xpoints.append(np.array([np.mean([bins[n],bins[n+1]]) for n in range(0,len(bins)-1)]))

            ZMeans = []
            ZStdDevs = []
            Z_ids = []

            # For each bin we make
            for iii in range(len(bins)-1):

                # Store the ids of the Property1 to calculate the mean and std.dev of Property2
                ids = np.where(np.logical_and(np.log10(Prop1list[i][ii])>=bins[iii],
                                          np.log10(Prop1list[i][ii])<=bins[iii+1]))[0]
                
                Vals = Prop2list[i][ii][ids][np.nonzero(Prop2list[i][ii][ids])]
                
                if len(Vals) > 5:
                    
                    ZMeans.append(np.mean(np.log10(Vals)))
                    ZStdDevs.append(np.std(np.log10(Vals)))
                    Z_ids.append(ids)
                
                else: 
                    
                    ZMeans.append(np.nan)
                    ZStdDevs.append(np.nan)
                    Z_ids.append(ids)

            BoxMeans.append(ZMeans)
            BoxStdDevs.append(ZStdDevs)
            Box_ids.append(Z_ids)

        AllBoxMeans.append(BoxMeans)
        AllBoxStdDevs.append(BoxStdDevs)
        Allids.append(Box_ids)

    AllBoxMeans = np.array(AllBoxMeans)
    AllBoxStdDevs = np.array(AllBoxStdDevs)

    return(AllBoxMeans,AllBoxStdDevs,Xpoints)


def get_particle_property_within_postprocessed_groups_adj(output_path,particle_property,p_type,desired_redshift,subhalo_index,requested_property,group_type='groups',list_all=True,store_all_offsets=1, public_simulation=0,file_format='fof_subfind'):

    
    '''
    This function was directly adapted from Aklant Bohmwick's get_particle_property_within_postprocessed_groups function from his version of arepo_package. This adapted version takes requested_property as an input so as to avoid loading in all the data every time the property of some halo is desired. 
    '''
    output_redshift,output_snapshot=arepo_package.desired_redshift_to_output_redshift(output_path,desired_redshift,list_all=False,file_format=file_format)
#     if (public_simulation==0):
#         requested_property=il.snapshot.loadSubset_groupordered(output_path,output_snapshot,p_type,fields=particle_property)

    if (group_type=='groups'):
        if(public_simulation==0):              
            group_lengths,output_redshift=(arepo_package.get_group_property(output_path,'GroupLenType', desired_redshift,list_all=False,file_format=file_format,stack_style='vstack',postprocessed=1))
            group_lengths=group_lengths[:,p_type] 
            if (store_all_offsets==0):
                    group_offsets=np.array([sum(group_lengths[0:i]) for i in range(0,subhalo_index+1)]) 
            else:
                if (os.path.exists(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot))):
                    group_offsets=np.load(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot),allow_pickle = True)
                    print("offsets were already there")
                else:
                    group_offsets=np.array([sum(group_lengths[0:i]) for i in range(0,len(group_lengths))])
                    np.save(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot),group_offsets)
                    print("Storing the offsets")
            group_particles=requested_property[group_offsets[subhalo_index]:group_offsets[subhalo_index]+group_lengths[subhalo_index]]
        else:
            group_particles=il.snapshot.loadHalo(output_path, output_snapshot, subhalo_index, p_type, fields=particle_property)
        return group_particles,output_redshift

    elif (group_type=='subhalo'):              
        if(public_simulation==0):
            group_lengths,output_redshift=(arepo_package.get_group_property(output_path,'GroupLenType', desired_redshift,postprocessed=1))
            group_lengths=group_lengths[:,p_type] 
            if (store_all_offsets==0):
                    group_offsets=np.array([sum(group_lengths[0:i]) for i in range(0,subhalo_index+1)]) 
            else:
                if (os.path.exists(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot))):
                    group_offsets=np.load(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot),allow_pickle = True)
                    # print("offsets were already there")
                else:
                    group_offsets=np.array([sum(group_lengths[0:i]) for i in range(0,len(group_lengths))])
                    np.save(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot),group_offsets)
                    print("Storing the offsets")        
            subhalo_lengths,output_redshift=(arepo_package.get_subhalo_property(output_path,'SubhaloLenType', desired_redshift, postprocessed=1))
            subhalo_lengths=subhalo_lengths[:,p_type] 
            subhalo_indices=np.arange(0,len(subhalo_lengths))
            subhalo_group_number,output_redshift=(arepo_package.get_subhalo_property(output_path,'SubhaloGrNr', desired_redshift,list_all=False,postprocessed=1));
            desired_group_number=subhalo_group_number[subhalo_index]  
            subhalo_lengths=subhalo_lengths[subhalo_group_number==desired_group_number]
            subhalo_offsets=np.array([sum(subhalo_lengths[0:i]) for i in range(0,len(subhalo_lengths))])
            mask=subhalo_group_number==desired_group_number
            #print(len(mask)),mask
            subhalo_indices=subhalo_indices[mask]  
            subhalo_final_indices=np.arange(0,len(subhalo_indices))
            group_particles=requested_property[group_offsets[desired_group_number]:group_offsets[desired_group_number]+group_lengths[desired_group_number]]   
        
            del requested_property

            #subhalo_indices=subhalo_indices[subhalo_group_number==desired_group_number]
            final_index=(subhalo_final_indices[subhalo_indices==subhalo_index])[0]
        
            subhalo_particles=group_particles[subhalo_offsets[final_index]:subhalo_offsets[final_index]+subhalo_lengths[final_index]]
      
            #return subhalo_particles,group_particles,output_redshift     
        else:
            subhalo_group_number,output_redshift=(arepo_package.get_subhalo_property(output_path,'SubhaloGrNr', desired_redshift,list_all=False,postprocessed=1));
            desired_group_number=subhalo_group_number[subhalo_index]
            group_particles=il.snapshot.loadHalo(output_path, output_snapshot, desired_group_number, p_type, fields=particle_property)
            subhalo_particles=il.snapshot.loadSubhalo(output_path, output_snapshot, subhalo_index, p_type, fields=particle_property)
        return subhalo_particles,group_particles,output_redshift
    else:
        print("Error:Unidentified group type")
        
        

def Write2File(*args,fname='BrahmaData'):
    
    '''
    Write2File is a function designed to take the output of M_Sigma and write the data to a pickle file
    The pickle file format was chosen because of the inhomogeneous structure of the data, since each subhalo
    has a different number of stars
    
    Inputs:
    
    See outputs of M_Sigma function
    fname: Name of file to dump data to
    
    Outputs:
    
    Writes data to file named 'fname.pickle'
    '''
    
    Data=[arg for arg in args]
        
    # Now trying with cloudpickle
    with open(fname+'.pickle', 'wb') as f:
        cloudpickle.dump(Data, f)
        

def ReadBrahmaData(fname='BrahmaData'):
    '''
    ReadBrahmaData is a function designed to read in the data dumped by Write2File
    
    Inputs:
    fname: Name of file to read data from
    
    Outputs:
    Data: List of data from M_Sigma function in the following order: [M,Sigma,VelsMag,Velocities,Nstars,IgnoredBhs]
    See M_Sigma for details
    '''
        
    # Trying with cloudpickle: 
    with open(fname+'.pickle', 'rb') as f:
        Data = cloudpickle.load(f)
        
    return(Data)



def fixed_x(X_vals,Y_vals,fixed_vals,bin_width):

    '''
    fixed_x is a function that takes the traditional scaling relations (like M_BH-sigma) and provides
    y (M_BH) axis averages and std devs. vs. redshift for fixed values of the x axis (sigma).
    
    
    Inputs:
    
    X_vals: List of values typically on the x axis (like sigma) for each redshift desired
            Format: [X_vals_z0, X_vals_z1, ... ]
    Y_vals: List of values typically on the y axis (like M_BH) for each redshift desired
            Format: [Y_vals_z0, Y_vals_z1, ... ]
    fixed_vals: List of values of X_vals that you want to keep constant
    bin_width: Width of bins around fixed_vals to draw from X_vals
    
    Outputs:
    
    avgs: List of averages of X_vals at fixed_vals values
    stds: List of standard deviations around avgs
    
    '''
    
    # Avgs and std devs for all fixed x values 
    avgs = []
    stds = []
    
    # For each fixed value we are interested in
    for i in range(len(fixed_vals)):
        
        # Avgs and std devs for the current fixed x value 
        sigma_avgs = []
        sigma_stds = []
        
        # For each redshift in X_vals
        for ii in range(len(X_vals)):

            # Fetch indices of values within +/- bin_with of fixed_vals
            index = np.logical_and(X_vals[ii] > fixed_vals[i]-bin_width, X_vals[ii] < fixed_vals[i]+bin_width)
            
            # Calculate avg and std dev for y_vals at (redshift) index ii for the current fixed_val
            avg = np.mean(np.array(Y_vals[ii])[index])
            std = np.std(np.array(Y_vals[ii])[index])
                        
            # Append to lists
            sigma_avgs.append(avg)
            sigma_stds.append(std)
        
        avgs.append(sigma_avgs)
        stds.append(sigma_stds)
        
    return(avgs,stds)



def precent_growth(array,array_index,redshift_indices):

    '''
    Simple function to calculate the percent growth of a value across
    different redshifts
    
    Inputs:
    array: Array of values at each redshift for each line
    array_index: Index of array parsing through (typically 0, 1, or 2 for 3 different lines)
    redshift_indices: Indices of redshifts to calculate percent growth for. Format: (high_z index,low_z index)
    
    Outputs:
    vals: List of [highzval,lowzval]
    perc_growth: Percent growth from high_z index to low_z index
    '''
    
    highzval = array[array_index][redshift_indices[0]]
    lowzval = array[array_index][redshift_indices[1]]
    
    perc_growth = (lowzval - highzval)/highzval
    
    vals = [highzval,lowzval]
    
    return(vals,perc_growth)
    



def Center_subhalo(ParticleProps,Subhaloprops,box_size,redshift,h,subhalo_id):

    '''
    Center_subhalo is designed to center the coordinates and velocities of star
    particles on the subhalo, as well as correct units from internal units to physical units
    
    Inputs:
    ParticleProps: Output from il.snapshot.loadSubhalo for stars 
                   Assumed to have coordinate, velocity, mass, and potential fields
    Subhaloprops: Output from il.groupcat.loadSubhalos for subhalo data
                  Assumed to have coordinate and velocity fields
    box_size: Boxsize of the sim as read in from the header via
    
              hdr  = il.groupcat.loadHeader(basePath, snap_num)
              box_size = hdr["BoxSize"]
              
    redshift: Redshift of the current snapshot, can also be read in from header
    h: Hubble parameter for the simulation, also read in from header
    subhalo_id: Index of the subhalo
    
    Outputs:
    Coordinates: Coordinates of stars in km, centered on subhalo center
    Velocities: Velocities of stars in km/s, centered on subhalo center
    Potentials: Potential of stars in (km/s)^2
    '''
    
    a = 1/(1+redshift)
    
    Coordinates = ParticleProps['Coordinates']
    Velocities = ParticleProps['Velocities']
    Masses = ParticleProps['Masses']
    Potentials = ParticleProps['Potential']
    
    Subhalo_Pos = Subhaloprops['SubhaloPos'][subhalo_id]
    Subhalo_Vel = Subhaloprops['SubhaloVel'][subhalo_id]
    
    Coordinates = center(Coordinates,Subhalo_Pos,box_size)
    Velocities = Velocities - Subhalo_Vel

    # Correcting units, scale factor = 1 for z = 0
    kpc2km = 3.0857e16 # Conversion rate from kpc to km
    
    Coordinates *= a/h # New units: kpc
    Coordinates *= kpc2km # New units: km
    Velocities *= np.sqrt(a) # New units: km/s
    Potentials /= a # New units: (km/s)^2

    ri   = 0 * kpc2km  # from 0
    ro   = 20 * kpc2km # to 20 kpc
    incl = calc_incl(Coordinates, Velocities, Masses, ri, ro) # rotate based on stars

    Coordinates = trans(Coordinates, incl)
    Velocities = trans(Velocities, incl)

    return(Coordinates,Velocities,Potentials)
    


def kinematic_decomp_r(Coordinates,Velocities,Potentials,nbins=500,nstars_min=1000):

    '''
    kinematic_decomp_r does a decomposition of subhalos based on the energetics of the stars.
    Stars with a lower specific angular momentum than 50% of the angular momentum for a star
    on a circular orbit at its position are classified as belonging to the spheroid, whereas
    those with grater than 50% are classified as belonging to the disk.
    
    This was an initial attempt at doing a decomposition that left many stars from the disk
    contaminating what was classified as the bulge component. Hence, the next function was developed.
    
    Inputs:
    Coordinates: Array containing the coordinates of stars, assumed to be in km
    Velocities:  Array containing the velocities of stars, assumed to be in km/s
    Potentials:  Array containing the potentials of stars, assumed to be in (km/s)^2
                 Coordinates and velocities are assumed to be centered on the subhalo
    nbins: Number of radial bins to make along the disk radially when calculating stellar potentials
    nstars_min: Minimum number of stars required in a subhalo to do the decomposition
    
    Outputs:
    pos: Radial positions at which the gradients are given
    grad: Gravitational potential gradient at the radial positions 
    ratio: Ratio of j_z to j_circ for each star given its radius
    negids: ids of stellar angular momentums that were set to np.nan
    '''
    
    # Only do decomposition if there are at least nstars_min stars
    if len(Coordinates)<nstars_min:
        return
    
    kpc2km = 3.0857e16 # Conversion rate from kpc to km
    # radial distance from subhalo center in the xy plane
    r = np.sqrt(Coordinates[:,0]**2 + Coordinates[:,1]**2)
    
    height = 3 * kpc2km # kpc for height of disk
    ri   = 0 * kpc2km  # from 0
    ro   = np.max(r) # to the max disk size of the subhalo
    n = 30 # Number of stars required per bin 
    
    bins = overlapping_bins(ri,ro,nbins,dx=0.5)

    # Only stars within the height of the disk
    disk_mask = (Coordinates[:,2] > -height) & (Coordinates[:,2] < height)
    disk_coords = Coordinates[disk_mask]
    disk_pot = Potentials[disk_mask]
    disk_r = r[disk_mask]    

    # Potentials at each radial bin
    potential_binned = np.zeros(shape=len(bins))

    for i in range(len(bins)):

        # Mask of stars within the current radial bin
        r_mask = (disk_r > bins[i][0]) & (disk_r < bins[i][1])

        # Coordinates, potentials of stars in current bin
        r_bin = disk_coords[r_mask]
        r_pot = disk_pot[r_mask]

        # Require at least n stars in the radial bin to consider the radial potential well-defined
        if len(r_bin) < n:
            potential_binned[i] = np.nan

        # Otherwise, take the average of the potentials in the bin
        else:

            # Calculate mean potential
            potential = np.mean(r_pot)

            # Append to list
            potential_binned[i] = potential
        
    # Removing nan values
    no_nans = ~np.isnan(potential_binned)
    potential_binned = potential_binned[no_nans]
            
    # Positions in the middle of the bins
    pos = np.array([np.mean(bins[n]) for n in range(len(bins))])
    
    # Removing nan potential indices from position
    pos = pos[no_nans]
    
    # Calculating the gradient based on positions and potentials
    grad = np.gradient(potential_binned,pos)
        
    # Interpolating the gradient function with scipy 
    gradient_interp = interp1d(pos, grad, kind='linear', fill_value="extrapolate")
    
    # Calculate interpolated gradients
    grad_phi_interp = np.array(gradient_interp(r))
    
    # Find ids of negative potential gradients
    negids = grad_phi_interp < 0
    
    # Set negative potential gradients to np.nan
    grad_phi_interp[negids] = np.nan
    
    # Calculate circular angular momentum
    v_circ = np.sqrt(r * grad_phi_interp)
    j_circ = r * v_circ
    
    # Calculate actual angular momentum
    j_z = np.cross(Coordinates,Velocities)[:,2]
    
    # Take the ratio of the two
    ratio = j_z/j_circ
    
    # Return the radial positions, gradients, and ratio of the angular momentums to the specific angular momentums
    return(pos,grad,ratio,negids)
    



def kinematic_decomp_e(Coordinates,Velocities,Potentials,nstars=150,nstars_min=1000):

    '''
    kinematic_decomp_e does a decomposition of a subhalo based on the energetics of the stars.
    Stars with a lower specific angular momentum than 50% of the angular momentum for a star
    on a circular orbit with its specific binding energy are classified as belonging to the 
    spheroid, whereas those with greater than 50% are classified as belonging to the disk.
    
    Inputs:
    Coordinates: Array containing the coordinates of stars, assumed to be in km
    Velocities:  Array containing the velocities of stars, assumed to be in km/s
    Potentials:  Array containing the potentials of stars, assumed to be in (km/s)^2
                 Coordinates and velocities are assumed to be centered on the subhalo
    nstars: Number of stars per bin for calculating radially binned stellar potentials
    nstars_min: Minimum number of stars required in a subhalo to do the decomposition
    
    Outputs:
    ratio: Ratio of j_z to j_circ for each star given its specific binding energy
    negids: ids of stellar angular momentums that were set to np.nan
    rcs: Calculated circular radii of stars
    '''
    
    # Only do decomposition if there are at least nstars_min stars
    if len(Coordinates)<nstars_min:
        return(np.nan)

    # Removing linear potential gradient
    corrected_potential = remove_linear_gradient(Coordinates,Potentials)
    
    kpc2km = 3.0857e16 # Conversion rate from kpc to km
    # radial distance from subhalo center in the xy plane
    r = np.sqrt(Coordinates[:,0]**2 + Coordinates[:,1]**2)
    
    height = 3 * kpc2km # kpc for height of disk
    ri   = 0 * kpc2km  # from 0
    ro   = np.percentile(r, 97.5) # 97.5th percentile of stars
    
    disk_mask = (Coordinates[:,2] > -height) & (Coordinates[:,2] < height)
    disk_coords = Coordinates[disk_mask]
    disk_pot = corrected_potential[disk_mask]
    disk_r = r[disk_mask] 

    # Only care about stars within ro
    mask = disk_r < ro
    rstars_masked = disk_r[mask]
    potentials_masked = disk_pot[mask]

    # Generating bin centers and averages
    bin_centers,bin_averages = equal_num_bins(rstars_masked,potentials_masked,N=nstars)

    # Smoothing average values and calculating gradient
    window=10
    if len(bin_averages) < window: # If less than 10 points, just interpolate based on however many points there are
        potental_interp = interp1d(bin_centers, bin_averages, kind='linear', fill_value="extrapolate")
    else:
        smoothed_p = savgol_filter(bin_averages, window_length=window, polyorder=1)
        potental_interp = interp1d(bin_centers, smoothed_p, kind='linear', fill_value="extrapolate")
    
    xvals = np.linspace(0,ro,150)
    yvals = np.array([potental_interp(i) for i in xvals])
    
    grad = np.gradient(yvals,xvals)
    smoothed_grad= savgol_filter(grad, window_length=window, polyorder=1)
    gradient_interp = interp1d(xvals, smoothed_grad, kind='linear', fill_value="extrapolate")

    # Defining new function for root finder to calculate rc
    def f(r,args): # args: [stellar specific binding energy]
        val = potental_interp(r) + 0.5*r*np.max([0,gradient_interp(r)]) - args[0] 
        return(val)
    
    rcs = []
    skipped_stars = 0
    
    # Calculating circular radii for all stars given their binding energies e
    for i in range(len(corrected_potential)):
        args = [corrected_potential[i] + 0.5*np.linalg.norm(Velocities[i])**2]
        try:
            a = 0
            b = 2*np.max(r)
            rc = brentq(f,a,b,args=args)
            rcs.append(rc)

        # Inevitably, not all stars will have solutions:
        except Exception as Ex:
            
            skipped_stars+=1
            rcs.append(np.nan)
            
    print("Nonzero rcs:",len(np.array(rcs)[~np.isnan(rcs)]), "Skipped stars: {}".format(skipped_stars))
    
    # Calculate interpolated gradients at rc
    grad_phi_interp = np.array(gradient_interp(rcs))
    
    # Find ids of negative potential gradients
    negids = grad_phi_interp < 0
    
    # Set negative potential gradients to np.nan
    grad_phi_interp[negids] = np.nan
    
    # Calculate circular angular momentum
    v_circ = np.sqrt(rcs * grad_phi_interp)
    j_circ = rcs * v_circ
    
    # Calculate actual angular momentum
    j_z = np.cross(Coordinates,Velocities)[:,2]
    
    # Take the ratio of the two
    ratio = j_z/j_circ
    
    return(ratio,negids,rcs)



def cal_avg(xvals,yvals,bins):
    
    '''
    cal_avg is like mean_trends, but simplified for only one set of x and y values
    
    xvals and yvals in normal values, bins in log10
    '''
    
    # Getting rid of zero vals
    nonzeroy = yvals != 0
    yvals=yvals[nonzeroy]
    
    # Initialize empty lists
    Means = []
    StdDevs = []
    
    # Store average values of bins for plotting
    xpoints = np.array([np.mean([bins[n],bins[n+1]]) for n in range(0,len(bins)-1)])
    
    for i in range(len(bins)-1):
        
        # Store the ids of the Property1 to calculate the mean and std.dev of Property2
        ids = np.where(np.logical_and(np.log10(xvals)>=bins[i],np.log10(xvals)<=bins[i+1]))[0]
        Vals = yvals[ids]
        Means.append(np.mean(np.log10(Vals)))
        StdDevs.append(np.std(np.log10(Vals)))
        
    return(Means,StdDevs,xpoints)



def overlapping_bins(start,end,nbins,dx=0.5):
    
    '''
    overlapping_bins is pretty straightforward: It makes overlapping bins
    This wasn't the best at reducing noise in my binning, so ended up 
    using the next function instead
    
    Inputs:
    start: Where to begin binning
    end: Where to end binning
    nbins: Number of bins desired
    dx: Fraction of a full step l to be taken when constructing next bin
        For non-overlapping bins, this would be 1
        For dx > 1, you will miss regions within your given range
        1 - dx gives the proportion of overlap between subsequent bins
    
    Output:
    bins: Overlapping bins as an array with bin edges in tuples
    '''
    
    if dx > 1:
        print("Warning: Bins will not cover full range")
    
    # Bin length
    l = (end-start)/(1 + (nbins - 1)*dx)
    
    # Initialize empty list
    bins = []
    
    # Initialize start value for bins
    bin_start=start
    
    for i in range(nbins):
        bin_end = bin_start + l # Value for current bin to end at
        bin_i = (bin_start,bin_end) # Defining current bin
        bins.append(bin_i)
        bin_start += dx*l # Increase bin_start for next bin
        
    return(bins)



def equal_num_bins(r,vals,N=150):

    '''
    equal_num_bins creates bins of a value (for my use, potentials or potential gradients)
    with an equal number of objects (stars) in each bin, then returns the average value 
    and position for each bin
    
    r: Radii
    vals: Potentials
    N: number of stars per bin
    '''
    
    indices=np.argsort(r)
    r_sorted = r[indices]
    vals_sorted = vals[indices]
        
    n = int(len(r)/N) # Number of bins
    print("Number of stars per bin:",N)
    
    bin_centers = []
    bin_medians = []
    
    # Loop through bins and calculate the average potential and the center of distances in each bin
    for i in range(n):
        start = i * N
        end = (i + 1) * N
        # Extract the bin slice for both distances and vals
        bin_distances = r_sorted[start:end]
        bin_vals = vals_sorted[start:end]
        # Compute the average (mean) for the bin
        bin_center = np.mean(bin_distances)
        bin_median = np.median(bin_vals)
        bin_centers.append(bin_center)
        bin_medians.append(bin_median)

    return(bin_centers,bin_medians)



def remove_linear_gradient(Coordinates,Potentials):

    '''
    remove_linear_gradient removes any linear trend in the potentials of stars
    
    Inputs:
    Coordinates: Coordinates of the stars of the subhalo, in km
    Potentials: Potentials of the stars of the subhalo, in (km/s)^2
    
    Outputs:
    corrected_potential: Potentials with the linear component subtracted off
    '''

    kpc2km = 3.0857e16
    
    def model(xy, a, b, c):
        x, y = xy
        return a * x + b * y + c

    # Only select stars at large radii to avoid fitting the linear component to the central potential well
    rstars = np.sqrt(Coordinates[:,0]**2 + Coordinates[:,1]**2)
    
    r_out_mask = rstars > np.percentile(rstars, 95) # Stars outside of 95th percentile
    r_out = rstars[r_out_mask]

    # Some small subhalos will have very few stars
    if len(r_out) < 3:
        return(Potentials)
    
    # Get the coordinates of stars at greater than the 95th percentile of stellar radii
    x = Coordinates[:,0][r_out_mask]
    y = Coordinates[:,1][r_out_mask]

    Potentials_masked = Potentials[r_out_mask]
    
    params, covariance = curve_fit(model, (x, y), Potentials_masked)

    fit_potential = model((Coordinates[:,0], Coordinates[:,1]), *params)

    # Subtract off the linear potential while keeping the normalization
    corrected_potential = Potentials - fit_potential + params[2]
    
    return(corrected_potential)
