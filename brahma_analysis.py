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
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

'''
load_data is a function designed to load in BRAHMA data and store the desired data in a list of lists (of lists).
The lists are stacked according to simulation box and redshift.

Inputs:

path_to_output: Path to run
run: Name of simulation runs
Outputlist: Simulation boxes you want to analyze
redshifts: Redshifts you want to look at 
Property1, Property2: the particle properties to load
part_type: Type of particle to analyze (dark matter=0, gas=1, star=4, black hole=5)
conversion1,conversion2: Conversion from internal units to physical units for properties 1 and 2 respectively
Lbol: Whether we want to convert to Bolometric Luminosity. This only applied if reading in BH Mdot

Outputs:

Prop1list, Prop2list: Lists of lists of properties you specified to pull
outputzlist: the actual redshifts of the snapshots taken from BRAHMA
limits: the minimum and maximum values of Property1 (assumed to be x axis) across the entire dataset in log10(physical units). 

The limits will be useful for binning to make mean trends
'''

# Defaults for the conversions are 1e10*h, where h is 0.6774, to convert to Solar Masses
def load_data(path_to_output,run,outputlist,redshifts,Property1,Property2,part_type,
              conversion1=1e10*0.6774,conversion2=1e10*0.6774,Lbol=[False,False]):

    basePaths = []
    for output in outputlist:
        basePaths.append(path_to_output+run+output)

    # Initiating lists to store data from all boxes
    # Not using np arrays because the size of each dimension can be different for each box
    Prop1list = []
    Prop2list = []
    outputz_list = []
    minx = []
    maxx = []
    
    # Some constants to be used later if converting to Lbol: Lbol = e_r * Mdot * c^2
    e_r = 0.2 # Radiative efficiency assumed for BH accretion
    c = 3e10 # cm/s
    Lbolconv = 6.3e16 # From Mdot/Gyr*(cm/s)^2 to erg/s

    for basePath in basePaths:
    
        # List for all z's for the current Box
        BoxProp1 = []
        BoxProp2 = []
        Boxoutputz = []
        Boxminx = []
        Boxmaxx = []

        for z in redshifts:

            #Reading Prop1 for each type, dark matter=0, gas=1, star=4, black hole=5 into an np.array
            GroupProp1,output_redshift=arepo_package.get_group_property(basePath,Property1,z,postprocessed=1)

            #Reading Prop2 of the halo into an np.array
            GroupProp2,output_redshift=arepo_package.get_group_property(basePath,Property2,z,postprocessed=1)
            
            if Lbol[0]:
                
                # Selecting only the particle type that we're interested in
                # Converting to datatype double, otherwise we will overflow
                Groupdata=GroupProp1[:,part_type].astype(np.double) * conversion1 * e_r * Lbolconv * c**2
                BoxProp1.append(Groupdata)
                BoxProp2.append(GroupProp2*conversion2)
                Boxoutputz.append(output_redshift)
                
            elif Lbol[1]:
                
                GroupProp2 = GroupProp2.astype(np.double)
                Groupdata=GroupProp1[:,part_type] * conversion1
                BoxProp1.append(Groupdata)
                BoxProp2.append(GroupProp2 * conversion2 * e_r * Lbolconv * c**2)
                Boxoutputz.append(output_redshift)
                
            else:
                
                Groupdata=GroupProp1[:,part_type] * conversion1
                BoxProp1.append(Groupdata)
                BoxProp2.append(GroupProp2*conversion2)
                Boxoutputz.append(output_redshift)


            # Nonzero ids for our properties
            nonzero1id = set(np.nonzero(Groupdata)[0])
            nonzero2id = set(np.nonzero(GroupProp2)[0])
            
            # Finding these max and min x are going to help us set our x and y lims for plotting
            Boxminx.append(np.min( np.log10(np.array(Groupdata[list(nonzero1id & nonzero2id)]) ) ) )
            Boxmaxx.append(np.max( np.log10(np.array(Groupdata[list(nonzero1id & nonzero2id)]) ) ) )

        Prop1list.append(BoxProp1)
        Prop2list.append(BoxProp2)
        outputz_list.append(Boxoutputz)
        minx.append(np.min(Boxminx))
        maxx.append(np.max(Boxmaxx))

    minx = np.min(minx)
    maxx = np.max(maxx)
    limits = [minx,maxx]

    return(Prop1list,Prop2list,outputz_list,limits)




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

def mean_trends(Prop1list,Prop2list,redshifts,limits,bins:int):

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
                    
#                     print("Plotting!\n")
#                     if i==0:
                        
#                         print("Simulation: Illustris","Redshift:",redshifts[ii%2],"Sigma:",Xpoints[i][iii],"Num of points:",len(Vals))
#                     else:
#                         print("Simulation: TNG","Redshift:",redshifts[ii%2],"Sigma:",Xpoints[i][iii],"Num of points:",len(Vals))
                
                # Otherwise, skip this bin
                else: 
                    
                    ZMeans.append(np.nan)
                    ZStdDevs.append(np.nan)
                    Z_ids.append(ids)
                    
#                     print("Not plotting!\n")
#                     if i==0:
                        
#                         print("Simulation: Illustris","Redshift:",redshifts[ii%2],"Sigma:",Xpoints[i][iii],"Num of points:",len(Vals))
#                     else:
#                         print("Simulation: TNG","Redshift:",redshifts[ii%2],"Sigma:",Xpoints[i][iii],"Num of points:",len(Vals))

            BoxMeans.append(ZMeans)
            BoxStdDevs.append(ZStdDevs)
            Box_ids.append(Z_ids)

        AllBoxMeans.append(BoxMeans)
        AllBoxStdDevs.append(BoxStdDevs)
        Allids.append(Box_ids)

    AllBoxMeans = np.array(AllBoxMeans)
    AllBoxStdDevs = np.array(AllBoxStdDevs)

    return(AllBoxMeans,AllBoxStdDevs,Xpoints)


'''
Adjusted version of mean_trends to produce bins for each redshift in a box as opposed
to for each box at a given redshift
'''

def mean_trends_adj(Prop1list,Prop2list,redshifts,limits,bins:int):

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


'''
plot_brahma is a function designed to plot the data as binned and formatted by mean_trends()
It will produce a (1,n) length plot where n is the number of redshift you are considering
This is obviously not ideal for high n, so hopefully you don't want to look at 20 redshifts

Inputs:

AllBoxMeans: A numpy array of the mean binned y values for each redshift, for each box
AllBoxStdDevs: A numoy array of the std dev of y values for each redshift, for each box
XPoints: Points along the x axis to plot your average y values
redshifts: Redshifts you want to look at 
legendnames: Names for labels for the plots in the legend
axislabels: Names for labels of the x, and y axis respectively. Assumed to be a list [Xname,Yname]
savefig: Name of image to save, or don't save an image by default

Outputs:

This function should produce a plot of the input data.
By default, no figure is saved. To save a figure, set savefig equal to the name of the resulting image file
'''

def plot_brahma(AllBoxMeans,AllBoxStdDevs,XPoints,redshifts,legend_names,axislabels,savefig = False):
    
    tick_size=15
    label_font_size=25

    f,axes = plt.subplots(1,len(redshifts),figsize=(15,5),sharey=True,sharex=True)

    for i in range(AllBoxMeans.shape[0]):
        for ii in range(len(redshifts)):
        
            # Select only the values that are non nan or inf
            ids = np.logical_and(~np.isnan(AllBoxMeans[i,ii,:]),~np.isinf(AllBoxMeans[i,ii,:]))
        
            if ii == 0:
                # There is probably a better solution to this, but I don't want to duplicate labels
                axes[ii].errorbar(XPoints[i][ids],AllBoxMeans[i,ii,:][ids],
                                  yerr = AllBoxStdDevs[i,ii,:][ids],label=legend_names[i])
            else:
                axes[ii].errorbar(XPoints[i][ids],AllBoxMeans[i,ii,:][ids],yerr = AllBoxStdDevs[i,ii,:][ids])
    n = 0
    
    for ax in axes.flat:

        ax.grid(alpha = 0.5)
        ax.tick_params(labelsize=tick_size)
        ax.set_xlim(np.min(XPoints)-0.5,np.max(XPoints)+0.5)
        ax.set_title('Z={}'.format(redshifts[n]),size = 25)
        n+=1

    # f.legend(fontsize = 12,loc=[0.575,0.65])
    f.supxlabel('{}'.format(axislabels[0]),fontsize=label_font_size)
    f.supylabel('{}'.format(axislabels[1]),fontsize=label_font_size,x=0)

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    
    if savefig != False:
        plt.savefig(savefig)
        
    return(f,axes)




'''
get_group_particle_data is designed to grab some desired property of a subhalo of a 
specified index. Only subhalos with more than 100 stars are considered.

Inputs:

basePath: Path to data file
desired_redshift: Redshifts you want to look at 
p_type: Type of particle to analyze (dark matter=0, gas=1, star=4, black hole=5)
desired_index: Index of the subhalo
particle_property: Property to grab: https://www.tng-project.org/data/docs/specifications/#sec2b

Outputs:

Prop_subhalo: Desired property of the subhalo specified
Prop_group: Desired property for the group containing the specified subhalo
output_redshift: The actual redshift of the snapshots taken from BRAHMA
'''

def get_group_particle_data(basePath,desired_redshift,p_type,desired_index,particle_property):

    #----------------Get the subhalo catalog and select only those subhalos that have enough stars
    SubhaloLenType,o = arepo_package.get_subhalo_property(basePath,'SubhaloLenType',desired_redshift,postprocessed=1)
    SubhaloSMLen = SubhaloLenType[:,p_type]
    SubhaloIndices = np.arange(0,len(SubhaloSMLen))
    mask= SubhaloSMLen>100
    SubhaloIndicesWithStars = SubhaloIndices[mask]

    #---------------Assign the true index of the subhalo to be selected 
    ActualSubhaloIndex = SubhaloIndicesWithStars[desired_index]

    #---------------Retrieve the desired property of the subhalo and the parent group
    Prop_subhalo,Prop_group,output_redshift=arepo_package.get_particle_property_within_postprocessed_groups(basePath,particle_property,p_type,desired_redshift,ActualSubhaloIndex,store_all_offsets=1,group_type='subhalo')
    
    return(Prop_subhalo,Prop_group,output_redshift)



'''
M_Sigma uses get_group_particle_data to read in the central MBH mass and stellar subhalo 
velocity dispersion for specified subhalo indices at a specified redshift. The central
MBH is assumed to be the most massive BH in the subhalo

Inputs:
basePath: Path to data file
desired_redshift: Redshifts you want to look at 
desired_indices: Indices of subhalos you want to look at
file_format: Format of file being pointed to by basePath

Outputs:
M: Array of central massive black hole masses in the specified subhalo, units: 10^10 Msun/h
MStars: Array of stellar masses for each subhalo, units: 10^10 Msun/h
Sigma: 3D Velocity dispersion of the stars in the specified subhalo, units: km/s
VelsMag: The magnitude of each star's velocity for each subhalo, units: km/s
Velocities: The 3D velocity components of each star for each subhalo, units: km/s
NStars: The number of stars in each subhalo
BH_Progs: The number of progenitor BHs for each central BH (or number of past mergers)
IgnoredBhs: The total number of BHs in subhalos that are not represented in the M array
'''

def M_Sigma(basePath,desired_redshift,desired_indices,file_format='fof_subfind',h = 0.6774):
    
    M = []
    MStars=[]
    Sigma = []
    VelsMag = []
    Velocities = []
    NStars = []
    BH_Progs=[]
    IgnoredBhs = 0
    
    SubhaloLenType,o = arepo_package.get_subhalo_property(basePath,'SubhaloLenType',desired_redshift,postprocessed=1)
    SubhaloBHLen = SubhaloLenType[:,5]
    SubhaloStarsLen = SubhaloLenType[:,4]
    SubhaloIndices = np.arange(0,len(SubhaloBHLen))
    mask = np.logical_and(SubhaloBHLen>0,SubhaloStarsLen>0)  # Only subhalos with a BH and with stars
    SubhaloIndicesWithStars = SubhaloIndices[mask]
    
    # From get_particle_property_within_postprocessed_groups
    output_redshift,output_snapshot=arepo_package.desired_redshift_to_output_redshift(basePath,
                                                                        desired_redshift,list_all=False,file_format=file_format)
    requested_property1=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Velocities')
    requested_property2=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=4,fields='Masses')
    requested_property3=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=5,fields='Masses')
    requested_property4=il.snapshot.loadSubset_groupordered(basePath,output_snapshot,partType=5,fields='BH_Progs')
    
    # Scale factor calculation
    a = 1/(1+output_redshift)
    
    for index in desired_indices:
        
        ActualSubhaloIndex = SubhaloIndicesWithStars[index]
        Vel_subhalo,Vel_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Velocities',4,output_redshift,ActualSubhaloIndex,requested_property1,store_all_offsets=1,group_type='subhalo')
        MStars_subhalo,Mstars_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Masses',4,output_redshift,ActualSubhaloIndex,requested_property2,store_all_offsets=1,group_type='subhalo')
        BHMasses_subhalo,BHMasses_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'Masses',5,desired_redshift,ActualSubhaloIndex,requested_property3,store_all_offsets=1,group_type='subhalo')
        BHProgs_subhalo,BHProgs_group,output_redshift=get_particle_property_within_postprocessed_groups_adj(basePath,'BH_Progs',5,desired_redshift,ActualSubhaloIndex,requested_property4,store_all_offsets=1,group_type='subhalo')
        
        # Velocity calculations
        N = len(Vel_subhalo) # number of stars
        Vel_subhalo=np.array(Vel_subhalo)/np.sqrt(a) # New units: km/s
        VelocityMags = np.linalg.norm(Vel_subhalo, axis=1) # Calculate velocity magnitudes
        mu = np.mean(VelocityMags) # Average stellar velocity
        Sigmasubhalo = np.sqrt(np.sum((VelocityMags - mu) ** 2) / N)  # Calculate velocity dispersion
       
        M.append(np.max(BHMasses_subhalo)*1e10*h)
        MStars.append(MStars_subhalo*1e10*h)
        Sigma.append(Sigmasubhalo)
        VelsMag.append(VelocityMags)
        Velocities.append(Vel_subhalo)
        NStars.append(N)
        BH_Progs.append(BHProgs_subhalo[0])
        IgnoredBhs += len(BHMasses_subhalo)-1
    
    return(M,MStars,Sigma,VelsMag,Velocities,NStars,BH_Progs,IgnoredBhs)



'''
This function was directly adapted from Aklant Bohmwick's get_particle_property_within_postprocessed_groups function from his version of arepo_package. This adapted version takes requested_property as an input so as to avoid loading in all the data every time the property of some halo is desired. 
'''

def get_particle_property_within_postprocessed_groups_adj(output_path,particle_property,p_type,desired_redshift,subhalo_index,requested_property,group_type='groups',list_all=True,store_all_offsets=1, public_simulation=0,file_format='fof_subfind'):
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

def Write2File(*args,fname='BrahmaData'):
    
    Data=[arg for arg in args]
    
    # Originally did this with pickle, then needed pickle5 but couldn't install
    # with open(fname+'.pickle', 'wb') as handle:
    #     pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Now trying with cloudpickle
    with open(fname+'.pickle', 'wb') as f:
        cloudpickle.dump(Data, f)
        
        
'''
ReadBrahmaData is a function designed to read in the data dumped by Write2File

Inputs:

fname: Name of file to read data from

Outputs:

Data: List of data from M_Sigma function in the following order: [M,Sigma,VelsMag,Velocities,Nstars,IgnoredBhs]
See M_Sigma for details
'''

def ReadBrahmaData(fname='BrahmaData'):
    # with open(fname+'.pickle', 'rb') as handle:
    #     Data = pickle.load(handle)
        
    # Trying with cloudpickle: 
    with open(fname+'.pickle', 'rb') as f:
        Data = cloudpickle.load(f)
        
    return(Data)


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

def fixed_x(X_vals,Y_vals,fixed_vals,bin_width):
    
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

def precent_growth(array,array_index,redshift_indices):
    
    highzval = array[array_index][redshift_indices[0]]
    lowzval = array[array_index][redshift_indices[1]]
    
    perc_growth = (lowzval - highzval)/highzval
    
    vals = [highzval,lowzval]
    
    return(vals,perc_growth)
    

    
    
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

def Center_subhalo(ParticleProps,Subhaloprops,box_size,redshift,h,subhalo_id):
    
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
    


'''
kinematic_decomp_r does a decomposition of subhalos based on the energetics of the stars.
Stars with a lower specific angular momentum than 50% of the angular momentum for a star
on a circular orbit at its position are classified as belonging to the spheroid, whereas
those with grater than 50% are classified as belonging to the disk.

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


def kinematic_decomp_r(Coordinates,Velocities,Potentials,nbins=500,nstars_min=1000):
    
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
nbins: Number of radial bins to make along the disk radially when calculating stellar potentials
nstars_min: Minimum number of stars required in a subhalo to do the decomposition

Outputs:
pos: Radial positions at which the gradients are given
grad: Gravitational potential gradient at the radial positions 
ratio: Ratio of j_z to j_circ for each star given its specific binding energy
negids: ids of stellar angular momentums that were set to np.nan
'''

def kinematic_decomp_e(Coordinates,Velocities,Potentials,nbins=300,nstars_min=1000):
    
    # Only do decomposition if there are at least nstars_min stars
    if len(Coordinates)<nstars_min:
        return
    
    kpc2km = 3.0857e16 # Conversion rate from kpc to km
    # radial distance from subhalo center in the xy plane
    r = np.sqrt(Coordinates[:,0]**2 + Coordinates[:,1]**2)
    
    height = 3 * kpc2km # kpc for height of disk
    ri   = 0 * kpc2km  # from 0
    ro   = np.percentile(r, 97.5) # to where 97.5% of the stars exist. This prevents large outlier outlier radii from messing up the bins
    n = 3 # Number of stars required per bin 
    
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
    if len(potential_binned) > 1:
        grad = np.gradient(potential_binned,pos)
    else: # Temporary fix; subhalo 170 had no values in potential_binned? Need a better fix
        return(np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)

    window=10
    
    if 0.5*len(grad) > window: # Only want to use smoothing if the binned data is large enough

        # Trying smoothing
        smoothed_g = savgol_filter(grad, window_length=window, polyorder=1)
        gradient_interp = interp1d(pos, smoothed_g, kind='linear', fill_value="extrapolate")
    
        smoothed_p = savgol_filter(potential_binned, window_length=window, polyorder=1)
        potental_interp = interp1d(pos, smoothed_p, kind='linear', fill_value="extrapolate")
                
    else: # Otherwise, just the use interpolation functions

        # Interpolating the potentials and potential gradient function with scipy 
        gradient_interp = interp1d(pos, grad, kind='linear', fill_value="extrapolate")
        potental_interp = interp1d(pos, potential_binned, kind='linear', fill_value="extrapolate")

    # Defining new function for root finder to calculate rc
    def f(r,args): # args: [stellar specific binding energy]
        val = potental_interp(r) + 0.5*r*np.max([0,gradient_interp(r)]) - args[0] # np.max prevents negative potentials
        return(val)
    
    rcs = []
    skipped_stars = 0
    
    # Calculating circular radii for all stars given their binding energies e
    for i in range(len(Potentials)):
        args = [Potentials[i] + 0.5*np.linalg.norm(Velocities[i])**2]
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
    
    # Return the radial positions, gradients, and ratio of the angular momentums to the specific angular momentums
    return(pos,grad,ratio,negids,rcs,potential_binned,gradient_interp,potental_interp)

'''
Testing the constant number of stars in each bin

nstars: stars per bin
'''

def kinematic_decomp_e2(Coordinates,Velocities,Potentials,nstars=150,nstars_min=1000):
    
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
        val = potental_interp(r) + 0.5*r*np.max([0,gradient_interp(r)]) - args[0] # np.max prevents negative potentials
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
    
    # Return the radial positions, gradients, and ratio of the angular momentums to the specific angular momentums
    return(ratio,negids,rcs)#,xvals,grad,bin_centers,bin_averages,gradient_interp,potental_interp)



'''
cal_avg is like mean_trends, but simplified for only one set of x and y values

xvals and yvals in normal values, bins in log10
'''

def cal_avg(xvals,yvals,bins):
    
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


'''
overlapping_bins is pretty straightforward: It makes overlapping bins

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

def overlapping_bins(start,end,nbins,dx=0.5):
    
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


'''
equal_num_bins creates bins of a value (for my use, potentials or potential gradients)
with an equal number of objects (stars) in each bin, then returns the average value 
and position for each bin

r: Radii
vals: Potentials
N: number of stars per bin
'''


def equal_num_bins(r,vals,N=150):
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


'''
remove_linear_gradient removes any linear trend in the potentials of stars

Inputs:
Coordinates: Coordinates of the stars of the subhalo, in km
Potentials: Potentials of the stars of the subhalo, in (km/s)^2

Outputs:
corrected_potential: Potentials with the linear component subtracted off
'''

def remove_linear_gradient(Coordinates,Potentials):

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
