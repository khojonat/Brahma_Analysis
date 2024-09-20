import numpy as np
import arepo_package
import math
import matplotlib.pyplot as plt

'''
load_data is a function designed to load in BRAHMA data and store the desired data in a list of lists (of lists).
The lists are stacked according to simulation box and redshift.

Inputs:

Outputlist: Aimulation boxes you want to analyze
redshifts: Redshifts you want to look at 
Property1, Property2: the particle properties to load
part_type: Type of particle to analyze (dark matter=0, gas=1, star=4, black hole=5)
conversion1,conversion2: Conversion from internal units to physical units for properties 1 and 2 respectively
Lbol: Whether we want to convert to Bolometric Luminosity. This only applied if reading in BH Mdot
e_r: Radiative efficiency used in calculating Lbol

Outputs:

Prop1list, Prop2list: Lists of lists of properties you specified to pull
outputzlist: the actual redshifts of the snapshots taken from BRAHMA
limits: the minimum and maximum values of Property1 (assumed to be x axis) across the entire dataset in log10(physical units). 

The limits will be useful for binning to make mean trends
'''

# Defaults for the conversions are 1e10*h, where h is 0.6774
def load_data(path_to_output,run,outputlist,redshifts,Property1,Property2,part_type,
              conversion1=1e10*0.6774,conversion2 = 1e10*0.6774,Lbol=[False,False],e_r=0.2):
    
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

    for basePath in basePaths:
    
        # List for all z's for the current Box
        BoxProp1 = []
        BoxProp2 = []
        Boxoutputz = []
        Boxminx = []
        Boxmaxx = []
    
        for z in redshifts:
    
            #Reading Prop1 for each type, dark matter=0, gas=1, star=4, black hole=5
            GroupProp1,output_redshift=arepo_package.get_group_property(basePath,Property1,z,postprocessed=1)
        
            #Reading Prop2 of the halo
            GroupProp2,output_redshift=arepo_package.get_group_property(basePath,Property2,z,postprocessed=1)
            
            # If we want the BH bolometric luminsoity from Mdot, set Lbol=True
            # This is definitely not the most efficient, but it works
            
            c = 3e10 # cm/s
            gperMsun = 1.9884099e33 # grams per solar mass
            
            if Lbol[0]:
                
                # Selecting only the particle type that we're interested in
                Groupdata=np.array(GroupProp1[:,part_type])*conversion1*gperMsun*e_r*c**2
                BoxProp1.append(Groupdata)
                BoxProp2.append(GroupProp2*conversion2)
                Boxoutputz.append(output_redshift)
                
            elif Lbol[1]:
                
                Groupdata=np.array(GroupProp1[:,part_type])*conversion1
                BoxProp1.append(Groupdata)
                BoxProp2.append(GroupProp2*conversion2*gperMsun*e_r*c**2)
                Boxoutputz.append(output_redshift)
                
            else:
                
                Groupdata=np.array(GroupProp1[:,part_type])*conversion1
                BoxProp1.append(Groupdata*conversion1)
                BoxProp2.append(GroupProp2*conversion2)
                Boxoutputz.append(output_redshift)
            
            # Nonzero ids for our properties to grab limits
            nonzero1id = set(np.nonzero(Groupdata)[0])
            nonzero2id = set(np.nonzero(GroupProp2)[0])
            Boxminx.append(np.min( np.log10(Groupdata[list(nonzero1id & nonzero2id)] ) ))
            Boxmaxx.append(np.max( np.log10(Groupdata[list(nonzero1id & nonzero2id)] ) ))
        
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

Prop1list,Prop2list: Lists of lists of properties you specified to pull
redshifts: Redshifts you want to look at 
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

    # Want to find mean and std dev for each box
    for i in range(len(Prop1list)):
    
        BoxMeans = []
        BoxStdDevs = []
        Box_ids = []
        
        low = math.floor(limits[0])
        high = math.ceil(limits[1])
        # Add a manual shift of i in log scale to the bins to prevent overlap
        bins = np.log10((i+1)*np.logspace(low,high,num=numbins))

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
                
                Vals = np.array(Prop2list[i][ii][ids][np.nonzero(Prop2list[i][ii][ids])])
                ZMeans.append(np.mean(np.log10(Vals)))
                ZStdDevs.append(np.std(np.log10(Vals)))
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
        # ax.set_ylim(4,9) # Maybe don't need to do this since the plots should be sharing y axes?
        ax.set_xlim(np.min(XPoints)-0.5,np.max(XPoints)+0.5)
        ax.set_title('Z={}'.format(redshifts[n]),size = 25)
        n+=1

    # f.legend(fontsize = 12)
    f.supxlabel('{}'.format(axislabels[0]),fontsize=label_font_size)
    f.supylabel('{}'.format(axislabels[1]),fontsize=label_font_size)

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    
    if savefig != False:
        plt.savefig(savefig)
        
    return(f,axes)

