import sys
from brahma_analysis import * # Imports illustris_python here
import arepo_package
import h5py
import os
import numpy as np

sys.path.append('/home/yja6qa/arepo_package/')

h = 0.6774

TNGpath='/standard/torrey-group/IllustrisTNG/Runs/L75n1820TNG'
basePath = TNGpath
snap_num=99
subhalo_id = 0

fields = ['Masses','Coordinates','Velocities']
Star_Props = il.snapshot.loadSubhalo(TNGpath, snap_num, id=subhalo_id, partType=4, fields=fields)
fields = ['Coordinates']
DM_Props = il.snapshot.loadSubhalo(TNGpath, snap_num, id=subhalo_id, partType=1, fields=fields)
fields = ['Masses','Coordinates']
Gas_Props = il.snapshot.loadSubhalo(TNGpath, snap_num, id=subhalo_id, partType=0, fields=fields)
BH_Props = il.snapshot.loadSubhalo(TNGpath, snap_num, id=subhalo_id, partType=5, fields=fields)

Star_Props = [Star_Props['Masses'],Star_Props['Coordinates'],Star_Props['Velocities']]
Gas_Props = [Gas_Props['Masses'],Gas_Props['Coordinates']]
BH_Props = [BH_Props['Masses'],BH_Props['Coordinates']]

Grav_Pot_E = Grav_Pot_E_efficient(Star_Props, DM_Props, Gas_Props,BH_Props)


Write2File(Grav_E,fname='Brahma_Data/Grav_Pot_E_efficient')
