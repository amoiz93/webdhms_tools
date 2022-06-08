#================================Reprojection==================================
'''
Purpose:
    Creates the _river and _morph files for each sub-basin.
    Sub-catchment.dat produced in the previous step needs to be modified with
    appropriate values.

       Record of revision:
           Date            Author          Description of change
           ====            ======          =====================
           2018/02/10      Abdul Moiz      Original code (Developer Version)
'''
#==============================================================================

#============================Importing Modules=================================
print("Importing modules...")
import pandas as pd
import os
import archook
archook.get_arcpy(pro=True)
import arcpy
import math
import numpy as np
import linecache
#==============================================================================

#========================Defining Custom Functions=============================
#------------------------------------------------------------------------------
def createfolder(folderName):
    if os.path.isdir(folderName)==False:
        os.mkdir(folderName)
#------------------------------------------------------------------------------
def rootpath(filepath):
    string=filepath.split('\\')
    folderpath=''
    for x in range(0,len(string)-1):
        folderpath=folderpath+str(string[x])+"\\"
    return folderpath
    del string
    del x
    del folderpath
#------------------------------------------------------------------------------
def outpaths(scratch, output, data, parameter, mapping):
    createfolder(scratch)
    createfolder(output)
    createfolder(os.path.join(output, data))
    createfolder(os.path.join(output, parameter))
    createfolder(mapping)
#------------------------------------------------------------------------------
def read_ascii_grid(array):
    ncols = int(((linecache.getline(array, 1)).split())[-1])
    nrows = int(((linecache.getline(array, 2)).split())[-1])
    xll = float(((linecache.getline(array, 3)).split())[-1])
    yll = float(((linecache.getline(array, 4)).split())[-1])
    cellsize = int(((linecache.getline(array, 5)).split())[-1])
    nodata_value = int(((linecache.getline(array, 6)).split())[-1])
    array = np.loadtxt(array, skiprows=6)
    array = np.ma.masked_where(array == nodata_value, array)
    return array, ncols, nrows, xll, yll, cellsize, nodata_value
    linecache.clearcache()
#==============================================================================

#==============================Inputs==========================================
#Directories
project = '../data_prep_arcgis/'
scratch = 'scratch'
inputs = 'inputs'
output = 'output'
data = 'data'
parameter = 'parameter'
mapping = 'mapping'

dx_max = 500

pfaf_level = 1
pfaf_level0_basins = [0]      #Which subbasins to sub-diivde upto level 1 (Always 0)
pfaf_level1_basins = []      #Which subbasins to sub-divide upto level 2
pfaf_level2_basins = []       #Which subbasins to sub-divide upto level 3
max_level = 10
#==============================================================================

#====================Setting Environment Variables=============================
print("Setting enviornment variables...")
os.chdir(project)
outpaths(scratch, output, data, parameter, mapping)

arcpy.env.workspace = project
arcpy.env.scratchWorkspace = os.path.join(project, scratch)

arcpy.env.overwriteOutput = True
#==============================================================================

#========================Checking out extensions===============================
print("Checking out extensions...")
extensions = ["Spatial"]
for extension in extensions:
    if arcpy.CheckExtension(extension) == "Available":
        print("Checking out " + extension + " Analyst...")
        arcpy.CheckOutExtension(extension)
    else:
        print(extension + " Analyst license is unavailable")
#==============================================================================

#==================================Main========================================
cell_area = arcpy.Raster(os.path.join(project, scratch, 'cell_area.tif'))
cell_area_ascii = read_ascii_grid(os.path.join(project, output, parameter, 'cell_area.asc'))
flow_int_grid = cell_area_ascii[0]
flow_int_grid = np.ma.filled(flow_int_grid, -9999)


coarse_resolution = arcpy.Describe(cell_area).meanCellHeight
spatial_reference = arcpy.Describe(cell_area).SpatialReference

subcatchment = pd.read_csv(os.path.join(project, output, parameter, 'subcatchment.dat'), header = 0, delim_whitespace=True)
for ws in subcatchment['sub_basin']:
    ws_path = os.path.join(project, scratch, ws + '_morph.pkl')
    ws_df = pd.read_pickle(ws_path)
    ws_df.sort_values(by = ['distance'], ascending = False, inplace = True)
    dis_max = ws_df['distance'].iloc[0]
    n_totalgrid = len(ws_df)
    flow_num = int(dis_max/dx_max)+1
    
    width_min = subcatchment['width_min'].loc[subcatchment['sub_basin'] == ws].item()
    width_max = subcatchment['width_max'].loc[subcatchment['sub_basin'] == ws].item()
    
    height_min = subcatchment['height_min'].loc[subcatchment['sub_basin'] == ws].item()
    height_max = subcatchment['height_max'].loc[subcatchment['sub_basin'] == ws].item()
    
    roughness_min = subcatchment['roughness_min'].loc[subcatchment['sub_basin'] == ws].item()
    roughness_max = subcatchment['roughness_max'].loc[subcatchment['sub_basin'] == ws].item()
    
    with open(os.path.join(project, output, parameter, ws + '_river'), 'w') as f:
        f.write(str(flow_num) + '\n')
        
        dx_total = 0
        for i in range(flow_num)[::-1]:
            flow_int = ws_df.loc[(ws_df['distance'] >= (i*dx_max)) & (ws_df['distance'] < ((i+1)*dx_max))]
            flow_int_area = len(flow_int)*coarse_resolution
            flow_int_angle = flow_int['bedslope'].mean()
            flow_int_tan = round(math.tan(math.radians(flow_int_angle)),6)
            
            width = width_max + (i+1)*(width_min-width_max)/float(flow_num)
            height = height_max + (i+1)*(height_min-height_max)/float(flow_num)
            roughness = roughness_max + (i+1)*(roughness_min-roughness_max)/float(flow_num)
            
            if i == 0:
                dx = dis_max - (flow_num-1)*dx_max
            else:
                dx = dx_max
                
            f.write(str(len(flow_int)) + '\t' +
                    str(round(dx,2)) + '\t' + 
                    str(round(flow_int_tan,6)) + '\t' +
                    str(round(width,2)) + '\t' +
                    str(round(roughness, 5)) + '\t' +
                    str(round(height,2)) + '\n')
            
            for j in range(len(flow_int)):
                flow = flow_int.iloc[j]
                f.write(str(flow.row) + '\t')
                f.write(str(flow.col) + '\t')
                
                flow_int_grid[flow.row-1][flow.col-1] = (flow_num - i)
                
            f.write('\n')
            
np.savetxt(os.path.join(project, output, parameter, 'flow_int.asc'), flow_int_grid, 
           comments ='', fmt = '%i',
           header = linecache.getline(os.path.join(project, output, parameter, 'cell_area.asc'), 1)\
                  + linecache.getline(os.path.join(project, output, parameter, 'cell_area.asc'), 2)\
                  + linecache.getline(os.path.join(project, output, parameter, 'cell_area.asc'), 3)\
                  + linecache.getline(os.path.join(project, output, parameter, 'cell_area.asc'), 4)\
                  + linecache.getline(os.path.join(project, output, parameter, 'cell_area.asc'), 5)\
                  + linecache.getline(os.path.join(project, output, parameter, 'cell_area.asc'), 6).rstrip())
#==============================================================================
print('Finished Successfully!')