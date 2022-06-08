#================================Reprojection==================================
'''
Purpose: [Still under development]
    Reprojects, resamples and masks out the soil hydraulic parameters:
        (1) Soil Grid (Code)
        (2) Soil Depth
        (3) Alpha
        (4) n
        (5) Hydraulic Conductivity (Ksat)
        (6) Residual Soil Moisture Content (WC_res)
        (7) Saturation Soil Moisture Content (WC_sat)
        
    Data source: (FutureWater, 2015) ~ HighHydroSoil 1km

       Record of revision:
           Date            Author          Description of change
           ====            ======          =====================
           2018/02/10      Abdul Moiz      Original code (Developer Version)
'''
#==============================================================================

#============================Importing Modules=================================
print("Importing modules...")
import archook
archook.get_arcpy()
import arcpy
import os
import numpy as np
import pandas as pd
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
#==============================================================================
    
#==============================Inputs==========================================
#Directories
project = '../data_prep_arcgis/'
scratch = 'scratch'
inputs = 'inputs'
output = 'output'
data = 'data'
parameter = 'parameter/hihydro'
mapping = 'mapping/hihydro'
soil = 'hihydro_soil'
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
        
#===================================Main=======================================
cell_area = arcpy.Raster(os.path.join(project, scratch, 'cell_area.tif'))
target_cellsize = arcpy.Describe(cell_area).meanCellHeight
target_projection = arcpy.Describe(cell_area).SpatialReference

print('Reading HiHydro Soil data...')
soil_unit = arcpy.Raster(os.path.join(project, inputs, soil, 'soil_unit_wrb.tif'))

alpha_ts = arcpy.Raster(os.path.join(project, inputs, soil, 'alpha_ts.tif'))
alpha_ss = arcpy.Raster(os.path.join(project, inputs, soil, 'alpha_ss.tif'))

watern_ts = arcpy.Raster(os.path.join(project, inputs, soil, 'n_ts.tif'))
watern_ss = arcpy.Raster(os.path.join(project, inputs, soil, 'n_ss.tif'))

ks_ts = arcpy.Raster(os.path.join(project, inputs, soil, 'ksat_ts.tif'))
ks_ss = arcpy.Raster(os.path.join(project, inputs, soil, 'ksat_ss.tif'))

thetar_ts = arcpy.Raster(os.path.join(project, inputs, soil, 'wcres_ts.tif'))
thetar_ss = arcpy.Raster(os.path.join(project, inputs, soil, 'wcres_ss.tif'))

thetas_ts = arcpy.Raster(os.path.join(project, inputs, soil, 'wcsat_ts.tif'))
thetas_ss = arcpy.Raster(os.path.join(project, inputs, soil, 'wcsat_ss.tif'))


soil_depth = arcpy.Raster(os.path.join(project, inputs, soil, 'soil_depth.tif'))


print('Re-projecting...')
for raster in [soil_unit, alpha_ts, alpha_ss, ks_ts, ks_ss, 
               thetar_ts, thetar_ss, thetas_ts, thetas_ss, 
               watern_ts, watern_ss, soil_depth]:
    arcpy.ProjectRaster_management(raster, 
                                   os.path.join(project, mapping, str(raster).split('\\')[-1]), 
                                   target_projection, 
                                   'NEAREST', 
                                   target_cellsize)
    raster1 = arcpy.Raster(os.path.join(project, mapping, str(raster).split('\\')[-1]))
    raster_extract = arcpy.sa.ExtractByMask(raster1, cell_area)
    if (str(raster) != 'soil_unit_wrb.tif') or (str(raster) != 'soil_depth.tif'):
        raster_extract = raster_extract/10000.0
    raster_extract.save(str(raster1))
    arcpy.RasterToASCII_conversion(raster, os.path.join(project, output, parameter, (str(raster).split('\\')[-1]).split('.')[0] + '.asc'))

##Making soil_unit.asc and soil_dpeth.asc
#soil_unit = arcpy.Raster(os.path.join(project, mapping, hihydro, 'TAXNWRB_1km_ll.tif'))
#soil_depth = arcpy.Raster(os.path.join(project, mapping, hihydro, 'BDRICM_M_1km_ll.tif.tif'))
#for raster in [soil_unit, soil_depth]:
#    print('Processing' + (str(raster).split('\\')[-1]).split('.')[0] + '.asc ' + '...')
#    arcpy.RasterToASCII_conversion(raster, os.path.join(project, output, parameter, hihydro, (str(raster).split('\\')[-1]).split('.')[0] + '.asc'))
#
##Making soil_code.txt
#print('Processing soil_code.txt ...')
#soil_unit = arcpy.RasterToNumPyArray(soil_unit, nodata_to_value = -9999)
#soil_code = np.unique(soil_unit, return_counts = True)
#soil_code = {'Record':range(1, len(soil_code[0])),
#             'VALUE':soil_code[0][1:].tolist(),
#             'COUNT':soil_code[1][1:].tolist()}
#soil_code = pd.DataFrame(data=soil_code)
#soil_code = soil_code[['Record', 'VALUE', 'COUNT']]
#soil_code.to_csv(os.path.join(project, output, parameter, 'soil_code.txt'), sep = '\t', index = False)
#
#
##Making soil_water_para.dat
#print('Processing soil_water_para.dat ...')
#alpha = arcpy.Raster(os.path.join(project, mapping, 'alpha.tif'))
#ks = arcpy.Raster(os.path.join(project, mapping, 'ks.tif'))
#thetar = arcpy.Raster(os.path.join(project, mapping, 'thetar.tif'))
#thetas = arcpy.Raster(os.path.join(project, mapping, 'thetas.tif'))
#watern = arcpy.Raster(os.path.join(project, mapping, 'watern.tif'))
#
#soil_code['alpha'] = -9999.0
#soil_code['n'] = -9999.0
#soil_code['theta_r'] = -9999.0
#soil_code['theta_s'] = -9999.0
#soil_code['ks1'] = -9999.0
#soil_code['ks2'] = -9999.0
#soil_code['ksg'] = -9999.0
#soil_code['GWcs'] = 0.15
#soil_code['apow'] = -9999.0
#soil_code['bpow'] = -9999.0
#soil_code['lamdaf_max'] = -9999.0
#soil_code['vlcmin'] = -9999.0
#
#
#for raster in [alpha, ks, thetar, thetas, watern]:
#    raster_pd = arcpy.RasterToNumPyArray(raster, nodata_to_value = -9999.0)
#    raster_pd = np.unique(raster_pd, return_counts = True)
#    raster_pd = {'Record':range(1,len(raster_pd[0])),
#              'VALUE':raster_pd[0][1:].tolist(),
#              'COUNT':raster_pd[1][1:].tolist()}
#    raster_pd = pd.DataFrame(data=raster_pd)
#    raster_pd = raster_pd[['Record', 'VALUE', 'COUNT']]
#    for i in range(0,len(soil_code)):
#        value = soil_code['VALUE'].iloc[i]
#        count = soil_code['COUNT'].iloc[i]
#        j = raster_pd.loc[raster_pd['COUNT'] == count, 'VALUE']
#        if len(j) == 0:
#            raster_value = -9999.0
#        else:
#            raster_value = round(j.item(), 5)
#        
#        if str(raster) == str(alpha):
#            soil_code['alpha'].iloc[i] = raster_value
#        
#        elif str(raster) == str(ks):
#            if raster_value != -9999.0:
#                raster_value = round(raster_value*10.0/24.0, 5)
#                soil_code['ks1'].iloc[i] = raster_value
#                soil_code['ks2'].iloc[i] = round(0.1*raster_value,5)
#                soil_code['ksg'].iloc[i] = round(0.05*raster_value,5)
#                
#            else:
#                soil_code['ks1'].iloc[i] = -9999.0
#                soil_code['ks2'].iloc[i] = -9999.0
#                soil_code['ksg'].iloc[i] = -9999.0
#        
#        elif str(raster) == str(thetar):
#            soil_code['theta_r'].iloc[i] = raster_value
#            
#        elif str(raster) == str(thetas):
#            soil_code['theta_s'].iloc[i] = raster_value
#            
#        elif str(raster) == str(watern):
#            soil_code['n'].iloc[i] = raster_value
#with open(os.path.join(project, output, parameter, 'soil_water_para.dat'), 'w') as f:
#    f.write('Soil water parameters \n')
#    f.write('soil_code \n')
#    for col in soil_code[['theta_s', 'theta_r', 'alpha', 'n', 'ks1', 'ks2', 'ksg', 'GWcs', 'apow', 'bpow', 'lamdaf_max', 'vlcmin']].columns:
#        f.write(col + '\t')
#    f.write('\n')
#    for i in range(0, len(soil_code)):
#        f.write(str(soil_code['VALUE'].iloc[i]) + '\n')
#        for col in soil_code[['theta_s', 'theta_r', 'alpha', 'n', 'ks1', 'ks2', 'ksg', 'GWcs', 'apow', 'bpow', 'lamdaf_max', 'vlcmin']].columns:
#            f.write(str(soil_code[col].iloc[i]) + '\t')
#        f.write('\n')
#==============================================================================

#==========================Checking in extensions==============================
print("Checking in extensions...")
for extension in extensions:
    if arcpy.CheckExtension(extension) == "Available":
        arcpy.CheckInExtension(extension)
#==============================================================================
        
print("Finished Sucessfully!")
        