#================================SiB2 Land Cover==================================
'''
Purpose:
    Reprojects, resamples and masks out Land Cover.
    
    Data Source: SiB2 land classification (USGS) ~ 1km

       Record of revision:
           Date            Author          Description of change
           ====            ======          =====================
           2018/02/10      Abdul Moiz      Original code (Developer Version)
'''
#==============================================================================

#============================Importing Modules=================================
print("Importing modules...")
import archook
archook.get_arcpy(pro=True)
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
parameter = 'parameter'
mapping = 'mapping'
land = 'usgs_land'
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
        
#====================================Main======================================
cell_area = arcpy.Raster(os.path.join(project, scratch, 'cell_area.tif'))
target_cellsize = arcpy.Describe(cell_area).meanCellHeight
target_projection = arcpy.Describe(cell_area).SpatialReference
arcpy.env.extent = cell_area

print('Reading USGS land cover data...')
land = arcpy.Raster(os.path.join(project, inputs, land, 'gbsbm2geo20.tif'))

print('Re-projecting ...')
land_reprojected = arcpy.ProjectRaster_management(land, os.path.join(project, scratch, 'land_use.tif'),
                                                  target_projection, 'NEAREST', target_cellsize)

print('Subsetting ...')
land_subset = arcpy.sa.ExtractByMask(land_reprojected, cell_area)
land_subset.save(os.path.join(project, scratch, 'land_use.tif'))
arcpy.CopyRaster_management(os.path.join(project, scratch, 'land_use.tif'),
                            os.path.join(project, mapping, 'land_use.tif'),
                            '#', '#', '-9999', '#', '#', '16_BIT_SIGNED', '#', '#', 'TIFF', '#')
land_subset = arcpy.sa.ExtractByMask(os.path.join(project, mapping, 'land_use.tif'),
                                     cell_area)
land_subset.save(os.path.join(os.path.join(project, mapping, 'land_use.tif')))

print('Processing land_use.asc ...')
arcpy.RasterToASCII_conversion(os.path.join(os.path.join(project, mapping, 'land_use.tif')), os.path.join(project, output, parameter, 'land_use.asc'))
#==============================================================================

#==========================Checking in extensions==============================
print("Checking in extensions...")
for extension in extensions:
    if arcpy.CheckExtension(extension) == "Available":
        arcpy.CheckInExtension(extension)
#==============================================================================
        
print("Finished Sucessfully!")
