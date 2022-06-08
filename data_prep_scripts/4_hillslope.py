#================================Reprojection==================================
'''
Purpose:
    This scripts is used for sub-grid parameterization. It generates the
    following outputs:
        (1) slope_angle.asc (fine slope mean resampled to coarse slope)
        (2) bedslope.asc    (corase slope mean value using 5 x 5 neighbourhood)
        (3) slope_length.asc(A/(2*sum(L)))

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
import math
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
fineFill = arcpy.Raster(os.path.join(project, scratch, 'fineFill_full.tif'))
coarseFill = arcpy.Raster(os.path.join(project, scratch, 'coarseFill_full.tif'))

fineFDR = arcpy.Raster(os.path.join(project, scratch, 'fineFDR_full.tif'))
fineFAC = arcpy.Raster(os.path.join(project, scratch, 'fineFAC_full.tif'))

fine_resolution = arcpy.Describe(fineFill).meanCellHeight
coarse_resolution = arcpy.Describe(coarseFill).meanCellHeight

ratio = coarse_resolution/fine_resolution

#BedSlope
fineSlope = arcpy.sa.Slope(fineFill, "DEGREE")
fineSlope.save(os.path.join(project, scratch, 'fineSlope.tif'))
fineSlope = arcpy.sa.BlockStatistics(fineSlope, arcpy.sa.NbrRectangle(ratio, ratio, '#'), "MEAN")
coarseSlope = arcpy.Resample_management(fineSlope, os.path.join(project, scratch, 'coarseSlope.tif'), str(coarse_resolution), 'NEAREST')
bedslope = arcpy.sa.BlockStatistics(coarseSlope, arcpy.sa.NbrRectangle(5, 5, '#'), "MEAN") #Question: Why use 5 x 5 neighbourhood?
                                                                       
#Slope Length
fineFDR = arcpy.sa.Con(fineFAC > 1, fineFDR)
slope_length = arcpy.sa.Con((fineFDR == 1) | (fineFDR == 4) | (fineFDR == 16) | (fineFDR == 64), fine_resolution,
               arcpy.sa.Con((fineFDR == 2) | (fineFDR == 8) | (fineFDR == 32) | (fineFDR == 128), fine_resolution*math.sqrt(2)))
slope_length_sum = arcpy.sa.BlockStatistics(slope_length, arcpy.sa.NbrRectangle(ratio, ratio, '#'), "SUM")
coarseSlope_length = arcpy.Resample_management(slope_length_sum, os.path.join(project, scratch, 'coarseSlope_length.tif'), str(coarse_resolution), 'NEAREST')
coarseSlope_length = arcpy.Raster(os.path.join(project, scratch, 'coarseSlope_length.tif'))
slope_length = (float(coarse_resolution)* float(coarse_resolution))/(coarseSlope_length*2)
slope_length.save(os.path.join(project, scratch, 'slope_length_full.tif'))


arcpy.Clip_management(coarseSlope, '#', os.path.join(project, scratch, 'slope_angle.tif'), os.path.join(project, mapping, 'coarseBasin.shp'), '#', 'ClippingGeometry', '#')
arcpy.RasterToASCII_conversion(os.path.join(project, scratch, 'slope_angle.tif'), os.path.join(project, output, parameter, 'slope_angle.asc'))

arcpy.Clip_management(bedslope, '#', os.path.join(project, scratch, 'bedslope.tif'), os.path.join(project, mapping, 'coarseBasin.shp'), '#', 'ClippingGeometry', '#')
arcpy.RasterToASCII_conversion(os.path.join(project, scratch, 'bedslope.tif'), os.path.join(project, output, parameter, 'bedslope.asc'))

arcpy.Clip_management(slope_length, "#", os.path.join(project, scratch, 'slope_length.tif'),os.path.join(project, mapping, 'coarseBasin.shp'), "#", "ClippingGeometry", "#")
arcpy.RasterToASCII_conversion(os.path.join(project, scratch, 'slope_length.tif'), os.path.join(project, output, parameter, 'slope_length.asc'))
#==============================================================================

#==========================Checking in extensions==============================
print("Checking in extensions...")
for extension in extensions:
    if arcpy.CheckExtension(extension) == "Available":
        arcpy.CheckInExtension(extension)
#==============================================================================
    
print("Finished Sucessfully!")