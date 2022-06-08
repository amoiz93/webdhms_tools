#================================Reprojection==================================
'''
Purpose:
    This scripts delineates the watershed and generates the following:
        (1) cell_area.asc
        (2) elevation.asc

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

#Inputs
outlet = 'unazuki.shp'
stream_def_threshold = 20
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
print("Processing...")
print("Delineating Watershed")

outlet = os.path.join(project, inputs, outlet)
#Fine Watershed
print("Fine Watershed")
fineDEM = arcpy.Raster(os.path.join(project, scratch, 'fineDEM.tif'))
fineFill = arcpy.Raster(os.path.join(project, scratch, 'fineFill_full.tif'))
fineFDR = arcpy.Raster(os.path.join(project, scratch, 'fineFDR_full.tif'))
fineFAC = arcpy.Raster(os.path.join(project, scratch, 'fineFAC_full.tif'))

fineBasin = arcpy.sa.Watershed(fineFDR, outlet)
fineBasin.save(os.path.join(project, scratch, 'fineBasin.tif'))
fineWatershed = arcpy.RasterToPolygon_conversion(os.path.join(project, scratch, 'fineBasin.tif'), os.path.join(project, mapping, 'fineBasin.shp'), "NO_SIMPLIFY")

outfineDEM = arcpy.sa.ExtractByMask(fineDEM, fineBasin)
outfineDEM.save(os.path.join(project, scratch, 'fineDEM.tif'))

outfineFill = arcpy.sa.ExtractByMask(fineFill, fineBasin)
outfineFill.save(os.path.join(project, scratch, 'fineFill.tif'))

outfineFDR = arcpy.sa.ExtractByMask(fineFDR, fineBasin)
outfineFDR.save(os.path.join(project, scratch, 'fineFDR.tif'))

outfineFAC = arcpy.sa.ExtractByMask(fineFAC, fineBasin)
outfineFAC.save(os.path.join(project, scratch, 'fineFAC.tif'))

fineStream = arcpy.sa.Con(outfineFAC > int(stream_def_threshold), 1)
arcpy.sa.StreamToFeature(fineStream, outfineFDR, \
                         os.path.join(project, mapping, 'fineStream.shp'), "NO_SIMPLIFY")

#Coarse Watershed
print("Coarse Watershed")
coarseDEM = arcpy.Raster(os.path.join(project, scratch, 'coarseDEM.tif'))
coarseFill = arcpy.Raster(os.path.join(project, scratch, 'coarseFill_full.tif'))
coarseFDR = arcpy.Raster(os.path.join(project, scratch, 'coarseFDR_full.tif'))
coarseFAC = arcpy.Raster(os.path.join(project, scratch, 'coarseFAC_full.tif'))

coarseBasin = arcpy.sa.Watershed(coarseFDR, outlet)
coarseBasin.save(os.path.join(project, scratch, 'coarseBasin.tif'))
coarseWatershed = arcpy.RasterToPolygon_conversion(os.path.join(project, scratch, 'coarseBasin.tif'), os.path.join(project, mapping, 'coarseBasin.shp'), "NO_SIMPLIFY")

outcoarseDEM = arcpy.sa.ExtractByMask(coarseDEM, coarseBasin)
outcoarseDEM.save(os.path.join(project, scratch, 'coarseDEM.tif'))

outcoarseFill = arcpy.sa.ExtractByMask(coarseFill, coarseBasin)
outcoarseFill.save(os.path.join(project, scratch, 'coarseFill.tif'))

outcoarseFDR = arcpy.sa.ExtractByMask(coarseFDR, coarseBasin)
outcoarseFDR.save(os.path.join(project, scratch, 'coarseFDR.tif'))

outcoarseFAC = arcpy.sa.ExtractByMask(coarseFAC, coarseBasin)
outcoarseFAC.save(os.path.join(project, scratch, 'coarseFAC.tif'))

coarseStream = arcpy.sa.Con(outcoarseFAC > int(stream_def_threshold), 1)
arcpy.sa.StreamToFeature(coarseStream, outcoarseFDR, \
                         os.path.join(project, mapping, 'coarseStream.shp'), "NO_SIMPLIFY")

arcpy.Clip_management(coarseFill, "#", os.path.join(project, scratch, 'elevation.tif'),os.path.join(project, mapping, 'coarseBasin.shp'), "#", "ClippingGeometry", "#")
arcpy.RasterToASCII_conversion(os.path.join(project, scratch, 'elevation.tif'), os.path.join(project, output, parameter, 'elevation.asc'))

coarseBasin = arcpy.sa.Con(coarseBasin == 0, 1)
arcpy.Clip_management(coarseBasin, '#', os.path.join(project, scratch, 'cell_area.tif'), os.path.join(project, mapping, 'coarseBasin.shp'), '#', 'ClippingGeometry', '#')
arcpy.RasterToASCII_conversion(os.path.join(project, scratch, 'cell_area.tif'), os.path.join(project, output, parameter, 'cell_area.asc'))
#==============================================================================

#==========================Checking in extensions==============================
print("Checking in extensions...")
for extension in extensions:
    if arcpy.CheckExtension(extension) == "Available":
        arcpy.CheckInExtension(extension)
#==============================================================================
    
print("Finished Sucessfully!")