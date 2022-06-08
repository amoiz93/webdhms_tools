#===========================Terrain Preprocessing==============================
'''
Purpose:
    This scripts does the following:
        (1) Creates two DEMs (Coarse and Fine)
        (2) Fill Sinks
        (3) Flow Direction
        (4) Flow Accumulation
        (5) Stream Definition

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
stream_def_threshold = 20
model_resolution = 250
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
#Fine DEM
print("FineDEM")
fineDEM = arcpy.Raster(os.path.join(project, scratch, 'fineDEM.tif'))
fineFill = arcpy.sa.Fill(fineDEM)
fineFDR = arcpy.sa.FlowDirection(fineFill)
fineFAC = arcpy.sa.FlowAccumulation(fineFDR)
fineFill.save(os.path.join(project, scratch, 'fineFill_full.tif'))
fineFDR.save(os.path.join(project, scratch, 'fineFDR_full.tif'))
fineFAC.save(os.path.join(project, scratch, 'fineFAC_full.tif'))
fineStream = arcpy.sa.Con(fineFAC > int(stream_def_threshold), 1)
fineStream.save(os.path.join(project, scratch, 'fineStream_full.tif'))

#Coarse DEM
print("CoarseDEM")
coarseDEM = arcpy.Resample_management(fineFill, 
                                      os.path.join(scratch, 'coarseDEM.tif'), 
                                      model_resolution, 'BILINEAR')
coarseFill = arcpy.sa.Fill(coarseDEM)
coarseFDR = arcpy.sa.FlowDirection(coarseFill)
coarseFAC = arcpy.sa.FlowAccumulation(coarseFDR)
coarseFill.save(os.path.join(project, scratch, 'coarseFill_full.tif'))
coarseFDR.save(os.path.join(project, scratch, 'coarseFDR_full.tif'))
coarseFAC.save(os.path.join(project, scratch, 'coarseFAC_full.tif'))
coarseStream = arcpy.sa.Con(coarseFAC > int(stream_def_threshold), 1)
coarseStream.save(os.path.join(project, scratch, 'coarseStream_full.tif'))
#==============================================================================

#==========================Checking in extensions==============================
print("Checking in extensions...")
for extension in extensions:
    if arcpy.CheckExtension(extension) == "Available":
        arcpy.CheckInExtension(extension)
#==============================================================================
    
print("Finished Sucessfully!")