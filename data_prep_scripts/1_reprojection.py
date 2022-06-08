#================================Reprojection==================================
'''
Purpose:
    To reproject an input Raster (Digital Elevation Model) to target projection 
    system.

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
project = '../data_prep_arcgis'
scratch = 'scratch'
inputs = 'inputs'
output = 'output'
data = 'data'
parameter = 'parameter'
mapping = 'mapping'

#Inputs
inDEM = 'GSI_5m.tif'
target_projection = 'WGS 1984 UTM Zone 53N'
target_cellsize = 10
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
inDEM = arcpy.Raster(os.path.join(inputs, inDEM))
outDEM = os.path.join(project, scratch, 'fineDEM.tif')
target_projection = arcpy.SpatialReference(target_projection)
arcpy.ProjectRaster_management(inDEM, outDEM, target_projection, 'BILINEAR', \
                               target_cellsize)
#==============================================================================

#==========================Checking in extensions==============================
print("Checking in extensions...")
for extension in extensions:
    if arcpy.CheckExtension(extension) == "Available":
        arcpy.CheckInExtension(extension)
#==============================================================================
    
print("Finished Sucessfully!")