#================================MODIS LAI FPAR==================================
'''
Purpose:
    Reprojects, resamples and masks out MODIS LAI FPAR.
    Followed by wrinting to Binary File.
    Valid values are 0-100. Fill values are 249-255.
    Scaling is not done here as it is done inside the model
    
    Data Source: MCD15A2H Product (MODIS LAI FPAR)
                 Downloaded from AppEEARS
    
       Record of revision:
           Date            Author          Description of change
           ====            ======          =====================
           2018/09/27      Abdul Moiz      Original code (Developer Version)
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
import xarray as xr
import glob
import linecache

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
def outpaths(scratch, output, data, parameter, mapping, netcdf):
    createfolder(scratch)
    createfolder(output)
    createfolder(os.path.join(output, data))
    createfolder(os.path.join(output, parameter))
    createfolder(os.path.join(mapping, netcdf))
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
modis_folder = 'modis_laifpar/geotiff'
netcdf = 'modis_laifpar'

#Write Options
write_netcdf = True
write_binary = True

#Time
start_year = 2015
end_year = 2018

#Variables
varnames = ['Fpar', 'Lai']

#====================Setting Environment Variables=============================
print("Setting enviornment variables...")
os.chdir(project)
outpaths(scratch, output, data, parameter, mapping, netcdf)

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

#=================================Main=========================================
cell_area = arcpy.Raster(os.path.join(project, scratch, 'cell_area.tif'))
target_cellsize = arcpy.Describe(cell_area).meanCellHeight
target_projection = arcpy.Describe(cell_area).SpatialReference
arcpy.env.extent = cell_area

modis_input_folder = os.path.join(project, inputs, modis_folder)


#Setting Regrid
cell_area_ascii = read_ascii_grid(os.path.join(project, output, parameter, 'cell_area.asc'))
wsheds = cell_area_ascii[0]
wsheds_mask = np.ma.getmask(wsheds)
ncols = cell_area_ascii[1]
nrows = cell_area_ascii[2]
cellsize = cell_area_ascii[5]
xll = cell_area_ascii[3]
yll = cell_area_ascii[4]
xur = xll + cellsize*ncols
yur = yll + cellsize*nrows

xi = np.linspace(xll + 0.5*cellsize, xur - 0.5*cellsize, ncols)  #Check if this is needed
yi = np.linspace(yll + 0.5*cellsize, yur - 0.5*cellsize, nrows)


print('Processing ...')

for var in varnames:
    for year in range(start_year, end_year + 1): 
        print(var + ': ' + str(year))
        arr_fin = []
        
        #Open file for Direct-Access Fortran
        filename = var.upper() + '.' + str(year)
        
        if write_binary == True:
            f = open(os.path.join(project, output, data, filename + '.direct'), 'wb')
        
    
        modis_files = glob.glob(os.path.join(modis_input_folder, '*_' + var + '*_doy' + str(year) + '*_*.tif'))
        modis_files.sort()
        date_range = pd.date_range(pd.datetime(year, 01, 01),pd.datetime(year, 12, 31),freq = '8D')
        
        if len(modis_files) != len(date_range):
            date_range = date_range[:len(modis_files)]
        
        for modis_file, date in zip(modis_files, date_range):
            print(date)
            inRaster = arcpy.Raster(modis_file)
            outRaster = os.path.join(project, scratch, var + '_modis.tif')
            arcpy.ProjectRaster_management(inRaster, outRaster, target_projection, 'NEAREST', \
                               target_cellsize)
            outRaster = arcpy.Raster(outRaster)
            modis_subset = arcpy.sa.ExtractByMask(outRaster, cell_area)
            arr = (arcpy.RasterToNumPyArray(modis_subset,nodata_to_value=255)).astype(np.float32)
            arr[np.isnan(arr)] = -9999.0
            arr[arr >=249.0] = -9999.0 #From documentation
            arr = np.ma.masked_where(wsheds_mask, arr)
            arr = np.ma.filled(arr, -9999.0)

            #Writing to Direct_Access Fortran
            arr_fortran = np.asfortranarray(arr, 'float32')
            
            arr[arr == -9999.0] = np.nan
            if write_binary == True:
                arr_fortran.tofile(f)
            
            arr_fin.append(arr)
        arr_fin = np.stack(arr_fin)
        
        if write_netcdf == True:
            arr_xr = xr.DataArray(arr_fin, coords=[date_range, yi[::-1], xi], dims=['time', 'y', 'x'])
            arr_xr.to_netcdf(os.path.join(project, mapping, netcdf, var.upper() + '.' + str(year) + '.nc'))
        
        if write_binary == True:
            f.close()
        
        
print("Finished Sucessfully!")
