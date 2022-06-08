#====================================JRA55=====================================
'''
Purpose:
    Reprojects, resamples and masks out JMA Radar Precipitation.
    Inputs are netcdf files
    Writes output to netcdf and binary (both are optional)
    Note: It is assumed that the time is already JST
    
    Data Source: JMA C-Band Radar (Corrected by AMeDAS)
    
       Record of revision:
           Date            Author          Description of change
           ====            ======          =====================
           2018/10/25      Abdul Moiz      Original code (Developer Version)

'''
#==============================================================================

#============================Importing Modules=================================

print("Importing modules...")
import os
import numpy as np
import pandas as pd
import xarray as xr
import glob
import linecache
import pyproj as proj
from scipy.interpolate import griddata
import multiprocessing
from functools import partial
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
#------------------------------------------------------------------------------  
def proc_ppt(month,year,sel_year,
                     project,output,data,mapping,netcdf,
                     xi,yi,x,y,wsheds_mask,
                     write_netcdf,write_binary):
    sel_month = sel_year.sel(time=str(year)+'-'+str(month).zfill(2))
    
    print(str(year),str(month).zfill(2))
    
    #Regridding & Writing to Direct-Access Fortran
    arr_fin = []
    if write_binary == True:
        f = open(os.path.join(project,output,data,'PPT_RAD'+str(year)+'_'+str(month).zfill(2)+'.direct'),'wb')
    for t in sel_month.time:
        arr = (sel_month.sel(time=t)).values
        arr = griddata((x.flatten(), y.flatten()), arr.flatten(), (xi,yi), method = 'linear')
        arr = np.flip(arr,0)
        arr = np.ma.masked_where(wsheds_mask, arr)
        arr = np.ma.filled(arr, -9999.0)
        arr_fortran = np.asfortranarray(arr,'float32')
        arr[wsheds_mask] = np.nan
        if write_binary == True:
            arr_fortran.tofile(f)
        arr_fin.append(arr) # For saving to netcdf
    arr_fin = np.stack(arr_fin)
    if write_netcdf == True:
        arr_xr = xr.DataArray(arr_fin, coords=[sel_month.time.values, yi[::-1,0], xi[0,:]], dims=['time', 'y', 'x'])
        arr_xr.to_netcdf(os.path.join(project, mapping, netcdf, 'PPT_RAD'+str(year)+'_'+str(month).zfill(2) + '.nc'))
    
    if write_binary == True:
        f.close()
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
jmaradar_folder = '/mnt/disk1/repository/data/jma-radar/jmatonc/subsets/kurobe/10min_aggregated/utc'
netcdf = 'jmaradar'

#Write Options
write_netcdf = True
write_binary = True

#Time
start_time = '2010-01-01 00:00:00'
end_time = '2014-12-31 23:00:00'

#Coordinate systems
crs_wgs = proj.Proj(init='epsg:4326') # WGS84
crs_utm = proj.Proj(init='epsg:32653') # UTM

t_offset = 9    # Local_Time - UTC = Time Offset (Hours)
#==============================================================================

#====================Setting Environment Variables=============================
print("Setting enviornment variables...")
os.chdir(project)
outpaths(scratch, output, data, parameter, mapping, netcdf)
#==============================================================================
cell_area = os.path.join(project, output, parameter, 'cell_area.asc')

#Setting Regrid
cell_area = read_ascii_grid(cell_area)
wsheds = cell_area[0]
wsheds_mask = np.ma.getmask(wsheds)
ncols = cell_area[1]
nrows = cell_area[2]
cellsize = cell_area[5]
xll = cell_area[3]
yll = cell_area[4]
xur = xll + cellsize*ncols
yur = yll + cellsize*nrows

#Target Grid (xi,yi)
xi = np.linspace(xll + 0.5*cellsize, xur - 0.5*cellsize, ncols)  
yi = np.linspace(yll + 0.5*cellsize, yur - 0.5*cellsize, nrows)
xi, yi = np.meshgrid(xi,yi)

print('Processing ...')
ds = xr.open_mfdataset(sorted(glob.glob(os.path.join(jmaradar_folder,'*.nc'))))
ds = ds.resample(time='1H').mean()
ds = ds['Precipitation']
ds = ds.shift(time=t_offset)          # UTC --> Local Time
ds = ds.sel(time=slice(start_time,end_time))

#Original Grid (x,y)
x = ds.lon.values
y = ds.lat.values
x, y = np.meshgrid(x,y)
x, y = proj.transform(crs_wgs, crs_utm, x, y)       # WGS --> UTM

#Setting Pool for multiprocessing
pool = multiprocessing.Pool(multiprocessing.cpu_count())

years = pd.to_datetime(ds.time.values).year.unique()

for year in years:
    sel_year = ds.sel(time=str(year))
    months = pd.to_datetime(sel_year.time.values).month.unique()
    
    pool.map(partial(proc_ppt,year=year,sel_year=sel_year,
                     project=project,output=output,data=data,mapping=mapping,netcdf=netcdf,
                     xi=xi,yi=yi,x=x,y=y,wsheds_mask=wsheds_mask,
                     write_netcdf=write_netcdf,write_binary=write_binary), months)
        
                
print("Finished Sucessfully!")