#=============================Temperature Correction=====================================
'''
Purpose:
    This script reads the JRA55 vertical profile for lapse rate and calculates
    the elevation corrected surface temperature
    
    Need to write more explanation here.
    
       Record of revision:
           Date            Author          Description of change
           ====            ======          =====================
           2018/09/27      Abdul Moiz      Original code (Developer Version)
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
def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)
#------------------------------------------------------------------------------
def simple_idw(x, y, z, xi, yi, power = 2):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / ((dist)**power)

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi
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
jra55_folder = 'C:/Cloud/GoogleDrive/Research/Doctoral/4_WEBDHM/kurobe/data_prep_arcgis/inputs/jra55_3d/anl_p125'
tair_pkl = 'C:/Cloud/GoogleDrive/Research/Doctoral/1_Data_Collection/kurobe/database/insitu/pickled/kansai_tair.pkl'
tair_pkl_info = 'C:/Cloud/GoogleDrive/Research/Doctoral/1_Data_Collection/kurobe/database/insitu/pickled/kansai_stations.pkl'
netcdf = 'corr_tair_jra55'

#Write Options
write_netcdf = True
write_binary = True

#Time
start_time = '2010-01-01 00:00:00'
end_time = '2010-01-02 23:00:00'
t_offset = 9    # Local_Time - UTC = Time Offset (Hours)

#Coordinate systems
crs_wgs = proj.Proj(init='epsg:4326') # WGS84
crs_utm = proj.Proj(init='epsg:32653') # UTM


stations = ['Kurobe','Dashidaira']
#==============================================================================

#====================Setting Environment Variables=============================
print("Setting enviornment variables...")
os.chdir(project)
outpaths(scratch, output, data, parameter, mapping, netcdf)
#==============================================================================



cell_area = os.path.join(project, output, parameter, 'cell_area.asc')
elevation = os.path.join(project, output, parameter, 'elevation.asc')

obs_info = (pd.read_pickle(tair_pkl_info))[stations]
tmp_obs = (pd.read_pickle(tair_pkl) + 273.15)[stations]
tmp_obs = tmp_obs.interpolate()

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


print('Processing..')
jra55_3d = xr.open_mfdataset(sorted(glob.glob(os.path.join(jra55_folder,'*.nc'))))
jra55_3d.load()
jra55_3d = jra55_3d.resample(time='1H').interpolate('linear')    # 3H  --> 1H
jra55_3d = jra55_3d.shift(time=t_offset)                         # UTC --> Local Time
jra55_3d = jra55_3d.sel(time=slice(start_time,end_time))

#Original JRA55 Grid (x,y)
x = jra55_3d.lon.values
y = jra55_3d.lat.values
x, y = np.meshgrid(x,y)
x, y = proj.transform(crs_wgs, crs_utm, x, y)       # WGS --> UTM

#Original Observed Station Points (xp,yp)
xp, yp = proj.transform(crs_wgs, crs_utm, 
                        obs_info.loc['Longitude'].values, obs_info.loc['Latitude'].values)       # WGS --> UTM

#Read Elevation
elevation = read_ascii_grid(elevation)[0]
elevation = np.stack([elevation]*len(jra55_3d.lev))


#Interpolation Observed Station Elevation
obs_hgt = obs_info.loc['Model Elevation'].values  #Can choose only elevation
obs_hgt = simple_idw(xp,yp,obs_hgt,xi.flatten(),yi.flatten())
obs_hgt = obs_hgt.reshape(nrows,ncols)
obs_hgt = np.flip(obs_hgt,0)
obs_hgt = np.stack([obs_hgt]*len(jra55_3d.lev))

years = pd.to_datetime(jra55_3d.time.values).year.unique()

for year in years:
    jra55_3d_sel_year = jra55_3d.sel(time=str(year))
    months = pd.to_datetime(jra55_3d_sel_year.time.values).month.unique()
    
    for month in months:

        jra55_3d_sel_month = jra55_3d_sel_year.sel(time=str(year)+'-'+str(month).zfill(2))
        
        print(year,str(month).zfill(2))
        
        #Regridding & Writing to Direct-Access Fortran
        tmp_corrected_arr_fin = []
        lapse_arr_fin = []
        bias_arr_fin = []
        tmp_obs_arr_fin = []
        
        
        if write_binary == True:
            f = open(os.path.join(project,output,data,'Tair_C'+str(year)+'_'+str(month).zfill(2)+'.direct'),'wb')
            
        for t in jra55_3d_sel_month.time:
            print(t.values)
            
            #Interpolating Observed Temperature spaptially
            obs_tmp_t = tmp_obs.loc[t.values].values
            obs_tmp_t = simple_idw(xp,yp,obs_tmp_t,xi.flatten(),yi.flatten())
            obs_tmp_t = obs_tmp_t.reshape(nrows,ncols)
            obs_tmp_t = np.flip(obs_tmp_t,0)
            
            
            jra55_3d_t = (jra55_3d_sel_month.sel(time=t))
            
            jra55_3d_t_hgt_arr = []
            jra55_3d_t_tmp_arr = []
            
            for lev in jra55_3d.lev:
                jra55_3d_l = jra55_3d_t.sel(lev=lev)
                
                jra55_3d_l_hgt = griddata((x.flatten(), y.flatten()), jra55_3d_l.hgtprs.values.flatten(), (xi,yi), method = 'linear')
                jra55_3d_l_hgt = np.flip(jra55_3d_l_hgt,0)
                
                jra55_3d_l_tmp = griddata((x.flatten(), y.flatten()), jra55_3d_l.tmpprs.values.flatten(), (xi,yi), method = 'linear')
                jra55_3d_l_tmp = np.flip(jra55_3d_l_tmp,0)
                
                jra55_3d_t_hgt_arr.append(jra55_3d_l_hgt)
                jra55_3d_t_tmp_arr.append(jra55_3d_l_tmp)
            
            jra55_3d_t_hgt_arr=np.stack(jra55_3d_t_hgt_arr)
            jra55_3d_t_tmp_arr=np.stack(jra55_3d_t_tmp_arr)
            
            
            #Calcuating Lapse Rate
            u_arg = np.argmin(np.ma.masked_where(jra55_3d_t_hgt_arr<elevation,jra55_3d_t_hgt_arr),0)
            l_arg = np.argmax(np.ma.masked_where(jra55_3d_t_tmp_arr>elevation,jra55_3d_t_tmp_arr),0)
            
            I,J = np.ogrid[:u_arg.shape[0],:u_arg.shape[1]]
            
            u_hgt = np.ma.masked_where(wsheds_mask,jra55_3d_t_hgt_arr[u_arg,I,J])
            l_hgt = np.ma.masked_where(wsheds_mask,jra55_3d_t_hgt_arr[l_arg,I,J])
            
            u_tmp = np.ma.masked_where(wsheds_mask,jra55_3d_t_tmp_arr[u_arg,I,J])
            l_tmp = np.ma.masked_where(wsheds_mask,jra55_3d_t_tmp_arr[l_arg,I,J])
            
            lapse = (u_tmp-l_tmp)/(u_hgt-l_hgt)
            
            #Calulating BIAS
            u_arg = np.argmin(np.ma.masked_where(jra55_3d_t_hgt_arr<obs_hgt,jra55_3d_t_hgt_arr),0)
            l_arg = np.argmax(np.ma.masked_where(jra55_3d_t_tmp_arr>obs_hgt,jra55_3d_t_tmp_arr),0)
            
            I,J = np.ogrid[:u_arg.shape[0],:u_arg.shape[1]]
            
            u_hgt_jra55_at_obs = np.ma.masked_where(wsheds_mask,jra55_3d_t_hgt_arr[u_arg,I,J])
            l_hgt_jra55_at_obs = np.ma.masked_where(wsheds_mask,jra55_3d_t_hgt_arr[l_arg,I,J])
            
            u_tmp_jra55_at_obs = np.ma.masked_where(wsheds_mask,jra55_3d_t_tmp_arr[u_arg,I,J])
            l_tmp_jra55_at_obs = np.ma.masked_where(wsheds_mask,jra55_3d_t_tmp_arr[l_arg,I,J])
            
            
            tmp_jra55_at_obs = ((u_tmp_jra55_at_obs-l_tmp_jra55_at_obs)/(u_hgt_jra55_at_obs - l_hgt_jra55_at_obs))*(obs_hgt[0]-l_hgt_jra55_at_obs)+l_tmp_jra55_at_obs
            
            bias = obs_tmp_t - tmp_jra55_at_obs
            
            
            # Calculating Corrected Temperature
            t_temp_corrected = lapse*(elevation[0] - l_hgt) + l_tmp
            t_temp_corrected = t_temp_corrected + bias
            
            if write_binary == True:
                arr_fortran = np.ma.masked_where(wsheds_mask, t_temp_corrected)
                arr_fortran = np.ma.filled(arr_fortran, -9999.0)
                arr_fortran = np.asfortranarray(arr_fortran,'float32')
                arr_fortran.tofile(f)
            
            #Masking out wsheds
            t_temp_corrected[wsheds_mask] = np.nan
            lapse[wsheds_mask] = np.nan
            bias[wsheds_mask] = np.nan
            obs_tmp_t[wsheds_mask] = np.nan
            
            
            #Appending variables for each month
            tmp_corrected_arr_fin.append(t_temp_corrected)
            lapse_arr_fin.append(lapse)
            bias_arr_fin.append(bias)
            tmp_obs_arr_fin.append(obs_tmp_t)
            
            
            
        #Stacking for each month
        tmp_corrected_arr_fin = np.stack(tmp_corrected_arr_fin)
        lapse_arr_fin = np.stack(lapse_arr_fin)
        bias_arr_fin = np.stack(bias_arr_fin)
        tmp_obs_arr_fin = np.stack(tmp_obs_arr_fin)
        
        if write_netcdf == True:  
            ds_out = xr.Dataset({'corr_t2m':(['time','y','x'], tmp_corrected_arr_fin),
                                 'lapse_rate':(['time','y','x'],lapse_arr_fin),
                                 'bias':(['time','y','x'],bias_arr_fin),
                                 'obs_t2m':(['time','y','x'],tmp_obs_arr_fin)},
                                coords = {'time':(['time'],jra55_3d_sel_month.time.values), 
                                          'y':(['y'],yi[::-1,0]),
                                          'x':(['x'], xi[0,:])})
            ds_out.to_netcdf(os.path.join(project, mapping, netcdf, 'Tair_corr_'+str(year)+'_'+str(month).zfill(2) + '.nc'))
            
        if write_binary == True:
            f.close()
            
obs_hgt = obs_hgt[0]
obs_hgt[wsheds_mask] = np.nan
obs_hgt_ds = xr.Dataset({'obs_hgt':(['y','x'],obs_hgt)},coords = {'y':(['y'],yi[::-1,0]),'x':(['x'], xi[0,:])})            
obs_hgt_ds.to_netcdf(os.path.join(project, mapping, netcdf, 'obs_hgt.nc'))
print('Finished Successfully!')    
                