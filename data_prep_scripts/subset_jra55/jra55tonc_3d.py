import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from py3grads import Grads

def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

variables = ['anl_p125_hgt','anl_mdl_hgt','anl_p125_tmp','anl_mdl_tmp']
save = '/dias/groups/webdhm-01/moiz/webdhm/kurobe/data_repository/jra55'


base = '/dias/data/jra55/Hist/Daily'
str_year = 2000
str_mon = 1
end_year = 2017
end_mon = 12
years = range(str_year,end_year+1)

ga = Grads(verbose=False)
for var in variables:
    for year in years:
        if (year == str_year) and (year == end_year):
            months = range(str_mon, end_mon + 1)
        elif (year == str_year):
            months = range(str_mon, 12 + 1)
        elif (year == end_year):
            months = range(1, end_mon + 1)
        else:
            months = range(1, 12 + 1)
        
        for month in months:
            print(var,str(year)+str(month).zfill(2))
            inpath = os.path.join(base,var.split('_')[0]+'_'+var.split('_')[1],str(year)+str(month).zfill(2),var+'.ctl')
            outpath = os.path.join(save,var.split('_')[0]+'_'+var.split('_')[1])
            makedir(outpath)
            
            ga('run gradstonc_3d.gs ' + inpath + ' ' + outpath)
            output = xr.open_mfdataset(glob.glob(os.path.join(outpath,'*_temp.nc')))
            output.to_netcdf(os.path.join(outpath,var+'_'+str(year)+str(month).zfill(2)+'.nc'))
            for temp in glob.glob(os.path.join(outpath,'*_temp.nc')):
                os.remove(temp)
