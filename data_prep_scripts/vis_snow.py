#NOTE!!!: This script needs to be further refined

import xarray as xr
import pandas as pd
import numpy as np
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
import matplotlib.patches as mpatches
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.img_tiles import GoogleTiles


#========================Defining Custom Functions=============================
#------------------------------------------------------------------------------
def createfolder(folderName):
    if os.path.isdir(folderName)==False:
        os.mkdir(folderName)
#------------------------------------------------------------------------------
class ShadedReliefESRI(GoogleTiles):
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
               'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg').format(
               z=z, y=y, x=x)
        return url
#==============================================================================
var = 'MAXIMUM_SNOW_EXTENT' #LAI or FPAR        
utm = 53
in_folder = os.path.join('..', 'data_prep_arcgis', 'mapping', 'modis_snow')
out_folder = os.path.join(in_folder, 'figures')
basin = os.path.join('..', 'data_prep_arcgis', 'mapping', 'coarseBasin.shp')

snow_nc = glob.glob(os.path.join(in_folder, var + '*.nc'))
start = '2003-01-01'
end = '2020-12-31'

#Plot Properties
extents = [137.5, 137.8, 36.38, 36.91]
xticks  = np.linspace(137.5, 138.5, 11).tolist()
yticks  = np.linspace(36.0, 37.0, 11).tolist()

flatui = ['#7CFC00','#1E90FF','#A9A9A9'] #Green Gray Green Blue Green
cmap = colors.ListedColormap(sns.color_palette(flatui).as_hex())
bounds=[1,2,3,5]
norm = colors.BoundaryNorm(bounds, cmap.N)

snow_patch = mpatches.Patch(facecolor='#1E90FF', label='Snow',edgecolor='k')
land_patch = mpatches.Patch(facecolor='#7CFC00', label='Land',edgecolor='k')
cloud_patch = mpatches.Patch(facecolor='#A9A9A9', label = 'Cloud',edgecolor='k')

#Main
createfolder(out_folder)
basin = ShapelyFeature(shpreader.Reader(basin).geometries(), ccrs.UTM(utm), edgecolor='black')

#Subsetting
snow_nc = xr.open_mfdataset(snow_nc)
snow_nc = snow_nc['Snow Covered Area'].sel(time=slice(start,end))

snow_nc = snow_nc.where((snow_nc != -9999) & 
                        (snow_nc != 0) &
                        (snow_nc != 11) &
                        (snow_nc != 254))

snow_nc = snow_nc.where((snow_nc != 11) &
                        (snow_nc != 25) &
                        (snow_nc != 37) &
                        (snow_nc != 39) &
                        (snow_nc != 100), 1) #Land == 0

snow_nc = snow_nc.where(snow_nc != 200, 2)   #Snow == 1

snow_nc = snow_nc.where(snow_nc != 50, 4)    #Cloud == 2

for t in range(len(snow_nc)):
    print(t)
    plt.figure(figsize = (14,6))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True, alpha = 0.15, linestyle = '--', color = 'k')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    gl.xlabel_style = {'size':10, 'rotation':0, 'color': 'black'}
    gl.ylabel_style = {'size':10, 'rotation':0, 'color': 'black'}
    gl.xlabels_top = False
    gl.ylabels_right = False
    
    snow_plot = snow_nc.isel(time = t).plot(ax=ax, 
                                            transform = ccrs.UTM(utm),
                                            cmap = cmap, norm=norm)
    ax.add_feature(basin,facecolor = 'none')
    ax.set_extent(extents)
    snow_plot.colorbar.remove()
    plt.legend(handles=[snow_patch,land_patch,cloud_patch], loc='lower center', ncol=1, bbox_to_anchor=(0.75,0), prop={'size':15}, shadow = True, edgecolor = 'k')
    ax.set_title('Time = ' + str(snow_nc.isel(time = t).time).split("'")[3].split('.')[0], 
                 fontsize = 12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, str(t).zfill(3) + '.png'), dpi = 300, bbox_inches = 'tight')
    


