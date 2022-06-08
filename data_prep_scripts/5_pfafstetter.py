#================================Reprojection==================================
'''
Purpose:
    This script performs Pfafstetter delineation. Basins can be delineated upto
    Level 3.

       Record of revision:
           Date            Author          Description of change
           ====            ======          =====================
           2018/02/10      Abdul Moiz      Original code (Developer Version)
           2018/09/27      Abdul Moiz      Correction made for Level 2 or higher
                                           (A bug was fixed which caused some sub-basins
                                           to be defined by an empty grid)
'''
#==============================================================================

#============================Importing Modules=================================
print("Importing modules...")
import archook
archook.get_arcpy(pro=True)
import arcpy
import os
from simpledbf import Dbf5
import pandas as pd
import numpy as np
from collections import OrderedDict
import linecache
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
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
#------------------------------------------------------------------------------
def subbasin_path(a_dir, key):
    path = get_immediate_subdirectories(a_dir)
    folder = [s for s in path if s.endswith(str(key))][0]
    path = os.path.join(a_dir, folder)
    return path
#------------------------------------------------------------------------------
def outpaths(scratch, output, data, parameter, mapping):
    createfolder(scratch)
    createfolder(output)
    createfolder(os.path.join(output, data))
    createfolder(os.path.join(output, parameter))
    createfolder(mapping)
#------------------------------------------------------------------------------
def basin_directories(root_dir, level, level0, level1, level2):
    
    id_level0 = np.zeros((1), object)
    id_level1 = np.zeros((9), object)
    id_level2 = np.zeros((9,9), object)
    id_level3 = np.zeros((9,9,9), object)
    
    id_level0[:] = np.nan
    id_level1[:] = np.nan
    id_level2[:,:] = np.nan
    id_level3[:,:,:] = np.nan
    
    
    #Level 0
    path_level0 = os.path.join(root_dir, 'basin' + str(level0[0]))
    createfolder(path_level0)
    id_level0[0] = path_level0
    if level == 1:
        #Level 1
        for i in range(1,10):
            path_level1 = os.path.join(path_level0, 'basin' + str(i))
            createfolder(path_level1)
            id_level1[i-1] = path_level1
            
    elif level == 2:
        #Level 1
        for i in range(1,10):
            path_level1 = os.path.join(path_level0, 'basin' + str(i))
            createfolder(path_level1)
            id_level1[i-1] = path_level1
        #Level 2
        for level1_id in level1:
            for j in range(1,10):
                path_level2 = os.path.join(path_level0, 'basin' + str(level1_id)[0], 'basin' + str(level1_id) + str(j))
                createfolder(path_level2)
                id_level2[int(str(level1_id)[0])-1,j-1] = path_level2#int(str(level1_id) + str(j))
                
    elif level == 3:
        #Level 1
        for i in range(1,10):
            path_level1 = os.path.join(path_level0, 'basin' + str(i))
            createfolder(path_level1)
            id_level1[i-1] = path_level1
        #Level 2
        for level1_id in level1:
            for j in range(1,10):
                path_level2 = os.path.join(path_level0, 'basin' + str(level1_id)[0], 'basin' + str(level1_id) + str(j))
                createfolder(path_level2)
                id_level2[int(str(level1_id)[0])-1,j-1] = path_level2#int(str(level1_id) + str(j))
        #Level 3
        for level2_id in level2:
            for k in range(1,10):
                path_level3 = os.path.join(path_level0, 'basin' + str(level2_id)[0], 'basin' + str(level2_id), 'basin' + str(level2_id) + str(k))
                createfolder(path_level3)
                id_level3[int(str(level2_id)[0])-1,int(str(level2_id)[1])-1,k-1] = path_level3#int(str(level2_id) + str(k))
    
    return id_level0, id_level1, id_level2, id_level3
#------------------------------------------------------------------------------
def FeatureToDataFrame(acc, dis, link): #Shapefiles
    acc_list = []
    dis_list = []
    link_list =[]
    to_list = []
    from_list = []
    fcs = [acc, dis, link]
    for fc in fcs:
        df = pd.DataFrame()
        cursor = arcpy.da.SearchCursor(fc, ['GRID_CODE', 'FROM_NODE', 'TO_NODE'])
        for row in cursor:
            if fc == acc:
                acc_list.append(row[0])
            elif fc == dis:
                dis_list.append(row[0])
            elif fc == link:
                link_list.append(row[0])
                to_list.append(row[2])
                from_list.append(row[1])
    df = {'link':link_list, 'fac':acc_list, 'len':dis_list, 'dn_node':to_list, 'up_node':from_list}
    df = pd.DataFrame(data=df)
    df = df[['link', 'fac', 'len', 'dn_node', 'up_node']]
    return df
#------------------------------------------------------------------------------
def pfaf(network, point):
    outlet = network.loc[stream_df.fac.idxmax()]
    outletxy = point.loc[point.fac.idxmax()]
    
    link_list = [outlet.link]
    dn_node_list = [outlet.dn_node]
    up_node_list = [outlet.up_node]
    fac_list = [outlet.fac]
    len_list = [outlet.len]
    x_list = [outletxy.x]
    y_list = [outletxy.y]
    
    #Initialize    
    link1 = outlet
    possible_link2 = network
    
    while not(possible_link2.empty):
        possible_link2 = network.loc[network.dn_node == link1.up_node]
        link2 = possible_link2.loc[possible_link2.fac.idxmax()]
        link2_xy = (point.loc[point.link == link2.link])
        #Update Lists
        link_list.append(link2.link)
        dn_node_list.append(link2.dn_node)
        up_node_list.append(link2.up_node)
        fac_list.append(link2.fac)
        len_list.append(link2.len)
        x_list.append(link2_xy.loc[link2_xy.fac.idxmax()].x)
        y_list.append(link2_xy.loc[link2_xy.fac.idxmax()].y)
        
        link1 = link2
        possible_link2 = network.loc[network.dn_node == link1.up_node]
    
    main_river = {'link':link_list, 'fac':fac_list, 'len':len_list, 'x':x_list, 'y':y_list, 'dn_node':dn_node_list, 'up_node':up_node_list}
    main_river = pd.DataFrame(data=main_river)
    main_river = main_river[['link','fac','len','x','y','dn_node','up_node']]
    #Identifying Tributaries
    tributaries_network = network.loc[~network.link.isin(main_river.link)]
    tributaries_outlet = tributaries_network.loc[tributaries_network.dn_node.isin(main_river.dn_node)]
    
    tributaries_outlet = tributaries_outlet.sort_values(by='fac', ascending=False)
    tributaries_outlet = tributaries_outlet.groupby('dn_node').first()
    tributaries_outlet.reset_index(inplace=True)
    if len(tributaries_outlet) >= 4:
        tributaries_outlet = (tributaries_outlet.sort_values(by='fac', ascending = False)).head(n=4)
        
    else:
        tributaries_outlet = (tributaries_outlet.sort_values(by='fac', ascending = False)).head(n=len(tributaries_outlet))
    tributaries_outlet = tributaries_outlet.sort_values(by='len', ascending = True) 
    tributaries = []
    for index, tributary_outlet in tributaries_outlet.iterrows():
        tributary_outlet_xy = (point.loc[(point.link == tributary_outlet.link) & (point.fac == tributary_outlet.fac)]).iloc[0]
        link_list = [tributary_outlet.link]
        dn_node_list = [tributary_outlet.dn_node]
        up_node_list = [tributary_outlet.up_node]
        fac_list = [tributary_outlet.fac]
        len_list = [tributary_outlet.len]
        x_list = [tributary_outlet_xy.x]
        y_list = [tributary_outlet_xy.y]
        
        link1 = tributary_outlet
        possible_link2 = network
        while not(possible_link2.empty):
            possible_link2 = network.loc[network.dn_node == link1.up_node]
            if not possible_link2.empty:
                link2 = possible_link2.loc[possible_link2.fac.idxmax()]
                link2_xy = (point.loc[point.link == link2.link])
                #Update Lists
                link_list.append(link2.link)
                dn_node_list.append(link2.dn_node)
                up_node_list.append(link2.up_node)
                fac_list.append(link2.fac)
                len_list.append(link2.len)
                x_list.append(link2_xy.loc[link2_xy.fac.idxmax()].x)
                y_list.append(link2_xy.loc[link2_xy.fac.idxmax()].y)
                
                link1 = link2
                possible_link2 = network.loc[network.dn_node == link1.up_node]
        tributary = {'link':link_list, 'fac':fac_list, 'len':len_list, 'x':x_list, 'y':y_list, 'dn_node':dn_node_list, 'up_node':up_node_list}
        tributary = pd.DataFrame(data=tributary)
        tributary = tributary[['link','fac','len','x','y','dn_node','up_node']]
        tributaries.append(tributary)
    tributaries_outlet = []
    for t in tributaries:
        tributaries_outlet.append(t.iloc[[0]])
    tributaries_outlet = pd.concat(tributaries_outlet)
    tributaries_outlet.sort_values(by = 'len', ascending = True, inplace = True)
    tributaries_outlet.reset_index(inplace = True, drop = True)
    tributaries_outlet.index = range(2, len(tributaries_outlet)*2+1, 2)
    tributaries_outlet.sort_values(by = 'len', ascending = False, inplace = True)
    
    main_river_outlet = main_river.loc[main_river.dn_node.isin(tributaries_outlet.dn_node)]
    main_river_outlet = main_river_outlet.loc[main_river_outlet.link.isin(point.link)]
    main_river.dn_node = main_river.dn_node.astype('int64')
    main_river.up_node = main_river.up_node.astype('int64')
    
    main_river_outlet = pd.concat([main_river_outlet, main_river.iloc[[0]]])
    main_river_outlet.sort_values(by='fac', ascending=False, inplace = True)
    main_river_outlet.reset_index(inplace = True, drop = True)
    temp = range(1, len(main_river_outlet)*2, 2)
#    temp[-1] = 9
    main_river_outlet.index = temp
    main_river_outlet.sort_values(by='fac', ascending=True, inplace = True)
    return main_river_outlet, tributaries_outlet
#------------------------------------------------------------------------------
def createPoint(x, y, sptl_ref):
    point = arcpy.Point(x,y)
    pointgeometry = arcpy.Geometry('point', point, sptl_ref)
    return pointgeometry
#------------------------------------------------------------------------------
def pfaf_delineate(main_river_outlet, tributaries_outlet, fdr, sptl_ref, path):
    watershed_list = []
    for i in main_river_outlet.index:
        if i == (len(main_river_outlet) + len(tributaries_outlet)):
            folder = subbasin_path(path, i-1)
            x_coord = tributaries_outlet.x[i-1]
            y_coord = tributaries_outlet.y[i-1]
            outlet = createPoint(x_coord, y_coord, sptl_ref)
            basin_trib = arcpy.sa.Watershed(fdr, outlet)
            
            folder = subbasin_path(path, 9)
            x_coord = main_river_outlet.x[i]
            y_coord = main_river_outlet.y[i]
            outlet = createPoint(x_coord, y_coord, sptl_ref)
            basin = arcpy.sa.Watershed(fdr, outlet)
            watershed_name = folder.split('\\')[-1]
            basin_main = arcpy.sa.SetNull(~(arcpy.sa.IsNull(basin_trib)),basin)
            watershed = arcpy.RasterToPolygon_conversion(basin_main, os.path.join(folder, watershed_name + '.shp'), "NO_SIMPLIFY")
            watershed_list.append(str(watershed))
        if (i in range(3,tributaries_outlet.index[0],2)) and (len(main_river_outlet.index) >= 3):
            folder = subbasin_path(path, i+1)
            x_coord = tributaries_outlet.x[i+1]
            y_coord = tributaries_outlet.y[i+1]
            outlet = createPoint(x_coord, y_coord, sptl_ref)
            basin_trib_up = arcpy.sa.Watershed(fdr, outlet)
            watershed_name = folder.split('\\')[-1]
            watershed = arcpy.RasterToPolygon_conversion(basin_trib_up, os.path.join(folder, watershed_name + '.shp'), "NO_SIMPLIFY")
            watershed_list.append(str(watershed))
            
            folder = subbasin_path(path, i-1)
            x_coord = tributaries_outlet.x[i-1]
            y_coord = tributaries_outlet.y[i-1]
            outlet = createPoint(x_coord, y_coord, sptl_ref)
            basin_trib_dn = arcpy.sa.Watershed(fdr, outlet)
            
            folder = subbasin_path(path, i+2)
            x_coord = main_river_outlet.x[i+2]
            y_coord = main_river_outlet.y[i+2]
            outlet = createPoint(x_coord, y_coord, sptl_ref)
            basin_up = arcpy.sa.Watershed(fdr, outlet)

            folder = subbasin_path(path, i)
            x_coord = main_river_outlet.x[i]
            y_coord = main_river_outlet.y[i]
            outlet = createPoint(x_coord, y_coord, sptl_ref)
            basin = arcpy.sa.Watershed(fdr, outlet)
            watershed_name = folder.split('\\')[-1]
            basin_main = arcpy.sa.SetNull(~(arcpy.sa.IsNull(basin_trib_up)),basin)
            basin_main = arcpy.sa.SetNull(~(arcpy.sa.IsNull(basin_trib_dn)),basin_main)
            basin_main = arcpy.sa.SetNull(~(arcpy.sa.IsNull(basin_up)),basin_main)
            watershed = arcpy.RasterToPolygon_conversion(basin_main, os.path.join(folder, watershed_name + '.shp'), "NO_SIMPLIFY")
            watershed_list.append(str(watershed))
        if i == 1:
            folder = subbasin_path(path, i+1)
            x_coord = tributaries_outlet.x[i+1]
            y_coord = tributaries_outlet.y[i+1]
            outlet = createPoint(x_coord, y_coord, sptl_ref)
            basin_trib = arcpy.sa.Watershed(fdr, outlet)
            watershed_name = folder.split('\\')[-1]
            watershed = arcpy.RasterToPolygon_conversion(basin_trib, os.path.join(folder, watershed_name + '.shp'), "NO_SIMPLIFY")
            watershed_list.append(str(watershed))
            
            folder = subbasin_path(path, i+2)
            x_coord = main_river_outlet.x[i+2]
            y_coord = main_river_outlet.y[i+2]
            outlet = createPoint(x_coord, y_coord, sptl_ref)
            basin_up = arcpy.sa.Watershed(fdr, outlet)
            
            folder = subbasin_path(path, i)
            x_coord = main_river_outlet.x[i]
            y_coord = main_river_outlet.y[i]
            outlet = createPoint(x_coord, y_coord, sptl_ref)
            basin = arcpy.sa.Watershed(fdr, outlet)
            watershed_name = folder.split('\\')[-1]
            basin_main = arcpy.sa.SetNull(~(arcpy.sa.IsNull(basin_up)),basin)
            basin_main = arcpy.sa.SetNull(~(arcpy.sa.IsNull(basin_trib)),basin_main)
            watershed = arcpy.RasterToPolygon_conversion(basin_main, os.path.join(folder, watershed_name + '.shp'), "NO_SIMPLIFY")
            watershed_list.append(str(watershed))
    watershed_list.sort()
    return watershed_list
#------------------------------------------------------------------------------
def clip_network(gridLink, gridFAC, gridFL, gridFDR, boundary, basin_path):
    gridLink = arcpy.Clip_management(gridLink, "#", os.path.join(basin_path, 'coarseLink.tif'), boundary, "#", "ClippingGeometry", "#")
    gridFL = arcpy.Clip_management(gridFL, "#", os.path.join(basin_path, 'coarseFL.tif'), boundary, "#", "ClippingGeometry", "#")
    gridFAC = arcpy.Clip_management(gridFAC, "#", os.path.join(basin_path, 'coarseFAC.tif'), boundary, "#", "ClippingGeometry", "#")
    gridFDR = arcpy.Clip_management(gridFDR, "#", os.path.join(basin_path, 'coarseFDR.tif'), boundary, "#", "ClippingGeometry", "#")
    
    zonalFAC = arcpy.sa.Int(arcpy.sa.ZonalStatistics(gridLink, 'VALUE', gridFAC, 'MAXIMUM'))
    zonalDIS = arcpy.sa.Int(arcpy.sa.ZonalStatistics(gridLink, 'VALUE', gridFL, 'MINIMUM'))
                                    
    #Extract Network Grid (Needed for Coordinates)
    arcpy.sa.Sample([gridLink, gridFAC, gridFL], gridLink, os.path.join(basin_path, 'net.dbf'), 'NEAREST')   #Modified from cell_area to gridLink [2018/09/27]
    coordinates = Dbf5(os.path.join(basin_path, 'net.dbf'))
    coordinates = coordinates.to_dataframe()
    coordinates.columns = ['cell_area', 'x', 'y', 'link', 'fac', 'len']
    coordinates = coordinates[coordinates.link != -9999.0]
    coordinates.link = coordinates.link.astype(int)
    
    
    #Extract Network Vector (Needed to identify hierarchy)
    vecLink =  arcpy.sa.StreamToFeature(gridLink, gridFDR, os.path.join(basin_path, 'vectorLink.shp'), 'NO_SIMPLIFY')
    vecFAC =  arcpy.sa.StreamToFeature(zonalFAC, gridFDR, os.path.join(basin_path, 'vectorFAC.shp'), 'NO_SIMPLIFY')
    vecDIS =  arcpy.sa.StreamToFeature(zonalDIS, gridFDR, os.path.join(basin_path, 'vectorDIS.shp'), 'NO_SIMPLIFY')
    stream_links = FeatureToDataFrame(vecFAC, vecDIS, vecLink)
    
    return stream_links, coordinates, gridFDR
#------------------------------------------------------------------------------
def clip_geomorphology(gridBedslope, gridFDR, boundary, basin_shp):
    
    bedslope_extract = arcpy.sa.ExtractByMask(gridBedslope, basin_shp)
    basin_path = basin_shp.split('\\')[:-1]
    basin_path.append('bedslope.asc')
    arcpy.RasterToASCII_conversion(bedslope_extract, os.path.join(*basin_path))
    
    coarseFDR_extract = arcpy.sa.ExtractByMask(gridFDR, basin_shp)
    coarseFL_extract = arcpy.sa.FlowLength(coarseFDR_extract)
    basin_path = basin_shp.split('\\')[:-1]
    basin_path.append('distance.asc')
    arcpy.RasterToASCII_conversion(coarseFL_extract, os.path.join(*basin_path))
    
    cell_area_extract = arcpy.sa.ExtractByMask(boundary, basin_shp)
    basin_path = basin_shp.split('\\')[:-1]
    basin_path.append('watershed.asc')
    arcpy.RasterToASCII_conversion(cell_area_extract, os.path.join(*basin_path))
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

pfaf_level = 1
pfaf_level0_basins = [0]      #Which subbasins to sub-diivde upto level 1 (Always 0)
pfaf_level1_basins = []       #Which subbasins to sub-divide upto level 2 (e.g. [1,2,3...])
pfaf_level2_basins = []       #Which subbasins to sub-divide upto level 3 (e.g. [11,22,21...])
max_level = 10
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
cell_area = arcpy.Raster(os.path.join(project, scratch, 'cell_area.tif'))
coarseFDR = arcpy.Raster(os.path.join(project, scratch, 'coarseFDR.tif'))
coarseFAC = arcpy.Raster(os.path.join(project, scratch, 'coarseFAC.tif'))
coarseBedslope = arcpy.Raster(os.path.join(project, scratch, 'bedslope.tif'))
arcpy.env.extent = cell_area
coarse_resolution = arcpy.Describe(cell_area).meanCellHeight
spatial_reference = arcpy.Describe(cell_area).SpatialReference

#Creating folders
basin_paths = basin_directories(os.path.join(project, scratch), pfaf_level, pfaf_level0_basins, pfaf_level1_basins, pfaf_level2_basins)

#Processing
coarseFL = arcpy.sa.FlowLength(coarseFDR)
coarseFL.save(os.path.join(project, scratch, 'coarseFL.tif'))

coarseStream = arcpy.sa.Con(coarseFAC > 1, 1)
coarseStream.save(os.path.join(project, scratch, 'coarseStream.tif'))

coarseLink = arcpy.sa.StreamLink(coarseStream, coarseFDR)
coarseLink.save(os.path.join(project, scratch, 'coarseLink.tif'))

netFAC = arcpy.sa.Int(arcpy.sa.ZonalStatistics(coarseLink, 'VALUE', coarseFAC, 'MAXIMUM'))
netFAC.save(os.path.join(project, scratch, 'netFAC.tif'))

netDIS = arcpy.sa.Int(arcpy.sa.ZonalStatistics(coarseLink, 'VALUE', coarseFL, 'MINIMUM'))
netDIS.save(os.path.join(project, scratch, 'netDIS.tif'))

vectorLink = arcpy.sa.StreamToFeature(coarseLink, coarseFDR, os.path.join(project, scratch, 'vectorLink.shp'), 'NO_SIMPLIFY')
vectorFAC = arcpy.sa.StreamToFeature(netFAC, coarseFDR, os.path.join(project, scratch, 'vectorFAC.shp'), 'NO_SIMPLIFY')
vectorDIS = arcpy.sa.StreamToFeature(netDIS, coarseFDR, os.path.join(project, scratch, 'vectorDIS.shp'), 'NO_SIMPLIFY')

#Extract Network Grid (Needed for Coordinates)
arcpy.sa.Sample([coarseLink, coarseFAC, coarseFL], cell_area, os.path.join(project, scratch, 'net.dbf'), 'NEAREST')
sample = Dbf5(os.path.join(project, scratch, 'net.dbf'))
sample = sample.to_dataframe()
sample.columns = ['cell_area', 'x', 'y', 'link', 'fac', 'len']
sample = sample[sample.link != -9999.0 ]
sample.link = sample.link.astype(int)

#Extract Network Vector (Needed to identify hierarchy)
stream_df = FeatureToDataFrame(vectorFAC, vectorDIS, vectorLink)


#Pfafstetter Level 1
if pfaf_level in range(1,max_level):
    pfaf_range = range(pfaf_level)
    for pf0 in pfaf_level0_basins:
        i = int(str(pf0)[0])
        root_path0 = basin_paths[pfaf_range[0]][i-1].split('\\')[:-1]
        root_basin0 = 'basin' + str(pf0)
        root_path0.append(root_basin0)
        root_path0 = '\\'.join(root_path0)
        
        print '\n Sub-dividing basin: ' + str(pf0), root_path0
        main_outlet, trib_outlet = pfaf(stream_df, sample)
        subbasins1 = pfaf_delineate(main_outlet, trib_outlet, coarseFDR, spatial_reference, root_path0)
        for subbasin1 in subbasins1:
            print 'Generated ' + subbasin1.split('\\')[-1].split('.')[0]
            print 'Extracting basin properties... \n'
            clip_geomorphology(coarseBedslope, coarseFDR, cell_area, subbasin1)
            

#Pfafstetter Level 2
if pfaf_level in range(2,max_level):
    pfaf_range = range(pfaf_level)
    for pf1 in pfaf_level1_basins:
        i = int(str(pf1)[0])
        root_path1 = basin_paths[pfaf_range[1]][i-1].split('\\')[:-1]
        root_basin1 = 'basin' + str(pf1)
        root_path1.append(root_basin1)
        root_path1 = '\\'.join(root_path1)
        shp_path1 = os.path.join(root_path1, root_path1.split('\\')[-1] + '.shp')
        
        if os.path.exists(shp_path1):
            print '\n Sub-dividing basin: ' + str(pf1), root_path1
            
            stream_df, sample, clip_FDR = clip_network(coarseLink, coarseFAC, coarseFL, coarseFDR, shp_path1, root_path1)
            main_outlet, trib_outlet = pfaf(stream_df, sample)
            subbasins2 = pfaf_delineate(main_outlet, trib_outlet, clip_FDR, spatial_reference, root_path1)
            for subbasin2 in subbasins2:
                print 'Generated ' + subbasin2.split('\\')[-1].split('.')[0]
                print 'Extracting basin properties... \n'
                clip_geomorphology(coarseBedslope, coarseFDR, cell_area, subbasin2)

#Pfafstetter Level 3
if pfaf_level in range(3,max_level):
    pfaf_range = range(pfaf_level)
    for pf2 in pfaf_level2_basins:
        i, j = int(str(pf2)[0]), int(str(pf2)[1])
        root_path2 = basin_paths[pfaf_range[2]][i-1][j-1].split('\\')[:-1]
        root_basin2 = 'basin' + str(pf2)
        root_path2.append(root_basin2)
        root_path2 = '\\'.join(root_path2)
        shp_path2 = os.path.join(root_path2, root_path2.split('\\')[-1] + '.shp')
        
        if os.path.exists(shp_path1):
            print '\n Sub-dividing basin: ' + str(pf2), root_path2
            
            stream_df, sample, clip_FDR = clip_network(coarseLink, coarseFAC, coarseFL, coarseFDR, shp_path2, root_path2)
            main_outlet, trib_outlet = pfaf(stream_df, sample)
            subbasins3 = pfaf_delineate(main_outlet, trib_outlet, clip_FDR, spatial_reference, root_path2)
            for subbasin3 in subbasins3:
                print 'Generated ' + subbasin3.split('\\')[-1].split('.')[0]
                print 'Extracting basin properties... \n'
                clip_geomorphology(coarseBedslope, coarseFDR, cell_area, subbasin3)

#Making kfs.dat
print('Creating kfs.dat ...')
location = pd.DataFrame(np.array([1,2,3,4,5,6,7,8,9]), columns = ["'Location'"], index = range(1,10))

lvl1 = np.zeros((9,), dtype = int)
if pfaf_level >=2:
    for basin1 in pfaf_level1_basins:
        i = basin1
        lvl1[i-1] = 1
lvl1 = pd.DataFrame(lvl1, columns = ["'Basin:'"], index = range(1,10))

lvl2 = np.zeros((9,9), dtype = int)
if pfaf_level >= 3:
    for basin2 in pfaf_level2_basins:
        i = int(str(basin2)[0])
        j = int(str(basin2)[1])
        lvl2[j-1, i-1] = 1
lvl2 = pd.DataFrame(lvl2, columns = ["'Basin1:'", "'Basin2:'", "'Basin3:'", "'Basin4:'", "'Basin5:'", "'Basin6:'", "'Basin7:'", "'Basin8:'", "'Basin9:'"], index = range(1,10))


with open(os.path.join(project, output, parameter, 'kfs.dat'), 'w') as f:
    f.write("'Maximum_level:'" + str(pfaf_level) + "\n")
    location.transpose().to_csv(f, header=False, sep = '\t')
    f.write("'Level_1'\n")
    lvl1.transpose().to_csv(f, header=False, sep = '\t')
    f.write("'Level_2'\n")
    lvl2.transpose().to_csv(f, header=False, sep = '\t')
    f.write("'END'")


#Making subcatchment.dat
print('Creating sub-catchment.dat ...')
sub_catch = []
a=0
for i in basin_paths[1]:
    a+=1
    if pd.notnull(i):
        b=0
        for j in basin_paths[2][a-1]:
            b+=1
            if pd.notnull(j):
                c=0
                for k in basin_paths[3][a-1][b-1]:
                    c+=1
                    if pd.notnull(k):
                        sub_catch.append('ws' + '{:<03d}'.format(int(str(a) + str(b) + str(c))))
                    else:
                        sub_catch.append('ws' + '{:<03d}'.format(int(str(a) + str(b))))
            else:
                sub_catch.append('ws' + '{:<03d}'.format(a))
sub_catch = list(OrderedDict.fromkeys(sub_catch))
sub_catch = list(reversed(sub_catch))
data = {'Code':range(1, len(sub_catch)+1), 'sub_basin':sub_catch}
subcatchment = pd.DataFrame(data=data, columns = ['Code', 'sub_basin', 'width_min', 'width_max', 'height_min', 'height_max', 'roughness_min', 'roughness_max'])
subcatchment.to_csv(os.path.join(project, output, parameter, 'subcatchment.dat'), sep = '\t', index = False)

#Creating morph files
print('Creating morph files...')
for ws in sub_catch:
    print(ws)
    ws_strp = ws[2:].rstrip('0')
    if len(ws_strp) == 1:
        i = int(ws_strp[0])
        ws_path = basin_paths[1][i-1]
    elif len(ws_strp) == 2:
        i = int(ws_strp[0])
        j = int(ws_strp[1])
        ws_path = basin_paths[2][i-1][j-1]
    elif len(ws_strp) == 3:
        i = int(ws_strp[0])
        j = int(ws_strp[1])
        k = int(ws_strp[2])
        ws_path = basin_paths[3][i-1][j-1][k-1]
    
    sub_watershed = read_ascii_grid(os.path.join(ws_path, 'watershed.asc'))
    ncols = sub_watershed[1]
    nrows = sub_watershed[2]
    sub_distance = read_ascii_grid(os.path.join(ws_path, 'distance.asc'))[0]
    sub_bedslope = read_ascii_grid(os.path.join(ws_path, 'bedslope.asc'))[0]
    ws_df = pd.DataFrame(columns = ['distance', 'row', 'col', 'bedslope'])
    
    for i in range(nrows):
        for j in range(ncols):
            if sub_watershed[0][i,j] != -9999:
                ws_df1 = pd.DataFrame({'distance':[sub_distance[i,j]],
                                       'row':[i+1],
                                       'col':[j+1],
                                       'bedslope':[sub_bedslope[i,j]]})
                ws_df = pd.concat([ws_df, ws_df1], ignore_index = True,sort=False)
    ws_df = ws_df[['distance', 'row', 'col', 'bedslope']]
    ws_df.to_csv(os.path.join(project, output, parameter, ws + '_morph'), header = False, index = False, sep = '\t')
    ws_df.to_pickle(os.path.join(project, scratch, ws + '_morph.pkl'))
#==============================================================================
print('Finished Successfully!')