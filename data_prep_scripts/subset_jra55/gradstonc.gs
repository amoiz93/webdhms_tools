function main(args)
'reinit'
fname = subwrd(args,1)
folder = subwrd(args,2)

folder_str = strlen(folder)
year_month = substr(fname, folder_str + 1, 6)
say year_month
say 'opening ' fname
'open 'fname
'q file'
t = sublin(result,5)
t = subwrd(t,12)
'set t 1 't
'set lon 136.5 138.5'
'set lat 35.5 37.5'

'q file'
v = sublin(result,6)
v = subwrd(v, 5)

vi = 1

while (vi <= v)
 'q file'
 var = sublin(result, 6 + vi)
 var = subwrd(var, 1)
 say var
 'define 'var'= 'var

 'set sdfwrite -nc4 'folder'/'year_month'/'var'_temp.nc'
 'sdfwrite 'var
 vi = vi + 1
endwhile	
