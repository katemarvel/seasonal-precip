import glob
import sys
import cdms2 as cdms
import numpy as np
import MV2 as MV
import difflib
import scipy.stats as stats
global crunchy
import socket
import pickle
if socket.gethostname().find("crunchy")>=0:
    crunchy = True
else:
    crunchy = False

#import peakfinder as pf
import cdtime,cdutil,genutil
from eofs.cdms import Eof
from eofs.multivariate.cdms import MultivariateEof
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from Plotting import *
import CMIP5_tools as cmip5


### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)

def penmont_vpd_SH(Ta,Rnet,q,press,u):
    """
    
    Using the modified FAO Penman-Monteith approach, calculation reference
    (potential) evapotranspiration. As part of this, also calculate the vapor
    pressure deficit (VPD) and relative humidity RH.
    
    Inputs:
    Ta      = temperature, degrees K
    Rnet    = surface net radiation, W/m2
    u       = wind speed at 2 meters, m/s
    q       = specific humidity, kg/kg
    press   = surface pressure, Pascals

    Outputs:
    PET  = potential evapotranspiration (mm/day)       
    VPD  = vapor presssure deficit (kPa)  
    RH   = relative humidity, fraction

    Written by Benjamin I. Cook and translated to Python by Kate Marvel
 
    Based on:
    Xu, C-Y., and V. P. Singh. "Cross comparison of empirical equations 
               for calculating potential evapotranspiration with data 
               from Switzerland." Water Resources Management,
               16.3 (2002): 197-219.

           FAO Document:
           http://www.fao.org/docrep/X0490E/x0490e00.htmContents

    For Tetens, both above and below zero:
       http://cires.colorado.edu/~voemel/vp.html
    """

    #Convert temperature to degrees C
    Ta = Ta -273.15
    
    # ground heat flux, set to zero (works fine on >monthly timescales, and is 
    # accurate if one calculates Rnet as SH+LH)
    gflux=0.

    #Extrapolate wind speed from 10m to 2m (ADDED BY KATE)
    u2 = u*(4.87/np.log(67.8*10.-5.42))


    # Calculate the latent heat of vaporization (MJ kg-1)
    lambda_lv=2.501-(2.361e-3)*Ta

    # Calculate Saturation vapor pressure (kPa)
    es=0.611*np.exp((17.27*Ta)/(Ta+237.3))  

    # Convert specific humidity (kg/kg) to actual vapor pressure
    ea=(press*q)/0.6213 # Pascals

    # Convert ea to kilopascals
    ea = ea/1000.

    # Convert Pressure to kPa
    press=press/1000.

    # Use es and relative humidity to calculate ea in kPa
    #ea=es.*RH

    # Slope of the vapor pressure curve (kPa C-1)
    delta_vpc=(4098*es)/((Ta+237.3)**2)

    # Psychometric constant (kPa C-1)
    # 0.00163 is Cp (specific heat of moist air) divided by epsilon (0.622, 
    # ratio molecular weight of water vapour/dry air)
    psych_const=0.00163*(press/lambda_lv)

    # Net radiation
    Rnet=Rnet/11.6  # convert W/m2 to MJ/m2/d

    # Calculate VPD
    VPD=es-ea

    # Calculate Relative Humidity
    RH = ea/es

    # Potential Evapotranspiration (mm/day)
    PET=(0.408*delta_vpc*(Rnet-gflux)+psych_const*(900./(Ta+273))*u2*(VPD))/(delta_vpc+psych_const*(1+0.34*u))

    return PET, VPD, RH

def PET_from_cmip(fname):
    """
    fname must be a temperature file
    
    Calculate PET from cmip5 data on crunchy
    #Ta      = temperature, degrees K = tas
    #   Rnet    = surface net radiation, W/m2 = (hfls+hfss)
    #  u       = wind speed at 2 meters, m/s = sfcWind (??)
    #   q       = specific humidity, kg/kg = huss

    #  press   = surface pressure, Pascals = ps
     """
    #Get land and ice masks
if 1:
    temp_variable = fname.split(".")[-4]
    
    f_ta =cdms.open(fname)
    Ta = f_ta(temp_variable)
    f_ta.close()
    
    fland = cdms.open(cmip5.landfrac(fname))
    land = fland("sftlf")
    fland.close()
    try:
        fglac = cdms.open(cmip5.glacierfrac(fname))
    
        glacier=fglac("sftgif")
        fglac.close()
    except:
        glacier = MV.zeros(land.shape)
    #mask ocean and ice sheets
    totmask = np.logical_or(land==0,glacier==100.)
    nt = len(Ta.getTime())

    totmask =np.repeat(totmask.asma()[np.newaxis],nt,axis=0)

    if cmip5.get_corresponding_file(fname,"sfcWind") is None:
        #if no surface wind is available, use NCEP reanalyis
        f_wind = cdms.open("wspd.mon.ltm.nc")
        wspd = f_wind("wspd")
        ny=nt/12
        u=np.repeat(wspd,ny,axis=0) #Repeat monthly climatologies
        u.setAxisList([Ta.getTime()]+wspd.getAxisList()[1:])
        u.id="sfcWind"
        f_wind.close()
    else:
        f_wind = cdms.open(cmip5.get_corresponding_file(fname,"sfcWind"))
        u = f_wind("sfcWind")
        
        f_wind.close()

    
    
    f_hfls =cdms.open(cmip5.get_corresponding_file(fname,"hfls"))
    hfls = f_hfls("hfls")
    f_hfls.close()
    # Some models put sfcWind on a different grid to calculate surface fluxes
    if u.shape != hfls.shape:
        u = u.regrid(hfls.getGrid(),regridTool='regrid2')
    
    f_hfss =cdms.open(cmip5.get_corresponding_file(fname,"hfss"))
    hfss = f_hfss("hfss")
    f_hfss.close()

    Rnet = hfss + hfls
    
        

    

    f_huss =cdms.open(cmip5.get_corresponding_file(fname,"huss"))
    q = f_huss("huss")
    f_huss.close()

    f_ps =cdms.open(cmip5.get_corresponding_file(fname,"ps"))
    press = f_ps("ps")
    f_ps.close()

    corresponding_files = [cmip5.get_corresponding_file(fname,var) for var in ["hfss",temp_variable,"huss","ps","hfls"]]
    L=cmip5.get_common_timeax(corresponding_files)
    if type(L)==type(()):
        PET,VPD,RH = penmont_vpd_SH(Ta(time=L),Rnet(time=L),q(time=L),press(time=L),u(time=L))
    else:
    
        PET,VPD,RH = penmont_vpd_SH(Ta[:L],Rnet[:L],q[:L],press[:L],u[:L])

    PET = MV.masked_where(totmask,PET)
    PET.setAxisList(Ta.getAxisList())
    PET.id = "PET"

    VPD = MV.masked_where(totmask,VPD)
    VPD.setAxisList(Ta.getAxisList())
    VPD.id = "VPD"

    RH = MV.masked_where(totmask,RH)
    RH.setAxisList(Ta.getAxisList())
    RH.id = "RH"

    
    
    return PET,VPD,RH

        
             
def PET_experiment(experiment):

    
    fnames_tasmax = np.array(cmip5.get_datafiles(experiment,"tasmax"))
    fnames_tasmin = np.array(cmip5.get_datafiles(experiment,"tasmin"))
    path="/kate/PET/"+experiment+"/"
    os.system("mkdir "+path)
# TASMAX    
    for fname in fnames_tasmax:
        try:
            PET,VPD,RH =  PET_from_cmip(fname)
            writefname = fname.split("/")[-1].replace("xml","nc").replace("tasmax","PET_tasmax")
            fw = cdms.open(path+writefname,"w")
            fw.write(PET)
            fw.write(VPD)
            fw.write(RH)
            fw.close()

           
            
        except:
            print "bad file: "+fname

                
    for fname in fnames_tasmin:
        try:
            PET,VPD,RH =  PET_from_cmip(fname)
            writefname = fname.split("/")[-1].replace("xml","nc").replace("tasmin","PET_tasmin")
            fw = cdms.open(path+writefname,"w")
            fw.write(PET)
            fw.write(VPD)
            fw.write(RH)
            fw.close()

           
            
        except:
            print "bad file: "+fname


if __name__ == "__main__":
    for experiment in ["piControl","1pctCO2","historical","rcp85"]:
        PET_experiment(experiment)








