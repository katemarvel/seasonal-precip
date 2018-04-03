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
import seasonal_cycle_utils as sc

### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)


def observations():

    # Read in PET and PR files
    #regridded PET
    f = cdms.open("OBS/cru_ts4.01.1901.2016.pet.dat.REGRID.nc")
    pet = f("pet")
    f.close()
    landmask = pet[0].mask
    # Regridded GPCC
    fpr = cdms.open("OBS/precip.mon.total.2.5x2.5.v7.nc")
    pr = fpr("precip")
    fpr.close()
    pr = pr/30. #Approximate mm-> mm/day by assuming 30 days/month
    #Put them both on the same time axis
    startpet = cmip5.start_time(pet)
    stoppet = cmip5.stop_time(pet)
    startpr = cmip5.start_time(pr)
    stoppr = cmip5.stop_time(pr)

    if cdtime.compare(startpet,startpr) >0:
        start = startpet
    else:
        start = startpr
    if cdtime.compare(stoppet,stoppr) >0:
        stop = stoppr
    else:
        stop = stoppet
    start = cdtime.comptime(start.year,start.month,1)
    stop = cdtime.comptime(stop.year,stop.month,31)

    pr = pr(time=(start,stop))
    pet = pet(time=(start,stop))

    #Calculate R and P
    Rpr,Ppr = sc.fast_annual_cycle(pr)
    Rpet,Ppet = sc.fast_annual_cycle(pet)
    #Convert phase to month of maximum (VECTORIZE THIS?)
    #How to handle "month of maximum" if it's fluctuating between 1 and 12?  (physically, what does this mean when our timesteps are every year?)
    #Should we modify the code in phase detection to start at phase 0?  Ie start in a month such that the maximum is 6 months away?

    #Calculate variance maps for p and pet
    test_period  = ('1979-1-1','2004-12-31') #for overlap with CMIP5 historical
    pet_vmap = sc.variance_map(pet(time=test_period))
    pr_vmap = sc.variance_map(pr(time=test_period))

    #make phase maps
    variance_threshold = 0.25 #Can we come up with a physically meaningful threshold here? Null hypothesis of no correlation ruled out at 99% confidence? 
    Ppet_clim = sc.phase_climatology(Ppet)
    Ppet_clim_month = sc.mask_data(sc.phase_to_month(Ppet_clim),landmask)

    Ppr_clim = sc.phase_climatology(Ppr)
    Ppr_clim_month = sc.mask_data(sc.phase_to_month(Ppr_clim),landmask)
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    from land_seasonal_cycle import landplot
    plt.subplot(211)
    m = landplot(MV.masked_where(pet_vmap<variance_threshold,Ppet_clim_month),cmap=cm.hsv,vmin=0,vmax=12)
    m.drawcoastlines(color='gray')
    cbar = plt.colorbar(orientation="horizontal")
    cbar.set_ticks(np.arange(12))
    cbar.set_ticklabels(months)
    plt.title("PET phase")

    plt.subplot(212)
    m = landplot(MV.masked_where(pr_vmap<variance_threshold,Ppr_clim_month),cmap=cm.hsv,vmin=0,vmax=12)
    m.drawcoastlines(color='gray')
    cbar = plt.colorbar(orientation="horizontal")
    cbar.set_ticks(np.arange(12))
    cbar.set_ticklabels(months)
    plt.title("PR phase")

    #phase trends
    Pa_pet = sc.get_phase_anomalies(Ppet)
    Pa_pet_trends = cmip5.get_linear_trends(Pa_pet)

    Pa_pr = sc.get_phase_anomalies(Ppr)
    Pa_pr_trends = cmip5.get_linear_trends(Pa_pr)



def temperature_phase():
    fpr = cdms.open("OBS/precip.mon.total.2.5x2.5.v7.nc")
    pr = fpr["precip"]
    the_grid = pr.getGrid()
    ft = cdms.open("OBS/GHCN_CAMS_NOAA.mon.mean.nc")
    temp=ft("air").regrid(the_grid,regridTool='regrid2')
    ft.close()
    fa = cdms.open("OBS/GHCN_CAMS_NOAA.mon.1981-2010.ltm.nc")
    climatology=fa("air").regrid(the_grid,regridTool='regrid2')
    fa.close()
    fpr.close()

    start = cmip5.start_time(temp)
    if start.month is not 1:
        start = cdtime.comptime(start.year+1,1,1)
    stop = cmip5.stop_time(temp)
    if stop.month is not 12:
        stop=cdtime.comptime(stop.year-1,12,31)
    temp=temp(time=(start,stop))
    temp_year = temp.shape[0]/12
    temp_ac = cmip5.cdms_clone(temp+np.repeat(climatology.asma(),temp_year,axis=0),temp)
    return temp_ac


def phase_1pct(pet):
    nmod,nt,nlat,nlon = pet.shape
    Pmma = MV.zeros((nmod,nt/12,nlat,nlon))
    for i in range(nmod):
        R,P = sc.fast_annual_cycle(pet[i])
        yax = P.getTime()
        Pa = sc.get_phase_anomalies(P,historical=False)
        Pmma[i]=Pa
    Pmma.setAxis(0,pet.getAxis(0))
    Pmma.setAxis(1,yax)
    Pmma.setAxis(2,pet.getAxis(2))
    Pmma.setAxis(3,pet.getAxis(3))
    Pmma.id = pet.id+"_phase"
    return Pmma
    
import string
def phase_data_store():
    for variable in ["PET","pr","evspsbl"]:
        fname = "/kate/TEST_DATA/"+string.upper(variable)+"_ensemble.nc"
        f=cdms.open(fname)
        writename="/kate/TEST_DATA/"+string.upper(variable)+"_phase.nc"
        fw = cdms.open(writename,"w")
        X = f(variable)
        P = phase_1pct(X)
        fw.write(P)
        fw.close()

def only_ESMS(fnames):
    #ESMs from Swann et al Table S4 http://www.pnas.org/content/pnas/suppl/2016/08/25/1604581113.DCSupplemental/pnas.201604581SI.pdf
    ESMs = ["bcc-csm1-1", "CanESM2", "CESM1-BGC", "GFDL-ESM2M","HadGEM2-ES", "IPSL-CM5A-LR","NorESM1-ME"]
    only_ESMs=[]
    for esm in ESMs:
        I=np.where(np.array([x.find(esm+".")>=0 for x in fnames]))[0]
        only_ESMs += np.array(fnames)[I].tolist()
    return np.array(only_ESMs)

def PETFUNC(X):
    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs("precip").getGrid()
    Xt = X[:140*12]
    Xr = Xt.regrid(the_grid,regridTool='regrid2')
    fobs.close()
    return Xr

def get_P_and_E(experiment="1pctCO2"):
    pr_fnames_all = np.array(cmip5.get_datafiles(experiment,"pr"))
    pr_esm = only_ESMS(pr_fnames_all)
    if experiment == "1pctCO2": #GFDL p1 increases CO2 only to doubling so get rid of it
        i=np.where(np.array([x.find(".GFDL-ESM2M.1pctCO2.r1i1p1.")>=0 for x in pr_esm]))[0]
        pr_esm=np.delete(pr_esm,i)
    nmods = len(pr_esm)

    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs["precip"].getGrid()
    nlat,nlon=the_grid.shape
    fobs.close()
    PR = MV.zeros((nmods,140*12,nlat,nlon))

    for i in range(nmods):
        f=cdms.open(pr_esm[i])
        X = f("pr")
        Xregrid=PETFUNC(X)
        PR[i]=Xregrid
    axes = [cmip5.make_model_axis(pr_esm)]+Xregrid.getAxisList()
    PR.setAxisList(axes)
    PR.id="pr"
    

    evspsbl_fnames_all = np.array(cmip5.get_datafiles(experiment,"evspsbl"))
    evspsbl_esm = only_ESMS(evspsbl_fnames_all)
    if experiment == "1pctCO2": #GFDL p1 increases CO2 only to doubling so get rid of it
        i=np.where(np.array([x.find(".GFDL-ESM2M.1pctCO2.r1i1p1.")>=0 for x in evspsbl_esm]))[0]
        evspsbl_esm=np.delete(evspsbl_esm,i)
    nmods = len(evspsbl_esm)

    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs["precip"].getGrid()
    nlat,nlon=the_grid.shape
    fobs.close()
    EVSPSBL = MV.zeros((nmods,140*12,nlat,nlon))

    for i in range(nmods):
        f=cdms.open(evspsbl_esm[i])
        X = f("evspsbl")
        EVSPSBL[i]=PETFUNC(X)
    axes = [cmip5.make_model_axis(evspsbl_esm)]+Xregrid.getAxisList()
    EVSPSBL.setAxisList(axes)
    EVSPSBL.id="evspsbl"

    fw = cdms.open("/kate/TEST_DATA/ESM_PR_EVSPSBL.nc","w")
    fw.write(PR)
    fw.write(EVSPSBL)
    fw.close()
    
def get_LAI_and_GPP(experiment="1pctCO2"):
   
    

    lai_fnames_all = np.array(cmip5.get_datafiles(experiment,"lai",realm="land"))
    lai_esm = only_ESMS(lai_fnames_all)
    if experiment == "1pctCO2": #GFDL p1 increases CO2 only to doubling so get rid of it
        i=np.where(np.array([x.find(".GFDL-ESM2M.1pctCO2.r1i1p1.")>=0 for x in lai_esm]))[0]
        lai_esm=np.delete(lai_esm,i)
    nmods = len(lai_esm)

    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs["precip"].getGrid()
    nlat,nlon=the_grid.shape
    fobs.close()
    LAI = MV.zeros((nmods,140*12,nlat,nlon))

    for i in range(nmods):
        f=cdms.open(lai_esm[i])
        X = f("lai")
        LAI[i]=PETFUNC(X)
    axes = [cmip5.make_model_axis(lai_esm)]+Xregrid.getAxisList()
    LAI.setAxisList(axes)
    LAI.id="lai"

        

    gpp_fnames_all = np.array(cmip5.get_datafiles(experiment,"gpp",realm="land"))
    gpp_esm = only_ESMS(gpp_fnames_all)
    if experiment == "1pctCO2": #GFDL p1 increases CO2 only to doubling so get rid of it
        i=np.where(np.array([x.find(".GFDL-ESM2M.1pctCO2.r1i1p1.")>=0 for x in gpp_esm]))[0]
        gpp_esm=np.delete(gpp_esm,i)
    nmods = len(gpp_esm)

    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs["precip"].getGrid()
    nlat,nlon=the_grid.shape
    fobs.close()
    GPP = MV.zeros((nmods,140*12,nlat,nlon))

    for i in range(nmods):
        f=cdms.open(gpp_esm[i])
        X = f("gpp")
        GPP[i]=PETFUNC(X)
    axes = [cmip5.make_model_axis(gpp_esm)]+Xregrid.getAxisList()
    GPP.setAxisList(axes)
    GPP.id="gpp"
    
    fw = cdms.open("/kate/TEST_DATA/ESM_LAI_GPP.nc","w")
    fw.write(LAI)
    fw.write(GPP)
    fw.close()
    return LAI,GPP

def get_evap_variables(experiment="1pctCO2"):

    evspsblveg_fnames_all = np.array(cmip5.get_datafiles(experiment,"evspsblveg",realm="land"))
    evspsblveg_esm = only_ESMS(evspsblveg_fnames_all)
    if experiment == "1pctCO2": #GFDL p1 increases CO2 only to doubling so get rid of it
        i=np.where(np.array([x.find(".GFDL-ESM2M.1pctCO2.r1i1p1.")>=0 for x in evspsblveg_esm]))[0]
        evspsblveg_esm=np.delete(evspsblveg_esm,i)
    nmods = len(evspsblveg_esm)

    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs["precip"].getGrid()
    nlat,nlon=the_grid.shape
    fobs.close()
    EVSPSBLVEG = MV.zeros((nmods,140*12,nlat,nlon))

    for i in range(nmods):
        f=cdms.open(evspsblveg_esm[i])
        X = f("evspsblveg")
        Xregrid=PETFUNC(X)
        EVSPSBLVEG[i]=Xregrid
    axes = [cmip5.make_model_axis(evspsblveg_esm.tolist())]+Xregrid.getAxisList()
    EVSPSBLVEG.setAxisList(axes)
    EVSPSBLVEG.id="evspsblveg"
    

    evspsblsoi_fnames_all = np.array(cmip5.get_datafiles(experiment,"evspsblsoi",realm="land"))
    evspsblsoi_esm = only_ESMS(evspsblsoi_fnames_all)
    if experiment == "1pctCO2": #GFDL p1 increases CO2 only to doubling so get rid of it
        i=np.where(np.array([x.find(".GFDL-ESM2M.1pctCO2.r1i1p1.")>=0 for x in evspsblsoi_esm]))[0]
        evspsblsoi_esm=np.delete(evspsblsoi_esm,i)
    nmods = len(evspsblsoi_esm)

    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs["precip"].getGrid()
    nlat,nlon=the_grid.shape
    fobs.close()
    EVSPSBLSOI = MV.zeros((nmods,140*12,nlat,nlon))

    for i in range(nmods):
        f=cdms.open(evspsblsoi_esm[i])
        X = f("evspsblsoi")
        EVSPSBLSOI[i]=PETFUNC(X)
    axes = [cmip5.make_model_axis(evspsblsoi_esm.tolist())]+Xregrid.getAxisList()
    EVSPSBLSOI.setAxisList(axes)
    EVSPSBLSOI.id="evspsblsoi"

    
    tran_fnames_all = np.array(cmip5.get_datafiles(experiment,"tran",realm="land"))
    tran_esm = only_ESMS(tran_fnames_all)
    if experiment == "1pctCO2": #GFDL p1 increases CO2 only to doubling so get rid of it
        i=np.where(np.array([x.find(".GFDL-ESM2M.1pctCO2.r1i1p1.")>=0 for x in tran_esm]))[0]
        tran_esm=np.delete(tran_esm,i)
    nmods = len(tran_esm)

    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs["precip"].getGrid()
    nlat,nlon=the_grid.shape
    fobs.close()
    TRAN = MV.zeros((nmods,140*12,nlat,nlon))

    for i in range(nmods):
        f=cdms.open(tran_esm[i])
        X = f("tran")
        TRAN[i]=PETFUNC(X)
    axes = [cmip5.make_model_axis(tran_esm)]+Xregrid.getAxisList()
    TRAN.setAxisList(axes)
    TRAN.id="tran"

    fw = cdms.open("/kate/TEST_DATA/land_evap.nc","w")
    fw.write(EVSPSBLVEG)
    fw.write(EVSPSBLSOI)
    fw.write(TRAN)
    fw.close()
    
            
    

