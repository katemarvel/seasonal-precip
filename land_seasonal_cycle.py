import glob
import sys
import cdms2 as cdms
import numpy as np
import MV2 as MV
import difflib
import scipy.stats as stats
global crunchy
import socket
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
### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)

from scipy import signal
import seasonal_cycle_utils as sc
from Plotting import *
import CMIP5_tools as cmip5



def gpcp_land_mask():
    """Create a land mask from GPCC number of observations """
    f = cdms.open("OBS/precip.mon.nobs.2.5x2.5.v7.nc")
    test = f("precip")
    nobs = MV.sum(test,axis=0) # Total # of observations
    gpcp_land_mask = nobs[::-1] == 0 # Have to flip lat axis to be on same grid
    return gpcp_land_mask


def landplot(data,vmin=None,vmax=None,cmap=cm.RdBu_r):
    """ Plot data on cyl projection with lon_0 = prime meridian"""
    if vmin is None:
        a = np.max(np.abs(data))
        vmin=-a
        vmax=a
    m = bmap(data,projection="cyl",lon_0=0,vmin=vmin,vmax=vmax,cmap=cmap)
    return m


def regrid_pr_historical(X):
    """ regrid CMIP5 pr to obs grid """
    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs("precip").getGrid()
    fobs.close()
    start = '1900-1-1'
    stop = '2005-12-31'
    return X(time=(start,stop)).regrid(the_grid,regridTool='regrid2')



def regrid_pr_rcp85(X):
    """ regrid CMIP5 pr to obs grid """
    fobs = cdms.open("/work/marvel1/SEASONAL/OBS/GPCP.precip.mon.mean.nc")
    the_grid = fobs("precip").getGrid()
    fobs.close()
    start = '2006-1-1'
    stop = '2099-12-31'
    return X(time=(start,stop)).regrid(the_grid,regridTool='regrid2')

if __name__ == "__main__":
    histdirec = "/work/cmip5/historical/atm/mo/pr/"
    hist = cmip5.get_ensemble(histdirec,"pr",func=regrid_pr_historical)
    hist.id="pr"
    f = cdms.open("/kate/PR_ENSEMBLES/cmip5.MMA.historical.pr.nc","w")
    f.write(hist)
    f.close()

    rcp85direc = "/work/cmip5/rcp85/atm/mo/pr/"
    rcp85 = cmip5.get_ensemble(rcp85direc,"pr",func=regrid_pr_rcp85)
    rcp85.id="pr"
    f = cdms.open("/kate/PR_ENSEMBLES/cmip5.MMA.rcp85.pr.nc","w")
    f.write(rcp85)
    f.close()

    




