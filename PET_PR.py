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







