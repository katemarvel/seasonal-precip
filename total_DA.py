#regrid all observations to common grid and combine
#Write code to plot maps of specific regions
import CMIP5_tools as cmip5
import MV2 as MV
import glob
import cdms2 as cdms
import numpy as np
import cdutil
from seasonal_cycle_utils import mask_data
from eofs.cdms import Eof
import DA_tools as da
from seasonal_cycle_utils import mask_data
def combine_observations(list_of_obs,grid='2.5',writename=None):
    """ Place observations on common grid and combine """
    #Common time axis
    starts = [cmip5.start_time(x) for x in list_of_obs]
    start = np.max(starts)
    stops = [cmip5.stop_time(x) for x in list_of_obs]
    stop = np.min(stops)
    newlist = [x(time=(start,stop)) for x in list_of_obs]
    L = newlist[0].shape[0]

     #get target grid
    if grid == '2.5':
        fgrid = cdms.open("OBS/gpcp.precip.mon.mean.nc")
        print "writing to GPCP grid"
    else:
        fgrid = cdms.open("OBS/gpcc.precip.mon.total.v7.nc")
        print "writing to gpcc grid"
    
    gpcp_grid = fgrid("precip").getGrid()
    #set up the combined observations
    combined_obs = MV.zeros((L,)+gpcp_grid.shape)+1.e20

    for X in newlist:
        Xr = X.regrid(gpcp_grid,regridTool='regrid2')
        indices=np.where(~Xr.flatten().mask)
        MV.put(combined_obs,indices,Xr.compressed())
    combined_obs = MV.masked_where(combined_obs>1.e10,combined_obs)
    combined_obs.setAxisList([newlist[0].getTime()]+gpcp_grid.getAxisList())
    combined_obs.id = "pdsi"
    
    if writename is not None:
        fw = cdms.open(writename,"w")
        fw.write(combined_obs)
        fw.close()
    fgrid.close()
    return combined_obs

def mask_models(CO,writename=None):
    """Create mask based on availability of observations at first time step and apply to CMIP5 hist+85 PDSI."""
    f = cdms.open("../DROUGHT_ATLAS/CMIP5/pdsi.ensemble.hist.rcp85.nc")
    models = f("pdsi")
    models = MV.masked_where(np.isnan(models),models)
    masked_models = mask_data(models,CO[0].mask)
    masked_models.id='pdsi'
    if writename is not None:
        fw = cdms.open(writename,"w")
        fw.write(masked_models)
        fw.close()
    return masked_models
    
def rewrite_downloaded():
    ##NEED TO FIX TIME AXIS for all obs
    ##Transpose so axes are [time,lat,lon] and set time axis units to be years since summer 0 AD
    files = glob.glob("../DROUGHT_ATLAS/DOWNLOADED/*")
    for fil in files:
        write_transposed_and_masked(fil)
def write_transposed_and_masked(fil):
  
    atlas_name = fil.split("/")[-1].split(".")[0]
    print atlas_name
    f = cdms.open(fil)
    try:
        pdsi = f("pdsi")
        tpdsi = MV.transpose(pdsi)
    except:
        pdsi = f("PDSI")
        tpdsi = pdsi #For NADA the axes are OK
    mask_tpdsi=MV.masked_where(np.isnan(tpdsi),tpdsi)
    taxis = mask_tpdsi.getTime()
    taxis.units = "years since 0000-7-1"
    mask_tpdsi.setAxis(0,taxis)
    mask_tpdsi.id="pdsi"
    mask_tpdsi.name="pdsi"
    for key in f.attributes.keys():
        setattr(mask_tpdsi,key,f.attributes[key])
    writefname = "../DROUGHT_ATLAS/PROCESSED/"+atlas_name+".nc"
    wf = cdms.open(writefname,"w")
    wf.write(mask_tpdsi)
    wf.close()


def write_regridded():
    fnames = glob.glob("../DROUGHT_ATLAS/PROCESSED/*")
    for fil in fnames:
        atlas_name = fil.split("/")[-1].split(".")[0]
        if atlas_name != "ANZDA":
            f = cdms.open("../DROUGHT_ATLAS/PROCESSED/"+atlas_name+".nc")
            obs = f("pdsi")
            obs = MV.masked_where(np.isnan(obs),obs)
            obs = MV.masked_where(np.abs(obs)>90,obs)
            obs = obs(time=('1100-7-1','2020-12-31'))

            obs=mask_data(obs,obs.mask[0]) #Make all the obs have the same mask as the first datapoint
            f.close()
            writefname = "../DROUGHT_ATLAS/PROCESSED/"+atlas_name+"2.5.nc"
            CO = combine_observations([obs],grid='2.5',writename=writefname)
            modelwritename = "../DROUGHT_ATLAS/CMIP5/pdsi."+atlas_name+"2.5.hist.rcp85.nc"
            mm = mask_models(CO,writename=modelwritename)
            
def write_combinations():
    """
    Make two combined NH drought atlases: ALL_OBS, which starts in 1100 and includes MADA, OWDA, and NADA, and ALL_OBS_plus_MEX, which includes the former + MXDA
    """
    list_of_obs = []
    for atlas_name in ["OWDA2.5","MADA2.5","NADA2.5"]:
        f = cdms.open("../DROUGHT_ATLAS/PROCESSED/"+atlas_name+".nc")
        obs = f("pdsi")
        obs = MV.masked_where(np.isnan(obs),obs)
        obs = MV.masked_where(np.abs(obs)>90,obs)
        obs = obs(time=('1100-7-1','2020-12-31'))
        obs=mask_data(obs,obs.mask[0]) #Make all the obs have the same mask as the first datapoint
        f.close()
        list_of_obs+=[obs]
    writename = "../DROUGHT_ATLAS/PROCESSED/ALL_OBS.nc"
    CO = combine_observations(list_of_obs,writename=writename)
    modelwritename = "../DROUGHT_ATLAS/CMIP5/pdsi.ALL_OBS.hist.rcp85.nc"
    mm = mask_models(CO, writename = modelwritename)
    #now add in MXDA:
    atlas_name = "MXDA2.5"
    f = cdms.open("../DROUGHT_ATLAS/PROCESSED/"+atlas_name+".nc")
    obs = f("pdsi")
    obs = MV.masked_where(np.isnan(obs),obs)
    obs = MV.masked_where(np.abs(obs)>90,obs)
    obs = obs(time=('1100-7-1','2020-12-31'))
    obs=mask_data(obs,obs.mask[0]) #Make all the obs have the same mask as the first datapoint
    f.close()
    list_of_obs+=[obs]
    writename = "../DROUGHT_ATLAS/PROCESSED/ALL_OBS_plus_MEX.nc"
    CO = combine_observations(list_of_obs,writename=writename)
    modelwritename = "../DROUGHT_ATLAS/CMIP5/pdsi.ALL_OBS_plus_MEX.hist.rcp85.nc"
    mm = mask_models(CO, writename = modelwritename)
    
          
def dai_jja():
    f = cdms.open("../DROUGHT_ATLAS/pdsi.mon.mean.selfcalibrated.nc")
    dai=f("pdsi")
    cdutil.setTimeBoundsMonthly(dai)
    dai_jja = cdutil.JJA(dai)
    fgrid = cdms.open("OBS/gpcp.precip.mon.mean.nc")
    gpcp_grid = fgrid("precip").getGrid()
    fgrid.close()
    dai2=dai_jja.regrid(gpcp_grid,regridTool='regrid2')
    dai2.id="pdsi"
    for att in dai.attributes.keys():
        setattr(dai2,att,dai.attributes[att])
    fw = cdms.open("../DROUGHT_ATLAS/OBSERVATIONS/DAI_selfcalibrated.nc","w")
    fw.write(dai2)
    fw.close()
    return dai2

#NEED TO MAKE SURE HAS THE SAME MASK AT ALL TIME STEPS
    
def project_dai_on_solver(ALL,start='1970-1-1'):

    f = cdms.open("../DROUGHT_ATLAS/OBSERVATIONS/DAI_selfcalibrated.nc")
    dai_jja=f("pdsi")
    f.close()
    dai_jja_mask = mask_data(dai_jja,ALL.obs[0].mask)(time=(start,'2018-12-31'))
    newmask = np.prod(~dai_jja_mask.mask,axis=0)
    dai_jja_mask = mask_data(dai_jja_mask,newmask==0)
    solver = Eof(mask_data(ALL.mma,newmask==0))
    dai_jja_mask = mask_data(dai_jja_mask,solver.eofs()[0].mask)
    fac = da.get_orientation(solver)
    return solver.projectField(dai_jja_mask)[:,0]*fac
    
