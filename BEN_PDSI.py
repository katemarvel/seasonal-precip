### Import useful routines
import numpy as np
import string
import glob
import os
from collections import Counter
import scipy.stats as stats 
### Import CDAT routines ###
import MV2 as MV
import cdms2 as cdms
import genutil
import cdutil
import cdtime
from eofs.cdms import Eof

### Import scipy routines for smoothing, interpolation
from scipy.interpolate import interp1d
from scipy.optimize import brentq,fminbound
import scipy.ndimage as ndimag

import CMIP5_tools as cmip5
import DA_tools as da
from Plotting import *


### Import plotting routines
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap  
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import mpl

### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)


def landplot(data,**kwargs):
    """ Plot data on cyl projection with lon_0 = prime meridian"""
    if 'cmap' not in kwargs.keys():
        kwargs['cmap']=cm.BrBG
    if 'vmin' not in kwargs.keys():
        a = np.max(np.abs(data))
        kwargs['vmin']=-a
        kwargs['vmax']=a
        
    #if 'mask' not in kwargs.keys():
     #   kwargs['mask']=False
    
   # else:
    #    land_mask = gpcp_land_mask()
     #   data = MV.masked_where(land_mask,data)
    m = bmap(data,projection="cyl",lon_0=0,**kwargs)
    
    return m

def jja_pdsi(X):
    """Average CMIP5 PDSI over boreal summer months """
    jja=MV.average(X,axis=1)
    tax = jja.getAxis(0)
    tax.units = 'years since 0000-7-1'
    tax.id="time"
    jja.setAxis(0,tax)
    jja.id = 'pdsi'
    return jja

def transfer_to_local():
    """Transfer from Ben Cook's hard drive to local desktop on 4/17"""
    direc = "/Volumes/Apollo/PIERREPDSI/PDSI/"
    files = glob.glob(direc+"*")
    for fil in files:
        print fil
        writefname = "/Users/kmarvel/Documents/PDSI/"+fil.split("/")[-1].replace("all.indices","jja")
        f = cdms.open(fil)
        fw = cdms.open(writefname,"w")
        X=f["PDSI"]
        jja=jja_pdsi(X)
        fw.write(jja)
        fw.close()
        f.close()

def signal_to_noise_map(drought_atlas,start_year=1100,modern_start=1979):

    preindustrial = drought_atlas[start_year:1850]
    modern = drought_atlas[modern_start:]
    modern_times = modern.shape[0]
    nt,nlat,nlon=drought_atlas.shape
    SN = MV.zeros(drought_atlas.shape[1:])+1.e20
    signals = cmip5.get_linear_trends(modern)
    for i in range(nlat):
        for j in range(nlon):
            if not drought_atlas.mask[-1,i,j]:
                noise = da.get_slopes(preindustrial[:,i,j],modern_times)/365. #get_slopes assumes time units are days; these are years
                width=np.ma.std(noise)
                signal = signals[i,j]
                SN[i,j]=signal/width
    return SN
    

    
def regrid_and_truncate(X,the_grid=None):
    if the_grid is None:
        fobs = cdms.open("OBS/gpcp.precip.mon.mean.nc")
        the_grid = fobs["precip"].getGrid()
    Xt = X(time=('1900-1-1','2099-12-31'))
    Xr = Xt.regrid(the_grid,regridTool='regrid2')
    if the_grid is None:
        fobs.close()
    return Xr
def average_and_truncate(X,region):
    
    Xt = X(time=('1900-1-1','2099-12-31'))
    Xr = cdutil.averager(Xt(region),axis='xy')
    
    return Xr

def write_average_ensemble(region,label):
    direc = "/Users/kmarvel/Google Drive/PDSI/"
    jjafiles = glob.glob(direc+"*")
    nf=len(jjafiles)
    i=0
    f=cdms.open(jjafiles[i])
    X=f("pdsi")
    Xt=average_and_truncate(X,region)
    
    ens = MV.zeros((nf,)+Xt.shape)+1.e20
    ens[i]=Xt
    for i in range(nf)[1:]:
        f=cdms.open(jjafiles[i])
        X=f("pdsi")
        Xt=average_and_truncate(X,region)
        try:
            ens[i]=Xt
        except:
            continue
        axes = Xt.getAxisList()
        f.close()
    ens=MV.masked_where(np.abs(ens)>1.e10,ens)
    ens = MV.masked_where(np.isnan(ens),ens)
    ens.id="pdsi"
    modax = cmip5.make_model_axis(jjafiles)
    ens.setAxisList([modax]+axes)
    fw = cdms.open("../DROUGHT_ATLAS/CMIP5/pdsi."+label+".hist.rcp85.nc","w")
    ens.id='pdsi'
    fw.write(ens)
    fw.close()
    return ens
def write_regridded_ensemble(the_grid=None,label="ensemble"):
    direc = "/Users/kmarvel/Google Drive/PDSI/"
    jjafiles = glob.glob(direc+"*")
    nf=len(jjafiles)
    i=0
    f=cdms.open(jjafiles[i])
    X=f("pdsi")
    Xt=regrid_and_truncate(X,the_grid=the_grid)
    
    ens = MV.zeros((nf,)+Xt.shape)+1.e20
    ens[i]=Xt
    for i in range(nf)[1:]:
        f=cdms.open(jjafiles[i])
        X=f("pdsi")
        Xt=regrid_and_truncate(X,the_grid=the_grid)
        try:
            ens[i]=Xt
        except:
            continue
        axes = Xt.getAxisList()
        f.close()
    ens=MV.masked_where(np.abs(ens)>1.e10,ens)
    ens = MV.masked_where(np.isnan(ens),ens)
    ens.id="pdsi"
    modax = cmip5.make_model_axis(jjafiles)
    ens.setAxisList([modax]+axes)
    fw = cdms.open("../DROUGHT_ATLAS/CMIP5/pdsi."+label+".hist.rcp85.nc","w")
    ens.id='pdsi'
    fw.write(ens)
    fw.close()
    return ens
    
def get_rid_of_bad(h85):
    #Fgoals and bcc have the wrong grid (FIX THIS LATER???)
    models = cmip5.models(h85)
    newmodels=models[:3]+models[4:22]+models[23:]
    h85_mod = MV.zeros((len(newmodels),)+h85.shape[1:])
    h85_mod = MV.concatenate((h85[:3],h85[4:22],h85[23:]))
    h85_mod.setAxis(0,cmip5.make_model_axis(newmodels))
    for i in range(len(h85.shape))[1:]:
        h85_mod.setAxis(i,h85.getAxis(i))
    h85_mod.id="pdsi"
    return h85_mod

def europe_plot(data,projection,**kwargs):
    laterr=2
    lonerr=20
    v=max([np.abs(np.ma.min(data)),np.abs(np.ma.max(data))])
    if "vmin" not in kwargs.keys():
        kwargs["vmin"]=-v
    if "vmax" not in kwargs.keys():
        kwargs["vmax"]=v
    
    data = MV.masked_where(np.isnan(data),data)
    #fig = plt.figure()
    #ax = fig.add_axes([0.1,0.1,0.8,0.8])
    llcrnrlat=data.getLatitude()[:][0]-laterr
    urcrnrlat=data.getLatitude()[:][-1]+laterr

    llcrnrlon=data.getLongitude()[:][0]-2
    urcrnrlon=data.getLongitude()[:][-1]+20

    lat_0=np.median(data.getLatitude())
    lon_0=np.median(data.getLongitude())

   # m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,rsphere=(6378137.00,6356752.3142),resolution='l',area_thresh=1000.,projection='lcc',lat_0=lat_0,lon_0=lon_0,ax=ax)
    m=Basemap(llcrnrlon=llcrnrlon, \
            llcrnrlat=llcrnrlat, \
            urcrnrlon=urcrnrlon, \
            urcrnrlat= urcrnrlat, \
            llcrnrx=None, \
            llcrnry=None, \
            urcrnrx=None, \
            urcrnry=None, \
            width=None, \
            height=None, \
            projection=projection, \
            resolution='c', \
            area_thresh=1000, \
            rsphere=6370997.0, \
            ellps=None, \
            lat_ts=None, \
            lat_1=None, \
            lat_2=None, \
            lat_0=51.0, \
            lon_0=13.0, \
            lon_1=None, \
            lon_2=None, \
            o_lon_p=None, \
            o_lat_p=None, \
            k_0=None, \
            no_rot=True, \
            suppress_ticks=True, \
            satellite_height=35786000, \
            boundinglat=None, \
            fix_aspect=True, \
            anchor='C', \
            celestial=False, \
            round=False, \
            epsg=None, \
            ax=None)
    lon = data.getLongitude().getBounds()[:,0]
    lat = data.getLatitude().getBounds()[:,0]
    x,y=m(*np.meshgrid(lon,lat))
    stuff=m.pcolormesh(x,y,data,**kwargs)
    plt.colorbar(stuff)
    return m
def scratch(good):
    #TO DO:
    # mask OWDA, NADA, MADA regions


    f=cdms.open("../DROUGHT_ATLAS/PROCESSED/OWDA.nc")
    owda=f("pdsi")
    f.close()
    f = cdms.open("../DROUGHT_ATLAS/CMIP5/pdsi.ensemble.hist.rcp85.nc")
    h85=f("pdsi")
    good = get_rid_of_bad(h85)
    

    owda_region = cdutil.region.domain(latitude=(np.min(owda.getLatitude()[:]),np.max(owda.getLatitude()[:])),longitude= (np.min(owda.getLongitude()[:]),np.max(owda.getLongitude()[:])))
    
    ow = MV.average(good(owda_region),axis=0)
    
    owsolver = Eof(MV.average(sc.mask_data(good(owda_region),owda_regrid[-1].mask),axis=0))
    ow_fingerprint = owsolver.eofs()[0]
    
    owda_regrid = owda.regrid(ow.getGrid(),regridTool='regrid2') 
    owda_regrid_mask=sc.mask_data(owda_regrid,ow_fingerprint.mask)


    
    
    nada_region = cdutil.region.domain(latitude=(np.min(nada.getLatitude()[:]),np.max(nada.getLatitude()[:])),longitude= (np.min(nada.getLongitude()[:]),np.max(nada.getLongitude()[:])))



    nasolver = Eof(MV.average(good(nada_region),axis=0))
    na_fingerprint = nasolver.eofs()[0] 

    nada_regrid = nada.regrid(na_fingerprint.getGrid(),regridTool='regrid2') 
    nada_regrid_mask=sc.mask_data(nada_regrid,na_fingerprint.mask)

from seasonal_cycle_utils import mask_data

#Testing: is there a better way to put model data on drought atlas grids?
class DroughtAtlas():
    def __init__(self,name,cutoff='0001-1-1'):
        self.name=name
        #if name.find("+")<0:
        f = cdms.open("../DROUGHT_ATLAS/PROCESSED/"+name+".nc")
        obs = f("pdsi")
        self.obs = MV.masked_where(np.isnan(obs),obs)
        self.obs = MV.masked_where(np.abs(self.obs)>90,self.obs)
        self.obs = self.obs(time=(cutoff,'2020-12-31'))

        self.obs=mask_data(self.obs,self.obs.mask[0]) #Make all the obs have the same mask as the first datapoint
        f.close()
        fm = cdms.open("../DROUGHT_ATLAS/CMIP5/pdsi."+name+".hist.rcp85.nc")
        self.model=get_rid_of_bad(fm("pdsi"))
        self.model=MV.masked_where(np.isnan(self.model),self.model)
        fm.close()

        
        
        
        # else:
        #DEPRECATED: MERGE observations onto common grid using old code
        #     name1,name2=name.split("+")
        #     f1 = cdms.open("../DROUGHT_ATLAS/PROCESSED/"+name1+".nc")
        #     obs1 = f1("pdsi")
        #     obs1 = MV.masked_where(np.isnan(obs1),obs1)
        #     obs1 = MV.masked_where(np.abs(obs1)>90,obs1)
        #     obs1 = obs1(time=(cutoff,'2017-12-31'))
       
        #     obs1=mask_data(obs1,obs1.mask[0])
        #     f1.close()
        #     fm1 = cdms.open("../DROUGHT_ATLAS/CMIP5/pdsi."+name1+".hist.rcp85.nc")
        #     model1=get_rid_of_bad(fm1("pdsi"))
        #     model1=MV.masked_where(np.isnan(model1),model1)
        #     fm1.close()

            
        #     f2 = cdms.open("../DROUGHT_ATLAS/PROCESSED/"+name2+".nc")
        #     obs2 = f2("pdsi")
        #     obs2 = MV.masked_where(np.isnan(obs2),obs2)
        #     obs2 = MV.masked_where(np.abs(obs2)>90,obs2)
        #     obs2 = obs2(time=(cutoff,'2017-12-12'))
       
        #     obs2=mask_data(obs2,obs2.mask[0])
        #     f2.close()
        #     fm2 = cdms.open("../DROUGHT_ATLAS/CMIP5/pdsi."+name2+".hist.rcp85.nc")
        #     model2=get_rid_of_bad(fm2("pdsi"))
        #     model2=MV.masked_where(np.isnan(model2),model2)
        #     fm2.close()


        #     self.obs=merge.merge(obs1,obs2)
        #     self.model=merge.merge(model1,model2)
            

            
       
        
    
        mma = MV.average(self.model,axis=0)
        self.mma = mask_data(mma,self.obs[0].mask) #make all the models have the same mask
        self.solver = Eof(self.mma)
        eofmask=self.solver.eofs()[0].mask
        self.fac=da.get_orientation(self.solver)
        self.projection = self.solver.projectField(mask_data(self.obs,eofmask))[:,0]*self.fac
        self.noise = self.projection(time=('1-1-1','1850-1-1'))

            
    def plot_fingerprint(self,ax1=None,ax2=None):
        eof1=self.solver.eofs()[0]*self.fac
        #v=max([np.abs(np.ma.min(eof1)),np.abs(np.ma.max(eof1))])
        pc1 = self.solver.pcs()[:,0]*self.fac
        if ax1 is None:
            ax1=plt.subplot(211)
        if self.name not in ["OWDA","MXDA","NADA","MADA"]:
            #m=bmap(eof1,cmap=cm.BrBG,vmin=-v,vmax=v)
            m=landplot(eof1)
            m.drawcoastlines()
            #plt.colorbar(orientation="horizontal",)
        else:
            m=plot_regional(self.solver.eofs()[0],self.name,cmap=cm.BrBG)
        plt.subplot(212)
        time_plot(pc1)
    def model_projections(self):
        to_proj = mask_data(self.model,self.solver.eofs()[0].mask)
        P=MV.zeros(to_proj.shape[:2])
        for i in range(to_proj.shape[0]):
            tp = to_proj[i]
            mma_mask = mask_data(self.mma,tp[0].mask)
            solver = Eof(mma_mask)
            fac=da.get_orientation(solver)
            
            P[i] = solver.projectField(tp)[:,0]*fac
        P.setAxisList(to_proj.getAxisList()[:2])
        self.P=P
    def sn_at_time(self,start_time,L,overlapping=True):
        if not hasattr(self,"P"):
            self.model_projections()
        stop_time=start_time.add(L,cdtime.Years)
        modslopes = cmip5.get_linear_trends(self.P(time=(start_time,stop_time)))
        if overlapping:
            noiseterm = bootstrap_slopes(self.noise,L)
        else:
            noiseterm = da.get_slopes(self.noise,L)/365.
        return modslopes,noiseterm
    def obs_SN(self,start_time,stop_time=None,overlapping=True,include_dai=False):
        if stop_time is None:
            stop_time=cmip5.stop_time(self.projection)
        target_obs = self.projection(time=(start_time,stop_time))
        L=len(target_obs)
        modslopes,noiseterm = self.sn_at_time(start_time,L,overlapping=True)
        ns=np.std(noiseterm)
        signal = float(cmip5.get_linear_trends(target_obs))/ns
        plt.hist(modslopes/ns,20,normed=True,color=cm.Oranges(.8),alpha=.5)
        lab = str(start_time.year)+"-"+str(stop_time.year)
        da.fit_normals_to_data(modslopes/ns,color=cm.Oranges(.9),label=lab+" Model projections")

        plt.hist(noiseterm/ns,20,normed=True,color=cm.Greens(.8),alpha=.5)
        da.fit_normals_to_data(noiseterm/ns,color=cm.Greens(.9),label="Pre-1850 tree-ring reconstructions")
        plt.axvline(signal,color=cm.Blues(.8),lw=3,label=lab+" Tree-ring reconstructions")
        print signal
        if include_dai:
            dai_proj = self.project_dai_on_solver(start=start_time)
            daitrend = cmip5.get_linear_trends(dai_proj(time=(start_time,stop_time)))
            
            
            
        plt.legend(loc=0)
        
        
        
    def time_of_emergence(self,start_time,times = np.arange(10,76),plot=True,**kwargs):
        if not hasattr(self,"P"):
            self.model_projections()
        nmod,nyears = self.P.shape
        self.TOE=MV.zeros((nmod,len(times)))
        for i in range(len(times)):
            L=times[i]
            modslopes,noiseterm = self.sn_at_time(start_time,L)
            sns=modslopes/np.std(noiseterm)
            self.TOE[:,i]=sns
        self.TOE.setAxis(0,self.P.getAxis(0))
        if plot:
            endyears = start_time.year+times
            plt.plot(endyears,np.ma.average(self.TOE.asma(),axis=0),lw=4,label=self.name+" model mean signal",**kwargs)
            plt.fill_between(endyears,np.ma.min(self.TOE.asma(),axis=0),np.ma.max(self.TOE.asma(),axis=0),alpha=.4,**kwargs)
            plt.axhline(stats.norm.interval(.9)[-1],c="r",lw=3)
            plt.xlabel("Trend end year")
            plt.ylabel("Signal-to-noise ratio")

    def project_dai_on_solver(self,start='1970-1-1'):

        f = cdms.open("../DROUGHT_ATLAS/OBSERVATIONS/DAI_selfcalibrated.nc")
        dai_jja=f("pdsi")
        f.close()
        dai_jja_mask = mask_data(dai_jja,self.obs[0].mask)(time=(start,'2018-12-31'))
        newmask = np.prod(~dai_jja_mask.mask,axis=0)
        dai_jja_mask = mask_data(dai_jja_mask,newmask==0)
        solver = Eof(mask_data(self.mma,newmask==0))
        dai_jja_mask = mask_data(dai_jja_mask,solver.eofs()[0].mask)
        fac = da.get_orientation(solver)
        return solver.projectField(dai_jja_mask)[:,0]*fac
        
            
        
def regional_DA(OWDA,region,start_time=None,typ='fingerprint',return_noise=False):
    if start_time is None:
        start_time=cdtime.comptime(2000,1,1)
    times = np.arange(10,76)
    modeldata = mask_data(OWDA.model(region),OWDA.obs(region)[0].mask)
    if typ == 'fingerprint':
        mma = MV.average(modeldata,axis=0)
        solver = Eof(mma)
        
        to_proj = mask_data(modeldata,solver.eofs()[0].mask)
        P=MV.zeros(to_proj.shape[:2])
        for i in range(to_proj.shape[0]):
            tp = to_proj[i]
            mma_mask = mask_data(mma,tp[0].mask)
            solver = Eof(mma_mask)
            fac=da.get_orientation(solver)
            P[i] = solver.projectField(tp)[:,0]*fac
        P.setAxisList(to_proj.getAxisList()[:2])
        noise = solver.projectField(OWDA.obs(region))[:,0]
    else:
        P = cdutil.averager(modeldata,axis='xy')
        noise = cdutil.averager(OWDA.obs(region),axis='xy')
    if return_noise:
        return P,noise
    else:
        nmod,nyears = P.shape
        TOE=MV.zeros((nmod,len(times)))
        for i in range(len(times)):
            L=times[i]
            stop_time=start_time.add(L,cdtime.Years)
            modslopes = cmip5.get_linear_trends(P(time=(start_time,stop_time)))
            
            noiseterm = np.std(bootstrap_slopes(noise,L))

      
            TOE[:,i]=modslopes/noiseterm
        TOE.setAxis(0,P.getAxis(0))
        return TOE
        
        
        
def regionplot(TOES,cmap=cm.Set1):
    endyears = np.arange(10,76)+2000
    counter=0.
    L=float(len(TOES.keys()))
    for k in sorted(TOES.keys()):
        y=TOES[k]
        sigma=np.ma.std(y.asma(),axis=0)
        mu = np.ma.average(y.asma(),axis=0)
        plt.fill_between(endyears,np.min(y.asma(),axis=0),np.max(y.asma(),axis=0),color=cmap(counter/L),alpha=.3)
        plt.plot(endyears,mu,color=cmap(counter/L),lw=3,label=k)
        counter+=1.
        
                


def merge_grids(X,Y):
    xlats = X.getLatitude()[:]
    ylats = Y.getLatitude()[:]
    mylats = np.union1d(xlats,ylats)

    xlons = X.getLongitude()[:]
    ylons = Y.getLongitude()[:]
    mylons = np.union1d(xlons,ylons)

    test = MV.zeros(X.shape[:-2]+(len(mylats),len(mylons)))
    for i in range(len(mylats)):
        for j in range(len(mylons)):
            lat = mylats[i]
            lon=mylons[j]
            #figure out what grid(lat,lon) belong to
            if (lat in xlats) and (lon in xlons):
                print "X"
                ti = int(np.where(xlats == lat)[0])
                tj = int(np.where(xlons == lon)[0])
                test[:,i,j]=X[:,ti,tj]
            elif (lat in ylats) and (lon in ylons):
                print "Y"
                ti = int(np.where(ylats == lat)[0])
                tj = int(np.where(ylons == lon)[0])
                test[:,i,j]=Y[:,ti,tj]
            else:
                continue
        return test
        
               
#Test levant
latmin=27.25
latmax = 37
lonmin=32
lonmax=44.75
levant = cdutil.region.domain(latitude=(latmin,latmax),longitude=(lonmin,lonmax))

WestMED = cdutil.region.domain(latitude=(32,42),longitude=(-10,0))
EastMED = cdutil.region.domain(latitude=(36,41),longitude=(20,37))
MidEast = cdutil.region.domain(latitude=(30,34),longitude=(33,47))



def bootstrap_slopes(noise,L):
    nt=noise.shape[0]-L
    test = MV.zeros((nt,L))
    
    for i in range(nt):
        test[i]=noise[i:L+i]
    test.setAxis(1,noise[:L].getAxis(0))
    return cmip5.get_linear_trends(test)  


def get_map(name):
    if name == "OWDA":
        m =  Basemap(projection='lcc', resolution='c',width=8E6, height=8E6, lat_0=45, lon_0=20,)
    elif name == "NADA":
        m = Basemap(projection='lcc', resolution='c',width=8E6, height=8E6, lat_0=50, lon_0=-100,)
    elif name == "MADA":
        m = Basemap(projection='lcc', resolution='c',width=8E6, height=8E6, lat_0=30, lon_0=90,)
    elif name == "MXDA":
        m = Basemap(projection='lcc', resolution='c',width=8E6, height=8E6, lat_0=45, lon_0=-100,)
    return m
        
        
def plot_regional(data,name,**kwargs):
    v=max([np.abs(np.ma.min(data)),np.abs(np.ma.max(data))])
    if "vmin" not in kwargs.keys():
        kwargs["vmin"]=-v
    if "vmax" not in kwargs.keys():
        kwargs["vmax"]=v
    if cmap not in kwargs.keys():
        kwargs['cmap']=cm.BrBG
    m = get_map(name)    
    lon = data.getLongitude().getBounds()[:,0]
    lat = data.getLatitude().getBounds()[:,0]
    x,y=m(*np.meshgrid(lon,lat))
    stuff=m.pcolormesh(x,y,data,**kwargs)
    plt.colorbar(stuff)
    #m.drawcoastlines(color='gray')
    return m



def soilmoisture(depth,mask=None):
    if depth == "pdsi":
        f = cdms.open("../DROUGHT_ATLAS/CMIP5/pdsi.ensemble.hist.rcp85.nc")
        variable="pdsi"
    else:
        f = cdms.open("../DROUGHT_ATLAS/CMIP5/sm"+depth+".ensemble.hist.rcp85.nc")
        variable = "sm"+depth
    sm = get_rid_of_bad(f(variable))
    sm = MV.masked_where(np.isnan(sm),sm)
    f.close()
    if mask is not None:
        sm = mask_data(sm,mask)
    return sm

def soilmoisture_fingerprints(mask,name=None):
    depths = ["30cm","2m","pdsi"]
    letters = ["(a): ","(b): ","(c): "]
    pcs = []
    pclabels = []
    for depth in depths:
        i=depths.index(depth)
        plt.subplot(2,2,i+1)
        sm = soilmoisture(depth,mask=mask)
        solver = Eof(MV.average(sm,axis=0))
        fac = da.get_orientation(solver)
        if name is None:
            m=landplot(fac*solver.eofs()[0],vmin=-.1,vmax=.1)
            plt.colorbar(orientation='horizontal',label='EOF loading')
        else:
            m=plot_regional(fac*solver.eofs()[0],name,vmin=-.1,vmax=.1)
            m.drawcountries()
        m.drawcoastlines(color='gray')
        
        
        plt.title(letters[i]+depth+" fingerprint")
        pcs+=[fac*solver.pcs()[:,0]]

    
    plt.subplot(2,2,4)
    for i in range(3):
        time_plot(pcs[i],label=depths[i])
    plt.legend(loc=0)
    plt.title("(d): Principal Components")
    plt.xlabel("Time")
    plt.ylabel("Temporal amplitude")
    
