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

from Plotting import *
import CMIP5_tools as cmip5

### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)
def version_num(fname):
    v = fname.split("ver-")[1].split(".")[0]
    if v[0]=='v':
        return int(v[1:])
    else:
        return int(v)
def get_corresponding_pr(fname):
    vari =fname.split(".")[7]
    if len(glob.glob(fname.replace(vari,"pr"))) == 1:
        return glob.glob(fname.replace(vari,"pr"))[0]
    else:
        fnames = glob.glob(fname.replace(vari,"pr").split(".ver")[0]+"*")
        if len(fnames)>0:
            i = np.argmax(map(version_num,fnames))
            return fnames[i]
        else:
            possmats = glob.glob(fname.replace(vari,"pr").split("cmip5.")[0]+"*")
            return difflib.get_close_matches(fname,possmats)[0]


test = cmip5.get_datafiles("historical","tas")

def Koeppen(fname):
    base = string.join(fname.split(".")[1:4],".")
    f = cdms.open(fname)
    tas = cdutil.ANNUALCYCLE.climatology(f("tas",time=('1979-1-1','2005-12-31')))-273.15 #convert from K to C
    fpr = cdms.open(get_corresponding_pr(fname))
    pr = cdutil.ANNUALCYCLE.climatology(fpr("pr",time=('1979-1-1','2005-12-31')))*(86400 * 30) #convert to mm
    
    fland = cdms.open(cmip5.landfrac(fname))
    fglac = cdms.open(cmip5.glacierfrac(fname))
    land = fland("sftlf")
    glacier=fglac("sftgif")

    #mask ocean and ice sheets
    totmask = np.logical_or(land==0,glacier==100.)
    totmask =np.repeat(totmask.asma()[np.newaxis],12,axis=0)
    tasmask = MV.masked_where(totmask,tas)
    prmask = MV.masked_where(totmask,pr)
    
    #mean annual precip
    pr_ann = MV.average(prmask,axis=0)
    pr_ann_sum = MV.sum(prmask,axis=0)
    
    #wettest month precip
    pr_wet = MV.max(prmask,axis=0)
    #driest month precip
    pr_dry = MV.min(prmask,axis=0)


    #mean annual T
    tas_ann = MV.average(tasmask,axis=0)
    tas_ann_sum = MV.sum(tasmask,axis=0)
    #wettest month T
    tas_hot = MV.max(tasmask,axis=0)
    #driest month T
    tas_cold = MV.min(tasmask,axis=0)
   
    #Calculate Summer/Winter (Summer (winter) is defined as the warmer (cooler) six month period of ONDJFM and AMJJAS.)
    Oct_Mar_index = np.array([10,11,12,1,2,3])-1
    Apr_Sep_index=np.arange(4,10)-1
    hasborealsummer = MV.average(tasmask.asma()[Apr_Sep_index],axis=0)>MV.average(tasmask.asma()[Oct_Mar_index],axis=0)
    hasaustralsummer = ~hasborealsummer
    
    Pmax_boreal_summer = np.ma.max(prmask.asma()[Apr_Sep_index],axis=0)
    Pmax_austral_winter = Pmax_boreal_summer

    Pmin_boreal_summer = np.ma.min(prmask.asma()[Apr_Sep_index],axis=0)
    Pmin_austral_winter = Pmin_boreal_summer
    
    Pmax_boreal_winter = np.ma.max(prmask.asma()[Oct_Mar_index],axis=0)
    Pmax_austral_summer = Pmax_boreal_winter

    Pmin_boreal_winter = np.ma.min(prmask.asma()[Oct_Mar_index],axis=0)
    Pmin_austral_summer = Pmin_boreal_winter

    Ptot_boreal_winter =  np.ma.sum(prmask.asma()[Oct_Mar_index],axis=0)
    Ptot_austral_winter = np.ma.sum(prmask.asma()[Apr_Sep_index],axis=0)
   
    
    
    
    #Number of months where the temperature > 10c
    tas_mon10 = np.ma.sum(np.ma.masked_where(totmask,np.array(tasmask>10,dtype=np.int)),axis=0)

    #Precip of the driest month in summer
   
    pr_sdry = np.ma.zeros(land.shape)+1.e20
    pr_sdry[np.where(hasborealsummer)]=Pmin_boreal_summer[hasborealsummer]
    pr_sdry[np.where(hasaustralsummer)]=Pmin_austral_summer[hasaustralsummer]
    
    pr_sdry=MV.masked_where(totmask[0],pr_sdry)
    pr_sdry.setAxisList(land.getAxisList())

    #Precip of the wettest month in summer
    pr_swet = np.ma.zeros(land.shape)+1.e20
    pr_swet[np.where(hasborealsummer)]=Pmax_boreal_summer[hasborealsummer]
    pr_swet[np.where(~hasborealsummer)]=Pmax_austral_summer[hasaustralsummer]
    pr_swet=MV.masked_where(totmask[0],pr_swet)
    pr_swet.setAxisList(land.getAxisList())

    #Precip of the driest month in winter
    pr_wdry = np.ma.zeros(land.shape)+1.e20
    pr_wdry[np.where(hasborealsummer)]=Pmin_boreal_winter[hasborealsummer]
    pr_wdry[np.where(~hasborealsummer)]=Pmin_austral_winter[hasaustralsummer]
    pr_wdry=MV.masked_where(totmask[0],pr_sdry)
    pr_wdry.setAxisList(land.getAxisList())

     #Precip of the wettest month in winter
    
    pr_wwet = np.ma.zeros(land.shape)+1.e20
    pr_wwet[np.where(hasborealsummer)]=Pmax_boreal_winter[hasborealsummer]
    pr_wwet[np.where(~hasborealsummer)]=Pmax_austral_winter[hasaustralsummer]
    pr_wwet=MV.masked_where(totmask[0],pr_swet)
    pr_wwet.setAxisList(land.getAxisList())

    #Does >70% of precipitation occur in winter?

    pr_winter = np.ma.zeros(land.shape)+1.e20
    pr_winter[np.where(hasborealsummer)]=Ptot_boreal_winter[hasborealsummer]
    pr_winter[np.where(~hasborealsummer)]=Ptot_austral_winter[~hasborealsummer]

    pr_summer = np.ma.zeros(land.shape)+1.e20
    pr_summer[np.where(hasborealsummer)]=Ptot_austral_winter[hasborealsummer]
    pr_summer[np.where(~hasborealsummer)]=Ptot_boreal_winter[~hasborealsummer]
    
    thresh1 = pr_winter>=pr_ann_sum*0.7
    #Does >70% of precipitation occur in summer?
    thresh2 = pr_summer>=pr_ann_sum*0.7
    #Thresholds defined as in Kottek et al http://koeppen-geiger.vu-wien.ac.at/present.htm
    pr_threshold =np.array(2*tas_ann+14)
    pr_threshold[thresh1] = 2*tas_ann.asma()[thresh1]
    pr_threshold[thresh2]=2*tas_ann.asma()[thresh2]+28
        

    K = np.zeros(land.shape,dtype=np.string0)
    #A climates
    A = tas_cold>18
    #Tropical rain forest
    Af = np.logical_and(A,pr_dry>60)
    # Tropical monsoon
    Am = np.logical_and(A,pr_dry >= 100 - (pr_ann_sum / 25))
    #Tropical savannah
    Aw = np.logical_and(A,pr_dry< 100 - (pr_ann_sum / 25))
    #K[A]="A"
    K[Af]="Af"
    K[Am]="Am"
    K[Aw]="Aw"

    #B Climates (Dry)
    B = pr_ann_sum < 10*pr_threshold
    #Desert
    BW = np.logical_and(B,pr_ann_sum <= 5*pr_threshold)
    #Steppe
    BS = np.logical_and(B,pr_ann_sum > 5*pr_threshold)
    #Hot arid desert
    BWh = np.logical_and(BW,tas_ann>=18)
    #Cold arid desert
    BWk = np.logical_and(BW,tas_ann<18)
    #Hot arid steppe
    BSh = np.logical_and(BS,tas_ann>=18)
    #Cold arid desert
    BSk = np.logical_and(BS,tas_ann<18)
    
    K[BWk]="BWk"
    K[BWh]="BWh"
    K[BSk]="BSk"
    K[BSh]="BSh"

    #C climates (mild)
    notAorB = np.logical_and(~A,~B)
    trange = np.logical_and(tas_cold>0,tas_cold<18)
    trange = np.logical_and(trange,tas_hot>10)
    C = np.logical_and(notAorB,trange)

    #dry summer
    drysummer=np.logical_and(pr_sdry < 40,pr_sdry < pr_wwet/3)
    Cs = np.logical_and(C,drysummer)
    
    #dry winter
    drywinter = pr_wdry< pr_swet / 10.
    Cw = np.logical_and(drywinter,C)
    #no dry season
    noseas = np.logical_and(~Cs,~Cw)
    Cf = np.logical_and(C,noseas)

    #what if both summer and winter are dry?
    bothdry=np.logical_and(Cw,Cs)
    summerdrier = pr_summer<pr_winter
    #If winter is drier, assign to Cw
    Cw.asma()[bothdry] = ~summerdrier[bothdry]
    #If summer is drier, assign to Cs
    Cs.asma()[bothdry] = summerdrier[bothdry]

    #Hot summer
    hotsummer = tas_hot>22
    #warm summer
    warmsummer = np.logical_and(~hotsummer,tas_mon10 >= 4)
    #cold summer
    nothotorwarm = np.logical_and(~hotsummer,~warmsummer)
    tasmonbounds = np.logical_and(tas_mon10 < 4,tas_mon10 >=1)
    coldsummer = np.logical_and(nothotorwarm, tasmonbounds)

    Csa = np.logical_and(Cs,hotsummer)
    Csb = np.logical_and(Cs,warmsummer)
    Csc = np.logical_and(Cs,coldsummer)

    Cwa = np.logical_and(Cw,hotsummer)
    Cwb = np.logical_and(Cw,warmsummer)
    Cwc = np.logical_and(Cw,coldsummer)

    Cfa = np.logical_and(Cf,hotsummer)
    Cfb = np.logical_and(Cf,warmsummer)
    Cfc = np.logical_and(Cf,coldsummer)

    K[Csa] = "Csa"
    K[Csb] = "Csb"
    K[Csc] = "Csc"

    K[Cwa] = "Cwa"
    K[Cwb] = "Cwb"
    K[Cwc] = "Cwc"

    K[Cfa] = "Cfa"
    K[Cfb] = "Cfb"
    K[Cfc] = "Cfc"

    #D climates (cold)
    D=np.logical_and(tas_hot>10,tas_cold<=0)
     #dry summer
    drysummer=np.logical_and(pr_sdry < 40,pr_sdry < pr_wwet/3)
    Ds = np.logical_and(D,drysummer)
    
    #dry winter
    drywinter = pr_wdry< pr_swet / 10.
    Dw = np.logical_and(drywinter,D)
    #no dry season
    noseas = np.logical_and(~Ds,~Dw)
    Df = np.logical_and(D,noseas)

    #what if both summer and winter are dry?
    bothdry=np.logical_and(Dw,Ds)
    summerdrier = pr_summer<pr_winter
    #If winter is drier, assign to Dw
    Dw.asma()[bothdry] = ~summerdrier[bothdry]
    #If summer is drier, assign to Ds
    Ds.asma()[bothdry] = summerdrier[bothdry]

    #Hot summer
    hotsummer = tas_hot>22
    #warm summer
    warmsummer = np.logical_and(~hotsummer,tas_mon10 >= 4)
    #very cold winter
    verycoldwinter = tas_cold<-38
    #cold summer: here, not a,b, or d
    nothotorwarm = np.logical_and(~hotsummer,~warmsummer)

    coldsummer = np.logical_and(nothotorwarm, ~verycoldwinter)

    Dsa = np.logical_and(Ds,hotsummer)
    Dsb = np.logical_and(Ds,warmsummer)
    Dsc = np.logical_and(Ds,coldsummer)
    Dsd = np.logical_and(Ds,verycoldwinter)
    
    Dwa = np.logical_and(Dw,hotsummer)
    Dwb = np.logical_and(Dw,warmsummer)
    Dwc = np.logical_and(Dw,coldsummer)
    Dwd = np.logical_and(Dw,verycoldwinter)

    Dfa = np.logical_and(Df,hotsummer)
    Dfb = np.logical_and(Df,warmsummer)
    Dfc = np.logical_and(Df,coldsummer)
    Dfd = np.logical_and(Df,verycoldwinter)

    K[Dsa] = "Dsa"
    K[Dsb] = "Dsb"
    K[Dsc] = "Dsc"
    K[Dsd] = "Dsd"

    K[Dwa] = "Dwa"
    K[Dwb] = "Dwb"
    K[Dwc] = "Dwc"
    K[Dwd] = "Dwd"

    K[Dfa] = "Dfa"
    K[Dfb] = "Dfb"
    K[Dfc] = "Dfc"
    K[Dfd] = "Dfd"

    #E Climates (polar)
    E = tas_hot<10
    ET = np.logical_and(E,tas_hot>0)
    EF = np.logical_and(E,tas_hot<0)

    K[ET]="ET"
    K[EF] = "EF"
    f.close()
    fpr.close()
    fland.close()
    fglac.close()
    return K
    
    
    
    

    
    
    

    
    
    

    
