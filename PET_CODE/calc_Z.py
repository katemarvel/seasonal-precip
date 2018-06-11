import glob
import sys
import cdms2 as cdms
import numpy as np
import MV2 as MV
import difflib
import scipy.stats as stats
global crunchy
import socket
import datetime

if socket.gethostname().find("crunchy")>=0:
    crunchy = True
else:
    crunchy = False


import cdtime,cdutil,genutil
import calendar
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from Plotting import *
import CMIP5_tools as cmip5


### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)

def bucket2d(PET,P,WCBOT,WCTOP,SS,SU):
    """
    Written by Kate Marvel.  Adapted from code by Jacob Wolf (UIdaho) and Park Williams (LDEO)
    Inputs:
    PET: Potential evapotranspiration (mm)
    PR: Precipitation (mm)
    WCBOT: Soil moisture holding capacity in bottom layer (mm)
    WCTOP: Water holding capacity in top layer (mm)
    SS: Available moisture in surface layer at the start of the month (mm)
    SU:  Available moisture in underlying layer at start of month (mm)


    Outputs:
    PL: Potential loss
    ET: Actual evapotranspiration
    TL: Total loss
    RO: Runoff
    R: Recharge
    SSS: Updated surface soil moisture
    SSU: Updated moisture of underlying soil layers
    """

    SP = SS+SU #total soil moisture
    WCTOT = WCBOT+WCTOP #total soil moisture holding capacity
    PR=WCTOT-SP #potential recharge
    PRS=WCTOP-SS #recharge from the surface
    PRU=WCBOT-SU #recharge from the underlying levels
    
    #Potential loss
    
    straw = SU/WCTOT
    demand = PET-SS
    loss_from_soil = demand*straw+SS
    PL = MV.where(SS>=PET,PET,loss_from_soil)
    #If surface moisture exceeds PET, potential loss = PET
    #Otherwise, potential loss = surface moisture + remaining soil moisture times fraction of soil remaining (moisture gets increasingly difficult to extract)
    PL = MV.where(PL<=SP,PL,SP)
    #Potential loss can't exceed total soil moisture


    
    #If precip exceeds PET then get recharge and possibly runoff
    p_exceeds_PET = P>=PET
    recharge_only_top = np.ma.logical_and(p_exceeds_PET,(P-PET)<=PRS)
    recharge_under_layer = np.ma.logical_and(p_exceeds_PET,(P-PET)>PRS)
    runoff_occurs = np.ma.logical_and(recharge_under_layer,(P-PET-PRS)>PRU)

    #Recharge of surface layer
    RS = MV.where(recharge_under_layer,PRS,0)
    RS = MV.where(recharge_only_top,P-PET,RS)

    #Recharge of under layer
    RU = MV.where(recharge_under_layer,P-PET-RS,0)
    RU = MV.where(runoff_occurs,WCBOT-SU,RU)

    #Runoff
    RO = MV.where(runoff_occurs,P-PET-RS-RU,0)

    #Update soil moistures
    SSS = MV.where(recharge_under_layer,WCTOP,0)
    SSS = MV.where(recharge_only_top,SS+P-PET,SSS)

    SSU = MV.where(recharge_under_layer,SU+RU,0)
    SSU = MV.where(recharge_only_top,SU,SSU)

    R=RU+RS
    #Runoff can't exceed total precipitation
    R=MV.where(R>PR,PR,R)

    #Evaporation exceeds precip: no runoff or recharge
    pet_exceeds_p = P<PET
    evap_from_surface_only = np.logical_and(pet_exceeds_p,SS>=(PET-P))
    evap_from_both = np.logical_and(pet_exceeds_p,SS<(PET-P))

    #evaporation from the surface layer only
    SL = MV.where(evap_from_surface_only,PET-P,0.)
    SSS = MV.where(evap_from_surface_only,SS-SL,SSS)
    SSU = MV.where(evap_from_surface_only,SU,SSU)

    #evaporation from both layers
    SL = MV.where(evap_from_both,SS,SL)
    demand = PET-P-SL
    UL = MV.where(evap_from_both,demand*SU/WCTOT,0)
    UL = MV.where(UL>SU,SU,UL)


    TL = MV.where(pet_exceeds_p,SL+UL,0.)
    ET = MV.where(pet_exceeds_p,P+SL+UL,PET) #If P>PET, then ET = PET.  Otherwise ET balances precip and loss to surface/underlying layers
    ET = MV.where(PET<ET,PET,ET)

    return PL,ET,TL,RO,R,SSS,SSU
    
def pad_by_10(X,year1,year2):
    """
    Pad an array at the beginning with an artificial 10 year spinup, in which each value is set to the climatology
    """
    tax = X.getTime()
    lastten=[start.sub(x,cdtime.Months) for x in range(121)[1:]][::-1]
    dayax=np.array([x.torel(tax.units).value for x in lastten])
    tax_new = np.append(dayax,tax)
    new_time_axis=cdms.createAxis(tax_new)
    new_time_axis.designateTime()
    new_time_axis.id="time"
    new_time_axis.units=tax.units
    Xnew = MV.zeros((len(tax_new),)+X.shape[1:])
    Xnew[:120]=np.repeat(MV.average(X(time=(year1,year2)),axis=0).asma()[np.newaxis,:,:],120,axis=0)
    Xnew[120:]=X
    for k in X.attributes.keys():
        setattr(Xnew,k,X.attributes[k])
    Xnew.setAxisList([new_time_axis]+X.getAxisList()[1:])
    return Xnew



def pdsi_coeff(X,Y):
    """
    Calculate alpha,beta,gamma,delta by dividing variable by its potential value
    """
    denom_nonzero = Y !=0
    coeff = MV.where(denom_nonzero,X/Y,0)
    coeff = MV.where(X==0,1,coeff)
    return coeff


        
def calculate_Z(PET,P,WCTOP,WCBOT,year1,year2):
    """
    Written by Kate Marvel.  Adapted from code by  Park Williams (LDEO).
    Calculates the Palmer z-index
    
    Inputs:
    PET: Potential evapotranspiration (mm)
    PR: Precipitation (mm).  NOTE CMIP5 standard is kg/m s-2 so need to run convert_to_mm on CMIP5 output
    WCBOT: Soil moisture holding capacity in bottom layer (mm)
    WCTOP: Water holding capacity in top layer (mm)
    year1: start date of calibration period
    year2: stop date of calibration period


    Outputs:
    PL: Potential loss
    ET: Actual evapotranspiration
    TL: Total loss
    RO: Runoff
    R: Recharge
    SSS: Updated surface soil moisture
    SSU: Updated moisture of underlying soil layers
    """

    #INITIALIZE VARS
    WCTOT=WCBOT+WCTOP  # Total water holding capacity of the soil layers
    SS=WCTOP           # Surface soil moisture start at full capacity
    SU=WCBOT           # Underlying layer soil moisture start at full capacity
    SP=SS+SU           # Combined surface and underlying soil moisture

    #Get arrays for output
    pldat=MV.zeros(PET.shape)
    spdat=MV.zeros(PET.shape)
    prdat=MV.zeros(PET.shape)
    rdat=MV.zeros(PET.shape)
    tldat=MV.zeros(PET.shape)
    etdat=MV.zeros(PET.shape)
    rodat=MV.zeros(PET.shape)
    sssdat=MV.zeros(PET.shape)
    ssudat=MV.zeros(PET.shape)

    #get rid of badly calibrated negative PET
    PET = MV.where(PET<0,0,PET)

    #fake 10 year spinup
    PET_ex=pad_by_10(PET,year1,year2)
    P_ex=pad_by_10(P,year1,year2)

    #get rid of badly calibrated negative PET
    PET = MV.where(PET<0,0,PET)

    #run 2-layer bucket model
    for i in range(PET_ex.shape[0]):
        PR = WCTOT-SP
        PL,ET,TL,RO,R,SSS,SSU=bucket2d(PET_ex[i],P_ex[i],WCBOT,WCTOP,SS,SU)
        ET = MV.where(ET<0,0,ET)
        SS=SSS
        SU=SSU
        SP = SS+SU

        if i>=120:
            pldat[i-120] = PL
            spdat[i-120] = SS+SU
            prdat[i-120] = PR
            
            rdat[i-120] = R
           
           
            tldat[i-120] = TL
           
                
            etdat[i-120] = ET
            rodat[i-120] = RO
            sssdat[i-120] = SSS
            ssudat[i-120] = SSU

    rdat=MV.where(rdat>=prdat,prdat,rdat)
    tldat = MV.where(tldat>=pldat,pldat,tldat)

    for X in [spdat,pldat,prdat,rdat,tldat,etdat,rodat]:
        X.setAxisList(PET.getAxisList())
    
    #calculate means over calibration period
    SPSUM = cdutil.ANNUALCYCLE.climatology(spdat(time=(year1,year2)))
    PLSUM = cdutil.ANNUALCYCLE.climatology(pldat(time=(year1,year2)))
    PRSUM = cdutil.ANNUALCYCLE.climatology(prdat(time=(year1,year2)))
    RSUM = cdutil.ANNUALCYCLE.climatology(rdat(time=(year1,year2)))
    TLSUM = cdutil.ANNUALCYCLE.climatology(tldat(time=(year1,year2)))
    ETSUM = cdutil.ANNUALCYCLE.climatology(etdat(time=(year1,year2)))
    PESUM = cdutil.ANNUALCYCLE.climatology(PET(time=(year1,year2)))
    ROSUM = cdutil.ANNUALCYCLE.climatology(rodat(time=(year1,year2)))
    PSUM = cdutil.ANNUALCYCLE.climatology(P(time=(year1,year2)))
    
    # CAFEC: Climatology Appropriate for Existing Conditions (Palmer 1965, p 12)
    # Calculate the CAFEC coefficients
    
    #Coefficient of evaporation:  fraction of mean ET to mean potential ET
    alpha = pdsi_coeff(ETSUM,PESUM)

    #Coef of Recharge: ratio of mean recharge to mean potential recharge
    beta = pdsi_coeff(RSUM,PRSUM)

    #Coefficient of Runoff: Ratio of mean runoff to mean potential runoff
    gamma = pdsi_coeff(ROSUM,SPSUM)

    #Coefficient of Loss: Ratio of mean moisture loss to mean potential moisture loss
    delta = pdsi_coeff(TLSUM,PLSUM)

    TRAT=(PESUM+RSUM+ROSUM)/(PSUM+TLSUM)

    #CAFEC precipitation (needed to maintain "normal" moisture)
    nyears = PET.shape[0]/12
    repeat_it=lambda coef: np.ma.repeat(coef,nyears,axis=0)
    Phat = PET*repeat_it(alpha) + prdat*repeat_it(beta) + spdat*repeat_it(gamma) - pldat*repeat_it(delta)
    
   # Moisture departure (difference between precip and precip needed for normal conditions)
    DD=P-Phat
    DD.setAxisList(P.getAxisList())
    SABSD=MV.absolute(DD)
    DBAR = cdutil.ANNUALCYCLE.climatology(SABSD(time=(year1,year2)))
    
    # Weird empirical scaling thing to standardize moisture availability departures. THIS IS DUMB.        
    AKHAT = 1.5*MV.log10((TRAT+2.8*25.4)/DBAR)+0.5
    AKHAT = MV.where(AKHAT<0,0,AKHAT)
    
    annsum = MV.sum(AKHAT*DBAR,axis=0)

    AK = 17.67*25.4*AKHAT/annsum
    
    Z = DD/25.4*repeat_it(AK)

    #Force Z between -16 and 16
    Z=MV.where(Z>16,16,Z)
    Z=MV.where(Z<-16,-16,Z)
    
    Z.setAxisList(P.getAxisList())
    Z.id = "z"
    Z.units = "mm"
    Z.info = "Created by Kate Marvel using calc_Z.py"
    Z.creation_date = datetime.datetime.now().isoformat()
    
    return Z
    
        
    
    
                  
    
    
    
    
    
    
    
    
