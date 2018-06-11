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


def end_spell_P(U,Ze,nmonths,current_month,typ="wet"):
    """
    Calculate the probability that a dry/wet spell has ended
    Ref: Eq (30), pg 29 Palmer 1965
    
    """
    nmonths=nmonths.astype(np.int64).asma()
    if type(U)==type(MV.array([])):
        U = U.asma()
    Ut = U[:current_month+1][::-1] #flip to go back in time
    CS = np.ma.cumsum(Ut,axis=0) #sum backwards in time. CS[0] = Ut[current_month]; CS[n] = Ut[current_month]+Ut[1 month_before]+ ... Ut[n months before]
    Uij=nmonths.choose(CS) #summation limit, different for each spatial point, is the first month of the current spell
    if typ=="dry":
        Uij=MV.where(Uij<0,0,Uij)
    elif typ=="wet":
        Uij=MV.where(Uij>0,0,Uij)
    Q =Ze+Uij-U[current_month]
    Pe = 100*Uij/Q
   
    return Pe
    
    


def calc_PDSI(Z,BTthresh,calc_modified=True):
    PDSI = MV.zeros(Z.shape)+1.e20
    
    if calc_modified:
        PMDI = MV.zeros(Z.shape)+1.e20
    #Allocate space for effective wetness/dryness

    Ud = MV.zeros(Z.shape)+1.e20
    Uw = MV.zeros(Z.shape)+1.e20
    X1 = MV.zeros(Z.shape)+1.e20
    X2 = MV.zeros(Z.shape)+1.e20
    X3 = MV.zeros(Z.shape)+1.e20
    Pe = MV.zeros(Z.shape)+1.e20
    montho = MV.zeros(Z.shape) #log how many months wet/dry spells have been going on
    X = MV.zeros(Z.shape)
    
    #Start in month 1, inheriting no wetness/dryness from previous month
    i = 0
    XX = Z[i]/3. #PDSI value if there is no wet or dry spell underway
    X1i = MV.where(XX>0,XX,0)
    X1i = MV.where(XX>1,0,X1i)

    X2i = MV.where(XX<0,XX,0)
    X2i = MV.where(XX<-1,0,X2i)

    X3i = MV.where(MV.absolute(XX)>1,XX,0)
    #how many months has a wet/dry spell lasted?

    nmonths = MV.where(MV.absolute(XX)>1,1,0)
    montho[i] = nmonths
    X[i]=XX
    X1[i]=X1i
    X2[i]=X2i
    X3[i]=X3i
    for i in range(Z.shape[0]): #Loop over time
        XX=Z[i]/3.

        nmonths = montho[i-1] #months so far in dry spell
        #These value will be used when calculating the probability of ending a wet/dry spell.
        #An established spell will be maintained at |PDSI|>=0.5 with a value of |Z| >=0.15
        Ud[i] = Z[i]-0.15 
        Uw[i] = Z[i] + 0.15

        this_month_wet = XX>0
        this_month_dry = XX<0
        this_month_neutral = XX==0

        #PDSI value if initiating a dry spell
        X2i = MV.where((X2[i-1]*(1-0.103)+XX)<XX,X2[i-1]*(1-0.103)+XX,XX)
        X1i =  MV.where((X1[i-1]*(1-0.103)+XX)>XX,X1[i-1]*(1-0.103)+XX,XX)
                     
       

        transitional_value = MV.where(this_month_wet,X1[i],0)
        transitional_value = MV.where(this_month_dry,X2[i],0)
        extreme_transitional_value = MV.absolute(transitional_value)>1

        
        
        last_month_neutral = X3[i-1] ==0
        last_month_dry = X3[i-1]<0
        last_month_wet = X3[i-1]>0

        
        #Moisture needed to end drought or wet spell:
        Ze_drought = -2.691*X3[i-1]-1.5
        Ze_wet_spell = -2.691*X3[i-1]+1.5
        Ze = MV.where(last_month_dry,Ze_drought,0)
        Ze = MV.where(last_month_wet,Ze_wet_spell,0)

        #Probability that last month's wet or dry spell will end
        P_end_wet = end_spell_P(Ud,Ze_wet_spell,nmonths,i,typ="wet")
        P_end_dry = end_spell_P(Uw,Ze_drought,nmonths,i,typ="dry")
        P_end_spell = MV.where(last_month_dry,P_end_dry,0)
        P_end_spell = MV.where(last_month_wet,P_end_wet,P_end_spell)
        Pe[i]=P_end_spell

        # If this was a wet month
        condition = this_month_wet
        #If last month was in a dry spell
        condition1 = np.ma.logical_and(condition,last_month_dry)
        #If the dry period is definitely ending
        dry_definitely_ending = P_end_dry >=100
        condition2 = np.ma.logical_and(condition1,dry_definitely_ending)
        #PDSI = wet transitional value
        Xi=MV.where(condition2,X1i,0)
        #  If wet transitional value indicates very wet conditions
        very_wet=X1i>1
        # set X3 to wet transitional value
        condition3=np.logical_and(condition2,very_wet)
        X3i=MV.where(condition3,X1i,0)
        
        X1i=MV.where(condition3,0,X1i)
        #Log beginning of wet spell
        nmonths = MV.where(condition3,1,0)

        #If the dry period is definitely NOT ending
        dry_definitely_NOT_ending = P_end_dry <= 0
        condition2 = np.ma.logical_and(condition1,dry_definitely__NOT_ending)
        X2i=MV.where(condition2,0,X2i)

        #If last month was in a wet spell
        condition1 = np.ma.logical_and(condition,last_month_wet)
        #If the wet period is definitely ending
        wet_definitely_ending = P_end_wet >=100
        condition2 = np.ma.logical_and(condition1,wet_definitely_ending)
        #PDSI = dry transitional value
        Xi=MV.where(condition2,X2i,Xi)

         #  If dry transitional value indicates very dry conditions
        very_dry=X2i<=-1
        # set X3 to dry transitional value
        condition3=np.logical_and(condition2,very_dry)
        X3i=MV.where(condition3,X2i,X3i)
        
        X2i=MV.where(condition3,0,X1i)
        #Log beginning of dry spell
        nmonths = MV.where(condition3,1,nmonths)

        #If the wet spell is definitely NOT ending
        wet_definitely_NOT_ending = P_end_wet <= 0
        condition2 = np.ma.logical_and(condition1,wet_definitely__NOT_ending)
        X1i=MV.where(condition2,0,X1i)

        #if last month was neutral but this month has a positive Z:
        condition1 = np.ma.logical_and(condition,last_month_neutral)
        wet_spell_establishing = X1i>0.5
        condition2=np.logical_and(condition1,wet_spell_establishing)
        #PSDI = wet transitional value
        Xi=MV.where(condition2, X1i,Xi)
        #Begin wet spell
        nmonths=MV.where(condition2,1,nmonths)
        #if wet spell is atarting off strong
        wet_strong = X1i>=1
        condition3=np.logical_and(condition2,wet_strong)
        #Set wet persistence value to wet transitional value
        X3i=MV.where(condition3,X1i,X31)
        X1i=MV.where(condition3,0,X1i)

        #If this was a dry month
        condition = this_month_dry
         #If last month was in a wet spell
        condition1 = np.ma.logical_and(condition,last_month_wet)
        #If the wet period is definitely ending
        wet_definitely_ending = P_end_wet >=100
        condition2 = np.ma.logical_and(condition1,wet_definitely_ending)
        #PDSI = dry transitional value
        Xi=MV.where(condition2,X2i,Xi)
        #  If dry transitional value indicates very dry conditions
        very_dry=X2i<=-1
        # set X3 to dry transitional value
        condition3=np.logical_and(condition2,very_dry)
        X3i=MV.where(condition3,X2i,X3i)
        
        X2i=MV.where(condition3,0,X1i)
        #Log beginning of dry spell
        nmonths = MV.where(condition3,1,nmonths)

        #If the wet spell is definitely NOT ending
        wet_definitely_NOT_ending = P_end_wet <= 0
        condition2 = np.ma.logical_and(condition1,wet_definitely__NOT_ending)
        X1i=MV.where(condition2,0,X1i)

        #If last month was in a dry spell
        condition1 = np.ma.logical_and(condition,last_month_dry)
        #If the dry period is definitely ending
        dry_definitely_ending = P_end_dry >=100
        condition2 = np.ma.logical_and(condition1,dry_definitely_ending)
        #PDSI = wet transitional value
        Xi=MV.where(condition2,X1i,0)
        #  If wet transitional value indicates very wet conditions
        very_wet=X1i>1
        # set X3 to wet transitional value
        condition3=np.logical_and(condition2,very_wet)
        X3i=MV.where(condition3,X1i,0)
        
        X1i=MV.where(condition3,0,X1i)
        #Log beginning of wet spell
        nmonths = MV.where(condition3,1,0)

        #If the dry period is definitely NOT ending
        dry_definitely_NOT_ending = P_end_dry <= 0
        condition2 = np.ma.logical_and(condition1,dry_definitely__NOT_ending)
        X2i=MV.where(condition2,0,X2i)

         #if last month was neutral but this month has a negative Z:
        condition1 = np.ma.logical_and(condition,last_month_neutral)
        dry_spell_establishing = X2i<=0.5
        condition2=np.logical_and(condition1,dry_spell_establishing)
        #PSDI = dry transitional value
        Xi=MV.where(condition2, X2i,Xi)
        #Begin dry spell
        nmonths=MV.where(condition2,1,nmonths)
        #if dry spell is atarting off strong
        dry_strong = X2i<=-1
        condition3=np.logical_and(condition2,dry_strong)
        #Set dry persistence value to dry transitional value
        X3i=MV.where(condition3,X2i,X31)
        X2i=MV.where(condition3,0,X2i)

        X1i=MV.where(X1i<0,0,X1i)
        X2i = MV.where(X2i>0,0,X2i)

        #if the probability the spell has ended is not 100% and it wasn't a neutral month
        condition = P_end_spell<100
        X3i = MV.where(condition,X3[i-1]*.897+XX,X3i)
        #If the sign of X3 didn't change over the time step
        sign_same = np.ma.sign(X3[i-i]) == np.ma.sign(X3i)
        condition1=np.ma.logical_and(condition,sign_same)
        #If X3 is still extreme
        extreme = MV.absolute(X3i)>0.5
        condition2=np.ma.logical_and(condition1,extreme)
        #The spell continues
        nmonths=MV.where(condition2,nmonths+1,nmonths)
        #Otherwise, the spell ends
        condition2=np.ma.logical_and(condition1,~extreme)
        nmonths=MV.where(condition2,0,nmonths)
        #If the sign did change
        sign_different = ~sign_same
        condition1=np.ma.logical_and(condition,sign_different)
        #IF the X3 anomaly is large
        very_extreme = MV.absolute(X3i)>=1
        condition2=np.ma.logical_and(condition1,very_extreme)
        #new dry or wet spell begins
        nmonths=MV.where(condition2,1,nmonths)
        #otherwise neutral conditions continue
        condition2=np.ma.logical_and(condition1,~very_extreme)
        nmonths=MV.where(condition2,0,nmonths)

        montho[i]=nmonths
        #Decide what PDSI is
        #If there is no event established or in progress
        no_event = nmonths ==0
        tending_toward_wet = X1i>-X2i
        wet_condition = np.logical_and(no_event,tending_toward_wet)
        Xi=MV.where(wet_condition,X1i,Xi)
        tending_toward_dry = X1i<=-X2i
        dry_condition = np.logical_and(no_event,tending_toward_dry)
        Xi=MV.where(dry_condition,X2i,Xi)

        #If a wet or dry spell has begun
        event_begun = nmonths == 1
        began_last_month = X3i==0
        condition = np.logical_and(event_begun,began_last_month)
        condition1= np.logical_and(condition,tending_toward_wet)
        Xi = MV.where(condition1,X1i,Xi)
        condition1= np.logical_and(condition,tending_toward_dry)
        Xi = MV.where(condition1,X2i,Xi)
        began_this_month = X3i !=0
        condition = np.logical_and(event_begun,began_this_month)
        pdsi_not_assigned = Xi==0
        Xi=MV.where(np.logical_and(pdsi_not_assigned,condition),X3i,Xi)
        #If beyond the first month of a wet or dry spell
        Xi=MV.where(nmonths>1,X3i,Xi)
    
        #BACKTRACK
        ending_or_confirmed = np.logical_or(P_end_spell>=100,P_end_spell<=0)
        condition = np.logical_and(montho[i]==1,ending_or_confirmed)
        s_orig = i
        
        
        
        
        
        


        
        

        
        
        
        
                           
        
        
        

        
        
        

        
        
        
    
