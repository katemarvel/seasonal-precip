#This is a modification of Ben's code.  It will apply Park's PET/PDSI code to generate monthly zindex and scPDSI calculations from model output on LLNL's crunchy
#Import necessary modules
import cdms2 as cdms
import MV2 as MV
import numpy as np
import cdtime,cdutil,genutil
import calendar
#import the PET code that I wrote
import calc_Z as z


def convert_to_mm(X):
    if X.units == "mm":                                                 
        return X
    if X.units == 'kg m-2 s-1':    
        X=X*60*60*24 #convert to mm/day                             
    days_in_month=np.array([calendar.monthrange(x.year,x.month)[1] for x in X.getTime().asComponentTime()])      
    if len(X.shape)==3:                                             
        Xd = cmip5.cdms_clone(X*days_in_month[:,np.newaxis,np.newaxis],X)                                      
    elif len(X.shape)==4:               
        Xd = cmip5.cdms_clone(X*days_in_month[np.newaxis,:,np.newaxis,np.newaxis],X)                                
    Xd.units = "mm"                                        
    return Xd    

#Set up calibration period.  Use same period as Park
start_year = cdtime.comptime(1921,1,1)
stop_year = cdtime.comptime(2000,12,31)
yrcalib = (start_year,stop_year)


def calculate_all_Z(experiment,start_year,stop_year):
    maxfiles = sorted(glob.glob("/kate/PET/"+experiment+"/*tasmax*"))
    minfiles = sorted(glob.glob("/kate/PET/"+experiment+"/*tasmin*"))

    # get PET: average of PET calculated from min and max temperatures
    maxfiles_stub = [fil.split("tasmax")[0] for fil in maxfiles]
    minfiles_stub  = [fil.split("tasmin")[0] for fil in minfiles]


    for i in range(len(maxfiles)):
        try:
            imin=minfiles_stub.index(maxfiles_stub[i])
        except:
            continue 

        petmax_file = cdms.open(maxfiles[i])
        petmin_file = cdms.open(minfiles[imin])
        petmax = petmax_file("PET")
        petmin = petmin_file("PET")
        petraw = 0.5*(petmax+petmin)
        petmax_file.close()
        petmin_file.close()

        #get the corresponding precip data
        model=maxfiles_stub[i].split(".")[1]
        rip=maxfiles_stub[i].split(".")[3]
        prpath = "/work/cmip5/"+experiment+"/atm/mo/pr/"
        pr_file = cdms.open(cmip5.get_latest_version(glob.glob(prpath+"*"+model+"*"+rip+"*")))
        prraw=pr_file("pr")
        pr_file.close()

        #convert to mm
        PET = convert_to_mm(petraw)
        P = convert_to_mm(prraw)

        #define soil moisture holding capacity
        WCTOP = MV.ones(PET.shape[1:])*25.4 # top layer: 1 inch (converted to mm)
        WCBOT = MV.ones(PET.shape[1:])*25.4*5 #underlying layers: 5 inches

        Z = z.calculate_Z(PET,P,WCTOP,WCBOT,start_year,stop_year)
        cmd = "mkdir "+"/kate/Zindex/"+experiment+"/"
        zwritename = "/kate/Zindex/"+maxfiles_stub[i].split("/")[-1].split("PET")[0]+"z.nc"
        fzw=cdms.open(zwritename,"w")
        fzw.write(Z)
        fzw.close()

calculate_all_Z(historical,start_year,stop_year)

    

