import Biome_bin as b
#import sc_utilities as sc


import glob
import sys,os
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


b.write_all_Koeppen()

def average_variable_over_biome(forcing,variable,realm="atm"):
    fnames = cmip5.get_datafiles(forcing,variable,realm=realm)
    writedirec = "/kate/biomes/"+forcing+"/"+variable+"/"
    cmd = "mkdir -p writedirec"
    os.system(cmd)
    for fname in fnames:
        try:
            model = fname.split(".")[1]
            K = b.get_koeppen_classification(model)
            f = cdms.open(fnames)
            X = f(variable)
            
            AV = average_over_all_biomes(K,X)
            AV.id = variable
    
            writefname = fname.split("/")[-1].replace("xml","nc").replace(variable,"Koeppen")
            fw = cdms.open(writedirec+writefname,"w")
            fw.write(AV)
            fw.close()
    
            f.close()
        except:
            print "BAD: "+fname


for forcing in ["historical","rcp85","1pctCO2","piControl"]:
    for variable in ["evspsbl","pr"]:
        average_variable_over_biome(forcing,variable)

        
        
