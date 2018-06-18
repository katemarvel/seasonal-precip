#This is a modification of Ben's code.  It will apply Park's PET/PDSI code to generate monthly zindex and scPDSI calculations from model output on LLNL's crunchy
#Import necessary modules
import cdms2 as cdms
import MV2 as MV
import numpy as np
import cdtime,cdutil,genutil

#Set up calibration period.  Use same period as Park
start_year = cdtime.comptime(1921,1,1)
stop_year = cdtime.comptime(2000,12,31)
yrcalib = (start_year,stop_year)

#Net radiation

#wind

#prep data



