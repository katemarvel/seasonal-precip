
def penmont_vpd_SH(Ta,Rnet,q,press,u):
    """
    
    Using the modified FAO Penman-Monteith approach, calculation reference
    (potential) evapotranspiration. As part of this, also calculate the vapor
    pressure deficit (VPD) and relative humidity RH.
    
    Inputs:
    Ta      = temperature, degrees C
    Rnet    = surface net radiation, W/m2
    u       = wind speed at 2 meters, m/s
    q       = specific humidity, kg/kg
    press   = surface pressure, Pascals

    Outputs:
    PET  = potential evapotranspiration (mm/day)       
    VPD  = vapor presssure deficit (kPa)  
    RH   = relative humidity, fraction

    Written by Benjamin I. Cook and translated to Python by Kate Marvel
 
    Based on:
    Xu, C-Y., and V. P. Singh. "Cross comparison of empirical equations 
               for calculating potential evapotranspiration with data 
               from Switzerland." Water Resources Management,
               16.3 (2002): 197-219.

           FAO Document:
           http://www.fao.org/docrep/X0490E/x0490e00.htmContents

    For Tetens, both above and below zero:
       http://cires.colorado.edu/~voemel/vp.html
    """
    # ground heat flux, set to zero (works fine on >monthly timescales, and is 
    # accurate if one calculates Rnet as SH+LH)
    gflux=0.

    #Extrapolate wind speed from 10m to 2m (ADDED BY KATE)
    u2 = u*(4.87/np.log(67.8*10.-5.42))
    

    # Calculate the latent heat of vaporization (MJ kg-1)
    lambda_lv=2.501-(2.361e-3)*Ta

    # Calculate Saturation vapor pressure (kPa)
    es=0.611*exp((17.27*Ta)/(Ta+237.3))  

    # Convert specific humidity (kg/kg) to actual vapor pressure
    ea=(press*q)/0.6213 # Pascals

    # Convert ea to kilopascals
    ea = ea/1000.

    # Convert Pressure to kPa
    press=press/1000.

    # Use es and relative humidity to calculate ea in kPa
    #ea=es.*RH

    # Slope of the vapor pressure curve (kPa C-1)
    delta_vpc=(4098*es)/((Ta+237.3)^2)

    # Psychometric constant (kPa C-1)
    # 0.00163 is Cp (specific heat of moist air) divided by epsilon (0.622, 
    # ratio molecular weight of water vapour/dry air)
    psych_const=0.00163*(press/lambda_lv)

    # Net radiation
    Rnet=Rnet/11.6  # convert W/m2 to MJ/m2/d

    # Calculate VPD
    VPD=es-ea

    # Calculate Relative Humidity
    RH = ea/es

    # Potential Evapotranspiration (mm/day)
    PET=(0.408*delta_vpc*(Rnet-gflux)+psych_const*(900./(Ta+273))*u2*(VPD))/(delta_vpc+psych_const*(1+0.34*u))

    return PET, VPD, RH

def PET_from_cmip(fname,temp_variable = "tas"):

    #Ta      = temperature, degrees C = tas
    #   Rnet    = surface net radiation, W/m2 = (hfls+hfss)
    #  u       = wind speed at 2 meters, m/s = sfcWind (??)
    #   q       = specific humidity, kg/kg = huss
    #  press   = surface pressure, Pascals = ps

    f_wind = cdms.open(fname)
    u = f_wind("sfcWind")
    f_wind.close()
    
    f_hfls =cdms.open(cmip5.get_corresponding_file(fname,"hfls"))
    hfls = f_hfls("hfls")
    f_hfls.close()

    f_hfss =cdms.open(cmip5.get_corresponding_file(fname,"hfss"))
    hfss = f_hfss("hfss")
    f_hfss.close()

    R_net = hfss + hfls

    f_ta =cdms.open(cmip5.get_corresponding_file(fname,temp_variable))
    Ta = f_ta(temp_variable)
    f_ta.close()

    f_huss =cdms.open(cmip5.get_corresponding_file(fname,"huss"))
    q = f_huss("huss")
    f_huss.close()

    f_ps =cdms.open(cmip5.get_corresponding_file(fname,"ps"))
    press = f_ps("ps")
    f_ps.close()

    PET,VPD,RH = penmont_vpd_SH(Ta,Rnet,q,press,u)
    return PET,VPD,RH

    
    
    
             











