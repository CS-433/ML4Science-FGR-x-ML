""" Useful functions used to get Mass_HI predictions from the models currently used to approximate the M_HI vs Mass_Halo relation"""
import numpy as np
#import astropy.units as u

def MHI_Modi2019(Mh, z, model='A'):
    # from Modi+ 2019
    if(model == 'A'):
        alpha = (1+2*z)/(2+2*z)
        Mcut = (3e9*(1+10*(3/(1+z))**8)) # in Msun/h units
        Ah = 8e5*(1+(3.5/z)**6)*(1+z)**3 
    elif(model == 'C'):
        alpha = 0.9
        Mcut = 1.e10 # in Msun/h units
        Ah = 3.e6*(1.+1./z)*(1+z)**3
    else:
        ValueError(' Model B is not implemented')
    M_HI = Ah * np.power(Mh/Mcut, alpha) * np.exp(-Mcut/Mh)
    return M_HI


def MHI_Padmanabhan2017(Mh, z, Om0, delta_c=200.):
    ''' Mh must be in Msun units (without small h)'''
    fHc = (1-0.2486) * 0.0486 / Om0 

    # Barnes+ (2014), Eq 3 : https://arxiv.org/abs/1403.1873v3
    vM_c0 = 96.6 * np.power(delta_c*Om0 / 24.4, 1./6) * np.sqrt((1+z)/3.3) * np.power(Mh/1e11, 1./3) # * u.km / u.s

    #from Padmanabhan+ (2017), Eq. 1 and Table 3 : https://arxiv.org/abs/1611.06235
    alpha, cHI, v_c0, beta, gamma = 0.9, 28.65, np.power(10, 1.56), -0.58, 1.45
    M_HI = alpha * fHc * Mh * np.power(Mh/1e11, beta) * np.exp(-1*np.power(v_c0 / vM_c0, 3))
    return M_HI.astype(np.float32)