import numpy as np
from astropy import units as un
from astropy.constants import G

"""
Within the FIRE dataset, the following parameters in the halo catalog are needed

 mass.200c     =  M200 (virial mass) in solar masses
 scale.radius  =  rs (scale radius) in kpc
 radius        =  R200 (virial radius) in kpc
"""

def rho_nfw(r, rs, M200, R200):
    """
    NFW dark matter density profile

    Parameters
    ----------
    r : float or Quantity
        Radius
    rs : float or Quantity
        NFW scale radius
    M200: float or Quantity
        Virial mass
    R200: float or Quantity
        Virial radius

    Returns
    -------
    float or Quantity
        Density at radius r
    """
    conc = R200 / rs
    factor = np.log(1 + conc) - conc/(1 + conc)
    rho0 = M200 / (4 * np.pi * rs**3) / factor

    y = r / rs
    return rho0 / (y * (y + 1)**2)

def mass_nfw(r, rs, M200, R200):
    """
    Mass contained by radius r

    Parameters
    ----------
    r : float or Quantity
        Radius
    rs : float or Quantity
        NFW scale radius
    M200: float or Quantity
        Virial mass
    R200: float or Quantity
        Virial radius

    Returns
    -------
    float or Quantity
        mass contained within radius r
    """
    conc = R200 / rs
    factor = np.log(1 + conc) - conc/(1 + conc)
    rho0 = M200 / (4 * np.pi * rs**3) / factor

    y = r / rs

    return 4 * np.pi * rho0 * rs**3 * (np.log(1 + y) - y/(1 + y))


def phi_nfw(r, rs, M200, R200):
    '''
    potential from nfw proile
    '''
    conc = R200 / rs
    factor = np.log(1 + conc) - conc/(1 + conc)
    rho0 = M200 / (4 * np.pi * rs**3) / factor

    y = r / rs
    
    return -4 *G* np.pi * rho0 * rs**3 * np.log(1 + y)/r

## M200 = 5e12 * un.Msun
## R200 = 360 * un.kpc
## rs = 51 * un.kpc        # conc = 6.95
## 
## rvals = np.linspace(0.0001*rs, 3*R200, 500)
## masses = mass_nfw(rvals, rs, M200, R200).to_value("Msun")
## rhos = rho_nfw(rvals, rs, M200, R200).to_value("Msun/Mpc3")
## 
## rvals = rvals.to_value("Mpc")
## dr = rvals[1] - rvals[0]
## mass_test = np.cumsum(4 * np.pi * rvals**2 * rhos * dr)
## 
## import pylab as pl
## pl.plot(rvals, masses, label='true')
## pl.plot(rvals, mass_test, label='test')
## pl.legend()
## pl.show()
## 
## import IPython; IPython.embed()