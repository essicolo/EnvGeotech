r"""
variational formulation described at https://www.authorea.com/users/23640/articles/61529
Inspired from http://sfepy.org/doc-devel/examples/diffusion/poisson_field_dependent_material.html

./simple.py Richards-Buckingham_SfePy.py --save-regions-as-groups
./postproc.py cube_big_tetra_regions.vtk # works with mayavi <= 4.2
"""

from sfepy import data_dir
from sfepy.base.base import output
import numpy as np

filename_mesh = data_dir + '/meshes/3d/cube_big_tetra.mesh'

# Constitutive relationships

## Van Genuchten water retention curve (wrc) and hydraulic conductivity function (hcf)
## Expressed in terms of psi, where psi is suction (psi = -h)
class VanGenuchten(object):
    """
    Returns the water retention curve and hydraulic conductivity function
    according to van Genuchten (1980)'s functions
    """
    def __init__(self, thR, thS, aVG, nVG, mVG, ksat, psi, lVG = 0.5):
        self.aVG = aVG
        self.nVG = nVG
        self.mVG = mVG
        self.lVG = lVG
        self.ksat = ksat
        self.thR = thR
        self.thS = thS
        self.psi = psi

    def wrc(self):
        if np.all(self.psi <= 0):
            self.th = np.repeat(self.thS, len(self.psi))
        elif np.all(self.psi > 0):
            self.th = self.thR + (self.thS - self.thR) * (1+(self.aVG * self.psi) ** self.nVG) ** (-self.mVG)
        else:
            self.th = np.zeros(len(self.psi))
            self.th[self.psi <= 0] = self.thS
            self.th[self.psi > 0] = self.thR + (self.thS - self.thR) * (1+(self.aVG * self.psi[self.psi > 0]) ** self.nVG) ** (-self.mVG)
        return([self.psi, self.th])

    def hcf(self):
        if np.all(self.psi <= 0):
            self.k = np.repeat(self.ksat, len(self.psi))
        elif np.all(self.psi > 0):
            self.k = self.ksat*((1-((self.aVG*self.psi)**(self.nVG*self.mVG))* \
            ((1+((self.aVG*self.psi)**self.nVG))**(-self.mVG)))**2) / \
            ((1+((self.aVG*self.psi)**self.nVG))**(self.mVG*self.lVG))
        else:
            self.k = np.zeros(len(self.psi))
            self.k[self.psi <= 0] = self.ksat
            self.k[self.psi > 0] = self.ksat*((1-((self.aVG*self.psi[self.psi > 0])**(self.nVG*self.mVG))*      ((1+((self.aVG*self.psi[self.psi > 0])**self.nVG))**(-self.mVG)))**2) / \
            ((1+((self.aVG*self.psi[self.psi > 0])**self.nVG))**(self.mVG*self.lVG))
        return([self.psi, self.k])

## Brooks and Corey water retention curve (wrc) and hydraulic conductivity function (hcf)
## Expressed in terms of psi, where psi is suction (psi = -h)
class BrooksCorey(object):
    """
    Returns the water retention curve and hydraulic conductivity function
    according to Brooks and Corey (1964)'s functions
    """
    def __init__(self, thR, thS, aev, lBC, ksat, psi):
        self.aev = aev
        self.lBC = lBC
        self.ksat = ksat
        self.thR = thR
        self.thS = thS
        self.psi = psi

    def wrc(self):
        if np.all(self.psi <= 0):
            self.th = np.repeat(self.thS, len(self.psi))
        else:
            self.th = np.zeros(len(self.psi))
            self.th[self.psi <= self.aev] = self.thS
            self.th[self.psi > self.aev] = self.thR + (self.thS - self.thR) * (self.psi[self.psi > self.aev] / self.aev) ** (-self.lBC)
        return([self.psi, self.th])

    def hcf(self):
        if np.all(self.psi <= 0):
            self.k = np.repeat(self.ksat, len(self.psi))
        else:
            self.k = np.zeros(len(self.psi))
            self.k[self.psi <= self.aev] = self.ksat
            self.k[self.psi > self.aev] = self.ksat * (self.psi[self.psi > self.aev] / self.aev) ** (-2 - 3*self.lBC)
        return([self.psi, self.k])

# Material parameters
## the initial residual is almost zero - this might mean that you need some
## scaling of parameters, so that the entries in the rhs (and the matrix) are not
## too small. - Robert Cimrman, https://groups.google.com/forum/#!topic/sfepy-devel/dbEy3I3jSOg
scaling = 1

## typical uniform silt
silt_thR = 0.034
silt_thS = 0.460
silt_ksat = 7E-7 * scaling

### van Genuchten parameters
silt_aVG = 1.6 / scaling
silt_nVG = 1.37
silt_mVG = 1 - (1 / silt_nVG)
silt_lVG = 0.5

### Brooks and Corey parameters
silt_aev = 0.35 * scaling
silt_lBC = 0.35

def get_conductivity(ts, coors, problem, equations = None, mode = None, **kwargs):
    """
    Calculates the conductivity with VanGenuchten returns it.
    This relation results in larger h gradients where h is small.
    """
    if mode == 'qp':
        # note: verifier ce qui sort de la forumaltion originale
        # et comparer a ici
        # http://sfepy.org/doc-devel/examples/diffusion/poisson_field_dependent_material.html
        # reprendre une formulation plus simple (Brooks and Corey?)
        h_values = problem.evaluate('ev_volume_integrate.i.Omega(h)',
                                    mode = 'qp', verbose = False)
        #hcf = VanGenuchten(thR = silt_thR, thS = silt_thS, aVG = silt_aVG, nVG = silt_nVG, mVG = 1-1/silt_nVG,
        #               lVG = silt_lVG, ksat = silt_ksat, psi = -h_values).hcf()
        #hcf = BrooksCorey(thR = silt_thR, thS = silt_thS, aev = silt_aev, lBC = silt_lBC, ksat = silt_ksat, psi = -h_values).hcf()
        val = 7E-7 / (-(h_values) + 0.01) # hcf[1]
        output('h_values: min:', h_values.min(), 'max:', h_values.max())
        output('conductivity: min:', val.min(), 'max:', val.max())

        val.shape = (val.shape[0], 1, 1)
        # print("***print val***: ", val) # the outpu must be of shape [[[ 700.]] [[700.]] ]]]
        return {'val' : val}


materials = {
    'coef' : 'get_conductivity',
    'flux' : ({'val' : 1E-7 * scaling},),
}

fields = {
    'pressure' : ('real', 1, 'Omega', 1),
}

variables = {
    'h' : ('unknown field', 'pressure', 0),
    'v' : ('test field',    'pressure', 'h'),
}

regions = {
    'Omega' : 'all',
    'Gamma_Bottom' : ('vertices in (z < -0.49999)', 'facet'),
    'Gamma_Top' : ('vertices in (z > 0.49999)', 'facet'),
}

ebcs = {
    'P1' : ('Gamma_Bottom', {'h.0' : 0}),
}

functions = {
    'get_conductivity' : (get_conductivity,),
}

integrals = {
    'i' : 1,
}

equations = {
    'Pressure' : """dw_laplace.i.Omega(coef.val, v, h) - dw_surface_integrate.2.Gamma_Top(flux.val, v) = 0"""
}

solvers = {
    'ls' : ('ls.scipy_direct', {}),
    'newton' : ('nls.newton', {
        'i_max' : 1,
        'eps_a' : 1e-10,
        'eps_r' : 1.0,
    }),
}

options = {
    'nls' : 'newton',
    'ls' : 'ls',
}
