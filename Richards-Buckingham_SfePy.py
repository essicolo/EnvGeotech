r"""
variational formulation described at https://www.authorea.com/users/23640/articles/61529
Inspired from http://sfepy.org/doc-devel/examples/diffusion/poisson_field_dependent_material.html
"""

from sfepy import data_dir
from sfepy.base.base import output
import numpy as np

filename_mesh = data_dir + '/meshes/3d/cube_big_tetra.mesh'

# Constitutive relationships
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
        if np.all(self.psi > 0):
            self.th = self.thS
        elif np.all(self.psi <= 0):
            self.th = self.thR + (self.thS - self.thR) * (1+(self.aVG * self.psi) ** self.nVG) ** (-self.mVG)
        else:
            self.th = np.zeros(len(self.psi))
            self.th[self.psi <= 0] = self.thS
            self.th[self.psi > 0] = self.thR + (self.thS - self.thR) * (1+(self.aVG * self.psi[self.psi > 0]) ** self.nVG) ** (-self.mVG)
        return([self.psi, self.th])

    def hcf(self):
        if np.all(self.psi > 0):
            self.k = self.ksat
        elif np.all(self.psi <= 0):
            self.k = self.ksat*((1-((self.aVG*self.psi)**(self.nVG*self.mVG))* \
            ((1+((self.aVG*self.psi)**self.nVG))**(-self.mVG)))**2) / \
            ((1+((self.aVG*self.psi)**self.nVG))**(self.mVG*self.lVG))
        else:
            self.k = np.zeros(len(self.psi))
            self.k[self.psi > 0] = self.ksat
            self.k[self.psi <= 0] = self.ksat*((1-((self.aVG*self.psi[self.psi <= 0])**(self.nVG*self.mVG))*      ((1+((self.aVG*self.psi[self.psi <= 0])**self.nVG))**(-self.mVG)))**2) / \
            ((1+((self.aVG*self.psi[self.psi <= 0])**self.nVG))**(self.mVG*self.lVG))
        return([self.psi, self.k])

# Material parameters
cbl_thR = 0.045
cbl_thS = 0.430
cbl_aVG = 14.5
cbl_nVG = 2.68
cbl_mVG = 1-1/2.68
cbl_lVG = 0.5
cbl_ksat = 8.25e-5

def get_conductivity(ts, coors, problem, equations = None, mode = None, **kwargs):
    """
    Calculates the conductivity with VanGenuchten returns it.
    This relation results in larger h gradients where h is small.
    """
    if mode == 'qp':
        h_values = problem.evaluate('ev_volume_integrate.i.Omega(h)',
                                    mode = 'qp', verbose = False)
        val = VanGenuchten(thR = cbl_thR, thS = cbl_thS, aVG = cbl_aVG, nVG = cbl_nVG, mVG = 1-1/cbl_nVG,
                       lVG = cbl_lVG, ksat = cbl_ksat, psi = -h_values).hcf()[0]
        val.shape = (val.shape[0] * val.shape[1], 1, 1)
        return {'val' : val}


materials = {
    'coef' : 'get_conductivity',
    'flux' : ({'val' : 1E-7},),
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
    'Gamma_Bottom' : ('vertices in (z < 0.00001)', 'facet'),
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
