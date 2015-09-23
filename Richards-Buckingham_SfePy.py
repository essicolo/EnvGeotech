r"""
variational formulation described at https://www.authorea.com/users/23640/articles/61529
Inspired from http://sfepy.org/doc-devel/examples/diffusion/poisson_field_dependent_material.html

cd ~bin/sfepy
./simple.py  SEP_pdf/EnvGeotech/Richards-Buckingham_SfePy.py --save-regions-as-groups
./postproc.py cube_big_tetra_regions.vtk # works with mayavi <= 4.2
"""

from sfepy import data_dir
from sfepy.base.base import output
import numpy as np

filename_mesh = data_dir + '/meshes/3d/cube_big_tetra.mesh'

# Constitutive relationships expressed in terms of psi, where psi is suction (psi = -h)
def vanGenuchten(ksat, aVG, nVG, mVG, lVG, psi):
    k = np.piecewise(psi, [psi < 0, psi >= 0],
    [ksat, ksat*((1-((aVG*psi[psi >= 0])**(nVG*mVG))*((1+((aVG*psi[psi >= 0])**nVG))**(-mVG)))**2) / ((1+((aVG*psi[psi >= 0])**nVG))**(mVG*lVG))])
    return(k)

def brooksCorey(aev, lBC, ksat, psi):
    k = np.piecewise(psi, [psi < aev, psi >= aev],
    [ksat, ksat * (psi[psi >= aev] / aev) ** (-2 - 3 * lBC)])
    return(k)

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
    Calculates the conductivity with a constitutive k(psi) relation,
    where psi = -h.
    """
    if mode == 'qp':

        ## Get pressure values
        h_values = problem.evaluate('ev_volume_integrate.i.Omega(h)',
                                    mode = 'qp', verbose = False) * scaling
        psi_values = -h_values

        # van Genuchten
        val = vanGenuchten(ksat = silt_ksat, aVG = silt_aVG, nVG = silt_nVG,
                           mVG = silt_mVG, lVG = silt_lVG, psi = psi_values)

        # Brooks and Corey
        #val = brooksCorey(ksat = silt_ksat, aev = silt_aev, lBC = silt_lBC,
        #                  psi = psi_values)

        # Reshape the val vector to match SfePy expectations
        val.shape = (val.shape[0] * val.shape[1], 1, 1)

        # Check output
        output('h_values: min:', h_values.min(), 'max:', h_values.max())
        output('conductivity: min:', val.min(), 'max:', val.max())

        return {'val' : val}


materials = {
    'coef' : 'get_conductivity',
    'flux' : ({'val' : 3E-8 * scaling},),
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
    'i' : 2, # not sure of value to use here
}

equations = {
    'Pressure' : """dw_laplace.i.Omega(coef.val, v, h) - dw_surface_integrate.i.Gamma_Top(flux.val, v) = 0"""
}

solvers = {
     'ls' : ('ls.scipy_direct', {}),
     'newton' : ('nls.newton', {
         'i_max' : 1,
         'eps_a' : -1e-10,
         'eps_r' : -1,
     }),
     'ts' : ('ts.simple', {
         't0' : 0.0,
         't1' : 1.0,
         'dt' : None,
         'n_step' : 5,
         'quasistatic' : True,
     }),
}

options = {
     'ts' : 'ts',
     'nls' : 'newton',
     'ls' : 'ls',
}

options = {
    'nls' : 'newton',
    'ls' : 'ls',
}
