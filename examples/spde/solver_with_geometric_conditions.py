import os
import numpy as np


"""
We define a class ``SteadyStateHeat2DSolver'' which
solves the steady state heat equation
in a two-dimensional square grid.

For now we don't add any forcing function and the
boundary conditions are Dirichlet.

"""


__all__ = ['SteadyStateHeat2DSolver']

import fipy
import numpy as np
from pdb import set_trace as keyboard
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 


class SteadyStateHeat2DSolver(object):
    
    """
    Solves the 2D steady state heat equation with dirichlet boundary conditions.
    It uses the stochastic model we developed above to define the random conductivity.
    _
    Arguments:
    nx           -    Number of grid points along x direction.
    ny           -    Number of grid points along y direction.
    value_left   -    The value at the left side of the boundary.
    value_right  -    The value at the right side of the boundary.
    value_top    -    The value at the top of the boundary.
    value_bottom -    The value at the bottom of the boundary.
    """

    def __init__(self, nx=100, ny=100, value_left=1.,
        value_right=0., value_top=0., value_bottom=0.,
        q=None, division_axis="vertical"):
        """
        ::param nx:: Number of cells in the x direction.
        ::param ny:: Number of cells in the y direction.
        ::param value_left:: Boundary condition on the left face.
        ::param value_right:: Boundary condition on the right face.
        ::param value_top:: Boundary condition on the top face.
        ::param value_bottom:: Boundary condition on the bottom face.
        ::param q:: Source function. 
        ::param division_axis:: Direction of the division ("vertical" or "horizontal").
        """
        # Set domain dimensions
        self.nx = nx
        self.ny = ny
        self.dx = 1. / nx
        self.dy = 1. / ny
        
        # Define mesh
        self.mesh = fipy.Grid2D(nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy)
        
        # Define cell variable phi and apply the division
        self.phi = fipy.CellVariable(name='$T(x)$', mesh=self.mesh, value=0.)
        x, y = self.mesh.cellCenters
        if division_axis == "vertical":
            self.phi.setValue(value_left, where=x < 0.5)  # Left side
            self.phi.setValue(value_right, where=x >= 0.5)  # Right side
        elif division_axis == "horizontal":
            self.phi.setValue(value_top, where=y >= 0.5)  # Top side
            self.phi.setValue(value_bottom, where=y < 0.5)  # Bottom side
        
        # Random conductivity and source
        self.C = fipy.CellVariable(name='$C(x)$', mesh=self.mesh, value=1.)
        self.source = fipy.CellVariable(name='$f(x)$', mesh=self.mesh, value=0.)
        
        # Apply boundary conditions (Dirichlet)
        self.phi.constrain(value_left, self.mesh.facesLeft)
        self.phi.constrain(value_right, self.mesh.facesRight)
        
        # Homogeneous Neumann
        self.phi.faceGrad.constrain(value_top, self.mesh.facesTop)
        self.phi.faceGrad.constrain(value_bottom, self.mesh.facesBottom)
        
        # Setup the diffusion problem
        self.eq = -fipy.DiffusionTerm(coeff=self.C) == self.source

    
    def set_source(self, source):
        """
        Initialize the source field.
        """
        self.source.setValue(source)
        
    def set_coeff(self, C):
        """
        Initialize the random conductivity field.
        """
        self.C.setValue(C)
        
    def solve(self):
        self.eq.solve(var=self.phi)

    def ObjectiveFunction(self):
        """
        We look at the temperature in the middle of the domain.
        """
        return self.phi.value[self.loc]

    def NeumannSpatialAverage(self):
        """
        Spatial average of the independent variable on the right side 
        Neumann boundary. 
        """
        loc = np.where(np.int32(self.mesh.facesRight.value) == 1)[0]
        val = self.phi.faceValue.value[loc]
        return np.mean(val)

    
    def RandomField(self):
        facecenters=np.array(self.mesh.faceCenters).T
        xf=facecenters[:, 0]
        yf=facecenters[:, 1]
        zf=self.C.value
        xif=yif=np.linspace(0.01, 0.99, 32)
        zif=griddata((xf, yf), zf, (xif[None,:], yif[:,None]), method='cubic')
        return zif


