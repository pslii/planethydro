import numpy as np

from .utility import azAverage, colMul, centralDiff2D


__author__ = 'pslii'


class reduceData:
    """
    Usage:
    processor = reduceData(grid, params, data, metadata)
    rho_avg, rho_pertb = processor.rhoPertb()
    """

    def __init__(self, grid, params, data):
        self.grid = grid
        self.data = data
        self.params = params

    def sigma_pertb(self):
        sigma_avg = (self.data.rho * self.grid.dphi).sum(axis=1) / (2 * np.pi)        
        return self.data.rho - sigma_avg[:, np.newaxis]

    def calculate_velocity(self):
        phi_pl, omega_pl = self.data.phiPlanet, self.data.omegaPlanet
        vr, vphi = self.data.u, self.data.v - omega_pl * self.grid.r[:, np.newaxis]

        phi_grid = self.grid.phi - phi_pl
        vx = vr * np.cos(phi_grid) - vphi * np.sin(phi_grid)
        vy = vr * np.sin(phi_grid) + vphi * np.cos(phi_grid)
        return vx, vy

        

