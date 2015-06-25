import numpy as np

from . import utility


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

    def omega(self):
        return self.data.v / self.grid.r[:, np.newaxis]

    def oortB(self):
        """
        B = \Omega + (r/2)(d\Omega/dr)
        """
        GM = self.params.get('gm')
        r = self.grid.r
        r_in = (self.grid.r_edge[1] + self.grid.r_edge[2])/2
        r_out = (self.grid.r_edge[-2] + self.grid.r_edge[-3])/2

        omega = self.omega()
        domegadr = utility.d_dr(self.grid, omega, 
                                         arr_start = np.sqrt(GM/r_in**3),
                                         arr_end = np.sqrt(GM/r_out**3))
        return omega + (r[:,np.newaxis]/2.0) * domegadr

    def vortensity_gradient(self):
        """
        d(\Sigma/B)/dr
        """
        sigma = self.data.rho
        B = self.oortB()
        dsigmadr = utility.d_dr(self.grid, sigma)
        dBdr = utility.d_dr(self.grid, B)
        r = self.grid.r[:, np.newaxis]
        return  r * (dsigmadr - (sigma/B) * dBdr)

    def vorticity(self):
        """
        \omega = curl(v)
        """
        omega_pl = self.data.omegaPlanet
        u, v = self.data.u, self.data.v - omega_pl * self.grid.r[:, np.newaxis]
        return utility.curl2D(self.grid, u, v)

    def vortensity(self):
        """
        \omega / \Sigma
        """
        return self.vorticity()/self.data.rho

    def vortensity_source(self):
        sigma = self.data.rho
        dsigdr, dsigdphi = utility.grad2D(self.grid, sigma)
        dpidr, dpidphi = utility.grad2D(self.grid, self.data.p)
        return (dsigdr*dpidphi - dpidr*dsigdphi) / sigma**3


