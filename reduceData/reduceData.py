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
        """

        :param grid:
        :param params:
        :param data:
        :type grid: gridReader
        :type params: dict
        :type data: SimData
        :return:
        """
        self.grid = grid
        self.data = data
        self.params = params

    def sigma_pertb(self):
        sigma_avg = (self.data.rho * self.grid.dphi).sum(axis=1) / (2 * np.pi)        
        return self.data.rho - sigma_avg[:, np.newaxis]

    def calculate_velocity(self):
        """
        Calculate velocities in frame of planet.
        :return:
        """
        phi_pl = self.data.phiPlanet
        (vr_p, vphi_p, _) = self.data.cyl_planet_velocity
        vr, vphi = self.data.u - vr_p, self.data.v - vphi_p

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
        (vr_p, vphi_p, _) = self.data.cyl_planet_velocity
        vr, vphi = self.data.u - vr_p, self.data.v - vphi_p

        return utility.curl2D(self.grid, vr, vphi)

    def vortensity(self):
        """
        \omega / \Sigma
        """
        return self.vorticity()/self.data.rho

    def vortensity_source(self):
        """
        In planet frame
        :return:
        """
        sigma = self.data.rho
        (vr_p, vphi_p, _) = self.data.cyl_planet_velocity

        vr, vphi = self.data.u - vr_p, self.data.v - vphi_p
        dsigdr, dsigdphi = utility.grad2D(self.grid, sigma)
        dpidr, dpidphi = utility.grad2D(self.grid, self.data.p)
        # (\nabla \Sigma \times \nabla p) / \Sigma^3
        source = (dsigdr*dpidphi - dpidr*dsigdphi) / sigma**3
        # \mathbf{v} \cdot \nabla(\omega/\Sigma)
        # dvortensitydr, dvortensitydphi = utility.grad2D(self.grid, self.vortensity()/sigma)

        return source # - (vr * dvortensitydr + vphi * dvortensitydphi)

    def torque_density(self, plot=False):
        sigma = self.data.rho
        GM_p = self.params.get("gm_p")
        eps = self.params.get('eps_d') * np.sqrt(self.params.get('temp_d')) * self.params.get('r_d')

        r, phi = self.grid.r[:, np.newaxis], self.grid.phi[np.newaxis, :]
        xp, yp, zp, rp = self.data.xp, self.data.yp, self.data.zp, self.data.rp
        phi_p = self.data.phiPlanet
        S = self.grid.distance(xp, yp, zp)
        dPhi = rp * r * np.sin(phi - phi_p) * (GM_p / (S**2 + eps**2)**1.5)
        torq_dens = sigma * dPhi

        if plot:
            # rescale to -1, 1 and plot as SymLogNorm
            maxval = np.abs(torq_dens).max()
            sign = np.sign(torq_dens)
            log = np.log(np.abs(torq_dens)/maxval)
            log *= sign
            return log

        return torq_dens

