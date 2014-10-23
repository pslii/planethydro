import numpy as np

from .utility import azAverage, colMul, centralDiff2D


__author__ = 'pslii'


class reduceData:
    """
    Usage:
    processor = reduceData(grid, params, data, metadata)
    rho_avg, rho_pertb = processor.rhoPertb()
    """

    def __init__(self, grid, params, data, metadata):
        self.ndat = metadata['ndat']
        self.grid = grid
        self.data = self.__processData(data)
        self.metadata = metadata
        self.params = params

        self.xn, self.yn, self.zn = metadata['xn'], \
                                    metadata['yn'], metadata['zn']
        self.un, self.vn, self.wn = metadata['up'], \
                                    metadata['vp'], metadata['wp']

        self.phi_pl = self.phiPlanet()
        self.omega_pl = self.omegaPlanet()

        self.omega_arr, self.domega_arr = None, None


    def __processData(self, data):
        newdata = {}
        for key, value in data.iteritems():
            if len(value.shape) > 2:
                newdata[key] = self._getMidplane(value)
        return newdata

    @staticmethod
    def _getMidplane(arr):
        _, _, nztot = arr.shape
        return arr[:, :, nztot / 2]

    def rhovr(self):
        return self.data['rho'] * self.data['u']

    def rhovphi(self):
        return self.data['rho'] * self.data['v']

    def rhov(self):
        return (self.rhovr() ** 2 + self.rhovphi() ** 2) ** 0.5

    def vPertb(self):
        """
        Returns:
        2D array: v_phi-v_Kep
        2D array: v_phi-v_planet
        """
        GM = self.params['GM']
        v_kep = np.sqrt(GM / self.grid.r)
        v_pl = self.omega_pl * self.grid.r
        subtractkep = lambda arr: arr - v_kep
        subtractpl = lambda arr: arr - v_pl
        return np.apply_along_axis(subtractkep, 0, self.data['v']), \
               np.apply_along_axis(subtractpl, 0, self.data['v'])

    def rhoPertb(self):
        """
        Returns:
        1D array: rho_avg
        2D array: rho-rho_avg
        """
        rho_avg = azAverage(self.grid, self.data['rho'])
        rho_pertb = self.data['rho'] - rho_avg.reshape((self.grid.nxtot, 1))
        return rho_avg, rho_pertb

    def calcVelocities(self, vr=None, vphi=None, phi=None):
        """
        Converts vr, vphi into vx, vy
        """
        if phi is None:
            phi = self.phi_pl
        if vr is None:
            vr = self.data['u']
        if vphi is None:
            vphi = self.data['v']
        phi_grid = (self.grid.phi - phi + np.pi) % (2 * np.pi)
        vx = colMul(np.cos(phi_grid), vr.transpose()) - \
             colMul(np.sin(phi_grid), vphi.transpose())
        vy = colMul(np.sin(phi_grid), vr.transpose()) + \
             colMul(np.cos(phi_grid), vphi.transpose())
        return vx.transpose(), vy.transpose()

    def oortB(self):
        """
        Computes Oort's second constant:
        B = 1/(2r) d(r^2 \Omega)/dr
        = r/2 (v_phi + r dv_phi/dr)

        WRONG!!!
        """
        vphi = self.data['v']
        r_in = (self.grid.r_edge[1] + self.grid.r_edge[2]) / 2.0
        r_out = (self.grid.r_edge[-1] + self.grid.r_edge[-2]) / 2.0
        dvphidr = centralDiff2D(self.grid, vphi,
                              arr_start=np.sqrt(1.0 / r_in),
                              arr_end=np.sqrt(1.0 / r_out))
        terms = vphi + colMul(self.grid.r, dvphidr)
        return colMul(self.grid.r / 2.0, terms)

    def vortensity(self):
        """
        \Sigma * dlog(\Sigma/B)/dr = \Sigma * dlogR/dr * dlog(\Sigma/B)/dlogR
        -> dlog(\Sigma/B)/dlogR = (dlog(\Sigma/B)/dr) / (dlogR/dr) = r * dlog(\Sigma/B)/dr
        :return: \Sigma * r * dlog(\Sigma/B)/dr
        """

        sigma, vphi = self.data['rho'], self.data['v']
        r = self.grid.r
        logB = np.log(self.oortB())
        logSigma = np.log(sigma)

        dlogB = colMul(r, centralDiff2D(self.grid, logB))
        dlogSigma = colMul(r, centralDiff2D(self.grid, logSigma))
        return sigma * dlogB, sigma * dlogSigma, sigma * (dlogSigma - dlogB)

    def omega(self):
        if self.omega_arr is None:
            GM = self.params['GM']
            omega = colMul(self.grid.r, self.data['v'])
            r_in = (self.grid.r_edge[1] + self.grid.r_edge[2]) / 2.0
            r_out = (self.grid.r_edge[-1] + self.grid.r_edge[-2]) / 2.0
            domega = centralDiff2D(self.grid, omega,
                                 arr_start=-1.5 * np.sqrt(GM / r_in ** 5),
                                 arr_end=-1.5 * np.sqrt(GM / r_out ** 5))
            self.omega_arr, self.domega_arr = omega, domega
        return self.omega_arr, self.domega_arr

    def kappa2(self):
        """
        Epicyclic frequency squared
        """
        omega, domega = self.omega()
        vphi = self.data['v']
        kappa2 = 4.0 * omega ** 2 + 2.0 * vphi * domega
        return kappa2

    def lindbladRes(self):
        lr = np.zeros((self.grid.nxtot, self.grid.nytot))
        rp = np.sqrt(self.xn ** 2 + self.yn ** 2)
        for m in np.arange(1, 6):
            lr_in, lr_out = self.grid.r.min(), self.grid.r.max()

            lr_out = ((1.0 + 1.0 / m) ** (2.0 / 3.0)) * rp
            if m != 1: lr_in = ((1.0 - 1.0 / m) ** (2.0 / 3.0)) * rp
            loc = np.where((self.grid.r >= lr_in) & (self.grid.r <= lr_out))
            lr[loc, :] = m

        dist = np.sqrt((self.grid.x - self.xn) ** 2 + (self.grid.y - self.yn) ** 2)
        lr[np.where(dist < .1)] = -1.0
        return lr

    def cs2(self):
        """
        Sound speed squared
        """
        gamma = self.params['gam']
        return gamma * self.data['p'] / self.data['rho']

    def torque_c(self):
        GM_p = self.params['GM_p']
        rp = np.sqrt(self.xn ** 2 + self.yn ** 2 + self.zn ** 2)
        r = self.grid.r
        sigma = self.data['rho']
        B = self.oortB()
        omega, domega = self.omega()
        _, _, vortensity = self.vortensity()
        m = 1

        Phi_p = GM_p / (r - rp)
        gamma_c = colMul((m * (Phi_p * np.pi) ** 2 / (2.0 * r)), (vortensity / (B * domega)))
        return gamma_c

    def torques(self, smoothing=True, pos=False):
        """
        Computes torque on planet from the disk
        """
        return self.torque_sigma(self.data['rho'],
                                 smoothing=smoothing, pos=pos)

    def torque_sigma(self, sigma, smoothing=True, pos=False):
        """
        Computes torque on planet given a density distribution
        """
        GM, GM_p = self.params['GM'], self.params['GM_p']
        xn, yn, zn = self.xn, self.yn, self.zn
        r_dist = np.array([xn - self.grid.x, yn - self.grid.y])
        if smoothing:
            t_disk, r_gap = self.params['temp_d'], self.params['r_d']
            asp_ratio = np.sqrt(t_disk / (GM / r_gap))
            h_disk = asp_ratio * r_gap
            eps = 0.1 * h_disk
            r_mag = np.sqrt(r_dist[0, :, :] ** 2 + r_dist[1, :, :] ** 2 + eps ** 2)
        else:
            r_mag = np.sqrt(r_dist[0, :, :] ** 2 + r_dist[1, :, :] ** 2)
        # torque = np.zeros((self.params['nxtot'], self.params['nytot']))
        force = -GM_p * sigma / r_mag ** 2
        cross = (yn * self.grid.x - xn * self.grid.y)
        if pos:
            # not physical, for ease of plotting only
            cross = -np.abs(-cross)

        torque = cross / r_mag * force
        return force, torque

    def planetPos(self):
        return np.sqrt(self.xn ** 2 + self.yn ** 2 + self.zn ** 2)

    def hillRadius(self):
        rp = np.sqrt(self.xn ** 2 + self.yn ** 2 + self.zn ** 2)
        GM, GM_p = self.params['GM'], self.params['GM_p']
        return rp * (GM_p / (3.0 * GM)) ** (1.0 / 3.0)

    def __rotateCoords(self, frame='planet'):
        xn, yn, zn = self.xn, self.yn, self.zn
        up, vp, wp = self.metadata['up'], \
                     self.metadata['vp'], \
                     self.metadata['wp']

        if frame == 'disk':
            o_syst = 0.0
            phi_syst = 0.0
        elif frame == 'planet':
            o_syst = -self.omega_pl
            phi_syst = self.phi_pl
        elif frame == 'inertial':
            GM, time, r0 = self.metadata['GM'], \
                           self.metadata['time'], \
                           self.params['r_d']
            o_syst = -np.sqrt(GM / r0 ** 3)
            phi_syst = o_syst * time
        else:
            assert False
        xp = xn * np.cos(phi_syst) + yn * np.sin(phi_syst)
        yp = xn * np.cos(phi_syst) + yn * np.sin(phi_syst)
        zp = xn * np.cos(phi_syst) + yn * np.sin(phi_syst)

        ui = up - o_syst * yn
        vi = vp + o_syst * xn
        up = ui * np.cos(phi_syst) + vi * np.sin(phi_syst)
        vp = vi * np.cos(phi_syst) - ui * np.sin(phi_syst)
        wp = wp
        return (xp, yp, zp), (up, vp, wp)

    def omegaPlanet(self):
        """
        Compute omega of planet
        """
        xn, yn, zn = self.xn, self.yn, self.zn
        un, vn, wn = self.un, self.vn, self.wn
        return (vn * xn - un * yn) / (xn ** 2 + yn ** 2)

    def phiPlanet(self):
        """
        Compute phi of planet
        """
        xn, yn, zn = self.xn, self.yn, self.zn
        return np.arctan2(yn, xn) + 0.5 * np.pi

    def findPlanet(self):
        """
        Given xyz position of planet, calculates phi position
        and identifies j coordinate.
        """
        xn, yn, zn = self.xn, self.yn, self.zn
        phi_p = (np.arctan2(yn, xn) + 2.0 * np.pi) % (2.0 * np.pi)
        j = 0
        for j in range(self.grid.nytot):
            if self.grid.phi_edge[j] <= phi_p <= self.grid.phi_edge[j + 1]: break
        return j

    def setPlanetFrame(self):
        xn = self.xn, yn = self.yn, zn = self.zn
        un = self.metadata['up']
        vn = self.metadata['vp']
        wn = self.metadata['wp']
        o_syst = -(vn * xn - un * yn) / (xn ** 2 + yn ** 2)
        phi_p = np.arctan2(yn, xn)
        phi = (self.grid.phi_edge[2:-2] - phi_p + np.pi) % (2 * np.pi)
        indices = np.argsort(phi[np.where(phi < np.max(phi))[0]])
        phi = np.sort(phi)
        return phi, indices, o_syst