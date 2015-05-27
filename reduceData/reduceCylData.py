import numpy as np

from . import utility

__author__ = 'pslii'

class reduceCylData:
    def __init__(self, grid, params, data):
        """
        Data reduction module for 3D cylindrical code.

        :type grid: planetHydro.parseData.gridReader.gridReader
        :param grid:

        :type params: dict
        :param params:

        :type data: planetHydro.parseData.dataReader.SimData
        :param data:

        """
        self.data = data
        self.grid = grid
        self.params = params
        self.rhoXZ = None

    def get1D(self, arr, threshold=None):
        """
        Returns 1D distribution.
        :param arr:
        :param threshold:
        :return:
        """
        return utility.azAverage(self.grid, self.diskAverage(arr, threshold))

    def diskFlatten(self, arr):
        """
        \int arr dz
        Integrates array in z direction
        :param arr: array to be integrated
        :return: 2D vertically integrated array
        """
        return (arr * self.grid.dz[np.newaxis, np.newaxis, :]).sum(axis=2)

    def diskAverage(self, arr, rhoThreshold):
        """
        Takes a height average of array in z direction,
        using threshold density as cutoff for disk height
        :param arr: array to be averaged
        :param rhoThreshold:  threshold density
        :return: 2D, z-averaged array
        """
        if rhoThreshold is None:
            rhoThreshold = self.params.get('rho_disk') / 10.0

        outputArr = np.zeros((self.grid.nxtot, self.grid.nytot))
        if self.rhoXZ is None:
            self.rhoXZ = utility.azAverage(self.grid, self.data.rho)

        for i in range(self.grid.nxtot):
            intRange = np.where(self.rhoXZ[i, :] >= rhoThreshold)[0]
            if intRange.shape[0] == 0: continue

            subArr = arr[i, :, intRange]
            dz = self.grid.dz[intRange]
            zrange = dz.sum()
            outputArr[i, :] = np.dot(subArr.transpose(), dz) / zrange
        return outputArr

    def diskAverageAsp(self, arr, asp_ratio=None):
        """
        Vertically averages the given array over the height of the disk (given an aspect ratio).
        Otherwise, simply averages over the entire height of the simulation region.

        :param arr: 3D array to be vertically averaged. Must have same dimensions as grid.
        :return: Vertically averaged 2D array.
        """
        assert arr.shape == self.grid.shape, "Input array must have same number of dimensions as the grid."

        if (asp_ratio is not None) | ('asp_ratio' in self.params.keys()):
            asp_ratio = self.params['asp_ratio'] if asp_ratio is None else asp_ratio
            H = self.grid.r * asp_ratio
            outputArr = np.zeros((self.grid.nxtot, self.grid.nytot))

            for i in range(self.grid.nxtot):
                intRange = np.where((self.grid.z <= H[i]) & (self.grid.z >= -H[i]))[0]
                if intRange.shape == (0,): continue
                dz = self.grid.dz[intRange]
                zrange = dz.sum()
                subArr = arr[i, :, intRange]
                outputArr[i, :] = np.dot(subArr.transpose(), dz) / zrange
        else:
            print "No disk aspect ratio detected. Integrating over whole simulation region."
            outputArr = utility.integrateZ(self.grid, arr)
        return outputArr

    def eps(self):
        if self.params.get('eps') is None:
            return self.params.get('asp_ratio') * self.params.get('smooth_ratio') * self.params.get('sax_init')
        else:
            return self.params.get('eps')

    def sigma(self):
        """
        Calculate surface density and azimuthally averaged surface density.
        \int \rho dz, \int \int \rho dphi dz / 2 \pi
        :return: sigma, sigma1d
        """
        sigma = self.diskFlatten(self.data.rho) 
        sigma1D = (sigma * self.grid.dphi[np.newaxis, :]).sum(1)/(2*np.pi)
        return sigma, sigma1D

    def pi(self):
        return self.diskFlatten(self.data.p)

    def vPhi(self, rhoThreshold=None):
        return self.diskAverage(self.data.v, rhoThreshold)

    def calcVelocities(self, phi_pl=None):
        k_mid = self.grid.nztot/2
        u, v = [x[:,:,k_mid] for x in [self.data.u, self.data.v]]
        if phi_pl is None:
            phi_pl = self.data.phiPlanet
        omega_pl = self.data.omegaPlanet
        v -= omega_pl * self.grid.r[:, np.newaxis]

        phi_grid = self.grid.phi - phi_pl
        vx = u * np.cos(phi_grid) - v * np.sin(phi_grid)
        vy = u * np.sin(phi_grid) + v * np.cos(phi_grid)
        return vx, vy

    def sigmaPertb(self):
        sigma, sigma1D = self.sigma()
        return sigma - sigma1D[:, np.newaxis]

    def rhoPertb(self, rhoThreshold=None):
        azAvg = utility.azAverage(self.grid, self.data.rho)
        rho_pertb = self.data.rho - azAvg[:, np.newaxis, :]
        return rho_pertb
    
    def force(self):
        """
        Computes the total force on the disk.
        """
        rho = self.data.rho
        GM, GM_p = self.params['gm'], self.params['gm_p']
        xp, yp, zp = self.data.xp, self.data.yp, self.data.zp
        r_sph = np.sqrt(self.grid.r3D**2 + self.grid.z3D**2)

        x, y, z = self.grid.x[:,:,np.newaxis], \
            self.grid.y[:,:,np.newaxis], \
            self.grid.z[np.newaxis,np.newaxis,:]

        # stellar term
        fs = -GM / r_sph**3
        fs_x, fs_y, fs_z = (fs * rho * q for q in (x, y, z))

        # planet term
        rrp = self.grid.distance(xp, yp, zp) # r-rp
        fp = -GM_p / (rrp**2 + self.eps()**2)**(3.0/2.0)
        fp_x, fp_y, fp_z = (fp * rho * q for q in ((x-xp), (y-yp), (z-zp)))
        
        # non-inertial term
        fi = -GM_p / np.sqrt(xp**2 + yp**2 + zp**2)**3
        fi_x, fi_y, fi_z = (fi * rho * q for q in (xp, yp, zp))

        # total
        ftot_x = fp_x + fi_x + fs_x
        ftot_y = fp_y + fi_y + fs_y
        ftot_z = fp_z + fi_z + fs_z

        torque_x = y * ftot_z - z * ftot_y
        torque_y = z * ftot_x - x * ftot_z
        torque_z = x * ftot_y - y * ftot_x
        
        tovec = lambda q : [self.grid.integrate(vec) for vec in utility.cart2cyl(self.grid, q)]
        
        fs = tovec((fs_x, fs_y, fs_z))
        fp = tovec((fp_x, fp_y, fp_z))
        fi = tovec((fi_x, fi_y, fi_z))
        ftot = tovec((ftot_x, ftot_y, ftot_z))
        
        return  fs, fp, fi, ftot, (torque_x, torque_y, torque_z)

    def _zTorque(self, rho, zavg=True, rhoThreshold=0.0, plot=False):
        """
        Computes torque from the disk.
        """
        GM, GM_p = self.params['gm'], self.params['gm_p']
        xp, yp, zp = self.data.xp, self.data.yp, self.data.zp
        r = np.sqrt(self.grid.distance(xp, yp, zp) ** 2.0 + self.eps() ** 2.0)
        force = -GM_p * rho / r ** 3.0
        rpr = (xp * self.grid.yDist(yp) - yp * self.grid.xDist(xp))  # r_p cross r
        if plot:
            rpr = -np.abs(rpr)

        torque = force * np.dstack((rpr,) * self.grid.nztot)
        if zavg:
            return self.diskAverage(torque, rhoThreshold)
        else:
            return torque

    def r_hill(self, circular_orbit=False):
        """
        Compute the Hill radius of the planet.
        :return:
        """
        sax, ecc, incl = self.orb_elements()
        GM_p, GM = self.params.get('gm_p'), self.params.get('gm')
        if circular_orbit:
            return sax * (GM_p/ (3.0 * GM))**(1.0/3.0)
        else:
            return sax * (1.0-ecc) * (GM_p/ (3.0 * GM))**(1.0/3.0)

    def cs(self):
        """
        :return: Sound speed in simulation region
        """
        gamma = self.params.get('gam')
        return np.sqrt(gamma * self.data.p/self.data.rho)

    def omega(self):
        return (self.data.v.transpose() * self.grid.r).transpose()

    def toomre_q(self):
        k_mid = self.grid.nztot/2
        cs, omega = self.cs()[:,:,k_mid], self.omega()[:,:,k_mid]
        sigma, _ = self.sigma()
        return cs * omega / (np.pi * sigma)

    def m_disk(self):
        """
        :return: The disk mass
        """
        rho = self.data.rho
        rho_xy = (rho * self.grid.dz).sum(axis=2)
        rho_x  = (rho_xy * self.grid.dphi).sum(axis=1)
        m = (rho_x * self.grid.r * self.grid.dr).sum(axis=0)
        return m

    def resonance_torques(self, spacing=3):
        """
        Calculate torques from various resonances.
        :return: LR torque, corotation torque
        """
        r = self.grid.r

        r_p, r_hill = self.data.rp, self.r_hill(circular_orbit=True)
        fs, fp, fi, ftot, (_, _, torque) = self.force()

        i_in = r<(r_p)
        i_out = r>=(r_p)

        i_ILR = r<=(r_p-spacing*r_hill)
        i_OLR = r>=(r_p+spacing*r_hill)
        i_COR = (r>(r_p-spacing*r_hill)) & (r<(r_p+spacing*r_hill))

        integrate = lambda x, y : (((x[y,:,:] * self.grid.dz).sum(2)
                                    * self.grid.dphi).sum(1)
                                    * self.grid.r[y] * self.grid.dr[y]).sum()

        ILR_Torque = integrate(torque, i_ILR)
        OLR_Torque = integrate(torque, i_OLR)
        Cor_Torque = integrate(torque, i_COR)
        in_Torque = integrate(torque, i_in)
        out_Torque = integrate(torque, i_out)
        return fs, fp, fi, ftot, \
            ILR_Torque, OLR_Torque, Cor_Torque, \
            ILR_Torque+OLR_Torque, \
            ILR_Torque+OLR_Torque+Cor_Torque, \
            in_Torque, out_Torque

    def zTorque(self, zavg=False, plot=False):
        return self._zTorque(self.data.rho, zavg, plot=plot)

    def lindbladRes(self):
        """
        :return: 2D array with Lindblad resonances
        """
        lr = np.zeros((self.grid.nxtot, self.grid.nytot))
        rp = np.sqrt(self.data.xp ** 2 + self.data.yp ** 2)
        for m in np.arange(1, 6):
            lr_in, lr_out = self.grid.r.min(), self.grid.r.max()

            lr_out = ((1.0 + 1.0 / m) ** (2.0 / 3.0)) * rp
            if m != 1: lr_in = ((1.0 - 1.0 / m) ** (2.0 / 3.0)) * rp
            loc = np.where((self.grid.r >= lr_in) & (self.grid.r <= lr_out))
            lr[loc, :] = m

        dist = np.sqrt((self.grid.x - self.data.xp) ** 2 + (self.grid.y - self.data.yp) ** 2)
        lr[np.where(dist < .1)] = -1.0
        return lr

    def oortB(self):
        """
        Computes Oort's second constant:
        B = 1/(2r) d(r^2 \Omega)/dr
        = 1/(2r) * [v_phi + r dv_phi/dr]
        """
        r_in = (self.grid.r_edge[1] + self.grid.r_edge[2]) / 2.0
        r_out = (self.grid.r_edge[-2] + self.grid.r_edge[-3]) / 2.0
        dvphidr = utility.centralDiff3D(self.grid, self.data.v,
                                        arr_end=(np.sqrt(1.0 / r_out),) * self.grid.nztot)
        return (self.data.v + self.grid.r3D * dvphidr) / (2 * self.grid.r3D)

    def vortensity(self):
        (Sigma, _), B = self.sigma(), (self.oortB())[:, :, self.grid.nztot / 2]
        return B/Sigma

    def midplane_vortensity(self):
        """
        \Sigma * dlog(\Sigma/B)/dr = \Sigma * dlogR/dr * dlog(\Sigma/B)/dlogR
        -> dlog(\Sigma/B)/dlogR = (dlog(\Sigma/B)/dr) / (dlogR/dr) = r * dlog(\Sigma/B)/dr
        :return: \Sigma * r * dlog(\Sigma/B)/dr
        :rtype: np.ndarray
        """
        (Sigma, _), B = self.sigma(), (self.oortB())[:, :, self.grid.nztot / 2]
        dlogBdr =  utility.centralDiff2D(self.grid, B)/B
        dlogSigmadr = utility.centralDiff2D(self.grid, Sigma)/Sigma
        dlogSigmaBdr = Sigma * self.grid.r2D * (dlogSigmadr-dlogBdr)
        return dlogSigmaBdr

    def orb_elements(self):
        GM = self.params.get('gm')
        xp, yp, zp = self.data.xp, self.data.yp, self.data.zp
        up, vp, wp = self.data.up, self.data.vp, self.data.wp

        rp = np.sqrt(xp**2+yp**2+zp**2)
        energy = 0.5 * (up**2+vp**2+wp**2) - GM/rp

        angmomx = yp*wp-zp*vp
        angmomy = zp*up-xp*wp
        angmomz = xp*vp-yp*up
        angmom = np.sqrt(angmomx**2+angmomy**2+angmomz**2)

        sax = -0.5 * GM / energy
        ecc = 1.0-angmom**2/(GM*sax)
        ecc = np.sqrt(ecc) if ecc >= 0 else 0

        incl = np.arccos(angmomz/angmom) * 180 / np.pi

        return sax, ecc, incl

    def disk_boundary(self, sigma_disk=None):
        """
        Finds the interior edge of the disk using several different methods:
        1. derivative of sigma
        2. density threshold based on max or initial sigma
        :return: float
        """
        _, sigma1d = self.sigma()
        dsdr = utility.centralDiff1D(self.grid, sigma1d)

        if not (sigma_disk is None):
            i_thresh = np.where(sigma1d >= sigma_disk)[0][0]
        else:
            i_thresh = np.where(sigma1d >= (sigma1d.max()/2.0))[0][0]

        return self.grid.r[dsdr.argmax()], \
               self.grid.r[i_thresh]

    def torque_density(self):
        """
        Computes the torque density.

        See D'Angelo & Lubow 2008
        """
        GM_p, eps = self.params.get('gm_p'), self.eps()
        
        xp, yp, zp, rp = self.data.xp, self.data.yp, self.data.zp, self.data.rp
        phi_p = self.data.phiPlanet

        rho = self.data.rho
        S = self.grid.distance(xp, yp, zp)

        r, phi = self.grid.r[:, np.newaxis, np.newaxis], self.grid.phi[np.newaxis, :, np.newaxis]
        
        dPhi = rp * r * np.sin(phi - phi_p) * (GM_p / (S**2 + eps**2)**1.5)
        integrand = (rho * dPhi * self.grid.dz).sum(2)
        integrand = (self.grid.dphi * integrand).sum(1)
        
        # Torque density
        torque_dens = integrand * self.grid.r

        # Torque per unit disk mass
        _, sigma = self.sigma()
        torque_per_unit_mass = integrand / (2 * np.pi * sigma)
        
        return torque_dens, torque_per_unit_mass
