import numpy as np

from planetHydro.reduceData import utility

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
        return (arr * self.grid.dz3D).sum(axis=2)

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
        sigma1D = (self.data.rho * self.grid.dz3D * self.grid.dphi3D).sum(axis=1) / (2.0 * np.pi)
        sigma1D = sigma1D.sum(axis=1)
        return self.diskFlatten(self.data.rho), sigma1D

    def pi(self):
        return self.diskFlatten(self.data.p)

    def vPhi(self, rhoThreshold=None):
        return self.diskAverage(self.data.v, rhoThreshold)

    def sigmaPertb(self):
        sigma, sigma1D = self.sigma()
        return sigma - sigma1D[:, np.newaxis]

    def rhoPertb(self, rhoThreshold=None):
        azAvg = utility.azAverage(self.grid, self.data.rho)
        rho_pertb = self.data.rho - azAvg[:, np.newaxis, :]
        return rho_pertb

    def _zTorque(self, rho, zavg=True, rhoThreshold=None, plot=False):
        GM_p = self.params['gm_p']
        xp, yp, zp = self.data.xp, self.data.yp, self.data.zp
        r = np.sqrt(self.grid.distance(xp, yp, zp) ** 2 + self.eps() ** 2)
        force = -GM_p * rho / r ** 3
        rpr = (xp * self.grid.yDist(yp) - yp * self.grid.xDist(xp))  # r_p cross r
        if plot:
            rpr = -np.abs(-rpr)

        torque = force * np.dstack((rpr,) * self.grid.nztot)
        if zavg:
            return self.diskAverage(torque, rhoThreshold)
        else:
            return torque

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
                                        arr_start=(np.sqrt(1.0 / r_in),) * self.grid.nztot,
                                        arr_end=(np.sqrt(1.0 / r_out),) * self.grid.nztot)
        return (self.data.v + self.grid.r3D * dvphidr) / (2 * self.grid.r3D)

    def midplane_vortensity(self):
        """
        \Sigma * dlog(\Sigma/B)/dr = \Sigma * dlogR/dr * dlog(\Sigma/B)/dlogR
        -> dlog(\Sigma/B)/dlogR = (dlog(\Sigma/B)/dr) / (dlogR/dr) = r * dlog(\Sigma/B)/dr
        :return: \Sigma * r * dlog(\Sigma/B)/dr
        """
        (Sigma, _), B = self.sigma(), (self.oortB())[:, :, self.grid.nztot / 2]

        logSigmaB = np.log(Sigma / B)
        dlogSigmaBdr = utility.centralDiff(self.grid, logSigmaB)

        return Sigma * self.grid.r2D * dlogSigmaBdr


