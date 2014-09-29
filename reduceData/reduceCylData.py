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

    def get1D(self, arr, asp_ratio=None):
        """
        Returns 1D distribution.
        :param arr:
        :param asp_ratio:
        :return:
        """
        return utility.azAverage(self.grid, self.diskAverage(arr, asp_ratio=asp_ratio))

    def diskAverage(self, arr, asp_ratio=None):
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

    def sigma(self, asp_ratio=None):
        return self.diskAverage(self.data.rho, asp_ratio=asp_ratio)

    def pi(self, asp_ratio=None):
        return self.diskAverage(self.data.p, asp_ratio=asp_ratio)

    def vPhi(self, asp_ratio=None):
        return self.diskAverage(self.data.v, asp_ratio=asp_ratio)

    def rhoPertb(self, zavg=False):
        azAvg = utility.azAverage(self.grid, self.data.rho)
        rho_pertb = self.data.rho - azAvg[:, np.newaxis, :]
        if zavg:
            return self.diskAverage(rho_pertb)
        else:
            return rho_pertb

    def _zTorque(self, rho, zavg):
        GM_p = self.params['gm_p']
        xp, yp, zp = self.data.xp, self.data.yp, self.data.zp
        r = self.grid.distance(xp, yp, zp)
        force = -GM_p * rho / r ** 3
        rpr = (xp * self.grid.yDist(yp) - yp * self.grid.xDist(xp))  # r_p cross r
        torque = force * np.dstack((rpr,) * self.grid.nztot)
        if zavg:
            return self.diskAverage(torque)
        else:
            return torque

    def zTorque(self, zavg=False):
        return self._zTorque(self.data.rho, zavg)

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