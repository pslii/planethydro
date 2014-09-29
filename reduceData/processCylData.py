#!/home/marina/envs/planet/bin/python
import numpy as np

from .. import parseData
from . import processData
from ..tecOutput import tecOutput


__author__ = 'Patrick'


class cylDataReduction:
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
        return processData.azAverage(self.grid, self.diskAverage(arr, asp_ratio=asp_ratio))

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
            outputArr = processData.integrateZ(self.grid, arr)
        return outputArr

    def sigma(self, asp_ratio=None):
        return self.diskAverage(self.data.rho, asp_ratio=asp_ratio)

    def pi(self, asp_ratio=None):
        return self.diskAverage(self.data.p, asp_ratio=asp_ratio)

    def vPhi(self, asp_ratio=None):
        return self.diskAverage(self.data.v, asp_ratio=asp_ratio)

    def rhoPertb(self, zavg=False):
        azAvg = processData.azAverage(self.grid, self.data.rho)
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


def _test3d():
    params, grid, dataReader = parseData.initialize()
    tecOut = tecOutput.outputTec(['rho', 'rho_i', 'rho_pertb'], grid, output=True, suffix='_3d.dat')

    start, end, num = tecOutput.detectData()

    for ndat in range(start, end):
        data = dataReader.readData(ndat, legacy=False)
        process = cylDataReduction(grid, params, data)
        if ndat == 0:
            data0 = data

        output_dict = {'rho': data.rho,
                       'rho_i': data.rho - data0.rho,
                       'rho_pertb': process.rhoPertb(zavg=False)}
        tecOut.writeNDat(ndat, output_dict, data.phiPlanet)


def _test():
    params, grid, dataReader = parseData.initialize()
    varlist = ['sigma', 'pi', 'sigma_i', 'rhoPertb', 'torque', 'torque_i', 'vphi']
    tecOut = tecOutput.outputTec(varlist, grid, outDim=2, output=True)
    start, end, num = tecOutput.detectData()

    for ndat in range(start, end):
        data = dataReader.readData(ndat, legacy=False)
        process = cylDataReduction(grid, params, data)
        if ndat == 0:
            data0 = data
            reduce0 = process
            sigma0 = reduce0.sigma(asp_ratio=0.03)

        output_dict = {'sigma': process.sigma(asp_ratio=0.03)}
        output_dict['sigma_i'] = output_dict['sigma'] - sigma0
        output_dict['pi'] = process.pi(asp_ratio=0.03)
        output_dict['rhoPertb'] = process.rhoPertb(zavg=True)
        output_dict['torque'] = process.zTorque(zavg=True)
        output_dict['torque_i'] = output_dict['torque'] - process._zTorque(data0.rho, zavg=True)
        output_dict['vphi'] = process.vPhi(asp_ratio=0.03)

        tecOut.writeNDat(ndat, output_dict, data.phiPlanet)
    """
    plt.clf()
    for i in np.arange(.03, .15, .03):
    plt.plot(grid.r, reduce.get1D(data.rho, asp_ratio=i), label=str(i))
    plt.plot(grid.r, reduce.get1D(data.rho, asp_ratio=None))
    plt.legend(loc='upper left')
    plt.show()
    """


if __name__ == "__main__":
    _test3d()
    _test()