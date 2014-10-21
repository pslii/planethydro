#!/home/marina/envs/planet/bin/python -u

__author__ = 'pslii'
import numpy as np

import planetHydro.parseData as parseData
import planetHydro.reduceData as reduceData


params, grid, dataReader = parseData.initialize()
data = dataReader.readData(5)
process = reduceData.reduceCylData(grid, params, data)

vmid = data.v[:, :, grid.nztot / 2]
# r_in = (grid.r_edge[1]+grid.r_edge[2])/2.0
r_out = (grid.r_edge[-2] + grid.r_edge[-3]) / 2.0
dvdr = reduceData.utility.centralDiff(grid, vmid, arr_end=np.sqrt(1.0 / r_out))
B = (dvdr * grid.r2D + vmid) / (2.0 * grid.r2D)
sigma, _ = process.sigma()

logSigmaB = np.log(sigma / B)
dlogSigmaBdr = reduceData.utility.centralDiff(grid, logSigmaB)