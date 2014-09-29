#!/home/marina/envs/planet/bin/python

from .. import parseData
from planetHydro.reduceData.reduceCylData import reduceCylData
from ..tecOutput import tecOutput

__author__ = 'Patrick'


def processCylData():
    params, grid, dataReader = parseData.initialize()
    start, end, num = tecOutput.detectData()

    varlist3D = ['rho', 'rho_i', 'rho_pertb']
    tecOut3D = tecOutput.outputTec(varlist3D, grid, output=True, suffix='_3d.dat')

    varlist = ['sigma', 'pi', 'sigma_i', 'rho_pertb', 'torque', 'torque_i', 'vphi', 'LR']
    tecOut = tecOutput.outputTec(varlist, grid, outDim=2, output=True)

    if start > 0:
        data0 = dataReader.readData(0, legacy=False)
        reduce0 = reduceCylData(grid, params, data0)
        sigma0 = reduce0.sigma(asp_ratio=0.03)

    for ndat in range(start, end):
        data = dataReader.readData(ndat, legacy=False)
        process = reduceCylData(grid, params, data)
        if ndat == 0:
            data0 = data
            reduce0 = process
            sigma0 = reduce0.sigma(asp_ratio=0.03)

        # process 3D data
        output_dict = {'rho': data.rho,
                       'rho_i': data.rho - data0.rho,
                       'rho_pertb': process.rhoPertb(zavg=False)}
        tecOut3D.writeNDat(ndat, output_dict, data.phiPlanet)

        # process 2D data
        output_dict = {'sigma': process.sigma(asp_ratio=0.03)}
        output_dict['sigma_i'] = output_dict['sigma'] - sigma0
        output_dict['pi'] = process.pi(asp_ratio=0.03)
        output_dict['rho_pertb'] = process.rhoPertb(zavg=True)
        output_dict['torque'] = process.zTorque(zavg=True)
        output_dict['torque_i'] = output_dict['torque'] - process._zTorque(data0.rho, zavg=True)
        output_dict['vphi'] = process.vPhi(asp_ratio=0.03)
        output_dict['LR'] = process.lindbladRes()

        tecOut.writeNDat(ndat, output_dict, data.phiPlanet)