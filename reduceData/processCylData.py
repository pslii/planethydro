#!/home/marina/envs/planet/bin/python

import pandas as pd

from .. import parseData
from .reduceCylData import reduceCylData
from .utility import azAverage
from ..tecOutput import tecOutput


__author__ = 'Patrick'


def processCylData(verbose=False):
    params, grid, dataReader = parseData.initialize()
    start, end, num = tecOutput.detectData()

    varlist1D = ['r', 'sigma_avg', 'vphi_avg', 'torque_avg']
    asciiheader = 'variables = ' + ','.join(varlist1D)

    varlist2D = ['sigma', 'pi', 'sigma_i', 'rho_pertb', 'torque', 'torque_i', 'vphi', 'LR']
    tecOutXY = tecOutput.outputTec(varlist2D, grid, outDim='xy', output=True, suffix='xy')

    varlist2Dxz = ['rho', 'p', 'vr', 'vphi', 'vz']
    tecOutXZ = tecOutput.outputTec(varlist2Dxz, grid, outDim='xz', output=True, suffix='xz')

    varlist3D = ['rho', 'rho_i', 'rho_pertb']
    tecOut3D = tecOutput.outputTec(varlist3D, grid, outDim='xyz', output=True, suffix='xyz')

    if start > 0:
        data0 = dataReader.readData(0, legacy=False)
        reduce0 = reduceCylData(grid, params, data0)
        sigma0 = reduce0.sigma(asp_ratio=0.03)

    for ndat in range(start, end):
        data = dataReader.readData(ndat, legacy=False, verbose=verbose)
        process = reduceCylData(grid, params, data)
        if ndat == 0:
            data0 = data
            reduce0 = process
            sigma0 = reduce0.sigma(asp_ratio=0.03)

        # process 3D data
        output_dict = {'rho': data.rho,
                       'rho_i': data.rho - data0.rho,
                       'rho_pertb': process.rhoPertb(zavg=False)}
        tecOut3D.writeCylindrical(ndat, output_dict, data.phiPlanet)

        # process 2D data
        output_dict = {'rho': data.rho[:, 0, :],
                       'p': data.p[:, 0, :],
                       'vr': data.u[:, 0, :],
                       'vphi': data.v[:, 0, :],
                       'vz': data.w[:, 0, :]}
        tecOutXZ.writeRZ(ndat, output_dict)

        output_dict = {'sigma': process.sigma(asp_ratio=0.03)}
        output_dict['sigma_i'] = output_dict['sigma'] - sigma0
        output_dict['pi'] = process.pi(asp_ratio=0.03)
        output_dict['rho_pertb'] = process.rhoPertb(zavg=True)
        output_dict['torque'] = process.zTorque(zavg=True)
        output_dict['torque_i'] = output_dict['torque'] - process._zTorque(data0.rho, zavg=True)
        output_dict['vphi'] = process.vPhi(asp_ratio=0.03)
        output_dict['LR'] = process.lindbladRes()

        tecOutXY.writeCylindrical(ndat, output_dict, data.phiPlanet)

        # process 1D data
        rho_avg = azAverage(grid, output_dict['sigma'])
        torque_avg = azAverage(grid, output_dict['torque'])
        vphi_avg = azAverage(grid, output_dict['vphi'])

        output_dict = {'r': grid.r,
                       'sigma_avg': rho_avg,
                       'torque_avg': torque_avg,
                       'vphi_avg': vphi_avg}
        df = pd.DataFrame(output_dict, columns=varlist1D)
        fname = str(ndat).zfill(4) + 'cut.dat'
        print "Writing file: " + fname
        with open(fname, 'w') as fout:
            fout.write(asciiheader + '\n')
            df.to_csv(fout, index=False, header=False, sep='\t')