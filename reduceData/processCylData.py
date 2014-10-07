#!/home/marina/envs/planet/bin/python

import pandas as pd

from .. import parseData
from .reduceCylData import reduceCylData
from .utility import azAverage
from ..tecOutput import tecOutput


__author__ = 'Patrick'


def processCylData(outputs=['x', 'xy', 'xz', 'xyz'], datarange=(None, None), verbose=False):
    params, grid, dataReader = parseData.initialize()
    if (datarange[0] is None) | (datarange[1] is None):
        start, end, _ = tecOutput.detectData()
    else:
        start, end = datarange

    if 'x' in outputs:
        varlist1D = ['r', 'sigma_avg', 'vphi_avg', 'torque_avg']
        asciiheader = 'variables = ' + ','.join(varlist1D)

    if 'xy' in outputs:
        varlist2D = ['sigma', 'sigma_i', 'sigma_pertb', 'rho_mid', 'pi', 'torque', 'torque_i', 'vphi', 'LR']
        tecOutXY = tecOutput.outputTec(varlist2D, grid, outDim='xy', output=True, suffix='xy')

    if 'xz' in outputs:
        varlist2Dxz = ['rho', 'p', 'vr', 'vphi', 'vz']
        tecOutXZ = tecOutput.outputTec(varlist2Dxz, grid, outDim='xz', output=True, suffix='xz')

    if 'xyz' in outputs:
        varlist3D = ['rho', 'rho_i', 'rho_pertb']
        tecOut3D = tecOutput.outputTec(varlist3D, grid, outDim='xyz', output=True, suffix='xyz')

    if start > 0:
        data0 = dataReader.readData(0, legacy=False)
        reduce0 = reduceCylData(grid, params, data0)
        sigma0, _ = reduce0.sigma()

    for ndat in range(start, end):
        data = dataReader.readData(ndat, legacy=False, verbose=verbose)
        process = reduceCylData(grid, params, data)
        if ndat == 0:
            data0 = data
            reduce0 = process
            sigma0, _ = reduce0.sigma()

        # process 3D data
        if 'xyz' in outputs:
            output_dict = {'rho': data.rho,
                           'rho_i': data.rho - data0.rho,
                           'rho_pertb': process.rhoPertb()}
            tecOut3D.writeCylindrical(ndat, output_dict, data.phiPlanet)

        # process 2D data
        if 'xz' in outputs:
            output_dict = {'rho': data.rho[:, 0, :],
                           'p': data.p[:, 0, :],
                           'vr': data.u[:, 0, :],
                           'vphi': data.v[:, 0, :],
                           'vz': data.w[:, 0, :]}
            tecOutXZ.writeRZ(ndat, output_dict)

        sigma, sigma1d = process.sigma()
        if 'xy' in outputs:
            output_dict = {'sigma': sigma}
            # noinspection PyUnboundLocalVariable
            output_dict['sigma_i'] = output_dict['sigma'] - sigma0
            output_dict['sigma_pertb'] = process.sigmaPertb()
            output_dict['rho_mid'] = data.rho[:, :, grid.nztot / 2]
            output_dict['pi'] = process.pi()
            output_dict['torque'] = process.zTorque(zavg=True, plot=True)
            output_dict['torque_i'] = output_dict['torque'] - process._zTorque(data0.rho, zavg=True, plot=True)
            output_dict['vphi'] = process.vPhi()
            output_dict['LR'] = process.lindbladRes()

            tecOutXY.writeCylindrical(ndat, output_dict, data.phiPlanet)

        if 'x' in outputs:
            # process 1D data
            torque_avg = azAverage(grid, process.zTorque(zavg=True))
            vphi_avg = azAverage(grid, process.vPhi())

            output_dict = {'r': grid.r,
                           'sigma_avg': sigma1d,
                           'torque_avg': torque_avg,
                           'vphi_avg': vphi_avg}
            df = pd.DataFrame(output_dict, columns=varlist1D)
            fname = str(ndat).zfill(4) + 'cut.dat'
            print "Writing file: " + fname
            with open(fname, 'w') as fout:
                fout.write(asciiheader + '\n')
                df.to_csv(fout, index=False, header=False, sep='\t')