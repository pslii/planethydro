#!/home/marina/envs/planet/bin/python

import pandas as pd
from .. import parseData
from .reduceCylData import reduceCylData
from .utility import azAverage
from .utility import polar_plot
from ..tecOutput import tecOutput
import numpy as np
import os


__author__ = 'Patrick'


def processCylData(outputs=['x', 'xy', 'xz', 'xyz', 'time'],
                   datarange=(None, None), verbose=False, hydro=True):
    params, grid, dataReader = parseData.initialize(hydro=hydro)
    if (datarange[0] is None) | (datarange[1] is None):
        start, end, _ = tecOutput.detectData()
    else:
        start, end = datarange

    if 'x' in outputs:
        varlist1D = ['r', 'sigma_avg', 'vphi_avg', 'torque_avg', 'vort_avg']
        asciiheader = 'variables = ' + ','.join(varlist1D)

    if 'xy' in outputs:
        varlist2D = ['sigma', 'sigma_i', 'sigma_pertb', 'rho_mid',
                     'pi', 'torque', 'torque_i', 'vphi', 'LR',
                     'vortensity', 'vort_grad']
        tecOutXY = tecOutput.outputTec(varlist2D, grid, outDim='xy', output=True, suffix='xy')

    if 'xz' in outputs:
        varlist2Dxz = ['rho', 'p', 'T', 'vr', 'vphi', 'vz']
        tecOutXZ = tecOutput.outputTec(varlist2Dxz, grid, outDim='xz', output=True, suffix='xz')

    if 'xyz' in outputs:
        varlist3D = ['rho', 'rho_i', 'rho_pertb']
        tecOut3D = tecOutput.outputTec(varlist3D, grid, outDim='xyz', output=True, suffix='xyz')

    if 'png' in outputs:
        if not (os.path.isdir('./png')):
            print("Creating directory png")
            os.mkdir('png')

    if 'time' in outputs:
        varlistTime = ['time', 'sax', 'r_gap_sigma', 'r_gap_thresh',
                       'in_torque', 'out_torque',
                       'ILR_torque', 'OLR_torque', 'LR_torque',
                       'COR_torque', 'torque_tot']
        timeDict = {k: [] for k in varlistTime}
        timeheader = 'variables = ' + ','.join(varlistTime)


    i_disk = np.where(grid.r > params.get('r_gap'))[0][0]
    if start > 0:
        data0 = dataReader.readData(0, legacy=False)
        reduce0 = reduceCylData(grid, params, data0)
        sigma0, sigma1d = reduce0.sigma()
        sigma_disk = sigma1d[i_disk]

    for ndat in range(start, end):
        data = dataReader.readData(ndat, legacy=False)
        process = reduceCylData(grid, params, data)
        if ndat == 0:
            data0 = data
            reduce0 = process
            sigma0, _ = reduce0.sigma()
            sigma0, sigma1d = reduce0.sigma()
            sigma_disk = sigma1d[i_disk]

        """
        m = data.rho*grid.r3D*grid.dr3D*grid.dphi3D*grid.dz3D
        import numpy as np
        m_interior = m[np.where(grid.r < params['r_gap'])[0],:,:].sum()
        print m_interior, m.sum()
        """

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
                           'T': data.p[:, 0, :]/data.rho[:, 0, :],
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
            output_dict['vortensity'] = process.vortensity()
            output_dict['vort_grad'] = process.midplane_vortensity()

            tecOutXY.writeCylindrical(ndat, output_dict, data.phiPlanet)

        if 'x' in outputs:
            # process 1D data
            torque_avg = azAverage(grid, process.zTorque(zavg=True))
            vphi_avg = azAverage(grid, process.vPhi())
            vort_avg = azAverage(grid, process.midplane_vortensity())

            output_dict = {'r': grid.r,
                           'sigma_avg': sigma1d,
                           'rho_mid': data.rho[:, 0, grid.nztot / 2],
                           'torque_avg': torque_avg,
                           'vort_avg': vort_avg,
                           'vphi_avg': vphi_avg}
            df = pd.DataFrame(output_dict, columns=varlist1D)
            fname = str(ndat).zfill(4) + 'cut.dat'
            print "Writing file: " + fname
            with open(fname, 'w') as fout:
                fout.write(asciiheader + '\n')
                df.to_csv(fout, index=False, header=False, sep='\t')

        if 'png' in outputs:
            print "Generating png..."
            polar_plot(grid, params, data.rp, data.phiPlanet,
                                 sigma, save='png/sigma_'+str(ndat).zfill(4))
            polar_plot(grid, params, data.rp, data.phiPlanet,
                                 data.rho[:,:,grid.nztot/2], save='png/rho_'+str(ndat).zfill(4))

        if 'time' in outputs:
            # compute time series data
            timeDict['time'].append(data.time/params.get('period'))
            sax, ecc, incl = process.orb_elements()
            timeDict['sax'].append(sax)
            r_gap_sigma, r_gap_thresh = process.disk_boundary(sigma_disk=sigma_disk)
            timeDict['r_gap_sigma'].append(r_gap_sigma)
            timeDict['r_gap_thresh'].append(r_gap_thresh)
            ilr, olr, cor, lrtot, tot, in_torque, out_torque = process.resonance_torques()
            timeDict['in_torque'].append(in_torque)
            timeDict['out_torque'].append(out_torque)
            timeDict['ILR_torque'].append(ilr)
            timeDict['OLR_torque'].append(olr)
            timeDict['LR_torque'].append(lrtot)
            timeDict['COR_torque'].append(cor)
            timeDict['torque_tot'].append(tot)

    if 'time' in outputs:
        df = pd.DataFrame(timeDict, columns=varlistTime)
        with open('time.dat', 'w') as fout:
            fout.write(timeheader + '\n')
            df.to_csv(fout, index=False, header=False, sep='\t')



