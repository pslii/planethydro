#!/home/marina/envs/planet/bin/python

import pandas as pd
from .. import parseData
from .reduceCylData import reduceCylData
from .utility import azAverage
from .utility import polar_plot
from ..tecOutput import tecOutput
import numpy as np
import os, sys


__author__ = 'Patrick'


def get_range((start, end)):
    startNone, endNone = start is None, end is None
    start_detect, end_detect, _ = tecOutput.detectData()
    if startNone and endNone: # process all files
        start, end = start_detect, end_detect
    elif startNone & (not endNone): # start at 0 and process up to specified file
        start = 0
    elif (not startNone) & endNone: # process only 1 file
        end = start+1
    elif (not startNone) & (not endNone):
        end = end+1

    end = min(end, end_detect)
    assert start >= start_detect, 'Error: starting ndat must be positive.'
    assert end > start, 'Error: ending ndat must be greater than starting ndat.'

    return start, end

def processCylData(outputs=['x', 'xy', 'xz', 'xyz', 'time'],
                   datarange=(None, None), skip=None, verbose=False, hydro=True):
    params, grid, dataReader = parseData.initialize(hydro=hydro)
    start, end = get_range(datarange)
    if skip is None: skip = 1

    if 'x' in outputs:
        varlist1D = ['r', 'r_norm', 'sigma_avg', 'rho_mid', \
                         'torque_avg', 'torque_density', 'torque_per_unit_mass', 'torque_int', \
                         'vort_avg', 'vphi_avg']
        asciiheader = 'variables = ' + ','.join(varlist1D)

    if ('xy' in outputs) or ('rphi' in outputs):
        varlist2D = ['sigma', 'sigma_i', 'sigma_pertb', 'rho_mid',
                     'pi', 'torque', 'torque_i', 'torque_pertb', 'realtorque', 'LR',
                     'vortensity', 'vort_grad']
        if 'xy' in outputs:
            tecOutXY = tecOutput.outputTec(varlist2D+['vx', 'vy'], grid, outDim='xy', output=True, suffix='xy')
        if 'rphi' in outputs:
            tecOutRPHI = tecOutput.outputTec(varlist2D+['vphi', 'vr'], grid, outDim='rphi',
                                             output=True, suffix='rphi')

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
        varlistTime = ['time', 'sax', 'rp', 'r_gap_sigma', 'r_gap_thresh',                       
                       'in_torque', 'out_torque',
                       'ILR_torque', 'OLR_torque', 'LR_torque',
                       'COR_torque', 'torque_tot', 'm_disk',
                       'fs_r', 'fp_r', 'fp_phi', 'fp_z',
                       'ftot_r', 'ftot_phi', 'ftot_z']
        timeDict = {k: [] for k in varlistTime}
        timeheader = 'variables = ' + ','.join(varlistTime)


    i_disk = np.where(grid.r > params.get('r_gap'))[0][0]
    if start > 0:
        try:
            data0 = dataReader.readData(0, legacy=False)
        except OSError:
            print "Error: 0th datafile is incomplete. Returning."
            sys.exit(-1)
        reduce0 = reduceCylData(grid, params, data0)
        sigma0, sigma1d = reduce0.sigma()
        sigma_disk = sigma1d[i_disk]

    for ndat in xrange(start, end, skip):
        data = dataReader.readData(ndat, legacy=False)

        process = reduceCylData(grid, params, data)
        if ndat == 0:
            data0 = data
            reduce0 = process
            sigma0, _ = reduce0.sigma()
            sigma0, sigma1d = reduce0.sigma()
            sigma_disk = sigma1d[i_disk]


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
        realtorque = process.zTorque(zavg=True, plot=False)
        torque_avg = azAverage(grid, realtorque)

        if ('xy' in outputs) or ('rphi' in outputs):
            torque = process.zTorque(zavg=True, plot=True)
            torque0 = process._zTorque(data0.rho, zavg=True, plot=True)

            output_dict = {'sigma': sigma}
            # noinspection PyUnboundLocalVariable
            output_dict['sigma_i'] = output_dict['sigma'] - sigma0
            output_dict['sigma_pertb'] = process.sigmaPertb()
            output_dict['rho_mid'] = data.rho[:, :, grid.nztot / 2]
            output_dict['pi'] = process.pi()
            output_dict['torque'] = torque
            output_dict['realtorque'] = realtorque
            output_dict['torque_pertb'] = (realtorque.transpose() - torque_avg).transpose()
            output_dict['torque_i'] = output_dict['torque'] - torque0
            output_dict['LR'] = process.lindbladRes()
            output_dict['vortensity'] = process.vortensity()
            output_dict['vort_grad'] = process.midplane_vortensity()

            if 'xy' in outputs:
                output_dict['vx'], output_dict['vy'] = process.calcVelocities()
                tecOutXY.writeCylindrical(ndat, output_dict, data.phiPlanet+0.5*np.pi)
            if 'rphi' in outputs:
                output_dict['vr'] = process.vr()
                output_dict['vphi'] = process.vPhi()
                tecOutRPHI.writeRPHI(ndat, output_dict, data.phiPlanet)

        if 'x' in outputs:
            # process 1D data
            vphi_avg = azAverage(grid, process.vPhi())
            vort_avg = azAverage(grid, process.midplane_vortensity())
            torque_dens, torque_int, torque_per_unit_mass = process.torque_density()
            r_norm = (grid.r - data.rp) / data.rp

            output_dict = {'r': grid.r,
                           'r_norm' : r_norm,
                           'sigma_avg': sigma1d,
                           'rho_mid': data.rho[:, 0, grid.nztot / 2],
                           'torque_avg': torque_avg,
                           'torque_density' : torque_dens,
                           'torque_int' : torque_int, 
                           'torque_per_unit_mass' : torque_per_unit_mass,
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
            timeDict['rp'].append(data.rp)
            r_gap_sigma, r_gap_thresh = process.disk_boundary(sigma_disk=sigma_disk)
            timeDict['r_gap_sigma'].append(r_gap_sigma)
            timeDict['r_gap_thresh'].append(r_gap_thresh)
            
            fs, fp, fi, ftot, ilr, olr, cor, lrtot, tot, in_torque, out_torque = process.resonance_torques()
            timeDict['in_torque'].append(in_torque)
            timeDict['out_torque'].append(out_torque)
            timeDict['ILR_torque'].append(ilr)
            timeDict['OLR_torque'].append(olr)
            timeDict['LR_torque'].append(lrtot)
            timeDict['COR_torque'].append(cor)
            timeDict['torque_tot'].append(tot)
            timeDict['m_disk'].append(process.m_disk())
            timeDict['fs_r'].append(fs[0])
            timeDict['fp_r'].append(fp[0])
            timeDict['fp_phi'].append(fp[1])
            timeDict['fp_z'].append(fp[2])
            timeDict['ftot_r'].append(ftot[0])
            timeDict['ftot_phi'].append(ftot[1])
            timeDict['ftot_z'].append(ftot[2])

            if ndat % 10 == 0:
                df = pd.DataFrame(timeDict, columns=varlistTime)
                with open('time.dat', 'w') as fout:
                    fout.write(timeheader + '\n')
                    df.to_csv(fout, index=False, header=False, sep='\t')
                
    if 'time' in outputs:
        df = pd.DataFrame(timeDict, columns=varlistTime)
        with open('time.dat', 'w') as fout:
            fout.write(timeheader + '\n')
            df.to_csv(fout, index=False, header=False, sep='\t')



