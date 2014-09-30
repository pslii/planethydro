#!/usr/local/lib/miniconda/bin/python
import pandas as pd

from .. import parseData
from planetHydro.reduceData.reduceData import reduceData
from .utility import azAverage
from ..tecOutput import tecOutput

"""
Data reduction
"""

"""
Bulk processing options

def reduceData((ndat, datReader, grid, params)):
    print "Now reading", ndat
    data, metadata = datReader.readData(ndat)
    dat = reduceData(grid, params, data, metadata)
    
    logB, logSigma, vort_grad = vort = dat.vortensity()
    rho_avg, rho_pertb = dat.rhoPertb()
    vx, vy = dat.calcVelocities()
    phi_edge, indx, o_syst = dat.setPlanetFrame()
    vphi = data['v']-o_syst*np.tile(grid.r, (grid.nytot,1)).transpose()
    force, torque = dat.torques()
    
    maparr = lambda arr : np.reshape(arr[:, indx],
                                     (grid.nxtot, grid.nytot, 1), order='F')    
    gridToVTK(str(ndat).zfill(4),
        phi_edge, grid.r_edge[2:-2], np.arange(1), 
        cellData = {"density": maparr(data['rho']),
                    "pressure": maparr(data['p']),            
                    "entropy": maparr(data['s']),
                    "vr": maparr(data['u']), "vphi": maparr(vphi),
                    "vx": maparr(vx), "vy": maparr(vy),
                    "vortensity": maparr(vort_grad),
                    "torque": maparr(torque),
                    "rho_pertb": maparr(rho_pertb)})
    
def parallelProcess():
    params, grid, datReader = parseData.initialize()
    
    # detect files to be processed
    start = 0
    while isfile(str(start).zfill(4)+'.vtr'):
        start += 1
    end = start
    while (isfile(str(end).zfill(4)+'dat') and
           not isfile(str(end).zfill(4)+'.vtr')):
        end += 1
    num = end-start
    
    # process in parallel
    print "Processing {2} files from {0} to {1} in parallel".format(start, end-1, num)
    inputs = zip(range(start, end),
                 (datReader,)*num,
                 (grid,)*num,
                 (params,)*num)
    pool = Pool(4)
    pool.map(reduceData, inputs)

def main():
    parallelProcess()
    
if __name__=="__main__":
    main()
    
def plotStreamlines(minlength=0.5):
    params, grid, datReader = parseData.initialize()
    data, metadata = datReader.readData(50)
    process = reduceData(grid, params, data, metadata)

    phi, indices, _ = process.setPlanetFrame()
    
    vKep, vPl = process.vPertb()
    vr, vphi = data['u'], data['v']
    vr_interp, r_int, phi_int = interp_r(grid, vr[:, indices], ratio=6)
    vphi_interp, _, _ = interp_r(grid, vPl[:, indices], ratio=6)

    # Restrict range
    lim_range = np.where((r_int>=1.4) & (r_int<=1.8))[0]
    plt.clf()
    plt.streamplot(phi_int, r_int[lim_range],
                   vphi_interp[lim_range,:],
                   vr_interp[lim_range,:], minlength=minlength, density=2)
    plt.show()
    return grid.r, vPl
"""


def processAll(path='.'):
    """
    Detects all subdirectories in given directory and
    traverses it to process the data within
    """
    import os
    from os.path import isfile

    subfolders = os.walk(path).next()[1]
    print subfolders
    for subfolder in subfolders:
        subpath = path + '/' + subfolder + '/'
        if not (isfile(subpath + 'grid') &
                    isfile(subpath + 'param.inc') &
                    isfile(subpath + '0000dat')): continue
        print "Processing:", subfolder
        os.chdir(subfolder)
        processData()
        os.chdir('../')


def processData(path='.', n_start=1, n_skip=1):
    params, grid, datReader = parseData.initialize(path=path)
    start, end, ndats = tecOutput.detectData(path=path, n_start=n_start)

    varlist = ['rho', 'p', 's', 'vr', 'vphi', 'vKep', 'vx', 'vy',
               'rho_i', 'p_i', 'v_i',
               'omega', 'LR', 'vortensity',
               'rhoPertb', 'rhovr', 'torque', 'torque_i']
    output = tecOutput.outputTec(varlist, grid, output=True, path=path)
    varlist1d = ['r', 'sigma_avg', 'pi_avg',
                 'vphi_avg', 'torque_avg', 'torque_int']
    asciiheader = 'variables = ' + ', '.join(varlist1d)

    datrange = range(start, end, n_skip)
    if not (0 in datrange):
        datrange.insert(0, 0)
    for n_dat in datrange:
        data, metadata = datReader.readData(n_dat)
        if n_dat == 0:
            data0, metadata0 = data, metadata

        processor = reduceData(grid, params, data, metadata)

        # output 2D data
        data['rho_i'] = data['rho'] - data0['rho']
        data['p_i'] = data['p'] - data0['p']
        data['v_i'] = data['v'] - data0['v']

        data['omega'], _ = processor.omega()
        data['LR'] = processor.lindbladRes()
        _, _, vortensity = processor.vortensity()
        data['vortensity'] = vortensity

        rho_avg, data['rhoPertb'] = processor.rhoPertb()
        data['rhovr'] = processor.rhovr()
        _, data['torque'] = processor.torques(pos=True)

        _, torque = processor.torques()
        _, torque0 = processor.torque_sigma(data0['rho'])
        data['torque_i'] = torque - torque0

        vKep, vPl = processor.vPertb()
        data['vr'], data['vphi'] = data['u'], vPl
        data['vKep'] = vKep
        data['vx'], data['vy'] = processor.calcVelocities(data['u'], vPl)
        output.writeCylindrical(n_dat, data, processor.phiPlanet())


        # output 1D averages
        _, torque = processor.torques(pos=False)
        data1d = {'r': grid.r,
                  'sigma_avg': rho_avg,
                  'pi_avg': azAverage(grid, data['p']),
                  'vphi_avg': azAverage(grid, data['v']),
                  'torque_avg': azAverage(grid, data['torque']),
                  'torque_int': azAverage(grid, torque)}
        df = pd.DataFrame(data1d, columns=varlist1d)

        fname = str(n_dat).zfill(4) + 'cut.dat'
        with open(fname, 'w') as fout:
            fout.write(asciiheader + '\n')
            df.to_csv(fout, index=False, header=False, sep='\t')