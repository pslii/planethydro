#!/usr/local/lib/miniconda/bin/python
import numpy as np
import pandas as pd

from planetHydro.parseData.initialize import initialize
from planetHydro.tecOutput.tecOutput import _detectData, outputTec


"""
Utility routines
"""


def interp_r(grid, arr, ratio=4):
    """
    Interpolate a structured, rectilinear grid to a
    regular grid in the r and phi directions.
    """
    r, phi = grid.r, grid.phi

    from scipy.interpolate import interp2d

    newarr = interp2d(phi, r, arr)

    nr, nphi = arr.shape
    nmin = (r.max() - r.min()) / grid.dr.min()
    if nr * ratio < nmin:
        print "Warning: current grid ratio will cause loss of fidelity near star."
        print "         Increase ratio to resolve this issue."
        print nr * ratio, nmin

    r_newarr = np.linspace(r.min(), r.max(), num=nr * ratio)
    phi_newarr = np.linspace(phi.min(), phi.max(), num=nphi * ratio)
    return newarr(phi_newarr, r_newarr), r_newarr, phi_newarr


def azAverage(grid, arr):
    """
    Finds azimuthal (phi) average of an array
    """
    return integratePhi(grid, arr, phiavg=True)


def intRange(gridarr, gridmin=None, gridmax=None):
    if gridmin is None: gridmin = gridarr.min()
    if gridmax is None: gridmax = gridarr.max()
    assert (gridmax > gridmin)

    return np.where((gridarr >= gridmin) & (gridarr <= gridmax))[0]


def integrateR(grid, arr, rmin=None, rmax=None, ravg=False):
    """
    Array must be in 3D cylindrical coordinates
    """
    intrange = intRange(grid.r, rmin, rmax)

    r, dr = grid.r[intrange], grid.dr[intrange]
    rrange = r.max() - r.min()

    integral = np.einsum('ijk,i->jk', arr[intrange, :, :], r * dr)
    return integral / rrange if ravg else integral


def integratePhi(grid, arr, phimin=0.0, phimax=(2 * np.pi), phiavg=False):
    """
    Input array may be in polar or cylindrical coordinates
    """
    intrange = intRange(grid.phi, phimin, phimax)

    phirange, dphi = grid.phi[intrange].max() - grid.phi[intrange].min(), \
                     grid.dphi[intrange]

    if len(arr.shape) == 2:
        integral = np.dot(arr[:, intrange], dphi.transpose())
    else:
        integral = np.einsum('ijk,j->ik', arr[:, intrange, :], dphi)
    return integral / phirange if phiavg else integral


def integrateZ(grid, arr, zmin=None, zmax=None, zavg=False):
    """
    Computes avged vertical integral. Input array must be in
    3D cylindrical coordinates.
    sigma = flatten(grid, rho, zmin=-1,  zmax=1)
    """
    intrange = intRange(grid.z, zmin, zmax)

    H, dz = grid.z[intrange].max() - grid.z[intrange].min(), \
            grid.dz[intrange]

    assert (len(arr.shape) == 3)
    integral = np.einsum('ijk,k->ij', arr[:, :, intrange], dz)
    return integral / H if zavg else integral


def colMul(x, arr):
    """
    Multiplies every column in an array elementwise with a vector
    """
    return np.dot(np.diag(x), arr)


def centralDiff(grid, arr, arr_start=None, arr_end=None):
    """
    Takes derivative in r direction using central difference method.
    """
    dr = grid.dr

    if arr_start is None: arr_start = arr[0]
    if arr_end is None: arr_end = arr[-1]
    nxtot, nytot = arr.shape
    f1 = np.vstack((np.ones(nytot) * arr_start, arr[:-1, :]))
    f2 = np.vstack((arr[1:, :], np.ones(nytot) * arr_end))

    dx1 = np.hstack((dr[0], dr[:-1]))
    dx2 = np.hstack((dr[1:], dr[-1]))
    dx = 1.0 / (dx1 + dx2)

    return colMul(dx, f2 - f1)


"""
Data reduction
"""


class dataReduction:
    """
    Usage:
    processor = dataReduction(grid, params, data, metadata)
    rho_avg, rho_pertb = processor.rhoPertb()
    """

    def __init__(self, grid, params, data, metadata):
        self.ndat = metadata['ndat']
        self.grid = grid
        self.data = self.__processData(data)
        self.metadata = metadata
        self.params = params

        self.xn, self.yn, self.zn = metadata['xn'], \
                                    metadata['yn'], metadata['zn']
        self.un, self.vn, self.wn = metadata['up'], \
                                    metadata['vp'], metadata['wp']

        self.phi_pl = self.phiPlanet()
        self.omega_pl = self.omegaPlanet()

        self.omega_arr, self.domega_arr = None, None


    def __processData(self, data):
        newdata = {}
        for key, value in data.iteritems():
            if len(value.shape) > 2:
                newdata[key] = self._getMidplane(value)
        return newdata

    @staticmethod
    def _getMidplane(arr):
        _, _, nztot = arr.shape
        return arr[:, :, nztot / 2]

    def rhovr(self):
        return self.data['rho'] * self.data['u']

    def rhovphi(self):
        return self.data['rho'] * self.data['v']

    def rhov(self):
        return (self.rhovr() ** 2 + self.rhovphi() ** 2) ** 0.5

    def vPertb(self):
        """
        Returns:
        2D array: v_phi-v_Kep
        2D array: v_phi-v_planet
        """
        GM = self.params['GM']
        v_kep = np.sqrt(GM / self.grid.r)
        v_pl = self.omega_pl * self.grid.r
        subtractkep = lambda arr: arr - v_kep
        subtractpl = lambda arr: arr - v_pl
        return np.apply_along_axis(subtractkep, 0, self.data['v']), \
               np.apply_along_axis(subtractpl, 0, self.data['v'])

    def rhoPertb(self):
        """
        Returns:
        1D array: rho_avg
        2D array: rho-rho_avg
        """
        rho_avg = azAverage(self.grid, self.data['rho'])
        rho_pertb = self.data['rho'] - rho_avg.reshape((self.grid.nxtot, 1))
        return rho_avg, rho_pertb

    def calcVelocities(self, vr=None, vphi=None, phi=None):
        """
        Converts vr, vphi into vx, vy
        """
        if phi is None:
            phi = self.phi_pl
        if vr is None:
            vr = self.data['u']
        if vphi is None:
            vphi = self.data['v']
        phi_grid = (self.grid.phi - phi + np.pi) % (2 * np.pi)
        vx = colMul(np.cos(phi_grid), vr.transpose()) - \
             colMul(np.sin(phi_grid), vphi.transpose())
        vy = colMul(np.sin(phi_grid), vr.transpose()) + \
             colMul(np.cos(phi_grid), vphi.transpose())
        return vx.transpose(), vy.transpose()

    def oortB(self):
        """
        Computes Oort's second constant:
        B = 1/(2r) d(r^2 \Omega)/dr
        = r/2 (3 v_phi + r dv_phi/dr)
        """
        vphi = self.data['v']
        r_in = (self.grid.r_edge[1] + self.grid.r_edge[2]) / 2.0
        r_out = (self.grid.r_edge[-1] + self.grid.r_edge[-2]) / 2.0
        dvphidr = centralDiff(self.grid, vphi,
                              arr_start=np.sqrt(1.0 / r_in),
                              arr_end=np.sqrt(1.0 / r_out))
        terms = 3 * vphi + colMul(self.grid.r, dvphidr)
        return colMul(self.grid.r / 2.0, terms)

    def vortensity(self):
        sigma, vphi = self.data['rho'], self.data['v']
        r = self.grid.r
        logB = np.log(self.oortB())
        logSigma = np.log(sigma)

        dlogB = colMul(r, centralDiff(self.grid, logB))
        dlogSigma = colMul(r, centralDiff(self.grid, logSigma))
        return sigma * dlogB, sigma * dlogSigma, sigma * (dlogSigma - dlogB)

    def omega(self):
        if self.omega_arr is None:
            GM = self.params['GM']
            omega = colMul(self.grid.r, self.data['v'])
            r_in = (self.grid.r_edge[1] + self.grid.r_edge[2]) / 2.0
            r_out = (self.grid.r_edge[-1] + self.grid.r_edge[-2]) / 2.0
            domega = centralDiff(self.grid, omega,
                                 arr_start=-1.5 * np.sqrt(GM / r_in ** 5),
                                 arr_end=-1.5 * np.sqrt(GM / r_out ** 5))
            self.omega_arr, self.domega_arr = omega, domega
        return self.omega_arr, self.domega_arr

    def kappa2(self):
        """
        Epicyclic frequency squared
        """
        omega, domega = self.omega()
        vphi = self.data['v']
        kappa2 = 4.0 * omega ** 2 + 2.0 * vphi * domega
        return kappa2

    def lindbladRes(self):
        lr = np.zeros((self.grid.nxtot, self.grid.nytot))
        rp = np.sqrt(self.xn ** 2 + self.yn ** 2)
        for m in np.arange(1, 6):
            lr_in, lr_out = self.grid.r.min(), self.grid.r.max()

            lr_out = ((1.0 + 1.0 / m) ** (2.0 / 3.0)) * rp
            if m != 1: lr_in = ((1.0 - 1.0 / m) ** (2.0 / 3.0)) * rp
            loc = np.where((self.grid.r >= lr_in) & (self.grid.r <= lr_out))
            lr[loc, :] = m

        dist = np.sqrt((self.grid.x - self.xn) ** 2 + (self.grid.y - self.yn) ** 2)
        lr[np.where(dist < .1)] = -1.0
        return lr

    def cs2(self):
        """
        Sound speed squared
        """
        gamma = self.params['gam']
        return gamma * self.data['p'] / self.data['rho']

    def torque_c(self):
        GM_p = self.params['GM_p']
        rp = np.sqrt(self.xn ** 2 + self.yn ** 2 + self.zn ** 2)
        r = self.grid.r
        sigma = self.data['rho']
        B = self.oortB()
        omega, domega = self.omega()
        _, _, vortensity = self.vortensity()
        m = 1

        Phi_p = GM_p / (r - rp)
        gamma_c = colMul((m * (Phi_p * np.pi) ** 2 / (2.0 * r)), (vortensity / (B * domega)))
        return gamma_c

    def torques(self, smoothing=True, pos=False):
        """
        Computes torque on planet from the disk
        """
        return self.torque_sigma(self.data['rho'],
                                 smoothing=smoothing, pos=pos)

    def torque_sigma(self, sigma, smoothing=True, pos=False):
        """
        Computes torque on planet given a density distribution
        """
        GM, GM_p = self.params['GM'], self.params['GM_p']
        xn, yn, zn = self.xn, self.yn, self.zn
        r_dist = np.array([xn - self.grid.x, yn - self.grid.y])
        if smoothing:
            t_disk, r_gap = self.params['temp_d'], self.params['r_d']
            asp_ratio = np.sqrt(t_disk / (GM / r_gap))
            h_disk = asp_ratio * r_gap
            eps = 0.1 * h_disk
            r_mag = np.sqrt(r_dist[0, :, :] ** 2 + r_dist[1, :, :] ** 2 + eps ** 2)
        else:
            r_mag = np.sqrt(r_dist[0, :, :] ** 2 + r_dist[1, :, :] ** 2)
        # torque = np.zeros((self.params['nxtot'], self.params['nytot']))
        force = -GM_p * sigma / r_mag ** 2
        cross = (yn * self.grid.x - xn * self.grid.y)
        if pos:
            # not physical, for ease of plotting only
            cross = -np.abs(-cross)

        torque = cross / r_mag * force
        return force, torque

    def planetPos(self):
        return np.sqrt(self.xn ** 2 + self.yn ** 2 + self.zn ** 2)

    def hillRadius(self):
        rp = np.sqrt(self.xn ** 2 + self.yn ** 2 + self.zn ** 2)
        GM, GM_p = self.params['GM'], self.params['GM_p']
        return rp * (GM_p / (3.0 * GM)) ** (1.0 / 3.0)

    def __rotateCoords(self, frame='planet'):
        xn, yn, zn = self.xn, self.yn, self.zn
        up, vp, wp = self.metadata['up'], \
                     self.metadata['vp'], \
                     self.metadata['wp']

        if frame == 'disk':
            o_syst = 0.0
            phi_syst = 0.0
        elif frame == 'planet':
            o_syst = -self.omega_pl
            phi_syst = self.phi_pl
        elif frame == 'inertial':
            GM, time, r0 = self.metadata['GM'], \
                           self.metadata['time'], \
                           self.params['r_d']
            o_syst = -np.sqrt(GM / r0 ** 3)
            phi_syst = o_syst * time
        else:
            assert False
        xp = xn * np.cos(phi_syst) + yn * np.sin(phi_syst)
        yp = xn * np.cos(phi_syst) + yn * np.sin(phi_syst)
        zp = xn * np.cos(phi_syst) + yn * np.sin(phi_syst)

        ui = up - o_syst * yn
        vi = vp + o_syst * xn
        up = ui * np.cos(phi_syst) + vi * np.sin(phi_syst)
        vp = vi * np.cos(phi_syst) - ui * np.sin(phi_syst)
        wp = wp
        return (xp, yp, zp), (up, vp, wp)

    def omegaPlanet(self):
        """
        Compute omega of planet
        """
        xn, yn, zn = self.xn, self.yn, self.zn
        un, vn, wn = self.un, self.vn, self.wn
        return (vn * xn - un * yn) / (xn ** 2 + yn ** 2)

    def phiPlanet(self):
        """
        Compute phi of planet
        """
        xn, yn, zn = self.xn, self.yn, self.zn
        return np.arctan2(yn, xn) + 0.5 * np.pi

    def findPlanet(self):
        """
        Given xyz position of planet, calculates phi position
        and identifies j coordinate.
        """
        xn, yn, zn = self.xn, self.yn, self.zn
        phi_p = (np.arctan2(yn, xn) + 2.0 * np.pi) % (2.0 * np.pi)
        j = 0
        for j in range(self.grid.nytot):
            if self.grid.phi_edge[j] <= phi_p <= self.grid.phi_edge[j + 1]: break
        return j

    def setPlanetFrame(self):
        xn = self.xn, yn = self.yn, zn = self.zn
        un = self.metadata['up']
        vn = self.metadata['vp']
        wn = self.metadata['wp']
        o_syst = -(vn * xn - un * yn) / (xn ** 2 + yn ** 2)
        phi_p = np.arctan2(yn, xn)
        phi = (self.grid.phi_edge[2:-2] - phi_p + np.pi) % (2 * np.pi)
        indices = np.argsort(phi[np.where(phi < np.max(phi))[0]])
        phi = np.sort(phi)
        return phi, indices, o_syst


"""
Bulk processing options

def reduceData((ndat, datReader, grid, params)):
    print "Now reading", ndat
    data, metadata = datReader.readData(ndat)
    dat = dataReduction(grid, params, data, metadata)
    
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
    process = dataReduction(grid, params, data, metadata)

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
        makeTecplot()
        os.chdir('../')


def makeTecplot(path='.', n_start=1, n_skip=1):
    params, grid, datReader = initialize(path=path)

    # detect number of files to be processed
    start, end, ndats = _detectData(path=path, n_start=n_start)

    varlist = ['rho', 'p', 's', 'vr', 'vphi', 'vKep', 'vx', 'vy',
               'rho_i', 'p_i', 'v_i',
               'omega', 'LR', 'vortensity',
               'rhoPertb', 'rhovr', 'torque', 'torque_i']
    output = outputTec(varlist, grid, output=True, path=path)
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

        processor = dataReduction(grid, params, data, metadata)

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
        output.writeNDat(n_dat, data, processor.phiPlanet())


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