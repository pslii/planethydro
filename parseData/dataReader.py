import numpy as np
import struct
import time
import matplotlib.pyplot as plt

__author__ = 'pslii'


class dataReader:
    """
    Parses Fortran binaries into Numpy arrays.
    """

    VARLIST_DEFAULT = ['rho', 'p', 's',
                       'hr', 'hf', 'hz',
                       'u', 'v', 'w',
                       'br', 'bphi', 'bz',
                       'time', 'ndat', 'it', 'tau', 'zdisk',
                       'sax', 'incl', 'ecc',
                       'xn', 'yn', 'zn',
                       'up', 'vp', 'wp']
    VARLIST_HYDRO = ['rho', 'p', 's',
                     '_', '_', '_',
                     'u', 'v', 'w',
                     '_', '_', '_',
                     'time', 'ndat', 'it', 'tau', 'zdisk',
                     'sax', 'incl', 'ecc',
                     'xn', 'yn', 'zn',
                     'up', 'vp', 'wp']

    FMTLIST_DEFAULT = ['darr', 'darr', 'darr', 'darr',
                       'darr', 'darr', 'darr', 'darr',
                       'darr', 'darr', 'darr', 'darr',
                       'd', 'i', 'i', 'd', 'd',
                       'd', 'd', 'd',
                       'd', 'd', 'd',
                       'd', 'd', 'd']

    def __init__(self, grid,
                 fmtlist=FMTLIST_DEFAULT,
                 varlist=VARLIST_DEFAULT, path='.'):
        self.path = path
        self.grid = grid
        self.nx = grid.nxtot / grid.nlayers_i
        self.ny = grid.nytot / grid.nlayers_j

        self.nlayers_ij = grid.nlayers_i * grid.nlayers_j
        if grid.ndims == 3:
            self.nz = grid.nztot / grid.nlayers_k
            self.nblocks = self.nlayers_ij * grid.nlayers_k
            self.block_shape = (self.nblocks, self.nz + 4, self.ny + 4, self.nx + 4)
        else:
            self.nz = 0
            self.nblocks = self.nlayers_ij
            self.block_shape = (self.nblocks, self.ny + 4, self.nx + 4)
        self.block_size = np.prod(self.block_shape)

        # precompute binary indices
        assert (len(fmtlist) == len(varlist))
        self.unpackList, self.fmtlist, self.varlist, self.fileSize = \
            self._computeUnpack(fmtlist, varlist)

    def __call__(self, ndat):
        return self.readData(ndat)

    def _progressbar(self, fname, progress, total):
        """
        Outputs a progressbar.
        Reading 0039dat: [==================================================] 100%

        Note: need to set -u flag in python call for this to work properly
        :param progress:
        :param total:
        :return:
        """
        percentage = int(progress * 100.0 / total)
        percentage_bar = ('=' * (percentage / 2)).ljust(50)
        print '\rReading {0}: [{1}] {2}%'.format(fname, percentage_bar, percentage),

    def _computeUnpack(self, fmtlist, varlist):
        """
        Computes start and end indices of arrays/values in the binary file.
        """
        fmtlist_out, varlist_out = [], []
        formats, starts, ends = [], [], []
        start, end = 0, 0

        for fmt, var, i in zip(fmtlist, varlist, range(len(fmtlist))):
            start = end
            block_size = self.block_size if ('arr' == fmt[1:]) else 1
            end += block_size * self.getByteSize(fmt[0])

            if not (var == '' or var == '_'):  # strip out unused arrays
                fmtlist_out.append(fmt)
                varlist_out.append(var)
                formats.append('<' + fmt[0] * block_size)
                starts.append(start)
                ends.append(end)
        sizes = np.array(ends) - np.array(starts)
        return zip(starts, sizes, formats), fmtlist_out, varlist_out, sizes.sum()

    @staticmethod
    def getByteSize(c):
        assert (len(c) == 1)
        if c in 'd':
            return 8
        elif c in 'fil':
            return 4
        elif c in 'c?':
            return 1
        else:
            print "Error: invalid character format"
            assert False

    def _repack(self, fmt, binary):
        unpack = struct.unpack(fmt, binary)
        if len(unpack) > 1:
            var = np.empty(self.grid.shape)
            unpack = np.reshape(unpack, self.block_shape)
            if self.grid.ndims == 2:
                for l in range(self.nblocks):
                    li = l % self.grid.nlayers_i
                    lj = l / self.grid.nlayers_i
                    var[li * self.nx:(li + 1) * self.nx, \
                    lj * self.ny:(lj + 1) * self.ny] = \
                        unpack[l,
                        2:2 + self.ny,
                        2:2 + self.nx].transpose()
            elif self.grid.ndims == 3:
                for l in range(self.nblocks):
                    li = l % self.grid.nlayers_i
                    lj = (l % self.nlayers_ij) / self.grid.nlayers_i
                    lk = l / self.nlayers_ij
                    var[li * self.nx:(li + 1) * self.nx,
                    lj * self.ny:(lj + 1) * self.ny,
                    lk * self.nz:(lk + 1) * self.nz] = \
                        unpack[l,
                        2:2 + self.nz,
                        2:2 + self.ny,
                        2:2 + self.nx].transpose()
        else:
            var = unpack
        return var

    def readData(self, ndat, suffix="dat", n_digits=4, legacy=False):
        t0 = time.time()
        fname = str(ndat).zfill(n_digits) + suffix
        data, metadata = {}, {}
        bytesRead = 0
        self._progressbar(fname, bytesRead, self.fileSize)

        with open(self.path + '/' + fname, 'rb') as datfile:
            for ((start, size, fmt), varname) in zip(self.unpackList, self.varlist):
                datfile.seek(start)
                binary = datfile.read(size)
                var = self._repack(fmt, binary)

                if isinstance(var, np.ndarray) and len(var) > 1:
                    assert var.shape == self.grid.shape
                    data[varname] = var
                else:
                    metadata[varname] = var[0]
                bytesRead += size
                self._progressbar(fname, bytesRead, self.fileSize)
        t1 = time.time()
        print " {:.2f} s".format(t1-t0)

        if legacy:
            return data, metadata
        else:
            return SimData(self.grid, data, metadata)

class SimData(object):
    def __init__(self, grid, data_dict, metadata_dict):
        """
        Wrapper for data dictionaries.

        :type data_dict: dict
        :type metadata_dict: dict
        :param data_dict:
        :param metadata_dict:
        """
        self.shape = grid.shape
        for value in data_dict.values():
            assert self.shape == value.shape, "Error: shape mismatch."

        self.rho, self.p, self.s = data_dict.get('rho'), data_dict.get('p'), data_dict.get('s')
        self.temp = self.p / self.rho
        self.u, self.v, self.w = data_dict.get('u'), data_dict.get('v'), data_dict.get('w')
        self.data = data_dict

        self.xp, self.yp, self.zp = metadata_dict.get('xn'), metadata_dict.get('yn'), metadata_dict.get('zn')
        self.up, self.vp, self.wp = metadata_dict.get('up'), metadata_dict.get('vp'), metadata_dict.get('wp')
        self.time, self.ndat = metadata_dict.get('time'), metadata_dict.get('ndat')
        self.metadata = metadata_dict

    def keys(self):
        return self.metadata.keys() + self.data.keys()

    def get(self, var, default=None):
        if var in self.data.keys():
            return self.data[var]
        elif var in self.metadata.keys():
            return self.metadata[var]
        else:
            print "No variable named {0} in the data.".format(var)
            return default

    def addArray(self, key, value):
        print "Adding {0} to data dictionary".format(key)
        assert value.shape == self.shape, "Error: array must have shape {0}".format(self.shape)
        self.data[key] = value

    @property
    def rp(self):
        return np.sqrt(self.xp**2 + self.yp**2 + self.zp**2)

    @property
    def phiPlanet(self):
        return np.arctan2(self.yp, self.xp) + 0.5 * np.pi

    @property
    def omegaPlanet(self):
        return (self.vp * self.xp - self.up * self.yp) / (self.xp ** 2 + self.yp ** 2)


    def planetPos(self, frame='planet'):
        xn, yn, zn = self.xp, self.yp, self.zp
        up, vp, wp = self.up, self.vp, self.wp

        if frame == 'disk':
            o_syst = 0.0
            phi_syst = 0.0
        elif frame == 'planet':
            o_syst = self.omegaPlanet
            phi_syst = self.phiPlanet
        elif frame == 'inertial':
            GM, time, r0 = self.get('gm'), self.get('time'), self.get('r_gap')
            o_syst = -np.sqrt(GM / r0**3)
            phi_syst = o_syst * time
        else:
            print "Invalid frame selected."
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


def _sandbox(evallist):
    """
    Takes in list of variables, evaluates them and returns a dict
    """
    exec (';'.join(evallist))

    _dict = {}
    _varlist = []
    for _var in evallist:
        _var = _var.split('=')

        _varname = _var[0].strip()
        _varlist.append(_varname)
        _dict[_varname.lower()] = eval(_varname)
    return _dict, _varlist