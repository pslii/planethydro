import numpy as np
import struct
import time

__author__ = 'pslii'


class dataReader:
    VARLIST_DEFAULT = ['rho', 'p', 's',
                       'hr', 'hf', 'hz',
                       'u', 'v', 'w',
                       'br', 'bphi', 'bz',
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
        self.unpackList, self.fmtlist, self.varlist = \
            self._computeUnpack(fmtlist, varlist)

    def __call__(self, ndat):
        return self.readData(ndat)

    def _computeUnpack(self, fmtlist, varlist):
        """
        Computes start and end indices of arrays/values
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
        return zip(starts, ends, formats), fmtlist_out, varlist_out

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

    def _readVar(self, binary, (start, end, fmt)):
        """
        Unpacks an individual variable from the Fortran binary file.
        """
        unpack = struct.unpack(fmt, binary[start:end])
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

    def readData(self, ndat, suffix='dat', n_digits=4, verbose=False, legacy=True):
        """
        Parses an unformatted Fortran binary file
        """
        t0 = time.time()
        assert (isinstance(ndat, int))
        fname = str(ndat).zfill(n_digits) + suffix
        print "Reading {0}...".format(fname)

        data, metadata = {}, {}
        with open(self.path + '/' + fname, 'rb') as datfile:
            binary = datfile.read()
            for unpack, varname in zip(self.unpackList, self.varlist):
                if verbose: print "Parsing {0}".format(varname)
                var = self._readVar(binary, unpack)
                if isinstance(var, np.ndarray) and len(var) > 1:
                    data[varname] = var
                else:
                    metadata[varname] = var[0]
        t1 = time.time()
        if verbose: print "Read time: {0}".format(t1 - t0)
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
    def phiPlanet(self):
        return np.arctan2(self.yp, self.xp) + 0.5 * np.pi

    @property
    def omegaPlanet(self):
        return (self.vp * self.xp - self.up * self.yp) / (self.xp ** 2 + self.yp ** 2)


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