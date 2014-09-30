import numpy as np
import struct

__author__ = 'pslii'


class gridReader:
    """
    Reads in and parses binary grid file for 2D polar and 3D cyl coordiantes
    """

    def __init__(self, params, fname='grid', path='.'):
        if ('nztot' in params.keys()) & ('nlayers_k' in params.keys()):
            self.ndims = 3
        else:
            self.ndims = 2

        self.nxtot = params['nxtot']
        self.nytot = params['nytot']
        self.nlayers_i = params['nlayers_i']
        self.nlayers_j = params['nlayers_j']

        if self.ndims == 3:
            self.nztot = params['nztot']
            self.nlayers_k = params['nlayers_k']
            self.shape = (self.nxtot, self.nytot, self.nztot)
        else:
            self.nztot = None
            self.nlayers_k = None
            self.shape = (self.nxtot, self.nytot)

        self.r_edge, self.phi_edge, self.z_edge = self._readGrid(fname, path)
        self.r, self.phi, self.z = self._computeCellCoordinates()
        self.dr, self.dphi, self.dz = self._computeCellSize()

        self.xy_area, self.xz_area, self.vol = self._computeCellVol()

        self.x = np.outer(self.r, np.cos(self.phi))
        self.y = np.outer(self.r, np.sin(self.phi))
        self.x_edge = np.outer(self.r_edge, np.cos(self.phi_edge))
        self.y_edge = np.outer(self.r_edge, np.sin(self.phi_edge))

        # spherical radius
        if self.ndims == 3:
            self.r_sph = np.sqrt(np.tile(self.r ** 2, (self.nztot, 1)) + \
                                 np.tile(self.z ** 2, (self.nxtot, 1)).transpose())
        else:
            self.r_sph = self.r

        self.rPlot = np.vstack((self.r,) * self.nztot).transpose()
        self.zPlot = np.vstack((self.z,) * self.nxtot)

    def xDist(self, xp):
        return self.x - xp

    def yDist(self, yp):
        return self.y - yp

    def zDist(self, zp):
        return self.z - zp

    def plotCoordsRZ(self):
        return self.rPlot, self.zPlot

    def plotCoords(self, phi, ndims=2):
        (x_edge, y_edge), z_edge = self._rotate(self.x_edge[2:-3, 2:-2],
                                                self.y_edge[2:-3, 2:-2], phi), \
                                   self.z_edge[2:-3]
        if ndims == 2:
            shape = (self.nxtot, self.nytot + 1)
            assert x_edge.shape == shape
            assert y_edge.shape == shape
            return x_edge, y_edge, None
        elif ndims == 3:
            x_edge = np.dstack((x_edge,) * self.nztot)
            y_edge = np.dstack((y_edge,) * self.nztot)
            z_edge = np.tile(z_edge, (self.nxtot, self.nytot + 1, 1))

            shape = (self.nxtot, self.nytot + 1, self.nztot)
            assert x_edge.shape == shape
            assert y_edge.shape == shape
            assert z_edge.shape == shape
            return x_edge, y_edge, z_edge


    def distance(self, xp, yp, zp):
        if self.ndims == 2:
            dist2 = self.xDist(xp) ** 2 + self.yDist(yp) ** 2
        elif self.ndims == 3:
            dist2 = np.dstack((self.xDist(xp) ** 2 + self.yDist(yp) ** 2,) * self.nztot) + \
                    np.tile(self.zDist(zp) ** 2, (self.nxtot, self.nytot, 1))
        else:
            assert False
        assert dist2.shape == self.shape
        return np.sqrt(dist2)

    @staticmethod
    def _rotate(x, y, phi):
        return x * np.cos(phi) + y * np.sin(phi), y * np.cos(phi) - x * np.sin(phi)

    def frame(self, phi):
        """
        Rotates coordinate system by phi
        """
        return self._rotate(self.x, self.y, phi), \
               self._rotate(self.x_edge, self.y_edge, phi)

    def _readGrid(self, fname, path):
        with open(path + '/' + fname, 'rb') as gridfile:
            binary = gridfile.read()
            start, num = 0, self.nxtot + 5
            r_edge = np.array(struct.unpack('<' + 'd' * num,
                                            binary[:num * 8]))
            start, num = self.nxtot + 5, self.nytot + 5
            phi_edge = np.array(struct.unpack('<' + 'd' * num,
                                              binary[start * 8:(start + num) * 8]))
            if self.ndims == 3:
                start, num = self.nytot + 5 + self.nxtot + 5, self.nztot + 5
                z_edge = np.array(struct.unpack('<' + 'd' * num,
                                                binary[-num * 8:]))
            else:
                z_edge = np.array([0.0])
        return r_edge, phi_edge, z_edge

    def _computeCellCoordinates(self):
        r_edge, phi_edge, z_edge = self.r_edge, self.phi_edge, self.z_edge
        r = (np.roll(r_edge, -1) + r_edge)[2:-3] / 2.0
        phi = (np.roll(phi_edge, -1) + phi_edge)[2:-3] / 2.0
        if self.ndims == 3:
            z = (np.roll(z_edge, -1) + z_edge)[2:-3] / 2.0
        else:
            z = np.array([0.0])
        return r, phi, z

    def _computeCellSize(self):
        r_edge, phi_edge, z_edge = self.r_edge, self.phi_edge, self.z_edge
        dr = (np.roll(self.r_edge, -1) - self.r_edge)[2:-3]
        dphi = (np.roll(self.phi_edge, -1) - self.phi_edge)[2:-3]
        if self.ndims == 3:
            dz = (np.roll(self.z_edge, -1) - self.z_edge)[2:-3]
        else:
            dz = np.array([0.0])
        return dr, dphi, dz

    def _computeCellVol(self):
        xy_area = np.outer(self.r * self.dr, self.dphi)
        xz_area = np.outer(self.dr, self.dz)
        # einsum black magic
        vol = np.einsum('i,j,k->ijk', self.dr * self.r, self.dphi, self.dz)
        return xy_area, xz_area, vol