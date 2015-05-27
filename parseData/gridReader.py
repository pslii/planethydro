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
            self.rPlot = np.vstack((self.r_edge[2:-3],) * self.nztot).transpose()
        else:
            self.r_sph = self.r
            self.rPlot = self.r_edge[2:-3]

        self.zPlot = np.vstack((self.z_edge[2:-3],) * self.nxtot)

        # TODO: find more efficient method to generate these
        self.r2D, self.phi2D, self.z2D = self._grid2d(self.r, self.phi, self.z)
        self.dr2D, self.dphi2D, self.dz2D = self._grid2d(self.dr, self.dphi, self.dz)
        if self.ndims == 3:
            self.r3D, self.phi3D, self.z3D = self._grid3d(self.r2D, self.phi2D, self.z)
            self.dr3D, self.dphi3D, self.dz3D = self._grid3d(self.dr2D, self.dphi2D, self.dz)
            self.dV3D = self.r3D * self.dr3D * self.dphi3D * self.dz3D
        else:
            self.r3D, self.phi3D, self.z3D = None, None, None
            self.dr3D, self.dphi3D, self.dz3D = None, None, None
            self.dV3D = None

        # optional variables for processing
        self._jacobian = None

    def _grid2d(self, r, phi, z):
        r2D = np.vstack((r,) * self.nytot).transpose()  # rphi
        phi2D = np.vstack((phi,) * self.nxtot)  # rphi
        z2D = np.vstack((z[:, np.newaxis].transpose(),) * self.nxtot)  # rz
        return r2D, phi2D, z2D

    def _grid3d(self, r2D, phi2D, z):
        r3D = np.dstack((r2D,) * self.nztot)
        phi3D = np.dstack((phi2D,) * self.nztot)
        zZPhi = np.vstack((z[:, np.newaxis].transpose(),) * self.nytot).transpose()  # zphi
        z3D = np.dstack((zZPhi,) * self.nxtot).transpose()
        return r3D, phi3D, z3D

    def xDist(self, xp):
        return self.x - xp

    def yDist(self, yp):
        return self.y - yp

    def zDist(self, zp):
        return self.z - zp

    def plotCoordsRZ(self):
        return self.rPlot, self.zPlot

    def integrate(self, arr):
        """
        Given a scalar array with the same dimensions as the grid, performs a volume integral 
        over the entire array and returns a scalar.
        """
        assert arr.shape == self.shape, "Error: input array must have same dimensions as the grid."
        if self._jacobian is None:
            self._jacobian = (self.r * self.dr)[:, np.newaxis, np.newaxis] * \
                self.dphi[np.newaxis, :, np.newaxis] * \
                self.dz[np.newaxis, np.newaxis, :]            
        return (arr * self._jacobian).sum()

    def meshCoords(self, phi_p):
        r, phi = self.r_edge[2:-2], self.phi_edge[2:-2]
        phi_grid, r_grid = np.meshgrid(phi, r)
        x0, y0 = r_grid*np.cos(phi_grid), r_grid*np.sin(phi_grid)

        x, y = x0 * np.cos(phi_p) + y0 * np.sin(phi_p), \
               y0 * np.cos(phi_p) - x0 * np.sin(phi_p)
        return x, y

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
