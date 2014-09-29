import numpy as np

__author__ = 'pslii'


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