import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

__author__ = 'pslii'


def polar_plot(grid, params, rp, phi_p, array, save=None):
    # Generate the mesh
    x, y = grid.meshCoords(phi_p)

    # Plot
    fig, ax = plt.subplots(figsize=(6,6), dpi=300, frameon=False, tight_layout=True)

    normalize = colors.PowerNorm(2, vmin=array.min(), vmax=array.max())

    ax.set_aspect('equal', 'datalim', 'C')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-3, -0.5)
    ax.pcolormesh(x, y, array, norm=normalize)

    GM_p, GM = params.get('gm_p'), params.get('gm')
    r_hill = rp * (GM_p / (3 * GM))**(1.0/3.0)

    circles = []
    circles.append(plt.Circle((0, 0), rp, fill=False, lw=0.5, color='r'))
    for m in [1,3]:
        circles.append(plt.Circle((0, -rp), radius=m*r_hill, color='r', fill=False, lw=.5))
        circles.append(plt.Circle((0, 0), radius=(rp-m * r_hill), color='r', ls='dashed', fill=False, lw=0.5))
        circles.append(plt.Circle((0, 0), radius=(rp+m * r_hill), color='r', ls='dashed', fill=False, lw=0.5))
    for m in np.arange(1.0,4.0):
        circles.append(plt.Circle((0,0), radius=rp*(m/(m+1.0))**(2.0/3.0), color='g', fill=False, lw=.5))
    for m in np.arange(2.0,4.0):
        circles.append(plt.Circle((0,0), radius=rp*(m/(m-1.0))**(2.0/3.0), color='g', fill=False, lw=.5))
    [ax.add_artist(circle) for circle in circles]

    if not (save is None):
        fig.savefig(save)
    plt.close(fig)


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
    integral = (r * dr)[:, np.newaxis, np.newaxis] * arr[intrange, :, :]
    integral = integral.sum()
    # integral = np.einsum('ijk,i->jk', arr[intrange, :, :], r * dr)
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


def centralDiff3D(grid, arr, arr_start=None, arr_end=None):
    """
    :type grid: planetHydro.parseData.gridReader.gridReader
    :return:
    """
    if arr_start is None:
        arr_start = (None,) * arr.shape[2]
    if arr_end is None:
        arr_end = (None,) * arr.shape[2]

    out = np.empty(grid.shape)
    for i in range(grid.nztot):
        out[:, :, i] = centralDiff2D(grid, arr[:, :, i], arr_start=arr_start[i], arr_end=arr_end[i])
    return out


def extrapolate(x, (x0, y0), (x1, y1)):
    return (y1 - y0) / (x1 - x0) * (x - x0) + y0


def centralDiff2D(grid, arr, arr_start=None, arr_end=None):
    """
    Takes derivative in r direction using central difference method.
    :type grid: planetHydro.parseData.gridReader.gridReader
    """
    r = (grid.r_edge[1:] + grid.r_edge[:-1]) / 2.0
    dr = (r[2:] - r[:-2])[1:-1]  # NOTE: this is dr1+dr2, not regular dr

    if arr_start is None: arr_start = extrapolate(r[1], (grid.r[0], arr[0, :]), (grid.r[1], arr[1, :]))
    if arr_end is None: arr_end = extrapolate(r[-2], (grid.r[-2], arr[-2, :]), (grid.r[-1], arr[-1, :]))
    _, nytot = arr.shape
    f1 = np.vstack((np.ones(nytot) * arr_start, arr[:-1, :]))  # f(r_i-1)
    f2 = np.vstack((arr[1:, :], np.ones(nytot) * arr_end))  # f(r_i+1)
    return colMul(1.0 / dr, f2 - f1)

def centralDiff1D(grid, arr, arr_start=None, arr_end=None):
    """
    Takes derivative in r direction using central difference method.
    :type grid: planetHydro.parseData.gridReader.gridReader
    :type arr: np.ndarray
    """
    r = (grid.r_edge[1:] + grid.r_edge[:-1])[1:-1] / 2.0
    dr = r[2:] - r[:-2]  # NOTE: this is dr1+dr2, not regular dr
    if arr_start is None: arr_start = extrapolate(r[0], (grid.r[0], arr[0]), (grid.r[1], arr[1]))
    if arr_end is None: arr_end = extrapolate(r[-1], (grid.r[-2], arr[-2]), (grid.r[-1], arr[-1]))
    f1 = np.hstack((arr_start, arr[:-1]))
    f2 = np.hstack((arr[1:], arr_end))
    return (f2-f1)/dr

def cart2cyl(grid, (vx, vy, vz)):
    sin, cos = np.sin(grid.phi3D), np.cos(grid.phi3D)
    r = vx * cos + vy * sin
    phi = -vx * sin + vy * cos
    return (r, phi, vz)
