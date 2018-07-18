# Description: Local ancillary functions.
#
# Author:      André Palóczy Filho
# E-mail:      paloczy@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from netCDF4 import num2date
from mpl_toolkits.basemap import Basemap
from pygamman import gamma_n as gmn
from pygeodesy import Datums, VincentyError
from pygeodesy.ellipsoidalVincenty import LatLon as LatLon
from pygeodesy.sphericalNvector import LatLon as LatLon_sphere
from pandas import Series


def seasonal_avg(t, F):
    tmo = np.array([ti.month for ti in t])
    ftmo = [tmo==mo for mo in range(1, 13)]

    return np.array([F[ft].mean() for ft in ftmo])


def seasonal_std(t, F):
    """
    USAGE
    -----
    F_seasonal = seasonal_std(t, F)

    Calculates the seasonal standard deviation of variable F(t).
    Assumes 't' is a 'datetime.datetime' object.
    """
    tmo = np.array([ti.month for ti in t])
    ftmo = [tmo==mo for mo in range(1, 13)]

    return np.array([F[ft].std() for ft in ftmo])


def deseason(t, F):
    Fssn = seasonal_avg(t, F)
    nyears = int(t.size/12)
    aux = np.array([])
    for n in range(nyears):
        aux = np.concatenate((aux, Fssn))

    return F - aux


def blkavgt(t, x, every=2):
    """
    Block-averages a variable x(t). Returns its block average
    and standard deviation and new t axis.
    """
    nt = t.size
    t = np.array([tt.toordinal() for tt in t])
    tblk, xblk, xblkstd = np.array([]), np.array([]), np.array([])
    for i in range(every, nt+every, every):
        xi = x[i-every:i]
        tblk = np.append(tblk, t[i-every:i].mean())
        xblk = np.append(xblk, xi.mean())
        xblkstd = np.append(xblkstd, xi.std())

    tblk = num2date(tblk, units='days since 01-01-01')
    return tblk, xblk, xblkstd


def blkavg(x, y, every=2):
    """
    Block-averages a variable y(x). Returns its block average
    and standard deviation and new x axis.
    """
    nx = x.size
    xblk, yblk, yblkstd = np.array([]), np.array([]), np.array([])
    for i in range(every, nx+every, every):
        yi = y[i-every:i]
        xblk = np.append(xblk, x[i-every:i].mean())
        yblk = np.append(yblk, yi.mean())
        yblkstd = np.append(yblkstd, yi.std())

    return xblk, yblk, yblkstd


def blksum(x, y, every=2):
    """
    Sums a variable y(x) in blocks. Returns its block average
    and new x axis.
    """
    nx = x.size
    xblk, yblksum = np.array([]), np.array([])
    for i in range(every, nx+every, every):
        xblk = np.append(xblk, x[i-every:i].mean())
        yblksum = np.append(yblksum, y[i-every:i].sum())

    return xblk, yblksum


def stripmsk(arr, mask_invalid=False):
    if mask_invalid:
        arr = np.ma.mask_invalid(arr)
    if np.ma.isMA(arr):
        msk = arr.mask
        arr = arr.data
        arr[msk] = np.nan

    return arr


def rcoeff(x, y):
        """
        USAGE
        -----
        r = rcoeff(x, y)

        Computes the Pearson correlation coefficient r between series x and y.

        References
        ----------
        e.g., Thomson and Emery (2014),
        Data analysis methods in physical oceanography,
        p. 257, equation 3.97a.
        """
        x,y = map(np.asanyarray, (x,y))

        # Sample size.
        assert x.size==y.size
        N = x.size

        # Demeaned series.
        x = x - x.mean()
        y = y - y.mean()

        # Standard deviations.
        sx = x.std()
        sy = y.std()

        ## Covariance between series. Choosing unbiased normalization (N-1).
        Cxy = np.sum(x*y)/(N-1)

        ## Pearson correlation coefficient r.
        r = Cxy/(sx*sy)

        return r


def near(x, x0, npts=1, return_index=False):
    """
    USAGE
    -----
    xnear = near(x, x0, npts=1, return_index=False)

    Finds 'npts' points (defaults to 1) in array 'x'
    that are closest to a specified 'x0' point.
    If 'return_index' is True (defauts to False),
    then the indices of the closest points are
    returned. The indices are ordered in order of
    closeness.
    """
    x = list(x)
    xnear = []
    xidxs = []
    for n in range(npts):
        idx = np.nanargmin(np.abs(np.array(x)-x0))
        xnear.append(x.pop(idx))
        if return_index:
            xidxs.append(idx)
    if return_index: # Sort indices according to the proximity of wanted points.
        xidxs = [xidxs[i] for i in np.argsort(xnear).tolist()]
    xnear.sort()

    if npts==1:
        xnear = xnear[0]
        if return_index:
            xidxs = xidxs[0]
    else:
        xnear = np.array(xnear)

    if return_index:
        return xidxs
    else:
        return xnear


def near2(x, y, x0, y0, npts=1, return_index=False):
    """
    USAGE
    -----
    xnear, ynear = near2(x, y, x0, y0, npts=1, return_index=False)

    Finds 'npts' points (defaults to 1) in arrays 'x' and 'y'
    that are closest to a specified '(x0, y0)' point. If
    'return_index' is True (defauts to False), then the
    indices of the closest point(s) are returned.

    Example
    -------
    >>> x = np.arange(0., 100., 0.25)
    >>> y = np.arange(0., 100., 0.25)
    >>> x, y = np.meshgrid(x, y)
    >>> x0, y0 = 44.1, 30.9
    >>> xn, yn = near2(x, y, x0, y0, npts=1)
    >>> print("(x0, y0) = (%f, %f)"%(x0, y0))
    >>> print("(xn, yn) = (%f, %f)"%(xn, yn))
    """
    x, y = map(np.array, (x, y))
    shp = x.shape

    xynear = []
    xyidxs = []
    dx = x - x0
    dy = y - y0
    dr = dx**2 + dy**2
    for n in range(npts):
        xyidx = np.unravel_index(np.nanargmin(dr), dims=shp)
        if return_index:
            xyidxs.append(xyidx)
        xyn = (x[xyidx], y[xyidx])
        xynear.append(xyn)
        dr[xyidx] = np.nan

    if npts==1:
        xynear = xynear[0]
        if return_index:
            xyidxs = xyidxs[0]

    if return_index:
        return xyidxs
    else:
        return xynear


def xy2dist(x, y, cyclic=False, datum='WGS84'):
    """
    USAGE
    -----
    d = xy2dist(x, y, cyclic=False, datum='WGS84')
    """
    if datum is not 'Sphere':
        xy = [LatLon(y0, x0, datum=Datums[datum]) for x0, y0 in zip(x, y)]
    else:
        xy = [LatLon_sphere(y0, x0) for x0, y0 in zip(x, y)]
    d = np.array([xy[n].distanceTo(xy[n+1]) for n in range(len(xy)-1)])

    return np.append(0, np.cumsum(d))


def lon180to360(lon):
        """
        Converts longitude values in the range [-180,+180]
        to longitude values in the range [0,360].
        """
        lon = np.asanyarray(lon)
        return (lon + 360.0) % 360.0


def lon360to180(lon):
        """
        Converts longitude values in the range [0,360]
        to longitude values in the range [-180,+180].
        """
        lon = np.asanyarray(lon)
        return ((lon + 180.) % 360.) - 180.


def rot_vec(u, v, angle=-45, degrees=True):
        """
        USAGE
        -----
        u_rot,v_rot = rot_vec(u,v,angle=-45.,degrees=True)

        Returns the rotated vector components (`u_rot`,`v_rot`)
        from the zonal-meridional input vector components (`u`,`v`).
        The rotation is done using the angle `angle` positive counterclockwise
        (trigonometric convention). If `degrees` is set to `True``(default),
        then `angle` is converted to radians.
        is

        Example
        -------
        >>> from matplotlib.pyplot import quiver
        >>> from ap_tools.utils import rot_vec
        >>> u = -1.
        >>> v = -1.
        >>> u2,v2 = rot_vec(u,v, angle=-30.)
        """
        u,v = map(np.asanyarray, (u,v))
        if degrees:
                angle = angle*np.pi/180. # Degrees to radians.

        u_rot = +u*np.cos(angle) + v*np.sin(angle) # Usually the across-shore component.
        v_rot = -u*np.sin(angle) + v*np.cos(angle) # Usually the along-shore component.

        return u_rot, v_rot


def angle_isobath(xiso, yiso, cyclic=True):
    R = 6371000.0 # Mean radius of the earth in meters (6371 km), from gsw.constants.earth_radius.
    deg2rad = np.pi/180. # [rad/deg]

    if cyclic: # Add cyclic point.
        xiso = np.append(xiso, xiso[0])
        yiso = np.append(yiso, yiso[0])

    # From the coordinates of the isobath, find the angle it forms with the
    # zonal axis, using points k+1 and k.
    shth = yiso.size-1
    theta = np.zeros(shth)
    for k in range(shth):
        dyk = R*(yiso[k+1]-yiso[k])
        dxk = R*(xiso[k+1]-xiso[k])*np.cos(yiso[k]*deg2rad)
        theta[k] = np.arctan2(dyk,dxk)

    xisom = 0.5*(xiso[1:] + xiso[:-1])
    yisom = 0.5*(yiso[1:] + yiso[:-1])

    return xisom, yisom, theta


def gamman(Sp, T, p, x, y):
    assert Sp.shape==T.shape
    n = np.size(p)
    skel = Sp*np.nan
    Spmsk = np.ma.masked_invalid(Sp).mask
    Tmsk = np.ma.masked_invalid(T).mask
    gammamsk = np.logical_or(Tmsk, Spmsk)
    Sp[Spmsk] = 35
    T[Tmsk] = 0
    gm, dgl, dgh = gmn(Sp, T, p, n, x, y)
    # gm, dgl, dgh = gmn.gamma_n(Sp, T, p, n, x, y)
    gm[gammamsk] = np.nan

    return gm, dgl, dgh


def bmap_antarctica(ax, resolution='h'):
	"""
	Full Antartica basemap (Polar Stereographic Projection).
	"""
	m = Basemap(boundinglat=-60,
		        lon_0=60,
	            projection='spstere',
	            resolution=resolution,
	            ax=ax)

	m.fillcontinents(color='0.9', zorder=9)
	m.drawcoastlines(zorder=10)
	m.drawmapboundary(zorder=-9999)
	m.drawmeridians(np.arange(-180, 180, 20), linewidth=0.15, labels=[1, 1, 1, 1], zorder=12)
	m.drawparallels(np.arange(-90, -50, 5), linewidth=0.15, labels=[0, 0, 0, 0], zorder=12)

	return m


def near(x, x0, npts=1, return_index=False):
    """
    USAGE
    -----
    xnear = near(x, x0, npts=1, return_index=False)

    Finds 'npts' points (defaults to 1) in array 'x'
    that are closest to a specified 'x0' point.
    If 'return_index' is True (defauts to False),
    then the indices of the closest points are
    returned. The indices are ordered in order of
    closeness.
    """
    x = list(x)
    xnear = []
    xidxs = []
    for n in range(npts):
        idx = np.nanargmin(np.abs(np.array(x)-x0))
        xnear.append(x.pop(idx))
        if return_index:
            xidxs.append(idx)
    if return_index: # Sort indices according to the proximity of wanted points.
        xidxs = [xidxs[i] for i in np.argsort(xnear).tolist()]
    xnear.sort()

    if npts==1:
        xnear = xnear[0]
        if return_index:
            xidxs = xidxs[0]
    else:
        xnear = np.array(xnear)

    if return_index:
        return xidxs
    else:
        return xnear


def UVz2iso(U, V, hisopyc, z):
	ny, nx = hisopyc.shape
	Uisopyc = np.nan*np.ones((ny,nx))
	Visopyc = np.nan*np.ones((ny,nx))
	for j in range(ny):
		print("Row %d of %d"%(j+1,ny))
		for i in range(nx):
			if np.isnan(hisopyc[j,i]):
				continue
			else:
				Uisopyc[j,i] = np.interp(hisopyc[j,i], z, U[:,j,i])
				Visopyc[j,i] = np.interp(hisopyc[j,i], z, V[:,j,i])

	return Uisopyc, Visopyc


def isopyc_depth(dens0, z, isopyc=1027.75, dzref=1.):
    """
    USAGE
    -----
    hisopyc = isopyc_depth(z, dens0, isopyc=1027.75)

    Calculates the spatial distribution of the depth of a specified isopycnal 'isopyc'
    (defaults to 1027.75 kg/m3) from a 2D density section rho0 (in kg/m3) with shape
    (nz,ny,nx) and a 1D depth array 'z' (in m) with shape (nz).

    'dzref' is the desired resolution for the refined depth array (defaults to 1 m) which
    is generated for calculating the depth of the isopycnal. The smaller 'dzref', the smoother
    the resolution of the returned isopycnal depth array 'hisopyc'.
    """
    dens0, z = map(np.array, (dens0, z))
    if not np.all(np.diff(z)>0):
        z = np.flipud(z)
        dens0 = np.flipud(dens0)
    if dens0.ndim==2:
        nz, nx = dens0.shape
    else:
        nz = dens0.size
        nx = 1
    zref = np.arange(z.min(), z.max()+dzref, dzref)

    if np.ma.isMaskedArray(dens0):
        dens0 = np.ma.filled(dens0, np.nan)

    hisopyc = np.nan*np.ones((nx))
    for i in range(nx):
        if nx==1:
            dens0i = dens0
        else:
            dens0i = dens0[:,i]

        cond1 = np.logical_or(isopyc<np.nanmin(dens0i), np.nanmax(dens0i)<isopyc)
        if np.logical_or(cond1, np.isnan(dens0i).all()):
            continue
        else:
            dens0ref = np.interp(zref, z, dens0i) # Refined density profile.
            fz = near(dens0ref, isopyc, return_index=True)
            try:
                hisopyc[i] = zref[fz]
            except ValueError:
                print("Warning: More than 1 (%d) nearest depths found. Using the median of the depths for point (i=%d)."%(fz.sum(), i))
                hisopyc[i] = np.nanmedian(zref[fz])

    return hisopyc


def isopyc_depth2(z, dens0, isopyc=1027.75, dzref=1.):
    """
    USAGE
    -----
    hisopyc = isopyc_depth(z, dens0, isopyc=1027.75)

    Calculates the spatial distribution of the depth of a specified isopycnal 'isopyc'
    (defaults to 1027.75 kg/m3) from a 3D density array rho0 (in kg/m3) with shape
    (nz,ny,nx) and a 1D depth array 'z' (in m) with shape (nz).

    'dzref' is the desired resolution for the refined depth array (defaults to 1 m) which
    is generated for calculating the depth of the isopycnal. The smaller 'dzref', the smoother
    the resolution of the returned isopycnal depth array 'hisopyc'.
    """
    z, dens0 = map(np.asanyarray, (z, dens0))
    ny, nx = dens0.shape[1:]
    if not np.all(np.diff(z>0)):
        z = np.flipud(z)
        dens0 = np.flipud(dens0)
    zref = np.arange(z.min(), z.max(), dzref)

    if np.ma.isMaskedArray(dens0):
        dens0 = np.ma.filled(dens0, np.nan)

    hisopyc = np.nan*np.ones((ny,nx))
    for j in range(ny):
        print("Row %d of %d"%(j+1,ny))
        for i in range(nx):
            dens0ij = dens0[:,j,i]
            if np.logical_or(np.logical_or(isopyc<np.nanmin(dens0ij), np.nanmax(dens0ij)<isopyc), np.isnan(dens0ij).all()):
                continue
            else:
                dens0ref = np.interp(zref, z, dens0ij) # Refined density profile.
                dens0refn = near(dens0ref, isopyc)
                fz=dens0ref==dens0refn
                try:
                    hisopyc[j,i] = zref[fz]
                except ValueError:
                    print("Warning: More than 1 (%d) nearest depths found. Using the median of the depths for point (j=%d,i=%d)."%(fz.sum(), j, i))
                    hisopyc[j,i] = np.nanmedian(zref[fz])

    return hisopyc


def montecarlo_gamman(sp_mean, spSE, t_mean, tSE, p0, z0, x0, y0, wanted_isoneutrals, dzref,
                      nmc=1e4, alpha=0.95, nbins=100, plot_cdf=True, verbose=True):
    """Using block-averaged lon, lat (x0, y0)."""
    nmc = int(nmc)
    ngamma = len(wanted_isoneutrals)
    nz = z0.size
    if not np.all(np.diff(z0)>=0):
        z0 = np.flipud(z0)
        p0 = np.flipud(p0)
        sp_mean = np.flipud(sp_mean)
        spSE = np.flipud(spSE)
        t_mean = np.flipud(t_mean)
        tSE = np.flipud(tSE)
    z00 = z0.copy()
    # Mean neutral density profile for this segment.
    gamman_MEANSEG, _, _ = gamman(sp_mean, t_mean, p0, x0, y0)
    # Calculate derived variable (the isoneutral depth) with the synthetic random data.
    hgamman_mc = np.empty((ngamma, nmc))*np.nan
    hgamman_errs = np.empty(ngamma)*np.nan
    for n in range(nmc):
        if verbose:
            print("Simulated Monte Carlo profile %d of %d"%(n+1,nmc))
        SP_mc = spSE*np.random.randn(nz) + sp_mean
        T_mc = tSE*np.random.randn(nz) + t_mean
        gamman_mc, _, _ = gamman(SP_mc, T_mc, p0, x0, y0)
        # gamman_mc = np.ma.masked_invalid(gamman_mc)
        fgud = ~np.isnan(gamman_mc)
        gamman_mc = gamman_mc[fgud]
        z0 = z00[fgud]
        for ngm in range(ngamma):
            ison = wanted_isoneutrals[ngm]
            isodpth = isopyc_depth(gamman_mc, z0, isopyc=ison, dzref=dzref)
            # print(isodpth)
            hgamman_mc[ngm, n] = isodpth

    # Calculate PDF and CDF of the simulated data.
    hgamman_MEANSEG = np.empty(ngamma)*np.nan
    gamman_MEANSEG = gamman_MEANSEG[fgud]
    for ngm in range(ngamma):
        ison = wanted_isoneutrals[ngm]
        hgamman_mcm = hgamman_mc[ngm, :]
        if np.all(np.isnan(hgamman_mcm)):
            print("Skipping %.2f kg/m3 isoneutral (Not found in any of the simulated profiles)."%ison)
            continue
        # Mean depth of this isoneutral for this segment.
        hgamman_MEANSEG[ngm] = isopyc_depth(gamman_MEANSEG, z0, isopyc=ison, dzref=dzref)
        fig, ax = plt.subplots()
        # Remove the mean from the absolute value of the simulated depths
        # so the error is a +- (depth interval).
        hgamman_mcm = hgamman_mcm[~np.isnan(hgamman_mcm)]
        hgamman_mcm = hgamman_mcm - hgamman_MEANSEG[ngm]
        hgamman_mcm = np.abs(hgamman_mcm)
        hlb, hrb = hgamman_mcm.min(), hgamman_mcm.max()
        hgamman_hist, bins, _ = ax.hist(hgamman_mcm, bins=nbins, density=True,
                                        range=(hlb, hrb), color='b', histtype='bar')

        # Calculate and plot CDF.
        dbin = bins[1:] - bins[:-1]
        cdf = np.cumsum(hgamman_hist*dbin) # CDF [unitless].
        cdf = np.insert(cdf, 0, 0)
        fci_alpha = near(cdf, alpha, return_index=True)

        # Value of the Monte Carlo derived variable associated with the alpha*100 percentile.
        hgamman_err = bins[fci_alpha]

        if plot_cdf:
                ax2 = ax.twinx()
                ax2.plot(bins, cdf, 'k', linewidth=3.0)
                # ax2.plot(binm, cdf, 'k', linewidth=3.0)
                ax2.axhline(y=alpha, linewidth=1.5, linestyle='dashed', color='grey')
                ax2.axvline(x=hgamman_err, linewidth=1.5, linestyle='dashed', color='grey')
                ax.set_ylabel(r'Probability density', fontsize=18, fontweight='black')
                ax2.set_ylabel(r'Cumulative probability', fontsize=18, fontweight='black')
                ax.set_xlabel(r'Departure from mean isoneutral depth [m]', fontsize=18, fontweight='black')
                ytks = ax2.get_yticks()
                ytks = np.append(ytks, alpha)
                ytks.sort()
                ytks = ax2.set_yticks(ytks)
                fig.canvas.draw()
                plt.show()
        else:
                plt.close()

        # Error is symmetric about zero, because the
        # derived variable is assumed to be normal.
        hgamman_err = hgamman_err/2.
        if verbose:
            print("The Monte Carlo error for the derived variable is +- %.2f (alpha=%.2f)"%(hgamman_err, alpha))
        hgamman_errs[ngm] = hgamman_err
    # hgamman_err is the envelope containing 100*'alpha' % of the values
    # of the simulated derived variable. So the error bars
    # are +-hgamman_err/2 (symmetric about each data point).

    return hgamman_MEANSEG, hgamman_err, hgamman_mc

def crosscorr(x, y, nblks, maxlags=0, overlap=0, onesided=False, verbose=False):
    """
    Lag-N cross correlation averaged with Welch's Method.
    Parameters
    ----------
    x, y     : Arrays of equal length.
    nblks    : Number of blocks to average cross-correlation.
    maxlags  : int, default (0) calculates te largest possible number of lags,
               i.e., the number of points in each chunk.
    overlap  : float, fraction of overlap between consecutive chunks. Default 0.
    onesided : Whether to calculate the cross-correlation only at
               positive lags (default False). Has no effect if
               x and y are the same array, in which case the
               one-sided autocorrelation function is calculated.

    Returns
    ----------
    crosscorr : float array.
    """
    if x is y:
        auto = True
    else:
        auto = False
    x, y = np.array(x), np.array(y)
    nx, ny = x.size, y.size
    assert x.size==y.size, "The series must have the same length"

    nblks, maxlags = int(nblks), int(maxlags)
    ni = int(nx/nblks)               # Number of data points in each chunk.
    dn = int(round(ni - overlap*ni)) # How many indices to move forward with
                                     # each chunk (depends on the % overlap).

    if maxlags==0:
        if verbose:
            print("Maximum lag was not specified. Accomodating it to block size (%d)."%ni)
        maxlags = ni
    elif maxlags>ni:
        if verbose:
            print("Maximum lag is too large. Accomodating it to block size (%d)."%ni)
        maxlags = ni

    if onesided:
        lags = range(maxlags+1)
    else:
        lags = range(-maxlags, maxlags+1)

    # Array that will receive cross-correlation of each block.
    xycorr = np.zeros(len(lags))

    n=0
    il, ir = 0, ni
    while ir<=nx:
        xn = x[il:ir]
        yn = y[il:ir]

        # Calculate cross-correlation for current block up to desired maximum lag - 1.
        xn, yn = map(Series, (xn, yn))
        xycorr += np.array([xn.corr(yn.shift(periods=lagn)) for lagn in lags])

        il+=dn; ir+=dn
        n+=1

    # pandas.Series.corr(method='pearson') -> pandas.nanops.nancorr() ...
    # -> pandas.nanops.get_corr_function() -> np.corrcoef -> numpy.cov(bias=False as default).
    # So np.corrcoef() returns the UNbiased correlation coefficient by default
    # (i.e., normalized by N-k instead of N).

    xycorr /= n    # Divide by number of blocks actually used.
    ncap = nx - il # Number of points left out at the end of array.

    if verbose:
        print("")
        if ncap==0:
            print("No data points were left out.")
        else:
            print("Left last %d data points out (%.1f %% of all points)."%(ncap,100*ncap/nx))
        print("Averaged %d blocks, each with %d lags."%(n,maxlags))
        if overlap>0:
            print("Intended %d blocks, but could fit %d blocks, with"%(nblks,n))
            print('overlap of %.1f %%, %d points per block.'%(100*overlap,dn))
        print("")

    lags = np.array(lags)
    if auto:
        fo = np.where(lags==0)[0][0]
        xycorr[fo+1:] = xycorr[fo+1:] + xycorr[:fo]
        lags = lags[fo:]
        xycorr = xycorr[fo:]

    fgud=~np.isnan(xycorr)

    return lags[fgud], xycorr[fgud]


def Tdecorr(Rxx, M=None, dtau=1., verbose=False):
    """
    USAGE
    -----
    Td = Tdecorr(Rxx)

    Computes the integral scale Td (AKA decorrelation scale, independence scale)
    for a data sequence with autocorrelation function Rxx. 'M' is the number of
    lags to incorporate in the summation (defaults to all lags) and 'dtau' is the
    lag time step (defaults to 1).

    The formal definition of the integral scale is the total area under the
    autocorrelation curve Rxx(tau):

    /+inf
    Td = 2 * |     Rxx(tau) dtau
    /0

    In practice, however, Td may become unrealistic if all of Rxx is summed
    (e.g., often goes to zero for data dominated by periodic signals); a
    different approach is to instead change M in the summation and use the
    maximum value of the integral Td(t):

    /t
    Td(t) = 2 * |     Rxx(tau) dtau
    /0

    References
    ----------
    e.g., Thomson and Emery (2014),
    Data analysis methods in physical oceanography,
    p. 274, equation 3.137a.

    Gille lecture notes on data analysis, available
    at http://www-pord.ucsd.edu/~sgille/mae127/lecture10.pdf
    """
    Rxx = np.asanyarray(Rxx)
    C0 = Rxx[0]
    N = Rxx.size # Sequence size.

    # Number of lags 'M' to incorporate in the summation.
    # Sum over all of the sequence if M is not chosen.
    if not M:
        M = N

    # Integrate the autocorrelation function.
    Td = np.zeros(M)
    for m in range(M):
        Tdaux = 0.
        for k in range(m-1):
            Rm = (Rxx[k] + Rxx[k+1])/2. # Midpoint value of the autocorrelation function.
            Tdaux = Tdaux + Rm*dtau # Riemann-summing Rxx.

        Td[m] = Tdaux

    # Normalize the integral function by the autocorrelation at zero lag
    # and double it to include the contribution of the side with
    # negative lags (C is symmetric about zero).
    Td = (2./C0)*Td

    if verbose:
        print("")
        print("Theoretical integral scale --> 2 * int 0...+inf [Rxx(tau)] dtau: %.2f."%Td[-1])
        print("")
        print("Maximum value of the cumulative sum: %.2f."%Td.max())

    return Td


def Tdecorrw(x, nblks=30, ret_median=True, verbose=True):
    """
    USAGE
    -----
    Ti = Tdecorrw(x, nblks=30, ret_median=True, verbose=True)

    'Ti' is the integral timescale calculated from the
    autocorrelation function calculated for variable 'x'
    block-averaged in 'nblks' chunks.
    """
    x = np.array(x)
    dnblkslr = round(nblks/2)

    tis = [Tdecorr(crosscorr(x, x, nblks=n, verbose=verbose)[1]).max() for n in range(nblks-dnblkslr, nblks+dnblkslr+1)]
    tis = np.ma.masked_invalid(tis)

    if verbose:
        print("========================")
        print(tis)
        print("========================")
        p1, p2, p3, p4, p5 = map(np.percentile, [tis]*5, (10, 25, 50, 75, 90))
        print("--> 10 %%, 25 %%, 50 %%, 75 %%, 90 %% percentiles for Ti:  %.2f,  %.2f,  %.2f,  %.2f,  %.2f."%(p1, p2, p3, p4, p5))
        print("------------------------")

    if ret_median:
        return np.median(tis)
    else:
        return tis

def rsig(ndof_eff, alpha=0.95):
	"""
	USAGE
	-----
	Rsig = rsig(ndof_eff, alpha=0.95)

	Computes the minimum (absolute) threshold value 'rsig' that
	the Pearson correlation coefficient r between two normally-distributed
	data sequences with 'ndof_eff' effective degrees of freedom has to have
	to be statistically significant at the 'alpha' (defaults to 0.95)
	confidence level.

	For example, if rsig(ndof_eff, alpha=0.95) = 0.2 for a given pair of
	NORMALLY-DISTRIBUTED samples with a correlation coefficient r>0.7, there
	is a 95 % chance that the r estimated from the samples is significantly
	different from zero. In other words, there is a 5 % chance that two random
	sequences would have a correlation coefficient higher than 0.7.

	OBS: This assumes that the two data series have a normal distribution.

	Translated to Python from the original matlab code by Prof. Sarah Gille
	(significance.m), available at http://www-pord.ucsd.edu/~sgille/sio221c/

	References
	----------
	Gille lecture notes on data analysis, available
	at http://www-pord.ucsd.edu/~sgille/mae127/lecture10.pdf

	Example
	-------
	TODO
	"""
	rcrit_z = erfinv(alpha)*np.sqrt(2./ndof_eff)

	return rcrit_z


def arsig(r0, Ndt, T1, T2, verbose=True):
    """
    USAGE
    -----
    alpha_rsig = arsig(r0, Ndt, T1, T2, verbose=True)

    The returned 'alpha_sig' is the CL at which the
    correlation coefficient 'r0' between two variables
    with integral timescales 'T1' and 'T2' and length 'Ndt'
    is significant.
    """
    Tslow = np.maximum(T1, T2) # The effective number of
    edof = Ndt/Tslow           # DoFs is constrained by the
                               # slower-decorrelating variable.
    alphai = 0.4
    issig = True
    while issig and alphai<1.0:
        rsigi = rsig(edof, alpha=alphai)
        issig=r0>=rsigi
        alphai+=0.01

    if verbose:
        print("Queried r = %.3f with %.1f EDoF. It is significant at **%.2f** CL."%(r0, edof, alphai))

    return alphai
