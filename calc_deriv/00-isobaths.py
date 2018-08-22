# Description: Make a control volume encircling Antarctica
#              using the POP 0.1 deg run topography.
#
# Author:      André Palóczy Filho
# E-mail:      paloczy@gmail.com
# Date:        August/2016

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from ap_tools.datasets import topo_subset
from ap_tools.utils import fmt_isobath, lon360to180
from ap_tools.utils import get_isobath#, angle_isobath
from ap_tools.utils import near, fpointsbox
from ap_tools.utils import avgdir
from gsw import distance
import time
import pickle
import xarray as xr
from os import system
from os.path import isfile
# from stripack import trmesh
from stripy.spherical import sTriangulation as trmesh
from cartopy import crs
from pygeodesy.sphericalNvector import LatLon as LatLon_sphere
from pygeodesy import Datums, VincentyError
from pygeodesy.ellipsoidalVincenty import LatLon as LatLon
from pygeodesy.utils import wrap180
from reproducibility import stamp, savefig, repohash
from cmocean import cm as cmo

import pyximport; pyximport.install()
from near2 import nearidx2

def _update_distances(distisonnU, distisonnT, HTE, HTN, HUW, HUS, icjc, imjm):
    ic, jc = icjc
    im, jm = imjm
    di = ic - im
    dj = jc - jm
    if dj<-1:
        dj = 1
    elif dj>1:
        dj = -1
    assert not di==dj==0, "di and dj are both 0."
    if di==0:
        if dj==1:    # Moved E.
            distisonnT.append(HUS[ic-1,jc])
            distisonnU.append(HTN[ic,jc])
        elif dj==-1: # Moved W.
            distisonnT.append(HUS[ic,jc])
            distisonnU.append(HTN[ic,jc+1])
    if dj==0:
        if di==1:    # Moved N.
            distisonnT.append(HUW[ic-1,jc])
            distisonnU.append(HTE[ic,jc])
        elif di==-1: # Moved S.
            distisonnT.append(HUW[ic,jc])
            distisonnU.append(HTE[ic+1,jc])

    return distisonnU, distisonnT

def get_trmesh(lonurad, laturad, lontrad, lattrad):
    ftri_U = 'tri_U.pkl'
    if isfile(ftri_U):
    	tri_U = pickle.load(open(ftri_U, 'rb'))
    else:
    	tini = time.clock()
    	print('Triangulation of ' + str(lonu.size) + ' points')
    	tri_U = trmesh(lonurad, laturad) # Longitudes and latitudes in RADIANS.
    	print('Triangulation of U-points took ' + str((time.clock()-tini)/60.) + ' min')
    	pickle.dump(tri_U, open(ftri_U, 'wb'))
    #
    ftri_T = 'tri_T.pkl'
    if isfile(ftri_T):
    	tri_T = pickle.load(open(ftri_T, 'rb'))
    else:
    	tini = time.clock()
    	print('Triangulation of ' + str(lont.size) + ' points')
    	tri_T = trmesh(lontrad, lattrad) # Longitudes and latitudes in RADIANS.
    	print('Triangulation of T-points took ' + str((time.clock()-tini)/60.) + ' min')
    	pickle.dump(tri_T, open(ftri_T, 'wb'))

    return tri_T, tri_U

def getstep_quad(hdg):
    if np.logical_and(hdg>-45, hdg<=45):
        istep, jstep = 0, 1
    elif np.logical_or(hdg>135, hdg<=-135):
        istep, jstep = 0, -1
    elif np.logical_and(hdg>45, hdg<=135):
        istep, jstep = 1, 0
    elif np.logical_and(hdg>-135, hdg<=-45):
        istep, jstep = -1, 0
    else:
        print("Invalid heading")
        return None

    return istep, jstep

def getquad(hdg):
    if np.logical_and(hdg>-45, hdg<=45):
        quad0 = 1
    elif np.logical_or(hdg>135, hdg<=-135):
        quad0 = 3
    elif np.logical_and(hdg>45, hdg<=135):
        quad0 = 2
    elif np.logical_and(hdg>-135, hdg<=-45):
        quad0 = 4
    else:
        print("Invalid heading")
        return None

    return quad0

# Walk cell by cell only towards immediate neighbors (0, 90, -90 or 180 degrees),
# from each T-cell to its neighbors, until the next cell defining
# the control volume is reached.
def getstep(hdg, hdg_ref, quadm):
    dictpos = {1:(1,0), 2:(0,-1), 3:(-1,0), 4:(0,1)}
    dictneg = {1:(0, 1), 2:(1, 0), 3:(0, -1), 4:(-1, 0)}
    dictquad = {1:dictpos, -1:dictneg}
    dhdg = hdg - hdg_ref
    quadc = getquad(hdg)
    if np.logical_or(dhdg==0, quadc!=quadm):
        iwlk, jwlk = getstep_quad(hdg)
    else:
        dhdgsgn = dhdg/np.abs(dhdg)
        iwlk, jwlk = dictquad[dhdgsgn][quadc]

    return iwlk, jwlk, quadc

def bmap_antarctica(ax, resolution='i'):
	"""
	Full Antartica basemap (Polar Stereographic Projection).
	"""
	m = Basemap(boundinglat=-60,
		        lon_0=60,
	            projection='spstere',
	            resolution=resolution,
	            ax=ax)

	m.drawmapboundary(zorder=1)
	m.fillcontinents(color='0.9', zorder=98)
	m.drawcoastlines(zorder=98)
	m.drawmeridians(np.arange(-180, 180, 20), linewidth=0.15, labels=[1, 1, 1, 1], zorder=99)
	m.drawparallels(np.arange(-90, -50, 5), linewidth=0.15, labels=[0, 0, 0, 0], zorder=99)

	return m

def xy2dist(x, y, cyclic=False, datum='WGS84'):
    """
    USAGE
    -----
    d = xy2dist(x, y, cyclic=False, datum='WGS84')

    Calculates a distance axis from a line defined by longitudes and latitudes
    'x' and 'y', using either the Vicenty formulae on an ellipsoidal earth
    (ellipsoid defaults to WGS84) or on a sphere (if datum=='Sphere').
    """
    if datum is not 'Sphere':
        xy = [LatLon(y0, x0, datum=Datums[datum]) for x0, y0 in zip(x, y)]
    else:
        xy = [LatLon_sphere(y0, x0) for x0, y0 in zip(x, y)]
    d = np.array([xy[n].distanceTo(xy[n+1]) for n in range(len(xy)-1)])

    return np.append(0, np.cumsum(d))


def _wrapj(j, nx):
    if j==nx:
        print('Warning: Crossing edge of array.')
        j = j - nx
    return j


def get_xisobUVmsks(Ic, Jc, lont, latt, polyvolU, nx):
    """
    Get the (U, V) mask for each cell such that only the inward (cross-isobath)
    transports are considered.

    Returns two arrays 'Umsk' and 'Vmsk' with 1, 0, -1
    values, such that the cross-isobath flux velocity
    'Uxisob' at each grid cell of the discretized
    isobath is

    Uxisob[n] = Umsk[n]*UeT[n] + Vmsk[n]*VnT[n].
    """
    Pij = np.hstack((Ic[...,np.newaxis], Jc[...,np.newaxis])).tolist()
    Pij.append(Pij[0])
    Pij.insert(0, Pij[-2])
    nmax_msk = len(Pij)
    Umsk = np.zeros(nmax_msk)
    Vmsk = np.zeros(nmax_msk)
    moves = []
    for n in range(2, nmax_msk):
        i, j = Pij[n]
        if (j+1)==nx:
            print('Warning: Crossing edge of array. Wrapping around.')

        iijj = np.vstack((Pij[n-2], Pij[n-1], Pij[n]))
        # print(iijj)
        dij = np.diff(iijj, axis=0)
        dij[dij<-1] = 1
        dij[dij>1] = -1
        di, dj = dij[0,:], dij[1,:]
        dij = (di.tolist(), dj.tolist())
        ijm = tuple(Pij[n-1])
        # Is the PREVIOUS T-point inside the volume?
        Tin = np.bool8(LatLon_sphere(latt[ijm], lont[ijm]).isEnclosedBy(polyvolU))
        print("i, j = %d, %d"%(i, j))
        # print(dij)

        if dij==([0, 1], [0, 1]):
            move = 'EE'
            Umskn = 0
            if Tin:
                Vmskn = -1
            else:
                Vmskn = +1
        elif dij==([0, -1], [0, -1]):
            move = 'WW'
            Umskn = 0
            if Tin:
                Vmskn = -1
            else:
                Vmskn = +1
        elif dij==([1, 0], [1, 0]):
            move = 'NN'
            Vmskn = 0
            if Tin:
                Umskn = -1
            else:
                Umskn = +1
        elif dij==([-1, 0], [-1, 0]):
            move = 'SS'
            Vmskn = 0
            if Tin:
                Umskn = -1
            else:
                Umskn = +1
        elif dij==([0, 1], [1, 0]):
            move = 'EN'
            Umskn = 0
            if Tin:
                Vmskn = -1
            else:
                Vmskn = +1
        elif dij==([-1, 0], [0, -1]):
            move = 'SW'
            Umskn = 0
            if Tin:
                Vmskn = -1
            else:
                Vmskn = +1
        elif dij==([1, 0], [0, 1]):
            move = 'NE'
            Vmskn = 0
            if Tin:
                Umskn = -1
            else:
                Umskn = +1
        elif dij==([0, -1], [-1, 0]):
            move = 'WS'
            Vmskn = 0
            if Tin:
                Umskn = -1
            else:
                Umskn = +1
        elif dij==([-1, 0], [0, 1]):
            move = 'SE'
            Umskn = Vmskn = 0
            pass
        elif dij==([0, -1], [1, 0]):
            move = 'WN'
            Umskn = Vmskn = 0
            pass
        elif dij==([0, 1], [-1, 0]):
            move = 'ES'
            if Tin:
                Umskn = -1
                Vmskn = -1
            else:
                Umskn = +1
                Vmskn = +1
        elif dij==([1, 0], [0, -1]):
            move = 'NW'
            if Tin:
                Umskn = -1
                Vmskn = -1
            else:
                Umskn = +1
                Vmskn = +1
        else:
            move = 'x'
            Umskn = Vmskn = 0

        Umsk[n-1] = Umskn
        Vmsk[n-1] = Vmskn

        print("%d of %d, %s"%(n+1, nmax_msk, move))
        moves.append(move)

    # Remove cyclic points so that the mask
    # is properly aligned with UeT and VnT.
    Umsk = Umsk[1:-1]
    Vmsk = Vmsk[1:-1]

    return Umsk, Vmsk, moves

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
#===
plt.close('all')

# ISOBS = [2400, 1200]
# ISOBS = [2500, 2400, 2000, 1800, 1500, 1400, 1300, 1200, 1100, 1000, 750, 500, 250, 100, 50, 0] # Skip 1900 m, weird point.
# ISOBS = [2500, 2400, 1200, 1000] # Skip 2000 m and 1900 m, weird points.
# ISOBS = [2500, 2400, 1000, 800]
ISOBS = [2500, 1000, 800]
# ISOBS = [2000]# [2500, 2000, 1000, 800] #[2500, 2400, 2000, 1000, 800]
WEIRD_ISOBS = ISOBS # [2000]

nboxes = 3#5

fmt = 'png'
figname = 'isobath_antarctica.' + fmt
isobs_map = [1000., 2500., 4000.]
LATCROP = -56. # Grid of SO subsets goes up to -56.156734058642421 (T points) and -56.128857617446108 (U points).

head = 'data_reproduce_figs/'

cm2m = 1e-2
m2km = 1e-3
rad2deg = 180./np.pi
deg2rad = 1./rad2deg
fname_out = head + 'isobaths.nc'

## Get model topography and grid.
# head = '/lustre/atlas1/cli115/proj-shared/apaloczy/POP_full_simulation_from_monthly_outputs/'
fname_grd = '/home/andre/sio/phd/papers/pathways_antarctica/mkfigs/data_reproduce_figs/SOsubset_avg_2005-2009.nc'
# fname_grd = '/home/andre/sio/phd/papers/pathways_antarctica/mkfigs/data_reproduce_figs/POP_topog.nc'
ncg = Dataset(fname_grd)
lont = ncg.variables['TLONG'][:]
latt = ncg.variables['TLAT'][:]
lonu = ncg.variables['ULONG'][:]
latu = ncg.variables['ULAT'][:]
HTE = ncg.variables['HTE'][:]*cm2m
HTN = ncg.variables['HTN'][:]*cm2m
HUW = ncg.variables['HUW'][:]*cm2m
HUS = ncg.variables['HUS'][:]*cm2m
ht = ncg.variables['HT'][:]*cm2m
lont = lon360to180(lont)
lonu = lon360to180(lonu)
nx = lonu.shape[1]
ny = np.sum(latu[:,0]<=LATCROP)
fSO=latu<=LATCROP # Mask to get the Southern Ocean part of the grid.
sh = (ny,nx)
ht = ht[fSO].reshape(sh)
latt = latt[fSO].reshape(sh)
lont = lont[fSO].reshape(sh)
latu = latu[fSO].reshape(sh)
lonu = lonu[fSO].reshape(sh)
HTE = HTE[fSO].reshape(sh)
HTN = HTN[fSO].reshape(sh)
HUW = HUW[fSO].reshape(sh)
HUS = HUS[fSO].reshape(sh)
lonurad, laturad = lonu.ravel()*deg2rad, latu.ravel()*deg2rad
lontrad, lattrad = lont.ravel()*deg2rad, latt.ravel()*deg2rad

# Get isobaths.
for ISOB in ISOBS:
    xiso, yiso = get_isobath(lonu, latu, ht, ISOB, smooth_isobath=False)

    # Make sure longitude starts at -180 and grows monotonically.
    ff=xiso==near(xiso, -180., return_index=False)
    ff = np.where(ff)[0]
    ff = ff[0] # Make sure only the first point is taken if more than one is found.
    xiso_l = xiso[ff:]
    xiso_r = xiso[:ff]
    yiso_l = yiso[ff:]
    yiso_r = yiso[:ff]
    xiso = np.concatenate((xiso_l, xiso_r))
    yiso = np.concatenate((yiso_l, yiso_r))

    # 1) Fix bad longitude points manually BEFORE calculating xiso_mid/yiso_mid (midpoints).
    # 'Bad points' are sharp discontinuities in the along-isobath distance.
    # Now remove bad points produced by the averaging and calculate distance.
    fig, ax = plt.subplots()
    ax.plot(xiso, yiso, 'o')
    ax.grid()
    ax.plot(xiso, yiso, 'b', linestyle='solid')
    ax.plot(xiso[0], yiso[0], 'yo')
    ax.plot(xiso[-1], yiso[-1], 'yo')
    ax.set_title('%d m isobath'%int(ISOB), fontsize=18, fontweight='black')
    ax.set_xlabel('Longitude [degrees East]', fontsize=18, fontweight='black')
    ax.set_ylabel('Latitude [degrees North]', fontsize=18, fontweight='black')

    fbad_pts = fpointsbox(xiso, yiso, fig, ax, nboxes=nboxes, plot=True, pause_secs=2, return_index=True)
    fbad_pts = np.int32(fbad_pts)
    for bad in fbad_pts:
        xiso[bad] = np.nan
        yiso[bad] = np.nan
    fnnan = ~np.isnan(xiso)
    xiso = xiso[fnnan]
    yiso = yiso[fnnan]
    plt.close('all')

    # Add cyclic point on the raw isobaths
    # to ensure the model ones have it too.
    xiso = np.append(xiso, xiso[0])
    yiso = np.append(yiso, yiso[0])

    xiso_orig = xiso.copy()
    yiso_orig = yiso.copy()

    # Make isobath coordinates be evenly-spaced.
    distiso_orig = xy2dist(xiso, yiso, datum='Sphere')

    neven = distiso_orig.size # Same number of points, but evenly spaced.
    distiso_even = np.linspace(0, distiso_orig[-1], num=neven)
    xiso = np.interp(distiso_even, distiso_orig, xiso)
    yiso = np.interp(distiso_even, distiso_orig, yiso)

    # Get distances just for plotting before subsetting.
    distiso = distiso_even*m2km # [km].

    # For problematic isobaths.
    if ISOB in WEIRD_ISOBS:
        fig, ax = plt.subplots()
        ax.plot(xiso, yiso, 'o')
        ax.grid()
        ax.plot(xiso, yiso, 'b', linestyle='solid')
        ax.plot(xiso[0], yiso[0], 'yo')
        ax.plot(xiso[-1], yiso[-1], 'yo')
        ax.set_title('%d m isobath'%int(ISOB), fontsize=18, fontweight='black')
        ax.set_xlabel('Longitude [degrees East]', fontsize=18, fontweight='black')
        ax.set_ylabel('Latitude [degrees North]', fontsize=18, fontweight='black')

        fbad_pts = fpointsbox(xiso, yiso, fig, ax, nboxes=nboxes, plot=True, pause_secs=2, return_index=True)
        fbad_pts = np.int32(fbad_pts)
        for bad in fbad_pts:
            xiso[bad] = np.nan
            yiso[bad] = np.nan
            distiso[bad] = np.nan
        fnnan = ~np.isnan(xiso)
        xiso = xiso[fnnan]
        yiso = yiso[fnnan]
        distiso = distiso[fnnan]
        plt.close('all')

    # Interpolate grid points to isobath.
    tri_T, tri_U = get_trmesh(lonurad, laturad, lontrad, lattrad)
    xisorad, yisorad = xiso.ravel()*deg2rad, yiso.ravel()*deg2rad

    # Get nearest POP grid points (T points and U points).
    print("Finding the U-points closest to the new isobath.")
    xuu = tri_U.interpolate_nearest(xisorad, yisorad, lonurad)*rad2deg
    yuu = tri_U.interpolate_nearest(xisorad, yisorad, laturad)*rad2deg

    # Get indices of each U-cell and take the associated T-cells (same index).
    print("Getting nearest-neighbor indices.")
    fnameidxs1 = 'idx_step1_' + str(ISOB) + 'm.npz'
    if isfile(fnameidxs1):
        Uij = np.load(fnameidxs1)['Uij'].tolist()
    else:
        Uij = [nearidx2(lonu, latu, x0, y0) for x0, y0 in zip(xuu, yuu)]
        np.savez(fnameidxs1, Uij=Uij)

    Jc, Ic = [], []
    distisonnU = [] # [0]
    distisonnT = [] # [0]
    nmax = len(Uij) - 1
    for n in range(nmax):
        nn = n + 1
        print("%d of %d"%(nn,nmax))
        ic, jc = Uij[n]
        icp, jcp = Uij[nn]
        jc = _wrapj(jc, nx)
        jcp = _wrapj(jcp, nx)

        x0, y0 = lonu[ic, jc], latu[ic, jc]
        x0p, y0p = lonu[icp, jcp], latu[icp, jcp]
        hdg_init = LatLon_sphere(y0, x0).bearingTo(LatLon_sphere(y0p, x0p))
        hdg_init = -wrap180(hdg_init - 90)
        quadm = getquad(hdg_init)

        try:
            m.plot(x0, y0, 'y', linestyle='none', marker=',', alpha=1, zorder=10, latlon=True)
        except NameError:
            m = bmap_antarctica(ax, resolution='l')

        cont=0
        # print(jc,jcp)
        # print(ic,icp)
        if np.logical_and(jc==jcp, ic==icp):
            if nn==nmax: # Last point
                Ic.append(ic)
                Jc.append(jc)
                distisonnU, distisonnT = _update_distances(distisonnU, distisonnT, HTE, HTN, HUW, HUS, (icp, jcp), (ic, jc))
            else:
                pass
        else:
            while np.logical_or(jc!=jcp, ic!=icp):
                cont+=1
                # print(cont)
                Ic.append(ic)
                Jc.append(jc)

                # Compass degrees to [-180, 180].
                hdgi = -wrap180(LatLon_sphere(y0, x0).bearingTo(LatLon_sphere(y0p, x0p)) - 90)
                iwlk, jwlk, quad0 = getstep(hdgi, hdg_init, quadm)
                quadm = quad0
                #
                im, jm = ic, jc
                ic = ic + iwlk
                jc = jc + jwlk
                jc = _wrapj(jc, nx)
                jm = _wrapj(jm, nx)

                # Accumulate distances.
                distisonnU, distisonnT = _update_distances(distisonnU, distisonnT,
                HTE, HTN, HUW, HUS, (ic, jc), (im, jm))

                # update current point's lon, lat to get the
                # new headin on the next iteration.
                x0, y0 = lonu[ic, jc], latu[ic, jc]

                # Remove dead-end tails.
                if n>=2 and np.logical_and(ic==Ic[-2], jc==Jc[-2]):
                    _, _ = Ic.pop(), Jc.pop()
                    _, _ = Ic.pop(), Jc.pop()
                    _, _ = distisonnU.pop(), distisonnT.pop()
                    _, _ = distisonnU.pop(), distisonnT.pop()

    Ic, Jc = map(np.array, (Ic, Jc))
    distisonnU, distisonnT = map(np.array, (distisonnU, distisonnT))
    distisonnU = np.cumsum(distisonnU)*m2km # [km].
    distisonnT = np.cumsum(distisonnT)*m2km # [km].

    # Replace original isobath with step-by-step points.
    xisonnU = lonu[Ic, Jc]
    yisonnU = latu[Ic, Jc]
    xisonnT = lont[Ic, Jc]
    yisonnT = latt[Ic, Jc]

    print("replace the nn indices with the zig-zag indices.")
    fnameidxs2 = 'idx_step2_' + str(ISOB) + 'm.npz'
    if isfile(fnameidxs2):
        Uij = np.load(fnameidxs2)['Uij'].tolist()
    else:
        Uij = [nearidx2(lonu, latu, x0, y0) for x0, y0 in zip(xisonnU, yisonnU)]
        np.savez(fnameidxs2, Uij=Uij)

    # Compare distance axes again after nn-interpolating.
    fig, ax = plt.subplots()
    ax.plot(distiso, 'r', marker='o', ms=2, label=r'distiso')
    ax.plot(distisonnT, 'm', marker='o', ms=2, label=r'distisonnT')
    ax.plot(distisonnU, 'c', marker='o', ms=2, label=r'distisonnU')
    ax.grid()
    ax.legend(loc='best')
    ax.set_title('%d m isobath'%int(ISOB), fontsize=18, fontweight='black')
    ax.set_xlabel('Index [unitless]', fontsize=18, fontweight='black')
    ax.set_ylabel('Along-isobath distance [km]', fontsize=18, fontweight='black')

    # Compare isobaths again after subsetting.
    fig, ax = plt.subplots()
    ax.set_title('%d m isobath'%int(ISOB), fontsize=18, fontweight='black')
    ax.plot(xiso_orig, yiso_orig, 'grey', linestyle='--', marker='.', alpha=0.7, label=r'before evenly spacing')
    ax.plot(xiso, yiso, 'k', linestyle='-', marker='.', alpha=0.7, label='evenly spaced')
    ax.plot(xisonnT, yisonnT, 'r', linestyle='-', marker='.', alpha=1, label='nn-interp T-pts (model grid)')
    ax.plot(xisonnU, yisonnU, 'g', linestyle='-', marker='.', alpha=1, label='nn-interp U-pts (model grid)')
    ax.axis('tight')
    ax.grid()
    ax.legend(loc='best')

    # Plot extracted isobaths over bottom topography from POP.
    if True:
        fig, ax = plt.subplots()
        ax.set_title('%d m isobath'%int(ISOB), fontsize=18, fontweight='black')
        m = bmap_antarctica(ax, resolution='l')
        cs = m.pcolormesh(lont, latt, ht, cmap=cmo.deep, latlon=True, zorder=4)
        cc = m.contour(lont, latt, ht, isobs_map, colors='grey', latlon=True, zorder=4)
        m.plot(xiso, yiso, 'k', linestyle='-', marker='o', ms=4, latlon=True, zorder=5)
        m.plot(xisonnT, yisonnT, 'r', linestyle='--', marker='x', ms=4, alpha=1, zorder=5, label='T-pts', latlon=True)
        m.plot(xisonnU, yisonnU, 'm', linestyle='-', marker='x', ms=4, alpha=1, zorder=5, label='U-pts (boundary)', latlon=True)
        fmt_isobath(cc, manual=False)
        cbaxes = fig.add_axes([0.33, 0.18, 0.30, 0.015])
        cb = plt.colorbar(mappable=cs, cax=cbaxes, orientation='horizontal')
        cb.set_ticks(np.arange(0., 6000., 1000.))
        cb.update_ticks()
        cb.set_label(r'Local depth [m]', fontsize=14)
        plt.legend(loc='center')

        git_dir = repohash(return_gitobj=True).git_dir
        fhead, ftail = figname.split('.')
        figname = fhead + str(ISOB) + '.' + ftail
        savefig(figname, git_dir, format=ftail, bbox_inches='tight', dpi=125)

    # Isobath defined by U-points (i.e., by faces of T-cells).
    polyvolU = [LatLon_sphere(y0, x0) for y0, x0 in zip(yisonnU, xisonnU)]
    # Get masks for turning UeT and VnT into cross-isobath
    # flux velocities.
    Umsk, Vmsk, stepsUVmsk = get_xisobUVmsks(Ic, Jc, lont, latt, polyvolU, nx)

    # Save this isobath in xarray.Variable objs and move to next one.
    dims = 'distiso'
    attrsd = dict(units='km', long_name='Along-isobath distance')
    attrsx = dict(units='degrees East', long_name='Along-isobath longitude')
    attrsy = dict(units='degrees North', long_name='Along-isobath latitude')

    distiso = xr.Variable(dims, distiso, attrs=attrsd)
    xiso = xr.Variable(dims, xiso, attrs=attrsx)
    yiso = xr.Variable(dims, yiso, attrs=attrsy)

    # Save "sawtooth" isobath (made of nearest model grid points).
    attrsdnnU = dict(units='km', long_name='Along-isobath distance (nearest-neighbor model U-grid points)')
    attrsdnnT = dict(units='km', long_name='Along-isobath distance (nearest-neighbor model T-grid points)')
    attrsxnnU = dict(units='degrees East', long_name='Along-isobath longitude (nearest-neighbor model U-grid points)')
    attrsynnU = dict(units='degrees North', long_name='Along-isobath latitude (nearest-neighbor model U-grid points)')
    attrsxnnT = dict(units='degrees East', long_name='Along-isobath longitude (nearest-neighbor model T-grid points)')
    attrsynnT = dict(units='degrees North', long_name='Along-isobath latitude (nearest-neighbor model T-grid points)')
    attrsidxi = dict(units='unitless', long_name='Row index (nearest-neighbor model T- and U-grid points)')
    attrsidxj = dict(units='unitless', long_name='Column index (nearest-neighbor model T- and U-grid points)')
    attrsUmsk = dict(units='unitless', long_name='Mask array that converts the U-component of the flux velocity at the EAST face of T-cells into its contribution to the cross-isobath flux velocity.')
    attrsVmsk = dict(units='unitless', long_name='Mask array that converts the V-component of the flux velocity at the NORTH face of T-cells into its contribution to the cross-isobath flux velocity.')

    Uaux = np.array(Uij)
    Uiv = xr.Variable(dims, Uaux[:,0], attrs=attrsidxi)
    Ujv = xr.Variable(dims, Uaux[:,1], attrs=attrsidxj)
    Umsk = xr.Variable(dims, Umsk, attrs=attrsUmsk)
    Vmsk = xr.Variable(dims, Vmsk, attrs=attrsVmsk)
    xisonnU = xr.Variable(dims, xisonnU, attrs=attrsxnnU)
    yisonnU = xr.Variable(dims, yisonnU, attrs=attrsynnU)
    xisonnT = xr.Variable(dims, xisonnT, attrs=attrsxnnT)
    yisonnT = xr.Variable(dims, yisonnT, attrs=attrsynnT)
    distisonnU = xr.Variable(dims, distisonnU, attrs=attrsdnnU)
    distisonnT = xr.Variable(dims, distisonnT, attrs=attrsdnnT)

    ## Save isobath coordinates.[-1, 0 or 1] mask array. one for U, one for V.
    Vars = dict(xiso=xiso, yiso=yiso)
    VarsnnU = dict(xiso=xisonnU, yiso=yisonnU, i=Uiv, j=Ujv) # Save indices too.
    VarsnnT = dict(xiso=xisonnT, yiso=yisonnT)
    VarsUxisob = dict(Umsk=Umsk, Vmsk=Vmsk)
    coords = dict(diso=distiso)
    coordsnnU = dict(diso=distisonnU)
    coordsnnT = dict(diso=distisonnT)
    groupname = '%d m isobath'%int(ISOB)
    groupnamennU = '%d m isobath (U-points)'%int(ISOB)
    groupnamennT = '%d m isobath (T-points)'%int(ISOB)
    groupnameUxisob = '%d m isobath (x-isobath U, V masks)'%int(ISOB)
    attrsg = stamp() # Add stamp and save.

    ds = xr.Dataset(Vars, coords=coords, attrs=attrsg)
    dsnnU = xr.Dataset(VarsnnU, coords=coordsnnU)
    dsnnT = xr.Dataset(VarsnnT, coords=coordsnnT)
    dsUxisob = xr.Dataset(VarsUxisob, coords=coordsnnU)

    if ISOB==ISOBS[0]:
        system('rm %s'%fname_out)
    try:
        ds.to_netcdf(fname_out, mode='a', group=groupname)
    except OSError:
        ds.to_netcdf(fname_out, mode='w', group=groupname)
    dsnnU.to_netcdf(fname_out, mode='a', group=groupnamennU)
    dsnnT.to_netcdf(fname_out, mode='a', group=groupnamennT)
    dsUxisob.to_netcdf(fname_out, mode='a', group=groupnameUxisob)

    plt.draw()
    plt.show()
    _ = input("Press any key to proceed to the next isobath.")
    plt.close('all')

system('sleep 3')
plt.close('all')
