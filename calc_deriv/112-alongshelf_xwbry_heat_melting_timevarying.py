# -*- coding: utf-8 -*-
#
# Description: Calculate time-varying along-shelf
#              heat transports across W boundaries
#              of segments.
#
# Author:      André Palóczy Filho
# E-mail:      paloczy@gmail.com
# Date:        February/2018

import numpy as np
import matplotlib
matplotlib.use('Agg') # Uncomment if running in batch exec.
from matplotlib import pyplot as plt
from glob import glob
from os import system
from mpl_toolkits.basemap import Basemap
from scipy.integrate import simps
from netCDF4 import Dataset
from ap_tools.utils import bbox2ij, rot_vec, xy2dist
from ap_tools.utils import fmt_isobath, lon360to180, lon180to360
from ap_tools.utils import get_isobath
import time
from datetime import datetime
from os.path import isfile
# from stripack import trmesh
from stripy.spherical import sTriangulation as trmesh
from pygeodesy.sphericalNvector import LatLon as LatLon_sphere
from pygeodesy.utils import wrap180
from matplotlib.patches import Polygon
import pickle
from gsw import p_from_z
import gsw
from pandas import Timestamp


def stripmsk(arr):
    if np.ma.isMA(arr):
        return arr.data
    else:
        return arr


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

def bmap_antarctica(ax, resolution='i', projection='stereographic'):
    """
    Full Antartica basemap (Polar Stereographic Projection).
    """
    if projection=='stereographic':
        m = Basemap(boundinglat=-60,
                    lon_0=60,
                    projection='spstere',
                    resolution=resolution,
                    ax=ax)
    elif projection=='cyl':
        m = Basemap(llcrnrlon=-180, urcrnrlon=180,
                    llcrnrlat=-90, urcrnrlat=-55,
                    projection=projection,
                    resolution=resolution,
                    ax=ax)

    m.fillcontinents(color='0.9', zorder=9)
    m.drawcoastlines(zorder=10)
    m.drawmapboundary(zorder=9999)
    if projection=='stereographic':
        m.drawmeridians(np.arange(-180, 180, 20), linewidth=0.15, labels=[1, 1, 1, 1], zorder=12)
        m.drawparallels(np.arange(-90, -50, 5), linewidth=0.15, labels=[0, 0, 0, 0], zorder=12)

    return m

def applymsk(arr, thresh=1e30, squeeze=False):
    if squeeze:
        arr = np.squeeze(arr)
    arr = np.ma.masked_greater(arr, thresh)
    return np.ma.masked_less(arr, -thresh)

def add_cycl_col(arr, end=True):
    if end:
        return np.concatenate((arr, arr[...,0][...,np.newaxis]), axis=-1)
    else:
        return np.concatenate((arr[...,-1][...,np.newaxis], arr), axis=-1)

def _get_dV(dd, ssh, dA, dz_only=False):
    dd = applymsk(dd)
    ssh = applymsk(ssh) # IMPORTANT (mask out fill values like 1e36)!!!
    dd[0,...] = dd[0,...] + ssh # [m3].

    if dz_only:
        return dd         # [m3].
    else:
        dA = applymsk(dA)
        return dd, dA*dd  # [m3].

def _UT_layers(UET0, VNT0, UeTTeT0, VnTTnT0, Umsk, Vmsk, ii, ji, fz):
    UTx = UET0[:, ii, ji][fz, :]*Umsk + VNT0[:, ii, ji][fz, :]*Vmsk             # Total.
    UTxm = UeTTeT0[:, ii, ji][fz, :]*Umsk + VnTTnT0[:, ii, ji][fz, :]*Vmsk    # Mean.
    UTxe = UTx - UTxm                                               # Eddy.

    return UTxm, UTxe

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
##---
plt.close('all')

# Offshore and inshore isobaths bounding the strip.
isob = 1000 # [2500, 2000, 1000, 800]
Tf0 = -2.638742253002648 # [degC] use the minimum freezing temperature found on the wanted isobath during the entire run.

# Start and end years to calculate the time series with.
START_YEAR = 1959
END_YEAR = 1968

segnames = ['S-AP', 'N-AP', 'E-EA', 'Byrd', 'Bellingshausen', 'W-EA', 'Weddell', 'Amundsen', 'C-EA', 'Ross']
fname_isobs = 'isobaths.nc'
fname_msk = 'volmsk%dm.npz'%isob
head_fin = '/lustre/atlas1/cli115/proj-shared/ia_top_tx0.1_v2_60yrs/'
head_fout = '/lustre/atlas1/cli115/proj-shared/apaloczy/monthly_interp/'

fname_lonlatidxs = 'segment_bdry_lonlatgrid_indices.npz'
p5 = 0.5
W2TW = 1e-12
T2Y = 1e-9
cm2m = 1e-2          # m/cm.
m3toSv = 1e-6        # Sv/(m3/s).
deg2rad = np.pi/180. # rad/deg.
rad2deg = 1/deg2rad
fcap = 501
thresh = 1e30
fmt = 'png'
figname = 'isobaths_antarctica.' + fmt
isobs_map = [1000., 2500., 4000.]
# head = '/lustre/atlas1/cli115/proj-shared/apaloczy/POP_2005-2009_from_monthly_outputs/'
# fname = 'SOsubset_avg_2005-2009.nc'
head = '/lustre/atlas1/cli115/proj-shared/apaloczy/POP_1969-2009_from_monthly_outputs/'
fname = 'SOsubset_avg_1969-2009.nc'
fname_ncgrid = head + fname

rho_sw = 1.026*1000 # [kg/m3].
cp0 = 39960000.0    # [erg/g/K].
cp0 = cp0*1e-7*1e3  # [J/kg/degC].
T2Q = rho_sw*cp0*W2TW
W2TW = 1e-12
T2Y = 1e-9 # [Y(something)/T(something)].

didx = np.load(fname_lonlatidxs)
wbryilo, wbryila = didx['wbry_idxlon'].flatten()[0], didx['wbry_idxlat'].flatten()[0]

### Convert row indices to the UeT, TeT, UET, VNT, etc arrays.
### Their first row was removed. So now the second row is the
### first row, etc.
#ii = ii - 1
for seg in wbryila.keys():
    ilow, iupp = wbryila[seg]
    wbryila.update({seg:(ilow-1, iupp-1)})

fname = head + fname
fname_dzu_kzit = 'POP-dzu_dzt_kzit_subsetSO.nc'
# Get grid variables (dzu and kzit).
ncg = Dataset(fname_dzu_kzit)
DZU0 = ncg.variables['dzu'][:]*cm2m
DZT0 = ncg.variables['dzt'][:]*cm2m
nz, ny, nx = DZU0.shape

nc = Dataset(fname)
DXU = nc.variables['DXU'][:fcap, :]*cm2m
DYU = nc.variables['DYU'][:fcap, :]*cm2m
DXT = nc.variables['DXT'][:fcap, :]*cm2m
DYT = nc.variables['DYT'][:fcap, :]*cm2m
TAREA = nc.variables['TAREA'][:fcap, :]*cm2m**2
z = -nc.variables['z_t'][:]*cm2m
ht = ncg.variables['h_t'][:]*cm2m
latu = nc.variables['ULAT'][:fcap, :]
lonu = nc.variables['ULONG'][:fcap, :]
latt = nc.variables['TLAT'][:fcap, :]
lont = nc.variables['TLONG'][:fcap, :]

# Get discretized isobaths.
ncx = Dataset(fname_isobs)
xui = ncx["%d m isobath (U-points)"%isob]['xiso'][:]
yui = ncx["%d m isobath (U-points)"%isob]['yiso'][:]
dui = ncx["%d m isobath (U-points)"%isob]['diso'][:]

# Load mask to select the grid points inside the volume.
msk = np.bool8(np.load(fname_msk)['volmsk'])
TAREAvol = TAREA[msk]
TAREA = TAREA[np.newaxis, ...]

fdir_tail = '/ocn/hist/ia_top_tx0.1_v2_yel_patc_1948_intel.pop.h.????-??.nc'
fdirs = glob(head_fin+'ia_top_tx0.1_v2_yel_patc_1948_intel_def_year_????')
fdirs.sort()
if not isinstance(fdirs, list):
    fdirs = [fdirs]

fnames = []
for fdir in fdirs:
    ystr = int(fdir[-4:])
    if np.logical_or(ystr<START_YEAR, ystr>END_YEAR):
        continue
    fnamesi = glob(fdir + fdir_tail)
    fnamesi.sort()
    for f in fnamesi:
        fnames.append(f)
nt = len(fnames) # Total number of files.

p = p_from_z(z, latu[msk].mean()) # lats closest to the isobath.
t = []
#

UQx_segs = dict()
UQxm_segs = dict()
UQxe_segs = dict()
Ux_segs = dict()
for seg in segnames:
    UQx_segs.update({seg:[]})
    UQxm_segs.update({seg:[]})
    UQxe_segs.update({seg:[]})
    Ux_segs.update({seg:[]})

n=0
for fnamen in fnames:
    ti = fnamen.split('/')[-1].split('.')[-2]
    system("echo %s > counter_tseries.txt"%ti)
    print(ti)
    nc = Dataset(fnamen)
    try:
        U = nc.variables['UVEL'][0,...][:,:fcap,:]*cm2m
        V = nc.variables['VVEL'][0,...][:,:fcap,:]*cm2m
        T = nc.variables['TEMP'][0,...][:,:fcap,:]
        S = nc.variables['SALT'][0,...][:,:fcap,:]
    except KeyError:
        print("Skipping corrupted file: %s"%fnamen)
        continue
    t.append(ti)

    eta = nc.variables['SSH'][0,:fcap,:]*cm2m
    srfcvol_fac = np.ones((ny,nx))
    csi = eta/DZT0[0,:,:]
    srfcvol_fac = srfcvol_fac + csi

    DZU = _get_dV(DZU0, eta, TAREA, dz_only=True)
    dzt, dVt = _get_dV(DZT0, eta, TAREA)  # [m3].

    # Add cyclic column on the variables that will be averaged in the zonal direction.
    Vc = applymsk(add_cycl_col(V, end=False))
    DXUc = applymsk(add_cycl_col(DXU, end=False))
    DZUc = applymsk(add_cycl_col(DZU, end=False))
    Tc = applymsk(add_cycl_col(T, end=True))

    UeT = p5*(U[:,:-1,:]*DYU[:-1,:]*DZU[:,:-1,:] + U[:,1:,:]*DYU[1:,:]*DZU[:,1:,:])
    VnT = p5*(Vc[:,:,:-1]*DXUc[:,:-1]*DZUc[:,:,:-1] + Vc[:,:,1:]*DXUc[:,1:]*DZUc[:,:,1:])
    TnT = p5*(T[:,:-1,:] + T[:,1:,:])   # T at NORTH face of T-cells.
    TeT = p5*(Tc[:,:,:-1] + Tc[:,:,1:]) # T at EAST face of T-cells.
    #
    toprow0 = np.zeros((nz, 1, nx))
    TnT = np.concatenate((TnT, toprow0), axis=1)
    # TsT = TnT[:,:-1,:]                   # [m3/s]. Starting at j = 2 (second T-row).
    TnT = TnT[:,1:,:]                    # [m3/s]. Last row is just a dummy row of zeroes.
    TeT = TeT[:,1:,:]
    TeT = add_cycl_col(TeT, end=False)
    # TwT = TeT[:,:,:-1]                   # [m3/s].
    TeT = TeT[:,:,1:]                    # [m3/s].
    #

    UeT[0,:,:] = UeT[0,:,:]*srfcvol_fac[1:,:] # [m3/s].
    VnT[0,:,:] = VnT[0,:,:]*srfcvol_fac       # [m3/s].

    # Adjust transport across faces.
    VnT = np.concatenate((VnT, toprow0), axis=1)
    # VsT = VnT[:,:-1,:]                   # [m3/s]. Starting at j = 2 (second T-row).
    VnT = VnT[:,1:,:]                    # [m3/s].
    VnT = VnT[:,:-1,:] # Cap off the last row (dummy row of zeroes).
    UeT = add_cycl_col(UeT, end=False)
    # UwT = UeT[:,:,:-1]                   # [m3/s].
    UeT = UeT[:,:,1:]                    # [m3/s].
    # Now all flux transports [m3/s] start at j = 2 (second T-row).
    # e.g., lont = lont[1:,:], VnT = VnT[:,1:,:], etc.

    # Make sure mask is there.
    UeT, VnT, TeT, TnT = map(applymsk, (UeT, VnT, TeT, TnT))

    #
    UeT0 = UeT.copy()
    VnT0 = VnT.copy()
    #
    UeT = UeT.sum(axis=0)
    VnT = VnT.sum(axis=0)

    # Time-mean and total temperature transports.
    UET0 = nc.variables['UET'][0,...][:,:fcap,:]
    VNT0 = nc.variables['VNT'][0,...][:,:fcap,:]
    UET0 = applymsk(UET0)
    VNT0 = applymsk(VNT0)

    UET0 = UET0*dVt # [m3*degC/s].
    VNT0 = VNT0*dVt # [m3*degC/s].
    UET = np.sum(UET0, axis=0)
    VNT = np.sum(VNT0, axis=0)
    UET = UET[1:,:]
    VNT = VNT[1:,:]
    # Assuming UeT and Tf0 are uncorrelated (Tf0 constant).
    # Remove contribution from Tf0 to the time-mean transport.
    UeTTeT0 = UeT0*(TeT - Tf0)
    VnTTnT0 = VnT0*(TnT - Tf0)
    UeTTeT = np.sum(UeTTeT0, axis=0) # [m3*degC/s].
    VnTTnT = np.sum(VnTTnT0, axis=0) # [m3*degC/s].

    # Extract the cross-line, along-shelf heat transports [TW] for the W boundary of each segment.
    for seg in segnames:
        ii, jj = wbryilo[seg], wbryila[seg]
        jj = slice(jj[0], jj[1])
        UQx = UET[jj, ii].sum()*T2Q      # Total.
        UQxm = UeTTeT[jj, ii].sum()*T2Q  # Mean.
        UQxe = UQx - UQxm                # Eddy.

        # Volume transport.
        Ux = UeT[jj, ii].sum()*1e-6 # [Sv].

        UQx_aux = UQx_segs[seg]
        UQxm_aux = UQxm_segs[seg]
        UQxe_aux = UQxe_segs[seg]
        Ux_aux = Ux_segs[seg]
        #
        UQx_aux.append(UQx)
        UQxm_aux.append(UQxm)
        UQxe_aux.append(UQxe)
        Ux_aux.append(Ux)
        #
        UQx_segs.update({seg:UQx_aux})
        UQxm_segs.update({seg:UQxm_aux})
        UQxe_segs.update({seg:UQxe_aux})
        Ux_segs.update({seg:Ux_aux})

    n+=1

# Save time series of UQ.
t = np.array([Timestamp(tt).to_datetime() for tt in t])
fnamez = 'hflxmelt_alongshelf_xwbry_tseries%dm_%d-%d.npz'%(isob, START_YEAR, END_YEAR)
ii = ii + 1

zvars = dict(t=t, UQx=UQx_segs, UQxm=UQxm_segs, UQxe=UQxe_segs, Ux=Ux_segs)

for kzv in zvars.keys():
    zvars.update({kzv:zvars[kzv]})
np.savez(fnamez, **zvars)
