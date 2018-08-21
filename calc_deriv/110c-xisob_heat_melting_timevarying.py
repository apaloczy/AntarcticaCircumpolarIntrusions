# -*- coding: utf-8 -*-
#
# Description: Calculate time-varying cross-isobath
#              heat transports.
#
# Author:      André Palóczy Filho
# E-mail:      paloczy@gmail.com
# Date:        December/2017

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
from gsw import SA_from_SP, CT_from_pt, p_from_z
import gsw
from seawater import fp as Tinsitu_freezing_eos80
from seawater import ptmp
from pandas import Timestamp

import pyximport; pyximport.install()
from get_xisobvel import xisobvel
from get_mskgeo import mskgeo


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

CALC_minimumTf_ONLY = False

# Offshore and inshore isobaths bounding the strip.
isob = 1000 # [2500, 2000, 1000, 800]
# In [81]: Tfmins.min()
# Out[81]: -2.638742253002648
Tf0 = -2.638742253002648 # [degC] use the minimum freezing temperature found on the wanted isobath during the entire run.

# Start and end years to calculate the time series with.
START_YEAR = 1959
END_YEAR = 1968

fname_isobs = 'isobaths.nc'
fname_msk = 'volmsk%dm.npz'%isob
head_fin = '/lustre/atlas1/cli115/proj-shared/ia_top_tx0.1_v2_60yrs/'
head_fout = '/lustre/atlas1/cli115/proj-shared/apaloczy/monthly_interp/'

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
T2Y = 1e-9 # [Y(something)/T(something)].

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

# Indices and x-isobath transport masks.
Umsk = ncx['%d m isobath (x-isobath U, V masks)'%isob]['Umsk'][:]
Vmsk = ncx['%d m isobath (x-isobath U, V masks)'%isob]['Vmsk'][:]
ii = ncx['%d m isobath (U-points)'%isob]['i'][:]
ji = ncx['%d m isobath (U-points)'%isob]['j'][:]

### Convert row indices to the UeT, TeT, UET, VNT, etc arrays.
### Their first row was removed. So now the second row is the
### first row, etc.
ii = ii - 1

# Load mask to select the grid points inside the volume.
msk = np.bool8(np.load(fname_msk)['volmsk'])
TAREAvol = TAREA[msk]
TAREA = TAREA[np.newaxis, ...]

segs_lims_heat_transp = {
'Amundsen':[-136., -100., -76., -64.],
'Bellingshausen':[-100., -75., -77., -60.],
'WAP':[-75., -53., -74., -60.],
'Weddell':[-53., -11., -78., -59.],
'W-EA':[-11., 91., -72., -60.],
'E-EA':[91., 165., -72., -60.],
'Ross':[165., -136., -79., -68.]
}

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

Tfmins = []
p = p_from_z(z, latu[msk].mean()) # lats closest to the isobath.
t = []
#
UQxm_circ_all = []
UQxe_circ_all = []
UQxm_100m_all = []
UQxe_100m_all = []
UQxm_100m_700m_all = []
UQxe_100m_700m_all = []
UQxm_700m_1000m_all = []
UQxe_700m_1000m_all = []
#
Ux_all = []
#
UQx_all = []
UQxm_all = []
UQxe_all = []
SHT_all = []
Qm_all = []
n=0
# for fnamen in fnames[-6:]:
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
        SHF = nc.variables['SHF'][0,...][:fcap,:][msk]
    except KeyError:
        print("Skipping corrupted file: %s"%fnamen)
        continue
    t.append(ti)

    Ssec = S[:, ii, ji] # Practical salinity.
    Tfm = Tinsitu_freezing_eos80(Ssec, p[:, np.newaxis])
    Tfm = ptmp(Ssec, Tfm, p[:, np.newaxis], pr=0) # In situ temperature of freezing to potential temperature of freezing (which is the model variable).
    fisob=z>-isob # Only down to the depth of the middle isobath.
    Tfmin = Tfm[fisob,:].min()
    print("")
    print("[%s] Minimum freezing T along %d m isobath:  %.3f degC"%(ti, isob, Tfmin))

    if CALC_minimumTf_ONLY:
        Tfmins.append(Tfmin)
        continue

    eta = nc.variables['SSH'][0,:fcap,:]*cm2m
    srfcvol_fac = np.ones((ny,nx))
    csi = eta/DZT0[0,:,:]
    srfcvol_fac = srfcvol_fac + csi

    DZU = _get_dV(DZU0, eta, TAREA, dz_only=True)
    dzt, dVt = _get_dV(DZT0, eta, TAREA)  # [m3].

    dVtvol = dVt[:, msk]
    TmTf = T[:, msk] - Tf0 # Tf0 is pot. temp.
    Qm = np.sum(TmTf*dVtvol)*T2Q*T2Y # [YJ], volume-integrated heat content in the volume bounded by the isobath.
    SHT = np.sum(SHF*TAREAvol)*W2TW # [W]

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
    UET0 = nc.variables['UET'][0,...][:,:fcap,:] # u * Potential temperature referenced at 0 dbar.
    VNT0 = nc.variables['VNT'][0,...][:,:fcap,:]
    UET0 = applymsk(UET0)
    VNT0 = applymsk(VNT0)

    # Remove contribution from volume-averaged temperature.
    UET0 = UET0*dVt # [m3*degC/s].
    VNT0 = VNT0*dVt # [m3*degC/s].
    UET = np.sum(UET0, axis=0)
    VNT = np.sum(VNT0, axis=0)
    UET = UET[1:,:]
    VNT = VNT[1:,:]
    # Assuming UeT and Tf0 are uncorrelated (Tf0 constant).
    #
    UeTTeT0 = UeT0*(TeT - Tf0)
    VnTTnT0 = VnT0*(TnT - Tf0)
    UeTTeT = np.sum(UeTTeT0, axis=0) # [m3*degC/s].
    VnTTnT = np.sum(VnTTnT0, axis=0) # [m3*degC/s].

    # Volume transport.
    Ux = UeT[ii, ji]*Umsk + VnT[ii, ji]*Vmsk # [m3/s].
    Ux = Ux*1e-6 # [Sv].

    # Extract the cross-isobath temperature transports [m3*degC/s].
    UTx = UET[ii, ji]*Umsk + VNT[ii, ji]*Vmsk          # Total.
    UTxm = UeTTeT[ii, ji]*Umsk + VnTTnT[ii, ji]*Vmsk # Mean.
    UTxe = UTx - UTxm                                  # Eddy.

    # Heat transports [TW].
    UQx = UTx*T2Q
    UQxm = UTxm*T2Q
    UQxe = UTxe*T2Q
    n+=1

    print("")
    print("Area-integrated surface heat flux:     %5.2f TW"%SHT)
    print("Cross-isobath heat transport (NET):    %5.2f TW"%UQx.sum())
    print("Cross-isobath heat transport (MEAN):   %5.2f TW"%UQxm.sum())
    print("Cross-isobath heat transport (EDDY):   %5.2f TW"%UQxe.sum())

    # Calculate mean and eddy heat fluxes in the top, middle and bottom layers.
    fz_100m = z>-100
    fz_100m_700m = np.logical_and(z<=-100, z>-700)
    fz_700m_1000m = np.logical_and(z<=-700, z>-1000)
    UTxm_100m, UTxe_100m = _UT_layers(UET0, VNT0, UeTTeT0, VnTTnT0, Umsk, Vmsk, ii, ji, fz_100m)
    UTxm_100m_700m, UTxe_100m_700m = _UT_layers(UET0, VNT0, UeTTeT0, VnTTnT0, Umsk, Vmsk, ii, ji, fz_100m_700m)
    UTxm_700m_1000m, UTxe_700m_1000m = _UT_layers(UET0, VNT0, UeTTeT0, VnTTnT0, Umsk, Vmsk, ii, ji, fz_700m_1000m)

    # Save also full heat transport profile.
    UTx = UET0[:, ii, ji]*Umsk + VNT0[:, ii, ji]*Vmsk             # Total.
    UTxm = UeTTeT0[:, ii, ji]*Umsk + VnTTnT0[:, ii, ji]*Vmsk    # Mean.
    UTxe = UTx - UTxm                                             # Eddy.
    UTxm_circ = UTxm*T2Q
    UTxe_circ = UTxe*T2Q
    #
    UTxm_100m = UTxm_100m.sum(axis=0)*T2Q
    UTxe_100m = UTxe_100m.sum(axis=0)*T2Q
    UTxm_100m_700m = UTxm_100m_700m.sum(axis=0)*T2Q
    UTxe_100m_700m = UTxe_100m_700m.sum(axis=0)*T2Q
    UTxm_700m_1000m = UTxm_700m_1000m.sum(axis=0)*T2Q
    UTxe_700m_1000m = UTxe_700m_1000m.sum(axis=0)*T2Q

    UQxm_circ_all.append(UTxm_circ)
    UQxe_circ_all.append(UTxe_circ)
    UQxm_100m_all.append(UTxm_100m)
    UQxe_100m_all.append(UTxe_100m)
    UQxm_100m_700m_all.append(UTxm_100m_700m)
    UQxe_100m_700m_all.append(UTxe_100m_700m)
    UQxm_700m_1000m_all.append(UTxm_700m_1000m)
    UQxe_700m_1000m_all.append(UTxe_700m_1000m)

    UQx_all.append(UQx)
    UQxm_all.append(UQxm)
    UQxe_all.append(UQxe)
    Qm_all.append(Qm)
    SHT_all.append(SHT)
    #
    Ux_all.append(Ux)

# Save time series of UQ.
t = np.array([Timestamp(tt).to_pydatetime() for tt in t])
ii = ii + 1

if CALC_minimumTf_ONLY:
    zvars = dict(t=t, Tfmin=Tfmins)
    fnamez = 'Tfmin_tseries%dm_%d-%d.npz'%(isob, START_YEAR, END_YEAR)
else:
    fnamez = 'hflxmelt_tseries%dm_%d-%d.npz'%(isob, START_YEAR, END_YEAR)
    zvars = dict(t=t, z=z, d=dui, x=xui, y=yui, i=ii, j=ji, Ux=Ux_all, SHT=SHT_all, Qm=Qm_all,
                UQxm_circ=UQxm_circ_all, UQxe_circ=UQxe_circ_all,
                UQx=UQx_all, UQxm=UQxm_all, UQxe=UQxe_all, UQxm_100m=UQxm_100m_all, UQxe_100m=UQxe_100m_all,
                UQxm_100m_700m=UQxm_100m_700m_all, UQxe_100m_700m=UQxe_100m_700m_all, UQxm_700m_1000m=UQxm_700m_1000m_all,
                UQxe_700m_1000m=UQxe_700m_1000m_all, Tf0=Tf0)

for kzv in zvars.keys():
    zvars.update({kzv:stripmsk(zvars[kzv])})
np.savez(fnamez, **zvars)
