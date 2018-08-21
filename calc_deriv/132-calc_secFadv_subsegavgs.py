# -*- coding: utf-8 -*-
#
# Description: Calculate the Neutral Density Temporal Residual Mean (NDTRM)
#              eddy streamfunction (Stewart & Thompson, 2015) along
#              cross-isobath sections. Average over some along-isobath
#              distance in different segments.
#
# Author:      André Palóczy Filho
# E-mail:      paloczy@gmail.com
# Date:        April/2018

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime
from seawater import alpha as alpha_eos80
from seawater import beta as beta_eos80
from gsw import SA_from_SP, p_from_z
from glob import glob
from pandas import Timestamp
from os import system
from sys import exit
import pickle
from ap_tools.utils import lon360to180, xy2dist, near
from reproducibility import savez, savefig


def _slicegh(arr, slcv, slch, average=False, how='mean'):
    if not average:
        if slch.start > slch.stop:
            return np.concatenate((arr[..., slcv, slice(slch.start, -1)], arr[..., slcv, slice(0, slch.stop)]), axis=-1)
        else:
            return arr[..., slcv, slch]
    else:
        if how=='mean':
            favg = np.mean
        elif how=='median':
            favg = np.median
        if slch.start > slch.stop:
            return favg(np.concatenate((arr[..., slcv, slice(slch.start, -1)], arr[..., slcv, slice(0, slch.stop)])), axis=-1)
        else:
            return favg(arr[..., slcv, slch], axis=-1)


def _applymsk(arr, thresh=1e30, squeeze=False):
    if squeeze:
        arr = np.squeeze(arr)
    arr = np.ma.masked_greater(arr, thresh)

    return np.ma.masked_less(arr, -thresh)


def _stripmsk(arr, mask_invalid=True):
    if mask_invalid:
        arr = np.ma.masked_invalid(arr)
    if np.ma.isMA(arr):
        msk = arr.mask
        arr = arr.data
        arr[msk] = np.nan

    return arr


def _add_cycl_col(arr, end=True):
    if end:
        return np.concatenate((arr, arr[...,0][...,np.newaxis]), axis=-1)
    else:
        return np.concatenate((arr[...,-1][...,np.newaxis], arr), axis=-1)


segnames = ['weddell_C', 'Maud', 'Amery_wE-EA', 'E-EA_E', 'Ross_W', 'E_Ross', 'Byrd_W', 'Amundsen_E', 'Bellingshausen_E']


def _get_DV(dd, ssh, dA, dz_only=False):
    dd = _applymsk(dd)
    ssh = _applymsk(ssh) # IMPORTANT (mask out fill values like 1e36)!!!
    dd[0,...] = dd[0,...] + ssh # [m3].

    if dz_only:
        return dd         # [m3].
    else:
        dA = _applymsk(dA)
        return dd, dA*dd  # [m3].


def _get_dV(dd0, ssh0, dA0, dz_only=False, segnames=segnames):
    dd, ssh, dA = dd0.copy(), ssh0.copy(), dA0.copy()
    _ = [dd.update({sseg:_applymsk(dd[sseg])[0, ...]}) for sseg in segnames]
    _ = [ssh.update({sseg:_applymsk(ssh[sseg])}) for sseg in segnames] # IMPORTANT (mask out fill values like 1e36)!!!
    _ = [dd.update({sseg:dd[sseg]+ssh[sseg]}) for sseg in segnames] # [m3].

    if dz_only:
        return dd         # [m3].
    else:
        _ = [dA.update({sseg:_applymsk(dA[sseg])}) for sseg in segnames]
        dAdd = dict()
        _ = [dAdd.update({sseg:dA[sseg]*dd[sseg]}) for sseg in segnames]
        return dd, dAdd  # [m3].


def get_TSflux(V, T, S, ETA, TAREA, DXU, DYU, DZU0, DZT0, laslices, loslices, Tf=None, segnames=segnames):
    ny, nx = DXU.shape
    nz = V.shape[0]
    srfcvol_fac = np.ones((ny,nx))
    csi = ETA/DZT0[0, :, :]
    srfcvol_fac = srfcvol_fac + csi
    DZU = _get_DV(DZU0, ETA, TAREA, dz_only=True)

    # Add cyclic column on the variables that will be averaged in the zonal direction.
    Vc = _applymsk(_add_cycl_col(V, end=False))
    DXUc = _applymsk(_add_cycl_col(DXU, end=False))
    DZUc = _applymsk(_add_cycl_col(DZU, end=False))

    p5 = 0.5
    VnT = p5*(Vc[:,:,:-1]*DXUc[:,:-1]*DZUc[:,:,:-1] + Vc[:,:,1:]*DXUc[:,1:]*DZUc[:,:,1:])
    TnT = p5*(T[:,:-1,:] + T[:,1:,:])   # T at NORTH face of T-cells.
    SnT = p5*(S[:,:-1,:] + S[:,1:,:])   # S at NORTH face of T-cells.
    VnT[0,:,:] = VnT[0,:,:]*srfcvol_fac # [m3/s].

    toprow0 = np.zeros((nz, 1, nx))
    toprow0yx = np.zeros((1, nx))
    TnT = np.concatenate((TnT, toprow0), axis=1)
    SnT = np.concatenate((SnT, toprow0), axis=1)
    TAREA = np.concatenate((TAREA, toprow0yx), axis=0)

    # Last row is just a dummy row of zeroes. To replace the one that went away when the meridional average was taken.
    TnT = TnT[:,1:,:]
    SnT = SnT[:,1:,:]
    TAREA = TAREA[1:,:]

    # Adjust transport across faces.
    VnT = np.concatenate((VnT, toprow0), axis=1)
    VnT = VnT[:,1:,:]
    # VnT = VnT[:,1:-1,:] # [m3/s]. Start at j = 2.
                        # Cap off the last row (dummy row of zeroes).

    loslices0 = loslices.copy()
    laslices0 = laslices.copy()
    TAREAij = _skeldict()
    # Now all flux transports [m3/s] start at j = 2 (second T-row).
    # e.g., lont = lont[1:,:], VnT = VnT[:,1:,:], etc.
    # Im = Im - 1
    for sseg in segnames:
        laslices0.update({sseg:slice(laslices[sseg].start-1, laslices[sseg].stop-1)})
        TAREAij.update({sseg:_slicegh(TAREA, laslices0[sseg], loslices0[sseg])})

    VnT, TnT, SnT = map(_applymsk, (VnT, TnT, SnT))
    # Assuming UeT and Tf0 are uncorrelated (Tf0 constant).
    TnTij, VnTij, VNTbar_minusTf = dict(), dict(), dict()
    SnTij, VNTbarij, VNSbarij = dict(), dict(), dict()

    # Change units back to m/s to be consistent with UET, UES and VNT, VNS.
    _ = [VnTij.update({sseg:_slicegh(VnT, laslices0[sseg], loslices0[sseg])/TAREAij[sseg]}) for sseg in segnames] # [m/s].
    _ = [TnTij.update({sseg:_slicegh(TnT, laslices0[sseg], loslices0[sseg])}) for sseg in segnames]
    _ = [VNTbar_minusTf.update({sseg:VnTij[sseg]*(TnTij[sseg] - Tf0)}) for sseg in segnames]                    # [degC*m/s].

    # Calculate mean T and S fluxes across EAST and NORTH faces of T-cells.
    _ = [SnTij.update({sseg:_slicegh(SnT, laslices0[sseg], loslices0[sseg])}) for sseg in segnames]
    _ = [VNTbarij.update({sseg:VnTij[sseg]*TnTij[sseg]}) for sseg in segnames] # [degC*m/s]
    _ = [VNSbarij.update({sseg:VnTij[sseg]*SnTij[sseg]}) for sseg in segnames] # [psu*m/s]

    # Remove along-shelf average.
    _ = [VNTbarij.update({sseg:(VNTbarij[sseg] - VNTbarij[sseg].mean(axis=-1)[...,np.newaxis])}) for sseg in segnames]
    _ = [VNSbarij.update({sseg:(VNSbarij[sseg] - VNSbarij[sseg].mean(axis=-1)[...,np.newaxis])}) for sseg in segnames]

    return VNTbarij, VNSbarij, VNTbar_minusTf

##---
plt.close('all')

# Load bounding longitudes of the subsegments.
# fname_lonlats = 'segment_bdry_lonlatgrid_indices_for_streamfunction_calculations.npz'
# segbrys_psi = np.load(fname_lonlats)['segbrys_psi'].flatten()[0]
##### Avoid the 110 W line.
# segbrys_psi.update({'Amundsen_E':(-115.0, -110.5)})

# segbrys_psi = {'Amery_wE-EA': (55, 120),
#                'Amundsen_E': (-115.0, -110.5),
#                'Bellingshausen_E': (-86.0, -75.0),
#                'Byrd_W': (-150, -135),
#                'E-EA_E': (122.7, 153.7),
#                'E_Ross': (-173, -153),
#                'Maud': (20, 60),
#                'Ross_W': (165.0, 180.0),
#                'weddell_C': (-50.0, -25.0)}

segbrys_psi = {'Amery_wE-EA': (70, 95),
               'Amundsen_E': (-115, -110.5),
               'Bellingshausen_E': (-85, -75),
               'Byrd_W': (-150, -135),
               'E-EA_E': (135, 150),
               'E_Ross': (-180, -155),
               'Maud': (25, 50),
               'Ross_W': (155, 180),
               'weddell_C': (-50, -25)}


YEAR_START = 1959#1969
isob = 1000
fnamez_out_psi = 'sec_subsegments_psimean_psieddy_uqxm_uqxe.npz'
fnamez_out_psimean = 'sec_subsegments_psimean.npz'

head_fin = '/lustre/atlas1/cli115/proj-shared/ia_top_tx0.1_v2_60yrs/'
tail_file = 'ia_top_tx0.1_v2_yel_patc_1948_intel.pop.h.????-??.nc'

fdirs = glob(head_fin+'ia_top_tx0.1_v2_yel_patc_1948_intel_def_year_????/ocn/hist/')
fdirs.sort()
if not isinstance(fdirs, list):
    fdirs = [fdirs]
fnames = []
for fdir in fdirs:
    if int(fdir.split('/ocn/')[0][-4:])<YEAR_START:
        continue
    fnamesi = glob(fdir + tail_file)
    fnamesi.sort()
    for f in fnamesi:
        fnames.append(f)
nt = len(fnames) # Total number of files.

fcap = 501
cm2m = 1e-2
W2TW = 1e-12
rho_sw = 1026.0     # [kg/m3].
cp0 = 39960000.0    # [erg/g/K].
cp0 = cp0*1e-7*1e3  # [J/kg/degC].
T2Q = rho_sw*cp0*W2TW
Tf0 = -2.6431782684654603 # [degC] use the minimum freezing temperature found on the wanted isobath during the entire run.
CALC_PSIMEAN_ONLY = False

# Calculate subsegment averages of h(x), pseudo-cross isobath distance
# and the south and north row indices for each subsegment.
nc0 = Dataset(fnames[0])
ht = nc0.variables['HT'][:fcap,:]*cm2m # [m].
lattx = nc0.variables['TLAT'][:fcap,:]
lontx = lon360to180(nc0.variables['TLONG'][:fcap,:])
lontx0 = lontx[100,:]
hshallow, hdeep = 100, 2000
fhshs = dict()
fhdeeps = dict()
xdists = dict()
hx_yavgs = dict()
loslices = dict()
laslices = dict()
Lattxs = dict()
Lontxs = dict()
Lys = dict()
nlat_stretch = 25
for subseg in segbrys_psi.keys():
    lol, lor = segbrys_psi[subseg]
    flol, flor = near(lontx0, lol, return_index=True), near(lontx0, lor, return_index=True)
    if flol>flor:
        ht_yavg = np.hstack((ht[:,flol:], ht[:,:flor])).mean(axis=1)
    else:
        ht_yavg = ht[:,flol:flor].mean(axis=1)
    fhsh = near(ht_yavg, hshallow, return_index=True)
    fhdeep = near(ht_yavg, hdeep, return_index=True)
    if subseg not in ['Maud', 'Byrd_W']:
        fhdeep += nlat_stretch
    ht_yavg = ht_yavg[fhsh:fhdeep]
    lom = np.array([0.5*(lol + lor)]*(fhdeep - fhsh))
    if flol>flor:
        lattx0 = np.hstack((lattx[fhsh:fhdeep,flol:], lattx[fhsh:fhdeep,:flor])).mean(axis=1)
    else:
        lattx0 = lattx[fhsh:fhdeep, flol:flor].mean(axis=1)
    xdist = xy2dist(lom, lattx0)*1e-3 # [km].
    lattx00 = lattx0.mean()
    Ly = xy2dist([lol, lor], [lattx00, lattx00])[-1]*1e-3 # [km] width of segment.
    fx1000 = near(ht_yavg, isob, return_index=True)
    xdist = xdist - xdist[fx1000] # pseudo-cross-isobath distance. Origin is at the pseudo 1000 m isobath.
    xdist = -xdist # x coordinate increases onshore.
    xdists.update({subseg:xdist})
    hx_yavgs.update({subseg:ht_yavg})
    fhshs.update({subseg:fhsh})
    fhdeeps.update({subseg:fhdeep})
    loslices.update({subseg:slice(flol, flor)})
    laslices.update({subseg:slice(fhsh, fhdeep)})
    Lattxs.update({subseg:lattx0})
    Lontxs.update({subseg:lom})
    Lys.update({subseg:Ly})

# plt.plot(xdist, -ht_yavg, label=subseg.replace('_',' '))
# plt.grid(True)
# plt.legend()
# plt.show()
# savefig('hx_yavgs_profiles.png')
# stop

# Get indices of all sections and concatenate.


def get_Ux_profile(V, eta, TAREA, DXT, DXU, DYU, DZU0, DZT0, slcvs=laslices, slchs=loslices, segnames=segnames):
    DZU = _get_DV(DZU0, eta, TAREA, dz_only=True)
    dzt, dVt = _get_DV(DZT0, eta, TAREA)  # [m3].
    _, ny, nx = DZT0.shape
    srfcvol_fac = np.ones((ny,nx))
    csi = eta/DZT0[0,:,:]
    srfcvol_fac = srfcvol_fac + csi

    Vc = _applymsk(_add_cycl_col(V, end=False))
    DXUc = _applymsk(_add_cycl_col(DXU, end=False))
    DZUc = _applymsk(_add_cycl_col(DZU, end=False))
    p5 = 0.5
    VnT = p5*(Vc[:,:,:-1]*DXUc[:,:-1]*DZUc[:,:,:-1] + Vc[:,:,1:]*DXUc[:,1:]*DZUc[:,:,1:])/DXT
    # VnT = p5*(Vc[:,:,:-1]*DXUc[:,:-1]*DZUc[:,:,:-1] + Vc[:,:,1:]*DXUc[:,1:]*DZUc[:,:,1:])
    # VnT = p5*(Vc[:,:,:-1]*DZUc[:,:,:-1] + Vc[:,:,1:]*DZUc[:,:,1:])
    VnT[0,:,:] = VnT[0,:,:]*srfcvol_fac # [m3/s].
    VnT = VnT[:,1:,:]                   # [m3/s]. Start at j = 2.
    VnT = VnT[:,:-1,:] # Cap off the last row (dummy row of zeroes).

    d = _skeldict()
    _ = [d.update({sseg:_slicegh(VnT, slcvs[sseg], slchs[sseg])}) for sseg in segnames]

    return d  # [m2/s].
    # return d  # [m3/s].


def _stack_sseg_arrays(full_arr, target_dict, slcvs=laslices, slchs=loslices, segnames=segnames, average=False, how='mean'):
    for sseg in segnames:
        slcvi, slchi = slcvs[sseg], slchs[sseg]
        vnew = _slicegh(full_arr, slcvi, slchi, average=average, how=how)
        if len(target_dict[sseg])==0: # If the dictionary is empty.
            target_dict.update({sseg:vnew[np.newaxis, ...]})
        else:
            target_dict.update({sseg:np.concatenate((target_dict[sseg], vnew), axis=0)})

    return target_dict


def _skeldict(ndims=1, segnames=segnames):
    skeld = dict()
    _ = [skeld.update({k:np.array([], ndmin=ndims)}) for k in segnames]

    return skeld


# Get dzt and dzu.
fname_dzt = 'POP-dzu_dzt_kzit_subsetSO.nc'
ncg = Dataset(fname_dzt)
DZU0 = ncg.variables['dzu'][:]*cm2m
DZT0 = ncg.variables['dzt'][:]*cm2m
dzum =  _stack_sseg_arrays(DZU0, _skeldict())
dztm =  _stack_sseg_arrays(DZT0, _skeldict())
nci = Dataset(fnames[0])
Umask = nci.variables['UVEL'][0,:,:fcap,:].mask
Umask = _stack_sseg_arrays(Umask, _skeldict())
Tmask = nci.variables['TEMP'][0,:,:fcap,:].mask
Tmask = _stack_sseg_arrays(Tmask, _skeldict())
for sseg in segnames:
    dzum0, dztm0 = dzum[sseg], dztm[sseg]
    dzum0[Umask[sseg]] = np.nan
    dztm0[Tmask[sseg]] = np.nan
    dzum0 = np.ma.masked_invalid(dzum0).squeeze()
    dztm0 = np.ma.masked_invalid(dztm0).squeeze()
    dzum.update({sseg:dzum0})
    dztm.update({sseg:dztm0})

dztm_1p, dztm_p1, dztm_1pp1, dztm_1pp1_2 = _skeldict(), _skeldict(), _skeldict(), _skeldict()
dzwm, dzwm2, dzwm_1p, dzwm_p1 = _skeldict(), _skeldict(), _skeldict(), _skeldict()
lontx, latx, Hu, Ht = _skeldict(), _skeldict(), _skeldict(), _skeldict()
lontx0, lattx0 = _skeldict(), _skeldict()

_ = [dztm_1p.update({sseg:dztm[sseg][1:,:]}) for sseg in segnames]
_ = [dztm_p1.update({sseg:dztm[sseg][:-1,:]}) for sseg in segnames]
_ = [dztm_1pp1.update({sseg:dztm[sseg][1:-1]}) for sseg in segnames]
_ = [dztm_1pp1_2.update({sseg:dztm_1pp1[sseg]*2}) for sseg in segnames]
_ = [dzwm.update({sseg:0.5*(dztm_1p[sseg] + dztm_p1[sseg])}) for sseg in segnames]
_ = [dzwm2.update({sseg:dzwm[sseg]*2}) for sseg in segnames]
_ = [dzwm_1p.update({sseg:dzwm[sseg][1:,:]}) for sseg in segnames]
_ = [dzwm_p1.update({sseg:dzwm[sseg][:-1,:]}) for sseg in segnames]
_ = [Hu.update({sseg:dzum[sseg].sum(axis=0)}) for sseg in segnames]
_ = [Ht.update({sseg:dztm[sseg].sum(axis=0)}) for sseg in segnames]
lattxx = nci.variables['TLAT'][:fcap,:]
lontxx = lon360to180(nci.variables['TLONG'][:fcap,:])
lontx = _stack_sseg_arrays(lontxx, _skeldict())
lattx = _stack_sseg_arrays(lattxx, _skeldict())
_ = [lontx0.update({sseg:lontx[sseg].mean()}) for sseg in segnames]
_ = [lattx0.update({sseg:lattx[sseg].mean()}) for sseg in segnames]

DXU = nci.variables['DXU'][:fcap,:]*cm2m
DYU = nci.variables['DYU'][:fcap,:]*cm2m
DXT = nci.variables['DXT'][:fcap,:]*cm2m

px, pmx, px0, pmxm, pmxm0 = _skeldict(), _skeldict(), _skeldict(), _skeldict(), _skeldict()
z = -nci.variables['z_t'][:]*cm2m
zw = -nci.variables['z_w'][:]*cm2m
_ = [px.update({sseg:p_from_z(z, lattx0[sseg])[:,np.newaxis]}) for sseg in segnames]
_ = [pmx.update({sseg:p_from_z(zw, lattx0[sseg].mean())[1:][:,np.newaxis]}) for sseg in segnames] # Cap first W-cell (zw0 = 0 m).
_ = [px0.update({sseg:px[sseg].squeeze()}) for sseg in segnames]
_ = [pmxm.update({sseg:px[sseg][1:-1,:]}) for sseg in segnames]
_ = [pmxm0.update({sseg:pmxm[sseg].squeeze()}) for sseg in segnames]

TAREA = nci.variables['TAREA'][:fcap,:]*cm2m*cm2m
TAREAij = _stack_sseg_arrays(TAREA, _skeldict())
dxm = _stack_sseg_arrays(DXT, _skeldict()) # Width of T-cells to multiply NDTEM heat transport by.

nz = DZT0.shape[0]
Tmx, Smx, Vm = _skeldict(), _skeldict(), _skeldict()
Tmxs, Smxs = _skeldict(), _skeldict()
utx, usx = _skeldict(), _skeldict()
psimean, psimeans = _skeldict(), None
psieddy, psieddys = _skeldict(), None
uqxe_adv, uqxe_advs = _skeldict(), None
uqxe_stir, uqxe_stirs = _skeldict(), None
uqxe, uqxes = _skeldict(), None
uqxm, uqxms = _skeldict(), None
t = []
# fnames = fnames[240:243]
#fnames = fnames[240:252]
# fnames = fnames[-60:]
#fnames = fnames[240:300]

_ = system("rm t_processing.txt")
nt = len(fnames)
n = 0
for fnamen in fnames:
    ti = fnamen.split('/')[-1].split('.')[-2]
    print(ti)
    _ = system('echo "%s" >> t_processing.txt'%ti)
    nci = Dataset(fnamen)
    mo = int(ti[5:7])
    yr = int(ti[:4])

    if CALC_PSIMEAN_ONLY:
        try:
            # Positive SOUTHWARD (onshore).
            V = -nci.variables['VVEL'][0,:,:fcap,:]*cm2m
            ETA = nci.variables['SSH'][0,:fcap,:]*cm2m
        except:
            print("Skipping corrupted file %s"%fnamen)
            continue

        Umx = get_Ux_profile(V, ETA, TAREA, DXT, DXU, DYU, DZU0, DZT0) # [m2/s].
        psimean_aux = _skeldict()
        _ = [psimean_aux.update({sseg:np.mean(-np.cumsum(Umx[sseg], axis=0), axis=-1)}) for sseg in segnames] # [m2/s].

        if psimeans is not None:
            _ = [psimeans.update({sseg:psimeans[sseg]+psimean_aux[sseg]}) for sseg in segnames] # [m2/s].
        else:
            psimeans = psimean_aux.copy()
        t.append(datetime(yr, mo, 15, 12))
        n+=1
        continue

    try:
        # Positive SOUTHWARD (onshore).
        V = -nci.variables['VVEL'][0,:,:fcap,:]*cm2m # Positive SOUTHWARD (onshore).
        _ = [Vm.update({sseg:_slicegh(V, laslices[sseg], loslices[sseg])}) for sseg in segnames]
        T = nci.variables['TEMP'][0,:,:fcap,:]
        S = nci.variables['SALT'][0,:,:fcap,:]
        _ = [Tmx.update({sseg:_slicegh(T, laslices[sseg], loslices[sseg])}) for sseg in segnames]
        _ = [Smx.update({sseg:_slicegh(S, laslices[sseg], loslices[sseg])}) for sseg in segnames]
        utxa = _applymsk(-nci.variables['VNT'][0,:,:fcap,:]) # Positive SOUTHWARD (onshore).
        usxa = _applymsk(-nci.variables['VNS'][0,:,:fcap,:]) # Positive SOUTHWARD (onshore).
        _ = [utx.update({sseg:_slicegh(utxa, laslices[sseg], loslices[sseg])}) for sseg in segnames] # [degC/s]
        _ = [usx.update({sseg:_slicegh(usxa, laslices[sseg], loslices[sseg])}) for sseg in segnames] # [psu/s]
    except:
        print("Skipping corrupted file %s"%fnamen)
        continue

    ETA = nci.variables['SSH'][0,:fcap,:]*cm2m
    etaij = dict()
    _ = [etaij.update({sseg:_slicegh(ETA, laslices[sseg], loslices[sseg])}) for sseg in segnames]
    dztmn = _get_dV(dztm, etaij, TAREAij, dz_only=True)  # [m3].

    _ = [utx.update({sseg:(utx[sseg]*dztmn[sseg])}) for sseg in segnames] # [degC*m/s]
    _ = [usx.update({sseg:(usx[sseg]*dztmn[sseg])}) for sseg in segnames] # [psu*m/s]

    # utxbar and usxbar are now positive SOUTHWARD (onshore).
    utxbar, usxbar, utxbarmTf = get_TSflux(V, T, S, ETA, TAREA, DXU, DYU, DZU0, DZT0, laslices, loslices, Tf=Tf0)

    # Calculate alpha and beta at the faces of the T-cells defining the isobath.
    laslicesp = laslices.copy()
    _ = [laslicesp.update({sseg:slice(laslicesp[sseg].start+1, laslicesp[sseg].stop+1)}) for sseg in segnames]
    TmxTN, SmxTN = dict(), dict() # T and S @ North face of T-cell.
    _ = [TmxTN.update({sseg:0.5*(_slicegh(T, laslices[sseg], loslices[sseg]) + _slicegh(T, laslicesp[sseg], loslices[sseg]))}) for sseg in segnames]
    _ = [SmxTN.update({sseg:0.5*(_slicegh(S, laslices[sseg], loslices[sseg]) + _slicegh(S, laslicesp[sseg], loslices[sseg]))}) for sseg in segnames]
    alphamxTN, betamxTN = dict(), dict()
    alphamxTNw, betamxTNw = dict(), dict()
    _ = [alphamxTN.update({sseg:alpha_eos80(SmxTN[sseg], TmxTN[sseg], px[sseg][:, np.newaxis], pt=True)}) for sseg in segnames]
    _ = [betamxTN.update({sseg:beta_eos80(SmxTN[sseg], TmxTN[sseg], px[sseg][:, np.newaxis], pt=True)}) for sseg in segnames]
    _ = [betamxTNw.update({sseg:(betamxTN[sseg][1:,:]*dztm_1p[sseg] + betamxTN[sseg][:-1,:]*dztm_p1[sseg])/dzwm2[sseg]}) for sseg in segnames]   # @ W-points.
    _ = [alphamxTNw.update({sseg:(alphamxTN[sseg][1:,:]*dztm_1p[sseg] + alphamxTN[sseg][:-1,:]*dztm_p1[sseg])/dzwm2[sseg]}) for sseg in segnames]   # @ W-points.

    # UpTp and UpSp. Both positive SOUTHWARD (onshore).
    UTxe, USxe = dict(), dict()
    UTxew, USxew = dict(), dict()
    _ = [UTxe.update({sseg:utx[sseg]-utxbar[sseg]}) for sseg in segnames] # [degC*m/s].
    _ = [USxe.update({sseg:usx[sseg]-usxbar[sseg]}) for sseg in segnames] # [psu*m/s].
    _ = [UTxew.update({sseg:(UTxe[sseg][1:,:]*dztm_1p[sseg] + UTxe[sseg][:-1,:]*dztm_p1[sseg])/dzwm2[sseg]}) for sseg in segnames]   # @ W-points.
    _ = [USxew.update({sseg:(USxe[sseg][1:,:]*dztm_1p[sseg] + USxe[sseg][:-1,:]*dztm_p1[sseg])/dzwm2[sseg]}) for sseg in segnames]   # @ W-points.

    # Assuming UeT and Tf0 are uncorrelated (Tf0 constant).
    # Need to move numerator to W-points before taking ratio.
    NUMw, DENw = dict(), dict()
    dSmxTNdzw, dTmxTNdzw = dict(), dict()

    _ = [NUMw.update({sseg:betamxTNw[sseg]*USxew[sseg] - alphamxTNw[sseg]*UTxew[sseg]}) for sseg in segnames] # @ W-points.
    _ = [dSmxTNdzw.update({sseg:(SmxTN[sseg][:-1] - SmxTN[sseg][1:])/dzwm[sseg]}) for sseg in segnames]
    _ = [dTmxTNdzw.update({sseg:(TmxTN[sseg][:-1] - TmxTN[sseg][1:])/dzwm[sseg]}) for sseg in segnames]
    _ = [DENw.update({sseg:(betamxTNw[sseg]*dSmxTNdzw[sseg] - alphamxTNw[sseg]*dTmxTNdzw[sseg])}) for sseg in segnames]

    # psieddy = psieddy(z).
    psimean_aux, psieddy_aux, uqxe_aux, uqxm_aux = _skeldict(), _skeldict(), _skeldict(), _skeldict()
    ue, TmxTf0_1pp1 = _skeldict(), _skeldict()
    # peaux = _skeldict()
    uqxe_adv_aux, uqxe_stir_aux = _skeldict(), _skeldict()

    Umx = get_Ux_profile(V, ETA, TAREA, DXT, DXU, DYU, DZU0, DZT0) # [m2/s].
    _ = [Umx.update({sseg:_applymsk(Umx[sseg])}) for sseg in segnames]
    _ = [psimean_aux.update({sseg:-np.cumsum(np.mean(Umx[sseg], axis=-1), axis=0)}) for sseg in segnames] # [m2/s].

    _ = [NUMw.update({sseg:_applymsk(NUMw[sseg])}) for sseg in segnames]
    _ = [DENw.update({sseg:_applymsk(DENw[sseg])}) for sseg in segnames]
    _ = [psieddy_aux.update({sseg:np.median(NUMw[sseg]/DENw[sseg], axis=-1)}) for sseg in segnames] # [m2/s]. At W-points.
    _ = [uqxe_aux.update({sseg:(np.sum(np.sum(utx[sseg] - utxbarmTf[sseg], axis=0)*TAREAij[sseg], axis=-1).squeeze()*T2Q)}) for sseg in segnames]
    _ = [uqxm_aux.update({sseg:(np.sum(np.sum(utxbarmTf[sseg], axis=0)*TAREAij[sseg], axis=-1).squeeze()*T2Q)}) for sseg in segnames]

    _ = [ue.update({sseg:-(psieddy_aux[sseg][:-1,...] - psieddy_aux[sseg][1:,...])}) for sseg in segnames] # [m2/s]. Don't divide by dz because will have to weight the average of u_eddy*(T - Tf0) anyway.

    _ = [TmxTf0_1pp1.update({sseg:TmxTN[sseg][1:-1,...] - Tf0}) for sseg in segnames]
    for sseg in segnames:
        uei = ue[sseg]
        Lyi = Lys[sseg]*1e3 # [m].
        TmxTf0i = np.mean(TmxTf0_1pp1[sseg], axis=-1)
        TmxTf0i[TmxTf0i<0] = 0.
        uqxe_adv_aux[sseg] = np.sum(Lyi*uei*TmxTf0i, axis=0)*T2Q # [(degC*m3/s)*T2Q]
        uqxe_stir_aux[sseg] = uqxe_aux[sseg] - uqxe_adv_aux[sseg]

    if psieddys is not None:
        _ = [Smxs.update({sseg:Smxs[sseg]+Smx[sseg].mean(axis=-1)}) for sseg in segnames] # [psu].
        _ = [Tmxs.update({sseg:Tmxs[sseg]+Tmx[sseg].mean(axis=-1)}) for sseg in segnames] # [degC].
        _ = [psimeans.update({sseg:psimeans[sseg]+psimean_aux[sseg]}) for sseg in segnames] # [m2/s].
        _ = [psieddys.update({sseg:psieddys[sseg]+psieddy_aux[sseg]}) for sseg in segnames] # [m2/s].
        _ = [uqxes.update({sseg:uqxes[sseg]+uqxe_aux[sseg]}) for sseg in segnames] # Eddy heat transport.
        _ = [uqxms.update({sseg:uqxms[sseg]+uqxm_aux[sseg]}) for sseg in segnames]
        _ = [uqxes_adv.update({sseg:uqxes_adv[sseg]+uqxe_adv_aux[sseg]}) for sseg in segnames]
        _ = [uqxes_stir.update({sseg:uqxes_stir[sseg]+uqxe_stir_aux[sseg]}) for sseg in segnames]
    else:
        _ = [Smxs.update({sseg:Smx[sseg].mean(axis=-1)}) for sseg in segnames]
        _ = [Tmxs.update({sseg:Tmx[sseg].mean(axis=-1)}) for sseg in segnames]
        psimeans = psimean_aux.copy()
        psieddys = psieddy_aux.copy()
        uqxes = uqxe_aux.copy()
        uqxms = uqxm_aux.copy()
        uqxes_adv = uqxe_adv_aux.copy()
        uqxes_stir = uqxe_stir_aux.copy()

    t.append(datetime(yr, mo, 15, 12))
    n+=1

if CALC_PSIMEAN_ONLY: # Calculate time-mean and save.
    print("Total number of records being averaged: n = %d."%n)
    _ = [psimeans.update({sseg:psimeans[sseg]/n}) for sseg in segnames] # [m3/s].
    dpsimean = dict(xdists=xdists, hx_yavgs=hx_yavgs, fhshs=fhshs, fhdeeps=fhdeeps, z=z, loslices=loslices, laslices=laslices, psimean=psimeans)
    savez(fnamez_out_psimean, **dpsimean)
    exit()

_ = [Smxs.update({sseg:Smxs[sseg]/n}) for sseg in segnames]
_ = [Tmxs.update({sseg:Tmxs[sseg]/n}) for sseg in segnames]
_ = [psimeans.update({sseg:psimeans[sseg]/n}) for sseg in segnames]
_ = [psieddys.update({sseg:psieddys[sseg]/n}) for sseg in segnames]
_ = [uqxms.update({sseg:uqxms[sseg]/n}) for sseg in segnames]
_ = [uqxes.update({sseg:uqxes[sseg]/n}) for sseg in segnames]
_ = [uqxes_adv.update({sseg:uqxes_adv[sseg]/n}) for sseg in segnames]
_ = [uqxes_stir.update({sseg:uqxes_stir[sseg]/n}) for sseg in segnames]

for sseg in segnames:
    Smxs[sseg][~np.isfinite(Smxs[sseg])] = np.nan
    Tmxs[sseg][~np.isfinite(Tmxs[sseg])] = np.nan
    psimeans[sseg][~np.isfinite(psimeans[sseg])] = np.nan
    psieddys[sseg][~np.isfinite(psieddys[sseg])] = np.nan
    uqxes[sseg][~np.isfinite(uqxes[sseg])] = np.nan
    uqxms[sseg][~np.isfinite(uqxms[sseg])] = np.nan
    uqxes_adv[sseg][~np.isfinite(uqxes_adv[sseg])] = np.nan
    uqxes_stir[sseg][~np.isfinite(uqxes_stir[sseg])] = np.nan
    Smxs[sseg], Tmxs[sseg] = map(_applymsk, (Smxs[sseg], Tmxs[sseg]))
    psimeans[sseg], psieddys[sseg], uqxms[sseg], uqxes[sseg] = map(_applymsk, (psimeans[sseg], psieddys[sseg], uqxms[sseg], uqxes[sseg]))
    uqxes_adv[sseg], uqxes_stir[sseg] = map(_applymsk, (uqxes_adv[sseg], uqxes_stir[sseg]))
    psimeans[sseg], psieddys[sseg], uqxms[sseg], uqxes[sseg] = psimeans[sseg].data, psieddys[sseg].data, uqxms[sseg].data, uqxes[sseg].data
    uqxes_adv[sseg], uqxes_stir[sseg] = uqxes_adv[sseg].data, uqxes_stir[sseg].data
    Smxs[sseg], Tmxs[sseg] = Smxs[sseg].data, Tmxs[sseg].data

t = np.array(t)
dvars = dict(psimean=psimeans, psieddy=psieddys, uqxm=uqxms, uqxe=uqxes, uqxe_adv=uqxes_adv, uqxe_stir=uqxes_stir, Sp=Smxs, Theta=Tmxs, t=t, zm_psimean=z, zm_psieddy=zw, z=z, p=px, pw=pmx, dzum=dzum, dztm=dztm, lats=Lattxs, lons=Lontxs, dists=xdists, segbrys_psi=segbrys_psi, segnames=segnames, hshallow=hshallow, hdeep=hdeep, xdists=xdists, hx_yavgs=hx_yavgs, fhshs=fhshs, fhdeeps=fhdeeps, loslices=loslices, laslices=laslices, Lys=Lys)

# psimean is at z_t coordinates (full).
# psieddy is at z_w[1:] coordinates.
savez(fnamez_out_psi, **dvars)
