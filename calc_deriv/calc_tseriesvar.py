# Description: Calculate yearly averages from monthly files.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com
# Date:        January/2018

import numpy as np
import matplotlib
from glob import glob
from os import system
from datetime import datetime
from netCDF4 import Dataset, num2date
from pandas import Timestamp
from gsw import SA_from_SP, CT_from_pt
from gsw import alpha as falpha
from gsw import beta as fbeta
import xarray as xr
from ap_tools.utils import lon360to180, rot_vec
from reproducibility import savez

def deg2m_dist(lon, lat):
        """
        USAGE
        -----
        dx, dy = deg2m_dist(lon, lat)
        """
        lon, lat = map(np.array, (lon, lat))

        dlat, _ = np.gradient(lat)             # [deg]
        _, dlon = np.gradient(lon)             # [deg]
        deg2m = 111120.0                       # [m/deg]
        # Account for divergence of meridians in zonal distance.
        dx = dlon*deg2m*np.cos(lat*np.pi/180.) # [m]
        dy = dlat*deg2m                        # [m]

        return dx, dy

def ang_isob(xiso, yiso):
    xiso, yiso = map(np.array, (xiso, yiso))
    R = 6371000.0 # Mean radius of the earth in meters (6371 km), from gsw.constants.earth_radius.
    deg2rad = np.pi/180. # [rad/deg]

    # From the coordinates of the isobath, find the angle it forms with the
    # zonal axis, using points k+1 and k.
    shth = yiso.size-1
    theta = np.zeros(shth)
    for k in range(shth):
        dyk = R*(yiso[k+1] - yiso[k])
        dxk = R*(xiso[k+1] - xiso[k])*np.cos(yiso[k]*deg2rad)
        theta[k] = np.arctan2(dyk, dxk)

    xisom = 0.5*(xiso[1:] + xiso[:-1])
    yisom = 0.5*(yiso[1:] + yiso[:-1])

    return xisom, yisom, theta/deg2rad

def near(x, x0, npts=1, return_index=False):
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

def stripmsk(arr, mask_invalid=True):
    if mask_invalid:
        arr = np.ma.masked_invalid(arr)
    if np.ma.isMA(arr):
        msk = arr.mask
        arr = arr.data
        arr[msk] = np.nan

    return arr

##---
CALC_MULTIYEARLY_TSDUVKE = False
NYR_avg = 10 # Average T, S, u, v every 10 years.
#
CALC_UxVaISOB = False
CALC_U_zavg = False
zslabavg_top, zslabavg_bot = 0, 150
CALC_SSH = False
CALC_PT = False
#
# Also plot seasonal cycle for these.
#
CALC_KE = False
CALC_GRADRHO = False
CALC_Jb = True
CALC_Jb_shelf_integral_timeseries = False
CALC_Tauxy = False
#
CALC_PT_zavg = False
CALC_AICE = False
z_PT = 1000 # [m].
CALC_CLIM_DUVKE = False

# Start and end years.
START_YEAR = 1959
END_YEAR = 2009
fname_out_aice = 'aice.npz'
fname_out_eke = 'EKE_MKE.npz'
fname_out_drhomag = 'gradRHO.npz'
fname_out_Jb = 'Jb.npz'
fname_out_Jb_shelf_integral_timeseries = 'Jb_int.npz'
fname_out_Tauxy = 'tauxy.npz'
fname_out_ssh = 'yearly_SSH.npz'
fname_out_u = 'yearly_U.npz'
fname_out_uvxisob = 'yearly_UVxisob.npz'
fname_out_PT = 'yearly_PT.npz'
fname_out_tsduvke = 'decadal_TSD-UV-KE.npz'
fname_out_duvke_clim = 'clim_%d-%d_D-UV-KE.npz'%(START_YEAR, END_YEAR)
fname_dzu = 'POP-dzu_dzt_kzit_subsetSO.nc'

cm2m = 1e-2
fcap = 501
thresh = 1e10
fdir_tail = '/ocn/hist/ia_top_tx0.1_v2_yel_patc_1948_intel.pop.h.????-??.nc'
head_fin = '/lustre/atlas1/cli115/proj-shared/ia_top_tx0.1_v2_60yrs/'
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

nc = Dataset(fnames[0])
lont = nc.variables['TLONG'][:fcap,:]
latt = nc.variables['TLAT'][:fcap,:]
lonu = nc.variables['ULONG'][:fcap,:]
latu = nc.variables['ULAT'][:fcap,:]
kmt = nc.variables['KMT'][:fcap,:] - 1 # Convert fortran to python index.
ny, nx = kmt.shape
z = nc.variables['z_t'][:]*cm2m # [m].
t = []
tmo = []

fname_isobs = 'isobaths.nc'
ncx = Dataset(fname_isobs)
dmsm = ncx["1000 m isobath"]['diso'][:]
xmsm = ncx["1000 m isobath"]['xiso'][:]
ymsm = ncx["1000 m isobath"]['yiso'][:]

xm = ncx["1000 m isobath (U-points)"]['xiso'][:]
ym = ncx["1000 m isobath (U-points)"]['yiso'][:]
dm = ncx["1000 m isobath (U-points)"]['diso'][:]
Im = ncx["1000 m isobath (U-points)"]['i'][:]
Jm = ncx["1000 m isobath (U-points)"]['j'][:]
uxmsk = ncx['1000 m isobath (x-isobath U, V masks)']['Umsk'][:]
vxmsk = ncx['1000 m isobath (x-isobath U, V masks)']['Vmsk'][:]

dmm = 0.5*(dm[1:] + dm[:-1])
xmm, ymm, angm = ang_isob(xm, ym) # Angle of the U-points isobath.

##----
if CALC_AICE:
    iceconc_thresh = 0.15 # Ice concentration threshold.
    fnames = [fnamen.replace('ocn','ice') for fnamen in fnames]
    fnames = [fnamen.replace('.pop.h.','.cice.h.') for fnamen in fnames]
    AICE = np.array([])
    nfirst = True
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        nci = Dataset(fnamen)
        if nfirst:
            tarea = nci['tarea'][:].data*1e-6 # [km2]
            lon = lon360to180(nci['TLON'][:].data)
            lat = nci['TLAT'][:].data
            tmask = nci['tmask'][:]
            nfirst = False
        Aice = nci.variables['aice'][0,:fcap,:]/100.   # Convert to fractional sea ice concentration (0-1).
        # Calculate total ice area for valid ice cells.
        # iarea=aice(aice>=dc & aice<=1.0 & aice~=0).*tarea(aice>=dc & aice<=1.0 & aice~=0).*1e-6;
        fice=np.logical_and(Aice>=iceconc_thresh, Aice<=1.0)
        aice = np.sum(Aice[fice]*tarea[fice])
        t.append(yeari)
        AICE = np.append(AICE, aice)

    t = np.array([Timestamp(str(ti)+'-15').to_pydatetime() for ti in t])
    savez(fname_out_aice, icearea=AICE, lon=lont, lat=latt, tarea=tarea, t=t)

##---

if CALC_UxVaISOB:
    dzui = Dataset(fname_dzu).variables['dzu'][:]
    dzui = dzui[:,Im,Jm]*cm2m

    Uxyr, Ux, ux = None, None, None
    Vayr, Va, va = None, None, None
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        nci = Dataset(fnamen)
        # Zonal/meridional vel. components.
        uu = nci.variables['UVEL'][0,:,:fcap,:]
        vv = nci.variables['VVEL'][0,:,:fcap,:]
        ui = uu[:,Im,Jm]*cm2m
        vi = vv[:,Im,Jm]*cm2m
        if fnamen==fnames[0]:
            hmsk = ~ui.mask
            hi = np.array([dzui[hmsk[:,n],n].sum(axis=0) for n in range(Im.size)])

        Ui = np.sum(ui*dzui, axis=0) # [m2/s], zonal transport per unit along-isobath length.
        Vi = np.sum(vi*dzui, axis=0) # [m2/s], meridional transport per unit along-isobath length.
        uui = Ui/hi # [m/s], depth-averaged zonal vel.
        vvi = Vi/hi # [m/s], depth-averaged meridional vel.
        uui = 0.5*(uui[1:] + uui[:-1])
        vvi = 0.5*(vvi[1:] + vvi[:-1])
        Ui = 0.5*(Ui[1:] + Ui[:-1])
        Vi = 0.5*(Vi[1:] + Vi[:-1])

        # Rotate depth-averaged velocities using angles based on realistic isobaths.
        va, ux = rot_vec(uui, vvi, angle=angm, degrees=True) # ATTENTION: v_along, u_across = rot(u_east, v_north)***
        ux = -ux # Positive ONSHORE.
        Vva, Uux = rot_vec(Ui, Vi, angle=angm, degrees=True)
        Uux = -Uux

        ux = ux[np.newaxis,...]
        va = va[np.newaxis,...]
        Uux = Uux[np.newaxis,...]
        Vva = Vva[np.newaxis,...]

        if Ux is not None:
            Ux = np.vstack((Ux, ux))
            Va = np.vstack((Va, va))
            UUx = np.vstack((UUx, Uux))
            VVa = np.vstack((VVa, Vva))
        else:
            Ux = ux
            Va = va
            UUx = Uux
            VVa = Vva
        nmo+=1
        tmo.append(yeari)
        if nmo==12:
            Ux = Ux.mean(axis=0)[np.newaxis,...]
            Va = Va.mean(axis=0)[np.newaxis,...]
            UUx = UUx.mean(axis=0)[np.newaxis,...]
            VVa = VVa.mean(axis=0)[np.newaxis,...]
            if Uxyr is not None:
                Uxyr = np.vstack((Uxyr, Ux))
                Vayr = np.vstack((Vayr, Va))
                UUxyr = np.vstack((UUxyr, UUx))
                VVayr = np.vstack((VVayr, VVa))
            else:
                Uxyr = Ux.copy()
                Vayr = Va.copy()
                UUxyr = UUx.copy()
                VVayr = VVa.copy()
            t.append(yeari2)
            Ux, UUx = None, None
            Va, VVa = None, None
            nmo=0

    t = np.array([Timestamp(str(ti)+'-06-15').to_pydatetime() for ti in t])
    tmo = np.array([Timestamp(str(ti)+'-15').to_pydatetime() for ti in tmo])
    Uxyr, Vayr = Uxyr.data, Vayr.data
    Uxyr[Uxyr>thresh] = np.nan
    Vayr[Vayr>thresh] = np.nan
    UUxyr, VVayr = UUxyr.data, VVayr.data
    UUxyr[UUxyr>thresh] = np.nan
    VVayr[VVayr>thresh] = np.nan
    # Uxyr, Vayr = Uxyr*cm2m, Vayr*cm2m # [m/s].
    savez(fname_out_uvxisob, ux=Uxyr, va=Vayr, Ux=UUxyr, Va=VVayr, lonu=xm, latu=ym, dm=dmm, xm=xmm, ym=ymm, angm=angm, Im=Im, Jm=Jm, t=t, tmo=tmo, z=z, d=dm, x=xm, y=ym)

##----

if CALC_U_zavg:
    fzu = np.logical_and(z>=zslabavg_top, z<=zslabavg_bot)
    dzu0 = Dataset(fname_dzu).variables['dzu'][fzu,...]*cm2m # [m].
    h0 = dzu0.sum(axis=0) # [m].
    Uyr, U, u = None, None, None
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        nci = Dataset(fnamen)
        u = nci.variables['UVEL'][0,fzu,:fcap,:]
        u = np.sum(u*dzu0, axis=0)/h0
        u = u[np.newaxis,...]*cm2m # [m/s].
        if U is not None:
            U = np.vstack((U, u))
        else:
            U = u
        nmo+=1
        tmo.append(yeari)
        if nmo==12:
            if Umo is not None:
                Umo = np.vstack((Umo, U[:, Im, Jm]))
            else:
                Umo = U.copy()
            U = U.mean(axis=0)[np.newaxis,...]
            if Uyr is not None:
                Uyr = np.vstack((Uyr, U))
            else:
                Uyr = U.copy()
            t.append(int(yeari2))
            U = None
            nmo=0

    t = np.array([Timestamp(str(ti)+'-06-15').to_pydatetime() for ti in t])
    tmo = np.array([Timestamp(str(ti)+'-15').to_pydatetime() for ti in tmo])
    Uyr = Uyr.data
    Uyr[Uyr>thresh] = np.nan
    Uyr[Uyr==0.] = np.nan
    savez(fname_out_u, umonthly=Umo, u=Uyr, lon=lonu, lat=latu, t=t, tmo=tmo, z=z, d=dm, x=xm, y=ym, ztop=zslabavg_top, zbot=zslabavg_bot)

##---
if CALC_Tauxy: # Yearly wind stress.
    Tauxyr, Tauxmo, Taux, taux = None, None, None, None
    Tauyyr, Tauymo, Tauy, tauy = None, None, None, None
    skel = np.zeros((ny, nx))
    Tauxclm, Tauyclm = dict(), dict()
    _ = [Tauxclm.update({mo:skel}) for mo in range(1, 13)]
    _ = [Tauyclm.update({mo:skel}) for mo in range(1, 13)]
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        mo = int(yeari[-2:])
        _ = system('echo "%s" > t_processing.txt'%yeari)
        nci = Dataset(fnamen)

        taux = nci.variables['TAUX'][0,:fcap,:]
        tauy = nci.variables['TAUY'][0,:fcap,:]

        taux, tauy = taux[np.newaxis,...], tauy[np.newaxis,...]
        if Taux is not None:
            Taux = np.vstack((Taux, taux))
            Tauy = np.vstack((Tauy, tauy))
        else:
            Taux = taux.copy()
            Tauy = tauy.copy()

        nmo+=1
        tmo.append(yeari)

        # Update monthly climatological fields.
        Tauxclm.update({nmo:Tauxclm[nmo] + taux})
        Tauyclm.update({nmo:Tauyclm[nmo] + tauy})

        if nmo==12:
            if Tauxmo is not None:
                Tauxmo = np.vstack((Tauxmo, Taux[:, Im, Jm]))
                Tauymo = np.vstack((Tauymo, Tauy[:, Im, Jm]))
            else:
                Tauxmo, Tauymo = Taux[:, Im,Jm], Tauy[:, Im,Jm]

            Taux = Taux.mean(axis=0)[np.newaxis,...]
            Tauy = Tauy.mean(axis=0)[np.newaxis,...]
            if Tauxyr is not None:
                Tauxyr = np.vstack((Tauxyr, Taux))
                Tauyyr = np.vstack((Tauyyr, Tauy))
            else:
                Tauxyr, Tauyyr = Taux.copy(), Tauy.copy()
            t.append(int(yeari2))
            Taux = None
            Tauy = None
            nmo=0

    Tauxmom = 0.5*(Tauxmo[:, 1:] + Tauxmo[:, :-1])
    Tauymom = 0.5*(Tauymo[:, 1:] + Tauymo[:, :-1])
    Tauamo, _ = rot_vec(Tauxmom, Tauymom, angle=angm, degrees=True) # positive CLOCKWISE***

    dynecm2toNm2 = 1e-1 # 1e-5*1e4
    t = np.array([Timestamp(str(ti)+'-06-15').to_pydatetime() for ti in t])
    tmo = np.array([Timestamp(str(ti)+'-15').to_pydatetime() for ti in tmo])
    #
    # Along-isobath wind stress, positive CLOCKWISE around the isobath.
    Tauamo = Tauamo.data
    Tauamo[Tauamo>thresh] = np.nan
    Tauamo = Tauamo*dynecm2toNm2 # [N/m2].
    #
    #--- Climatological monthly fields.
    nt = len(fnames)/12
    for mo in range(1, 13):
        auxx = Tauxclm[mo].squeeze()*dynecm2toNm2/nt
        auxy = Tauyclm[mo].squeeze()*dynecm2toNm2/nt
        Tauxclm.update({mo:auxx})
        Tauyclm.update({mo:auxy})
    #
    Tauxyr, Tauyyr = Tauxyr.data, Tauyyr.data
    Tauxmo, Tauymo = Tauxmo.data, Tauymo.data
    Tauxyr[Tauxyr>thresh] = np.nan
    Tauyyr[Tauyyr>thresh] = np.nan
    Tauxyr = Tauxyr*dynecm2toNm2 # [N/m2].
    Tauyyr = Tauyyr*dynecm2toNm2 # [N/m2].
    Tauxmo[Tauxmo>thresh] = np.nan
    Tauymo[Tauymo>thresh] = np.nan
    Tauxmo = Tauxmo*dynecm2toNm2 # [N/m2].
    Tauymo = Tauymo*dynecm2toNm2 # [N/m2].
    savez(fname_out_Tauxy, tauxclm=Tauxclm, tauyclm=Tauyclm, tau_alongmo=Tauamo, tauxmo=Tauxmo, tauymo=Tauymo, taux=Tauxyr, tauy=Tauyyr, lon=lonu, lat=latu, dm=dmm, xm=xmm, ym=ymm, angm=angm, t=t, tmo=tmo, z=z, d=dm, x=xm, y=ym)



if CALC_Jb_shelf_integral_timeseries: # Monthly surface buoyancy flux integrated over the shelf.
    JbINT = np.array([])
    JqINT = np.array([])
    JsINT = np.array([])

    # Load in 1000 m mask.
    finvol = np.bool8(np.load('volmsk1000m.npz')['volmsk'])
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        mo = int(yeari[-2:])
        _ = system('echo "%s" > t_processing.txt'%yeari)
        nci = Dataset(fnamen)
        shf = nci.variables['SHF'][0,:fcap,:]          # [W/m2].
        if fnamen==fnames[0]:
            rho0 = nci.variables['rho_sw'][0]*1e3      # [kg/m3].
            rho_fw = nci.variables['rho_fw'][0]*1e3    # [kg/m3].
            g = nci.variables['grav'][0]*1e-2          # [m/s2].
            Cp = nci.variables['cp_sw'][0]*1e3*1e-7    # [J/kg/degC].
            rhoCp = rho0*Cp
            #
            wetmsk = np.float32(~shf.mask) # Ones in valid (non-continent) cells.
            tarea = nci.variables['TAREA'][:fcap,:]*wetmsk*cm2m*cm2m # [m2].
            tareain = tarea[finvol] # [m2], zeros on the continent.
            Tareain = tareain.sum() # [m2].

            JB = shf*0
            JQ = JB.copy()
            JS = JB.copy()

        sfwf = nci.variables['SFWF'][0,:fcap,:]/rho_fw # [(kg of freshwater)/m2/s] / [(kg of freshwater)/m3] = [m/s] = [m3/s/m2]. Volume flux density.
                                                       # positive SFWF = Ocean gains freshwater, so this is (P - E).
        SSSp = nci.variables['SALT'][0,0,:fcap,:]      # [g/kg].
        SST = nci.variables['TEMP'][0,0,:fcap,:]       # [degC].
        SSSA = SA_from_SP(SSSp, 0, lont, latt)         # [g/kg].
        SSCT = CT_from_pt(SSSA, SST)                   # [degC].
        alpha = falpha(SSSA, SSCT, 0)
        beta = fbeta(SSSA, SSCT, 0)
        coeffQ =  g*alpha/rhoCp
        coeffFW = g*beta*SSSA
        qb = coeffQ*shf
        sb = coeffFW*sfwf        # Positive SFWF, ocean gains freshwater, hence buoyancy.
        jb = qb + sb # Surface buoyancy flux [W/kg]. Hosegood et al. (2013).

        # Accumulate time-averaged 2D fields [W/kg].
        JB += jb
        JQ += qb
        JS += sb

        # Integrate over the 1000 m-bounded control surface.
        Jbint = np.sum(jb[finvol]*tareain)/Tareain
        Jqint = np.sum(qb[finvol]*tareain)/Tareain
        Jsint = np.sum(sb[finvol]*tareain)/Tareain

        JbINT = np.append(JbINT, Jbint)
        JqINT = np.append(JqINT, Jqint)
        JsINT = np.append(JsINT, Jsint)

        nmo+=1
        tmo.append(yeari)
        if nmo==12:
            nmo=0

    nt = len(tmo)
    JB /= nt
    JQ /= nt
    JS /= nt

    tmo = np.array([Timestamp(str(ti)+'-15').to_pydatetime() for ti in tmo])
    savez(fname_out_Jb_shelf_integral_timeseries, Jb=JbINT, Jq=JqINT, Js=JsINT, t=tmo, Jbxy=JB, Jqxy=JQ, Jsxy=JS, lon=lont, lat=latt)

##---

if CALC_Jb: # Yearly surface buoyancy flux.
    thresh = 1e3
    Jbyr, Jbmo, Jb, jb = None, None, None, None

    finvol = np.bool8(np.load('volmsk1000m.npz')['volmsk'])
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        mo = int(yeari[-2:])
        _ = system('echo "%s" > t_processing.txt'%yeari)
        nci = Dataset(fnamen)
        shf = nci.variables['SHF'][0,:fcap,:]          # [W/m2].
        if fnamen==fnames[0]:
            rho0 = nci.variables['rho_sw'][0]*1e3      # [kg/m3].
            rho_fw = nci.variables['rho_fw'][0]*1e3    # [kg/m3].
            g = nci.variables['grav'][0]*1e-2          # [m/s2].
            Cp = nci.variables['cp_sw'][0]*1e3*1e-7    # [J/kg/degC].
            rhoCp = rho0*Cp
            #
            wetmsk = np.float32(~shf.mask) # Ones in valid (non-continent) cells.
            tarea = nci.variables['TAREA'][:fcap,:]*wetmsk*cm2m*cm2m # [m2].
            tareain = tarea[finvol] # [m2], zeros on the continent.
            Tareain = tareain.sum() # [m2].

        sfwf = nci.variables['SFWF'][0,:fcap,:]/rho_fw # [(kg of freshwater)/m2/s] / [(kg of freshwater)/m3] = [m/s] = [m3/s/m2]. Volume flux density.
                                                       # positive SFWF = Ocean gains freshwater, so this is (P - E).
        SSSp = nci.variables['SALT'][0,0,:fcap,:]      # [g/kg].
        SST = nci.variables['TEMP'][0,0,:fcap,:]       # [degC].
        SSSA = SA_from_SP(SSSp, 0, lont, latt)         # [g/kg].
        SSCT = CT_from_pt(SSSA, SST)                   # [degC].
        alpha = falpha(SSSA, SSCT, 0)
        beta = fbeta(SSSA, SSCT, 0)
        coeffQ =  g*alpha/rhoCp
        coeffFW = g*beta*SSSA
        qb = coeffQ*shf
        sb = coeffFW*sfwf        # Positive SFWF, ocean gains freshwater, hence buoyancy.
        jb = qb + sb # Surface buoyancy flux [W/kg]. Hosegood et al. (2013).

        # Integrate over the 1000 m-bounded control surface.
        Jbint = np.sum(jb[finvol]*tareain)/Tareain

        jb = jb[np.newaxis,...]
        if Jb is not None:
            Jb = np.vstack((Jb, jb))
        else:
            Jb = jb.copy()
        nmo+=1
        tmo.append(yeari)
        if nmo==12:
            if Jbmo is not None:
                Jbmo = np.vstack((Jbmo, Jb[:, Im, Jm]))
            else:
                Jbmo = Jb[:, Im, Jm]

            Jb = Jb.mean(axis=0)[np.newaxis,...]
            if Jbyr is not None:
                Jbyr = np.vstack((Jbyr, Jb))
            else:
                Jbyr = Jb.copy()
            t.append(int(yeari2))
            Jb = None
            nmo=0

    t = np.array([Timestamp(str(ti)+'-06-15').to_pydatetime() for ti in t])
    tmo = np.array([Timestamp(str(ti)+'-15').to_pydatetime() for ti in tmo])
    Jbyr = Jbyr.data
    # Jbyr[np.abs(Jbyr)>thresh] = np.nan
    Jbmo = Jbmo.data
    # Jbmo[np.abs(Jbmo)>thresh] = np.nan
    savez(fname_out_Jb, Jb=Jbyr, Jbmonthly=Jbmo, lon=lont, lat=latt, t=t, tmo=tmo, z=z, d=dm, x=xm, y=ym)

##---
if CALC_GRADRHO:
    ncdzu = Dataset(fname_dzu)
    dzt = ncdzu.variables['dzt'][:]*cm2m # [m].
    dx, dy = deg2m_dist(lont, latt)      # [m].
    dx, dy = dx*1e-3, dy*1e-3            # [km].
    DRHOMAGyr, DRHOMAGmo, DRHOMAG, drhomag = None, None, None, None
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        mo = int(yeari[-2:])
        _ = system('echo "%s" > t_processing.txt'%yeari)
        nci = Dataset(fnamen)
        rho = nci.variables['RHO'][0,:,:fcap,:]*1e3 # [kg/m3].
        _, drhody, drhodx = np.gradient(rho)
        drhody, drhodx = drhody/dy, drhodx/dx         # [kg/m3/km].
        drhomag = np.sqrt(drhodx*drhodx + drhody*drhody) # [kg/m3/km].
        if fnamen==fnames[0]:
            solidmsk = np.float32(~rho.mask)
            tarea = nci.variables['TAREA'][:fcap,:]*cm2m*cm2m # [m2].
            dVt = tarea[np.newaxis,...]*dzt
            Vh = np.sum(dVt*solidmsk, axis=0) # [m3].
        # plt.figure(); plt.imshow(np.log10(np.flipud(drhomag[0,...])), vmin=-7.5, vmax=-5.5); plt.colorbar(orientation='horizontal')
        drhomag = np.sum(drhomag*dVt, axis=0)/Vh
        drhomag = drhomag[np.newaxis,...]

        if DRHOMAG is not None:
            DRHOMAG = np.vstack((DRHOMAG, drhomag))
        else:
            DRHOMAG = drhomag.copy()
        nmo+=1
        tmo.append(yeari)
        if nmo==12:
            if DRHOMAGmo is not None:
                DRHOMAGmo = np.vstack((DRHOMAGmo, DRHOMAG[:, Im, Jm]))
            else:
                DRHOMAGmo = DRHOMAG[:, Im, Jm]

            DRHOMAG = DRHOMAG.mean(axis=0)[np.newaxis,...]
            if DRHOMAGyr is not None:
                DRHOMAGyr = np.vstack((DRHOMAGyr, DRHOMAG))
            else:
                DRHOMAGyr = DRHOMAG.copy()
            t.append(int(yeari2))
            DRHOMAG = None
            nmo=0

    nmotot = 12*len(t) # Total number of each month.
    t = np.array([Timestamp(str(ti)+'-06-15').to_pydatetime() for ti in t])
    tmo = np.array([Timestamp(str(ti)+'-15').to_pydatetime() for ti in tmo])
    DRHOMAGyr = DRHOMAGyr.data
    DRHOMAGyr[DRHOMAGyr>thresh] = np.nan
    DRHOMAGmo = DRHOMAGmo.data
    DRHOMAGmo[DRHOMAGmo>thresh] = np.nan
    savez(fname_out_drhomag, gradrho_mag_monthly=DRHOMAGmo, gradrho_mag=DRHOMAGyr, lon=lont, lat=latt, t=t, tmo=tmo, z=z, d=dm, x=xm, y=ym)

##---

if CALC_KE:
    ncdzu = Dataset(fname_dzu)
    dzu = ncdzu.variables['dzu'][:]*cm2m # [m].
    EKEyr, EKEmo, EKE, eke = None, None, None, None
    KEyr, KEmo, KE, ke = None, None, None, None
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        mo = int(yeari[-2:])
        _ = system('echo "%s" > t_processing.txt'%yeari)
        nci = Dataset(fnamen)
        ke = nci.variables['KE'][0,:,:fcap,:]*cm2m*cm2m # [m2/s2].****
        if fnamen==fnames[0]:
            solidmsk = np.float32(~ke.mask)
            uarea = nci.variables['UAREA'][:fcap,:]*cm2m*cm2m # [m2].
            dVu = uarea[np.newaxis,...]*dzu
            Vh = np.sum(dVu*solidmsk, axis=0) # [m3].
        uu = nci.variables['UVEL'][0,:,:fcap,:]*cm2m # [m/s]
        vv = nci.variables['VVEL'][0,:,:fcap,:]*cm2m # [m/s]
        mke = 0.5*(uu*uu + vv*vv) # [m2/s2]
        eke = ke - mke # Monthly horizontal EKE [cm2/s2]
        # Depth-average.
        # plt.figure(); plt.imshow(np.log10(np.flipud(eke*cm2m**2)), vmin=-5, vmax=-1); plt.colorbar(orientation='horizontal')
        mke = np.sum(mke*dVu, axis=0)/Vh # [m2/s2].
        eke = np.sum(eke*dVu, axis=0)/Vh # [m2/s2].
        mke = mke[np.newaxis,...]
        eke = eke[np.newaxis,...]

        if EKE is not None:
            MKE = np.vstack((MKE, mke))
            EKE = np.vstack((EKE, eke))
        else:
            MKE = mke.copy()
            EKE = eke.copy()

        nmo+=1
        tmo.append(yeari)
        if nmo==12:
            if EKEmo is not None:
                MKEmo = np.vstack((MKEmo, MKE[:, Im, Jm]))
                EKEmo = np.vstack((EKEmo, EKE[:, Im, Jm]))
            else:
                EKEmo = EKE[:, Im, Jm]
                MKEmo = MKE[:, Im, Jm]

            MKE = MKE.mean(axis=0)[np.newaxis,...]
            EKE = EKE.mean(axis=0)[np.newaxis,...]
            if EKEyr is not None:
                MKEyr = np.vstack((MKEyr, MKE))
                EKEyr = np.vstack((EKEyr, EKE))
            else:
                MKEyr = MKE.copy()
                EKEyr = EKE.copy()
            t.append(int(yeari2))
            MKE = None
            EKE = None
            nmo=0

    t = np.array([Timestamp(str(ti)+'-06-15').to_pydatetime() for ti in t])
    tmo = np.array([Timestamp(str(ti)+'-15').to_pydatetime() for ti in tmo])
    MKEyr = MKEyr.data
    MKEyr[MKEyr>thresh] = np.nan
    EKEyr = EKEyr.data
    EKEyr[EKEyr>thresh] = np.nan
    MKEmo = MKEmo.data
    MKEmo[MKEmo>thresh] = np.nan
    EKEmo = EKEmo.data
    EKEmo[EKEmo>thresh] = np.nan
    savez(fname_out_eke, mke=MKEyr, eke=EKEyr, mkemonthly=MKEmo, ekemonthly=EKEmo, lon=lonu, lat=latu, t=t, tmo=tmo, z=z, d=dm, x=xm, y=ym)

##---

if CALC_SSH:
    SSHyr, SSH, ssh = None, None, None
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        nci = Dataset(fnamen)
        ssh = nci.variables['SSH'][:,:fcap,:]
        ssh = np.ma.masked_greater(ssh, thresh)
        if SSH is not None:
            SSH = np.vstack((SSH, ssh))
        else:
            SSH = ssh
        nmo+=1
        if nmo==12:
            SSH = SSH.mean(axis=0)[np.newaxis,...]
            if SSHyr is not None:
                SSHyr = np.vstack((SSHyr, SSH))
            else:
                SSHyr = SSH.copy()
            t.append(int(yeari2))
            SSH = None
            nmo=0

    t = np.array(t)
    SSHyr = SSHyr.data
    SSHyr[SSHyr>thresh] = np.nan
    SSHyr = SSHyr*cm2m # [m].
    savez(fname_out_ssh, ssh=SSHyr, lon=lont, lat=latt, t=t, z=z)

##---

if CALC_PT_zavg: # FIXME: Need to finish this option.
    fzt = np.logical_and(z>=zslabavg_top+pt, z<=zslabavg_bot_pt)
    dzt0 = Dataset(fname_dzu).variables['dzt'][fzt,...]*cm2m # [m].
    h0 = dzt0.sum(axis=0) # [m].
    jmax, imax = kmt.shape
    landmsk=kmt==-1
    idxlev = np.ones((jmax, imax))*idxlev0
    slabmsk[:,landmsk] = False
    for j in range(jmax):
        for i in range(imax):
            kmtji = kmt[j, i]
            if idxlev0>kmtji:
                idxlev[kmtji:, j, i] = False
    idxlev = np.int32(idxlev)

    PTyr, PT, pt = None, None, None
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        nci = Dataset(fnamen)
        pt = nci.variables['TEMP'][0,fzu,:fcap,:]
        pti = np.ones((jmax, imax))*np.nan
        for j in range(jmax):
            for i in range(imax):
                pti[j,i] = pt[idxlev[j,i],j,i]
        pti[landmsk] = np.nan
        pti = np.ma.masked_greater(pti, thresh)
        pt = np.ma.masked_invalid(pti)
        pt = np.sum(pt*dzt0, axis=0)/h0
        pt = pt[np.newaxis,...]
        if PT is not None:
            PT = np.vstack((PT, pt))
        else:
            PT = pt
        nmo+=1
        if nmo==12:
            PT = PT.mean(axis=0)[np.newaxis,...]
            if PTyr is not None:
                PTyr = np.vstack((PTyr, PT))
            else:
                PTyr = PT.copy()
            t.append(int(yeari2))
            PT = None
            nmo=0

    t = np.array(t)
    PTyr = PTyr.data
    PTyr[PTyr>thresh] = np.nan
    savez(fname_out_PT, pt=PTyr, lon=lont, lat=latt, t=t, z=z, zlev=z_PT)

##---

if CALC_PT:
    jmax, imax = kmt.shape
    idxlev0 = near(z, z_PT, return_index=True)
    idxlev = np.ones((jmax, imax))*idxlev0
    # Want temperature at desired level or bottom, whichever is shallower
    for j in range(jmax):
        for i in range(imax):
            kmtji = kmt[j, i]
            if idxlev0>kmtji:
                idxlev[j, i] = kmtji
    idxlev = np.int32(idxlev)
    landmsk=idxlev==-1

    PTyr, PT, pt = None, None, None
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        nci = Dataset(fnamen)
        pt = nci.variables['TEMP'][0,:,:fcap,:]
        pti = np.ones((jmax, imax))*np.nan
        for j in range(jmax):
            for i in range(imax):
                pti[j,i] = pt[idxlev[j,i],j,i]
        pti[landmsk] = np.nan
        pt = np.ma.masked_greater(pti, thresh)
        pt = pt[np.newaxis,...]
        if PT is not None:
            PT = np.vstack((PT, pt))
        else:
            PT = pt
        nmo+=1
        if nmo==12:
            PT = PT.mean(axis=0)[np.newaxis,...]
            if PTyr is not None:
                PTyr = np.vstack((PTyr, PT))
            else:
                PTyr = PT.copy()
            t.append(int(yeari2))
            PT = None
            nmo=0

    t = np.array(t)
    PTyr = PTyr.data
    PTyr[PTyr>thresh] = np.nan
    savez(fname_out_PT, pt=PTyr, lon=lont, lat=latt, t=t, z=z, zlev=z_PT)

##--
if CALC_CLIM_DUVKE:
    skel = np.zeros_like(nc.variables['TEMP'][0,:,:fcap,:])
    Ujfm, Uamj, Ujas, Uond = skel.copy(), skel.copy(), skel.copy(), skel.copy()
    Vjfm, Vamj, Vjas, Vond = skel.copy(), skel.copy(), skel.copy(), skel.copy()
    KEjfm, KEamj, KEjas, KEond = skel.copy(), skel.copy(), skel.copy(), skel.copy()
    PDjfm, PDamj, PDjas, PDond = skel.copy(), skel.copy(), skel.copy(), skel.copy()

    njfm, namj, njas, nond = 0, 0, 0, 0
    jfm = [1, 2, 3]
    amj = [4, 5, 6]
    jas = [7, 8, 9]
    ond = [10, 11, 12]
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2, mo = yeari[:-3], int(yeari[-2:])
        print(yeari)
        nci = Dataset(fnamen)
        u = nci.variables['UVEL'][0,:,:fcap,:]
        v = nci.variables['VVEL'][0,:,:fcap,:]
        ke = nci.variables['KE'][0,:,:fcap,:]
        pd = nci.variables['PD'][0,:,:fcap,:]
        if mo in jfm:
            Ujfm += u
            Vjfm += v
            KEjfm += ke
            PDjfm += pd
            njfm+=1
        elif mo in amj:
            Uamj += u
            Vamj += v
            KEamj += ke
            PDamj += pd
            namj+=1
        elif mo in jas:
            Ujas += u
            Vjas += v
            KEjas += ke
            PDjas += pd
            njas+=1
        elif mo in ond:
            Uond += u
            Vond += v
            KEond += ke
            PDond += pd
            nond+=1

    Ujfm, Vjfm, KEjfm, PDjfm = Ujfm/njfm, Vjfm/njfm, KEjfm/njfm, PDjfm/njfm
    Uamj, Vamj, KEamj, PDamj = Uamj/namj, Vamj/namj, KEamj/namj, PDamj/namj
    Ujas, Vjas, KEjas, PDjas = Ujas/njas, Vjas/njas, KEjas/njas, PDjas/njas
    Uond, Vond, KEond, PDond = Uond/nond, Vond/nond, KEond/nond, PDond/nond

    vnames = ['U', 'V', 'KE', 'PD']
    djfm, damj, djas, dond = dict(), dict(), dict(), dict()
    _ = [djfm.update({vname:vars()[vname+'jfm']}) for vname in vnames]
    _ = [damj.update({vname:vars()[vname+'amj']}) for vname in vnames]
    _ = [djas.update({vname:vars()[vname+'jas']}) for vname in vnames]
    _ = [dond.update({vname:vars()[vname+'ond']}) for vname in vnames]
    savez(fname_out_duvke_clim, jfm=djfm, amj=damj, jas=djas, ond=dond, lon=lont, lat=latt, t=t, z=z, start_year=START_YEAR, end_year=END_YEAR)

if CALC_MULTIYEARLY_TSDUVKE:
    nmyr = 12.
    NMO_avg = nmyr*NYR_avg
    skel = np.zeros_like(nc.variables['TEMP'][0,:,:fcap,:])
    Uyr, U = [], skel.copy()
    Vyr, V = [], skel.copy()
    KEyr, KE = [], skel.copy()
    TEMPyr, TEMP = [], skel.copy()
    SALTyr, SALT = [], skel.copy()
    PDyr, PD = [], skel.copy()
    nmo=0
    for fnamen in fnames:
        yeari = fnamen.split('/')[-1].split('.')[-2]
        yeari2 = yeari[:-3]
        print(yeari)
        nci = Dataset(fnamen)
        u = nci.variables['UVEL'][0,:,:fcap,:]
        v = nci.variables['VVEL'][0,:,:fcap,:]
        ke = nci.variables['KE'][0,:,:fcap,:]
        temp = nci.variables['TEMP'][0,:,:fcap,:]
        salt = nci.variables['SALT'][0,:,:fcap,:]
        pd = nci.variables['PD'][0,:,:fcap,:]

        U += u
        V += v
        KE += ke
        TEMP += temp
        SALT += salt
        PD += pd
        nmo+=1
        if nmo==1:
            yearl = yeari2
        elif nmo==NMO_avg:
            U, V, KE = map(stripmsk, (U, V, KE))
            TEMP, SALT, PD = map(stripmsk, (TEMP, SALT, PD))
            Uyr.append(U/nmo)
            Vyr.append(V/nmo)
            KEyr.append(KE/nmo)
            TEMPyr.append(TEMP/nmo)
            SALTyr.append(SALT/nmo)
            PDyr.append(PD/nmo)
            #
            U = U*0
            V = V*0
            KE = KE*0
            TEMP = TEMP*0
            SALT = SALT*0
            PD = PD*0
            yearr = yeari2
            t.append(yearl+'-'+yearr)
            nmo=0

    cm2msq = cm2m**2
    cmcub2mcub = 1e3
    fbad = np.logical_or(Uyr[0]>thresh, ~np.isfinite(Uyr[0]))
    for n in range(len(t)):
        Uyr[n][fbad] = np.nan
        Vyr[n][fbad] = np.nan
        KEyr[n][fbad] = np.nan
        TEMPyr[n][fbad] = np.nan
        SALTyr[n][fbad] = np.nan
        PDyr[n][fbad] = np.nan
    Uyr = [uyr*cm2m for uyr in Uyr]
    Vyr = [vyr*cm2m for vyr in Vyr]
    KEyr = [keyr*cm2msq for keyr in KEyr]
    PDyr = [rhoyr*cmcub2mcub for rhoyr in PDyr]

    savez(fname_out_tsduvke, u=Uyr, v=Vyr, ke=KEyr, temp=TEMPyr, salt=SALTyr, pdens=PDyr, lon=lont, lat=latt, t=t)
