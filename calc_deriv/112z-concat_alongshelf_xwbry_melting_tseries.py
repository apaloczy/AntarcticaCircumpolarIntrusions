# -*- coding: utf-8 -*-
#
# Description: Concatenate *npz files with the chunks
#              of the (along-shelf) heat transport time series.
#
# Author:      André Palóczy Filho
# E-mail:      paloczy@gmail.com
# Date:        February/2018

import numpy as np
from glob import glob
from datetime import datetime
from reproducibility import savez

isob = 1000

fnameglob = 'hflxmelt_alongshelf_xwbry_tseries%dm_????-????.npz'%isob
fname_concat = 'hflxmelt_alongshelf_xwbry_tseries%dm.npz'%isob

concat_vars = ['t', 'UQx', 'UQxm', 'UQxe', 'Ux']
segs = ['S-AP', 'N-AP', 'E-EA', 'Byrd', 'Bellingshausen', 'W-EA', 'Weddell', 'Amundsen', 'C-EA', 'Ross']
fnames = glob(fnameglob)
fnames.sort()
for cvar in concat_vars:
    skel = None
    for fname in fnames: # Get the variable from all files.
        v = np.array(np.load(fname)[cvar]).flatten()
        if cvar=='t': # v is an array here.
            if skel is not None:
                skel = np.concatenate((skel, v))
            else:
                skel = v
        else:
            v = v.flatten()[0] # v is a dictionary here.
            _ = [v.update({segn:v[segn]}) for segn in segs]
            if skel is not None:
                for segn in segs:
                    oldd = skel[segn]
                    newd = v[segn]
                    aux = np.concatenate((oldd, newd))
                    skel.update({segn:aux})
            else:
                skel = v
    vars().update({cvar:skel})

# Add along-self divergences for each segment for convenience.
segl = ['Amundsen', 'Bellingshausen', 'S-AP', 'N-AP', 'Weddell', 'W-EA', 'C-EA', 'E-EA', 'Ross', 'Byrd']
segr = segl[1:]
segr.append(segl[0])

UQxdiv_ashf, Uxdiv_ashf = dict(), dict()
for sl, sr in zip(segl, segr):
    UQxdiv_ashf.update({sl:(UQx[sr] - UQx[sl])})
    Uxdiv_ashf.update({sl:(Ux[sr] - Ux[sl])})

concat_vars.append('UQxdiv_ashf')
# Make sure months are centered on the 15th.
t = np.array([datetime(ti.year, ti.month, 15) for ti in t.tolist()])

ds = dict()
allvars = concat_vars + ['Uxdiv_ashf']
for k in allvars:
    ds.update({k:vars()[k]})

savez(fname_concat, **ds)
