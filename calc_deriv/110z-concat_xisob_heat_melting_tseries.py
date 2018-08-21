# -*- coding: utf-8 -*-
#
# Description: Concatenate *npz files with the chunks
#              of the heat transport time series.
#
# Author:      André Palóczy Filho
# E-mail:      paloczy@gmail.com
# Date:        December/2017

import numpy as np
from glob import glob
from datetime import datetime
from reproducibility import savez

isob = 1000
# fnameglob = 'Tfmin_tseries%dm_????-????.npz'%isob
# fname_concat = 'Tfmin_tseries%dm.npz'%isob
fnameglob = 'hflxmelt_tseries%dm_????-????.npz'%isob
fname_concat = 'hflxmelt_tseries%dm.npz'%isob

# In [81]: Tfmins.min()
# Out[81]: -2.638742253002648

# aux_vars = []
# concat_vars = ['t', 'Tfmins']
aux_vars = ['d', 'i', 'j', 'y', 'x', 'z', 'Tf0']
concat_vars = ['t', 'Ux', 'Qm', 'SHT', 'UQx', 'UQxe', 'UQxm', 'UQxm_100m', 'UQxe_100m', 'UQxm_100m_700m', 'UQxe_100m_700m', 'UQxm_700m_1000m', 'UQxe_700m_1000m', 'UQxm_circ', 'UQxe_circ']

fnames = glob(fnameglob)
fnames.sort()
for avar in aux_vars:
    v = np.load(fnames[0])[avar]
    vars().update({avar:v})

for cvar in concat_vars:
    skel = None
    for fname in fnames: # Get the variable from all files.
        v = np.load(fname)[cvar]
        if v.ndim==1:
            if skel is not None:
                skel = np.hstack((skel, v))
            else:
                skel = v
        else:
            if skel is not None:
                skel = np.vstack((skel, v))
            else:
                skel = v
    vars().update({cvar:skel})

# Make sure months are centered on the 15th.
t = np.array([datetime(ti.year, ti.month, 15) for ti in t.tolist()])

ds = dict(t=None)
allvars = aux_vars + concat_vars
for k in allvars:
    ds.update({k:vars()[k]})

savez(fname_concat, **ds)
