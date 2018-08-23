## AntarcticaCircumpolarIntrusions large files (100 MB - 1 GB)

This file contains download links for the large data files (larger than 100 MB but smaller than 1 GB). These are:

* [Jb.npz](https://www.dropbox.com/s/23iy2f625ozlhhn/Jb.npz?dl=0) (758 MB)
* [gradRHO.npz](https://www.dropbox.com/s/6qjxuh9fas7n6dg/gradRHO.npz?dl=0) (758 MB)
* [PTbot.npz](https://www.dropbox.com/s/s0qc65cp37zpwde/PTbot.npz?dl=0) (730 MB)
* [POP-dzu_dzt_kzit_subsetSO.nc](https://www.dropbox.com/s/f9se6hkjrxzjxlv/POP-dzu_dzt_kzit_subsetSO.nc?dl=0) (654 MB)
* [POP_topog.nc](https://www.dropbox.com/s/kl1k1vsj49uhxfr/POP_topog.nc?dl=0) (396 MB)
* [MEOPgrd_2005-2009.npz](https://www.dropbox.com/s/o8k8jdyxs0s0pik/MEOPgrd_2005-2009.npz?dl=0) (207 MB)

Once you download the file required by the notebook you are attempting to run, place it on
this directory **(AntarcticaCircumpolarIntrusions/data_reproduce_figs/)** on your local copy of the repository.

### Metadata for some of the .npz files:

---

__htadv_tseries.npz__

This file contains the mean, eddy and total cross-isobath heat transports integrated over the along-isobath span of each segment. It also contains the divergence of the total heat transport in the along-shelf direction at each segment.

__Variables:__

- __t__: Time index, 612 months from Jan/1959 through Dec/2009.
- __uqx_onshore__ (dictionary): Time series of the total (mean+eddy) heat transport entering the domain of each segment, positive onshore.
- __uqxm_onshore__ (dictionary): Same as uqx_onshore, but for the mean heat transport.
- __uqxe_onshore__ (dictionary): Same as uqx_onshore, but for the eddy heat transport.
- __uqx_alongshelf_convergence__ (dictionary): Same as _UQxdiv_ashf_ in the "hflxmelt_alongshelf_xwbry_tseries1000m.npz" file (see below), but with the sign flipped. It is positive when there is along-shelf heat __convergence__ in the segment.

---

__hflxmelt_alongshelf_xwbry_tseries1000m.npz__

This file contains time series of heat transports across the western walls of each segment, and the divergence of the transports in the along-shelf direction, for each segment.

__Variables:__

- __t__: Time index, 612 months from Jan/1959 through Dec/2009.
- __UQx__ (dictionary): Total (mean+eddy) along-shelf heat transport across the _west_ wall of the segment, positive eastward. For example, UQx['Bellingshausen'] is the time series of the total heat transport entering the Bellingshausen segment from the east wall of the Amundsen segment (or west wall of the Bellingshausen segment).
- __UQxm__ (dictionary): Same as UQx, but for the mean component of the heat transports.
- __UQxe__ (dictionary): Same as UQx, but for the eddy component of the heat transports.
- __Ux__ (dictionary): Same as UQx, but for the along-shelf _volume_ transports across segments.
- __UQxdiv_ashf__ (dictionary): The along-shelf divergence of the total heat transport in each segment. It is the difference between the east and west transports. It is positive when there is less heat entering the domain across the west wall than heat exiting the domain across the east wall.
- __Uxdiv_ashf__ (dictionary): Same as UQxdiv_ashf, but for the along-shelf _volume_ transports across segments.

---

Please [contact André Palóczy](mailto:apaloczy@ucsd.edu) with any questions or if you have issues downloading the files.
