#!/usr/bin/env python

import argparse
import numpy as np
import flashlib as fl
import cfpack as cfp
from cfpack import print, stop

# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make radial profiles from FLASH output files.')
    parser.add_argument('inputfiles', type=str, nargs='+', help='Input filename(s).')
    parser.add_argument("-dset", type=str, default='dens', help="HDF5 dataset name.")
    parser.add_argument("-radius", type=float, help="Radius in which to compute radial profile.")
    parser.add_argument("-center", type=float, nargs=3, help="Center for radial profile (default: center on max dens).")
    parser.add_argument("-nbins", type=int, default=100, help="Number of bins for radial profile.")
    parser.add_argument("-rmin", type=float, help="Min radius for binning.")
    parser.add_argument("-rmax", type=float, help="Max radius for binning.")
    args = parser.parse_args()

    files = sorted(args.inputfiles)

    # loop over all input files
    for filen in files:
        gg = fl.FlashGG(filen) # create FlashGG object for this file 'filen'

        if args.center is None:
            maxloc = gg.GetMinMax("dens").max_loc
            args.center = maxloc

        if args.radius is None:
            args.radius = gg.L[0] / 100

        cell_data = np.array([])
        cell_radius = np.array([])

        for b in range(gg.NumBlocks): # loop over all blocks
            if gg.NodeType[b] == 1: # if it is a leaf block
                x, y, z = gg.GetCellCoords(b) # get cell coordinate of this block
                # cell distance to center
                r = np.sqrt((x-args.center[0])**2 + (y-args.center[1])**2 + (z-args.center[2])**2)
                # index to all cells with r <= radius
                ind = r <= args.radius
                # read full block variable
                block_data = gg.ReadBlockVar(b, args.dset)
                # only select cells with r < args.radius (index 'ind'), and append to cell_data
                cell_data = np.append(cell_data, block_data[ind])
                cell_radius = np.append(cell_radius, r[ind])


        # now bin in radius
        # first define the bins
        if args.rmin is None: args.rmin = 0.0
        if args.rmax is None: args.rmax = np.max(cell_radius)
        rbins_faces = cfp.get_1d_coords(cmin=args.rmin, cmax=args.rmax, ndim=args.nbins+1, cell_centred=False)
        rbins = (rbins_faces[1:]+rbins_faces[:-1])/2
        binned_data = np.zeros(args.nbins)
        binned_norm = np.zeros(args.nbins)
        for i in range(len(cell_radius)):
            ind = np.argmin(np.abs(rbins-cell_radius[i]))
            binned_data[ind] += cell_data[i]
            binned_norm[ind] += 1

        # normalise
        binned_data /= binned_norm

        # plot
        ind = ~np.isnan(binned_data)
        cfp.plot(x=rbins[ind], y=binned_data[ind], show=True)

