#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2020-2025

import cfpack as cfp
from cfpack import hdfio, print, stop
from cfpack.mpi import MPI, comm, nPE, myPE
import argparse
import numpy as np
import h5py
import shutil
from scipy import interpolate, stats
import os
from tqdm import tqdm

# === START class 'FlashGG' similar to C++ FlashGG to handle Flash block data ===
class FlashGG:

    def __init__(self, filename, verbose=1):
        self.filename = filename
        self.verbose = verbose
        # determine grid type
        self.GridType = self.GetGridType()
        if self.verbose > 1: print("GridType = "+self.GridType)
        # determine boundary type
        self.BoundaryType = self.GetBoundaryType()
        if self.verbose > 1: print("BoundaryType = "+self.BoundaryType)
        # extracted grid not fully implemented
        if self.GridType == 'E': return
        # read bounding box
        self.BoundingBox = hdfio.read(self.filename, "bounding box")
        # set number of blocks
        self.NumBlocks = np.shape(self.BoundingBox)[0]
        self.NumBlocksRep = self.NumBlocks
        # read node type
        self.NodeType = self.ReadNodeType()
        # create list of leaf blocks
        leaf_blocks = []
        for b in range(self.NumBlocks): # loop over all blocks
            if self.NodeType[b] == 1: # if it is a leaf block
                leaf_blocks.append(b)
        self.LeafBlocks = np.array(leaf_blocks)
        # number of leaf blocks
        self.NumLeafBlocks = len(self.LeafBlocks)
        # read refine level
        self.RefineLevel = self.ReadRefineLevel()
        # read runtime parameters
        self.runtime_params = read_runtime_parameters(self.filename)
        # read scalars
        self.scalars = read_scalars(self.filename)
        # time
        self.time = self.scalars['time']
        # redshift
        self.redshift = self.scalars['redshift']
        # Ndim
        self.Ndim = self.scalars['dimensionality']
        # set number of cells per block (NB)
        self.NB = np.zeros(3).astype(int)
        self.NB[0] = self.scalars['nxb']
        self.NB[1] = self.scalars['nyb']
        self.NB[2] = self.scalars['nzb']
        # base grid resolution (number of cells)
        self.NBaseGrid = np.array([])
        for i in range(3):
            if self.GridType == 'A': parname = ["nblockx", "nblocky", "nblockz"]
            if self.GridType == 'U': parname = ["iprocs", "jprocs", "kprocs"]
            self.NBaseGrid = np.append(self.NBaseGrid, self.runtime_params[parname[i]]*self.NB[i])
        self.NBaseGrid = self.NBaseGrid.astype(int) * 2**(self.RefineLevel.min()-1)
        # maximum effective resolution (number of cells)
        self.NMax = self.NBaseGrid * 2**(self.RefineLevel.max()-1)
        # size of each block
        self.LB = self.BoundingBox[:,:,1]-self.BoundingBox[:,:,0]
        # set cell size of each block
        self.D = self.LB / self.NB
        # size of domain
        self.L = np.array([np.max(self.BoundingBox[:,x,:])-np.min(self.BoundingBox[:,x,:]) for x in range(3)])
        # domain bounds
        self.domain_bounds = np.array([[self.runtime_params['xmin'], self.runtime_params['xmax']],
                                       [self.runtime_params['ymin'], self.runtime_params['ymax']],
                                       [self.runtime_params['zmin'], self.runtime_params['zmax']]])
        # define an analysis centre (e.g., for radial profiles; call to SetAnalysisCentre)
        self.centre = np.mean(self.domain_bounds, axis=1) # center of the simulation box
        self.binned_stats_buffer = None # buffer storage for binned_statistics

    def GetMyBlocks(self, MyPE, NPE, BlockList=None, allow_idle_cores=False):
        if BlockList is None: BlockList = self.LeafBlocks # default is to distribute all leaf blocks
        nblk = len(BlockList) # number of blocks
        DivBlocks = int(np.ceil(nblk/NPE))
        NPE_main  = int(nblk/DivBlocks)
        ModBlocks = int(nblk-NPE_main*DivBlocks)
        if MyPE < NPE_main: # (NPE_main) cores get DivBlocks blocks
            MyBlocks = BlockList[MyPE*DivBlocks:MyPE*DivBlocks+DivBlocks]
        if self.verbose > 1:
            print("First "+str(NPE_main)+" core(s) carry(ies) "+str(DivBlocks)+" block(s) (each).")
        if (MyPE == NPE_main) and (ModBlocks > 0): # core (NPE_main + 1) gets the rest (ModBlocks)
            MyBlocks = BlockList[NPE_main*DivBlocks:NPE_main*DivBlocks+ModBlocks]
            if self.verbose > 1: print("Core #"+str(NPE_main+1)+" carries "+str(ModBlocks)+" block(s).")
        NPE_in_use = NPE_main
        if ModBlocks > 0:
            NPE_in_use += 1
        if NPE_in_use < NPE:
            print("Non-optimal load balancing; "+str(NPE-NPE_in_use)+" core(s) remain(s) idle.", warn=True)
            if not allow_idle_cores:
                print("Need to adjust number of cores to avoid idle cores.", error=True)
                exit()
        if self.verbose > 1:
            print("MyBlocks = ", MyBlocks, color='cyan', mpi=True)
        return MyBlocks

    # set centre (e.g., for making radial profiles); centre must be list or array of 3 elements
    def SetCentre(self, centre=None):
        c = np.array(centre)
        if len(c) != 3: print("centre must have 3 elements for x, y, z", error=True)
        self.centre = c

    # get grid type (extracted: 'E', AMR: 'A', uniform: 'U')
    def GetGridType(self):
        grid_type = None
        if "minmax_xyz" in hdfio.get_dataset_names(self.filename): grid_type = 'E'
        else:
            runtime_params = read_runtime_parameters(self.filename)
            if 'lrefine_max' in runtime_params.keys(): grid_type = 'A'
            else: grid_type = 'U'
        return grid_type

    # get boundary type (extracted: 'E', AMR: 'A', uniform: 'U')
    def GetBoundaryType(self):
        boundary_type = 'isolated'
        if self.GridType == 'E': # E grid
            if "pbc" in hdfio.get_dataset_names(self.filename):
                if hdfio.read(self.filename, "pbc") == 1: boundary_type = 'periodic'
        else: # A or U grid
            runtime_params = read_runtime_parameters(self.filename)
            if runtime_params["xl_boundary_type"] == 'periodic': boundary_type = 'periodic'
        return boundary_type

    # get uniform grid
    @cfp.timer_decorator
    def GetUniformGrid(self, dset="dens"):
        import copy
        if self.NumBlocks != self.NumLeafBlocks:
            print("highest refinement level is not a uniform grid", error=True)
        grid_loc = np.zeros(self.NMax) # empty return array of full size
        LB = copy.deepcopy(self.LB) # fix 1D and 2D, which have zero-sized LB directions
        # get block position multiplier in domain
        for b in range(self.NumBlocks): LB[b,np.where(LB[b]==0.0)[0]] = 1.0
        block_pos_ind = np.round((self.BoundingBox[:,:,0] - self.domain_bounds[:,0]) / LB).astype(int)
        MyBlocks = self.GetMyBlocks(myPE, nPE) # domain decomposition
        for b in tqdm(MyBlocks, disable=(self.verbose==0 or myPE!=0), desc=cfp.get_frame().signature): # block loop
            # put block into ret array
            ib = block_pos_ind[b]*self.NB
            ie = ib + self.NB
            grid_loc[ib[0]:ie[0],ib[1]:ie[1],ib[2]:ie[2]] = self.ReadBlockVar(b, dsets=dset)
        # MPI reduction operation
        if MPI:
            ret = np.zeros(np.shape(grid_loc))
            comm.Reduce(grid_loc, ret, op=MPI.SUM) # master gets total sum
        else:
            ret = grid_loc
        return ret.squeeze() # return with irrelevant dimensions removed

    # read entire variable (all blocks)
    # Note that this will create an array that may be too big if the simulation contains too many blocks and cells
    # Use ReadBlockVar instead, to read the data block-by-block
    def ReadVar(self, dsets="dens", leaf=True):
        if type(dsets) != list: dsets = [dsets] # if user requests only a single dset, turn into list of 1 element for internal processing
        data = [] # prep return array
        for dset in dsets:
            all_blocks_data = []
            if leaf: loop_elements = self.LeafBlocks
            else: loop_elements = range(self.NumBlocks)
            for b in loop_elements:
                all_blocks_data.append(self.ReadBlockVar(b, dset))
            data.append(all_blocks_data) # append to output list
        if len(dsets) == 1: data = data[0] # strip first dim if only 1 dset was requested
        return np.array(data)

    # read block variable
    def ReadBlockVar(self, block, dsets="dens"):
        if type(dsets) != list: dsets = [dsets] # if user requests only a single dset, turn into list of 1 element for internal processing
        b = block % self.NumBlocks # dealing with PBCs if looping on block replica indices
        data = [] # prep return array
        for dset in dsets:
            # check if dset is in file
            if dset in hdfio.get_dataset_names(self.filename):
                data.append(hdfio.read(self.filename, dset, ind=b).T) # read directly from file
            else:
                data.append(self.CreateDerivedBlockVar(b, dset)) # create a derived variable
        if len(dsets) == 1: data = data[0] # strip first dim if only 1 dset was requested
        return np.array(data)

    # create derived block variable (in case of dset='radius', make use of the boundary conditions)
    def CreateDerivedBlockVar(self, block, dset="radius"):
        if dset == "radius":
            coords = np.array(self.GetCellCoords(block)) # get cell coordinates (x, y, z in first index)
            coords = (coords.T - self.centre).T # subtract the centre
            if self.BoundaryType == 'periodic':
                for d in range(3): # loop over x, y, z
                    ind = coords[d] < self.domain_bounds[d][0]; coords[d][ind] += self.L[d] # coord < lower domain bound
                    ind = coords[d] > self.domain_bounds[d][1]; coords[d][ind] -= self.L[d] # coord > upper domain bound
            return np.sqrt(np.sum(coords**2, axis=0)) # return radial distance to centre
        elif dset == "temperature":
            dens = self.ReadBlockVar(block, dsets="dens")
            pres = self.ReadBlockVar(block, dsets="pres")
            temp = pres / dens * 2.3*cfp.constants.m_p / cfp.constants.k_b
            return temp
        elif dset == "vorticity": # vorticity magnitude
            vort_x = self.ReadBlockVar(block, dsets="vorticity_x")
            vort_y = self.ReadBlockVar(block, dsets="vorticity_y")
            vort_z = self.ReadBlockVar(block, dsets="vorticity_z")
            vort = np.sqrt(vort_x**2 + vort_y**2 + vort_z**2)
            return vort
        else:
            print("derived dset = '"+dset+"' unknown; not implemented yet", error=True)

    # read bounding box
    def ReadBoundingBox(self):
        return hdfio.read(self.filename, "bounding box")

    # read node type
    def ReadNodeType(self):
        return hdfio.read(self.filename, "node type")

    # read refinement level
    def ReadRefineLevel(self):
        return hdfio.read(self.filename, "refine level")

    # get information on refinement (number of blocks and volume filling fraction on each level)
    def GetRefinementInfo(self):
        refine_level = self.RefineLevel[self.LeafBlocks]
        print("min/max level = "+str(refine_level.min())+", "+str(refine_level.max()))
        levels = np.unique(refine_level)
        nbl = []
        for level in levels: nbl.append((refine_level == level).sum())
        nbl = np.array(nbl)
        frac = nbl/self.NumLeafBlocks
        for il, level in enumerate(levels):
            print("Level "+str(level)+": nblocks="+str(nbl[il])+" (fraction: "+cfp.round(frac[il]*100, str_ret=True)+"%)")
        return nbl, frac

    # get the cell coordinates of a block
    def GetCellCoords(self, block):
        b = block % self.NumBlocks # dealing with PBCs if looping on block replica indices
        x, y, z = cfp.get_3d_coords(cmin=self.BoundingBox[b,:,0], cmax=self.BoundingBox[b,:,1], ndim=self.NB, cell_centred=True)
        return x, y, z

    # get the cell sizes (deltas) of a block
    def GetCellDeltas(self, block):
        b = block % self.NumBlocks # dealing with PBCs if looping on block replica indices
        d = self.D[b]
        dx, dy, dz = np.full(self.NB, d[0]), np.full(self.NB, d[1]), np.full(self.NB, d[2])
        return dx, dy, dz

    # get integral quantity of a dataset
    def GetIntegralQuantity(self, datasetname="dens", statistic="mean", weighting="volume"):
        # catch user settings
        statistics = ["mean", "std"]
        if statistic not in statistics: print("statistic='"+statistic+"' not supported; statistic must be any of", statistics, error=True)
        weightings = ["volume", "mass"]
        if weighting not in weightings: print("weighting='"+weighting+"' not supported; weighting must be any of", weightings, error=True)
        # init accumulators
        dsum1 = 0.0
        dsum2 = 0.0
        wsum = 0.0
        # accumulate (loop over all blocks)
        for b in range(self.NumBlocks):
            if self.NodeType[b] == 1: # leaf block
                dset = self.ReadBlockVar(b, datasetname)
                cell_vol = np.prod(self.D[b])
                if weighting == "volume": weight = np.full(dset.shape, cell_vol)
                if weighting == "mass": weight = self.ReadBlockVar(b, "dens") * cell_vol
                if statistic == "mean":
                    dsum1 += np.sum(dset*weight)
                    wsum += np.sum(weight)
                if statistic == "std":
                    dsum1 += np.sum(dset*weight)
                    dsum2 += np.sum(dset**2*weight)
                    wsum += np.sum(weight)
        # finally, define the return values for each statistics case
        ret = None # default return value
        if statistic == "mean": ret = dsum1 / wsum
        if statistic == "std": ret = np.sqrt( (dsum2/wsum) - (dsum1/wsum)**2 )
        return ret

    # get minimum and maximum of dataset and their location
    def GetMinMax(self, datasetname="dens"):
        minmax_loc = [np.zeros(3), np.zeros(3)] # initialise location
        minmax = [np.inf, -np.inf] # initialise minmax
        MyBlocks = self.GetMyBlocks(myPE, nPE) # domain decomposition
        for b in tqdm(MyBlocks, disable=(self.verbose==0 or myPE!=0), desc=cfp.get_frame().signature): # block loop
            dset = self.ReadBlockVar(b, datasetname)
            minmax_dest = [np.min(dset), np.max(dset)]
            for i, sign in enumerate([-1,1]): # min/max
                if sign * minmax_dest[i] > sign * minmax[i]:
                    minmax[i] = minmax_dest[i]
                    array = np.where(dset == minmax_dest[i])
                    index = np.array([val[0] for val in array]) # take first element with min/max (in case there are more)
                    minmax_loc[i] = self.BoundingBox[b,:,0] + (index[:]+0.5)*self.D[b,:]
        # MPI reduction operation
        if MPI:
            # perform allreduce (lower-case) with MINLOC/MAXLOC to get the PE that holds the min/max
            # and broadcast min/max location
            minmax[0], PE_with_min = comm.allreduce((minmax[0], myPE), op=MPI.MINLOC)
            comm.Bcast(minmax_loc[0], root=PE_with_min)
            minmax[1], PE_with_max = comm.allreduce((minmax[1], myPE), op=MPI.MAXLOC)
            comm.Bcast(minmax_loc[1], root=PE_with_max)
        # define return object
        class ret:
            def __init__(self, minmax_, minmax_loc_):
                self.min = minmax_[0]
                self.max = minmax_[1]
                self.min_loc = minmax_loc_[0]
                self.max_loc = minmax_loc_[1]
        return ret(minmax, minmax_loc)

    # Performs statistical analysis (mean, percentile, standard deviation) of x vs. y based on 2D histogram.
    # For radial profile x='radius' around 'centre' (default: domain centre).
    # Options for 'weight' = ['vol', 'mass'] for volume and mass weighting.
    # Options for 'statistic' = ['sum', 'mean', 'std', 'per_<insert percentile desired>'] (e.g. 'per_50' for median).
    # 'bins' is number of bins or arrays (bin edges); 'range' is binning range in x and y.
    # Options for 'bin_type' = ['lin', 'log'] for linear or logarithmic binning.
    # Set 'remove_nan'=True to remove NaN from output.
    def binned_statistic(self, x='radius', y='dens', centre=None, weight='vol', statistic='mean', bins=[300, 200],
                         range=[[None, None], [None, None]], bin_type=['lin', 'lin'], remove_nan=False, use_hist=False, verbose=1):
        if verbose: print("Total number of MPI ranks = "+str(nPE))
        if MPI: comm.Barrier()
        # handle input arguments
        if use_hist:
            print("use_hist=True; using histogram mode to compute statistics depends on binning!", warn=True)
        if statistic.split('_')[0] == 'per' and not use_hist:
            print("Percentile statistic only available when use_hist=True", error=True)
        from inspect import currentframe, getargvalues
        frame = currentframe()
        args_info = getargvalues(frame)
        args = {name: args_info.locals[name] for name in args_info.args}
        if args_info.varargs: args[args_info.varargs] = args_info.locals[args_info.varargs]
        if args_info.keywords: args[args_info.keywords] = args_info.locals[args_info.keywords]
        # --- helper function to compute 2D histogram ---
        def get_binned_stats():
            if verbose: print('Computing binned statistic...')
            if verbose > 1: print('binning x='+x+' vs. y='+y)
            if centre is not None: self.SetCentre(centre) # defining the centre
            bin_edges = [[],[]]
            bin_centres = [[],[]]
            for iq, q in enumerate([x, y]): # x and y
                if isinstance(bins[iq], int): # number of bins (int) given as input
                    if np.any(range[iq] == [None, None]): # auto set range
                        minmax_obj = self.GetMinMax(q)
                        range[iq] = [minmax_obj.min, minmax_obj.max]
                    if bin_type[iq] == 'lin':
                        bin_edges  [iq] = np.linspace(range[iq][0], range[iq][1], bins[iq])
                        bin_centres[iq] = (bin_edges[iq][:-1]+bin_edges[iq][1:])/2
                    if bin_type[iq] == 'log':
                        if range[iq][0] <= 0:
                            bin_edges  [iq] = cfp.symlogspace(range[iq][0], range[iq][1], bins[iq])
                            bin_centres[iq] = cfp.symlogspace(range[iq][0], range[iq][1], bins[iq], return_centre=True)
                        else:
                            bin_edges  [iq] = cfp.logspace(range[iq][0], range[iq][1], bins[iq])
                            bin_centres[iq] = cfp.logspace(range[iq][0], range[iq][1], bins[iq], return_centre=True)
                    if verbose > 1: print('bins determined')
                else:
                    if verbose > 1: print('bins already defined')
                    bin_edges  [iq] = np.array(bins[iq])
                    bin_centres[iq] = (bin_edges[iq][:-1]+bin_edges[iq][1:])/2 # assume centres are in the arithmetic middle
            if not use_hist:
                weights_loc = np.zeros(len(bin_edges[0])-1)
                data_loc    = np.zeros(len(bin_edges[0])-1)
                data_sq_loc = np.zeros(len(bin_edges[0])-1)
            else:
                weights_loc = np.zeros([len(bin_edges[0])-1, len(bin_edges[1])-1]) # init the local 2D weights array with zeros
            MyBlocks = self.GetMyBlocks(myPE, nPE) # domain decomposition
            for b in tqdm(MyBlocks, disable=(self.verbose==0 or myPE!=0), desc=cfp.get_frame().signature): # block loop
                xdat, ydat = self.ReadBlockVar(b, dsets=[x, y]) # contains all dataset values
                vol = np.zeros([self.NB[0], self.NB[1], self.NB[2]]) + np.prod(self.D[b]) # contains all volume values
                if weight == 'vol': weights = vol
                if weight == 'mass': weights = vol * self.ReadBlockVar(b, dsets='dens')
                if statistic == 'sum': weights = np.ones(self.NB)
                # For 2D binning, making the dset_arr, weight_arr 1D is necessary
                xdat = xdat.ravel(); ydat = ydat.ravel(); weights = weights.ravel()
                # Accumulate 2D histogram of x vs. y
                if not use_hist:
                    weights_loc += stats.binned_statistic(xdat, values=weights, statistic='sum', bins=bin_edges[0])[0]
                    if statistic == 'sum':
                        data_loc += stats.binned_statistic(xdat, values=ydat, statistic='sum', bins=bin_edges[0])[0]
                    else:
                        data_loc += stats.binned_statistic(xdat, values=weights*ydat, statistic='sum', bins=bin_edges[0])[0]
                        if statistic == 'std':
                            data_sq_loc += stats.binned_statistic(xdat, values=weights*ydat**2, statistic='sum', bins=bin_edges[0])[0]
                else:
                    weights_loc += stats.binned_statistic_2d(xdat, ydat, values=weights, statistic='sum', bins=bin_edges)[0]
            # MPI reduction operation
            if MPI:
                weights_tot = np.zeros(np.shape(weights_loc))
                comm.Allreduce(weights_loc, weights_tot, op=MPI.SUM) # every MPI rank gets the summed global weights
                if not use_hist:
                    data_tot = np.zeros(np.shape(data_loc))
                    comm.Allreduce(data_loc, data_tot, op=MPI.SUM) # every MPI rank gets the summed global
                    data_sq_tot = np.zeros(np.shape(data_sq_loc))
                    comm.Allreduce(data_sq_loc, data_sq_tot, op=MPI.SUM) # every MPI rank gets the summed global
            else:
                weights_tot = weights_loc
                if not use_hist:
                    data_tot = data_loc
                    data_sq_tot = data_sq_loc
            # fill the output buffer
            if use_hist:
                data_tot = None
                data_sq_tot = None
            self.binned_stats_buffer = {'args':args, 'weights':weights_tot, 'data':data_tot, 'data_sq':data_sq_tot,
                                        'bin_edges':bin_edges, 'bin_centres':bin_centres}
        # --- END: get_binned_stats() ---
        if use_hist:
            # check whether we can re-use the existing 2D histogram (e.g., if only 'stat' was changed)
            call_get_binned_stats = False
            if self.binned_stats_buffer is None: call_get_binned_stats = True # first time the function was called
            else: # the function was called before
                for arg in args.keys():
                    if arg in ['x', 'y', 'centre', 'weight', 'bins', 'range', 'bin_type']:
                        if np.any(self.binned_stats_buffer['args'][arg] != args[arg]): call_get_binned_stats = True
        else:
            call_get_binned_stats = True
        # get binned statistic
        if call_get_binned_stats:
            get_binned_stats()
        else:
            if verbose: print('Re-using existing binned statistic buffer...')
        # compute statistics
        if verbose: print('Computing '+weight+'-weighted '+statistic+' of y='+y+' as a function of x='+x)
        weights = self.binned_stats_buffer['weights']
        data = self.binned_stats_buffer['data']
        data_sq = self.binned_stats_buffer['data_sq']
        bin_edges = self.binned_stats_buffer['bin_edges']
        bin_centres = self.binned_stats_buffer['bin_centres']
        if not use_hist:
            good_ind = weights > 0
        else:
            good_ind = np.sum(weights, axis=1) > 0
        y_ret = np.full(shape=len(bin_centres[0]), fill_value=np.nan)
        if statistic == 'sum':
            if not use_hist:
                y_ret[good_ind] = data[good_ind]
            else:
                y_ret[good_ind] = np.sum(weights*bin_centres[1], axis=1)[good_ind]
        if statistic == 'mean':
            if not use_hist:
                y_ret[good_ind] = data[good_ind] / weights[good_ind]
            else:
                y_ret[good_ind] = np.sum(weights*bin_centres[1], axis=1)[good_ind] / np.sum(weights, axis=1)[good_ind]
        if statistic == 'std':
            if not use_hist:
                y_ret[good_ind] = np.sqrt( data_sq[good_ind] / weights[good_ind] - (data[good_ind] / weights[good_ind])**2 )
            else:
                y_ret[good_ind] = np.sqrt( np.sum(weights*bin_centres[1]**2, axis=1)[good_ind] / np.sum(weights, axis=1)[good_ind] -
                                           (np.sum(weights*bin_centres[1], axis=1)[good_ind] / np.sum(weights, axis=1)[good_ind])**2 )
        if statistic.split('_')[0] == 'per':
            def CDF_per(xarray, yarray, per):
                ycdf = 100 * np.cumsum(yarray) / np.sum(yarray)
                idx = np.max(np.where(ycdf<=per)[0])
                return xarray[idx]
            percentile = float(statistic.split('_')[1])
            for i in np.arange(len(good_ind)):
                if good_ind[i]: y_ret[i] = CDF_per(bin_centres[1], weights[i,:], percentile)
        if remove_nan: x_cen = x_cen[good_ind]; y_ret = y_ret[good_ind]
        # define return class
        class ret:
            def __init__(self, xe, xc, ye, yc, y, hist, rm_nan=False):
                if rm_nan:
                    ind_hist = hist > 0
                    ind_x = np.sum(weights, axis=1) > 0
                    ind_y = np.sum(weights, axis=0) > 0
                else:
                    ind_hist = slice(None)
                    ind_x = slice(None)
                    ind_y = slice(None)
                self.xe = xe # x-bin edges
                self.ye = ye # y-bin edges
                self.xc = xc[ind_x] # x-bin centres
                self.yc = yc[ind_y] # y-bin centres
                self.y  = y [ind_y] # statistic
                self.hist = hist[ind_hist] # 2D histogram
        return ret(bin_edges[0], bin_centres[0], bin_edges[1], bin_centres[1], y_ret, weights, rm_nan=remove_nan)

    # add block replicas to simplify handling of periodic boundary conditions (PBCs)
    def AddBlockReplicasPBCs(self):
        bb = np.copy(self.BoundingBox)
        nt = np.copy(self.NodeType)
        rl = np.copy(self.RefineLevel)
        pbc = [0,0,0]
        pbc_factor = [[0],[0],[0]]
        for dim in range(self.Ndim): pbc_factor[dim] = [-1,0,1]
        for pbc[2] in pbc_factor[2]:
            for pbc[1] in pbc_factor[1]:
                for pbc[0] in pbc_factor[0]:
                    if pbc[0]==0 and pbc[1]==0 and pbc[2]==0: continue # do not process the original
                    bbc = np.copy(bb)
                    for dim in range(self.Ndim): bbc[:,dim,:] += pbc[dim]*self.L[dim]
                    self.BoundingBox = np.append(self.BoundingBox, bbc, axis=0) # append to BoundingBox
                    self.NodeType = np.append(self.NodeType, nt, axis=0) # append to NodeType
                    self.RefineLevel = np.append(self.RefineLevel, rl, axis=0) # append to RefineLevel
        self.NumBlocksRep = np.shape(self.BoundingBox)[0] # update NumBlocksRep

    # return list of blocks overlapping with bounds
    def GetAffectedBlocks(self, box_bounds=None, sphere_radius=None, sphere_center=[0,0,0], leaf=True):
        extract_box = False; extract_sphere = False
        if sphere_radius is None: extract_box = True # do box extraction if no radius provided
        else: extract_sphere = True # extract spherical volume
        if extract_box and box_bounds is None:
            box_bounds = np.copy(self.domain_bounds) # use full domain if box_bounds not set
        affected_blocks = [] # start with empty return container
        for b in range(self.NumBlocks): # loop over all blocks
            if leaf:
                if self.NodeType[b] != 1: # not a leaf block
                    continue
            if extract_box: # check if block overlaps with rectangular box
                overlap = True
                for dim in range(3):
                    if self.BoundingBox[b,dim,1] <= box_bounds[dim][0]:
                        overlap = False; break
                    if self.BoundingBox[b,dim,0] >= box_bounds[dim][1]:
                        overlap = False; break
            if extract_sphere: # check if block overlaps with sphere
                overlap = False
                min_dist_squared = 0.0 # init squared minimum distance to 0.0
                for dim in range(3):
                    # check distance between sphere center and block's bounding box in each dimension
                    if sphere_center[dim] < self.BoundingBox[b,dim,0]:
                        min_dist_squared += (self.BoundingBox[b,dim,0] - sphere_center[dim])**2
                    elif sphere_center[dim] > self.BoundingBox[b,dim,1]:
                        min_dist_squared += (sphere_center[dim] - self.BoundingBox[b,dim,1])**2
                # if minimum distance from sphere center to the block is less than the sphere radius, we have overlap
                if min_dist_squared <= sphere_radius**2:
                    overlap = True
            if overlap: affected_blocks.append(b)
        return np.array(affected_blocks)

    # extract data in rectangular (box) or spherical region
    @cfp.timer_decorator
    def GetCells(self, dsets="dens", box_bounds=None, sphere_radius=None, sphere_center=[0,0,0]):
        if type(dsets) != list: dsets = [dsets] # if user only supplies a single dset, turn into list of 1 element for internal processing
        extract_box = False; extract_sphere = False
        if sphere_radius is None: extract_box = True # do box extraction if no radius provided
        else: extract_sphere = True # extract spherical volume
        if extract_box and box_bounds is None:
            box_bounds = np.copy(self.domain_bounds) # use full domain if box_bounds not set
        # get overlapping blocks
        blocks = self.GetAffectedBlocks(box_bounds=box_bounds, sphere_radius=sphere_radius, sphere_center=sphere_center)
        # init return data containers
        n_max_out = len(blocks)*self.NB.prod()
        cell_datas = np.empty((len(dsets),n_max_out)); cell_datas.fill(np.nan)
        cell_coord = np.empty((3,n_max_out)); cell_coord.fill(np.nan)
        cell_delta = np.empty((3,n_max_out)); cell_delta.fill(np.nan)
        ibeg = 0 # start index of selection for output array
        # loop blocks
        monitor = cfp.monitor(len(blocks), signature="GetCells: ")
        for ib, b in enumerate(blocks): # loop over all blocks
            if self.verbose > 1: print("working on block # "+str(ib+1)+" of total affected blocks "+str(len(blocks)))
            x, y, z = self.GetCellCoords(b) # get cell coordinate of this block
            dx, dy, dz = self.GetCellDeltas(b) # get cell sizes (deltas) of this block
            if extract_box: # check if cells overlap with rectangular box
                ind =   (x > box_bounds[0][0]) & (x < box_bounds[0][1]) & \
                        (y > box_bounds[1][0]) & (y < box_bounds[1][1]) & \
                        (z > box_bounds[2][0]) & (z < box_bounds[2][1])
            if extract_sphere: # check if cells overlap with sphere
                distance_squared = (x - sphere_center[0])**2 + (y - sphere_center[1])**2 + (z - sphere_center[2])**2
                ind = distance_squared <= sphere_radius**2
            n_cells = ind.sum() # number of affected cells (number of True values in ind)
            if n_cells > 0: # only extract if we have selected cells in extraction range
                iend = ibeg + n_cells # end index of output array for this selection of cells
                block_datas = self.ReadBlockVar(b, dsets) # read full block variable
                if len(dsets) == 1: block_datas = np.array([block_datas]) # turn into list if we only extract a single dset
                cell_datas[:,ibeg:iend] = block_datas[:,ind] # only select cells within the range and copy into right place in output
                cell_coord[:,ibeg:iend] = [ x[ind],  y[ind],  z[ind]] # coordinates
                cell_delta[:,ibeg:iend] = [dx[ind], dy[ind], dz[ind]] # cell sizes (deltas)
                ibeg = iend # new start index for next selection output
            monitor.report(ib)
        cell_datas = cell_datas[:,0:iend] # trim away all NaNs at the end of the array
        cell_coord = cell_coord[:,0:iend] # trim away all NaNs at the end of the array
        cell_delta = cell_delta[:,0:iend] # trim away all NaNs at the end of the array
        if len(dsets) == 1: cell_datas = cell_datas[0,:] # strip first dim if only 1 dset was requested
        class ret: # class object to be returned
            cell_dat = cell_datas # cell data (or multiple datasets, if requested)
            cell_pos = cell_coord # cell centre coordinates
            cell_del = cell_delta # cell sizes (deltas)
        return ret

# === END class 'FlashGG' ===

# === START class 'Particles' similar to C++ FlashParticles to handle Flash particles ===
class Particles:

    def __init__(self, filename, verbose=1):
        self.verbose = verbose
        filen = filename
        if is_plt_file(filen): # if plt file, we assume the part file has the same number
            filen = filen.replace("_hdf5_plt_cnt_", "_hdf5_part_")
        if not os.path.isfile(filen): # get out of here if there is no particle file
            self.n = 0
            return
        if self.verbose: print("Using '"+filen+"' as input file.")
        self.filename = filen
        # set number of particles and particle properties
        self.n = 0
        self.dsets = None
        h5_dsets = hdfio.get_dataset_names(self.filename)
        if "particle names" in h5_dsets:
            names = hdfio.read(self.filename, "particle names")
            # set particle properties
            self.dsets = [x[0].strip().decode() for x in names]
            self.n = hdfio.get_shape(self.filename, "tracer particles")[0]
        # guess the particle types
        self.set_type()

    # set/guess the particle types
    def set_type(self):
        if self.verbose: print("determining particle types...")
        self.tracer_type = None
        self.sink_type = None
        if self.dsets is None: return # return if there are no particles
        self.types = [1]
        self.n_by_type = [self.n]
        if 'type' in self.dsets:
            types = self.read('type')
            self.types = np.unique(types).astype(int)
            if self.verbose: print("file '"+self.filename+"' contains "+str(len(self.types))+" particle type(s) = "+str(self.types))
            if 'accr_rate' in self.dsets:
                self.tracer_type = 1
                self.sink_type = 2
            self.n_by_type = [(types==self.tracer_type).sum(), (types==self.sink_type).sum()]
            if self.verbose:
                if self.tracer_type is not None:
                    print("Guessing tracer / dark matter particle type is "+str(self.tracer_type)+" with "+str(self.n_by_type[self.tracer_type-1])+" particle(s).")
                if self.sink_type is not None: print("Guessing sink particle type is "+str(self.sink_type)+" with "+str(self.n_by_type[self.sink_type-1])+" particle(s).")
        else:
            if 'accr_rate' in self.dsets:
                self.sink_type = 1
            else:
                self.tracer_type = 1
            if self.verbose: print("There is only one particle type present.")
        return

    # print info
    def print_info(self):
        print("Particle filename = '"+self.filename+"'")
        print("Total number of particles = "+str(self.n))
        print("Particle types = ", self.types)
        if len(self.types) > 1:
            print("Number of tracer / dark matter particles = "+str(self.n_by_type[self.tracer_type-1])+" (tracer_type="+str(self.tracer_type)+")")
            print("Number of sink particles = "+str(self.n_by_type[self.sink_type-1])+" (sink_type="+str(self.sink_type)+")")
        return

    # read particles
    # dsets is the particle property(ies; in case of list input) to be read
    # type is particle type
    # box_bounds allows specifying a bounding box for the read
    def read(self, dsets=["posx", "posy", "posz"], type=None, box_bounds=None):
        from builtins import type as type_bi
        if self.n == 0: return None # return if we don't have any particles
        if type_bi(dsets) != list: dsets = [dsets] # if user only supplies a single dset, turn into list of 1 element for internal processing
        for dset in dsets:
            if dset not in self.dsets:
                print("requested dataset '"+dset+"' not in '"+self.filename+"'. Available datasets: ", self.dsets, error=True)
        # if user wants a specific type, we first check whether the type is actually present
        if type is not None:
            error = False
            if len(self.types) > 1: # there are mutliple particle types
                if type not in self.types: error = True # user requested a non-existent type
            elif type != 1: error = True # user requested a non-existent type
            if error:
                print("Particle type "+str(type)+" not in file (available types are", self.types, ").", error=True)
            if "type" in self.dsets:
                type_arr = self.read("type", box_bounds=box_bounds)
                type_ind = type_arr == type # get the indices matching the type
            else:
                type_ind = slice(None)
        if box_bounds is not None:
            box_bounds = np.array(box_bounds) # turn into numpy array
            pos = self.read(['posx', 'posy', 'posz'], type=type) # read particle positions
            bb_ind = np.full(pos[0].shape, True) # start with all True elements
            for dir in range(3): # find in box_bounds
                bb_ind = np.logical_and(bb_ind, np.logical_and(pos[dir,:] >= box_bounds[dir,0], pos[dir,:] <= box_bounds[dir,1]))
        if type is not None and box_bounds is not None:
            combined_ind = type_ind & bb_ind
        data = []
        for dset in dsets:
            index = self.dsets.index(dset) # find requested particle dataset index
            ind = np.s_[:,index] # index to pass to hdfio for hyperslab selection
            # read particle dset without type distinction
            data.append(hdfio.read(self.filename, "tracer particles", ind=ind))
            if type is not None and box_bounds is not None:
                data[-1] = data[-1][combined_ind] # combined slicing
            else:
                if type is not None: data[-1] = data[-1][type_ind] # type slicing
                if box_bounds is not None: data[-1] = data[-1][bb_ind] # box_bound slicing
        # return
        if len(dsets) == 1: data = data[0] # strip first dim if only 1 dset was requested
        return np.array(data)

    # return the total sink-gas gravitational interaction potential due to all sinks at a cell position
    def GetSinkGasPot(self, cell_pos):
        if self.n == 0: return None # return if we don't have any particles
        if read_runtime_parameters(self.filename)["grav_boundary_type"] == "periodic":
            print("Obtaining the gas potential with periodic boundary conditions not implemented yet. Returning None.")
            return None
        # Get sink-gas softening type
        soft_type = read_runtime_parameters(self.filename)["sink_softening_type_gas"]
        soft_radius = read_runtime_parameters(self.filename)["sink_softening_radius"]
        # Get min cell size to convert to physical units
        filen = self.filename
        filen = filen.replace("_hdf5_part_","_hdf5_plt_cnt_")
        dx_min = np.min(FlashGG(filen).D)
        # Convert soft_radius to dx_min
        soft_radius *= dx_min
        # Obtain radial distance to position
        sinks = self.read(["posx", "posy", "posz", "mass"], type=self.sink_type)
        # Get radial distances to each sink
        rdist = (np.ones(len(sinks[0])) * cell_pos[0] - sinks[0])**2 + \
                (np.ones(len(sinks[1])) * cell_pos[1] - sinks[1])**2 + \
                (np.ones(len(sinks[2])) * cell_pos[2] - sinks[2])**2
        # Linear softening
        if soft_type == 'linear':
            def linear_soft(r, r_soft):
                # linear acceleration kernel for r<r_soft: i.e. a = -GMr/r_soft^3
                # this gives phi = -3GM/(2r_soft) + GMr^2/(2r_soft^3) - the second term is to ensure continuity at r_soft
                if r < r_soft:
                    return -3/(2*r_soft) + r**2/(2*r_soft**3)
                # else usual potential for point mass, i.e. phi = -GM/r
                else:
                    return -1/r
            prefactor_grav = cfp.constants.g_n * sinks[3]
            phi_soften = np.vectorize(linear_soft, excluded=['r_soft'])
            return phi_soften(rdist, r_soft=soft_radius)
        else:
            print("Only linear softening implemented yet, spline softening not yet implemented. Returning None.")
            return None

# === END class 'Particles' ===

# === START class 'datfile' to handle time evolution data files ===
class datfile:

    @cfp.timer_decorator
    def __init__(self, filename, verbose=1, max_num_lines=1e7, read=True, clean=True):
        self.classname = self.__class__.__name__ # class name
        self.filename = filename # data filename
        self.verbose = verbose # for printing to stdout
        self.max_num_lines = int(max_num_lines) # maximum number of lines in data file
        self.header = None # columns header
        self.dat = np.array([]) # 2D data array
        if read: self.read() # read data
        self.sinks_evol = False
        if "sinks_" in self.filename: self.sinks_evol = True # for sinks_evol.dat, sinks_evol_after_outflow.dat, sinks_stellar_evolution.dat
        if clean: self.clean() # clean data
        if self.verbose > 1: print(self.classname+": class instance created.")

    # write cleaned file
    def write_cleaned(self):
        out_filename = self.filename+'_cleaned'
        backup_filename = self.filename+'_sav'
        if self.verbose: print("creating backup copy of '"+self.filename+"' as '"+backup_filename+"'")
        shutil.copyfile(self.filename, backup_filename)
        self.clean()
        self.write(out_filename)

    # remove data lines that were overwritten after FLASH restarts based on time_col
    def clean(self):
        time_col = 0
        if self.sinks_evol: time_col = 1
        completely_cleaned = False
        while not completely_cleaned:
            completely_cleaned = True # first set to True, but can switch to False below
            tmp = np.empty(self.dat.shape) # create temporary work array
            il = 0 # running index for adding cleaned data to tmp
            start_index = len(self.dat) # start index for looping through original data
            done_cleaning = False
            while not done_cleaning:
                for i in reversed(range(start_index)): # loop through all data times in reverse order
                    if i==0: # reached the first data point in time -> finished cleaning
                        done_cleaning = True; break # signal that we are finished cleaning
                    test = self.dat[i-1][time_col] / self.dat[i][time_col] # time ratio of previous line to current line
                    if test < 1:
                        tmp[il] = self.dat[i]; il += 1 # append to output data and increase output counter il
                    if self.sinks_evol and (test == 1):
                        if self.dat[i-1][0] != self.dat[i][0]: # time is the same, but is sink tag different?
                            tmp[il] = self.dat[i]; il += 1 # append to output data and increase output counter il
                    if (test > 1) or ((not self.sinks_evol) and (test == 1)): # keep moving backwards until we hit good data again
                        for j in reversed(range(i-1)):
                            if self.dat[j][time_col] < self.dat[i][time_col]:
                                if self.verbose > 1: print(">>> copied clean data between lines "+str(i+1)+" and "+str(start_index))
                                start_index = j+2; completely_cleaned = False; break
                        if j==0: done_cleaning = True # signal that we are finished cleaning
                        break # break loop over i
            if self.verbose: print("copied clean data between lines "+str(i+1)+" and "+str(start_index))
            tmp[il] = self.dat[i] # copy final missing element
            self.dat = (tmp[:il+1])[::-1] # resize table to correct size and reverse

    # get statistical moments over an interval ([xcol] from xs to xe) of quantity in column ycol
    def get_moments(self, ycol, xcol=0, xs=None, xe=None):
        x = self.dat[:,self.col_ind(xcol)] # extract x column
        y = self.dat[:,self.col_ind(ycol)] # extract y column
        moments = cfp.get_moments_from_time_series(x, y, ts=xs, te=xe)
        return moments

    # interpolate column ycol in the datafile onto xnew, where x is in xcol (usually time)
    def interpolate(self, ycol, xnew=None, xcol=0):
        tab = self.dat
        x = tab[:,self.col_ind(xcol)] # extract x column
        y = tab[:,self.col_ind(ycol)] # extract y column
        f = interpolate.interp1d(x, y, kind='cubic')
        if xnew is None: # if xnew is not given, interpolate onto uniform grid with the same number of points as the input
            xnew = np.linspace(np.nanmin(x), np.nanmax(x), len(x))
        ynew = f(xnew)
        return ynew

    # plot
    def plot_column(self, ycol, xcol=0, cfpack_plot_style=True):
        xc = self.col_ind(xcol)
        yc = self.col_ind(ycol)
        xlabel = cfp.tex_escape(self.header[xc])
        ylabel = cfp.tex_escape(self.header[yc])
        if cfpack_plot_style: cfp.load_plot_style()
        cfp.plot(x=self.dat[:,xc], y=self.dat[:,yc], xlabel=xlabel, ylabel=ylabel, show=True)
        if cfpack_plot_style: cfp.unload_plot_style()
        return

    # get column index
    def col_ind(self, col_id):
        if type(col_id) == int: return col_id # if int, return int as index
        if type(col_id) == str: # if string
            if col_id.isnumeric(): return int(col_id) # if string is number, return as int
            else: # it really is a string id for the column
                str_match_list = np.array([x.find(col_id)!=-1 for x in self.header]) # find wildcard match in header list
                if str_match_list.sum() == 1: # found unique column match
                    return np.argwhere(str_match_list).flatten()[0] # return index of matched col_id str
                else: # either no match at all or multiple matching columns by name
                    print("Error: no (unique) match for column id '"+col_id+"'")
                    print("Matched columns include ", self.header[str_match_list])
                    stop()

    # read datafile
    def read(self, read_header=True, to_float=True):
        with open(self.filename, 'r') as f:
            if read_header: self.header = f.readline() # read header (first line)
            self.header = np.array(self.header.split()) # make header numpy array
            ncol = len(self.header) # number of columns based on (first) header
            # (note that on re-simulate or restart with a different flash exec, this could in principle change)
            self.dat = np.empty((int(self.max_num_lines),ncol)) # init output data table
            il = 0 # index to append line to output table
            check_restart = False
            for line in f: # loop through all lines in file
                if check_restart:
                    check_restart = False
                    try:
                        if float(line.split()[0]) == 0.0: continue # skip this line if time=0 after restart
                    except:
                        continue # skip bad lines (some element on the line could not be converted to float)
                if line.strip() == "# simulation restarted":
                    check_restart = True # check next line for whether time=0
                try:
                    self.dat[il] = np.asarray(line.split()[:ncol], dtype=float); il += 1 # fill table with floats
                except:
                    pass # skip bad lines (some element on the line could not be converted to float)
        self.dat = self.dat[:il] # resize table to correct size
        if self.verbose: print("lines in table   : "+str(len(self.dat)))
        if self.verbose: print("columns in table : "+str(len(self.dat[0])))

    # write datafile
    def write(self, out_filename):
        if self.verbose: print("writing '"+out_filename+"'...")
        header = ''.join([' '+x.rjust(23)[:23] for x in self.header])[1:]
        np.savetxt(out_filename, self.dat, header=header, comments='', fmt='%23.16E')
        if self.verbose: print("'"+out_filename+"' written with "+str(len(self.dat))+" lines.", highlight=3)

# === END class 'datfile' ===


# === START class 'logfile' to handle log file operations ===
class logfile:

    def __init__(self, filename, verbose=1, read=True, run=-1, plot=False):
        self.classname = self.__class__.__name__ # class name
        self.filename = filename # filename
        self.verbose = verbose # for printing to stdout
        # a log file may contain multiple runs with multiple lines and other data
        self.n_runs = 0
        self.run = run
        self.run_str = ""
        self.lines = []
        self.time_stamp = []
        self.NB = []
        self.n_procs = []
        self.setup_line = []
        self.timestep_info = []
        if self.verbose > 1: print(self.classname+": class instance created.")
        if read:
            self.parse_file_into_runs()
            self.print_info()
            self.get_performance_info(plot=plot)

    # reads and parses the log file for useful information; separate into runs
    def parse_file_into_runs(self):
        # helper function to extract performance information per time step
        def parse_line_for_time_step_info(line):
            from datetime import datetime as datetime_class
            if line.find('step: n=') < 0: return None
            if self.verbose > 1: print("Found line: '"+line.rstrip()+"'")
            # Example: ' [ 05-09-2025  11:05:17.988 ] step: n=1 t=0.000000E+00 dt=1.000000E+08'
            parts = line.strip().split(']')
            date_time_str = parts[0].strip(' [')  # '05-09-2025  11:05:17.988'
            rest = parts[1].strip()  # 'step: n=1 t=0.000000E+00 dt=1.000000E+08'
            tokens = rest.split()
            n = int(tokens[1].split('=')[1])
            t = float(tokens[2].split('=')[1])
            dt = float(tokens[3].split('=')[1])
            date_str, time_str = date_time_str.split()
            datetime = datetime_class.strptime(f"{date_str} {time_str}", "%m-%d-%Y %H:%M:%S.%f")
            # define return class
            class ret:
                def __init__(self, n, t, dt, date_str, time_str, datetime):
                    self.n = n
                    self.t = t
                    self.dt = dt
                    self.date_str = date_str
                    self.time_str = time_str
                    self.datetime = datetime
            return ret(n, t, dt, date_str, time_str, datetime)
        # parse log file
        with open(self.filename, 'r') as f:
            minmax_leaf_blks = [1, 1] # default in case of UG mode
            lines = iter(f)
            for line in lines:
                if self.verbose > 1: print("line = ", line.rstrip())
                if "FLASH log file:" in line: # start of a new run
                    self.lines.append([line.rstrip()])
                    self.time_stamp.append(line.split()[3:5]) # get the time stamp of the run
                    self.timestep_info.append([])
                else:
                    # get number of MPI processes
                    self.lines[-1].append(line.rstrip())
                    if line.find("Number of MPI tasks:") > 0:
                        self.n_procs.append(int(line.split()[-1]))
                    # get block info
                    if line.find("Number x zones:") > 0:
                        self.NB.append([])
                        for _ in range(3): # x, y, z
                            self.NB[-1].append(int(line[-6:].rstrip()))
                            line = next(lines)
                    # get setup line
                    if line.find("Setup syntax:") > 0:
                        self.setup_line.append("")
                        for next_line in lines:
                            if "f compiler flags:" not in next_line:
                                self.setup_line[-1] += next_line[1:].rstrip()
                            else:
                                break
                    # get timestep info
                    timestep_info = parse_line_for_time_step_info(line)
                    if timestep_info is not None:
                        self.timestep_info[-1].append(timestep_info)
                    # get min/max blocks info
                    if line.find("[GRID amr_refine_derefine] min leaf blks") > 0:
                        minmax_leaf_blks = [int(line.split()[5]), int(line.split()[9])]
                    if len(self.timestep_info[-1]) > 0:
                        self.timestep_info[-1][-1].minmax_leaf_blks = minmax_leaf_blks
        # set number of runs
        self.n_runs = len(self.lines)
        # for each run, get time step information
        for irun in range(self.n_runs):
            # compute time differences (in seconds) between consecutive steps
            steps = self.timestep_info[irun]
            for istep in range(len(steps)-1, 0, -1):
                if steps[istep].n - steps[istep-1].n != 1: # check that the timesteps were in order
                    print("Consecutive time steps did not seem to be in order; irun, istep = ", irun, istep, error=True)
                steps[istep-1].walltime_used = (steps[istep].datetime - steps[istep-1].datetime).total_seconds()
                steps[istep-1].time_per_cell_per_step = steps[istep-1].walltime_used / np.prod(self.NB[irun]) / steps[istep-1].minmax_leaf_blks[1]
        # set run info
        if self.run == -1:
            self.run_str = "last run"
        else:
            if self.run == 0 or self.run > self.n_runs:
                print("Requested run must be in [1, "+str(self.n_runs)+"]", error=True)
            self.run_str = 'run #' + str(self.run)
            self.run -= 1 # for indexing into arrays

    def print_info(self):
        print("=== Information for log file '"+self.filename+"' (number of runs: "+str(self.n_runs)+") ===", color='green')
        print("Info for "+self.run_str+":", color='magenta')
        print(" Setup line: ", newline=False, color='cyan'); print(self.setup_line[self.run], no_prefix=True)
        print(" Start of run: ", newline=False, color='cyan'); print(self.time_stamp[self.run], no_prefix=True)
        print(" Block size: ", newline=False, color='cyan'); print(self.NB[self.run], no_prefix=True)
        print(" Number of cores used: ", newline=False, color='cyan'); print(str(self.n_procs[self.run]), no_prefix=True)

    def get_performance_info(self, dump=True, plot=False, cfpack_plot_style=True):
        steps = self.timestep_info[self.run]
        def get_wallclock_time(run):
            compute_time = 0.0
            steps = self.timestep_info[run]
            for istep in range(len(steps)-1):
                compute_time += steps[istep].walltime_used
            return compute_time
        if dump:
            # info about all runs combined
            total_compute_time = 0.0
            for irun in range(self.n_runs):
                total_compute_time += get_wallclock_time(irun) * self.n_procs[irun]
            print(" Total compute time used for all runs: "+cfp.eform(total_compute_time/3600,2)+" CPU-h", color='yellow')
            # info about the current run
            wallclock_time = get_wallclock_time(self.run) / 3600
            print(" Wallclock time used for "+self.run_str+": "+cfp.eform(wallclock_time,2)+" h", color='yellow')
            print(" Compute time used for "+self.run_str+": "+cfp.eform(wallclock_time*self.n_procs[self.run],2)+" CPU-h", color='yellow')
            def print_step_info(istep):
                s = steps[istep]
                t_str = cfp.eform(s.t,6)
                if s.t == 0.0: t_str = '0.000000E+00'
                minmax_leaf_blks_str = [f"{n:3d}" for n in s.minmax_leaf_blks]
                print("step: n, t, dt, minmax_leaf_blks, walltime_used, time_per_cell_per_step = ",
                        f"{s.n:7d}", t_str, cfp.eform(s.dt,6), f"[{', '.join(minmax_leaf_blks_str)}]", 
                        cfp.eform(s.walltime_used,2)+"s", cfp.eform(s.time_per_cell_per_step,2)+"s", no_prefix=True)
                return
            print(" Timestep info: ", color='cyan')
            n = len(steps) - 1
            for istep in range(min(3, n)): # first 3 steps
                print_step_info(istep)
            start = max(3, n-3)
            if start > 3:
                print("  ."*45, no_prefix=True, newline=False)
                print(no_prefix=True)
            for istep in range(start, n): # last 3 (skip overlap if total < 6)
                print_step_info(istep)
        # get average performance statistics
        time_per_cell_per_step = np.array([steps[istep].time_per_cell_per_step for istep in range(len(steps)-1)])
        nsteps = len(time_per_cell_per_step)
        time_per_cell_per_step_stats = np.percentile(time_per_cell_per_step,[16,50,84])
        time_per_cell_per_step_stats[0] = time_per_cell_per_step_stats[0]-time_per_cell_per_step_stats[1]
        time_per_cell_per_step_stats[2] = time_per_cell_per_step_stats[2]-time_per_cell_per_step_stats[1]
        if self.verbose:
            print(" Walltime per cell per step averaged over "+str(nsteps)+" time steps: "+
                  cfp.eform(time_per_cell_per_step_stats[1],2)+" (+"+cfp.eform(time_per_cell_per_step_stats[2],2)+
                  "/"+cfp.eform(time_per_cell_per_step_stats[0],2)+") (min/max="+cfp.eform(time_per_cell_per_step.min(),2)+
                  "/"+cfp.eform(time_per_cell_per_step.max(),2)+") s", color='yellow')
        if plot:
            if cfpack_plot_style: cfp.load_plot_style()
            x = np.array([steps[istep].n for istep in range(len(steps)-1)])
            y = np.array([steps[istep].time_per_cell_per_step for istep in range(len(steps)-1)])
            cfp.plot(x=x, y=y, xlabel=r'Time step \#', ylabel="Walltime per cell per step (s)", ylog=True, linewidth=0.5, show=True)
            bins = np.logspace(np.log10(time_per_cell_per_step.min()), np.log10(time_per_cell_per_step.max()), 500)
            po = cfp.get_pdf(time_per_cell_per_step, bins=bins)
            cfp.plot(x=po.bin_edges, y=po.pdf, type='pdf', xlabel="Walltime per cell per step (s)",
                     xlog=True, ylog=True, ylabel="PDF", linewidth=0.5, show=True)
            if cfpack_plot_style: cfp.unload_plot_style()
        return steps, nsteps, time_per_cell_per_step_stats

# === END class 'logfile' ===


# ==================== check for file types ===================
def is_movie_slice(filename):
    return filename.find('_slice_') != -1
def is_movie_proj(filename):
    return filename.find('_proj_') != -1
def is_movie_file(filename):
    return is_movie_slice(filename) or is_movie_proj(filename)
def is_part_file(filename):
    searchstr = '_hdf5_part_'
    index = filename.find(searchstr)
    return (index != -1) and (len(filename)-len(searchstr)-index == 4)
def is_plt_file(filename):
    searchstr = '_hdf5_plt_cnt_'
    index = filename.find(searchstr)
    return (index != -1) and (len(filename)-len(searchstr)-index == 4)
def is_chk_file(filename):
    searchstr = '_hdf5_chk_'
    index = filename.find(searchstr)
    return (index != -1) and (len(filename)-len(searchstr)-index == 4)
def is_extracted_file(filename):
    return filename.find('_extracted.h5') != -1
# ================= end: check for file types =================


# ================= read_runtime_parameters ===================
def read_runtime_parameters(flash_file):
    params_dsets = ['integer runtime parameters', \
                    'real runtime parameters', \
                    'logical runtime parameters', \
                    'string runtime parameters']
    runtime_parameters = dict()
    for dset in params_dsets:
        data = hdfio.read(flash_file, dset)
        for i in range(0, len(data)):
            datstr = data[i][0].strip().decode()
            if dset == 'string runtime parameters':
                datval = data[i][1].strip().decode()
            else:
                datval = data[i][1]
            runtime_parameters[datstr] = datval
    return runtime_parameters
# ================ end: read_runtime_parameters ===============


# ======================= read_scalars ========================
def read_scalars(flash_file):
    scalars_dsets = ['integer scalars', \
                     'real scalars', \
                     'logical scalars', \
                     'string scalars']
    scalars = dict()
    for dset in scalars_dsets:
        data = hdfio.read(flash_file, dset)
        for i in range(0, len(data)):
            datstr = data[i][0].strip().decode()
            if dset == 'string scalars':
                datval = data[i][1].strip().decode()
            else:
                datval = data[i][1]
            scalars[datstr] = datval
    return scalars
# ==================== end: read_scalars ======================


# ======================= write_unk_names ========================
# for flash_file, write "unknown names" given unput array unk_names
def write_unk_names(flash_file, unk_names=["dens", "velx", "vely", "velz"], strlen=40, overwrite=True):
    dsetname = "unknown names"
    unn = np.array([unk_names], dtype='|S'+str(strlen)+'') # create numpy string array
    # create datatype and dataspace
    type_id = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
    type_id.set_size(strlen)
    type_id.set_strpad(h5py.h5t.STR_NULLTERM)
    space = h5py.h5s.create_simple((len(unn[0]),1))
    # open file for read/write
    f = h5py.File(flash_file, "r+")
    if overwrite: del f[dsetname] # delete old "unknown names" dataset
    dset = h5py.h5d.create(f.id, dsetname.encode('utf-8'), type_id, space) # create "unknown names" dataset
    dset.write(space, dset.get_space(), unn) # write dataset
    dset.close() # close dataset
    f.close() # close file
    print("'"+dsetname+"' written in file '"+flash_file+"' with names = ", unk_names, highlight=2)
# ==================== end: write_unk_names ======================


# ======================= read_num_sinks ======================
def read_num_sinks(flash_file):
    num_sinks = 0
    # movie file
    if is_movie_file(flash_file):
        f = h5py.File(flash_file, "r")
        if 'numpart' in f.keys():
            num_sinks = hdfio.read(flash_file, 'numpart')[0]
        f.close()
    # flash particle file
    if is_part_file(flash_file) or is_plt_file(flash_file) or is_chk_file(flash_file):
        particles_obj = Particles(flash_file, verbose=0)
        if particles_obj.sink_type is not None:
            num_sinks = particles_obj.n_by_type[particles_obj.sink_type-1]
    return num_sinks
# ==================== end: read_num_sinks ====================

# ===================== read_sink_masses ======================
def read_sink_masses(flash_file):
    sink_masses = None
    num_sinks = read_num_sinks(flash_file)
    if num_sinks == 0: return
    # movie file
    if is_movie_file(flash_file):
        sink_masses = hdfio.read(flash_file, 'particlemasses')
    # flash particle file
    if is_part_file(flash_file) or is_plt_file(flash_file) or is_chk_file(flash_file):
        particles_obj = Particles(flash_file, verbose=0)
        if particles_obj.sink_type is not None:
            sink_masses = particles_obj.read("mass", type=particles_obj.sink_type)
    return sink_masses
# ================== end: read_sink_masses ====================

# ================= get_min ==================
def get_min(flash_file, dset):
    hdf = h5py.File(flash_file, "r")
    minimum = hdf[dset].attrs["minimum"]
    hdf.close()
    n = len(minimum)
    if n > 1:
        print("Found "+str(n)+" minima; picking one of them...", warn=1)
    return float(minimum.min())

# ================= get_max ==================
def get_max(flash_file, dset):
    hdf = h5py.File(flash_file, "r")
    maximum = hdf[dset].attrs["maximum"]
    hdf.close()
    n = len(maximum)
    if n > 1:
        print("Found "+str(n)+" maxima; picking one of them...", warn=1)
    return float(maximum.max())

# ================= get_time ==================
def get_time(flash_file):
    if is_movie_file(flash_file):
        time = hdfio.read(flash_file, 'time')[0]
    if is_part_file(flash_file) or is_plt_file(flash_file) or is_chk_file(flash_file):
        time = read_scalars(flash_file)['time']
    return time

# ================= get_dump ==================
def get_dump(flash_files, time=None, sink_mass=None, max_dens=None, low=None, high=None, quiet=True):
    # sort input list; note that this algorithm only works for sink masses, if sinks are growing in mass for increasing dumps
    flash_files = sorted(flash_files)
    # if not provided, extract low and high files
    if low is None: low = flash_files[0]
    if high is None: high = flash_files[-1]
    # set requested value (time or sink mass) and get low and high value info
    if time is not None: # search for dump with requested time
        req_val = time
        low_val = get_time(low)
        high_val = get_time(high)
    if sink_mass is not None: # search for dump with requested sink particle mass
        req_val = sink_mass
        low_val = read_sink_masses(low)
        if low_val is not None:
            low_val = sum(low_val)
        else:
            low_val = 0.0
        high_val = read_sink_masses(high)
        if high_val is not None:
            high_val = sum(high_val)
        else:
            high_val = 0.0
    if max_dens is not None: # search for dump with requested maximum density
        req_val = max_dens
        low_val = get_max(low, "dens")
        high_val = get_max(high, "dens")
    # if there are only 1 or 2 files (left)
    if (len(flash_files) == 1):
        return flash_files[0], low_val
    if (len(flash_files) == 2):
        if abs(low_val-req_val) < abs(high_val-req_val):
            return flash_files[0], low_val
        else:
            return flash_files[1], high_val
    # get middle info
    mid_index = len(flash_files)//2
    mid = flash_files[mid_index]
    if time is not None:
        mid_val = get_time(mid)
    if sink_mass is not None:
        mid_val = read_sink_masses(mid)
        if mid_val is not None:
            mid_val = sum(mid_val)
        else:
            mid_val = 0.0
    if max_dens is not None:
        mid_val = get_max(mid, "dens")
    if not quiet:
        if time is not None:
            print('--- Searching for time value: '+str(req_val)+' ---')
        if sink_mass is not None:
            print('--- Searching for sink mass value: '+str(req_val)+' ---')
        if max_dens is not None:
            print('--- Searching for maximum density value: '+str(req_val)+' ---')
        print('low  (dump, value): '+low+", "+str(low_val))
        print('mid  (dump, value): '+mid+", "+str(mid_val))
        print('high (dump, value): '+high+", "+str(high_val))
    # if element is smaller than mid, then it can only be present in left sub-array
    if req_val <= mid_val:
        flash_files = flash_files[:mid_index+1]
        if time is not None:
            return get_dump(flash_files, time=req_val,      low=flash_files[0], high=flash_files[-1])
        if sink_mass is not None:
            return get_dump(flash_files, sink_mass=req_val, low=flash_files[0], high=flash_files[-1])
        if max_dens is not None:
            return get_dump(flash_files, max_dens=req_val,  low=flash_files[0], high=flash_files[-1])
    # else the element can only be present in right sub-array
    else:
        flash_files = flash_files[mid_index:]
        if time is not None:
            return get_dump(flash_files, time=req_val,      low=flash_files[0], high=flash_files[-1])
        if sink_mass is not None:
            return get_dump(flash_files, sink_mass=req_val, low=flash_files[0], high=flash_files[-1])
        if max_dens is not None:
            return get_dump(flash_files, max_dens=req_val,  low=flash_files[0], high=flash_files[-1])


# ======================= get_sim_info =======================
def get_sim_info(filename):
    if not (is_plt_file(filename) or is_chk_file(filename)):
        print('Needs either plt or chk file; returning...')
        return
    # basic information
    print("--- Basic information ---", color="green")
    scalars = read_scalars(filename)
    rtparams = read_runtime_parameters(filename)
    print('time = ', cfp.eform(scalars['time']))
    redshift = scalars['redshift']
    a = 1 / (1 + redshift)
    if redshift > 0: print('redshift = ', cfp.eform(redshift))
    print('dt = ', cfp.eform(scalars['dt']))
    if 'cfl' in rtparams: print('cfl = ', rtparams['cfl'])
    if 'interpol_order' in rtparams: print('interpol_order = ', rtparams['interpol_order'])
    # grid information
    gg = FlashGG(filename)
    print("--- Grid information ---", color="green")
    print("dimensionality of grid =", gg.Ndim)
    print("grid type =", gg.GridType)
    gg.GetRefinementInfo()
    print("number of blocks (total, leaf) = "+str(gg.NumBlocks)+", "+str(gg.NumLeafBlocks))
    print("number of cells per block =", gg.NB)
    print("base grid resolution =", gg.NBaseGrid)
    print("maximum effective resolution =", gg.NMax)
    D = gg.D[gg.LeafBlocks]
    print('min/max cell size = '+cfp.eform(D.min())+", "+cfp.eform(D.max()))
    if redshift > 0:
        print('min/max cell size (proper) = '+cfp.eform(D.min()*a)+", "+cfp.eform(D.max()*a))
    print('domain bounds =', gg.domain_bounds.tolist())
    if redshift > 0:
        print('domain bounds (proper) =')
        print((gg.domain_bounds*a).tolist())
    # particle information
    part = Particles(filename)
    if part.n > 0:
        print("--- Particle information ---", color="green")
        part.print_info()
        sink_masses = read_sink_masses(filename)
        if sink_masses is not None:
            sink_masses = np.sort(sink_masses)
            print("Sink particle masses: total, min, max, median, mean, std =",
                cfp.round([np.sum(sink_masses), sink_masses.min(), sink_masses.max(), np.median(sink_masses), sink_masses.mean(), sink_masses.std()], 3, True))
# ==================== end: get_sim_info  =====================


# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser(description='FLASH library.')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand', description='valid subcommands', help='additional help', required=True)
    # sub parser for 'dumpfind' sub-command
    parser_dumpfind = subparsers.add_parser('dumpfind')
    parser_dumpfind.add_argument("inputfiles", nargs='+', type=argparse.FileType('r'), help="Input data file(s) to process")
    parser_dumpfind.add_argument("-t", "--time", dest='requested_time', type=float, help="Return filename based on requested time")
    parser_dumpfind.add_argument("-s", "--sink_mass", dest='requested_sink_mass', type=float, help="Return filename based on requested total sink mass")
    parser_dumpfind.add_argument("-m", "--max_dens", dest='requested_max_dens', type=float, help="Return filename based on requested maximum density")
    # sub parser for 'datfile' sub-command
    parser_datfile = subparsers.add_parser('datfile')
    parser_datfile.add_argument("inputfile", type=argparse.FileType('r'), help="Time evolution file to clean")
    parser_datfile.add_argument("-clean", action='store_true', default=False, help="Write a cleaned datfile (with old data after later restarts removed)")
    parser_datfile.add_argument("-moments_col", help="Get statistical moments of input column")
    parser_datfile.add_argument("-time_range", nargs=2, type=float, default=[-1e99, +1e99], help="limit to time range")
    parser_datfile.add_argument("-plot_col", help="Plot time evolution of input column")
    # sub parser for 'logfile' sub-command
    parser_logfile = subparsers.add_parser('logfile')
    parser_logfile.add_argument("inputfile", type=argparse.FileType('r'), help="Log file to parse")
    parser_logfile.add_argument("--run", "-r", type=int, default=-1, help="The run number for with information is requested (default: last run)")
    parser_logfile.add_argument("--plot", "-p", action='store_true', default=False, help="Plot timestep performance")
    # sub parser for 'siminfo' sub-command
    parser_siminfo = subparsers.add_parser('siminfo')
    parser_siminfo.add_argument("inputfiles", nargs='+', type=argparse.FileType('r'), help="Input data file(s) to process")
    # now parse arguments
    args = parser.parse_args()

    # ===================== handle dumpfind case =====================
    if args.subcommand == 'dumpfind':
        # searching for time and sink mass at the same time is not allowed
        if (args.requested_time is not None) and (args.requested_sink_mass is not None):
            print('Error: must not request both time and sink_mass in one call. Exiting...')
            exit()
        # sort input files
        inputfiles = sorted([x.name for x in list(args.inputfiles)])
        # find dump file
        filename_found, value_found = get_dump(inputfiles, time=args.requested_time, sink_mass=args.requested_sink_mass, max_dens=args.requested_max_dens)
        if filename_found is not None:
            if args.requested_time is not None:
                print("Found requested time "+str(args.requested_time)+" in file '"+filename_found+"' (exact time in file: "+str(value_found)+")")
            if args.requested_sink_mass is not None:
                print("Found requested total sink mass "+str(args.requested_sink_mass)+" in file '"+filename_found+"' (exact total sink mass in file: "+str(value_found)+")")
                # print number of sinks and sink masses
                print(filename_found+': number of sink particles = ', read_num_sinks(filename_found), '; mass of sink particles = ', read_sink_masses(filename_found))
            if args.requested_max_dens is not None:
                print("Found requested maximum density "+str(args.requested_max_dens)+" in file '"+filename_found+"' (exact maximum density in file: "+str(value_found)+")")
        else:
            print('No matching file found for request. Exiting...')

    # ===================== datfile case =====================
    if args.subcommand == 'datfile':
        # create new datfile class instance
        datfile_obj = datfile(args.inputfile.name)
        # clean if requested
        if args.clean: datfile_obj.write_cleaned()
        # get moments for column
        if args.moments_col:
            moments = datfile_obj.get_moments(args.moments_col, xs=args.time_range[0], xe=args.time_range[1])
            print("statistical moments for column '"+args.moments_col+"': ", moments)
        # plot column
        if args.plot_col: datfile_obj.plot_column(args.plot_col)

    # ===================== logfile case =====================
    if args.subcommand == 'logfile':
        # create new logfile class instance
        logfile_obj = logfile(args.inputfile.name, run=args.run, plot=args.plot)

    # ===================== datfile case =====================
    if args.subcommand == 'siminfo':
        # sort input files
        inputfiles = sorted([x.name for x in list(args.inputfiles)])
        for filename in inputfiles:
            get_sim_info(filename)

# =============================================================
