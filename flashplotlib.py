#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2019-2025

import os, sys
import numpy as np
import argparse
import timeit
import tempfile
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
from matplotlib import rcParams
try:
    import cmasher as cmr
except:
    pass
import flashlib as fl
import cfpack.constants as const
import cfpack as cfp
from cfpack import print, hdfio, stop
import glob

# =============== flashplotlib class ===============
class flashplotlib:

    # ============= __init__ =============
    def __init__(self):
        self.verbose = 1
        self.direction = 'z'
        self.bounds = None
        self.plotlabel = ''
        self.textlabel = [self.text()]
        self.labels_inside = None
        self.unitsystem = 'CGS'
        self.redshift = None # for cosmology
        self.convert2proper = False # convert co-moving quantities to proper quantities (for cosmology)
        self.time = None
        self.time_unit = None
        self.time_scale = None
        self.time_format = None
        self.time_transform = 'q'
        self.axes_label = [None, None, None]
        self.axes_unit = [None, None, None]
        self.axes_format = [None, None, None]
        self.axes_transform = ['q', 'q', 'q']
        self.data_transform = 'q'
        self.log = True
        self.facecolor = None
        self.colorbar = True # can also be 0, "none"/None, "false"/False to turn off, or "only", "aspect?", "panels?" for special options
        self.cmap = 'afmhot' # 'magma'
        self.cmap_label = None
        self.cmap_format = None
        self.vmin = None
        self.vmax = None
        self.pixel = None
        self.shift = None
        self.shift_periodic = None
        self.boundary = 'isolated' # can also be 'periodic'
        self.gauss_smooth_fwhm = [None, None]
        self.scale_sinks_by_mass = False
        self.show_sinks = False
        self.show_tracers = None
        self.particle_color = None
        self.particle_size_factor = 1.0
        self.particle_mark_tags = ["particle_mark_tags.h5", 'blue', 2.0] # [HDF5-file-with-tags, color, particle-size-factor]
        self.vec = False
        self.vec_transform = 'q'
        self.vec_var = 'vel'
        self.vec_unit = None
        self.vec_scale = None
        self.vec_scale_factor = 1.0
        self.vec_n = [None, None]
        self.vec_color = 'black'
        self.vec_key = False
        self.stream = False
        self.stream_transform = 'q'
        self.stream_var = 'mag'
        self.stream_vmin = None
        self.stream_thick = 1.0
        self.stream_n = [None, None]
        self.stream_color = 'blue'
        self.show_blocks = None
        self.show_grid = None
        self.bb = None # block bounding boxes
        self.rl = None # block refinement levels
        self.nb = None # number of cells per block
        self.outtype = ['screen']
        self.outname = None
        # init some matplotlib parameters
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{bm}'
        rcParams['xtick.top'] = True
        rcParams['xtick.direction'] = 'in'
        rcParams['xtick.minor.visible'] = True
        rcParams['ytick.right'] = True
        rcParams['ytick.direction'] = 'in'
        rcParams['ytick.minor.visible'] = True
        rcParams['axes.linewidth'] = 0.5
        rcParams['xtick.major.width'] = 0.5
        rcParams['xtick.minor.width'] = 0.5
        rcParams['ytick.major.width'] = 0.5
        rcParams['ytick.minor.width'] = 0.5
        rcParams['font.size'] = 10
    # ============= end: __int__ =============

    # ============= particles class =============
    class particles:
        def __init__(self):
            self.n = 0
            self.posx = None
            self.posy = None
            self.posz = None
            self.mass = None
            self.tag  = None
            self.type = None
            self.radius = None
    # ============= end: particles ==============

    # ============= text class =============
    class text:
        def __init__(self):
            self.text = ""
            self.pos = [0.0, 0.0]
            self.align = 'left'
            self.alpha = 1.0
            self.style = 0
            self.color = 'white'
            self.fontsize = 1.0
            self.fontweight = 1.0
        # function for plotting the text with different styles
        def plot(self):
            if self.text == "": return
            ax = plt.gca()
            # compute some text propteries to optimise text position
            scaled_fontsize = self.fontsize*rcParams['font.size']
            fontprops = mpl.font_manager.FontProperties(size=scaled_fontsize)
            textbb = mpl.textpath.TextPath((0,0), self.text, prop=fontprops).get_extents()
            xoffset = 0.0
            if self.align == 'right': xoffset = (textbb.width / scaled_fontsize)**0.8 - 2.0
            yoffset = -0.05*(textbb.height / scaled_fontsize + 1.0)**6 + (textbb.height / scaled_fontsize + 1.0)**2 - 3.0
            # plot text with specified style
            if self.style == 0 or self.style == None: # just normal text
                t = ax.text(self.pos[0], self.pos[1], self.text, transform=ax.transAxes, horizontalalignment=self.align, verticalalignment='center',
                            bbox=None, alpha=self.alpha, fontproperties=fontprops)
                t.set_path_effects([path_effects.PathPatchEffect(offset=(xoffset+0.0,yoffset+0.0),
                                    linewidth=0.20*self.fontweight, edgecolor=self.color, facecolor=self.color, alpha=self.alpha)])
            if self.style == 1: # text with grey shadow
                t = ax.text(self.pos[0], self.pos[1], self.text, transform=ax.transAxes, horizontalalignment=self.align, verticalalignment='center',
                            bbox=None, alpha=0.0, fontproperties=fontprops)
                t.set_path_effects([path_effects.PathPatchEffect(offset=(xoffset+0.4,yoffset+0.6),
                                    linewidth=0.20*self.fontweight, edgecolor='grey', facecolor='darkgrey', alpha=0.8*self.alpha),
                                    path_effects.PathPatchEffect(offset=(xoffset+0.0,yoffset+1.0),
                                    linewidth=0.20*self.fontweight, edgecolor=self.color, facecolor=self.color, alpha=self.alpha)])
            if self.style == 2: # text with grey shadow and transpararent background box
                bbox = dict(boxstyle='round', facecolor='white', edgecolor='white', alpha=0.67, pad=0.3)
                t = ax.text(self.pos[0], self.pos[1], self.text, transform=ax.transAxes, horizontalalignment=self.align, verticalalignment='center',
                            bbox=bbox, alpha=0.0, fontproperties=fontprops)
                t.set_path_effects([path_effects.PathPatchEffect(offset=(xoffset+0.4,yoffset+0.6),
                                    linewidth=0.20*self.fontweight, edgecolor='grey', facecolor='grey', alpha=0.5*self.alpha),
                                    path_effects.PathPatchEffect(offset=(xoffset+0.0,yoffset+1.0),
                                    linewidth=0.20*self.fontweight, edgecolor='black', facecolor='black', alpha=self.alpha)])

    # ============= end: text ==============

    # ============= transform_quantity =============
    def transform_quantity(self, q, string_expression):
        # evaluate string expression containing 'q' (e.g. "2*q-0.5")
        return eval(string_expression)
    # ============= end: transform_quantity =============

    # ============= get_map_var =============
    def get_map_var(self, filename=None, datasetname=None):
        if filename is None:
            print("Must specify filename.", error=True)
        if datasetname is None:
            print("Must specify datasetname.", error=True)
        # read 2D dataset
        if self.verbose > 1: print("Now reading data...")
        # dataset is derived or is an expression
        if type(datasetname) == list:
            if datasetname[0].find('derived:oadv') == 0:
                oadv = hdfio.read(filename[1], datasetname[1])
                dens = hdfio.read(filename[2], datasetname[2])
                map_var = oadv * dens
                self.datasetname = 'derived_oadv'
                self.filename = filename[-1] # pass the last one
        else:
            map_var = hdfio.read(filename, datasetname)
            self.filename = filename
            self.datasetname = datasetname
        return self.transform_quantity(map_var, self.data_transform)
    # ============= end: get_map_var =============

    # ============= prep_map =============
    def prep_map(self, filename=None, datasetname=None):
        # map variable
        if self.verbose > 1: print("Getting map variable...")
        self.data = self.get_map_var(filename=filename, datasetname=datasetname)
        # time
        if self.time is None:
            # format and scale the time
            def format_time(time_in_sec):
                if self.time_scale == 0: return '' # if we don't want to display time at all
                tsign = np.sign(time_in_sec)
                t = abs(time_in_sec)
                if self.time_transform != 'q':
                    tsign = 1
                    t = self.transform_quantity(time_in_sec, self.time_transform)
                if self.time_unit is None: # automatic
                    if t <= 1e-10: return r"$0$" # return a "0" string if time is really small
                    if 1e-10 < t <= 1e-6: self.time_unit = 'ns'
                    elif 1e-6 < t <= 1e7: self.time_unit = 's'
                    elif 1e7 < t <= 1e11: self.time_unit = 'yr'
                    elif 1e11 < t: self.time_unit = 'Myr'
                    else: self.time_unit = 's'
                if self.time_scale is None: # automatic
                    if self.time_unit.lower() == 'ns': self.time_scale = 1e-9
                    elif self.time_unit.lower() == 'yr': self.time_scale = const.year
                    elif self.time_unit.lower() == 'myr': self.time_scale = 1e6*const.year
                    else: self.time_scale = 1.0
                if self.time_transform != 'q': self.time_scale = 1.0
                if self.time_format is None: time_str = str(cfp.round(t/self.time_scale*tsign,3))
                else:
                    time_str = "{:"+self.time_format+"}"
                    time_str = time_str.format(t/self.time_scale*tsign)
                return r"$"+time_str+r"\,\mathrm{"+self.time_unit+"}$"
            # read time
            self.time = hdfio.read(self.filename, "time")[0]
            self.time_str = format_time(self.time)
        # redshift
        if self.redshift is None:
            def format_redshift(redshift):
                return r"$"+str(cfp.round(redshift,3))+r"$"
            dsetnames = hdfio.get_dataset_names(self.filename)
            if 'redshift' in dsetnames:
                self.redshift = hdfio.read(self.filename, "redshift")[0]
                self.redshift_str = format_redshift(self.redshift)
        # convert map quantity from co-moving to proper
        if self.convert2proper:
            self.data = cfp.comov2proper(self.data, self.redshift, qtype=self.datasetname)
        # get the slice/projection direction
        self.direction = hdfio.read(self.filename, "direction")[0].decode()
        # reorganise the bounds array depending on the view direction
        self.bounds = hdfio.read(self.filename, "minmax_xyz")
        bounds = np.array(self.bounds)
        if self.direction == 'z':
            if self.axes_label[0] in [None, "None"]: self.axes_label[0] = r'$x$'
            if self.axes_label[1] in [None, "None"]: self.axes_label[1] = r'$y$'
            if self.axes_label[2] in [None, "None"]: self.axes_label[2] = r'$z$'
            self.vx_name = self.vec_var + 'x'
            self.vy_name = self.vec_var + 'y'
            self.bx_name = self.stream_var + 'x'
            self.by_name = self.stream_var + 'y'
        if self.direction == 'y':
            if self.axes_label[0] in [None, "None"]: self.axes_label[0] = r'$x$'
            if self.axes_label[1] in [None, "None"]: self.axes_label[1] = r'$z$'
            if self.axes_label[2] in [None, "None"]: self.axes_label[2] = r'$y$'
            self.bounds[1] = bounds[2]
            self.bounds[2] = bounds[1]
            self.vx_name = self.vec_var + 'x'
            self.vy_name = self.vec_var + 'z'
            self.bx_name = self.stream_var + 'x'
            self.by_name = self.stream_var + 'z'
        if self.direction == 'x':
            if self.axes_label[0] in [None, "None"]: self.axes_label[0] = r'$y$'
            if self.axes_label[1] in [None, "None"]: self.axes_label[1] = r'$z$'
            if self.axes_label[2] in [None, "None"]: self.axes_label[2] = r'$x$'
            self.bounds[0] = bounds[1]
            self.bounds[1] = bounds[2]
            self.bounds[2] = bounds[0]
            self.vx_name = self.vec_var + 'y'
            self.vy_name = self.vec_var + 'z'
            self.bx_name = self.stream_var + 'y'
            self.by_name = self.stream_var + 'z'
        if self.convert2proper:
            for dir in range(3):
                self.bounds[dir] = cfp.comov2proper(self.bounds[dir], self.redshift, qtype='length')
        # box size
        box_size = np.array(self.bounds[:,1]) - np.array(self.bounds[:,0])
        # index used to extract dataset name (without '_slice' or '_proj')
        dsetname_index = max(self.datasetname.find('_proj'), self.datasetname.find('_slice'))
        # read vector data
        if self.vec:
            dname = self.vx_name+self.datasetname[dsetname_index:]
            fname = self.filename.replace(self.datasetname, dname)
            self.data_vx = self.transform_quantity(hdfio.read(fname, dname), self.vec_transform)
            dname = self.vy_name+self.datasetname[dsetname_index:]
            fname = self.filename.replace(self.datasetname, dname)
            self.data_vy = self.transform_quantity(hdfio.read(fname, dname), self.vec_transform)
        # read streamline data
        if self.stream:
            dname = self.bx_name+self.datasetname[dsetname_index:]
            fname = self.filename.replace(self.datasetname, dname)
            self.data_bx = self.transform_quantity(hdfio.read(fname, dname), self.stream_transform)
            dname = self.by_name+self.datasetname[dsetname_index:]
            fname = self.filename.replace(self.datasetname, dname)
            self.data_by = self.transform_quantity(hdfio.read(fname, dname), self.stream_transform)
        if self.verbose > 1: print("...done.")
        # change resolution to pixel
        if self.pixel is not None:
            dims_orig = np.shape(self.data)
            if dims_orig[0] != self.pixel[0] or dims_orig[1] != self.pixel[1]:
                if self.verbose: print("Interpolating to pixel resolution "+str(self.pixel[0])+" "+str(self.pixel[0]))
                self.data = cfp.congrid(self.data, [self.pixel[0],self.pixel[1]])
        # set dimensions
        self.dims = np.shape(self.data)
        # set data range
        if self.vmin is None: self.vmin = np.nanmin(self.data)
        if self.vmax is None: self.vmax = np.nanmax(self.data)
        # scale axes lengths by axes units
        if self.unitsystem == "CGS": self.length_factor = np.array([1.0, 1.0, 1.0])
        if self.unitsystem == "MKS": self.length_factor = np.array([1e-2, 1e-2, 1e-2]) # convert to m to cm
        for dir in range(3):
            if self.axes_unit[dir] in [None, "None"]:
                if box_size[dir] < 1e2: self.axes_unit[dir] = 'cm'
                elif box_size[dir] >= 1e2 and box_size[dir] < 1e5: self.axes_unit[dir] = 'm'
                elif box_size[dir] >= 1e5 and box_size[dir] < const.r_sol: self.axes_unit[dir] = 'km'
                elif box_size[dir] >= const.r_sol and box_size[dir] < const.au: self.axes_unit[dir] = 'rsol'
                elif box_size[dir] >= const.au and box_size[dir] < 1e4*const.au: self.axes_unit[dir] = 'AU'
                elif box_size[dir] >= 1e4*const.au and box_size[dir] < 1e3*const.pc: self.axes_unit[dir] = 'pc'
                elif box_size[dir] >= 1e3*const.pc and box_size[dir] < 1e6*const.pc: self.axes_unit[dir] = 'kpc'
                elif box_size[dir] >= 1e6*const.pc: self.axes_unit[dir] = 'Mpc'
            if self.axes_unit[dir].lower() == 'm': self.length_factor[dir] *= 1e2
            if self.axes_unit[dir].lower() == 'km': self.length_factor[dir] *= 1e5
            if self.axes_unit[dir].lower() == 'rsol': self.length_factor[dir] *= const.r_sol; self.axes_unit[dir] = r'R_\odot'
            if self.axes_unit[dir].lower() == 'au': self.length_factor[dir] *= const.au
            if self.axes_unit[dir].lower() == 'pc': self.length_factor[dir] *= const.pc
            if self.axes_unit[dir].lower() == 'kpc': self.length_factor[dir] *= 1e3*const.pc
            if self.axes_unit[dir].lower() == 'mpc': self.length_factor[dir] *= 1e6*const.pc
            if self.axes_unit[dir] == '1' or self.axes_unit[dir].strip() == '':
                self.axes_unit[dir] = '' # dimensionless units
            else: # add units in brackets
                if dir < 2: self.axes_unit[dir] = r'$\;(\mathrm{'+self.axes_unit[dir]+r'})$' # x, y axes
                else: self.axes_unit[dir] = r'$\;\mathrm{'+self.axes_unit[dir]+r'}$' # z axes (projection length)
            if self.axes_transform[dir] != 'q':
                self.length_factor[dir] = 1.0
        # add units to axes labels
        self.axes_label[0] += self.axes_unit[0]
        self.axes_label[1] += self.axes_unit[1]
        # set scaled bounds
        for dir in range(3):
            self.bounds[dir] /= self.length_factor[dir]
            self.bounds[dir] = self.transform_quantity(self.bounds[dir], self.axes_transform[dir])
        box_size[2] = np.array(self.bounds[2,1]) - np.array(self.bounds[2,0])
        # scale block bounding box (if present)
        if self.bb is not None:
            if self.direction == 'z': ind = (0,1,2)
            if self.direction == 'y': ind = (0,2,1)
            if self.direction == 'x': ind = (1,2,0)
            for dir in range(3):
                self.bb[:,ind[dir],:] /= self.length_factor[dir]
                self.bb[:,ind[dir],:] = self.transform_quantity(self.bb[:,ind[dir],:], self.axes_transform[dir])
                if self.convert2proper:
                    self.bb[:,ind[dir],:] = cfp.comov2proper(self.bb[:,ind[dir],:], self.redshift, qtype='length')
        # set colour map label
        if self.cmap_label is None and self.data_transform == 'q':
            self.cmap_label = r'$'+self.datasetname+r'$'
            if self.datasetname.find('dens') != -1:
                self.cmap_label = r'$\mathrm{density}$ $\left(\mathrm{g}\,\mathrm{cm}^{-3}\right)$'
            if self.datasetname.find('pden') != -1:
                self.cmap_label = r'$\mathrm{particle\;density}$ $\left(\mathrm{g}\,\mathrm{cm}^{-3}\right)$'
            if self.datasetname.find('temp') != -1:
                self.cmap_label = r'$\mathrm{temperature}$ $\left(\mathrm{K}\right)$'
            # velocity
            if self.datasetname.find('velx') != -1:
                self.cmap_label = r'$x\mathrm{-velocity}$ $\left(\mathrm{cm}\,\mathrm{s}^{-1}\right)$'
            if self.datasetname.find('vely') != -1:
                self.cmap_label = r'$y\mathrm{-velocity}$ $\left(\mathrm{cm}\,\mathrm{s}^{-1}\right)$'
            if self.datasetname.find('velz') != -1:
                self.cmap_label = r'$z\mathrm{-velocity}$ $\left(\mathrm{cm}\,\mathrm{s}^{-1}\right)$'
            # vorticity
            if self.datasetname.find('vorticity_x') != -1:
                self.cmap_label = r'$x\mathrm{-vorticity}$ $\left(\mathrm{s}^{-1}\right)$'
            if self.datasetname.find('vorticity_y') != -1:
                self.cmap_label = r'$y\mathrm{-vorticity}$ $\left(\mathrm{s}^{-1}\right)$'
            if self.datasetname.find('vorticity_z') != -1:
                self.cmap_label = r'$z\mathrm{-vorticity}$ $\left(\mathrm{s}^{-1}\right)$'
            # divergence of velocity
            if self.datasetname.find('divv') != -1:
                self.cmap_label = r'$\nabla\cdot\mathbf{v}$ $\left(\mathrm{s}^{-1}\right)$'
            # derived:oadv
            if self.datasetname.find('outflow_dens') == 0 or self.datasetname.find('derived_oadv') == 0:
                self.cmap_label = r'$\mathrm{outflow\;gas\;density}$ $\left(\mathrm{g}\,\mathrm{cm}^{-3}\right)$'
            # oadv
            if self.datasetname.find('oadv') == 0:
                self.cmap_label = r'$\mathrm{outflow\;tracer}$'
            # in case of projection
            if self.datasetname.find('proj') != -1:
                self.cmap_label += r'$\;\;(\mathrm{projection\;length\;}$'+self.axes_label[2]+r'\,=\,$'+str(cfp.round(box_size[2],2))+r'$'+self.axes_unit[2]+r')'
        # size of domain
        size_x = self.bounds[0][1]-self.bounds[0][0]
        size_y = self.bounds[1][1]-self.bounds[1][0]
        # periodic shift (roll)
        roll_dx = 0.0
        roll_dy = 0.0
        if self.shift_periodic is not None:
            data_tmp = cfp.congrid(self.data, [4096,4096])
            nd = np.shape(data_tmp)
            roll_nx = int(np.round(self.shift_periodic[0]/size_x*nd[0]))
            roll_dx = roll_nx / nd[0] * size_x
            roll_ny = int(np.round(self.shift_periodic[1]/size_y*nd[1]))
            roll_dy = roll_ny / nd[1] * size_y
            data_tmp = np.roll(data_tmp, roll_nx, axis=1)
            data_tmp = np.roll(data_tmp, roll_ny, axis=0)
            self.bounds[0][0] -= roll_dx
            self.bounds[0][1] -= roll_dx
            self.bounds[1][0] -= roll_dy
            self.bounds[1][1] -= roll_dy
            if self.verbose: print("applying periodic shift (roll) with dx,dy = "+str(roll_dx)+" "+str(roll_dy))
            self.data = cfp.congrid(data_tmp, (self.dims[0],self.dims[1]))
            if self.vec:
                # vx
                data_tmp = cfp.congrid(self.data_vx, [4096,4096])
                data_tmp = np.roll(data_tmp, roll_nx, axis=1)
                data_tmp = np.roll(data_tmp, roll_ny, axis=0)
                self.data_vx = cfp.congrid(data_tmp, (self.dims[0],self.dims[1]))
                # vy
                data_tmp = cfp.congrid(self.data_vy, [4096,4096])
                data_tmp = np.roll(data_tmp, roll_nx, axis=1)
                data_tmp = np.roll(data_tmp, roll_ny, axis=0)
                self.data_vy = cfp.congrid(data_tmp, (self.dims[0],self.dims[1]))
            if self.stream:
                # bx
                data_tmp = cfp.congrid(self.data_bx, [4096,4096])
                data_tmp = np.roll(data_tmp, roll_nx, axis=1)
                data_tmp = np.roll(data_tmp, roll_ny, axis=0)
                self.data_bx = cfp.congrid(data_tmp, (self.dims[0],self.dims[1]))
                # by
                data_tmp = cfp.congrid(self.data_by, [4096,4096])
                data_tmp = np.roll(data_tmp, roll_nx, axis=1)
                data_tmp = np.roll(data_tmp, roll_ny, axis=0)
                self.data_by = cfp.congrid(data_tmp, (self.dims[0],self.dims[1]))
        # read particles information
        if self.verbose > 1: print("Reading particles...", highlight=True)
        self.sinks = self.particles() # create new particles class object
        self.tracers = self.particles() # create new particles class object
        n_total = 0
        names = hdfio.get_dataset_names(self.filename)
        if "numpart_proj" in names:
            n_total = hdfio.read(self.filename, "numpart_proj")[0]
        if n_total > 0:
            p_type = np.array(hdfio.read(self.filename, "p_type_proj")).astype(int)
            ind_sinks = np.where(p_type == 0)
            ind_tracers = np.where(p_type == 1)
            self.sinks.n = ind_sinks[0].size
            if self.verbose: print("Number of sink particles: ", self.sinks.n)
            self.tracers.n = ind_tracers[0].size
            if self.verbose: print("Number of tracer particles: ", self.tracers.n)
            if (self.sinks.n + self.tracers.n != n_total):
                if self.verbose: print('WARNING. sum of sink and tracer particles does not match total number of particles in file '+self.filename)
            pos = np.array(hdfio.read(self.filename, "p_pos_proj"))
            tag  = np.array(hdfio.read(self.filename, "p_tag_proj")).astype(int)
            # sinks
            self.sinks.posx   = self.transform_quantity(pos[0][ind_sinks]/self.length_factor[0], self.axes_transform[0])
            self.sinks.posy   = self.transform_quantity(pos[1][ind_sinks]/self.length_factor[1], self.axes_transform[1])
            self.sinks.posz   = self.transform_quantity(pos[2][ind_sinks]/self.length_factor[2], self.axes_transform[2])
            self.sinks.mass   = np.array(hdfio.read(self.filename, "p_mass_proj"))[ind_sinks]
            self.sinks.tag    = tag[ind_sinks]
            # hack to scale sink radius in case of axes_transform
            length_factor_on_axes_transform = 1/(self.transform_quantity(1.0, self.axes_transform[0])-self.transform_quantity(0.0, self.axes_transform[0]))
            self.sinks.radius = hdfio.read(self.filename, "r_accretion")[0] / self.length_factor[0] / length_factor_on_axes_transform
            if self.convert2proper:
                self.sinks.posx = cfp.comov2proper(self.sinks.posx, self.redshift, qtype='length')
                self.sinks.posy = cfp.comov2proper(self.sinks.posy, self.redshift, qtype='length')
                self.sinks.posz = cfp.comov2proper(self.sinks.posz, self.redshift, qtype='length')
                self.sinks.radius = cfp.comov2proper(self.sinks.radius, self.redshift, qtype='length')
            if self.verbose: print("Total mass of sink particles: "+str(np.sum(self.sinks.mass)/const.m_sol)+" M_sol")
            # tracers
            self.tracers.posx = self.transform_quantity(pos[0][ind_tracers]/self.length_factor[0], self.axes_transform[0])
            self.tracers.posy = self.transform_quantity(pos[1][ind_tracers]/self.length_factor[1], self.axes_transform[1])
            self.tracers.posz = self.transform_quantity(pos[2][ind_tracers]/self.length_factor[2], self.axes_transform[2])
            self.tracers.tag  = tag[ind_tracers]
            # indices of marked particle tags
            self.tracers.marked_tags_ind = None
            if os.path.isfile(self.particle_mark_tags[0]): # test for existence of particle tags file for marking particles by tags
                dsetnames = hdfio.get_dataset_names(self.particle_mark_tags[0]) # get all the dataset names in the HDF5 file
                # read tags to be marked (assume there is only a single dataset, in dsetnames[0])
                self.tracers.marked_tags = np.array(hdfio.read(self.particle_mark_tags[0], dsetnames[0])).astype(int)
                if len(self.tracers.marked_tags) > 0: # get indices of marked tags into self.tracers.tag
                    self.tracers.marked_tags_ind = np.intersect1d(self.tracers.tag, self.tracers.marked_tags, return_indices=True)[1]
            # cosmology convert to proper from comoving
            if self.convert2proper:
                self.tracers.posx = cfp.comov2proper(self.tracers.posx, self.redshift, qtype='length')
                self.tracers.posy = cfp.comov2proper(self.tracers.posy, self.redshift, qtype='length')
                self.tracers.posz = cfp.comov2proper(self.tracers.posz, self.redshift, qtype='length')
            if self.shift_periodic is not None:
                # sinks
                ind = self.sinks.posx < self.bounds[0][0]
                self.sinks.posx[ind] += size_x
                ind = self.sinks.posx > self.bounds[0][1]
                self.sinks.posx[ind] -= size_x
                ind = self.sinks.posy < self.bounds[1][0]
                self.sinks.posy[ind] += size_y
                ind = self.sinks.posy > self.bounds[1][1]
                self.sinks.posy[ind] -= size_y
                # tracers
                ind = self.tracers.posx < self.bounds[0][0]
                self.tracers.posx[ind] += size_x
                ind = self.tracers.posx > self.bounds[0][1]
                self.tracers.posx[ind] -= size_x
                ind = self.tracers.posy < self.bounds[1][0]
                self.tracers.posy[ind] += size_y
                ind = self.tracers.posy > self.bounds[1][1]
                self.tracers.posy[ind] -= size_y
            if self.shift is not None:
                # sinks
                self.sinks.posx += self.shift[0]
                self.sinks.posy += self.shift[1]
                # tracers
                self.tracers.posx += self.shift[0]
                self.tracers.posy += self.shift[1]
        # shift coordinates
        if self.shift is not None:
            self.bounds[0] += self.shift[0]
            self.bounds[1] += self.shift[1]
        # apply Gaussian beam smoothing if requested
        if None not in self.gauss_smooth_fwhm:
            # convert FWHM given in physical coordinates FWHM into pixel units;
            fwhm_x = self.gauss_smooth_fwhm[0] / size_x * self.dims[0]
            fwhm_y = self.gauss_smooth_fwhm[1] / size_y * self.dims[1]
            mode = None
            if self.boundary == 'isolated': mode = 'reflect'
            if self.boundary == 'periodic': mode = 'wrap'
            if self.verbose: print("applying Gaussian smoothing with FWHM x,y = "+str(self.gauss_smooth_fwhm[0])+" "+str(self.gauss_smooth_fwhm[1])+\
                                    " ("+str(fwhm_x)+" "+str(fwhm_y)+" pixels), using "+self.boundary+" boundary conditions...")
            self.data = cfp.gauss_smooth(self.data, fwhm=[fwhm_x,fwhm_y], mode=mode)
            if self.vec:
                self.data_vx = cfp.gauss_smooth(self.data_vx, fwhm=[fwhm_x,fwhm_y], mode=mode)
                self.data_vy = cfp.gauss_smooth(self.data_vy, fwhm=[fwhm_x,fwhm_y], mode=mode)
            if self.stream:
                self.data_bx = cfp.gauss_smooth(self.data_bx, fwhm=[fwhm_x,fwhm_y], mode=mode)
                self.data_by = cfp.gauss_smooth(self.data_by, fwhm=[fwhm_x,fwhm_y], mode=mode)
        if self.verbose: print('min, max of data = '+str(np.nanmin(self.data))+", "+str(np.nanmax(self.data)))
    # ============= end: prep_map =============


    # ============= plot_map =============
    def plot_map(self, map_only=False, normalize=None, dpi=300):

        # function that shows or saves the plot window
        def show_or_save():
            globals()['end_time'] = timeit.default_timer()
            # save figure
            for outtype in self.outtype:
                if outtype == 'screen': continue
                pad_inches = 0.04
                if outtype in ["png", "jpg"]: pad_inches = 0.08
                if map_only: pad_inches = 0.0
                if self.outname is not None: outfilename = self.outname+"."+outtype
                else: outfilename = self.filename+"_"+self.datasetname+"."+outtype
                if self.verbose: print("saving '"+outfilename+"'...")
                plt.savefig(outfilename, format=outtype, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor='none')
                if self.verbose: print("'"+outfilename+"' written.", color="magenta")
            if 'screen' in self.outtype:
                if self.verbose: print("rendering for screen output...")
                plt.gcf().set_dpi(dpi)
                plt.tight_layout(pad=0.0)
                plt.show()
            plt.clf(); plt.cla(); plt.close() # clear figure after use

        # function that draws a colorbar
        def draw_colorbar(colorbar_only=False, norm=None):
            # set defaults
            aspect_default = 28
            aspect = aspect_default
            num_panels = 1
            horizontal_cb = False
            extend = "neither" # can be "both", "min", "max"
            # parse colorbar string
            for cbar_elem in self.colorbar:
                if type(cbar_elem) == str:
                    if cbar_elem.find(",") >= 0:
                        cbar_elem = cbar_elem.replace(" ", "")
                        split_str = cbar_elem.split(",")
                    else:
                        split_str = cbar_elem.split()
                    for cb_opt in split_str:
                        # aspect ratio option
                        aspect_index = cb_opt.find('aspect')
                        if aspect_index >= 0: aspect = float(cb_opt[aspect_index+len('aspect'):])
                        # panels option (to create a colorbar that spanning multiple panels)
                        panels_index = cb_opt.find('panels')
                        if panels_index >= 0: num_panels = float(cb_opt[panels_index+len('panels'):])
                        # vertical or horizontal
                        if cb_opt.find('horizontal') >= 0:
                            horizontal_cb = True
                        # extend
                        extend_index = cb_opt.find('extend')
                        if extend_index >= 0: extend = cb_opt[extend_index+len('extend'):]
            # in case we only produce a colorbar (no map)
            if colorbar_only:
                figsize_for_cb = np.copy(figsize)
                if horizontal_cb:
                    figsize_for_cb[0] *= num_panels
                else:
                    figsize_for_cb[1] *= num_panels
                fig.set_size_inches(figsize_for_cb)
                width = 0.025*aspect_default/aspect
                height = 0.9
                ypos = 0.05
                orientation = "vertical"
                if horizontal_cb: # flip width and height for horizontal colorbar
                    orientation = "horizontal"
                    width_saved = width
                    width = height
                    height = width_saved
                    ypos = 0.95
                cbax = fig.add_axes([0.05, ypos, width, height])
                cb = mpl.colorbar.ColorbarBase(cbax, cmap=self.cmap, norm=norm, orientation=orientation, extend=extend, label=self.cmap_label)
            else:
                cb = plt.colorbar(label=self.cmap_label, pad=0.01, aspect=aspect, extend=extend)
            # adjust final settings
            if not self.log: cb.ax.minorticks_on()
            cb.ax.yaxis.set_offset_position('left')
            if self.cmap_format is not None:
                if self.cmap_format not in ["None"]:
                    cb.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(self.cmap_format))
            return cb

        # see if we only produce the colorbar
        colorbar_only = False
        for cbar_elem in self.colorbar:
            if type(cbar_elem) == str:
                if cbar_elem.lower().find('only') >= 0:
                    colorbar_only = True

        # define how to normalise colours
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        if self.log:
            # fix negative or zero lower bound
            if self.vmin <= 0:
                # get the positive value of the number closest to zero
                linthresh = min([-self.data[self.data<0].max(), self.data[self.data>0].min()])
                linscale = 0.01
                norm = mpl.colors.SymLogNorm(vmin=self.vmin, vmax=self.vmax, linthresh=linthresh, linscale=linscale)
                # self.vmin = np.nanmin(self.data[self.data > 0])
            else:
                norm = mpl.colors.LogNorm(vmin=self.vmin, vmax=self.vmax)

        # create figure
        if "screen" not in self.outtype:
            plt.switch_backend('agg')
        else: # screen
            dpi /= 1.5 # set screen dpi
        figsize = np.array([6.5, 5.0])
        fig = plt.figure(figsize=figsize, dpi=dpi)

        if colorbar_only:
            cb = draw_colorbar(colorbar_only=True, norm=norm)
            show_or_save()
            return

        ax = plt.gca() # get current axes
        ax.set_aspect('equal')
        plt.minorticks_on()

        # set the coordinates of the bounding box
        extent = [self.bounds[0][0], self.bounds[0][1], self.bounds[1][0], self.bounds[1][1]]
        # normalize
        if normalize is not None:
            if normalize == 'mean':
                mean = np.mean(self.data)
                if self.verbose: print('normalizing to mean = ', mean)
                self.data /= mean
        # plot the map
        img = plt.imshow(self.data, cmap=self.cmap, origin='lower', interpolation='none', norm=norm, extent=extent)
        # add colorbar
        if not map_only:
            draw_cbar = True
            for cbar_elem in self.colorbar:
                if cbar_elem in [0, "0", False, "False", "false", None, "None", "none", "Off", "off"]:
                    draw_cbar = False
            if draw_cbar: cb = draw_colorbar()
        # add axis labels and ticks
        plt.xlabel(self.axes_label[0])
        if self.axes_format[0] is not None:
            if self.axes_format[0] not in ["None"]:
                ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(self.axes_format[0]))
        plt.ylabel(self.axes_label[1])
        if self.axes_format[1] is not None:
            if self.axes_format[1] not in ["None"]:
                ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(self.axes_format[1]))
        # move the exponential factor label a bit
        ax.get_yaxis().get_offset_text().set_x(-0.1)
        # show block bounding boxes and/or grid cells
        if self.show_blocks or self.show_grid:
            if self.bb is not None:
                if self.direction == 'z': ind = (0,1,2)
                if self.direction == 'y': ind = (0,2,1)
                if self.direction == 'x': ind = (1,2,0)
                for b in range(0,len(self.bb)): # loop over leaf blocks
                    # check if block inside projection range
                    overlap = True
                    for dim in range(3):
                        if self.bb[b,ind[dim],1] <= self.bounds[dim,0]:
                            overlap = False; break
                        if self.bb[b,ind[dim],0] >= self.bounds[dim,1]:
                            overlap = False; break
                    # if the block is inside the projection range, we draw...
                    if overlap:
                        LBx = self.bb[b,ind[0],1] - self.bb[b,ind[0],0] # block length x
                        LBy = self.bb[b,ind[1],1] - self.bb[b,ind[1],0] # block length y
                        if self.show_grid: # show grid cells in block
                            if 0 in self.show_grid or self.rl[b] in self.show_grid:
                                linewidth = 0.25
                                color = 'red'
                                dx = LBx / self.nb[ind[0]]
                                dy = LBy / self.nb[ind[1]]
                                for i in range(0,self.nb[ind[0]]): # lines along y
                                    corner = ( self.bb[b,ind[0],0]+i*dx, self.bb[b,ind[1],0] )
                                    patch = patches.Rectangle(corner, 0.0, LBy, fill=False, color=color, linewidth=linewidth)
                                    ax.add_patch(patch)
                                for j in range(0,self.nb[ind[1]]): # lines along x
                                    corner = ( self.bb[b,ind[0],0], self.bb[b,ind[1],0]+j*dy )
                                    patch = patches.Rectangle(corner, LBx, 0.0, fill=False, color=color, linewidth=linewidth)
                                    ax.add_patch(patch)
                        if self.show_blocks: # show block bounding box
                            linewidth = 0.5
                            color = 'blue'
                            if 0 in self.show_blocks or self.rl[b] in self.show_blocks:
                                corner = ( self.bb[b,ind[0],0], self.bb[b,ind[1],0] )
                                patch = patches.Rectangle(corner, LBx, LBy, fill=False, color=color, linewidth=linewidth)
                                ax.add_patch(patch)

        # vector field (e.g., velocity)
        if self.vec:
            # set number of vectors to draw
            vn = self.vec_n
            if vn[0] is None:
                vn_x = 50 # set of default vectors in x and scale number of vectors in y by size of extent
                vn = np.round(np.array([1.0, (extent[3]-extent[2])/(extent[1]-extent[0])]) * vn_x).astype(int).tolist()
            # interpolate to requested number of vectors
            vx = cfp.congrid(self.data_vx, vn)
            vy = cfp.congrid(self.data_vy, vn)
            vl = np.sqrt(vx**2 + vy**2)
            # get 2D coordinate grid
            x, y = cfp.get_2d_coords(cmin=[extent[0],extent[2]], cmax=[extent[1],extent[3]], ndim=vn)
            # scale and plot vectors
            if self.vec_scale is None:
                v_scl = np.mean(vl)
            else:
                v_scl = self.vec_scale
            qiver_obj = ax.quiver(x.T, y.T, vx, vy, pivot='mid', color=self.vec_color,
                                  minlength=0.0, width=0.0015, scale=2*vn[0]*v_scl/self.vec_scale_factor)
            # plot quiver legend (key)
            if self.vec_key and not map_only:
                v_lab = r'$'+str(cfp.round(v_scl))+r'$'
                v_unit = ''
                if (self.vec_var == 'vel'):
                    # velocity field
                    v_lab = r'$'+str(cfp.round(v_scl/1e5,2))+r'$'
                    v_unit = r'$\,\mathrm{km\,s^{-1}}$'
                elif (self.vec_var == 'moh'):
                    # radiation flux
                    v_lab = r'$'+str(cfp.eform(v_scl,2))+r'$'
                    v_unit = r'$\,\mathrm{erg\,cm^{-2}\,s^{-1}}$'
                if self.vec_transform != 'q': v_unit = ''
                if self.vec_unit is not None: v_unit = self.vec_unit
                q_lab = v_lab + v_unit
                q_lab_pos = [0.44, 1.02]
                if self.labels_inside is not None:
                    ypos = 0.94
                    if not isinstance(self.labels_inside, bool): ypos = float(self.labels_inside)
                    q_lab_pos = [0.44, ypos]
                    q_label = self.text()
                    q_label.text = q_lab
                    q_label.pos = q_lab_pos
                    q_label.style = 2
                    q_label.plot()
                    print("labels_inside with vector legend support not implemented yet", error=True)
                qiver_key = ax.quiverkey(qiver_obj, q_lab_pos[0], q_lab_pos[1], v_scl, q_lab, labelpos='E', labelsep=0.05)

        # magnetic field streamlines
        if self.stream:
            bn = self.stream_n
            if bn[0] is None:
                bn_x = 50 # set of default seed points in x and scale number of seed points in y by size of extent
                bn = np.round(np.array([1.0, (extent[3]-extent[2])/(extent[1]-extent[0])]) * bn_x).astype(int).tolist()
            # interpolate to requested number of vectors
            bx = cfp.congrid(self.data_bx, bn)
            by = cfp.congrid(self.data_by, bn)
            bl = np.sqrt(bx**2 + by**2)
            # scale length logarithmically
            scale_b_by_log = True
            if scale_b_by_log:
                if self.stream_vmin is None:
                    ind = bl > 0
                else:
                    if self.stream_vmin <= 0:
                        print("stream_vmin must be > 0 (for log scaling).", error=True)
                    ind = bl >= self.stream_vmin
                bl_log_floor = np.min(np.log10(bl[ind]))
                bl_log = bl*0 + bl_log_floor
                bl_log[ind] = np.log10(bl[ind])
                bx = bx/bl * (bl_log - bl_log_floor)
                by = by/bl * (bl_log - bl_log_floor)
                bl = np.sqrt(bx**2 + by**2) # new bl
                b_scl = 1.0 # this is a log scale for the thickness
            else:
                b_scl = np.rms(bl)
            # get 2D coordinate grid
            x, y = cfp.get_2d_coords(cmin=[extent[0],extent[2]], cmax=[extent[1],extent[3]], ndim=bn)
            # plot streamlines
            start_points = np.array([x.T.flatten().tolist(),y.T.flatten().tolist()])
            ax.streamplot(x.T, y.T, bx, by, color=self.stream_color, minlength=0.0, linewidth=bl/b_scl*self.stream_thick*0.25, start_points=start_points.T, density=5, arrowsize=0)

        # legend globals
        yoffset_font = 0.02 * (1.0 - rcParams['font.size'] / 10)
        legx = 0.03
        legy = 0.04 + yoffset_font
        legdy = 0.04 * rcParams['font.size'] / 10

        # plot tracer particles
        if self.show_tracers and self.tracers.n > 0:
            if self.particle_color is not None: color = self.particle_color
            else: color = 'white'
            markersize = 72./dpi * self.particle_size_factor
            plt.plot(self.tracers.posx[::self.show_tracers[0]], self.tracers.posy[::self.show_tracers[0]], linestyle = 'None', marker='.', markersize=markersize,
                     alpha=0.75, fillstyle='none', markeredgewidth=0.25*markersize, markerfacecolor=color, markeredgecolor=color)
            if self.tracers.marked_tags_ind is not None:
                color = self.particle_mark_tags[1]
                markersize *= float(self.particle_mark_tags[2])
                plt.plot(self.tracers.posx[self.tracers.marked_tags_ind][::self.show_tracers[0]], self.tracers.posy[self.tracers.marked_tags_ind][::self.show_tracers[0]],
                        linestyle = 'None', marker='.', markersize=markersize, alpha=0.75, fillstyle='none', markeredgewidth=0.25*markersize, markerfacecolor=color, markeredgecolor=color)

        # plot sink particles
        if self.show_sinks and self.sinks.n > 0:
            if self.particle_color is not None: color = self.particle_color
            else: color = 'blue'
            def scale_mass(mass):
                return 2**np.log10(mass/(1e-2*const.m_sol)) * 3*72./dpi
            # set markerwidth (boundary size)
            def markerwidth(markersize):
                return 0.025 * markersize
            # scale markersize by sink sink relative to extent
            one_length_unit_in_points = ax.get_window_extent().width  / (extent[1]-extent[0]) * 72/dpi
            markersize = np.zeros(self.sinks.n)
            # this gives the sink in its actual size and impose minimum plot size
            markersize[:] = max(self.sinks.radius * 2 * one_length_unit_in_points, 15*72/dpi)
            # scale markersize by sink mass
            if self.scale_sinks_by_mass: # scaling by mass, such that the size double every 10x in mass
                markersize = scale_mass(self.sinks.mass)
            markersize *= self.particle_size_factor
            # set symbol type
            symb = mpath.Path.unit_regular_star(6)
            # sort by mass
            ind = np.argsort(self.sinks.mass)
            posx = self.sinks.posx[ind]
            posy = self.sinks.posy[ind]
            for i in range(self.sinks.n):
                plt.plot(posx[i], posy[i], linestyle='None', marker=symb, markersize=markersize[i], markeredgewidth=markerwidth(markersize[i]), alpha=0.5, markerfacecolor=color)
                plt.plot(posx[i], posy[i], linestyle='None', marker=symb, markersize=markersize[i], markeredgewidth=markerwidth(markersize[i]), alpha=1.0, markeredgecolor='white', fillstyle='none')
            # if scaling by mass, add legend
            if self.scale_sinks_by_mass:
                masses_legend = np.array([0.1, 1, 10])*const.m_sol
                markersize = scale_mass(masses_legend)
                Lx = extent[1]-extent[0]
                Ly = extent[3]-extent[2]
                dx = 0.12
                px = np.linspace(0, len(masses_legend)-1, len(masses_legend)) * dx*Lx + extent[0] + legx*Lx + 0.9*markersize/one_length_unit_in_points
                py = np.zeros(len(masses_legend)) + extent[2] + 1.23*legy*Ly
                for i in range(len(masses_legend)):
                    plt.plot(px[i], py[i], linestyle='None', marker=symb, markersize=markersize[i], markeredgewidth=markerwidth(markersize[i]), alpha=0.5, markerfacecolor=color)
                    plt.plot(px[i], py[i], linestyle='None', marker=symb, markersize=markersize[i], markeredgewidth=markerwidth(markersize[i]), alpha=1.0, markeredgecolor='white', fillstyle='none')
                    self.textlabel.append(self.text())
                    self.textlabel[-1].text = r"$"+str(cfp.round(masses_legend[i]/const.m_sol,2))+r"\,\mathrm{M_\odot}$"
                    self.textlabel[-1].pos = [1.1*legx+dx*i+0.6*markersize[i]/one_length_unit_in_points, legy]
                    self.textlabel[-1].style = 1
            # add text labels for number of sinks and total sink mass
            shift_legy = 0
            if self.scale_sinks_by_mass: shift_legy = 1
            self.textlabel.append(self.text())
            self.textlabel[-1].text = r"$N_\mathrm{sinks}\,=\,"+str(self.sinks.n)+r"$"
            self.textlabel[-1].pos = [legx, (2+shift_legy)*legdy+yoffset_font]
            self.textlabel[-1].style = 1
            self.textlabel.append(self.text())
            self.textlabel[-1].text = r"$M_\mathrm{sinks}\,=\,"+str(cfp.round(np.sum(self.sinks.mass)/const.m_sol,3))+r"\,\mathrm{M_\odot}$"
            self.textlabel[-1].pos = [legx, (1+shift_legy)*legdy+yoffset_font]
            self.textlabel[-1].style = 1

        # add time and plot label
        if not map_only and not colorbar_only:
            style = None
            xoff = 0.0
            ypos = 1.025
            if self.labels_inside is not None:
                style = 2
                xoff = 0.03
                ypos = 0.96 + yoffset_font
                if not isinstance(self.labels_inside, bool): ypos = float(self.labels_inside)
            if self.time is not None or self.redshift:
                self.textlabel.append(self.text())
                self.textlabel[-1].pos = [0.0+xoff, ypos]
                self.textlabel[-1].style = style
                self.textlabel[-1].color = 'black'
                if self.redshift and self.show_redshift:
                    self.textlabel[-1].text = r"$\mathrm{Redshift}=\,$"+self.redshift_str+r"\phantom{8}"
                elif self.time is not None:
                    if self.time_scale != 0: self.textlabel[-1].text = r"$\mathrm{Time}=\,$"+self.time_str+r"\phantom{8}"
            if self.plotlabel != '':
                self.textlabel.append(self.text())
                self.textlabel[-1].text = self.plotlabel
                self.textlabel[-1].pos = [1.0-xoff, ypos]
                self.textlabel[-1].style = style
                self.textlabel[-1].color = 'black'
                self.textlabel[-1].align = 'right'

            # add text labels
            for i in range(len(self.textlabel)):
                self.textlabel[i].plot()

            # set axes limits
            ax.set_xlim([extent[0], extent[1]])
            ax.set_ylim([extent[2], extent[3]])

        # set the background colors (facecolor)
        if self.facecolor is not None:
            if self.facecolor == "cmap":
                color = mpl.cm.get_cmap(self.cmap)(0) # set to cmap low value color
            else: color = self.facecolor
            ax.set_facecolor(color) # background color of plot frame (inside the axes)
            if map_only:
                fig.set_facecolor(color) # figure background color

        if map_only:
            plt.axis('off')

        if colorbar_only:
            plt.axis('off')
            img.set_visible(False)
            ax.set_visible(False)

        show_or_save()

    # ============= end: plot_map =============


    # ============= plot_3d =============
    def plot_3d(self, filename, datasetname, slice=True, volume=False, streamlines=False, proj_range=None, data_range=None,
                pixel=128, ncpu=8, zoom=1.0, outdir=None, mass_weighting=False, mlab_axes=False, mlab_triad=True, mlab_cbar=True,
                vmin=None, vmax=None, view=None, move=None, map_only=False, outline=0, dpi=300):

        if self.verbose > 1: print("importing mayavi...")
        from mayavi import mlab
        if self.verbose > 1: print("done.")
        from tvtk.util import ctf

        # function that converts cmasher or mpl cmaps to color transfer function (ctf)
        def cmap_to_ctf(cmap_name):
            values = list(np.linspace(0, 1, 256))
            rgb = np.array(cmr.take_cmap_colors(cmap_name, 256, return_fmt='float')) # read cmasher RGB colors
            transfer_function = ctf.ColorTransferFunction()
            for i, v in enumerate(values):
                transfer_function.add_rgb_point(v, rgb[i, 0], rgb[i, 1], rgb[i, 2])
            return transfer_function

        # function that converts cmasher or mpl cmaps to RGBA
        def cmap_to_rgba(cmap_name, alpha=1.0):
            rgb = np.array(cmr.take_cmap_colors(cmap_name, 256, return_fmt='int')) # read cmasher RGB colors
            rgba = np.zeros((256,4), int) # RGBA container
            rgba[:,0:3] = rgb # RGB channels
            if type(alpha) == np.ndarray: # apply alpha array to color array
                if len(alpha) != 256:
                    print("If providing np.array, alpha must have length 256.", error=True)
                alpha_channel = 255*alpha
            else: # alpha is just a float, so we apply this alpha for all colors
                alpha_channel = np.zeros(256, int) + int(255*alpha)
            rgba[:,-1] = alpha_channel
            return rgba

        if 'screen' in self.outtype:
            offscreen = False
        else: # if we are saving the figure, we produce a matplotlib colorbar instead of the mlab_cbar
            offscreen = True
            mlab_cbar = False

        # start a mayavi figure
        plt.switch_backend('agg')
        mlab.options.offscreen = offscreen
        mfig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0,0,0), size=(1024,1024))

        # 3-slice
        if slice:
            if self.verbose: print("adding 3-slice...")
            filenames_slice = []
            datasetnames_slice = []
            for dir in ["x", "y", "z"]:
                if os.path.isfile(filename): # if source file is present we call projection
                    f, d = call_projection(filename, datasetname=datasetname, verbose=self.verbose, slice=True, direction=dir,
                                           proj_range=proj_range, zoom=zoom, pixel=pixel, ncpu=ncpu, outdir=outdir,
                                           data_range=data_range, do_particles=False)
                else: # we assume the slice files are already present
                    d = datasetname+"_slice"
                    f = filename+"_"+d+"_"+dir+".h5"
                filenames_slice.append(f)
                datasetnames_slice.append(d)

            # prep data ranges and extent
            vmin_set = False
            vmax_set = False
            if vmin is not None: vmin_set = True
            if vmax is not None: vmax_set = True
            if not vmin_set: vmin = +1e99
            if not vmax_set: vmax = -1e99
            extent = [+1e99,-1e99,+1e99,-1e99,+1e99,-1e99]
            for i in range(0, len(filenames_slice), 1):
                # prep datarange
                self.prep_map(filenames_slice[i], datasetnames_slice[i])
                if not vmin_set:
                    if self.vmin < vmin: vmin = self.vmin
                if not vmax_set:
                    if self.vmax > vmax: vmax = self.vmax
                # prep extent
                inds = [0,1,2]
                if self.direction == 'y': inds = [0,2,1]
                if self.direction == 'x': inds = [2,0,1]
                for i in range(0,3):
                    minimum = self.bounds[inds[i]][0]
                    maximum = self.bounds[inds[i]][1]
                    if inds[i]==2: # take the average of the slice thickness bounds
                        minimum = (self.bounds[inds[i]][0]+self.bounds[inds[i]][1])/2
                        maximum = (self.bounds[inds[i]][0]+self.bounds[inds[i]][1])/2
                    if (minimum < extent[2*i+0]): extent[2*i+0] = minimum
                    if (maximum > extent[2*i+1]): extent[2*i+1] = maximum
            if self.log:
                vmin = np.log10(vmin)
                vmax = np.log10(vmax)

            # plot 3 slices
            for i in range(0, len(filenames_slice), 1):
                self.prep_map(filenames_slice[i], datasetnames_slice[i])
                if self.direction == 'z':
                    X = cfp.get_1d_coords(self.bounds[0][0], self.bounds[0][1], self.dims[1])
                    Y = cfp.get_1d_coords(self.bounds[1][0], self.bounds[1][1], self.dims[0])
                    X, Y = np.meshgrid(X, Y)
                    Z    = np.zeros((self.dims[0], self.dims[1]))+(self.bounds[2][0]+self.bounds[2][1])/2 # place slice in middle of slice thickness bounds
                if self.direction == 'y':
                    X = cfp.get_1d_coords(self.bounds[0][0], self.bounds[0][1], self.dims[1])
                    Z = cfp.get_1d_coords(self.bounds[1][0], self.bounds[1][1], self.dims[0])
                    X, Z = np.meshgrid(X, Z)
                    Y    = np.zeros((self.dims[0], self.dims[1]))+(self.bounds[2][0]+self.bounds[2][1])/2
                if self.direction == 'x':
                    Y = cfp.get_1d_coords(self.bounds[0][0], self.bounds[0][1], self.dims[1])
                    Z = cfp.get_1d_coords(self.bounds[1][0], self.bounds[1][1], self.dims[0])
                    Y, Z = np.meshgrid(Y, Z)
                    X    = np.zeros((self.dims[0], self.dims[1]))+(self.bounds[2][0]+self.bounds[2][1])/2

                # scale data
                if self.log: data = np.log10(self.data)
                else: data = self.data

                # plot the surface(s) # the colormap is changed below
                mesh = mlab.mesh(X, Y, Z, scalars=data, colormap='viridis', vmin=vmin, vmax=vmax, extent=extent)

                # change the colors of the mesh plot to the target cmap (including cmasher)
                rgba = cmap_to_rgba(self.cmap, alpha=1.0) # we can also modify the alpha channel to add transparency
                mesh.module_manager.scalar_lut_manager.lut.table = rgba

        # add volume rendering
        if volume:
            if self.verbose: print("adding volume rendering...")
            # call extractor
            if os.path.isfile(filename): # if source file is present we call extractor
                extraction_file = call_extractor(filename, datasetname=datasetname, verbose=self.verbose,
                                                 mass_weighting=False, ncpu=ncpu, outdir=outdir)
            else: # we assume the extracted file is present
                extraction_file = filename+'_extracted.h5'
            # read extracted data
            dat3d = hdfio.read(extraction_file, datasetname)
            #if self.log: dat3d = np.log10(dat3d)
            minmax_xyz = hdfio.read(extraction_file, "minmax_xyz")
            extent = minmax_xyz.flatten()
            # generate coordinates
            cmin = np.array([minmax_xyz[0][0], minmax_xyz[1][0], minmax_xyz[2][0]])
            cmax = np.array([minmax_xyz[0][1], minmax_xyz[1][1], minmax_xyz[2][1]])
            ndim = dat3d.shape
            x, y, z = cfp.get_3d_coords(cmin=cmin, cmax=cmax, ndim=ndim)
            # add volume rendering
            source = mlab.pipeline.scalar_field(x, y, z, dat3d)
            vmin_3d = dat3d.min()
            vmax_3d = dat3d.max()
            vol = mlab.pipeline.volume(source, vmin=vmin_3d, vmax=vmax_3d)
            # change colormap via colour transfer function
            new_ctf = cmap_to_ctf(self.cmap)
            vol._volume_property.set_color(new_ctf)
            vol._ctf = new_ctf
            vol.update_ctf = True
            # change opacity transfer function
            if True:
                from tvtk.util.ctf import PiecewiseFunction
                otf = PiecewiseFunction()
                norm = extent[1]-extent[0]
                otf.add_point(0, 0.9999) # this is tricky; need to set the opacity transfer function well
                vol._otf = otf
                vol._volume_property.set_scalar_opacity(otf)

        # add volume rendering
        if streamlines:
            if self.verbose: print("adding volume rendering...")
            # call extractor
            if os.path.isfile(filename): # if source file is present we call extractor
                extraction_file = call_extractor(filename, datasetname="velx vely velz", verbose=self.verbose,
                                                 mass_weighting=False, ncpu=ncpu, outdir=outdir)
            else: # we assume the extracted file is present
                extraction_file = filename+'_extracted.h5'
            # read extracted data
            vx = hdfio.read(extraction_file, "velx")
            vy = hdfio.read(extraction_file, "vely")
            vz = hdfio.read(extraction_file, "velz")
            minmax_xyz = hdfio.read(extraction_file, "minmax_xyz")
            extent = minmax_xyz.flatten()
            # generate coordinates
            cmin = np.array([minmax_xyz[0][0], minmax_xyz[1][0], minmax_xyz[2][0]])
            cmax = np.array([minmax_xyz[0][1], minmax_xyz[1][1], minmax_xyz[2][1]])
            ndim = vx.shape
            x, y, z = cfp.get_3d_coords(cmin=cmin, cmax=cmax, ndim=ndim)
            # add streamlines
            vmin_3d = np.min((vx.min(), vy.min(), vz.min()))
            vmax_3d = np.min((vx.max(), vy.max(), vz.max()))
            flow = mlab.flow(x, y, z, vx, vy, vz, extent=extent, integration_direction='both', linetype='tube', vmin=vmin_3d, vmax=vmax_3d)
            flow.tube_filter.radius *= 0.5 # make the tubes a bit thinner

        # coordinate axes
        if mlab_axes:
            mlab.xlabel('x')
            mlab.ylabel('y')
            mlab.zlabel('z')
            axes = mlab.axes(extent=extent)
            axes.label_text_property.font_family = 'courier'
            axes.label_text_property.font_size = 2
            axes.label_text_property.bold = 0
            axes.label_text_property.italic = 0
            axes.title_text_property.font_family = 'courier'
            axes.title_text_property.font_size = 2
            axes.title_text_property.bold = 0
            axes.title_text_property.italic = 0

        # mlab colorbar
        if mlab_cbar:
            title = datasetname
            if self.log: title = "log10 "+title
            cbar = mlab.colorbar(title=title, orientation='vertical')
            cbar.label_text_property.font_family = 'courier'
            cbar.label_text_property.font_size = 5
            cbar.label_text_property.bold = 0
            cbar.label_text_property.italic = 0
            cbar.title_text_property.font_family = 'courier'
            cbar.title_text_property.font_size = 5
            cbar.title_text_property.bold = 0
            cbar.title_text_property.italic = 0

        # outline (box, plot frame)
        if outline is not None:
            mlab.outline(color=(outline,outline,outline), extent=extent)

        # coordinate axes triad
        if mlab_triad: mlab.orientation_axes(line_width=1)

        # from mlabtex import mlabtex
        #mid_x = (extent[1]-extent[0])/2+extent[0]
        #mid_y = (extent[3]-extent[2])/2+extent[2]
        #mid_z = (extent[5]-extent[4])/2+extent[4]
        #tex = mlabtex(mid_x, extent[3], extent[4], 'x', scale=0.05, color=(0., 0., 0.), orientation=(0., 0., 0.), dpi=1200)
        #tex = mlabtex(extent[1], mid_y, extent[4], 'y', scale=0.05, color=(0., 0., 0.), orientation=(0., 0., 0.), dpi=1200)
        #tex = mlabtex(extent[1], extent[2], mid_z, 'z', scale=0.05, color=(0., 0., 0.), orientation=(0., 0., 0.), dpi=1200)

        # select camera view
        if view is None:
            view = mlab.view()
            if self.verbose: print('camera view = ', view)
        else:
            mlab.view(azimuth=view[0], elevation=view[1], distance=view[2], focalpoint='auto')

        # select camera position
        if move is None:
            move = mlab.move()
            if self.verbose: print('camera move = ', move)
        else:
            mlab.move(right=move[0], up=move[1])

        if not offscreen:
            if self.verbose: print("rendering for screen output...")
            mlab.show()
            return

        # the following applies if we are saving the figure
        mlabfig = mlab.screenshot(mode='rgba', antialiased=True)
        fig = plt.figure()
        plt.imshow(mlabfig)
        plt.axis('off')

        # add matplotlib colorbar
        ax = plt.gca()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        if self.log:
            norm = mpl.colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
        if self.colorbar is not None:
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), ax=ax, label=self.cmap_label, pad=0.03)
            if not self.log: cb.ax.minorticks_on()
            cb.ax.yaxis.set_offset_position('left')

        # add labels
        if not map_only: ax.text(0.03, 0.97, r"$\mathrm{Time}=\,$"+self.time_str, transform=ax.transAxes, fontsize=12, horizontalalignment='left', verticalalignment='bottom')
        if self.plotlabel != '': ax.text(0.99, 0.97, self.plotlabel, transform=ax.transAxes, fontsize=12, horizontalalignment='right', verticalalignment='bottom')

        # save figure
        for outtype in self.outtype:
            if outtype == 'screen': continue
            pad_inches = 0.04
            if outtype in ["png", "jpg"]:
                pad_inches = 0.08
            if self.outname is not None:
                outfilename = self.outname+"_3d."+outtype
            else:
                outfilename = self.filename+"_"+self.datasetname+"_3d."+outtype
            if self.verbose: print("saving '"+outfilename+"'...")
            plt.savefig(outfilename, format=outtype, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)
            if self.verbose: print("'"+outfilename+"' written.")

        mlab.clf()
        mlab.close()
        plt.clf()
        plt.close()

    # ============= end: plot_3d =============

# ============= end: flashplotlib class =============


# ============= start: animation functions ==============
# easing functions to smoothen animation
# input x = [0,1]; returns in [0,1]
def easeInOutQuad(x):
    return 2*x**2 if x<0.5 else 1-(-2*x+2)**2/2
def easeInQuad(x):
    return x**2
def easeOutQuad(x):
    return 1-(1-x)**2
def easeInOutBack(x):
    c1 = 1.70158; c2 = c1*1.525
    return (4*x**2*((c2+1)*2*x-c2))/2 if x<0.5 else ((2*x-2)**2*((c2+1)*(x*2-2)+c2)+2)/2

# function that returns a list of values for each frame, with input beginning and end values
# for an arbitrary value, or for projection range animations (needs input centre and size)
def animate(nframes=100, dframe=1, ease_func=easeInOutQuad, value_beg=None, value_end=None,
            centre_beg=None, centre_end=None, size_beg=None, size_end=None, quiet=True):
    return_list = []
    for i in range(0, nframes, dframe):
        # increment [0 at the beginning, 1 at the end of frames]
        if nframes > 1: increment = ease_func(i/(nframes-1))
        else: increment = 0.0
        if not quiet: print("frame, increment = ", i+1, increment)
        # get value list bsaed on beginning and end values
        if value_beg is not None and value_end is not None:
            return_list.append(value_beg+(value_end-value_beg)*increment)
        # get range values based on centre and size (beginning to end)
        if centre_beg is not None and centre_end is not None and size_beg is not None and size_end is not None:
            if len(centre_beg)!=3 or len(centre_end)!=3 or len(size_beg)!=3 or len(size_end)!=3:
                print("centre_beg, centre_end, size_beg, size_end must be numpy arrays of size 3, i.e., X, Y, Z", error=True)
            range_min = (centre_beg+(centre_end-centre_beg)*increment) - (size_beg+(size_end-size_beg)*increment)/2.0
            range_max = (centre_beg+(centre_end-centre_beg)*increment) + (size_beg+(size_end-size_beg)*increment)/2.0
            return_list.append(np.array([range_min[0], range_max[0], range_min[1], range_max[1], range_min[2], range_max[2]]))
    return np.array(return_list)
# ============= end: animation functions ==============


# ========================================================
def call_projection(plt_filename, datasetname='dens', slice=None, direction='z', mass_weighting=False, proj_range=None,
                    zoom=None, pixel=[1024,1024], ncpu=8, outdir='.', data_range=None, view_angle=None,
                    rotation_angle=None, rotation_axis=None, rotation_centre=None, spherical_weight_radii=[None,None],
                    do_particles=True, boundary=None, opacity_weight=None, split_cells=False, verbose=1):
    # set the return filename and datasetname
    type_str = 'proj'
    if not fl.is_extracted_file(plt_filename):
        if fl.read_scalars(plt_filename)['dimensionality'] == 2: slice = 0.5 # trigger slice
    if slice is not None: type_str = 'slice'
    # get actual filename (with leading directory path removed)
    ind = plt_filename.rfind('/') + 1
    # construct output file name including path
    ret_filename = outdir+'/'+plt_filename[ind:]+'_'+datasetname+'_'+type_str+'_'+direction
    # construct output dataset name
    ret_datasetname = datasetname+'_'+type_str
    # define all calling options for FLASH projection (C++)
    output_str = " -o '"+ret_filename+"'"
    dataset_str = " -dset '"+datasetname+"'"
    slice_str = ''
    if slice is not None:
        slice_str = ' -slice'
    dir_str = ' -view xy'
    if direction=='y':
        dir_str = ' -view xz'
    if direction=='x':
        dir_str = ' -view yz'
    mw_str = ''
    if mass_weighting: mw_str = ' -mw'
    proj_range_str = ''
    if proj_range is not None:
        proj_range_str = ' -range'
        for x in proj_range: proj_range_str = proj_range_str+' '+str(x)
    zoom_str = ''
    if zoom is not None:
        zoom_str = ' -zoom ' + str(zoom)
    data_range_str = ''
    if data_range is not None:
        data_range_str = ' -data_range'
        for x in data_range: data_range_str = data_range_str+' '+str(x)
    pixel_str = ' -pixel 1024 1024'
    if pixel is not None:
        pixel_str = ' -pixel '+str(pixel[0])+' '+str(pixel[1])
    view_angle_str = ''
    if view_angle is not None:
        view_angle_str = ' -viewangle ' + str(view_angle)
    rotation_angle_str = ''
    if rotation_angle is not None:
        rotation_angle_str = ' -rotangle ' + str(rotation_angle)
    rotation_axis_str = ''
    if rotation_axis is not None:
        rotation_axis_str = ' -rotaxis'
        for x in rotation_axis: rotation_axis_str = rotation_axis_str+' '+str(x)
    rotation_centre_str = ''
    if rotation_centre is not None:
        rotation_centre_str = ' -rotcenter'
        for x in rotation_centre: rotation_centre_str = rotation_centre_str+' '+str(x)
    sp_r_max_str = ''
    if spherical_weight_radii[0] is not None:
        sp_r_max_str = ' -sp_r_max '+str(spherical_weight_radii[0])
    sp_r_chr_str = ''
    if spherical_weight_radii[1] is not None: # characteristic radius for spherical weighting
        sp_r_chr_str = ' -sp_r_chr '+str(spherical_weight_radii[1])
    no_particles_str = ''
    if not do_particles:
        no_particles_str = ' -no_particles'
    boundary_str = ""
    if boundary is not None:
        boundary_str = " -bc "+boundary
    opacity_str = ""
    if opacity_weight is not None:
        opacity_str = " -opacity "+str(opacity_weight)
    split_cells_str = ""
    if split_cells:
        split_cells_str = " -split_cells"
    starfield_str = ""
    if datasetname == "starfield": # special case to make starfield from sink particles
        dataset_str = " -dset 'dens'"
        no_particles_str = ""
        starfield_str = " -make_starfield"
    mpi_cmd = "mpirun -np "
    hostname = cfp.get_hostname()
    if hostname.find("setonix") != -1 or hostname.find("nid") != -1 or hostname.find("dm") != -1:
        mpi_cmd = "srun -n "
    if ncpu == 1:
        mpi_cmd = ""
        ncpu = ""
    cmd = mpi_cmd+str(ncpu)+" projection '"+plt_filename+"'"+dataset_str+pixel_str+slice_str+dir_str+mw_str+proj_range_str+\
          zoom_str+data_range_str+output_str+no_particles_str+view_angle_str+rotation_angle_str+rotation_axis_str+rotation_centre_str+\
          sp_r_max_str+sp_r_chr_str+boundary_str+opacity_str+split_cells_str+starfield_str
    # call projection
    if verbose > 1: print("============= START projection =============")
    p = cfp.run_shell_command(cmd)
    if p.returncode != 0:
        print("call to FLASH projection (C++) failed with an error - check output above.", error=True)
    if verbose > 1: print("============= END projection =============")
    # return
    if rotation_angle is not None: ret_filename += "_rotated"
    return ret_filename+'.h5', ret_datasetname
# ============= end: call_projection =============


# ========================================================
def call_extractor(plt_filename, datasetname='dens', mass_weighting=False, extraction_range=None,
                   pixel=[128,128,128], ncpu=8, outdir='.', verbose=1):
    # get actual filename (with leading directory path removed)
    ind = plt_filename.rfind('/') + 1
    # construct output file name including path
    ret_filename = outdir+'/'+plt_filename[ind:]+'_extracted.h5'
    # define all calling options for FLASH projection (C++)
    output_str = " -o '"+ret_filename+"'"
    dataset_str = " -dsets "+datasetname
    mw_str = ''
    if mass_weighting: mw_str = ' -mws 1'
    extraction_range_str = ''
    if extraction_range is not None:
        extraction_range_str = ' -range'
        for r in extraction_range: extraction_range_str += ' '+str(r)
    pixel_str = ' -pixel '+str(pixel[0])+' '+str(pixel[1])+' '+str(pixel[2])
    mpi_cmd = "mpirun -np "
    hostname = cfp.get_hostname()
    if hostname.find("setonix") != -1 or hostname.find("nid") != -1 or hostname.find("dm") != -1:
        mpi_cmd = "srun -n "
    if ncpu == 1:
        mpi_cmd = ""
        ncpu = ""
    cmd = mpi_cmd+str(ncpu)+" extractor_mpi '"+plt_filename+"'"+dataset_str+pixel_str+mw_str+extraction_range_str+output_str
    # call projection
    if verbose > 1: print("============= START extraction =============")
    run_shell_command(cmd)
    if verbose > 1: print("============= END extraction =============")
    # return
    return ret_filename
# ============= end: call_extractor =============


# ==================== parse_args =========================
def parse_args(process_args_locally=False):

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Make plots of FLASH output files (plt, part, chk, movie).')
    parser.add_argument("-i", "--i", dest='filename', nargs='*', help="HDF5 input file(s)")
    parser.add_argument("-d", "--d", dest='datasetname', default='dens', type=str, help="HDF5 dataset name")
    parser.add_argument("-verbose", "--verbose", type=int, default=1, help="Verbose level")
    parser.add_argument("-outtype", "--outtype", action='append', nargs='?', default=[], choices=['screen', 'pdf', 'eps', 'png', 'jpg'], help="Output type")
    parser.add_argument("-outname", "--outname", type=str, help="Output file name")
    parser.add_argument("-outdir", "--outdir", type=str, default='.', help="Output directory")
    parser.add_argument("-lowres", "--lowres", action='store_true', default=False, help="Do low-resolution screen output, e.g., for remote plotting")
    parser.add_argument("-facecolor", "--facecolor", type=str, help="Background color (facecolor) of plot frame ('cmap' sets to lower cmap color)")
    parser.add_argument("-colorbar", "--colorbar", nargs='*', default=[True], help="colorbar options (e.g., switch off, 'only' produces colorbar only; 'aspect50' makes it thinner; 'panels2' makes a bar spanning 2 panels; options can be combined)")
    parser.add_argument("-cmap", "--cmap", type=str, default="afmhot", help="Color map to use")
    parser.add_argument("-cmap_label", "--cmap_label", type=str, help="Colorbar label")
    parser.add_argument("-cmap_format", "--cmap_format", type=str, help="Colorbar tick format")
    parser.add_argument("-vmin", "--vmin", type=float, help="Min value for color map")
    parser.add_argument("-vmax", "--vmax", type=float, help="Max value for color map")
    parser.add_argument("-s", "-slice", "--slice", nargs='?', type=float, const=0.5, help="Instead of projection, make slice through fraction '-slice <value>' of the domain with default value=0.5 (center)")
    parser.add_argument("-direction", "--direction", choices=['x', 'y', 'z'], default='z', help="Projection direction")
    parser.add_argument("-nolog", "--nolog", action='store_true', default=False, help="Switches off log scaling for color map")
    parser.add_argument("-mw", "--mw", action='store_true', default=False, help="Apply mass-weighting in call to FLASH 'projection' (C++)")
    parser.add_argument("-range", "--range", nargs=6, type=float, help="Set the projection range (allows cutting, zooming, etc.)")
    parser.add_argument("-axes_label", "--axes_label", nargs='*', type=str, default=[None], help="Axes labels (use empty strings to turn axes labels off)")
    parser.add_argument("-axes_unit", "--axes_unit", nargs='*', type=str, default=[None], help="Axes units")
    parser.add_argument("-axes_format", "--axes_format", nargs='*', type=str, default=[None,None,None], help="String format for axes ticks (use empty strings to turn ticks off)")
    parser.add_argument("-axes_transform", "--axes_transform", nargs='*', type=str, default=['q','q','q'], help="Transform axes ticks by string expression (use 'q' in expression to access axes values)")
    parser.add_argument("-time_unit", "--time_unit", type=str, help="Time unit")
    parser.add_argument("-time_scale", "--time_scale", type=float, help="Time will be scaled (divided) by time_scale (can be 0 to remove time label)")
    parser.add_argument("-time_format", "--time_format", type=str, help="Time string format")
    parser.add_argument("-time_transform", "--time_transform", default='q', type=str, help="Transform time by string expression (use 'q' in expression to access time values)")
    parser.add_argument("-map_only", "--map_only", action='store_true', default=False, help="Produce a map without any annotations")
    choices_for_centre_on = ['max_dens', 'max_pden', 'max_sink', 'coord']
    help_for_centre_on = ", ".join(choices_for_centre_on)
    parser.add_argument("-centre_on", "--centre_on", help="Centre projection/slice on a position; choices are: "+help_for_centre_on)
    parser.add_argument("-data_range", "--data_range", nargs=2, type=float, help="Set the min and max data range when projecting/slicing.")
    parser.add_argument("-data_transform", "--data_transform", default='q', type=str, help="Transform data by string expression (use 'q' in expression to access data values)")
    parser.add_argument("-zoom", "--zoom", type=float, default=1.0, help="zoom option for FLASH 'projection' (C++)")
    parser.add_argument("-view_angle", "--view_angle", type=float, help="Viewing angle (in degrees) for perspective view (suggested value for rotation: 5.0).")
    parser.add_argument("-rot_angle", "--rot_angle", type=float, help="Rotation angle (in degrees).")
    parser.add_argument("-rot_axis", "--rot_axis", nargs=3, type=float, help="Rotation axis (e.g., 0 0 1 is the z axis).")
    parser.add_argument("-rot_centre", "--rot_centre", nargs=3, type=float, help="Rotation centre coordinate.")
    parser.add_argument("-spherical_weight_radii", "--spherical_weight_radii", nargs=2, type=float, default=[None,None], help="Apply spherical cutoff at radius [sp_r_max, sp_r_chr].")
    parser.add_argument("-opacity_weight", "--opacity_weight", type=float, help="Apply opacity weighting in call to FLASH 'projection' (C++).")
    parser.add_argument("-split_cells", "--split_cells", action='store_true', default=False, help="Split cells in call to FLASH 'projection' (C++)")
    parser.add_argument("-pixel", "--pixel", type=int, nargs='*', help="Pixels for projection")
    parser.add_argument("-plotlabel", "--plotlabel", type=str, default="", help="Label for the plot")
    parser.add_argument("-labels_inside", "--labels_inside", nargs='?', const=True, help="Put labels inside map")
    parser.add_argument("-shift_periodic", "--shift_periodic", nargs=2, type=float, help="Apply periodic coordinate shift (roll) (in units of axes_unit or automatic).")
    parser.add_argument("-shift", "--shift", nargs=2, type=float, help="Shift/adjust axes coordinates (in units of axes_unit or automatic)")
    parser.add_argument("-gauss_smooth_fwhm", "--gauss_smooth_fwhm", type=float, nargs='*', default=[None], help="Gaussian smoothing with FWHM (in units of axes_unit or automatic)")
    parser.add_argument("-boundary", "--boundary", type=str, choices=['isolated', 'periodic'], help="Boundary condition (used for Gaussian smoothing, and in FLASH 'projection' (C++))")
    parser.add_argument("-scale_sinks_by_mass", "--scale_sinks_by_mass", action='store_true', default=False, help="Scale sink particle plot size by their mass")
    parser.add_argument("-show_sinks", "--show_sinks", action='store_true', default=False, help="Show sink particles")
    parser.add_argument("-show_tracers", "--show_tracers", type=int, nargs='*', help="Show tracer particles (if int is supplied, draw only every int particles)")
    parser.add_argument("-show_redshift", '--show_redshift', action='store_true', help="Show redshift instead of time as plot header, if available")
    parser.add_argument("-particle_color", "--particle_color", help="Particle color")
    parser.add_argument("-particle_size_factor", "--particle_size_factor", type=float, default=1.0, help="Particle size factor")
    parser.add_argument("-particle_mark_tags", "--particle_mark_tags", nargs=3, default=['particle_mark_tags.h5', 'blue', 2.0], help="Mark particles by tags in [<HDF5-file>, <color>, <size_factor>] with default '%(default)s'")
    parser.add_argument("-convert2proper", "--convert2proper", action='store_true', default=False, help="Convert cosmological co-moving quantities to proper quantities")
    parser.add_argument("-v", "-vec", "--vec", action='store_true', default=False, help="Show vector field.")
    parser.add_argument("-vec_var","--vec_var", type=str, default='vel', help="FLASH variable for vector field. Default '%(default)s' for velx, vely, velz")
    parser.add_argument("-vec_mw","--vec_mw", action='store_true', default=False, help="Apply mass-weighting on vector quantities in call to FLASH 'projection' (C++). Default '%(default)s'")
    parser.add_argument("-vec_transform", "--vec_transform", default='q', type=str, help="Transform vec data by string expression (use 'q' in expression to access vec data values)")
    parser.add_argument("-vec_unit", "--vec_unit", type=str, help="Vector unit")
    parser.add_argument("-vec_scale", "--vec_scale", type=float, help="Vector scale (sets the value of the vector legend)")
    parser.add_argument("-vec_scale_factor", "--vec_scale_factor", type=float, default=1.0, help="Scale length of vectors by this factor")
    parser.add_argument("-vec_n", "--vec_n", type=int, nargs='*', default=[None], help="Number of vectors")
    parser.add_argument("-vec_color", "--vec_color", type=str, default='black', help="Color of vectors")
    parser.add_argument("-vec_key","--vec_key", action='store_true', default=False, help="Show vector legend (key)")
    parser.add_argument("-stream", "--stream", action='store_true', default=False, help="Show streamlines")
    parser.add_argument("-stream_var", "--stream_var", type=str, default='mag', help="FLASH variable for stream lines. Default '%(default)s' for magx, magy, magz")
    parser.add_argument("-stream_mw", "--stream_mw", action='store_true', default=False, help="Apply mass-weighting on streamline quantities in call to FLASH 'projection' (C++). Default '%(default)s'")
    parser.add_argument("-stream_transform", "--stream_transform", default='q', type=str, help="Transform streamline data by string expression (use 'q' in expression to access streamline data values)")
    parser.add_argument("-stream_vmin", "--stream_vmin", type=float, help="Minimum value for streamlines (must be > 0; default: min > 0)")
    parser.add_argument("-stream_thick", "--stream_thick", type=float, default=1.0, help="Streamline thickness in relative units (default: %(default)s)")
    parser.add_argument("-stream_n", "--stream_n", type=int, nargs='*', default=[None], help="Number of streamlines")
    parser.add_argument("-stream_color", "--stream_color", type=str, default='blue', help="Color of streamlines")
    parser.add_argument("-ncpu", "--ncpu", type=int, default=8, help="Number of cores for FLASH 'projection' (C++)")
    parser.add_argument("-show_blocks", "--show_blocks", type=int, nargs='*', help="Show FLASH block structure (can provide list of AMR levels)")
    parser.add_argument("-show_grid", "--show_grid", type=int, nargs='*', help="Show FLASH grid cells (can provide list of AMR levels)")
    parser.add_argument("-dpi", "--dpi", type=float, help="DPI for output")
    parser.add_argument("-fontsize", "--fontsize", type=float, default=1.0, help="fontsize factor (default: %(default)s)")
    parser.add_argument("-p3d", "--p3d", action='store', nargs='+', choices=['slice', 'vol', 'stream'], help="Plot in 3D")
    parser.add_argument("-p3d_options", "--p3d_options", action='store', nargs='+', choices=['axes', 'triad'], default=[], help="3D options")
    parser.add_argument("-p3d_view", "--p3d_view", nargs=3, type=float, help="3D camera view [azimuth, elevation, distance]")
    parser.add_argument("-p3d_move", "--p3d_move", nargs=2, type=float, help="3D camera move [right, up]")
    # this is so argparse handles negative arguments properly (for example, for range)
    for i, arg in enumerate(sys.argv):
        if len(arg) > 1:
            if arg[0] == '-' and arg[1].isdigit(): sys.argv[i] = ' ' + arg

    if process_args_locally:
        args_for_parser = None # normal parse_args mode
    else:
        args_for_parser = [] # this is when the module is included from another python script
    args = parser.parse_args(args=args_for_parser)
    # in case no args were passed at all, print usage details
    #if len(sys.argv) == 1:
    #    parser.print_usage()
    #    exit()

    # set actual defaults for special list args, if present
    if True in [x.find('show_blocks')!=-1 for x in sys.argv] and not args.show_blocks: args.show_blocks = [0] # show all levels
    if True in [x.find('show_grid')!=-1 for x in sys.argv] and not args.show_grid: args.show_grid = [0] # show all levels
    if True in [x.find('show_tracers')!=-1 for x in sys.argv] and not args.show_tracers: args.show_tracers = [1] # show all tracer particles

    # set default output to 'screen'
    if len(args.outtype) == 0: args.outtype.append('screen')

    # see if we have write permission in output directory; if not, create temporary dir
    if not os.access(args.outdir, os.W_OK | os.X_OK):
        args.outdir = tempfile.mkdtemp()

    # handle axes keywords
    if len(args.axes_label) > 3: # catch error
        print("The 'axes_label' option can only take <= 3 arguments.", error=True)
    if len(args.axes_label) == 1: args.axes_label.append(None)
    if len(args.axes_label) == 2: args.axes_label.append(None)
    if len(args.axes_unit) > 3: # catch error
        print("The 'axes_unit' option can only take <= 3 arguments.", error=True)
    if len(args.axes_unit) == 1: args.axes_unit.append(None)
    if len(args.axes_unit) == 2: args.axes_unit.append(None)
    if len(args.axes_format) > 3: # catch error
        print("The 'axes_format' option can only take <= 3 arguments.", error=True)
    if len(args.axes_format) == 1: args.axes_format.append(None)
    if len(args.axes_format) == 2: args.axes_format.append(None)
    if len(args.axes_transform) > 3: # catch error
        print("The 'axes_transform' option can only take <= 3 arguments.", error=True)
    if len(args.axes_transform) == 1: args.axes_transform.append('q')
    if len(args.axes_transform) == 2: args.axes_transform.append('q')

    # handle range keyword
    if args.range is not None:
        args.range = np.array(args.range)
        for dir in range(3):
            if args.axes_unit[dir] is not None:
                length_factor = 1.0
                if args.axes_unit[dir].lower() == 'm': length_factor = 1e2
                if args.axes_unit[dir].lower() == 'km': length_factor = 1e5
                if args.axes_unit[dir].lower() == 'rsol': length_factor = const.r_sol
                if args.axes_unit[dir].lower() == 'au': length_factor = const.au
                if args.axes_unit[dir].lower() == 'pc': length_factor = const.pc
                if args.axes_unit[dir].lower() == 'kpc': length_factor = 1e3*const.pc
                if args.axes_unit[dir].lower() == 'mpc': length_factor = 1e6*const.pc
                args.range[2*dir:2*dir+2] *= length_factor

    # handle pixel keyword
    if args.pixel is not None:
        if len(args.pixel) > 2: print("The 'pixel' option can only take <= 2 arguments.", error=True)
        if len(args.pixel) == 1: args.pixel.append(args.pixel[0])

    # handle vec_n keyword
    if len(args.vec_n) > 2: # catch error
        print("The 'vec_n' option can only take <= 2 arguments.", error=True)
    if len(args.vec_n) == 1: args.vec_n.append(args.vec_n[0])

    # handle stream_n keyword
    if len(args.stream_n) > 2: # catch error
        print("The 'stream_n' option can only take <= 2 arguments.", error=True)
    if len(args.stream_n) == 1: args.stream_n.append(args.stream_n[0])

    # handle gauss_smooth_fwhm keyword
    if len(args.gauss_smooth_fwhm) > 2: # catch error
        print("The 'gauss_smooth_fwhm' option can only take <= 2 arguments (x and y FWHM for Gaussian smoothing).", error=True)
    if len(args.gauss_smooth_fwhm) == 1: args.gauss_smooth_fwhm.append(args.gauss_smooth_fwhm[0])

    # FLASH 'projection' (C++)
    if args.mw and ((args.spherical_weight_radii[0] is not None) or (args.view_angle is not None) or (args.opacity_weight is not None)):
        if args.verbose: print("CAUTION: -mw should not be used together with -spherical_weight_radii or -view_angle view or -opacity_weight!")

    # handle centre_on keyword
    if args.centre_on is not None:
        valid_choice = False
        for choice in choices_for_centre_on:
            if args.centre_on.find(choice) == 0: valid_choice = True
        if not valid_choice: print("Invalid choice for '-centre_on' selected; valid choices are", choices_for_centre_on, error=True)

    # attach the parser to the args class instance
    args.parser = parser

    return args
# ==================== end: parse_args =========================

# ================= get_requested_datasetnames ==================
def get_requested_datasetnames(input_datasetname):
    if input_datasetname.find('derived:') == 0:
        # outflow tracer -> outflow gas density
        if input_datasetname[len('derived:'):] == 'oadv':
            return [input_datasetname, 'oadv', 'dens']
    else:
        return [input_datasetname]
# =========== end: get_requested_datasetnames ====================

# ======================= process_file ===========================
def process_file(filen, args):

    # save args.datasetname in case we call process_file multiple times with changed options
    datasetname_saved = args.datasetname

    # see if we want particles
    do_particles = False
    if args.show_tracers or args.show_sinks:
        do_particles = True

    # filen is a flash plt or chk file
    if fl.is_plt_file(filen) or fl.is_chk_file(filen) or fl.is_extracted_file(filen):

        if args.p3d is None: # normal 2D plotting

            # get unit system
            try: unitsystem = fl.read_runtime_parameters(filen)['pc_unitsbase']
            except: pass

            # get boundary conditions
            gg = fl.FlashGG(filen)
            if args.boundary is None: args.boundary = gg.BoundaryType

            if args.show_blocks or args.show_grid:
                nb = gg.NB
                if args.boundary == "periodic": gg.AddBlockReplicasPBCs()
                bb = gg.BoundingBox[gg.NodeType==1] # limit to leaf blocks
                rl = gg.RefineLevel[gg.NodeType==1] # limit to leaf blocks
                if args.verbose: print("Maximum refinement level = "+str(rl.max()))

            # handle centre_on_maxdens keyword
            if args.centre_on is not None:
                if args.centre_on.find("coord_") != -1:
                    split_str = args.centre_on.split("_")
                    split_str = split_str[1].split(",")
                    if len(split_str) != 3:
                        print("'-centre_on coord' must be formatted as '-centre_on coord_X,Y,Z', X,Y,Z are the values of the x, y, and z coordinates'", error=True)
                    max_loc = np.array([float(split_str[0]), float(split_str[1]), float(split_str[2])])
                    if args.verbose: print("Location of max density as supplied: "+str(max_loc))
                if args.centre_on == "max_dens":
                    max_loc = gg.GetMinMax(datasetname="dens").max_loc
                    if args.verbose: print("Location of max density = "+str(max_loc))
                if args.centre_on == "max_pden":
                    max_loc = gg.GetMinMax(datasetname="pden").max_loc
                    if args.verbose: print("Location of max particle density = "+str(max_loc))
                if args.centre_on == "max_sink":
                    fp = fl.Particles(filen.replace("plt_cnt", "part"), verbose=0)
                    sinks = fp.read(type=fp.sink_type)
                    imax = np.argmax(sinks["mass"])
                    max_loc = np.array([sinks["posx"][imax], sinks["posy"][imax], sinks["posz"][imax]])
                    if args.verbose: print("Location of most massive sink particle = "+str(max_loc))
                # if user provided a projection range, we use the user extent, but centre on the max location
                if args.range is not None:
                    proj_L = [args.range[2*x+1]-args.range[2*x] for x in range(3)]
                else:
                    proj_L = gg.L # use the full extent instead
                # new projection range that would use the respective extent
                args.range = [max_loc[x//2] + (-1+2*(x%2))*proj_L[x//2]/2 for x in range(6)]
                # correct projection range if we make a slice
                if args.slice is not None:
                    if args.direction == 'x':
                        args.range[0] = max_loc[0]
                        args.range[1] = max_loc[0]
                    if args.direction == 'y':
                        args.range[2] = max_loc[1]
                        args.range[3] = max_loc[1]
                    if args.direction == 'z':
                        args.range[4] = max_loc[2]
                        args.range[5] = max_loc[2]
            else: # not centering
                if args.slice is not None:
                    # default is whole domain, unless user has specified a range
                    if args.range is None: args.range = [gg.domain_bounds[x//2,x%2] for x in range(6)]
                    # center on requested slice location (in %)
                    if args.direction == 'x':
                        args.range[0] = gg.domain_bounds[0,0] + args.slice*gg.L[0]
                        args.range[1] = gg.domain_bounds[0,0] + args.slice*gg.L[0]
                    if args.direction == 'y':
                        args.range[2] = gg.domain_bounds[1,0] + args.slice*gg.L[1]
                        args.range[3] = gg.domain_bounds[1,0] + args.slice*gg.L[1]
                    if args.direction == 'z':
                        args.range[4] = gg.domain_bounds[2,0] + args.slice*gg.L[2]
                        args.range[5] = gg.domain_bounds[2,0] + args.slice*gg.L[2]

            # container for datasetnames
            datasetnames_for_projection = []
            mw_for_projection = [] # mass weighting

            # datasetnames for vector and/or streamline projections/slices
            if args.vec or args.stream:
                if args.direction == 'z': dirs = ['x', 'y']
                if args.direction == 'y': dirs = ['x', 'z']
                if args.direction == 'x': dirs = ['y', 'z']
                if args.vec:
                    datasetnames_for_projection.append(args.vec_var.lower()+dirs[0])
                    datasetnames_for_projection.append(args.vec_var.lower()+dirs[1])
                    mw_for_projection.append(args.vec_mw)
                    mw_for_projection.append(args.vec_mw)
                if args.stream:
                    datasetnames_for_projection.append(args.stream_var.lower()+dirs[0])
                    datasetnames_for_projection.append(args.stream_var.lower()+dirs[1])
                    mw_for_projection.append(args.stream_mw)
                    mw_for_projection.append(args.stream_mw)

            # get dataset projections/slices
            requested_datasetnames = get_requested_datasetnames(args.datasetname)
            if len(requested_datasetnames) == 1: # just one datasetname
                datasetnames_for_projection.append(args.datasetname)
                mw_for_projection.append(args.mw)
            else: # a derived dataset or expression was requested
                for dsetn in requested_datasetnames[1:]:
                    datasetnames_for_projection.append(dsetn)
                    mw_for_projection.append(args.mw)
                req_filen = []
                req_datasetname = []
            for i, dsetn in enumerate(datasetnames_for_projection):
                if args.verbose > 1: print("====================== START: calling projection for '"+dsetn+"' ======================")
                if i==len(datasetnames_for_projection)-1:
                    data_range = args.data_range # apply data_range only to main dataset
                else:
                    data_range = None
                file_ret, dset_ret = call_projection(filen, datasetname=dsetn, verbose=args.verbose, slice=args.slice, direction=args.direction,
                                                     mass_weighting=mw_for_projection[i], proj_range=args.range, view_angle=args.view_angle,
                                                     rotation_angle=args.rot_angle, rotation_axis=args.rot_axis, rotation_centre=args.rot_centre,
                                                     spherical_weight_radii=args.spherical_weight_radii, data_range=data_range,
                                                     zoom=args.zoom, pixel=args.pixel, ncpu=args.ncpu, outdir=args.outdir,
                                                     do_particles=do_particles, boundary=args.boundary, opacity_weight=args.opacity_weight,
                                                     split_cells=args.split_cells)
                if len(requested_datasetnames) == 1: # just one datasetname
                    if i == len(datasetnames_for_projection)-1:
                        filen = file_ret
                        args.datasetname = dset_ret
                else: # a derived dataset or expression was requested
                    if i >= len(datasetnames_for_projection)-len(requested_datasetnames)+1:
                        req_filen.append(file_ret)
                        req_datasetname.append(dset_ret)
                if args.verbose > 1: print("======================= END: calling projection for '"+dsetn+"' =======================")
            if len(requested_datasetnames) > 1: # a derived dataset or expression was requested
                req_filen.insert(0, '')
                filen = req_filen
                req_datasetname.insert(0, args.datasetname)
                args.datasetname = req_datasetname


    # print("Plotting with file '"+filen+"' and dataset '"+args.datasetname+"'...")

    # make a new flashplotlib object
    fpl = flashplotlib()
    fpl.data_transform = args.data_transform
    fpl.outname = args.outname if not args.outname else args.outdir+'/'+args.outname
    fpl.outtype = args.outtype
    fpl.verbose = args.verbose
    try: unitsystem; fpl.unitsystem=unitsystem
    except: pass
    fpl.facecolor = args.facecolor
    fpl.colorbar = args.colorbar
    fpl.cmap = args.cmap
    fpl.cmap_label = args.cmap_label
    fpl.cmap_format = args.cmap_format
    fpl.vmin = args.vmin
    fpl.vmax = args.vmax
    fpl.plotlabel = args.plotlabel
    fpl.labels_inside = args.labels_inside
    fpl.axes_label = args.axes_label
    fpl.axes_unit = args.axes_unit
    fpl.axes_format = args.axes_format
    fpl.axes_transform = args.axes_transform
    fpl.time_unit = args.time_unit
    fpl.time_scale = args.time_scale
    fpl.time_format = args.time_format
    fpl.time_transform = args.time_transform
    fpl.pixel = args.pixel
    fpl.shift = args.shift
    fpl.shift_periodic = args.shift_periodic
    fpl.boundary = args.boundary
    fpl.gauss_smooth_fwhm = args.gauss_smooth_fwhm
    fpl.scale_sinks_by_mass = args.scale_sinks_by_mass
    fpl.show_sinks = args.show_sinks
    fpl.show_tracers = args.show_tracers
    fpl.particle_color = args.particle_color
    fpl.particle_size_factor = args.particle_size_factor
    fpl.particle_mark_tags = args.particle_mark_tags
    fpl.vec = args.vec
    fpl.vec_transform = args.vec_transform
    fpl.vec_n = args.vec_n
    fpl.vec_color = args.vec_color
    fpl.vec_unit = args.vec_unit
    fpl.vec_scale = args.vec_scale
    fpl.vec_scale_factor = args.vec_scale_factor
    fpl.vec_var = args.vec_var
    fpl.vec_key = args.vec_key
    fpl.stream = args.stream
    fpl.stream_transform = args.stream_transform
    fpl.stream_vmin = args.stream_vmin
    fpl.stream_thick = args.stream_thick
    fpl.stream_n = args.stream_n
    fpl.stream_color = args.stream_color
    fpl.stream_var = args.stream_var
    fpl.convert2proper = args.convert2proper
    fpl.show_redshift = args.show_redshift
    if args.show_blocks or args.show_grid:
        fpl.nb = nb # number of cells per block
        fpl.bb = bb # block bounding boxes
        fpl.rl = rl # block refinement level
        if args.show_blocks: fpl.show_blocks = args.show_blocks
        if args.show_grid: fpl.show_grid = args.show_grid
    if args.nolog: fpl.log = False
    if args.lowres: args.dpi = 100
    # if dpi is None, we set dpi based on pixels, such that if map_only, we get exactly the pixel resolution
    if args.dpi is None:
        if args.pixel is not None: pixel_for_dpi = args.pixel[0]
        else: pixel_for_dpi = 1024
        args.dpi = pixel_for_dpi / 1000.0 * 259.8
    if args.fontsize is not None: rcParams['font.size'] *= args.fontsize # scale to requested fontsize
    # make a map plot
    if args.p3d is None: # 2D plotting
        fpl.prep_map(filen, args.datasetname) # read data
        fpl.plot_map(map_only=args.map_only, dpi=args.dpi) # make 2D plot
    else: # 3D plotting
        if 'slice' in args.p3d: slice_3d = True
        else: slice_3d = False
        if 'vol' in args.p3d: volume = True
        else: volume = False
        if 'stream' in args.p3d: stream_3d = True
        else: stream_3d = False
        if 'axes' in args.p3d_options: axes = True
        else: axes = False
        if 'triad' in args.p3d_options: triad = True
        else: triad = False
        fpl.plot_3d(filen, args.datasetname, slice=slice_3d, volume=volume, streamlines=stream_3d, proj_range=args.range, data_range=args.data_range,
            pixel=args.pixel, ncpu=args.ncpu, zoom=args.zoom, outdir=args.outdir, mass_weighting=args.mw, mlab_axes=axes, mlab_triad=triad, mlab_cbar=True,
            vmin=args.vmin, vmax=args.vmax, view=args.p3d_view, move=args.p3d_move, map_only=False, outline=0, dpi=args.dpi)
    # restore original args.datasetname
    args.datasetname = datasetname_saved
    return fpl
# ======================= end: process_file ===========================


# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":
    # time the script
    start_time = timeit.default_timer()
    # parse arguments and check if a file was specified
    args = parse_args(process_args_locally=True)
    if args.filename is None:
        print('No inputfile supplied.', warn=True)
        try:
            args.filename = [sorted(glob.glob("*chk_????"))[-1]]
            print('Using the most recent checkpoint file available: %s'%args.filename[0])
        except:
            print('Cannot find any checkpoint files in this directory. Please supply input.')
            args.parser.print_usage()
            exit(1)

    # loop over files
    for filen in args.filename:
        # parse arguments to restore original args
        args = parse_args(process_args_locally=True)
        # process file
        process_file(filen, args)
    # time the script
    total_time = end_time - start_time
    if args.verbose: print("***************** time to finish = "+str(total_time)+"s *****************")
