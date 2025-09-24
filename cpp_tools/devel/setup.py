#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2025

from cfpack import print, stop
import cfpack as cfp
import argparse
import glob
import os

##### MAIN #####

# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Makefile automator.')
    parser.add_argument("-compile", "--compile", action='store_true', help="compile", default=False)
    parser.add_argument("-test", "--test", action='store_true', help="test run", default=False)
    parser.add_argument("-clean", "--clean", action='store_true', help="clean", default=False)
    args = parser.parse_args()

    # get all relevant tools directories
    subdirs = sorted([entry.path for entry in os.scandir('../') if entry.is_dir() and not entry.name.startswith('.')])
    cwd = os.getcwd() # current working directory

    for dir in subdirs:

        print("Working on === "+dir[3:]+" ===", color='green')

        # copy template Makefile and set bin(s) names (executables to be created)
        cpp_files = sorted(glob.glob(dir+"/*.cpp")) # get all .cpp files

        # only do this if it's a source code folder (e.g., 'Libs' is excluded by this condition)
        if len(cpp_files) > 0:

            bins_list = [f.split('/')[-1][:-4] for f in cpp_files]
            bins = " ".join(bins_list)
            cfp.run_shell_command('cp Makefile '+dir) # copy template Makefile
            cfp.replace_line_in_file(dir+"/Makefile", "BINS = [BINS]", "BINS = "+bins)

            # deal with Makefile (library) requirements
            requires_fftw = False
            make_req_file = sorted(glob.glob(dir+"/Makefile.requires"))
            if len(make_req_file) > 0:
                make_req_file = make_req_file[0]
                if len(cfp.find_line_in_file(make_req_file, "fftw")) > 0:
                    requires_fftw = True

            if not requires_fftw:
                cfp.replace_line_in_file(dir+"/Makefile", "FFTW", "", search_str_position=None, debug=False)

            if args.compile:
                cfp.run_shell_command('cd '+dir+'; make clean; make; cd '+cwd)

            if args.test:
                for bin in bins_list:
                    cfp.run_shell_command('cd '+dir+'; ./'+bin+'; cd '+cwd)

            if args.clean:
                cfp.run_shell_command('cd '+dir+'; make clean; cd '+cwd)
