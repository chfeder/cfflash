#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2019-2024

import astropy.io.ascii as ap_ascii
import astropy.table as ap_table
import numpy as np
import argparse
import cfpack as cfp
from cfpack import print, stop
from scipy.special import iv, erfc


# =============================================================
def Pkin(k, pkin, kbn, pbn, knu, pnu):
    ret = ( (k/kbn)**pkin + (k/kbn)**pbn ) * np.exp(-(k/knu)**pnu)
    return ret

# =============================================================
def cauchy_pdf(s, s0=0.0, sigma_s=1.0, gamma=1.0):
    s = np.array(s)
    # turn into list in case of single value input
    if s.size == 1:
        s = [s]
        s = np.array(s)
    # Cauchy distribution
    cauchy = 1 / ( np.pi*gamma * (1.0 + (s-s0)**2/gamma**2 ) )
    # Normalisation of Cauchy * Normal distribution (see Mathematica integration)
    norm = np.exp(-gamma**2/(2*sigma_s**2)) * np.sqrt(2*np.pi) * sigma_s / erfc(gamma/np.sqrt(2)/sigma_s)
    stop()
    # PDF of Cauchy * Normal distribution
    pdf = norm * cauchy * normal_pdf(s, s0=s0, sigma_s=sigma_s, constrained=False)
    return pdf

# =============================================================
def cauchy_pdf_log(s, s0=0.0, sigma_s=1.0, gamma=1.0):
    return np.log(cauchy_pdf(s, s0, sigma_s, gamma))

# =============================================================
def kurt_pdf_log(s, p0=1.0, p1=1.0, p2=1.0, p3=1.0, p4=1.0):
    s = np.array(s)
    # turn into list in case of single value input
    if s.size == 1:
        s = [s]
        s = np.array(s)
    ret = p0 + p1*s + p2*s**2 + p3*s**3 + p4*s**4
    return ret

# =============================================================
def kurt_pdf(s, p0=1.0, p1=1.0, p2=1.0, p3=1.0, p4=1.0):
    log_pdf = kurt_pdf_log(s, p0, p1, p2, p3, p4)
    return np.exp(log_pdf)

# =============================================================
def normal_pdf(s, s0=0.0, sigma_s=1.0, constrained=True):
    if constrained: s0 = -0.5*sigma_s**2
    ret = 1.0/np.sqrt(2.0*np.pi*sigma_s**2) * np.exp(-0.5*(s-s0)**2/sigma_s**2)
    return ret

# =============================================================
def hopkins_pdf(s, sigma_s=1.0, theta=0.1):
    log_pdf = hopkins_pdf_log(s, sigma_s, theta)
    return np.exp(log_pdf)

# ============ see Hopkins (2013) =============================
def hopkins_pdf_log(s, sigma_s=1.0, theta=0.1):
    if theta < 1e-7: theta = 1e-7 # approximate limit of zero intermittency
    s = np.array(s)
    # turn into list in case of single value input
    if s.size == 1:
        s = [s]
        s = np.array(s)
    lamb = sigma_s**2 / (2.0*theta**2)
    u = -s/theta + lamb / (1.0+theta)
    # init return values
    ret = np.zeros(s.size) - 9999999999
    for i in range(0, s.size):
        # only work on u > 0 points
        if u[i] > 0:
            log_bessel = -9999999999
            # the argument of the 1st-order modified Bessel function of the 1st kind
            arg_bessel = 2.0*np.sqrt(lamb*u[i])
            if arg_bessel < 700: # we call the scipy iv function
                log_bessel = np.log( iv(1, arg_bessel) )
            else: # approximate log(iv) for large argument
                log_bessel = arg_bessel - np.log( np.sqrt(2*np.pi*arg_bessel) )
            # now define the log(PDF)
            ret[i] = log_bessel - lamb - u[i] + np.log(np.sqrt(lamb/u[i]/theta**2))
    return ret

# =============================================================
def squire_pdf(s, kappa=0.1, xi=1.5, L_over_l=1.0):
    log_pdf = squire_pdf_log(s, kappa, xi, L_over_l)
    return np.exp(log_pdf)

# =========== see Squire & Hopkins (2007) =====================
def squire_pdf_log(s, kappa=0.1, xi=1.5, L_over_l=2.0):
    # kappa = theta (from hopkins_pdf)
    if kappa < 1e-7: kappa = 1e-7 # approximate limit of zero intermittency
    s = np.array(s)
    # turn into list in case of single value input
    if s.size == 1:
        s = [s]
        s = np.array(s)
    lamb = xi * (1.0 + 1.0/kappa) * np.log(L_over_l)
    u = -s/kappa + lamb / (1.0+kappa)
    # init return values
    ret = np.zeros(s.size) - 9999999999
    for i in range(0, s.size):
        # only work on u > 0 points
        if u[i] > 0:
            log_bessel = -9999999999
            # the argument of the 1st-order modified Bessel function of the 1st kind
            arg_bessel = 2.0*np.sqrt(lamb*u[i])
            if arg_bessel < 700: # we call the scipy iv function
                log_bessel = np.log( iv(1, arg_bessel) )
            else: # approximate log(iv) for large argument
                log_bessel = arg_bessel - np.log( np.sqrt(2*np.pi*arg_bessel) )
            # now define the log(PDF)
            ret[i] = log_bessel - lamb - u[i] + np.log(np.sqrt(lamb/u[i]/kappa**2))
    return ret


# =============================================================
def read_pdf(filename, verbose=1):
    # open file
    if verbose: print("reading from file '"+filename+"'...")
    f = open(filename, 'r')
    # read headers
    head_mean = f.readline()
    linespl = f.readline().split()
    mean = float(linespl[0])
    head_rms = f.readline()
    linespl = f.readline().split()
    rms = float(linespl[0])
    head_skew = f.readline()
    linespl = f.readline().split()
    skew = float(linespl[0])
    head_kurt = f.readline()
    linespl = f.readline().split()
    kurt = float(linespl[0])
    head_sigma = f.readline()
    linespl = f.readline().split()
    sigma = float(linespl[0])
    head_tab = f.readline()
    header = {'tab_str': head_tab, 'mean_str' : head_mean , 'mean' : mean,  'mean_sigma'  : 0.0,
                                   'rms_str'  : head_rms  , 'rms'  : rms,   'rms_sigma'   : 0.0,
                                   'skew_str' : head_skew , 'skew' : skew,  'skew_sigma'  : 0.0,
                                   'kurt_str' : head_kurt , 'kurt' : kurt,  'kurt_sigma'  : 0.0,
                                   'sigma_str': head_sigma, 'sigma': sigma, 'sigma_sigma' : 0.0}
    # read the table data
    tab = f.read()
    data = ap_ascii.read(tab)
    n_rows = len(data)
    n_cols = len(data.columns)
    if verbose > 1: print('n_rows = ', n_rows)
    if verbose > 1: print('n_cols = ', n_cols)
    f.close()

    return data, header


# =============================================================
def write_pdf(filename, data, header, verbose=1):
    # write header first
    f = open(filename, mode='w')
    f.write('mean +/- sigma'+'\n')
    f.write(str(header['mean'])+' '+str(header['mean_sigma'])+'\n')
    f.write('rms +/- sigma'+'\n')
    f.write(str(header['rms'])+' '+str(header['rms_sigma'])+'\n')
    f.write('skew +/- sigma'+'\n')
    f.write(str(header['skew'])+' '+str(header['skew_sigma'])+'\n')
    f.write('kurt +/- sigma'+'\n')
    f.write(str(header['kurt'])+' '+str(header['kurt_sigma'])+'\n')
    f.write('sigma +/- sigma'+'\n')
    f.write(str(header['sigma'])+' '+str(header['sigma_sigma'])+'\n')
    f.close()
    # append data table to file
    f = open(filename, mode='a')
    # new astro py table with header neames
    ap_tab = ap_table.Table(data.transpose(), names=header['tab_str'].split())
    # write the table
    ap_tab.write(f, format='ascii.fixed_width', delimiter=' ')
    if verbose: print('write_pdf: '+filename+' written.', highlight=1)
    f.close()


# =============================================================
def aver_pdf(inputfiles, verbose=1):
    # read first file and allocate averaged-data container
    data, header = read_pdf(inputfiles[0])
    n_rows = len(data)
    aver_dat = np.zeros([5,n_rows])
    aver_dat[0] = data['col1'] # fill up grid variable
    header_aver = header # copy header; then overwrite
    header_aver['mean']  = 0.0
    header_aver['rms']   = 0.0
    header_aver['skew']  = 0.0
    header_aver['kurt']  = 0.0
    header_aver['sigma'] = 0.0
    # loop over all files and accumulate
    # TODO: option for log_average
    n = 0
    for filename in inputfiles:
        data, header = read_pdf(filename)
        # header content
        header_aver['mean']        += header['mean']
        header_aver['mean_sigma']  += header['mean']**2
        header_aver['rms']         += header['rms']
        header_aver['rms_sigma']   += header['rms']**2
        header_aver['skew']        += header['skew']
        header_aver['skew_sigma']  += header['skew']**2
        header_aver['kurt']        += header['kurt']
        header_aver['kurt_sigma']  += header['kurt']**2
        header_aver['sigma']       += header['sigma']
        header_aver['sigma_sigma'] += header['sigma']**2
        # check grid variable is all the same in all files
        if np.any(aver_dat[0] != data['col1']):
            print('aver_pdf: ERROR; something wrong with grid variable alignment', error=True)
        # tab content
        aver_dat[1] += data['col3'] # PDF
        aver_dat[2] += data['col3']**2
        aver_dat[3] += data['col4'] # CDF
        aver_dat[4] += data['col4']**2
        n += 1
    # now finalise the mean
    # header content
    header_aver['mean']        /= n
    header_aver['mean_sigma']  /= n
    header_aver['rms']         /= n
    header_aver['rms_sigma']   /= n
    header_aver['skew']        /= n
    header_aver['skew_sigma']  /= n
    header_aver['kurt']        /= n
    header_aver['kurt_sigma']  /= n
    header_aver['sigma']       /= n
    header_aver['sigma_sigma'] /= n
    header_aver['mean_sigma']  = np.sqrt(header_aver['mean_sigma']  - header_aver['mean']**2)
    header_aver['rms_sigma']   = np.sqrt(header_aver['rms_sigma']   - header_aver['rms']**2)
    header_aver['skew_sigma']  = np.sqrt(header_aver['skew_sigma']  - header_aver['skew']**2)
    header_aver['kurt_sigma']  = np.sqrt(header_aver['kurt_sigma']  - header_aver['kurt']**2)
    header_aver['sigma_sigma'] = np.sqrt(header_aver['sigma_sigma'] - header_aver['sigma']**2)
    # tab content
    aver_dat[1:5] /= n
    # compute log sigmas
    aver_dat[2] = cfp.get_sigma(aver_dat[1], aver_dat[2])
    aver_dat[4] = cfp.get_sigma(aver_dat[3], aver_dat[4])
    # new table header for columns
    header_aver['tab_str'] = 'variable PDF sigma_PDF CDF sigma_CDF'
    if verbose: print('aver_pdf: averaged over '+str(n)+' files', highlight=2)
    return aver_dat, header_aver


# =============================================================
def read_spect(filename, verbose=1):
    # open file
    if verbose: print("reading from file '"+filename+"'...")
    f = open(filename, 'r')
    # read headers
    head_ptot = 'dummy entry\n'; ptot = 0.0; ptot_sigma = 0.0
    head_plgt = 'dummy entry\n'; plgt = 0.0; plgt_sigma = 0.0
    first_line = f.readline()
    if '#00_BinIndex' in first_line: # it is a non-decomposed spectrum
        head_tab = first_line
    else: # it is a Helmholtz-decomposed spectrum (so we have extra header information to read)
        head_ptot = first_line
        linespl = f.readline().split()
        ptot = float(linespl[0])
        if len(linespl) == 1:
            ptot_sigma = 0.0
        else:
            ptot_sigma = float(linespl[1])
        head_plgt = f.readline()
        linespl = f.readline().split()
        plgt = float(linespl[0])
        if len(linespl) == 1:
            plgt_sigma = 0.0
        else:
            plgt_sigma = float(linespl[1])
        f.readline() # empty line
        head_tab = f.readline()
    # define output header dict
    header = {'tab_str': head_tab, 'ptot_str': head_ptot, 'ptot': ptot, 'ptot_sigma': ptot_sigma,
                                   'plgt_str': head_plgt, 'plgt': plgt, 'plgt_sigma': plgt_sigma}
    # read the table data
    tab = f.read()
    data = ap_ascii.read(tab)
    n_rows = len(data)
    n_cols = len(data.columns)
    if verbose > 1: print('n_rows = ', n_rows)
    if verbose > 1: print('n_cols = ', n_cols)
    f.close()
    return data, header


# =============================================================
def write_spect(filename, data, header, verbose=1):
    # write header first
    f = open(filename, mode='w')
    f.write(header['ptot_str'])
    f.write(str(header['ptot'])+' '+str(header['ptot_sigma'])+'\n')
    f.write(header['plgt_str'])
    f.write(str(header['plgt'])+' '+str(header['plgt_sigma'])+'\n')
    f.write('\n')
    f.close()
    # append data table to file
    f = open(filename, mode='a')
    # new astro py table with header neames
    ap_tab = ap_table.Table(data.transpose(), names=header['tab_str'].split())
    # write the table
    ap_tab.write(f, format='ascii.fixed_width', delimiter=' ')
    if verbose: print('write_spect: '+filename+' written.', highlight=1)
    f.close()


# =============================================================
def aver_spect(inputfiles, normalise=False, verbose=1):
    # read first file and allocate averaged-data container
    data, header = read_spect(inputfiles[0])
    n_rows = len(data)
    # check if this was a Helmholtz-decomposed dataset
    if 'col12' in data.keys():
        decomposed = True
        ptot_col = 'col16'
    else:
        decomposed = False
        ptot_col = 'col8'
    aver_dat = np.zeros([7,n_rows])
    aver_dat[0] = data['col2'] # fill up k
    header_aver = header # copy header; then overwrite
    header_aver['ptot']       = 0.0
    header_aver['ptot_sigma'] = 0.0
    header_aver['plgt']       = 0.0
    header_aver['plgt_sigma'] = 0.0
    # loop over all files and accumulate log_10(data)
    # we are computing a log-average
    n = 0
    if normalise: print("Normalising spectra to total power = 1 during averaging!", highlight=3)
    for filename in inputfiles:
        data, header = read_spect(filename)
        # normalisation = 1 (default) or by the total power of each spectrum (read from header p_tot), if requested
        norm = 1.0
        if normalise: norm = header['ptot']
        # header content
        header_aver['ptot']       += (header['ptot']/norm)
        header_aver['ptot_sigma'] += (header['ptot']/norm)**2
        header_aver['plgt']       += (header['plgt']/norm)
        header_aver['plgt_sigma'] += (header['plgt']/norm)**2
        # check k is all the same in all files
        if np.any(aver_dat[0] != data['col2']):
            print('aver_spect: ERROR; something wrong with k alignment', error=True)
        # tab content
        if decomposed:
            aver_dat[1] += np.log10(data['col12']/norm) # lgt spect
            aver_dat[2] += np.log10(data['col12']/norm)**2
            aver_dat[3] += np.log10(data['col14']/norm) # trv spect
            aver_dat[4] += np.log10(data['col14']/norm)**2
        aver_dat[5] += np.log10(data[ptot_col]/norm) # tot spect
        aver_dat[6] += np.log10(data[ptot_col]/norm)**2
        n += 1
    # now finalise the mean
    # header content
    header_aver['ptot']       /= n
    header_aver['ptot_sigma'] /= n
    header_aver['plgt']       /= n
    header_aver['plgt_sigma'] /= n
    header_aver['ptot_sigma'] = np.sqrt(header_aver['ptot_sigma'] - header_aver['ptot']**2)
    header_aver['plgt_sigma'] = np.sqrt(header_aver['plgt_sigma'] - header_aver['plgt']**2)
    # tab content
    aver_dat[1:7] /= n
    # compute log sigmas
    aver_dat[2] = cfp.get_sigma(aver_dat[1], aver_dat[2])
    aver_dat[4] = cfp.get_sigma(aver_dat[3], aver_dat[4])
    aver_dat[6] = cfp.get_sigma(aver_dat[5], aver_dat[6])
    # new table header for columns
    header_aver['tab_str'] = 'k log10_Plgt sigma_log10_Plgt log10_Ptrv sigma_log10_Ptrv log10_Ptot sigma_log10_Ptot'
    if verbose: print('aver_spect: averaged over '+str(n)+' files', highlight=2)
    return aver_dat, header_aver


# =============================================================
def read_strufu(filename, verbose=1):
    # open file
    if verbose: print("reading from file '"+filename+"'...")
    f = open(filename, 'r')
    # read headers
    head_tab = f.readline()
    header = {'tab_str': head_tab}
    # read the table data
    tab = f.read()
    data = ap_ascii.read(tab)
    n_rows = len(data)
    n_cols = len(data.columns)
    if verbose > 1: print('n_rows = ', n_rows)
    if verbose > 1: print('n_cols = ', n_cols)
    f.close()
    return data, header


# =============================================================
def write_strufu(filename, data, header, verbose=1):
    # write header first
    f = open(filename, mode='w')
    # new astro py table with header neames
    ap_tab = ap_table.Table(data.transpose(), names=header['tab_str'].split())
    # write the table
    ap_tab.write(f, format='ascii.fixed_width', delimiter=' ')
    if verbose: print('write_strufu: '+filename+' written.', highlight=1)
    f.close()


# =============================================================
def aver_strufu(inputfiles, log_average=False, verbose=1):
    # print info about which averaging scheme is used
    aver_type = 'linear'
    if log_average: aver_type = 'logarithmic (log_10)'
    if verbose: print("computing "+aver_type+" average...", highlight=3)
    # read first file and allocate averaged-data container
    data, header = read_strufu(inputfiles[0])
    n_rows = len(data)
    n_cols = len(data.columns)
    aver_dat = np.zeros([n_cols,n_rows])
    aver_dat[0] = data['col1'] # fill up col 1
    aver_dat[1] = data['col2'] # fill up col 2
    aver_dat[2] = data['col3'] # fill up col 3
    # loop over all files and accumulate log_10(data)
    # we are computing a log-average
    n = 0
    for filename in inputfiles:
        data, header = read_strufu(filename)
        # check first 3 columns are all the same in all files
        if (np.any(aver_dat[0] != data['col1']) or np.any(aver_dat[1] != data['col2']) or np.any(aver_dat[2] != data['col3'])):
            print('aver_strufu: ERROR; something wrong in first 3 columns; one or more of them are not the same in all files.', error=True)
        # tab content
        for i in range(3,n_cols-3,4):
            value = data['col'+str(i+2)]
            if log_average: value = np.log10(value)
            aver_dat[i+0] += value**2 # lgt strufu sq
            value = data['col'+str(i+2)]
            if log_average: value = np.log10(value)
            aver_dat[i+1] += value    # lgt strufu
            value = data['col'+str(i+4)]
            if log_average: value = np.log10(value)
            aver_dat[i+2] += value**2 # trv strufu sq
            value = data['col'+str(i+4)]
            if log_average: value = np.log10(value)
            aver_dat[i+3] += value    # trv strufu
        n += 1
    # now finalise the mean
    # tab content
    aver_dat[3:n_cols+1] /= n
    # compute log sigmas
    for i in range(3,n_cols-3,4):
        aver_dat[i+0] = cfp.get_sigma(aver_dat[i+1], aver_dat[i+0])
        aver_dat[i+2] = cfp.get_sigma(aver_dat[i+3], aver_dat[i+2])
        # convert back to non-log form if needed (except for the sigmas)
        value = aver_dat[i+1]
        if log_average: value = 10**value
        aver_dat[i+1] = value
        value = aver_dat[i+3]
        if log_average: value = 10**value
        aver_dat[i+3] = value
    # prep header
    header_aver = header # first copy header; then overwrite some entries
    header_split = header_aver['tab_str'].split()
    sigma_str = "sigma"
    if log_average: sigma_str = "log_"+sigma_str
    for i in range(3,n_cols-3,4):
        header_split[i+0] = header_split[i+0].replace("NP", sigma_str)
        header_split[i+2] = header_split[i+2].replace("NP", sigma_str)
    header_aver['tab_str'] = " ".join(header_split)
    if verbose: print('aver_strufu: averaged over '+str(n)+' files', highlight=2)
    return aver_dat, header_aver


# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make plots.')
    parser.add_argument("inputfiles", nargs='+', type=argparse.FileType('r'), help="input data file(s) to average")
    parser.add_argument("-o", "--o", dest='outputfile', type=str, help="Name of the output file (default: %(default)s)", default='out.dat')
    parser.add_argument("-log_average", "--log_average", action='store_true', help="compute logarithmic (log10) average (currently only supported for SFs); \
                                                                                    PDFs are linearly averaged, spectra are log10-averaged", default=False)
    parser.add_argument("-normalise", "--normalise", action='store_true', help="divide by total power before averaging spectra", default=False)
    parser.add_argument("-verbose", "--verbose", type=int, default=1, help="verbose level")

    args = parser.parse_args()

    inputfiles = sorted([x.name for x in list(args.inputfiles)])

    # averaging PDFs
    if inputfiles[0].find('.pdf') != -1:
        aver_dat, header_aver = aver_pdf(inputfiles, verbose=args.verbose)
        write_pdf(args.outputfile, aver_dat, header_aver, verbose=args.verbose)

    # averaging spectra
    if inputfiles[0].find('_spect_') != -1:
        aver_dat, header_aver = aver_spect(inputfiles, normalise=args.normalise, verbose=args.verbose)
        write_spect(args.outputfile, aver_dat, header_aver, verbose=args.verbose)

    # averaging structure functions
    if inputfiles[0].find('_sf_') != -1:
        aver_dat, header_aver = aver_strufu(inputfiles, args.log_average, verbose=args.verbose)
        write_strufu(args.outputfile, aver_dat, header_aver, verbose=args.verbose)
