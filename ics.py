#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath and Bella Gerrard, 2020-2025

from cfpack.defaults import *
import cfpack as cfp
import numpy as np
import argparse

# === get_ics ===
def get_ics(resolution=None, lrefine_max=1, nxb=8, nblockx=1, L=1.2e17, domain='sphere', R=5.0e16, H=0.0,
            rho=3.82e-18, sigma_v=1e-99, Omega=1.86e-13, B=100.e-6, mu=2.3, quiet=False):

    if not quiet:
        print('=== Overview of initial conditions based on input parameters ===:')
        print(' Size of box (L) = ', '{:5.4e}'.format(L), ' cm (', '{:5.4}'.format(L/au), 'AU)')
        print(' Domain type: ', domain)
    V = 0.0 # volume
    if domain == 'sphere':
        V = 4.0/3.0*np.pi * R**3
        if not quiet: print(' Radius of sphere (R) = ', '{:5.4e}'.format(R), ' cm (', '{:5.4}'.format(R/au), 'AU)')
    if domain == 'cylinder':
        V = np.pi*R**2 * 2.0*H
        if not quiet: print(' Radius of cylinder (R) = ', '{:5.4e}'.format(R), ' cm (', '{:5.4}'.format(R/au), 'AU)')
        if not quiet: print(' Half height of cylinder (H) = ', '{:5.4e}'.format(H), ' cm (', '{:5.4}'.format(H/au), 'AU)')
    if domain == 'box':
        V = L**3
        R = ( V / (4.0*np.pi/3.0) )**(1.0/3.0)
    if not quiet:
        print(' *Effective* radius from box volume = ', '{:5.4e}'.format(R), ' cm (', '{:5.4}'.format(R/au), 'AU)')
        print(' Mean molecular weight (mu) = ', '{:5.4e}'.format(mu))
        print(' Density (rho) = ', '{:5.4e}'.format(rho), ' g/cm^3 (', '{:5.4e}'.format(rho/(mu*m_p)), ' cm^-3)')
        print(' Angular frequency (Omega) = ', '{:5.4e}'.format(Omega), ' rad/s')
        print(' Magnetic field strength (B) = ', '{:5.4e}'.format(B), ' G')

    # mass
    M = rho*V
    # virial parameter
    alpha_vir = 5.0*sigma_v**2*L / (6.0*g_n*M)
    # freefall time
    t_ff = np.sqrt(3.0*np.pi/32.0/g_n/rho)
    # Mach number
    c_s = 0.2e5 # assume sound speed at L and rho to be 0.2 km/s (also assumed in call to cfp.polytropic_eos)
    Mach = sigma_v / c_s
    # gravitational energy
    E_grav = 3.0/5.0*g_n*M**2/R
    # turbulent energy
    E_turb = 0.5*M*sigma_v**2
    # rotational energy
    E_rot = 0.5*2.0/5.0*M*Omega**2*R**2
    # magnetic energy
    E_mag = 1.0/(8.0*np.pi)*B**2*V
    # Alfven speed
    AlfvenSpeed = B / np.sqrt(4*np.pi*rho)
    # Alfven Mach number
    AlfvenMach = np.inf
    # Plasma beta
    PlasmaBeta = np.inf
    # mass to flux ratio
    MoverPhiCrit = 0.53/(3.0*np.pi)*np.sqrt(5.0/g_n)
    MoverPhi = np.inf
    if B > 0:
        MoverPhi = M / (B*np.pi*R**2)
        AlfvenMach = sigma_v / AlfvenSpeed
        PlasmaBeta = 2 * c_s**2 / AlfvenSpeed**2

    if not quiet:
        print('=== Derived cloud properties ===:')
        print(' Mass (M) = ', '{:5.4e}'.format(M), ' g (', '{:5.4e}'.format(M/m_sol), ' M_sol)')
        if domain == 'box':
            print(' --- The following values are for a sphere of *effective* radius (see above), and only approximate for a box domain ---')
        print(' Virial parameter (isolated, spherical approximation) = ', '{:5.4}'.format(alpha_vir))
        print(' Number of Jeans masses (spherical Jeans mass approximation) = ', '{:5.4}'.format(M/cfp.MJ(rho, c_s)))
        print(' Freefall time (t_ff) = ', '{:5.4e}'.format(t_ff), ' s (', '{:5.4e}'.format(t_ff/year), ' yr)')
        print(' Turbulent Mach number = ', '{:5.4}'.format(Mach))
        print(' E_grav        = ', '{:5.4}'.format(E_grav))
        print(' E_turb        = ', '{:5.4}'.format(E_turb))
        print(' E_rot         = ', '{:5.4}'.format(E_rot))
        print(' E_mag         = ', '{:5.4}'.format(E_mag))
        print(' E_turb/E_grav = ', '{:5.4}'.format(E_turb/E_grav))
        if (E_rot > 0): print(' E_turb/E_rot  = ', '{:5.4}'.format(E_turb/E_rot))
        print(' E_rot/E_grav  = ', '{:5.4}'.format(E_rot/E_grav))
        print(' E_mag/E_grav  = ', '{:5.4}'.format(E_mag/E_grav))
        print(' Omega*t_ff    = ', '{:5.4}'.format(Omega*t_ff))
        print(' Alfven speed  = ', '{:5.4e}'.format(AlfvenSpeed), ' cm/s (', AlfvenSpeed/1e5, 'km/s)')
        print(' Alfven Mach number = ', '{:5.4}'.format(AlfvenMach))
        print(' Plasma beta        = ', '{:5.4}'.format(PlasmaBeta))
        print(' Mass-to-flux ratio = ', '{:5.4}'.format(MoverPhi/MoverPhiCrit))

    # get minimum cell size (dx)
    if resolution is not None:
        neff = resolution
    else:
        n_cells_on_level_1 = nblockx*nxb
        q = np.log(n_cells_on_level_1)/np.log(2)
        neff = 2**(q+lrefine_max-1)
    dx = L/neff

    if not quiet:
        print('=== Numerical resolution parameters ===:')
        print(' Grid resolution (maximum effective uniform) = ', neff)
        print(' Min. cell size (dx) = ', '{:5.4e}'.format(dx), ' cm (', '{:5.4}'.format(dx/au), 'AU)')
        if resolution is None:
            print(' AMR specific:')
            print('  Maximum refinement level (lrefine_max) = ', lrefine_max)
            print('  Number of cells per block per dimension (nxb) = ', nxb)
            print('  Number of blocks per dimension on base grid (nblockx) = ', nblockx)
            print('  Number of cells per dimension on level 1 (base grid) = ', n_cells_on_level_1)

    r_sink = 2.5*dx
    r_outflow = 16.0*dx
    v_outflow = 100e5 # 100 km/s

    if not quiet:
        print('=== Derived sink particle properties ===:')
        print(' Accretion radius (r_sink) = ', '{:5.4e}'.format(r_sink), ' cm (', '{:5.4}'.format(r_sink/au), 'AU)')
        print(' Outflow radius (r_outflow) = ', '{:5.4e}'.format(r_outflow), ' cm (', '{:5.4}'.format(r_outflow/au), 'AU)')
        print(' Outflow velocity (v_outflow) = ', '{:5.4e}'.format(v_outflow), ' cm/s (', v_outflow/1e5, 'km/s)')
        print(' Outflow_time_char = ', '{:5.4e}'.format(r_outflow/v_outflow), ' s (', '{:5.4}'.format(r_outflow/v_outflow/year), 'yr)')
        print(' Outflow_dt = ', '{:5.4e}'.format(dx/v_outflow), ' s (', '{:5.4}'.format(dx/v_outflow/year), 'yr)')

    # get the sink density threshold (needs to call eos, because the sound speed c_s depends on density)

    log_denstry_min = np.log10(1e-24)
    log_denstry_max = 1.0
    density_try = 10**(np.linspace(0,200000,num=200001)/199999.*(log_denstry_max-log_denstry_min)+log_denstry_min)
    sink_denstry = np.zeros(len(density_try))

    for i in range(len(density_try)-1):
        sink_denstry[i] = np.pi*(cfp.polytropic_eos(density_try[i],mu).cs)**2/4.0/g_n/r_sink**2

    index = np.where(density_try >= sink_denstry)[0]
    sink_dens = density_try[index[0]]
    m_sink = sink_dens*4.0*np.pi/3.0*r_sink**3

    if not quiet:
        print(' Density threshold (sink_dens) = ', '{:5.4e}'.format(sink_dens), ' g cm^-3 (', '{:5.4e}'.format(sink_dens/(mu*m_p)), ' cm^-3)')
        print(' M_sink (mass of gas at sink_dens and size r_sink) = ', '{:5.4e}'.format(m_sink), ' g (', '{:5.4e}'.format(m_sink/m_sol), ' M_sol)')
        print(' Sound speed c_s at sink_dens = ', '{:5.4e}'.format(cfp.polytropic_eos(sink_dens,mu).cs),' cm/s')

    # fill up dictionary for return
    ret = {"mass": M}
    ret["freefall_time"] = t_ff
    ret["virial_parameter"] = alpha_vir
    ret["sink_radius"] = r_sink
    ret["sink_density"] = sink_dens
    ret["resolution"] = neff

    return ret

# === end: get_ics ===

# ===== MAIN ====
# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Provides summary of cloud initial conditions.')
    parser.add_argument("-resolution", "--resolution", type=int,
                        help="(maximum effective uniform) grid resolution (if set, lrefine_max, nxb, nblockx, are ignored)")
    parser.add_argument("-lrefine_max", "--lrefine_max", type=int, default=1, help="maximum refinement level")
    parser.add_argument("-nxb", "--nxb", type=int, default=8, help="number of cells per block per dimension")
    parser.add_argument("-nblockx", "--nblockx", type=int, default=1, help="number of blocks per dimension on base grid")
    parser.add_argument("-L", "--L", type=float, default=1.2e17, help="size of box")
    parser.add_argument("-domain", "--domain", type=str, choices=['sphere', 'cylinder', 'box'],
                        help="domain shape (sphere, cylinder, or box; H applies only to cylinder; for box only L)", default='sphere')
    parser.add_argument("-R", "--R", type=float, default=5.0e16, help="radius of sphere")
    parser.add_argument("-H", "--H", type=float, default=3.31e16, help="half height of cylinder")
    parser.add_argument("-rho", "--rho", type=float, default=3.82e-18, help="density")
    parser.add_argument("-sigma_v", "--sigma_v", type=float, default=0.0, help="turbulent velocity dispersion")
    parser.add_argument("-Omega", "--Omega", type=float, default=0.0, help="angular frequency")
    parser.add_argument("-B", "--B", type=float, default=0.0, help="magnetic field")
    parser.add_argument("-mu", "--mu", type=float, default=2.3, help="mean molecular weight")
    args = parser.parse_args()

    get_ics(resolution=args.resolution, lrefine_max=args.lrefine_max, nxb=args.nxb, nblockx=args.nblockx,
            L=args.L, domain=args.domain, R=args.R, H=args.H, rho=args.rho, sigma_v=args.sigma_v, Omega=args.Omega,
            B=args.B, mu=args.mu, quiet=False)
