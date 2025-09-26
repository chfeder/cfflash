!!****if* source/Particles/ParticlesMain/Sink/Heating/Polar/Particles_sinkHeating
!!
!! NAME
!!
!!  Particles_sinkHeating
!!
!! SYNOPSIS
!!
!!  call Particles_sinkHeating()
!!
!! DESCRIPTION
!!
!!  Handles protostellar luminosity feedback (heating by radiation) from sink particles.
!!  It fills a global 3D grid variable SLHT_VAR with T_heat. This then needs to be applied
!!  or added in the respective EOS implementation used, in order to actually heat the gas
!!  to modify the gas pressure.
!!
!! ARGUMENTS
!!
!! NOTES
!!
!!   written by Sajay Sunny Mathew and Christoph Federrath, 2019
!!
!!***

subroutine Particles_sinkHeating()

    use Particles_sinkData
    use pt_sinkInterface, ONLY: pt_sinkFindList, pt_sinkGatherGlobal, &
                                pt_sinkCorrectForPeriodicBCs
    use Driver_interface, ONLY : Driver_abortFlash
    use Driver_data, ONLY : dr_globalMe, dr_simTime, dr_dt
    use RuntimeParameters_interface, ONLY : RuntimeParameters_get
    use Grid_interface, ONLY : Grid_getCellCoords, Grid_getBlkPhysicalSize, &
                               Grid_getBlkPtr, Grid_releaseBlkPtr, &
                               Grid_getBlkIndexLimits, Grid_getListOfBlocks, Grid_getMinCellSize
    use Cosmology_interface, ONLY : Cosmology_getRedshift
    use PhysicalConstants_interface, ONLY : PhysicalConstants_get
    use Timers_interface, ONLY : Timers_start, Timers_stop
    
    implicit none

#include "constants.h"
#include "Flash.h"
#include "Particles.h"

    real, save    :: sigma_sb, c, sb_factor
    real, save    :: delta_at_lrefmax, heating_time_char
    logical, save :: first_call = .true.

    integer, save :: sink_heating_model
    real, save    :: sink_heating_radius, sink_heating_inner_radius
    real, save    :: sink_heating_luminosity_fraction, sink_heating_tau0

    integer :: blockCount, lb, blockID, size_x, size_y, size_z, i, j, k, p
    real    :: heating_radius_comoving, redshift
    real    :: dx, dy, dz, radius, theta, lum, flux, angmom
    real    :: t_shape, tau, lx, ly, lz

    integer, dimension(MAXBLOCKS) :: blockList
    integer, dimension(2,MDIM) :: blkLimits, blkLimitsGC

    real, pointer, dimension(:,:,:,:) :: solnData
    real, dimension(:), allocatable :: xc, yc, zc

    integer, dimension(maxsinks) :: pindex_found
    integer :: np_found, npf

    real, parameter :: au = 1.496e13

#ifndef LUMINOSITY_PART_PROP
    real, parameter :: m_sol = 1.98855e33
    real, parameter :: r_sol = 6.957e10
    real, save :: eps_lum_g_factor
    real :: protostar_radius, accr_rate
    integer, parameter :: gather_nprops = 2
    integer, dimension(gather_nprops), save :: gather_propinds = &
       (/ integer :: CREATION_TIME_PART_PROP, OLD_PMASS_PART_PROP /)
#else
    integer, parameter :: gather_nprops = 3
    integer, dimension(gather_nprops), save :: gather_propinds = &
       (/ integer :: CREATION_TIME_PART_PROP, LUMINOSITY_PART_PROP, STELLAR_RADIUS_PART_PROP /)
#endif

    logical, parameter  :: debug = .false.

    ! ================= first call =================
    if (first_call) then

        call RuntimeParameters_get("sink_heating_model", sink_heating_model)
        call RuntimeParameters_get("sink_heating_radius", sink_heating_radius)
        call RuntimeParameters_get("sink_heating_luminosity_fraction", sink_heating_luminosity_fraction)
        call RuntimeParameters_get("sink_heating_tau0", sink_heating_tau0)

        ! limit inner radius based on maximum resolution below
        call Grid_getMinCellSize(delta_at_lrefmax)

        call PhysicalConstants_get("Stefan-Boltzmann", sigma_sb)
        call PhysicalConstants_get("speed of light", c)

        ! define sink heating constants
#ifndef LUMINOSITY_PART_PROP
        eps_lum_g_factor = sink_heating_luminosity_fraction * newton
#endif
        sb_factor = 1.0/(4.0*sigma_sb)

        if ((sink_heating_model .gt. 0) .and. (dr_globalMe == MASTER_PE)) &
            & print *, 'Particles_sinkHeating: activated and initialized.'

        first_call = .false.

    endif ! ================= end first call =================


    ! return if sink luminosity heating is off or if no sink particles present
    if (sink_heating_model .le. 0) return

    if (debug .and. (dr_globalMe == MASTER_PE)) print *, '[', dr_globalMe, '] Particles_sinkHeating: entering'

    call Timers_start("sinkHeating")

    call Cosmology_getRedshift(redshift)
    heating_radius_comoving = sink_heating_radius * (1.0 + redshift)

    ! characteristic heating time (for smooth start)
    ! note that with the default values, sink_heating_radius/c ~ 5e6s ~ 0.16yr;
    ! in most cases a smooth start is achieved based on the time step, so ramp up the
    ! luminosity feedback over a few time steps
    heating_time_char = max(sink_heating_radius/c, 2.0*dr_dt)

    ! Exchange particle information (only what's really needed)
    ! Note that we assume that pt_sinkGatherGlobal was called recently in pt_sinkDumpParticles,
    ! but that did not update LUMINOSITY_PART_PROP or OLD_PMASS_PART_PROP
    call pt_sinkGatherGlobal(gather_propinds, gather_nprops)

    call Grid_getListOfBlocks(LEAF, blockList, blockCount)

    ! Loop over blocks
    do lb = 1, blockCount

        blockID = blockList(lb)

        call Grid_getBlkPtr(blockID, solnData)

        call Grid_getBlkIndexLimits(blockID, blkLimits, blkLimitsGC)
        size_x = blkLimitsGC(HIGH,IAXIS)-blkLimitsGC(LOW,IAXIS) + 1
        size_y = blkLimitsGC(HIGH,JAXIS)-blkLimitsGC(LOW,JAXIS) + 1
        size_z = blkLimitsGC(HIGH,KAXIS)-blkLimitsGC(LOW,KAXIS) + 1

        allocate(xc(size_x))
        allocate(yc(size_y))
        allocate(zc(size_z))

        call Grid_getCellCoords(IAXIS, blockID, CENTER, .true., xc, size_x)
        call Grid_getCellCoords(JAXIS, blockID, CENTER, .true., yc, size_y)
        call Grid_getCellCoords(KAXIS, blockID, CENTER, .true., zc, size_z)

        ! loop over active cells (exclude guard cells)
        do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
          do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
            do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

                flux = 0.0

                ! find sink particles (create must be false, so we can use the global list)
                call pt_sinkFindList(xc(i), yc(j), zc(k), heating_radius_comoving, .false., pindex_found, np_found)

                if (np_found .GT. 0) then

                    do npf = 1, np_found

                        p = pindex_found(npf) ! this is the index in the global list, because we called pt_sinkFindList
                                              ! with create = .false.

                        ! compute relative distances
                        dx = particles_global(POSX_PART_PROP,p) - xc(i)
                        dy = particles_global(POSY_PART_PROP,p) - yc(j)
                        dz = particles_global(POSZ_PART_PROP,p) - zc(k)

                        if (sink_grav_boundary_type .eq. 2) call pt_sinkCorrectForPeriodicBCs(dx, dy, dz)

                        radius = sqrt(dx**2 + dy**2 + dz**2)

                        ! only compute heating inside sink_heating_radius
                        if (radius .le. sink_heating_radius) then

                            ! time shape function to make a smooth start of the luminosity feedback (linear ramp)
                            if ((dr_simTime-particles_global(CREATION_TIME_PART_PROP,p)) .lt. heating_time_char) then
                                t_shape = (dr_simTime-particles_global(CREATION_TIME_PART_PROP,p)) / heating_time_char
                            else
                                t_shape = 1.0
                            end if

#ifndef LUMINOSITY_PART_PROP
                            ! need to estimate the accretion luminosity; first need the protostar radius;
                            ! this can vary between 3 and 5 solar radii for low-mass stars
                            ! (Palla & Stahler 1992,1993; Robitaille et al. 2006)
                            if (particles_global(MASS_PART_PROP,p) .lt. m_sol) then
                                ! linear ramp for the protostar radius from 3 to 5 R_sol within the 0 < M < M_sol range
                                protostar_radius = (2.0*particles_global(MASS_PART_PROP,p)/m_sol+3.0)*r_sol
                            else
                                protostar_radius = 5.0*r_sol
                            end if

                            ! compute the accretion rate (after outflow!!!)
                            accr_rate = (particles_global(MASS_PART_PROP,p)-particles_global(OLD_PMASS_PART_PROP,p))/dr_dt

                            ! luminosity (see e.g., Palla & Stahler 1993); this contains a radiative efficiency factor
                            lum = eps_lum_g_factor * particles_global(MASS_PART_PROP,p) / protostar_radius * accr_rate
                            ! limit injection on small scales (dust sublimation at ~ 1 AU)
                            sink_heating_inner_radius = max(au, 2.0*delta_at_lrefmax, protostar_radius)
#else
                            ! if running proper stellar evolution, then we have the luminosity (protostellar and ZAMS)
                            ! appropriate for low- and high-mass stars
                            lum = particles_global(LUMINOSITY_PART_PROP,p)
                            ! limit injection on small scales (dust sublimation at ~ 1 AU)
                            sink_heating_inner_radius = max(au, 2.0*delta_at_lrefmax, particles_global(STELLAR_RADIUS_PART_PROP,p))
#endif
                            ! compute angular momentum
                            lx = particles_global(X_ANG_PART_PROP,p)
                            ly = particles_global(Y_ANG_PART_PROP,p)
                            lz = particles_global(Z_ANG_PART_PROP,p)
                            angmom = sqrt(lx**2 + ly**2 + lz**2)

                            theta = 0.0
                            if (angmom .gt. 0.0) then
                                theta = acos(abs(((lx*dx)+(ly*dy)+(lz*dz))/(radius*angmom)))
                            end if

                            tau = 0.0
                            if (theta .gt. 0) then
                                tau = sink_heating_tau0 * exp(-4*PI*((cos(theta)/sin(theta))**2)) * &
                                    log(max(radius,sink_heating_inner_radius)/au) / sin(theta)
                            end if
                            flux = flux + t_shape * lum * exp(-tau) / (4*PI*max(radius,sink_heating_inner_radius)**2)

                            !if (isnan(flux)) print *, '[', dr_globalMe, &
                            !    '] Particles_sinkHeating: flux, lum, t_shape, tau, exp(-tau), radius = ', &
                            !                              flux, lum, t_shape, tau, exp(-tau), radius

                        end if ! radius .le. sink_heating_radius

                    end do ! loop over all particles in heating radius

                end if ! particle found in heating radius

                ! update sink luminosity heating temperature (SLHT_VAR) field
                solnData(SLHT_VAR,i,j,k) = (sb_factor*flux)**0.25

            enddo ! i
          enddo ! j
        enddo ! k

        call Grid_releaseBlkPtr(blockID,solnData)

        deallocate(xc)
        deallocate(yc)
        deallocate(zc)

    enddo ! loop over blocks

    call Timers_stop("sinkHeating")

    if (debug .and. (dr_globalMe == MASTER_PE)) print *, '[', dr_globalMe, '] Particles_sinkHeating: exiting.'

    return

end subroutine Particles_sinkHeating
