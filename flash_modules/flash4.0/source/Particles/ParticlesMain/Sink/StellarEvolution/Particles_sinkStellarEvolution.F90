!!****if* source/Particles/ParticlesMain/Sink/StellarEvolution/Particles_sinkStellarEvolution
!!
!! NAME
!!
!!  Particles_sinkStellarEvolution
!!
!! SYNOPSIS
!!
!!  call Particles_sinkStellarEvolution()
!!
!! DESCRIPTION
!!
!!  - stellar evolution (including proto-stellar and ZAMS)
!!  - low-mass and high-mass stars are supported
!!  - updates stellar radius, deuterium fraction, luminosity, etc.
!!
!! ARGUMENTS
!!
!! NOTES
!!
!!   written by Christoph Federrath, 2016
!!   C++ class/function calls adapted from Offner et al. (2009, ApJ 703, 131)
!!
!!***

subroutine Particles_sinkStellarEvolution()

    use Particles_sinkData
    use pt_sinkInterface, ONLY: pt_sinkGatherGlobal
    use Driver_interface, ONLY : Driver_abortFlash
    use Driver_data, ONLY : dr_globalMe, dr_simTime, dr_dt
    use RuntimeParameters_interface, ONLY : RuntimeParameters_get
    use Timers_interface, ONLY : Timers_start, Timers_stop
    
    implicit none

#include "constants.h"
#include "Flash.h"
#include "Particles.h"

    logical, save :: first_call = .true.
    logical, save :: sink_stellar_evolution

    integer :: p
    real    :: m_in, accr_rate, luminosity_out
    real    :: mdeut_in_out, r_in_out, burnState_in_out

    logical, parameter :: log_stellar_evol = .true.
    integer, parameter  :: fu_selog = 48
    integer, parameter :: gather_nprops = 5
    integer, dimension(gather_nprops), save :: gather_propinds = &
      (/ integer :: OLD_PMASS_PART_PROP, DEUTERIUM_MASS_PART_PROP, STELLAR_RADIUS_PART_PROP, &
            &       BURN_STATE_PART_PROP, LUMINOSITY_PART_PROP /)

    logical, parameter :: debug = .false.

    ! ================= first call =================
    if (first_call) then

        call RuntimeParameters_get("sink_stellar_evolution", sink_stellar_evolution)

        if (sink_stellar_evolution .and. (dr_globalMe == MASTER_PE)) &
            & print *, 'Particles_sinkStellarEvolution: activated and initialized.'

        if (sink_stellar_evolution .and. log_stellar_evol .and. (dr_globalMe == MASTER_PE)) then
            open(fu_selog, file='sinks_stellar_evolution.dat', position='APPEND')
            write(fu_selog,'(8(1X,A16))') '[00]part_tag', '[01]time', '[02]mass', '[03]macc_rate', '[04]mdeut', &
              & '[05]stellar_rad', '[06]burn_state', '[07]luminosity'
            close(fu_selog)
        endif

        first_call = .false.

    endif ! ================= end first call =================


    ! return if switched off or if no sink particles present
    if ((.not. sink_stellar_evolution) .or. (localnpf .le. 0)) return

    if (debug .and. (dr_globalMe == MASTER_PE)) print *, '[', dr_globalMe, '] Particles_sinkStellarEvolution: entering'

    ! stop the code if the time step is ridiculous
    ! if (dr_dt .le. 1.0) call Driver_abortFlash("Particles_sinkStellarEvolution: dr_dt is too small")

    call Timers_start("sinkStellarEvolution")

    ! this is all just a local particles procedure
    do p = 1, localnp

        m_in = particles_local(MASS_PART_PROP,p)

        ! compute the actual accretion rate onto the star (after outflow)
        ! CAUTION: do not use ACCR_RATE_PART_PROP here, because that does not take outflow into account
        accr_rate = (particles_local(MASS_PART_PROP,p)-particles_local(OLD_PMASS_PART_PROP,p))/dr_dt

        ! these are input and output variables
        mdeut_in_out = particles_local(DEUTERIUM_MASS_PART_PROP,p)
        r_in_out = particles_local(STELLAR_RADIUS_PART_PROP,p)
        burnState_in_out = particles_local(BURN_STATE_PART_PROP,p)

        ! call Offner et al. (2009) stellar evolution code (C++)
        call update_stellar_evolution_c (m_in, accr_rate, mdeut_in_out, r_in_out, &
                                         & burnState_in_out, luminosity_out, dr_dt)

        ! set the new stellar evolution variables
        particles_local(DEUTERIUM_MASS_PART_PROP,p) = mdeut_in_out
        particles_local(STELLAR_RADIUS_PART_PROP,p) = r_in_out
        particles_local(BURN_STATE_PART_PROP,p) = burnState_in_out
        particles_local(LUMINOSITY_PART_PROP,p) = luminosity_out

    enddo  ! loop over particle list

    ! write out log file (note that this slows down the code, because it needs communication to gather particles info)
    if (log_stellar_evol) then
        ! communicate
        call pt_sinkGatherGlobal(gather_propinds, gather_nprops)
        ! write file
        if (dr_globalMe == MASTER_PE) then
            open(fu_selog, file='sinks_stellar_evolution.dat', position='APPEND')
            do p = 1, localnpf
                accr_rate = (particles_global(MASS_PART_PROP,p)-particles_global(OLD_PMASS_PART_PROP,p))/dr_dt
                write(fu_selog,'((1X,I16),5(1X,ES16.9),(1X,I16),(1X,ES16.9))') int(particles_global(TAG_PART_PROP,p)), &
                  dr_simTime, particles_global(MASS_PART_PROP,p), accr_rate, &
                  & particles_global(DEUTERIUM_MASS_PART_PROP,p), particles_global(STELLAR_RADIUS_PART_PROP,p), &
                  & int(particles_global(BURN_STATE_PART_PROP,p)), particles_global(LUMINOSITY_PART_PROP,p)
            enddo
            close(fu_selog)
        endif
    endif

    call Timers_stop("sinkStellarEvolution")

    if (debug .and. (dr_globalMe == MASTER_PE)) print *, '[', dr_globalMe, '] Particles_sinkStellarEvolution: exiting.'

    return

end subroutine Particles_sinkStellarEvolution
