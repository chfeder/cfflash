!!****if* source/Particles/ParticlesMain/Sink/StarCluster/Particles_sinkStellarEvolution
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
!!  Sets luminosity and radius of sink particles that represent star clusters.
!!
!! ARGUMENTS
!!
!! NOTES
!!
!!   written by Shyam Menon, 2022
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
    real, save :: cluster_lumpermass, cluster_imffactor
    integer, save:: cluster_type

    integer :: p
    real    :: m_in, luminosity_out
    real    :: chi

    logical, parameter :: debug = .false.
    real, parameter :: msun = 1.9891e+33, lsun = 3.839e+33, eV = 1.6021764620000066e-12

    ! ================= first call =================
    if (first_call) then
        call RuntimeParameters_get("cluster_type", cluster_type)
        call RuntimeParameters_get("cluster_imffactor", cluster_imffactor)
        if (cluster_type .eq. 1) call RuntimeParameters_get("cluster_lumpermass", cluster_lumpermass)
        if ((dr_globalMe == MASTER_PE)) then
            print *, 'Particles_sinkStellarEvolution: activated and initialized.'
            print *, 'Cluster type, IMFFactor = ', cluster_type, cluster_imffactor
        endif
        first_call = .false.
    endif ! ================= end first call =================


    ! return if no sink particles present
    if ((localnpf .le. 0)) return

    if (debug .and. (dr_globalMe == MASTER_PE)) print *, '[', dr_globalMe, '] Particles_sinkStellarEvolution: entering'

    call Timers_start("sinkStellarEvolution")

    ! loop over particles
    do p = 1, localnp

        m_in = particles_local(MASS_PART_PROP,p)

        if (cluster_type .eq. 1) then ! Constant user-defined luminosity per-unit mass
            luminosity_out = cluster_lumpermass * m_in
            particles_local(LUMFUV_PART_PROP,p) = luminosity_out
            ! EUV is 40% of FUV luminosity for a fully-sampled IMF + using 18eV per EUV photon
            particles_local(EUVRATE_PART_PROP,p) = 0.4*particles_local(LUMFUV_PART_PROP,p)/(18*eV)
        else if (cluster_type .eq. 2) then ! Fit to SLUG-determined median values for different cluster masses; Eqs. 33 and 34 of Kim et al. (2018)
            chi = log10(m_in/msun)
            ! FUV luminosity
            particles_local(LUMFUV_PART_PROP,p) = 10**(2.98*chi**6/(29.0 + chi**6)) * (m_in/msun) * lsun ! in erg/s
            ! EUV luminosity
            particles_local(EUVRATE_PART_PROP,p) = 10**(46.7*chi**6/(7.28+chi**6)) * (m_in/msun) ! in 1/s
        else
            call Driver_abortFlash("cluster_type needs value of 1 or 2; unsupported value provided by user.")
        endif

    enddo ! loop over particles

    call Timers_stop("sinkStellarEvolution")

    if (debug .and. (dr_globalMe == MASTER_PE)) print *, '[', dr_globalMe, '] Particles_sinkStellarEvolution: exiting.'

    return

end subroutine Particles_sinkStellarEvolution
