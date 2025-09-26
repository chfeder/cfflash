!!****if* source/physics/sourceTerms/Polytrope/PolytropeMain/Polytrope
!!
!! NAME
!!  Polytrope
!!
!! SYNOPSIS
!!  Polytrope(integer(IN)::blockCount,
!!            integer(IN)::blockList(blockCount),
!!            real(IN)::dt)
!!
!! DESCRIPTION
!!  Implement the polytropic eos as source term
!!
!! ARGUMENTS
!!  blockCount   : The number of blocks in the list
!!  blockList(:) : The list of blocks on which to apply the Polytrope operator
!!  dt           : the current timestep
!!
!! WRITTEN BY
!!   Christoph Federrath, 2007-2023
!!   Modified by Chalence Safranek-Shrader, 2011
!!
!!***

subroutine Polytrope(blockCount, blockList, dt)

  use Polytrope_data
  use Grid_interface,   ONLY : Grid_getBlkIndexLimits, &
                               Grid_getDeltas, Grid_getBlkPtr, Grid_releaseBlkPtr
  use Timers_interface, ONLY : Timers_start, Timers_stop
  use Driver_data,      ONLY : dr_globalMe, dr_simTime
  use Eos_interface,    ONLY : Eos_wrapped
  use Eos_data,         ONLY : eos_singleSpeciesA
  use PhysicalConstants_interface, ONLY : PhysicalConstants_get

  implicit none

#include "constants.h"
#include "Flash.h"
#include "Flash_mpi.h"

  integer, intent(IN)                        :: blockCount
  integer, dimension(blockCount), intent(IN) :: blockList
  real, intent(IN)                           :: dt

  integer                                    :: blockID, thisBlock, i, j, k, error
  integer, dimension(2,MDIM)                 :: blkLimits, blkLimitsGC
  real, pointer, dimension(:,:,:,:)          :: solnData
  real                                       :: del(MDIM)

  integer, parameter                         :: funit = 22
  character(len=80)                          :: outfile = "polytrope.dat"

  real                                       :: rho, press, eint_old_tot, eint_new_tot, d_eint, d_eint_red
  real                                       :: time, dvol
  logical, parameter                         :: polytropeDebug = .false.

  real, save                                 :: polytropeKonst1, polytropeKonst2, polytropeKonst3, &
                                                & polytropeKonst4, polytropeKonst5
#ifdef SLHT_VAR
  real, save                                 :: kB, mp, slhe_factor
#endif

  logical, save :: first_call = .true.

!! ====================================================================================

  if (.not. poly_usePolytrope) return

  if (first_call) then

     ! Here, we define the polytropic constants to maintain continuity of pressure
     polytropeKonst1 = polytropeKonst
     polytropeKonst2 = polytropeKonst1 * polytropeDens2**(polytropeGamma1 - polytropeGamma2)
     polytropeKonst3 = polytropeKonst2 * polytropeDens3**(polytropeGamma2 - polytropeGamma3)
     polytropeKonst4 = polytropeKonst3 * polytropeDens4**(polytropeGamma3 - polytropeGamma4)
     polytropeKonst5 = polytropeKonst4 * polytropeDens5**(polytropeGamma4 - polytropeGamma5)

#ifdef SLHT_VAR
     call PhysicalConstants_get("Boltzmann", kB)
     call PhysicalConstants_get("proton mass", mp)
     slhe_factor = kB/(eos_singleSpeciesA*mp)
#endif

     if (dr_globalMe .eq. MASTER_PE) then
        open(funit, file=trim(outfile), position='APPEND')
        write(funit,'(4(1X,A16))') '[00]time', '[01]dt', '[02]d(Eint)', '[03]d(Eint)/dt'
        close(funit)
     endif

     first_call = .false.

  end if

  call Timers_start("polytrope_timer")

  if (polytropeDebug .and. (dr_globalMe .eq. MASTER_PE)) then
    print *, 'Polytrope() entering ...'
  endif

  ! reset to zero for starting summation below
  eint_old_tot = 0.0
  eint_new_tot = 0.0

  ! loop over list of blocks passed in
  do thisBlock = 1, blockCount

    blockID = blockList(thisBlock)
    ! get block limits
    call Grid_getBlkIndexLimits(blockID, blkLimits, blkLimitsGC)
    ! get a pointer to solution data
    call Grid_getBlkPtr(blockID, solnData)
    ! getting the dx's
    call Grid_getDeltas(blockID, del)
#if NDIM == 1
    dvol = del(IAXIS)
#endif
#if NDIM == 2
    dvol = del(IAXIS) * del(JAXIS)
#endif
#if NDIM == 3
    dvol = del(IAXIS) * del(JAXIS) * del(KAXIS)
#endif

    ! only loop active cells (ghost cells must be filled after source terms anyway)
    do k = blkLimits(LOW, KAXIS), blkLimits(HIGH, KAXIS)
      do j = blkLimits(LOW, JAXIS), blkLimits(HIGH, JAXIS)
        do i = blkLimits(LOW, IAXIS), blkLimits(HIGH, IAXIS)

          rho = solnData(DENS_VAR,i,j,k)

          ! sum up the old internal energy
          eint_old_tot = eint_old_tot + solnData(EINT_VAR,i,j,k)*rho*dvol

          ! First polytropic regime
          if (rho .lt. polytropeDens1) then
             ! do nothing
             press = solnData(PRES_VAR,i,j,k)
          end if

          ! Second polytropic regime
          if (rho .ge. polytropeDens1 .and. rho .le. polytropeDens2) then
             press = polytropeKonst1 * rho**(polytropeGamma1)
          end if

          ! Third polytropic regime
          if (rho .gt. polytropeDens2 .and. rho .le. polytropeDens3) then
             press = polytropeKonst2 * rho**(polytropeGamma2)
          end if

          if (rho .gt. polytropeDens3 .and. rho .le. polytropeDens4) then
             press = polytropeKonst3 * rho**(polytropeGamma3)
          end if

          if (rho .gt. polytropeDens4 .and. rho .le. polytropeDens5) then
             press = polytropeKonst4 * rho**(polytropeGamma4)
          end if

          if (rho .gt. polytropeDens5) then
             press = polytropeKonst5 * rho**(polytropeGamma5)
          end if

          ! put back the new quantities
          solnData(PRES_VAR,i,j,k) = press

#ifdef SLHT_VAR
          ! sink luminosity heating feedback (SLHT_VAR is set in Particles_sinkHeating)
          solnData(PRES_VAR,i,j,k) = ( solnData(PRES_VAR,i,j,k)**4 + &
                                        (slhe_factor*solnData(SLHT_VAR,i,j,k)*rho)**4 )**0.25

#endif
        enddo ! end loop over i
      enddo ! end loop over j
    enddo ! end loop over k

    ! reset eint and temp by calling EOS
    call Eos_wrapped(MODE_DENS_PRES, blkLimits, blockID)

    ! loop again to compute the change of internal energy (only loop active cells)
    do k = blkLimits(LOW, KAXIS), blkLimits(HIGH, KAXIS)
      do j = blkLimits(LOW, JAXIS), blkLimits(HIGH, JAXIS)
        do i = blkLimits(LOW, IAXIS), blkLimits(HIGH, IAXIS)

          rho = solnData(DENS_VAR,i,j,k)

          ! sum up the new internal energy
          eint_new_tot = eint_new_tot + solnData(EINT_VAR,i,j,k)*rho*dvol

        enddo ! end loop over i
      enddo ! end loop over j
    enddo ! end loop over k

    call Grid_releaseBlkPtr(blockID,solnData)

  enddo ! loop over blocks

  ! compute total change of internal energy for this processor
  d_eint = eint_new_tot - eint_old_tot

  ! Sum up contributions from all blocks and processors
  d_eint_red = 0.0
  call MPI_AllReduce (d_eint, d_eint_red, 1, FLASH_REAL, MPI_Sum, MPI_Comm_World, error)
  d_eint = d_eint_red

  ! Write time evolution of the change of internal energy to file
  if (dr_globalMe .eq. MASTER_PE) then
    open(funit, file=trim(outfile), position='APPEND')
    write(funit,'(4(1X,ES16.9))') dr_simTime, dt, d_eint, d_eint/dt
    close(funit)
  endif

  if (polytropeDebug .and. (dr_globalMe .eq. MASTER_PE)) then
    print *, 'Polytrope() exiting ...'
  endif

  call Timers_stop ("polytrope_timer")

  return

end subroutine Polytrope
