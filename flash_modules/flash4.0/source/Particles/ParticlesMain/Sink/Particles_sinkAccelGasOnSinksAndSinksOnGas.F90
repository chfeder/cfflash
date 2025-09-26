!!****if* source/Particles/ParticlesMain/Sink/Particles_sinkAccelGasOnSinksAndSinksOnGas
!!
!! NAME
!!
!!  Particles_sinkAccelGasOnSinksAndSinksOnGas
!!
!! SYNOPSIS
!!
!!  call Particles_sinkAccelGasOnSinksAndSinksOnGas()
!!
!! DESCRIPTION
!!
!!  Computes gas -> sinks and sinks -> gas gravitational accelerations
!!  by direct summation over all sink particles and grid cells.
!!  For cosmology, will also want to get contribution from PDE
!!  (mapped DM delegate particle density).
!!
!! ARGUMENTS
!!
!! NOTES
!!
!!   written by Christoph Federrath, 2008-2015
!!   ported to FLASH3.3/4 by Chalence Safranek-Shrader, 2010-2012
!!   modified by Nathan Goldbaum, 2012
!!   refactored for FLASH4 by John Bachan, 2012
!!   debugged and renamed to reflect symmetry with Particles_sinkAccelSinksOnGas (Christoph Federrath, 2013)
!!   merged with Particles_sinkAccelSinksOnGas to speed up computation (Christoph Federrath, 2015)
!!
!!***

subroutine Particles_sinkAccelGasOnSinksAndSinksOnGas()

  use Particles_sinkData
  use pt_sinkSort
  use pt_sinkInterface, ONLY: pt_sinkGatherGlobal, pt_sinkEwaldCorrection, &
                              pt_sinkCorrectForPeriodicBCs
  use Driver_interface, ONLY : Driver_abortFlash, Driver_getSimTime
  use Driver_data, ONLY : dr_globalMe
  use Grid_interface, ONLY :  Grid_getCellCoords, Grid_getBlkPhysicalSize, &
                              Grid_getBlkPtr, Grid_releaseBlkPtr, Grid_getBlkIndexLimits, &
                              Grid_getListOfBlocks
  use Cosmology_interface, ONLY : Cosmology_getRedshift
  use Timers_interface, ONLY : Timers_start, Timers_stop

  implicit none

#include "constants.h"
#include "Flash.h"
#include "Particles.h"
  include "Flash_mpi.h"

  real               :: slope, hinv, h2inv, softening_radius_comoving
  real               :: pmass, paccx, paccy, paccz
  integer            :: i, j, k, p, ierr
  integer            :: size_x, size_y, size_z
  real               :: dx_block, dy_block, dz_block, dVol
  real               :: dx, dy, dz, radius, q, kernelvalue, r3, ax, ay, az
  real               :: exc, eyc, ezc
  real               :: prefactor_gos, prefactor_sog, redshift, oneplusz3
  real               :: size(3)

  integer :: blockCount, ib, lb
  integer :: blockList(MAXBLOCKS)
  
  integer, allocatable, dimension(:) :: id_sorted, QSindex
  real, allocatable, dimension(:) :: ax_sorted, ay_sorted, az_sorted, ax_total, ay_total, az_total
  real,pointer, dimension(:,:,:,: ) :: solnData
  real, dimension(:), allocatable :: xc, yc, zc
  integer, dimension(2,MDIM) :: blkLimits, blkLimitsGC

  integer, parameter :: gather_nprops = 5
  integer, dimension(gather_nprops), save :: gather_propinds = &
    (/ integer :: POSX_PART_PROP, POSY_PART_PROP, POSZ_PART_PROP, TAG_PART_PROP, MASS_PART_PROP /)

  logical, parameter :: Debug = .false.

  !==============================================================================

  if (.not. useSinkParticles) return

  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, 'Particles_sinkAccelGasOnSinksAndSinksOnGas: entering.'

  call Timers_start("AccelGasSinks-SinksGas")

  ! Exchange particle information (so we are up-to-date on localnpf)
  call pt_sinkGatherGlobal(gather_propinds, gather_nprops)

  if (localnpf .eq. 0) then
     call Timers_stop("AccelGasSinks-SinksGas")
     if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, 'Particles_sinkAccelGasOnSinksAndSinksOnGas: exiting.'
     return
  endif

  call Cosmology_getRedshift(redshift)
  softening_radius_comoving = softening_radius * (1.0 + redshift)
  hinv  = 2.0/softening_radius_comoving !!! makes sure that only for r < r_soft actual softening occurs
  h2inv = hinv**2
  slope = 1.0 / softening_radius_comoving**3
  oneplusz3 = (1.0 + redshift)**3.0
  prefactor_sog = -newton*oneplusz3

  ! Clear global accelerations
  particles_global(ACCX_PART_PROP, 1:localnpf) = 0.0
  particles_global(ACCY_PART_PROP, 1:localnpf) = 0.0
  particles_global(ACCZ_PART_PROP, 1:localnpf) = 0.0

  call Grid_getListOfBlocks(LEAF,blockList,blockCount)

  call Timers_start("loop")

  ! Loop over blocks
  do ib = 1, blockCount

        lb = blockList(ib)

        call Grid_getBlkPtr(lb,solnData)

        call Grid_getBlkIndexLimits(lb, blkLimits, blkLimitsGC)
        size_x = blkLimitsGC(HIGH,IAXIS)-blkLimitsGC(LOW,IAXIS) + 1
        size_y = blkLimitsGC(HIGH,JAXIS)-blkLimitsGC(LOW,JAXIS) + 1
        size_z = blkLimitsGC(HIGH,KAXIS)-blkLimitsGC(LOW,KAXIS) + 1

        allocate(xc(size_x))
        allocate(yc(size_y))
        allocate(zc(size_z))

        call Grid_getCellCoords(IAXIS, lb, CENTER, .true., xc, size_x)
        call Grid_getCellCoords(JAXIS, lb, CENTER, .true., yc, size_y)
        call Grid_getCellCoords(KAXIS, lb, CENTER, .true., zc, size_z)

        call Grid_getBlkPhysicalSize(lb,size)
        dx_block = size(1)/real(NXB)
        dy_block = size(2)/real(NYB)
        dz_block = size(3)/real(NZB)
        dVol = dx_block*dy_block*dz_block

        ! loop over cells (exclude guard cells)
        do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
           do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
              do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

                 prefactor_gos = -newton * solnData(DENS_VAR,i,j,k)  * dVol
#ifdef PDE_VAR
                 prefactor_gos = -newton * (solnData(DENS_VAR,i,j,k) + solnData(PDE_VAR,i,j,k)) * dVol
#endif
                 ! factor of (1+z)^3 needed in cosmological settings:
                 prefactor_gos = prefactor_gos * oneplusz3

                 ! reset particle accelerations
                 paccx = 0.0
                 paccy = 0.0
                 paccz = 0.0

                 ! Loop over all particles, local and global
                 do p = 1, localnpf

                       ! particle mass
                       pmass = particles_global(MASS_PART_PROP,p)

                       ! compute relative distances
                       dx = particles_global(POSX_PART_PROP,p) - xc(i)
                       dy = particles_global(POSY_PART_PROP,p) - yc(j)
                       dz = particles_global(POSZ_PART_PROP,p) - zc(k)

                       if (sink_grav_boundary_type .eq. 2) call pt_sinkCorrectForPeriodicBCs(dx, dy, dz)

                       radius = sqrt(dx**2 + dy**2 + dz**2)

                       ! compute accel
                       if (radius .lt. softening_radius_comoving) then
                          if (softening_type_gas .eq. 1) then ! spline softening
                             q = radius*hinv
                             kernelvalue = 0.0
                             if ((q.gt.1.0e-5) .and. (q.lt.1.0)) &
                                & kernelvalue = h2inv*(4.0/3.0*q-1.2*q**3+0.5*q**4)/radius
                             if ((q.ge.1.0)    .and. (q.lt.2.0)) &
                                & kernelvalue = h2inv * &
                                & (8.0/3.0*q-3.0*q**2+1.2*q**3-1.0/6.0*q**4-1.0/(15.0*q**2))/radius
                             ax = kernelvalue*dx
                             ay = kernelvalue*dy
                             az = kernelvalue*dz
                          end if

                          if (softening_type_gas .eq. 2) then ! linear kernel inside softening_radius
                             ax = dx*slope
                             ay = dy*slope
                             az = dz*slope
                          end if
                       else
                          r3 = 1.0 / radius**3
                          ax = dx*r3
                          ay = dy*r3
                          az = dz*r3
                       end if

                       if (sink_grav_boundary_type .eq. 2) then
                          call pt_sinkEwaldCorrection(abs(dx), abs(dy), abs(dz), exc, eyc, ezc)
                          ax = ax - sign(exc,dx)
                          ay = ay - sign(eyc,dy)
                          az = az - sign(ezc,dz)
                       endif

                       ! gas on sinks: add cell contribution to particle acceleration
                       particles_global(ACCX_PART_PROP,p) = particles_global(ACCX_PART_PROP,p) + & 
                            prefactor_gos * ax
                       particles_global(ACCY_PART_PROP,p) = particles_global(ACCY_PART_PROP,p) + & 
                            prefactor_gos * ay
                       particles_global(ACCZ_PART_PROP,p) = particles_global(ACCZ_PART_PROP,p) + & 
                            prefactor_gos * az

                       ! sinks on gas accelerations
                       paccx = paccx - ax * pmass
                       paccy = paccy - ay * pmass
                       paccz = paccz - az * pmass

                 end do ! loop over all particles

                 ! sinks on gas: x-acceleration:
                 solnData(SGAX_VAR,i,j,k) = paccx * prefactor_sog
                 ! sinks on gas: y-acceleration:
                 solnData(SGAY_VAR,i,j,k) = paccy * prefactor_sog
                 ! sinks on gas: z-acceleration:
                 solnData(SGAZ_VAR,i,j,k) = paccz * prefactor_sog

              enddo  ! i
           enddo  ! j
        enddo  ! k

        call Grid_releaseBlkPtr(lb,solnData)

        deallocate(xc)
        deallocate(yc)
        deallocate(zc)

  enddo  ! loop over blocks

  call Timers_stop("loop")

  call Timers_start("reduce")

  ! allocate temporary arrays
  allocate(id_sorted(localnpf), stat=ierr)
  if (ierr.ne.0) call Driver_abortFlash ("Particles_sinkAccelGasOnSinksAndSinksOnGas:  could not allocate id_sorted")
  allocate(QSindex(localnpf), stat=ierr)
  if (ierr.ne.0) call Driver_abortFlash ("Particles_sinkAccelGasOnSinksAndSinksOnGas:  could not allocate QSindex")
  allocate(ax_sorted(localnpf), stat=ierr)
  if (ierr.ne.0) call Driver_abortFlash ("Particles_sinkAccelGasOnSinksAndSinksOnGas:  could not allocate ax_sorted")
  allocate(ay_sorted(localnpf), stat=ierr)
  if (ierr.ne.0) call Driver_abortFlash ("Particles_sinkAccelGasOnSinksAndSinksOnGas:  could not allocate ay_sorted")
  allocate(az_sorted(localnpf), stat=ierr)
  if (ierr.ne.0) call Driver_abortFlash ("Particles_sinkAccelGasOnSinksAndSinksOnGas:  could not allocate az_sorted")
  allocate(ax_total(localnpf), stat=ierr)
  if (ierr.ne.0) call Driver_abortFlash ("Particles_sinkAccelGasOnSinksAndSinksOnGas:  could not allocate ax_total")
  allocate(ay_total(localnpf), stat=ierr)
  if (ierr.ne.0) call Driver_abortFlash ("Particles_sinkAccelGasOnSinksAndSinksOnGas:  could not allocate ay_total")
  allocate(az_total(localnpf), stat=ierr)
  if (ierr.ne.0) call Driver_abortFlash ("Particles_sinkAccelGasOnSinksAndSinksOnGas:  could not allocate az_total")

  ! sort global particles list before global all sum
  do p = 1, localnpf
     id_sorted(p) = int(particles_global(TAG_PART_PROP,p))
  enddo

  if (localnpf .gt. 0) call NewQsort_IN(id_sorted, QSindex)

  ! now particles are sorted by their tag
  do p = 1, localnpf
     ax_sorted(p) = particles_global(ACCX_PART_PROP, QSindex(p))
     ay_sorted(p) = particles_global(ACCY_PART_PROP, QSindex(p))
     az_sorted(p) = particles_global(ACCZ_PART_PROP, QSindex(p))
     ax_total(p) = 0.0
     ay_total(p) = 0.0
     az_total(p) = 0.0
  enddo

  ! Communicate to get total contribution from all cells on all procs
  call MPI_ALLREDUCE(ax_sorted, ax_total, localnpf, FLASH_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_ALLREDUCE(ay_sorted, ay_total, localnpf, FLASH_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_ALLREDUCE(az_sorted, az_total, localnpf, FLASH_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)

  do p = 1, localnpf
     particles_global(ACCX_PART_PROP, QSindex(p)) = ax_total(p)
     particles_global(ACCY_PART_PROP, QSindex(p)) = ay_total(p)
     particles_global(ACCZ_PART_PROP, QSindex(p)) = az_total(p)
  end do

  do p = 1, localnp
     particles_local(ACCX_PART_PROP,p) = particles_global(ACCX_PART_PROP,p)
     particles_local(ACCY_PART_PROP,p) = particles_global(ACCY_PART_PROP,p)
     particles_local(ACCZ_PART_PROP,p) = particles_global(ACCZ_PART_PROP,p)
  end do

  deallocate(id_sorted)
  deallocate(QSindex)
  deallocate(ax_sorted)
  deallocate(ay_sorted)
  deallocate(az_sorted)
  deallocate(ax_total)
  deallocate(ay_total)
  deallocate(az_total)

  call Timers_stop("reduce")
  call Timers_stop("AccelGasSinks-SinksGas")

  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, 'Particles_sinkAccelGasOnSinksAndSinksOnGas: exiting.'

  return

end subroutine Particles_sinkAccelGasOnSinksAndSinksOnGas
