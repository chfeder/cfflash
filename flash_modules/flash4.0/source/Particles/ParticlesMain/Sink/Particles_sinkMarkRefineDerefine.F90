!!****if* source/Particles/ParticlesMain/Sink/Outflow/Particles_sinkMarkRefineDerefine
!!
!! NAME
!!
!!  Particles_sinkMarkRefineDerefine
!!
!! SYNOPSIS
!!
!!  call Particles_sinkMarkRefineDerefine()
!!
!! DESCRIPTION
!!
!!  This routine takes care of grid refinement based on sink particles.
!!  If sink particles are present, it is recommended that they stay on the
!!  highest level of AMR, so this routine flags all cells within the sink particle
!!  accretion (or outflow) radius for refinement to the highest level.
!!
!! ARGUMENTS
!!
!! NOTES
!!
!!   written by Christoph Federrath, 2013-2022
!!
!!***

subroutine Particles_sinkMarkRefineDerefine()

  use tree
  use paramesh_dimensions
  use Grid_data, ONLY : gr_maxRefine
  use Cosmology_interface, ONLY : Cosmology_getRedshift
  use Grid_interface, ONLY : Grid_getBlkPhysicalSize, Grid_getBlkIndexLimits, &
                             Grid_getCellCoords
  use RuntimeParameters_interface, ONLY : RuntimeParameters_get
  use Particles_sinkData, ONLY : localnpf, particles_global, maxsinks, &
                                 accretion_radius, sink_grav_boundary_type
  use pt_sinkInterface, only: pt_sinkGatherGlobal, pt_sinkCorrectForPeriodicBCs
  use Driver_data, ONLY : dr_globalMe

  implicit none

#include "constants.h"
#include "Flash.h"

  logical, save :: refineOnSinkParticles, first_call = .true.
  integer, save :: outflow_mass_model = 0
  real, save :: outflow_radius, refine_radius

  real, dimension(:), allocatable :: xc, yc, zc
  integer, dimension(2,MDIM) :: blkLimits, blkLimitsGC
  real, dimension(MDIM) :: blockSize

  integer :: size_x, size_y, size_z, lb, kp, jp, ip, p
  real :: refine_radius_comoving, rad, distx, disty, distz, dxhalf, dyhalf, dzhalf
  logical :: p_found

  integer, parameter :: gather_nprops = 3
  integer, dimension(gather_nprops), save :: gather_propinds = &
    (/ integer :: POSX_PART_PROP, POSY_PART_PROP, POSZ_PART_PROP /)

#ifdef COSMOLOGY
  real :: redshift
#endif

  if (first_call) then
    call RuntimeParameters_get("refineOnSinkParticles", refineOnSinkParticles)
#ifdef SINKS_OUTFLOW
    call RuntimeParameters_get("outflow_mass_model", outflow_mass_model)
    call RuntimeParameters_get("outflow_radius", outflow_radius)
#endif
    ! turn off refinement on outflow radius if the outflow model is switched off or not present
    if (outflow_mass_model .le. 0) outflow_radius = 0.0
    ! use the maximum radius for refinement
    refine_radius = max(accretion_radius, outflow_radius)
    if (refineOnSinkParticles .and. (dr_globalMe .eq. MASTER_PE)) then
      write(*,'(A,ES16.9)') &
        ' Particles_sinkMarkRefineDerefine: Using sink particle refinement with refine_radius = ', refine_radius
    endif
    ! end of first call
    first_call = .false.
  end if

  if (.not. refineOnSinkParticles) return

#ifdef COSMOLOGY
  call Cosmology_getRedshift(redshift)
  refine_radius_comoving = refine_radius * (redshift + 1.0)
#else
  refine_radius_comoving = refine_radius
#endif

  ! update particles_global array
  call pt_sinkGatherGlobal(gather_propinds, gather_nprops)

  ! Any cell within refine_radius_comoving of sink particle should be at the
  ! highest refinement level (its block, to be precise)

  do lb = 1, lnblocks

    if (nodetype(lb) .eq. 1) then

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

      call Grid_getBlkPhysicalSize(lb, blockSize)

      dxhalf = blockSize(1)/real(NXB)/2.0
      dyhalf = blockSize(2)/real(NYB)/2.0
      dzhalf = blockSize(3)/real(NZB)/2.0

      ! see if particle is inside block
      do p = 1, localnpf
        if ( (particles_global(POSX_PART_PROP,p).ge.(xc(blkLimits(LOW,IAXIS))-dxhalf)) .and. &
             (particles_global(POSX_PART_PROP,p).le.(xc(blkLimits(HIGH,IAXIS))+dxhalf)) .and. &
             (particles_global(POSY_PART_PROP,p).ge.(yc(blkLimits(LOW,JAXIS))-dyhalf)) .and. &
             (particles_global(POSY_PART_PROP,p).le.(yc(blkLimits(HIGH,JAXIS))+dyhalf)) .and. &
             (particles_global(POSZ_PART_PROP,p).ge.(zc(blkLimits(LOW,KAXIS))-dzhalf)) .and. &
             (particles_global(POSZ_PART_PROP,p).le.(zc(blkLimits(HIGH,KAXIS))+dzhalf)) ) then
          if (lrefine(lb) .lt. gr_maxRefine) then
            derefine(lb) = .false.
            refine(lb) = .true.
            stay(lb) = .false.
          else
            derefine(lb) = .false.
            refine(lb) = .false.
            stay(lb) = .true.
          end if
        end if
      end do ! particles

      do kp = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
        do jp = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
          do ip = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

            ! cell within refinement radius?
            p_found = .false.
            do p = 1, localnpf
              distx = xc(ip) - particles_global(POSX_PART_PROP,p)
              disty = yc(jp) - particles_global(POSY_PART_PROP,p)
              distz = zc(kp) - particles_global(POSZ_PART_PROP,p)
              if (sink_grav_boundary_type .eq. 2) call pt_sinkCorrectForPeriodicBCs(distx, disty, distz)
              rad = sqrt(distx**2 + disty**2 + distz**2)
              if (rad .le. refine_radius_comoving) p_found = .true.
            end do

            ! derefinement
            if ((.not.p_found) .and. (.not.refine(lb)) .and. (.not.stay(lb))) then
              derefine(lb) = .true.
            else
              derefine(lb) = .false.
            end if

            ! refinement if cell/block is within sink radius
            if (p_found .and. (lrefine(lb) .lt. gr_maxRefine)) then
              derefine(lb) = .false.
              refine(lb) = .true.
              stay(lb) = .false.
            end if

            ! stay at highest level if it already is
            if (p_found .and. (lrefine(lb) .ge. gr_maxRefine)) then
              derefine(lb) = .false.
              refine(lb) = .false.
              stay(lb) = .true.
            end if

          end do
        end do
      end do

      deallocate(xc)
      deallocate(yc)
      deallocate(zc)

    end if ! leaf block

  end do ! loop over blocks

  return

end subroutine Particles_sinkMarkRefineDerefine
