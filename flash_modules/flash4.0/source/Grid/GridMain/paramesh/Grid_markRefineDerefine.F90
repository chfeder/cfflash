!!****if* source/Grid/GridMain/paramesh/Grid_markRefineDerefine
!!
!! NAME
!!  Grid_markRefineDerefine
!!
!! SYNOPSIS
!!
!!  Grid_markRefineDerefine()
!!
!! DESCRIPTION 
!!  Mark blocks for refinement or derefinement
!!  This routine is used with AMR only where individual 
!!  blocks are marked for refinement or derefinement based upon
!!  some refinement criterion. The Uniform Grid does not need
!!  this routine, and uses the stub.
!!
!! ARGUMENTS
!!
!! NOTES
!!
!! Every unit uses a few unit scope variables that are
!! accessible to all routines within the unit, but not to the
!! routines outside the unit. For Grid unit these variables begin with "gr_"
!! like, gr_meshMe or gr_eosMode, and are stored in fortran
!! module Grid_data (in file Grid_data.F90). The other variables
!! are local to the specific routines and do not have the prefix "gr_"
!!
!!***

subroutine Grid_markRefineDerefine()

  use Grid_data, ONLY : gr_refine_cutoff, gr_derefine_cutoff, &
                        gr_refine_filter, gr_refine_type, &
                        gr_numRefineVars, gr_refine_var, gr_refineOnParticleCount, &
                        gr_enforceMaxRefinement, gr_maxRefine, &
                        gr_lrefineMaxByTime, gr_lrefineMaxRedDoByTime, &
                        gr_lrefineMaxRedDoByLogR, gr_eosModeNow, &
                        gr_lrefineCenterI, gr_lrefineCenterJ, gr_lrefineCenterK
  use tree, ONLY : newchild, refine, derefine, stay
!!$  use physicaldata, ONLY : force_consistency
  use Logfile_interface, ONLY : Logfile_stampVarMask
  use Grid_interface, ONLY : Grid_fillGuardCells, Grid_markRefineSpecialized
  use Particles_interface, ONLY: Particles_sinkMarkRefineDerefine, Particles_massiveMarkRefineDerefine
  use RuntimeParameters_interface, ONLY : RuntimeParameters_get
  use Driver_data, ONLY : dr_globalMe

  implicit none

#include "constants.h"
#include "Flash.h"

  logical, save :: gcMaskArgsLogged = .false., first_call = .true.
  integer, save :: eosModeLast = 0
  logical, save :: ref_rectangle, ref_ellipse, ref_sphere ! for specialised refinement
  integer, save :: ref_rectangle_lref, ref_ellipse_lref, ref_sphere_lref ! for specialised refinement
  real, save :: specs_rectangle(7), specs_ellipse(7), specs_sphere(5) ! for specialised refinement

  logical :: doEos = .true.
  integer, parameter :: maskSize = NUNK_VARS+NDIM*NFACE_VARS
  logical, dimension(maskSize) :: gcMask
  real :: ref_cut, deref_cut, ref_filter
  integer :: l, i, iref

  if (first_call) then
    call RuntimeParameters_get("ref_rectangle", ref_rectangle)
    call RuntimeParameters_get("ref_rectangle_lref", ref_rectangle_lref)
    call RuntimeParameters_get("ref_rectangle_xmin", specs_rectangle(1))
    call RuntimeParameters_get("ref_rectangle_xmax", specs_rectangle(2))
    call RuntimeParameters_get("ref_rectangle_ymin", specs_rectangle(3))
    call RuntimeParameters_get("ref_rectangle_ymax", specs_rectangle(4))
    call RuntimeParameters_get("ref_rectangle_zmin", specs_rectangle(5))
    call RuntimeParameters_get("ref_rectangle_zmax", specs_rectangle(6))
    call RuntimeParameters_get("ref_rectangle_contained", specs_rectangle(7))
    if (ref_rectangle) then
      if (dr_globalMe .eq. MASTER_PE) then
        write(*,'(A,I2,A,6(1X,ES16.9),A,I1)') ' Grid_markRefineDerefine: refining in rectangle to level ', &
          ref_rectangle_lref, ', with bounds ', specs_rectangle(1:6), ', contained = ', int(specs_rectangle(7))
      endif
    endif
    call RuntimeParameters_get("ref_ellipse", ref_ellipse)
    call RuntimeParameters_get("ref_ellipse_lref", ref_ellipse_lref)
    call RuntimeParameters_get("ref_ellipse_center_x", specs_ellipse(1))
    call RuntimeParameters_get("ref_ellipse_center_y", specs_ellipse(2))
    call RuntimeParameters_get("ref_ellipse_center_z", specs_ellipse(3))
    call RuntimeParameters_get("ref_ellipse_semimajor_x", specs_ellipse(4))
    call RuntimeParameters_get("ref_ellipse_semimajor_y", specs_ellipse(5))
    call RuntimeParameters_get("ref_ellipse_semimajor_z", specs_ellipse(6))
    call RuntimeParameters_get("ref_ellipse_surface_only", specs_ellipse(7))
    if (ref_ellipse) then
      if (dr_globalMe .eq. MASTER_PE) then
        write(*,'(A,I2,A,3(1X,ES16.9),A,3(1X,ES16.9))') ' Grid_markRefineDerefine: refining in ellipse to level ', &
          ref_ellipse_lref, ', with center ', specs_ellipse(1:3), ' and semi-major axes sizes = ', specs_ellipse(4:6)
      endif
    endif
    call RuntimeParameters_get("ref_sphere", ref_sphere)
    call RuntimeParameters_get("ref_sphere_lref", ref_sphere_lref)
    call RuntimeParameters_get("ref_sphere_center_x", specs_sphere(1))
    call RuntimeParameters_get("ref_sphere_center_y", specs_sphere(2))
    call RuntimeParameters_get("ref_sphere_center_z", specs_sphere(3))
    call RuntimeParameters_get("ref_sphere_radius", specs_sphere(4))
    call RuntimeParameters_get("ref_sphere_treat_as_max_lref", specs_sphere(5))
    if (ref_sphere) then
      if (dr_globalMe .eq. MASTER_PE) then
        write(*,'(A,I2,A,3(1X,ES16.9),A,1(1X,ES16.9))') ' Grid_markRefineDerefine: refining in sphere to level ', &
          ref_sphere_lref, ', with center ', specs_sphere(1:3), ' and radius = ', specs_sphere(4)
        if (specs_sphere(5) .eq. 1.0) write(*,'(A,I2,A)') '   ...and limiting refinement to that level (', ref_sphere_lref, ').'
      endif
    endif
    first_call = .false.
  endif

  if (gr_lrefineMaxRedDoByTime) then
     call gr_markDerefineByTime()
  end if

  if (gr_lrefineMaxByTime) then
     call gr_setMaxRefineByTime()
  end if

  if (gr_eosModeNow .NE. eosModeLast) then
     gcMaskArgsLogged = .FALSE.
     eosModeLast = gr_eosModeNow
  end if

  ! that are implemented in this file need values in guardcells

  gcMask=.false.
  do i = 1,gr_numRefineVars
     iref = gr_refine_var(i)
     if (iref > 0) gcMask(iref) = .TRUE.
  end do

  gcMask(NUNK_VARS+1:min(maskSize,NUNK_VARS+NDIM*NFACE_VARS)) = .TRUE.
!!$  gcMask(NUNK_VARS+1:maskSize) = .TRUE.

  if (.NOT.gcMaskArgsLogged) then
     call Logfile_stampVarMask(gcMask, .true., '[Grid_markRefineDerefine]', 'gcArgs')
  end if

!!$  force_consistency = .FALSE.
  call Grid_fillGuardCells(CENTER_FACES,ALLDIR,doEos=.true.,&
       maskSize=maskSize, mask=gcMask, makeMaskConsistent=.true.,doLogMask=.NOT.gcMaskArgsLogged,&
       selectBlockType=ACTIVE_BLKS)
     gcMaskArgsLogged = .TRUE.
!!$  force_consistency = .TRUE.

  newchild(:) = .FALSE.
  refine(:)   = .FALSE.
  derefine(:) = .FALSE.
  stay(:)     = .FALSE.

  do l = 1, gr_numRefineVars
     iref = gr_refine_var(l)
     ref_cut = gr_refine_cutoff(l)
     deref_cut = gr_derefine_cutoff(l)
     ref_filter = gr_refine_filter(l)
     ! Loehner 1987 2nd-derivative refinement
     if (gr_refine_type(l) .eq. 0) call gr_markRefineDerefine(iref, ref_cut, deref_cut, ref_filter)
     ! Relative difference refinement
     if (gr_refine_type(l) .eq. 1) call gr_markRefineDerefineRelDiff(iref, ref_cut, deref_cut)
  end do

#ifdef FLASH_GRID_PARAMESH2
  ! For PARAMESH2, call gr_markRefineDerefine here if it hasn't been called above.
  ! This is necessary to make sure lrefine_min and lrefine_max are obeyed - KW
  if (gr_numRefineVars .LE. 0) then
     call gr_markRefineDerefine(-1, 0.0, 0.0, 0.0)
  end if
#endif

  if (gr_refineOnParticleCount) call gr_ptMarkRefineDerefine()

  if (gr_enforceMaxRefinement) call gr_enforceMaxRefine(gr_maxRefine)

  if (gr_lrefineMaxRedDoByLogR) &
        call gr_unmarkRefineByLogRadius(gr_lrefineCenterI, gr_lrefineCenterJ, gr_lrefineCenterK)

  ! Jeans refinement
  call gr_markRefineDerefineJeans()

  ! sink particle refinement
  call Particles_sinkMarkRefineDerefine()

  ! refine on massive particles
  call Particles_massiveMarkRefineDerefine()

  ! Refine in rectangular region
  if (ref_rectangle) then
    call gr_markInRectangle(specs_rectangle(1), specs_rectangle(2), specs_rectangle(3), &
                            specs_rectangle(4), specs_rectangle(5), specs_rectangle(6), &
                            ref_rectangle_lref, int(specs_rectangle(7)))
  endif
  ! Refine in ellipsoidal region
  if (ref_ellipse) then
    call gr_markEllipsoid(specs_ellipse(1), specs_ellipse(2), specs_ellipse(3), &
                          specs_ellipse(4), specs_ellipse(5), specs_ellipse(6), &
                          specs_ellipse(7), ref_ellipse_lref)
  endif
  ! Refine in spherical region
  if (ref_sphere) then
    call gr_markInRadius( specs_sphere(1), specs_sphere(2), specs_sphere(3), &
                          specs_sphere(4), specs_sphere(5), ref_sphere_lref)
  endif

  return

end subroutine Grid_markRefineDerefine
