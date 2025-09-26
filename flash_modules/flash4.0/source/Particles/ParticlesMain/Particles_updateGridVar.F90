!!****if* source/Particles/ParticlesMain/Particles_updateGridVar
!!
!! NAME
!!    Particles_updateGridVar
!!
!! SYNOPSIS
!!    Particles_updateGridVar(integer(IN)    :: partProp,
!!                            integer(IN)    :: varGrid,
!!                            integer(IN),optional    :: mode)
!!
!! DESCRIPTION
!!
!!    Updates the given grid variable with data from the given particle
!!    property.
!!
!! ARGUMENTS
!!               partProp:  Index of particle attribute to interpolate onto 
!!                          the mesh
!!               varGrid:   Index of gridded variable to receive interpolated
!!                          quantity
!!               mode:      (Optional) If zero (default), zero varGrid first;
!!                          if present and nonzero, do not zero varGrid first
!!                          but add data from particles to existing grid data.
!!
!! PARAMETERS
!! 
!!***

#include "Flash.h"
#include "Particles.h"
subroutine Particles_updateGridVar(partProp, varGrid, mode)
  use Grid_interface, only: Grid_mapParticlesToMesh
  use Particles_data, only: particles, pt_numLocal, pt_maxPerProc, pt_typeInfo
  implicit none

  integer, INTENT(in) :: partProp, varGrid
  integer, INTENT(in), optional :: mode
  
  integer :: mode1
  integer :: ty
  integer :: p0, pn
  logical :: skip
  
  mode1 = 0
  if(present(mode)) mode1 = mode
  
  do ty=1, NPART_TYPES
    skip = .false.
#if defined(MASS_PART_PROP) && defined(PDEN_VAR) && defined(SINK_PART_TYPE)
    skip = partProp==MASS_PART_PROP .and. varGrid==PDEN_VAR .and. ty==SINK_PART_TYPE
#endif
    if(.not. skip) then
      p0 = pt_typeInfo(PART_TYPE_BEGIN,ty)
      pn = pt_typeInfo(PART_LOCAL,ty)
      call Grid_mapParticlesToMesh( &
        particles(:,p0:p0+pn-1), NPART_PROPS, pn, pt_maxPerProc, &
        partProp, varGrid, mode1)
      mode1 = 1 ! stop zeroing out
    end if
  end do
end subroutine Particles_updateGridVar
