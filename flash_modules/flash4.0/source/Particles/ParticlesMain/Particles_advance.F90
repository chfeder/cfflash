!!****if* source/Particles/ParticlesMain/Particles_advance
!!
!! NAME
!!
!!  Particles_advance
!!
!! SYNOPSIS
!!
!!  Particles_advance(real(in) :: dtOld,
!!                    real(in) :: dtNew)
!!
!! DESCRIPTION
!!
!!  Time advancement routine for the particle module.
!!  Calls passive and active versions
!!  
!! ARGUMENTS
!!
!!   dtOld -- not used in this first-order scheme
!!   dtNew -- current time increment
!!  
!!
!! SIDE EFFECTS
!!
!!  Updates the POS{X,Y,Z} and VEL{X,Y,Z} properties of particles in the particles structure.
!!  Sorts particles in the particles structure by calling Grid_sortParticles (indirectly).
!!
!! NOTES
!!
!!  No special handling is done for the first call - it is assumed that particle
!!  initialization fills in initial velocity components properly.
!!***

!===============================================================================

subroutine Particles_advance(dtOld,dtNew)

  use Particles_data, ONLY: particles, pt_numLocal, pt_maxPerProc, useParticles, & 
       pt_gcMaskForAdvance, pt_gcMaskSizeForAdvance, pt_meshMe, pt_typeInfo,&
       pt_indexList, pt_indexCount, pt_keepLostParticles, pt_numLost, &
       pt_reduceGcellFills
  use pt_interface
  use Grid_interface, ONLY : Grid_moveParticles, Grid_fillGuardCells, &
                             Grid_mapMeshToParticles, Grid_sortParticles
  use Particles_interface, ONLY: Particles_sinkMoveParticles, Particles_sinkSortParticles, &
                                 Particles_sinkAdvanceParticles, &
                                 Particles_sinkCreateAccrete, Particles_sinkOutflow, &
                                 Particles_sinkStellarEvolution, Particles_sinkHeating, &
                                 Particles_sinkSupernova, Particles_splitting, &
                                 Particles_dynamicCreationDestruction
  use Driver_data, ONLY : dr_globalMe

  implicit none

#include "constants.h"
#include "Flash.h"
#include "Particles.h"
#include "GridParticles.h"

  real, INTENT(in) :: dtOld, dtNew

  integer :: i, p_begin, p_end, p_count, pfor, pbak, lostNow, istat, ntot
  integer, dimension(MAXBLOCKS,NPART_TYPES) :: particlesPerBlk
  logical, parameter :: regrid = .false., move_particles_inside_splitting = .false.
  logical, save :: gcMaskLogged = .FALSE.

  character(len=*), parameter :: funcsig = "Particles_advance: "
#ifdef DEBUG_PARTICLES
  logical, parameter :: Debug = .true.
#else
  logical, parameter :: Debug = .false.
#endif

!!------------------------------------------------------------------------------
  ! Don't do anything if runtime parameter isn't set
  if (.not. useParticles) return

  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//"entering."

!We need a work around for pure active particle simulations.
  !NONEXISTENT is known to be -1, which does not conflict with 
  !real particle types.
  !Note the undefine at the end of this subroutine.

  ! Prepare guardcell data needed for particle interpolation.
  !
  ! Experimentation with passive particles (with the old way of advancing particles)
  ! has shown that at least 2 layers of guardcells need to be filled
  ! with updated data for vel[xyz] and density, in order to get the
  ! same results as for a full guardcell fill, when using native grid interpolation. - KW
  ! With "monotonic" interpolation, even more layers are needed. - KW

  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling Grid_fillGuardCells'
  if (pt_reduceGcellFills) then
     call Grid_fillGuardCells(CENTER_FACES, ALLDIR, unitReadsMeshDataOnly=.true.)
  else
     call Grid_fillGuardCells(CENTER, ALLDIR, &
          maskSize=pt_gcMaskSizeForAdvance, mask=pt_gcMaskForAdvance, &
          doLogMask=.NOT.gcMaskLogged)
  end if

  ! Sort particles there so we only have to move the minimum of them.
  !  Sort by type, and then within each sort by block
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling Grid_sortParticles'
#ifdef TYPE_PART_PROP
  call Grid_sortParticles(particles, NPART_PROPS, pt_numLocal, NPART_TYPES, &
                          pt_maxPerProc, particlesPerBlk, BLK_PART_PROP, TYPE_PART_PROP)
#else
  call Grid_sortParticles(particles, NPART_PROPS, pt_numLocal, NPART_TYPES, &
                          pt_maxPerProc, particlesPerBlk, BLK_PART_PROP)
#endif
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'Grid_sortParticles-1 done.'

  if(pt_keepLostParticles) then
     pfor=pt_numLocal
     do while(particles(BLK_PART_PROP, pt_numLocal)==LOST)
        pt_numLocal=pt_numLocal-1
     end do
     lostNow=pfor-pt_numLocal
     pt_numLost=pt_numLost+lostNow
     pbak=pt_maxPerProc-pt_numLost
     if(pbak<pt_numLocal)call Driver_abortFlash("no more space for lost particles")
     do i = 1,lostNow
        particles(:,pbak+i)=particles(:,pt_numLocal+i)
     end do
  end if

  ! Now update the pt_typeInfo data structure
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling pt_updateTypeDS-1'
  call pt_updateTypeDS(particlesPerBlk)

  !! Now do actual movement, advance particles in time
  !! Here we assume that all passive particles have identical
  !! integration. The active particles may also chose to have
  !! identical integration, but they also have to option of picking
  !! different integration methods for different types.

  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling pt_advance*'
  do i = 1, NPART_TYPES
     p_begin=pt_typeInfo(PART_TYPE_BEGIN,i)
     p_count=pt_typeInfo(PART_LOCAL,i)
     p_end=p_begin+p_count-1
     select case(pt_typeInfo(PART_ADVMETHOD,i))
     case(RUNGEKUTTA)
        call pt_advanceRK(dtOld,dtNew,particles(:,p_begin:p_end), p_count, i)
     case(MIDPOINT)
        call pt_advanceMidpoint(dtOld,dtNew,particles(:,p_begin:p_end), p_count, i)
     case(ESTI)
        call pt_advanceEsti(dtOld,dtNew,particles(:,p_begin:p_end), p_count, i)
     case(EULER_TRA)
        call pt_advanceEuler_passive(dtOld,dtNew,particles(:,p_begin:p_end), p_count, i)
     case(EULER_MAS)
        call pt_advanceEuler_active(dtOld,dtNew,particles(:,p_begin:p_end), p_count, i)
     case(LEAPFROG)
        call pt_advanceLeapfrog(dtOld,dtNew,particles(:,p_begin:p_end), p_count, i)
     case(LEAPFROG_COSMO)
        call pt_advanceLeapfrog_cosmo(dtOld,dtNew,particles(:,p_begin:p_end), p_count, i)
     case(CHARGED)
        call pt_advanceCharged(dtOld,dtNew,particles(:,p_begin:p_end), p_count)
     case(CUSTOM)
        call pt_advanceCustom(dtOld,dtNew,particles(:,p_begin:p_end), p_count, i)
     end select
  end do

  ! sink particle routines
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling sink particle routines'
  ! advance sink particles based on velocity and acceleration
  call Particles_sinkAdvanceParticles(dtNew)
  ! check for creation and accretion
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling Particles_sinkCreateAccrete'
  call Particles_sinkCreateAccrete(dtNew)
  ! check for sink particle outflow feedback
  call Particles_sinkOutflow(dtNew) ! make sure this is called *after* Particles_sinkCreateAccrete
  ! check for sink particle stellar evolution
  call Particles_sinkStellarEvolution() ! must be called *after* Particles_sinkCreateAccrete and *before* Particles_sinkHeating
  ! check for sink particle accretion heating feedback
  call Particles_sinkHeating() ! must be called *after* Particles_sinkCreateAccrete or Particles_sinkOutflow
  ! check for sink particle supernova feedback
  call Particles_sinkSupernova(dtNew)
  ! move sink particles to the right blocks/procs
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling Particles_sinkMoveParticles'
  call Particles_sinkMoveParticles(regrid)
  ! sort sink particles
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling Particles_sinkSortParticles'
  call Particles_sinkSortParticles()
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'done with sink routines.'

  ! see if we need to create/destroy particles
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling Particles_dynamicCreationDestruction'
  call Particles_dynamicCreationDestruction(.false.)

  ! see if we need to split particles
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling Particles_splitting'
  call Particles_splitting(move_particles_inside_splitting)

  ! Put the particles in the appropriate blocks if they've moved off
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling Grid_moveParticles'
  call Grid_moveParticles(particles, NPART_PROPS, pt_maxPerProc, pt_numLocal, pt_indexList, pt_indexCount, regrid)

  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling Grid_sortParticles'
#ifdef TYPE_PART_PROP
  call Grid_sortParticles(particles, NPART_PROPS, pt_numLocal, NPART_TYPES, &
                            pt_maxPerProc, particlesPerBlk, BLK_PART_PROP, TYPE_PART_PROP)
#else
  call Grid_sortParticles(particles, NPART_PROPS, pt_numLocal, NPART_TYPES, &
                            pt_maxPerProc, particlesPerBlk, BLK_PART_PROP)
#endif
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'Grid_sortParticles-2 done.'

  if(pt_keepLostParticles) then
     pfor=pt_numLocal
     do while(particles(BLK_PART_PROP,pt_numLocal)==LOST)
        pt_numLocal=pt_numLocal-1
     end do
     lostNow=pfor-pt_numLocal
     pt_numLost=pt_numLost+lostNow
     pbak=pt_maxPerProc-pt_numLost
     if(pbak<pt_numLocal)call Driver_abortFlash("no more space for lost particles")
     do i = 1,lostNow
        particles(:,pbak+i)=particles(:,pt_numLocal+i)
     end do
  end if

  ! Now update the pt_typeInfo data structure
  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'calling pt_updateTypeDS-2'
  call pt_updateTypeDS(particlesPerBlk)
  
  ! If predictive routines are used, they will need to sort and prepare for the
  !  next time step.  Since sorting is so expensive, we suffer code duplication
  !  and do it in the pt_preparePassive routines.
  ! Many algorithms use the stub routines.

  do i=1, NPART_TYPES
     if (pt_typeInfo(PART_ADVMETHOD,i)==ESTI)then
        p_begin=pt_typeInfo(PART_TYPE_BEGIN,i)
        p_count=pt_typeInfo(PART_LOCAL,i)
        p_end=p_begin+p_count-1
        call pt_prepareEsti(dtOld,dtNew,particles(:,p_begin:p_end),p_count,i)
     end if
  end do

  gcMaskLogged = .TRUE.

  if (Debug .and. dr_globalMe .eq. MASTER_PE) print *, funcsig//'exiting.'

  return

  !!-----------------------------------------------------------------------
end subroutine Particles_advance
