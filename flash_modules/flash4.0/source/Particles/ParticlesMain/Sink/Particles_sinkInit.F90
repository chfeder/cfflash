!!****if* source/Particles/ParticlesMain/Sink/Particles_sinkInit
!!
!! NAME
!!
!!  Particles_sinkInit
!!
!! SYNOPSIS
!!
!!  call Particles_sinkInit(logical, INTENT(in) :: restart)
!!
!! DESCRIPTION
!!
!!  Initializes sink particle properties.
!!
!! ARGUMENTS
!!
!!   restart - logical flag indicating restart or not
!!
!! NOTES
!!
!!   written by Christoph Federrath, 2008-2015
!!   ported to FLASH3.3/4 by Chalence Safranek-Shrader, 2010-2012
!!   modified by Nathan Goldbaum, 2012
!!   refactored for FLASH4 by John Bachan, 2012
!!
!!***

subroutine Particles_sinkInit(restart)

   use Particles_sinkData
   use Driver_data, ONLY : dr_globalMe
   use pt_sinkInterface, ONLY: pt_sinkPrepareEwald
   use PhysicalConstants_interface, ONLY : PhysicalConstants_get
   use RuntimeParameters_interface, ONLY : RuntimeParameters_get
   use Driver_interface, ONLY : Driver_abortFlash
   use Grid_interface, ONLY : Grid_getMinCellSize

   implicit none

#include "constants.h"
#include "Flash.h"

   logical, INTENT(in) :: restart

   integer :: ierr
   real :: delta_at_lrefmax, sink_accretion_radius, sink_softening_radius

   character(len=MAX_STRING_LENGTH) :: sink_softening_type_gas, sink_softening_type_sinks
   character(len=MAX_STRING_LENGTH) :: sink_integrator, grav_boundary_type

#ifdef FLASH_GRID_PARAMESH
   logical :: refineOnJeansLength
#endif

   if (dr_globalMe .eq. MASTER_PE) write(*,'(A)') ' Particles_sinkInit: ------------ Initializing Sink Particles ------------'

   call RuntimeParameters_get("useSinkParticles", useSinkParticles)
   call RuntimeParameters_get("sink_maxSinks", sink_maxSinks)
   call RuntimeParameters_get("sink_AdvanceSerialComputation", sink_AdvanceSerialComputation)
   call RuntimeParameters_get("sink_offDomainSupport", sink_offDomainSupport)
   call RuntimeParameters_get("sink_stop_on_creation", sink_stop_on_creation)
   call RuntimeParameters_get("sink_stop_on_total_mass", sink_stop_on_total_mass)
   call RuntimeParameters_get("sink_merging", sink_merging)
   call RuntimeParameters_get("sink_density_thresh_factor", sink_density_thresh_factor)

   ierr = 0
   allocate (NumParticlesPerBlock(MAXBLOCKS))
   NumParticlesPerBlock(:) = 0

   ! gravitational constant
   call PhysicalConstants_get("Newton", newton)

#ifdef FLASH_GRID_PARAMESH
   ! see if Jeans refinement is switched on if we are using AMR
   call RuntimeParameters_get("refineOnJeansLength", refineOnJeansLength)
   if ((.not. refineOnJeansLength) .and. (dr_globalMe .eq. MASTER_PE)) then
      write(*,'(A)') ' Particles_sinkInit: WARNING: refineOnJeansLength = .false.'
      write(*,'(A)') ' Particles_sinkInit:          Jeans refinement is recommended with sink particles.'
   endif
#endif

   ! set sink particle accretion and softening radius
   call Grid_getMinCellSize(delta_at_lrefmax)
   call RuntimeParameters_get("sink_accretion_radius", sink_accretion_radius)
   call RuntimeParameters_get("sink_softening_radius", sink_softening_radius)
   if ((sink_accretion_radius .lt. 1e-3) .or. (sink_accretion_radius .gt. 1e3)) then
      call Driver_abortFlash('Particles_sinkInit: sink_accretion_radius seems wrong (provide in # cells)!')
   endif
   accretion_radius = sink_accretion_radius * delta_at_lrefmax
   softening_radius = sink_softening_radius * delta_at_lrefmax

   ! set sink softening type (gas)
   call RuntimeParameters_get("sink_softening_type_gas", sink_softening_type_gas)
   select case (sink_softening_type_gas)
      case ("spline")
         softening_type_gas = 1
      case ("linear")
         softening_type_gas = 2
      case default
         sink_softening_type_gas = "linear"
         softening_type_gas = 2
         if (dr_globalMe .eq. MASTER_PE) print *, &
            'Particles_sinkInit: invalid sink_softening_type_gas specified. Using default: ', trim(sink_softening_type_gas)
   end select

   ! set sink softening type (sinks)
   call RuntimeParameters_get("sink_softening_type_sinks", sink_softening_type_sinks)
   select case (sink_softening_type_sinks)
   case ("spline")
      softening_type_sinks = 1
   case ("linear")
      softening_type_sinks = 2
   case default
      sink_softening_type_sinks = "spline"
      softening_type_sinks = 1
      if (dr_globalMe .eq. MASTER_PE) print *, &
         'Particles_sinkInit: invalid sink_softening_type_sinks specified. Using default: ', trim(sink_softening_type_sinks)
   end select

   ! get gravitational boundary conditions
   call RuntimeParameters_get("grav_boundary_type", grav_boundary_type)
   select case (grav_boundary_type)
   case ("isolated")
      sink_grav_boundary_type = 1
   case ("periodic")
      sink_grav_boundary_type = 2
   case default
      if (useSinkParticles) call Driver_abortFlash('Sink particles only work with isolated or periodic gravity boundaries.')
   end select

   ! integrator
   call RuntimeParameters_get("sink_integrator", sink_integrator)
   select case (sink_integrator)
   case ("leapfrog")
      integrator = 1
   case ("euler")
      integrator = 2
   case ("leapfrog_cosmo")
      integrator = 3
   case default
      sink_integrator = "leapfrog"
      integrator = 1
      if (dr_globalMe .eq. MASTER_PE) print *, &
         'Particles_sinkInit: invalid sink_integrator specified. Using default: ', trim(sink_integrator)
   end select

   ! write out sink particle module info
   if (dr_globalMe .eq. MASTER_PE) then
      if (useSinkParticles) then
         write(*,'(A,F6.2,A)') ' Particles_sinkInit: accretion radius set to ', &
                              sink_accretion_radius, ' cells at the highest level of refinement'
         write(*,'(A,F6.2,A)') ' Particles_sinkInit: softening radius set to ', &
                              sink_softening_radius, ' cells at the highest level of refinement'
         if ((sink_accretion_radius .lt. 2.0) .or. (sink_accretion_radius .gt. 3.0)) then
            write(*,'(A)') ' CAUTION: Sink particle accretion radius is not within the recommended range (2-3 cells)!'
            write(*,'(A)') '          Sink particle creation checks might fail!'
         endif
         write(*,'(A,ES16.9,A)') ' Particles_sinkInit: -> accretion radius = ', accretion_radius, &
                                 ' * (1+redshift) in absolute simulation units'
         write(*,'(A,ES16.9,A)') ' Particles_sinkInit: -> softening radius = ', softening_radius, &
                                 ' * (1+redshift) in absolute simulation units'
         if (sink_merging) write(*,'(A)') ' Particles_sinkInit: Sink particles are allowed to merge.'
         write(*,'(2A)') ' Particles_sinkInit: sink_softening_type_gas   = ', trim(sink_softening_type_gas)
         write(*,'(2A)') ' Particles_sinkInit: sink_softening_type_sinks = ', trim(sink_softening_type_sinks)
         write(*,'(2A)') ' Particles_sinkInit: grav_boundary_type = ', trim(grav_boundary_type)
         write(*,'(3A)') ' Particles_sinkInit: sink_integrator = ', trim(sink_integrator)
      else
         write(*,'(A)') ' Particles_sinkInit: WARNING: Sink particles are compiled in, but useSinkParticles = .false.'
      endif
      write(*,'(A)') ' Particles_sinkInit: -----------------------------------------------------'
   endif

   ! Sink particle Properties (see Flash.h)
   ipx = POSX_PART_PROP
   ipy = POSY_PART_PROP
   ipz = POSZ_PART_PROP
   ipvx = VELX_PART_PROP
   ipvy = VELY_PART_PROP
   ipvz = VELZ_PART_PROP
   ipm = MASS_PART_PROP

   ipblk = BLK_PART_PROP
   iptag = TAG_PART_PROP
   ipcpu = PROC_PART_PROP

   iplx = X_ANG_PART_PROP
   iply = Y_ANG_PART_PROP
   iplz = Z_ANG_PART_PROP
   iplx_old = X_ANG_OLD_PART_PROP
   iply_old = Y_ANG_OLD_PART_PROP
   iplz_old = Z_ANG_OLD_PART_PROP
   ipt = CREATION_TIME_PART_PROP
   ipmdot = ACCR_RATE_PART_PROP 
   iold_pmass = OLD_PMASS_PART_PROP
   ipdtold = DTOLD_PART_PROP

   n_empty = sink_maxSinks
   RunningParticles = .true.
   if (.not. restart) then
      localnp = 0
      localnpf = 0
   end if

   if (sink_maxSinks .gt. maxsinks) then
      call Driver_abortFlash("Particles_sinkInit: sink_maxSinks > maxsinks. Must increase maxsinks in Particles_sinkData.")
   endif

   if (.not. restart) then !if we starting from scratch

      if (.not. allocated(particles_local)) then
         allocate (particles_local(pt_sinkParticleProps, sink_maxSinks), stat=ierr)
      endif
      if (ierr /= 0) then
         call Driver_abortFlash("Particles_sinkInit:  could not allocate particles_local array")
      endif

      if (.not. allocated(particles_global)) &
           allocate (particles_global(pt_sinkParticleProps, sink_maxSinks), stat=ierr)
      if (ierr /= 0) then
         call Driver_abortFlash("Particles_sinkInit:  could not allocate particles_global array for sink particles")
      endif

      particles_local = NONEXISTENT
      particles_global = NONEXISTENT

   end if  ! end of .not. restart

   if (allocated(particles_local)) particles_local(VELX_PART_PROP,:)=0.0
   if (allocated(particles_global)) particles_global(VELX_PART_PROP,:)=0.0

   if (allocated(particles_local)) particles_local(VELY_PART_PROP,:)=0.0
   if (allocated(particles_global)) particles_global(VELY_PART_PROP,:)=0.0

   if (allocated(particles_local)) particles_local(VELZ_PART_PROP,:)=0.0
   if (allocated(particles_global)) particles_global(VELZ_PART_PROP,:)=0.0

   if (.not. useSinkParticles) return

   ! See if we have to prepare an Ewald correction field, in case we run with periodic boundary conditions
   call pt_sinkPrepareEwald()

   return

end subroutine Particles_sinkInit
