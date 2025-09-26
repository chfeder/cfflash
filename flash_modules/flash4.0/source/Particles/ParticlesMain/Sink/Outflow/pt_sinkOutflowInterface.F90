!!****if* source/Particles/ParticlesMain/Sink/Outflow/pt_sinkOutflowInterface
!!
!! NAME
!!
!!  pt_sinkOutflowInterface
!!
!!
!! written by Martin Schroen (2011-2012) and Christoph Federrath (2012-2014)
!!
!!***

module pt_sinkOutflowInterface

 implicit none

#include "constants.h"
#include "Flash.h"
#include "Particles.h"

 type :: Sinktype 
     real, dimension(3)    :: dcm, massshape, delta, dp, prad1, prad2, l, dl
     integer, dimension(2) :: outflowcells
     real                  :: m, dm, angmom, v_kepler, macc, amomnorm, angle
 end type Sinktype

 type :: Celltype
     real, dimension(3) :: dist, dp, dprad, dprot, prot
     real               :: distance, theta, cos_theta, m, dm, ekin
     real               :: massshape, velshape, angmomshape
     integer            :: cone
 end type Celltype

 type :: Outflowtype
     integer :: mass_model, amom_model, time_model, speed_model, velocity_profile
     real    :: radius, mass_frac, amom_frac, theta, speed, cos_theta, time_char
 end type Outflowtype

 contains

!===============================================================================

  real function angular_smoothing_kernel(theta, theta_norm)

    implicit none
    real, intent(in) :: theta, theta_norm
    real, parameter :: pihalf = PI/2.

    if (abs(theta) .le. theta_norm) then 
       angular_smoothing_kernel = cos(theta/theta_norm*pihalf)
    else
       angular_smoothing_kernel = 0.0
    endif

    return

  end function angular_smoothing_kernel

!===============================================================================

  real function MassShape(Outflow, r, theta)

    implicit none
    type(Outflowtype), intent(in) :: Outflow
    real, intent(in) :: r, theta
    real, parameter :: pihalf = PI/2.

    if (Outflow%mass_model .eq. 0) then
        ! 0: no mass injection
        MassShape = 0.0
    else if (Outflow%mass_model .eq. 1) then
        ! 1: standard smoothing
        MassShape = sin(r/Outflow%radius*PI) * cos(theta/Outflow%theta*pihalf)
    else if (Outflow%mass_model .eq. 2) then
        ! 2: radial smoothing only; could be used for isotropic winds later
        MassShape = cos(r/Outflow%radius*pihalf)
    else
        ! constant
        MassShape = 1.0
    end if

    return

  end function MassShape

!===============================================================================

  real function VelocityShape(Outflow, theta)

    implicit none
    type(Outflowtype), intent(in) :: Outflow
    real, intent(in) :: theta

    if (Outflow%velocity_profile .eq. 1) then
        ! 1: makes an outflow (0.25*v_max) + jet (v_max) component
        ! (see e.g., Banerjee & Pudritz 2006; Machida et al. 2008)
        VelocityShape = 0.25*angular_smoothing_kernel(theta, Outflow%theta) + &
                        0.75*angular_smoothing_kernel(theta, Outflow%theta/6.0)
    else
        ! constant
        VelocityShape = 1.0
    end if

    return

  end function VelocityShape

!===============================================================================

  real function AmomShape(Outflow)

    implicit none
    type(Outflowtype), intent(in) :: Outflow
    real, parameter :: pihalf = PI/2.

    if (Outflow%amom_model .eq. 0) then
        ! 0: no angular momentum transfer
        AmomShape = 0.0
    else if (Outflow%amom_model .eq. 1) then
        ! 1: standard: constant distribution
        ! (note that the mass smoothing will also affect the smoothing of the momentum transfer,
        ! so don't use smoothing again for the angular momentum transfer
        AmomShape = 1.0
    else
        ! constant
        AmomShape = 1.0
    end if

    return

  end function AmomShape

!==================================================================

  real function TimeShape(Outflow, pt)

    implicit none
    type(Outflowtype), intent(in) :: Outflow
    real, intent(in) :: pt
    real, parameter :: pihalf = PI/2.

    ! pt is the time after sink creation

    if (Outflow%time_model .eq. 1) then
        ! 1: standard: linear ramp
        if (pt .lt. Outflow%time_char) then
            TimeShape = pt/Outflow%time_char
        else
            TimeShape = 1.0
        endif
    else
        ! constant
        TimeShape = 1.0
    end if

    return

  end function TimeShape
  
!==================================================================

  real function SpeedShape(Outflow, SP)

    implicit none
    type(Outflowtype),  intent(in) :: Outflow
    type(Sinktype), intent(in) :: SP
    real, parameter :: msol_half = 9.945e32

    if (Outflow%speed_model .eq. 1) then
        ! 1: scale with sqrt(M)
        SpeedShape = sqrt(SP%m / msol_half)
    else if (Outflow%speed_model .eq. 2) then
        ! 2: scale with sqrt(M), crop at 10 Msun
        if (SP%m .le. 1.989e34) then
            SpeedShape = sqrt(SP%m / msol_half)
        else
            SpeedShape = 0.0
        endif
    else
        ! constant
        SpeedShape = 1.0
    end if

    return

  end function SpeedShape

!==================================================================

  subroutine check_arrays(label)

    use Particles_sinkData
    use Driver_interface, ONLY : Driver_abortFlash
    use Driver_data, ONLY : dr_globalMe
    use Grid_interface, ONLY : Grid_getCellCoords, & 
         Grid_getBlkPhysicalSize, Grid_getBlkPtr, Grid_releaseBlkPtr,  & 
         Grid_getBlkIndexLimits, Grid_getListOfBlocks
    use Logfile_interface, ONLY : Logfile_stamp

#include "constants.h"
#include "Flash.h"
#include "Particles.h"

    implicit none

    character(len=80), intent(in) :: label

    integer :: blockCount, lb, blockID
    integer, dimension(MAXBLOCKS) :: blockList
  
    real, pointer, dimension(:,:,:,:) :: solnData
    real, dimension(:), allocatable :: xc, yc, zc

    integer, dimension(2,MDIM) :: blkLimits, blkLimitsGC

    logical, save       :: first_call = .true.

    integer, save       :: iXcoord, iYcoord, iZcoord, izn
    integer, save       :: MyPE, MasterPE

    integer             :: size_x, size_y, size_z, i, j, k, p
    real                :: dx_block, dy_block, dz_block, dVol
    real                :: size(3)

    if (first_call) then
      MyPE = dr_globalMe
      MasterPE = MASTER_PE
      iXcoord  = IAXIS
      iYcoord  = JAXIS
      iZcoord  = KAXIS
      izn = CENTER
      first_call = .false.
    endif


    if (MyPE .eq. MasterPE) write(*,'(2A)') trim(label), 'check_arrays: checking global particle arrays...'

    do p = 1, localnpf

          if (isnan(particles_global(ipvy,p)) .or. isinf(particles_global(ipvy,p))) then
             print *, trim(label), 'check_arrays: unphysical particle vel y (global list): ', particles_global(ipvy,p)
             write (*,'(2A,1(1X,I5))') trim(label), 'tag = ', int(particles_global(iptag,p))
          endif

    enddo


    if (MyPE .eq. MasterPE) write(*,'(2A)') trim(label), 'check_arrays: checking local particle arrays...'

    do p = 1, localnp

      if (isnan(particles_local(ipm,p)) .or. isinf(particles_local(ipm,p))) then
         print *, trim(label), 'check_arrays: unphysical particle mass: ', particles_local(ipm,p)
         write (*,'(2A,1(1X,I5))') trim(label), 'tag = ', int(particles_global(iptag,p))
      endif
                       
      if (isnan(particles_local(ipx,p)) .or. isinf(particles_local(ipx,p))) then
         print *, trim(label), 'check_arrays: unphysical particle pos x: ', particles_local(ipx,p)
         write (*,'(2A,1(1X,I5))') trim(label), 'tag = ', int(particles_global(iptag,p))
      endif
      if (isnan(particles_local(ipy,p)) .or. isinf(particles_local(ipy,p))) then
         print *, trim(label), 'check_arrays: unphysical particle pos y: ', particles_local(ipy,p)
         write (*,'(2A,1(1X,I5))') trim(label), 'tag = ', int(particles_global(iptag,p))
      endif
      if (isnan(particles_local(ipz,p)) .or. isinf(particles_local(ipz,p))) then
         print *, trim(label), 'check_arrays: unphysical particle pos z: ', particles_local(ipz,p)
         write (*,'(2A,1(1X,I5))') trim(label), 'tag = ', int(particles_global(iptag,p))
      endif

      if (isnan(particles_local(ipvx,p)) .or. isinf(particles_local(ipvx,p))) then
         print *, trim(label), 'check_arrays: unphysical particle vel x: ', particles_local(ipvx,p)
         write (*,'(2A,1(1X,I5))') trim(label), 'tag = ', int(particles_global(iptag,p))
      endif
      if (isnan(particles_local(ipvy,p)) .or. isinf(particles_local(ipvy,p))) then
         print *, trim(label), 'check_arrays: unphysical particle vel y: ', particles_local(ipvy,p)
         write (*,'(2A,1(1X,I5))') trim(label), 'tag = ', int(particles_global(iptag,p))
      endif
      if (isnan(particles_local(ipvz,p)) .or. isinf(particles_local(ipvz,p))) then
         print *, trim(label), 'check_arrays: unphysical particle vel z: ', particles_local(ipvz,p)
         write (*,'(2A,1(1X,I5))') trim(label), 'tag = ', int(particles_global(iptag,p))
      endif

    end do


    if (MyPE .eq. MasterPE) write(*,'(2A)') trim(label), 'check_arrays: checking grid arrays...'

    ! blockList,blockCount used to be passed in as args but not anymore
    call Grid_getListOfBlocks(LEAF,blockList,blockCount)

    ! loop over leaf blocks (note that passed blockList only contains leafs)
    do lb = 1, blockCount

        blockID = blockList(lb)

        call Grid_getBlkPtr(blockID,solnData)

        call Grid_getBlkIndexLimits(blockID, blkLimits, blkLimitsGC)
        size_x = blkLimitsGC(HIGH,IAXIS)-blkLimitsGC(LOW,IAXIS) + 1
        size_y = blkLimitsGC(HIGH,JAXIS)-blkLimitsGC(LOW,JAXIS) + 1
        size_z = blkLimitsGC(HIGH,KAXIS)-blkLimitsGC(LOW,KAXIS) + 1

        allocate(xc(size_x))
        allocate(yc(size_y))
        allocate(zc(size_z))

        call Grid_getCellCoords(iXcoord, blockID, izn, .true., xc, size_x)
        call Grid_getCellCoords(iYcoord, blockID, izn, .true., yc, size_y)
        call Grid_getCellCoords(iZcoord, blockID, izn, .true., zc, size_z)

        call Grid_getBlkPhysicalSize(blockID,size)
        dx_block = size(1)/real(NXB)
        dy_block = size(2)/real(NYB)
        dz_block = size(3)/real(NZB)
        dVol = dx_block*dy_block*dz_block

        ! loop over cells (not including guard cells)
        do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
           do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
              do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

                if (isnan(solnData(DENS_VAR,i,j,k)) .or. isinf(solnData(DENS_VAR,i,j,k)) .or. solnData(DENS_VAR,i,j,k) .lt. 0) then
                   print *, trim(label), 'check_arrays: unphysical density: ', solnData(DENS_VAR,i,j,k)
                   write (*,'(2A,3(1X,ES16.9))') trim(label), 'x,y,z = ', xc(i), yc(j), zc(k)
                endif
#ifdef PRES_VAR
                if (isnan(solnData(PRES_VAR,i,j,k)) .or. isinf(solnData(PRES_VAR,i,j,k)) .or. solnData(PRES_VAR,i,j,k) .lt. 0) then
                   print *, trim(label), 'check_arrays: unphysical pressure: ', solnData(PRES_VAR,i,j,k)
                   write (*,'(2A,3(1X,ES16.9))') trim(label), 'x,y,z = ', xc(i), yc(j), zc(k)
                endif
#endif
#ifdef ENER_VAR
                if (isnan(solnData(ENER_VAR,i,j,k)) .or. isinf(solnData(ENER_VAR,i,j,k)) .or. solnData(ENER_VAR,i,j,k) .lt. 0) then
                   print *, trim(label), 'check_arrays: unphysical total energy: ', solnData(ENER_VAR,i,j,k)
                   write (*,'(2A,3(1X,ES16.9))') trim(label), 'x,y,z = ', xc(i), yc(j), zc(k)
                endif
#endif
#ifdef EINT_VAR
                if (isnan(solnData(EINT_VAR,i,j,k)) .or. isinf(solnData(EINT_VAR,i,j,k)) .or. solnData(EINT_VAR,i,j,k) .lt. 0) then
                   print *, trim(label), 'check_arrays: unphysical internal energy: ', solnData(EINT_VAR,i,j,k)
                   write (*,'(2A,3(1X,ES16.9))') trim(label), 'x,y,z = ', xc(i), yc(j), zc(k)
                endif
#endif
                if (isnan(solnData(VELX_VAR,i,j,k)) .or. isinf(solnData(VELX_VAR,i,j,k))) then
                   print *, trim(label), 'check_arrays: unphysical x velocity: ', solnData(VELX_VAR,i,j,k)
                   write (*,'(2A,3(1X,ES16.9))') trim(label), 'x,y,z = ', xc(i), yc(j), zc(k)
                endif
                if (isnan(solnData(VELY_VAR,i,j,k)) .or. isinf(solnData(VELY_VAR,i,j,k))) then
                   print *, trim(label), 'check_arrays: unphysical y velocity: ', solnData(VELY_VAR,i,j,k)
                   write (*,'(2A,3(1X,ES16.9))') trim(label), 'x,y,z = ', xc(i), yc(j), zc(k)
                endif
                if (isnan(solnData(VELZ_VAR,i,j,k)) .or. isinf(solnData(VELZ_VAR,i,j,k))) then
                   print *, trim(label), 'check_arrays: unphysical z velocity: ', solnData(VELZ_VAR,i,j,k)
                   write (*,'(2A,3(1X,ES16.9))') trim(label), 'x,y,z = ', xc(i), yc(j), zc(k)
                endif

              end do
           end do
        end do

        call Grid_releaseBlkPtr(blockID,solnData)

        deallocate(xc)
        deallocate(yc)
        deallocate(zc)

    end do

    if (MyPE .eq. MasterPE) write(*,'(2A)') trim(label), 'check_arrays: done.'

  end subroutine check_arrays

!==================================================================

  logical function isinf(a)
    real, intent(in) :: a
    real, parameter :: PositiveHuge = +HUGE(real(1.0))
    real, parameter :: NegativeHuge = -HUGE(real(1.0))
    if ((a.lt.NegativeHuge).or.(a.gt.PositiveHuge)) then
      isinf = .true.
    else
      isinf = .false.
    end if
    return
  end function isinf

end module pt_sinkOutflowInterface
