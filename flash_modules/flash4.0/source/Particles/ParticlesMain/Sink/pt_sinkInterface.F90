!!****ih* source/Particles/ParticlesMain/Sink/pt_sinkInterface
!!
!! This is the header file for the sink particle module that defines its
!! public interfaces.
!!
!!***

module pt_sinkInterface

  implicit none
  
  interface
    subroutine pt_sinkAccelSinksOnSinks(local_min_radius, local_max_accel)
      real, intent(out) :: local_min_radius, local_max_accel
    end subroutine pt_sinkAccelSinksOnSinks
  end interface
  
  interface
    subroutine pt_sinkCorrectForPeriodicBCs(distx, disty, distz)
      real, intent(INOUT)  :: distx, disty, distz
    end subroutine pt_sinkCorrectForPeriodicBCs
  end interface
  
  interface
    function pt_sinkCreateParticle(x, y, z, pt, block_no, MyPE)
      real, intent(IN)    :: x, y, z, pt
      integer, intent(IN) :: block_no, MyPE
      integer :: pt_sinkCreateParticle
    end function pt_sinkCreateParticle
  end interface
  
  interface
    subroutine pt_sinkDumpParticles(simtime, outfilename)
      real, intent(IN) :: simtime
      character(len=*), intent(IN) :: outfilename
    end subroutine pt_sinkDumpParticles
  end interface
  
  interface
    subroutine pt_sinkEwaldCorrection(x, y, z, xcorr, ycorr, zcorr)
      real, intent(IN)  :: x, y, z
      real, intent(OUT) :: xcorr, ycorr, zcorr
    end subroutine pt_sinkEwaldCorrection
  end interface
  
  interface
    subroutine pt_sinkFindList(x, y, z, rad, create_part, pindex_found, np_found)
      use Particles_sinkData, only: maxsinks
      real, intent(IN) :: x, y, z, rad
      logical, intent(IN) :: create_part
      integer, dimension(maxsinks), intent(OUT) :: pindex_found
      integer, intent(OUT) :: np_found
    end subroutine pt_sinkFindList
  end interface
  
  interface
    subroutine pt_sinkGatherGlobal(propinds, nprops)
      integer, dimension(:), intent(in), optional :: propinds
      integer, intent(in), optional :: nprops
    end subroutine pt_sinkGatherGlobal
  end interface
  
  interface
    subroutine pt_sinkGetSubCycleTimeStep(dt, dt_global, local_min_radius, local_max_accel)
      real, intent(out) :: dt
      real, intent(in)  :: dt_global, local_min_radius, local_max_accel
    end subroutine pt_sinkGetSubCycleTimeStep
  end interface
  
  interface
    subroutine pt_sinkMergingAfterCreation(delta_at_lrefmax)
      real, intent(IN) :: delta_at_lrefmax
    end subroutine pt_sinkMergingAfterCreation
  end interface
  
  interface
    subroutine pt_sinkParticleMerging(dt)
      real, intent(IN) :: dt
    end subroutine pt_sinkParticleMerging
  end interface
  
  interface
    subroutine pt_sinkPrepareEwald()
    end subroutine pt_sinkPrepareEwald
  end interface


  contains

  !==================================================================

  subroutine pt_sinkMPIAllReduceReal(array, size, QSindex)

    use Driver_interface, ONLY : Driver_abortFlash

    implicit none

    include "Flash_mpi.h"

#include "Flash.h"
#include "Particles.h"
#include "constants.h"

    real, dimension(:), intent(INOUT) :: array
    integer, dimension(:), intent(IN) :: QSindex
    integer, intent(IN) :: size

    real, allocatable, dimension(:) :: array_loc, array_tot

    integer :: p, ps, ierr

    ! --------------

    allocate(array_loc(size), stat=ierr)
    allocate(array_tot(size), stat=ierr)
    if (ierr .ne. 0) call Driver_abortFlash("pt_sinkAllReduceReal: Allocation fault.")

    array_tot(:) = 0.0

    ! copy in sorted order
    do p = 1, size
        ps = QSindex(p) ! sorted index (in global particle list)
        array_loc(p) = array(ps)
    enddo

    ! Now communicate (MPI_ALLREDUCE, SUM) to get the total contribution of all cells from all processors
    call MPI_ALLREDUCE(array_loc, array_tot, size, FLASH_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)

    ! copy sum back in previous order
    do p = 1, size
        ps = QSindex(p) ! sorted index (in global particle list)
        array(ps) = array_tot(p)
    enddo

    deallocate(array_loc)
    deallocate(array_tot)

  end subroutine pt_sinkMPIAllReduceReal


  !==================================================================

  subroutine pt_sinkMPIAllReduceInt(array, size, QSindex)

    use Driver_interface, ONLY : Driver_abortFlash

    implicit none

    include "Flash_mpi.h"

#include "Flash.h"
#include "Particles.h"
#include "constants.h"

    integer, dimension(:), intent(INOUT) :: array
    integer, dimension(:), intent(IN) :: QSindex
    integer, intent(IN) :: size

    integer, allocatable, dimension(:) :: array_loc, array_tot

    integer :: p, ps, ierr

    ! --------------

    allocate(array_loc(size), stat=ierr)
    allocate(array_tot(size), stat=ierr)
    if (ierr .ne. 0) call Driver_abortFlash("pt_sinkAllReduceInt: Allocation fault.") 

    array_tot(:) = 0

    ! copy in sorted order
    do p = 1, size
        ps = QSindex(p) ! sorted index (in global particle list)
        array_loc(p) = array(ps)
    enddo

    ! Now communicate (MPI_ALLREDUCE, SUM) to get the total contribution of all cells from all processors
    call MPI_ALLREDUCE(array_loc, array_tot, size, FLASH_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)

    ! copy sum back in previous order
    do p = 1, size
        ps = QSindex(p) ! sorted index (in global particle list)
        array(ps) = array_tot(p)
    enddo

    deallocate(array_loc)
    deallocate(array_tot)

  end subroutine pt_sinkMPIAllReduceInt


end module pt_sinkInterface
