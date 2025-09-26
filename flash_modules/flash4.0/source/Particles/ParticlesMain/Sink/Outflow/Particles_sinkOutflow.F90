!!****if* source/Particles/ParticlesMain/Sink/Outflow/Particles_sinkOutflow
!!
!! NAME
!!
!!  Particles_sinkOutflow
!!
!! SYNOPSIS
!!
!!  call Particles_sinkOutflow(real, intent(IN)  :: dt)
!!
!! DESCRIPTION
!!
!!  Handles jet and outflow feedback from sink particles. Given some accreted mass, a
!!  physically calibrated fraction of that mass is transferred back to the gas with a
!!  collimated momentum and angular momentum that was calibrated with dedicated simulations
!!  and motivated by observations and theoretical outflow/jet models. For details, see
!!  Federrath et al. (2014, ApJ 790, 128).
!!
!! ARGUMENTS
!!
!!   dt : time step
!!
!! NOTES
!!
!!   started by Susanne Horn, 2008
!!   modified by Christoph Federrath, 2010
!!   rewritten and refined by Martin Schroen, 2012
!!   ported to FLASH4 by Christoph Federrath, 2013
!!   completely overhauled, simplified and improved by Christoph Federrath, 2013-2020
!!
!!***

subroutine Particles_sinkOutflow(dt)

    use Particles_sinkData
    use pt_sinkOutflowInterface
    use pt_sinkSort
    use pt_sinkInterface, ONLY: pt_sinkFindList, pt_sinkGatherGlobal, &
                                pt_sinkDumpParticles, pt_sinkCorrectForPeriodicBCs, &
                                pt_sinkMPIAllReduceReal, pt_sinkMPIAllReduceInt
    use Driver_interface, ONLY : Driver_abortFlash
    use Driver_data, ONLY : dr_globalMe, dr_simTime
    use RuntimeParameters_interface, ONLY : RuntimeParameters_get
    use Grid_interface, ONLY : Grid_getCellCoords, Grid_getBlkPhysicalSize, &
                               Grid_getBlkPtr, Grid_releaseBlkPtr, &
                               Grid_getBlkIndexLimits, Grid_getListOfBlocks, Grid_getMinCellSize
    use Eos_interface, ONLY : Eos_wrapped
    use Timers_interface, ONLY : Timers_start, Timers_stop

    implicit none

#include "constants.h"
#include "Flash.h"
#include "Particles.h"
#include "Eos.h"
#include "GridParticles.h"
    include "Flash_mpi.h"

    real, intent(IN) :: dt

    type(Outflowtype), save :: Outflow
    integer, save           :: MyPE, MasterPE
    real, save              :: delta_at_lrefmax, outflow_radius
    real, save              :: xmin, xmax, ymin, ymax, zmin, zmax
    logical, save           :: first_call = .true.
    character(4), save      :: MyPEchar, charfile
    character(9), save      :: SPchar

    integer :: blockCount
    integer, dimension(MAXBLOCKS) :: blockList
  
    real, pointer, dimension(:,:,:,: ) :: solnData
    real, dimension(:), allocatable :: xc, yc, zc

    type(Sinktype), dimension(maxsinks) :: SP
    type(Celltype) :: Cell

    integer, allocatable, dimension(:) :: id_sorted, QSindex
    integer :: ierr, p, dir

    integer, dimension(maxsinks) :: pindex_found
    integer             :: np_found, npf
    real, dimension(7)  :: tempvar
    real                :: dx_block, dy_block, dz_block, dVol, time, hilf, hilf2
    real                :: px_old, py_old, pz_old
    real                :: size(3)
    integer             :: ip, jp, kp
    integer             :: size_x, size_y, size_z
    integer, dimension(2,MDIM) :: blkLimits, blkLimitsGC

    integer, dimension(MAXBLOCKS) :: affected_blocks
    integer             :: blockID, lb, abn, iostat
    logical             :: affected_block

    integer, parameter  :: fu_outflowlog = 46, fu_outflowdat = 47
    logical             :: opened_successful = .false.
    logical, parameter  :: debug = .false.

#ifdef ALFL_MSCALAR
    real                :: alfl_dens
#endif

    integer, parameter :: gather_nprops = 3
    integer, dimension(gather_nprops), save :: gather_propinds = &
      (/ integer :: X_ANG_OLD_PART_PROP, Y_ANG_OLD_PART_PROP, Z_ANG_OLD_PART_PROP /)

#define get_pno(arg1) ((arg1)/65536)
#define get_ppe(arg1) ((arg1) - get_pno(arg1)*65536)

    ! ================= first call =================
    if (first_call) then

        call RuntimeParameters_get("outflow_mass_model",       Outflow%mass_model)
        call RuntimeParameters_get("outflow_mass_fraction",    Outflow%mass_frac)
        call RuntimeParameters_get("outflow_speed_model",      Outflow%speed_model)
        call RuntimeParameters_get("outflow_speed",            Outflow%speed)
        call RuntimeParameters_get("outflow_velocity_profile", Outflow%velocity_profile)
        call RuntimeParameters_get("outflow_amom_model",       Outflow%amom_model)
        call RuntimeParameters_get("outflow_amom_fraction",    Outflow%amom_frac)
        call RuntimeParameters_get("outflow_time_model",       Outflow%time_model)
        call RuntimeParameters_get("outflow_time_char",        Outflow%time_char)
        call RuntimeParameters_get("outflow_radius",           outflow_radius)
        call RuntimeParameters_get("outflow_theta",            Outflow%theta)

        if (Outflow%mass_model .le. 0) return

        ! convert outflow radius to physical size
        if ((outflow_radius .lt. 1e-3) .or. (outflow_radius .gt. 1e3)) then
          call Driver_abortFlash('Particles_sinkOutflow: outflow_radius seems wrong (provide in # cells)!')
        endif
        call Grid_getMinCellSize(delta_at_lrefmax)
        Outflow%radius = outflow_radius * delta_at_lrefmax

        MyPE = dr_globalMe
        MasterPE = MASTER_PE

        ! if outflow_time_char not set, then set it based on radius and velocity
        ! this makes the outflow reach full power when it would have propagated to
        ! the outflow radius (added by CF 2013)
        if (Outflow%time_char .le. 0.) then
           Outflow%time_char = Outflow%radius / Outflow%speed
           if (MyPE == MasterPE) write(*,'(A,ES16.9,A)') &
              'Particles_sinkOutflow: outflow_time_char set to ', Outflow%time_char, &
              ' based on outflow_radius and outflow_speed.'
        endif

        write (charfile,'(I4.4)') MyPE ! assumes not more than 10000 processors in use
        MyPEchar = charfile

        call RuntimeParameters_get("xmin", xmin)
        call RuntimeParameters_get("xmax", xmax)
        call RuntimeParameters_get("ymin", ymin)
        call RuntimeParameters_get("ymax", ymax)
        call RuntimeParameters_get("zmin", zmin)
        call RuntimeParameters_get("zmax", zmax)

        ! print the number of cells used for the sink outflow radius
        if (MyPE == MasterPE) write(*,'(A,F6.2,A)') &
           'Particles_sinkOutflow: You have set the sink outflow radius to ', &
           Outflow%radius/delta_at_lrefmax, ' cells at the highest level of refinement.'
        if (Outflow%radius/delta_at_lrefmax .LT. 8.0 .OR. Outflow%radius/delta_at_lrefmax .GT. 32.0) then
           if (MyPE == MasterPE) write(*,'(A)') &
               'CAUTION: Sink outflow radius is not within the recommended range (8-32 cells; 16 cells recommended)!'
        endif

        Outflow%theta = Outflow%theta*2.0*PI/360. ! convert deg to rad
        Outflow%cos_theta = cos(Outflow%theta)

        ! prepare file output
        if (MyPE == MasterPE) then
            open(fu_outflowlog, file='outflow.log', position='APPEND')
            write(fu_outflowlog,'(A)') 'Log file for the sink outflow module.'
            if (Outflow%mass_frac .le. 1) then
                write(fu_outflowlog,'(1(1X,A5),1(1X,I2),1(1X,A30),1(1X,F6.2))') 'MODEL', Outflow%mass_model, &
                  'mass transfer fraction', Outflow%mass_frac
            else
                write(fu_outflowlog,'(1(1X,A5),1(1X,I2),1(1X,A30),1(1X,ES16.9))') 'MODEL', Outflow%mass_model, &
                  'constant mass transfer', Outflow%mass_frac
            endif
            write(fu_outflowlog,'(1(1X,A5),1(1X,I2),1(1X,A30),1(1X,F6.2))') 'MODEL', Outflow%amom_model, &
              'amom transfer fraction', Outflow%amom_frac
            write(fu_outflowlog,'(1(1X,A5),1(1X,I2),1(1X,A30),1(1X,ES16.9))') 'MODEL', Outflow%time_model, &
              'characteristic time evolution', Outflow%time_char
            write(fu_outflowlog,'(1(1X,A5),1(1X,I2),1(1X,A30))') 'MODEL', Outflow%speed_model, &
              'outflow speed model'
            write(fu_outflowlog,'(1(1X,A5),1(1X,A11),1(1X,F16.9),A)') 'PARAM', 'theta', Outflow%theta, &
              ' rad half opening angle.'
            write(fu_outflowlog,'(1(1X,A5),1(1X,A11),1(1X,ES16.9),1(1X,A8),1(1X,F6.2),A)') 'PARAM', 'radius', &
              Outflow%radius, 'This is ', Outflow%radius/delta_at_lrefmax, ' cells at the highest level of refinement.'
            write(fu_outflowlog,'(1(1X,A5),1(1X,A11),1(1X,ES16.9),A)') 'PARAM', 'speed', Outflow%speed, &
              ' driving velocity.'

            write(fu_outflowlog,'(A)') 'See header for outflow*********.dat below. Notation:'
            write(fu_outflowlog,'(A)') '  M_out = outflowing mass'
            write(fu_outflowlog,'(A)') '  M_acc = accreted mass'
            write(fu_outflowlog,'(A)') '  L_out = outflowing angular momentum'
            write(fu_outflowlog,'(A)') '  L_acc = accreted angular momentum'
            write(fu_outflowlog,'(A)') '      S = sink spin before outflow'
            write(fu_outflowlog,'(A)') '     dR = sink position change during outflow'
            write(fu_outflowlog,'(A)') '   dV/V = fractional sink velocity change during outflow'
            write(fu_outflowlog,'(A)') 'N_cells = number of cells in outflow launching region (for cone 1 and cone 2 seperately)'
            write(fu_outflowlog,'(/1(1X,A16),1(1X,A10),11(1X,A16),2(1X,A10))') 'time', 'sink_tag', &
                'dM_out/dt', 'M_out/M_acc', '|dL_out|/dt', '|L_out|/|L_acc|', 'ang(L_out,S)[deg]', &
                'dR_x/cellwidth', 'dR_y/cellwidth', 'dR_z/cellwidth', 'dV_x/V_x', 'dV_y/V_y', 'dV_z/V_z', &
                'N_cells1', 'N_cells2'
            close(fu_outflowlog)
        endif

        first_call = .false.

    endif ! ================= end first call =================


    if (debug .and. (MyPE == MasterPE)) print *,'[', myPE, '] Particles_sinkOutflow: entering'

    ! return if outflow is off or if no sink particles present
    if ((Outflow%mass_model .le. 0) .or. (localnpf .le. 0)) return

    call Timers_start("sinkOutflow")

    call Timers_start("sinkOutflowInit")

    time = dr_simTime

    ! reset local variables
    SP(:)%dm = 0.0
    SP(:)%dcm(1) = 0.0
    SP(:)%dcm(2) = 0.0
    SP(:)%dcm(3) = 0.0
    SP(:)%dp(1) = 0.0
    SP(:)%dp(2) = 0.0
    SP(:)%dp(3) = 0.0
    SP(:)%angmom = 0.0
    SP(:)%amomnorm = 0.0
    SP(:)%prad1(1) = 0.0
    SP(:)%prad1(2) = 0.0
    SP(:)%prad1(3) = 0.0
    SP(:)%prad2(1) = 0.0
    SP(:)%prad2(2) = 0.0
    SP(:)%prad2(3) = 0.0
    SP(:)%l(1) = 0.0
    SP(:)%l(2) = 0.0
    SP(:)%l(3) = 0.0
    SP(:)%dl(1) = 0.0
    SP(:)%dl(2) = 0.0
    SP(:)%dl(3) = 0.0
    SP(:)%outflowcells(1) = 0
    SP(:)%outflowcells(2) = 0
    SP(:)%massshape(1) = 0.0
    SP(:)%massshape(2) = 0.0
    SP(:)%angle = 0.0

    ! Set local SP with particles_global variables.
    ! Note that we are using the global particle list throughout, execpt at the end,
    ! where we update the local particle list appropriately. Thus, ordering of the
    ! particle indexes is always according to the global list held on each processor.
    ! But remember that those indexes can be (and they actually are in general) different
    ! on each processor (see pt_sinkGatherGlobal), so we must sort all particles by tag below,
    ! before we reduce to global particle variables.

    ! Note that we assume that pt_sinkGatherGlobal was called recently in pt_sinkDumpParticles.
    ! Only communicate what's not been updated in pt_sinkDumpParticles and needed here
    ! (iplx_old, iply_old, iplz_old).
    call pt_sinkGatherGlobal(gather_propinds, gather_nprops)

    SP(1:localnpf)%m = particles_global(ipm,1:localnpf)
    SP(1:localnpf)%macc = dt * particles_global(ipmdot,1:localnpf)
    SP(1:localnpf)%l(1) = particles_global(iplx,1:localnpf)
    SP(1:localnpf)%l(2) = particles_global(iply,1:localnpf)
    SP(1:localnpf)%l(3) = particles_global(iplz,1:localnpf)
    SP(1:localnpf)%angmom  = sqrt(SP(1:localnpf)%l(1)**2 + SP(1:localnpf)%l(2)**2 + SP(1:localnpf)%l(3)**2)
    do p = 1, localnpf
      SP(p)%v_kepler = SpeedShape(Outflow, SP(p)) * Outflow%speed
    enddo

    call Timers_stop("sinkOutflowInit")

    ! ============= LOOP 1 (all blocks): gather information within outflow cones =============

    call Timers_start("sinkOutflowLoop1")

    call Grid_getListOfBlocks(LEAF,blockList,blockCount)

    abn = 0 ! affected blocks number

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

        affected_block = .false.
        call Grid_getCellCoords(IAXIS, blockID, CENTER, .true., xc, size_x)
        call Grid_getCellCoords(JAXIS, blockID, CENTER, .true., yc, size_y)
        call Grid_getCellCoords(KAXIS, blockID, CENTER, .true., zc, size_z)

        call Grid_getBlkPhysicalSize(blockID,size)
        dx_block = size(1)/real(NXB)
        dy_block = size(2)/real(NYB)
        dz_block = size(3)/real(NZB)
        dVol = dx_block*dy_block*dz_block

        ! loop over cells (not including guard cells)
        do kp = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
         do jp = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
          do ip = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

            ! reset outflow field
#ifdef OUTF_VAR
            solnData(OUTF_VAR,ip,jp,kp) = 0.0
#endif
#ifdef OADV_MSCALAR
            if (solnData(OADV_MSCALAR,ip,jp,kp) .le. 0) solnData(OADV_MSCALAR,ip,jp,kp) = 0.0
#endif
            ! LOOP particles in Outflow%radius (create must be false, so we can use the global list)
            call pt_sinkFindList(xc(ip), yc(jp), zc(kp), Outflow%radius, .false., pindex_found, np_found)

            if (np_found .GT. 0) then

              do npf = 1, np_found

                p = pindex_found(npf) ! this is the index in the global list, because we call pt_sinkFindList
                                      ! with create = .false.

                ! do not produce outflow immediately after sink creation,
                ! if dm_acc/dt = 0, if particle is near outflowing boundary
                if (time .gt. particles_global(ipt,p) &
                    .and. ((particles_global(ipmdot,p) .gt. 0.0) .or. (Outflow%mass_frac .gt. 1.0)) &
                    .and. ((sink_grav_boundary_type .eq. 2) .or. &
                           (particles_global(ipx,p) .gt. xmin+Outflow%radius .and. &
                            particles_global(ipx,p) .lt. xmax-Outflow%radius .and. &
                            particles_global(ipy,p) .gt. ymin+Outflow%radius .and. &
                            particles_global(ipy,p) .lt. ymax-Outflow%radius .and. &
                            particles_global(ipz,p) .gt. zmin+Outflow%radius .and. &
                            particles_global(ipz,p) .lt. zmax-Outflow%radius))) then

                    Cell%dist(1) = xc(ip) - particles_global(ipx,p)
                    Cell%dist(2) = yc(jp) - particles_global(ipy,p)
                    Cell%dist(3) = zc(kp) - particles_global(ipz,p)

                    if (sink_grav_boundary_type .eq. 2) call pt_sinkCorrectForPeriodicBCs(Cell%dist(1), Cell%dist(2), Cell%dist(3))

                    Cell%distance = sqrt( Cell%dist(1)**2 + Cell%dist(2)**2 + Cell%dist(3)**2)

                    ! if no spin or no speed, then stop
                    if ( SP(p)%angmom .gt. 0.0 .and. Cell%distance .gt. 0.0 .and. SP(p)%v_kepler .gt. 0.0) then

                        ! angle between sink spin and cell position inside r_outflow;
                        ! compare cos(chi) with cos(theta) for efficiency reasons: saves acos() for most cells
                        Cell%cos_theta = ( Cell%dist(1)*SP(p)%l(1) + Cell%dist(2)*SP(p)%l(2) + Cell%dist(3)*SP(p)%l(3) ) &
                            / Cell%distance / SP(p)%angmom
                        Cell%cos_theta = max(-1.0e0, min(1.0e0, Cell%cos_theta)) ! in cosine range
                        Cell%cone = 0

                        ! decide which of the 2 jets the cell relates to
                        if (Cell%cos_theta .gt. Outflow%cos_theta) then
                            Cell%theta = acos(Cell%cos_theta)
                            Cell%cone = 1
                        else if (-Cell%cos_theta .gt. Outflow%cos_theta) then
                            Cell%theta = acos(-Cell%cos_theta) ! acos(-x) == pi-acos(x)
                            Cell%cone = 2
                        endif

                        ! if cell is in cone
                        if ( Cell%cone .ne. 0 ) then

                            ! build integral sum of mass shape
                            Cell%massshape = MassShape(Outflow, Cell%distance, Cell%theta)
                            Cell%velshape = SP(p)%v_kepler * VelocityShape(Outflow, Cell%theta)
                            Cell%angmomshape = AmomShape(Outflow)

                            Cell%dm = Cell%massshape * TimeShape(Outflow, time - particles_global(ipt,p))
                            SP(p)%massshape(Cell%cone) = SP(p)%massshape(Cell%cone) + Cell%massshape
                            SP(p)%outflowcells(Cell%cone) = SP(p)%outflowcells(Cell%cone) + 1

                            ! dp_rad is the radial momentum change induced by the outflow
                            Cell%dprad(1) = Cell%dm * Cell%velshape * Cell%dist(1)/Cell%distance
                            Cell%dprad(2) = Cell%dm * Cell%velshape * Cell%dist(2)/Cell%distance
                            Cell%dprad(3) = Cell%dm * Cell%velshape * Cell%dist(3)/Cell%distance

                            if ( Cell%cone .eq. 1 ) then ! in 1st cone
                                SP(p)%prad1(1) = SP(p)%prad1(1) + Cell%dprad(1)
                                SP(p)%prad1(2) = SP(p)%prad1(2) + Cell%dprad(2)
                                SP(p)%prad1(3) = SP(p)%prad1(3) + Cell%dprad(3)
                            else ! in 2nd cone
                                SP(p)%prad2(1) = SP(p)%prad2(1) + Cell%dprad(1)
                                SP(p)%prad2(2) = SP(p)%prad2(2) + Cell%dprad(2)
                                SP(p)%prad2(3) = SP(p)%prad2(3) + Cell%dprad(3)
                            end if

                            ! block affected by outflow
                            affected_block = .true.

                        end if ! in cone
                    end if ! rotating
                end if ! not after sink creation
              end do ! particles
            end if ! particles found
          enddo ! i
         enddo ! j
        enddo ! k

        call Grid_releaseBlkPtr(blockID, solnData)

        deallocate(xc)
        deallocate(yc)
        deallocate(zc)

        ! update affected blocks list
        if ( affected_block ) then
            abn = abn + 1
            affected_blocks(abn) = blockID
        end if

    enddo ! blocks

    ! Sort global particle indexes (QSindex is valid and kept until the end!)
    allocate(id_sorted(localnpf), stat=ierr)
    allocate(QSindex(localnpf), stat=ierr)
    id_sorted(1:localnpf) = int(particles_global(iptag,1:localnpf))
    if (localnpf .gt. 0) call NewQsort_IN(id_sorted, QSindex) ! => sorted(i) = unsorted(QSindex(i))
    deallocate(id_sorted)

    ! Communicate (MPI_AllReduce) using particle ordering by tag
    call pt_sinkMPIAllReduceReal(SP(:)%massshape(1), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%massshape(2), localnpf, QSindex)
    call pt_sinkMPIAllReduceInt(SP(:)%outflowcells(1), localnpf, QSindex)
    call pt_sinkMPIAllReduceInt(SP(:)%outflowcells(2), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%prad1(1), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%prad1(2), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%prad1(3), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%prad2(1), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%prad2(2), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%prad2(3), localnpf, QSindex)

    do p = 1, localnpf

        ! CF 2013: fixed a FLASH2.5 bug that would cause delta = 0, if SP(p)%prad1 = 0.
        ! Plus: do x, y, z directions separately
        do dir = 1, 3
          if (SP(p)%prad1(dir) .eq. 0.0 .or. SP(p)%prad2(dir) .eq. 0.0) then
            SP(p)%delta(dir) = 1.0
          else
            SP(p)%prad1(dir) = (-1.0) * SP(p)%prad1(dir) / SP(p)%prad2(dir)
            ! the sign is - if pA/pB > 0 <=> pA, pB have same sign
            SP(p)%delta(dir) = sign(sqrt(abs(SP(p)%prad1(dir))), SP(p)%prad1(dir))
          end if
        enddo

    enddo ! loop over all particles

    call Timers_stop("sinkOutflowLoop1")

    ! =========== LOOP 2 (affected blocks): Gather information for amom shape only ===========

    call Timers_start("sinkOutflowLoop2")

    if (Outflow%amom_model .ne. 0) then

      do lb = 1, abn

        blockID = affected_blocks(lb)

        call Grid_getBlkPtr(blockID,solnData)

        call Grid_getBlkIndexLimits(blockID, blkLimits, blkLimitsGC)
        size_x = blkLimitsGC(HIGH,IAXIS)-blkLimitsGC(LOW,IAXIS) + 1
        size_y = blkLimitsGC(HIGH,JAXIS)-blkLimitsGC(LOW,JAXIS) + 1
        size_z = blkLimitsGC(HIGH,KAXIS)-blkLimitsGC(LOW,KAXIS) + 1

        allocate(xc(size_x))
        allocate(yc(size_y))
        allocate(zc(size_z))

        call Grid_getCellCoords(IAXIS, blockID, CENTER, .true., xc, size_x)
        call Grid_getCellCoords(JAXIS, blockID, CENTER, .true., yc, size_y)
        call Grid_getCellCoords(KAXIS, blockID, CENTER, .true., zc, size_z)

        call Grid_getBlkPhysicalSize(blockID,size)
        dx_block = size(1)/real(NXB)
        dy_block = size(2)/real(NYB)
        dz_block = size(3)/real(NZB)
        dVol = dx_block*dy_block*dz_block

        ! loop over cells (not including guard cells)
        do kp = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
         do jp = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
          do ip = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

            ! LOOP particles in Outflow%radius (create must be false, so we can use the global list)
            call pt_sinkFindList(xc(ip), yc(jp), zc(kp), Outflow%radius, .false., pindex_found, np_found)

            if (np_found .GT. 0) then

              do npf = 1, np_found

                p = pindex_found(npf) ! this is the index in the global list, because we call pt_sinkFindList
                                      ! with create = .false.

                ! do not produce outflow immediately after sink creation,
                ! if dm_acc/dt = 0, if particle is near outflowing boundary
                if (time .gt. particles_global(ipt,p) &
                    .and. ((particles_global(ipmdot,p) .gt. 0.0) .or. (Outflow%mass_frac .gt. 1.0)) &
                    .and. ((sink_grav_boundary_type .eq. 2) .or. &
                           (particles_global(ipx,p) .gt. xmin+Outflow%radius .and. &
                            particles_global(ipx,p) .lt. xmax-Outflow%radius .and. &
                            particles_global(ipy,p) .gt. ymin+Outflow%radius .and. &
                            particles_global(ipy,p) .lt. ymax-Outflow%radius .and. &
                            particles_global(ipz,p) .gt. zmin+Outflow%radius .and. &
                            particles_global(ipz,p) .lt. zmax-Outflow%radius))) then

                    Cell%dist(1) = xc(ip) - particles_global(ipx,p)
                    Cell%dist(2) = yc(jp) - particles_global(ipy,p)
                    Cell%dist(3) = zc(kp) - particles_global(ipz,p)

                    if (sink_grav_boundary_type .eq. 2) call pt_sinkCorrectForPeriodicBCs(Cell%dist(1), Cell%dist(2), Cell%dist(3))

                    Cell%distance = sqrt( Cell%dist(1)**2 + Cell%dist(2)**2 + Cell%dist(3)**2)

                    ! if no spin or no speed, then stop
                    if ( SP(p)%angmom .gt. 0.0 .and. Cell%distance .gt. 0.0 .and. SP(p)%v_kepler .gt. 0.0) then

                        ! angle between sink spin and cell position inside r_outflow;
                        ! compare cos(chi) with cos(theta) for efficiency reasons: saves acos() for most cells
                        Cell%cos_theta = ( Cell%dist(1)*SP(p)%l(1) + Cell%dist(2)*SP(p)%l(2) + Cell%dist(3)*SP(p)%l(3) ) &
                            / Cell%distance / SP(p)%angmom
                        Cell%cos_theta = max(-1.0e0, min(1.0e0, Cell%cos_theta)) ! in cosine range
                        Cell%cone = 0

                        ! decide which of the 2 jets the cell relates to
                        if (Cell%cos_theta .gt. Outflow%cos_theta) then
                            Cell%theta = acos(Cell%cos_theta)
                            Cell%cone = 1
                        else if (-Cell%cos_theta .gt. Outflow%cos_theta) then
                            Cell%theta = acos(-Cell%cos_theta) ! acos(-x) == pi-acos(x)
                            Cell%cone = 2
                        endif

                        ! if cell is in cone
                        if ( Cell%cone .ne. 0 ) then

                            ! build integral sum of mass shape
                            Cell%massshape = MassShape(Outflow, Cell%distance, Cell%theta) * &
                                             TimeShape(Outflow, time - particles_global(ipt,p))
                            Cell%angmomshape = AmomShape(Outflow)

                            ! momentum components associated with sink spin (sink spin cross relative cell position): S_sink x r
                            Cell%prot(1) = SP(p)%l(2)*Cell%dist(3) - SP(p)%l(3)*Cell%dist(2)
                            Cell%prot(2) = SP(p)%l(3)*Cell%dist(1) - SP(p)%l(1)*Cell%dist(3)
                            Cell%prot(3) = SP(p)%l(1)*Cell%dist(2) - SP(p)%l(2)*Cell%dist(1)

                            ! normalization factor (note that angmomshape=1 inside the outflow; massshape takes care of smoothing)
                            hilf = sqrt(Cell%prot(1)**2 + Cell%prot(2)**2 + Cell%prot(3)**2)
                            if (hilf .gt. 0.0) hilf = Cell%massshape * Cell%angmomshape / hilf

                            ! dp_rot = rotational momentum change induced by the outflow [here: dimensionless]
                            Cell%dprot(:) = Cell%prot(:) * hilf

                            ! note that the units of dl are here just [r] = cm
                            SP(p)%dl(1) = SP(p)%dl(1) + Cell%dist(2)*Cell%dprot(3) - Cell%dist(3)*Cell%dprot(2)
                            SP(p)%dl(2) = SP(p)%dl(2) + Cell%dist(3)*Cell%dprot(1) - Cell%dist(1)*Cell%dprot(3)
                            SP(p)%dl(3) = SP(p)%dl(3) + Cell%dist(1)*Cell%dprot(2) - Cell%dist(2)*Cell%dprot(1)

                        end if ! in cone
                    end if ! rotating
                end if ! not after sink creation
              end do ! particles
            end if ! particles found
          enddo ! i
         enddo ! j
        enddo ! k

        call Grid_releaseBlkPtr(blockID, solnData)

        deallocate(xc)
        deallocate(yc)
        deallocate(zc)

      enddo ! blocks

      ! Communicate (MPI_AllReduce) using particle ordering by tag
      call pt_sinkMPIAllReduceReal(SP(:)%dl(1), localnpf, QSindex)
      call pt_sinkMPIAllReduceReal(SP(:)%dl(2), localnpf, QSindex)
      call pt_sinkMPIAllReduceReal(SP(:)%dl(3), localnpf, QSindex)

      do p = 1, localnpf

        hilf = sqrt(SP(p)%dl(1)**2 + SP(p)%dl(2)**2 + SP(p)%dl(3)**2)

        if (hilf .eq. 0.0) then
            SP(p)%amomnorm = 0.0
        else
            ! CF 2013 (take the accreted fraction of angular momentum along the outflow axis)
            ! This has units of momentum [m*v], because [hilf] = cm
            SP(p)%amomnorm = Outflow%amom_frac * &
                 ( (SP(p)%l(1)-particles_global(iplx_old,p))*SP(p)%l(1) + &
                   (SP(p)%l(2)-particles_global(iply_old,p))*SP(p)%l(2) + &
                   (SP(p)%l(3)-particles_global(iplz_old,p))*SP(p)%l(3) ) / SP(p)%angmom / hilf
        endif

        ! clear these previously used fields to start accumulation below in the 3rd grid loop
        SP(p)%dl(1) = 0.0
        SP(p)%dl(2) = 0.0
        SP(p)%dl(3) = 0.0

      enddo ! loop over all particles

    endif ! Outflow%amom_model

    call Timers_stop("sinkOutflowLoop2")

    ! =========== LOOP 3 (affected blocks): Apply outflow properties to soln data ===========

    call Timers_start("sinkOutflowLoop3")

    do lb = 1, abn

        blockID = affected_blocks(lb)

        call Grid_getBlkPtr(blockID,solnData)

        call Grid_getBlkIndexLimits(blockID, blkLimits, blkLimitsGC)
        size_x = blkLimitsGC(HIGH,IAXIS)-blkLimitsGC(LOW,IAXIS) + 1
        size_y = blkLimitsGC(HIGH,JAXIS)-blkLimitsGC(LOW,JAXIS) + 1
        size_z = blkLimitsGC(HIGH,KAXIS)-blkLimitsGC(LOW,KAXIS) + 1

        allocate(xc(size_x))
        allocate(yc(size_y))
        allocate(zc(size_z))

        call Grid_getCellCoords(IAXIS, blockID, CENTER, .true., xc, size_x)
        call Grid_getCellCoords(JAXIS, blockID, CENTER, .true., yc, size_y)
        call Grid_getCellCoords(KAXIS, blockID, CENTER, .true., zc, size_z)

        call Grid_getBlkPhysicalSize(blockID,size)
        dx_block = size(1)/real(NXB)
        dy_block = size(2)/real(NYB)
        dz_block = size(3)/real(NZB)
        dVol = dx_block*dy_block*dz_block

        ! loop over cells (not including guard cells)
        do kp = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
         do jp = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
          do ip = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

            ! LOOP particles in Outflow%radius (create must be false, so we can use the global list)
            call pt_sinkFindList(xc(ip), yc(jp), zc(kp), Outflow%radius, .false., pindex_found, np_found)

            if (np_found .GT. 0) then

              do npf = 1, np_found

                p = pindex_found(npf) ! this is the index in the global list, because we call pt_sinkFindList
                                      ! with create = .false.

                ! do not produce outflow immediately after sink creation,
                ! if dm_acc/dt = 0, if particle is near outflowing boundary
                if (time .gt. particles_global(ipt,p) &
                    .and. ((particles_global(ipmdot,p) .gt. 0.0) .or. (Outflow%mass_frac .gt. 1.0)) &
                    .and. ((sink_grav_boundary_type .eq. 2) .or. &
                           (particles_global(ipx,p) .gt. xmin+Outflow%radius .and. &
                            particles_global(ipx,p) .lt. xmax-Outflow%radius .and. &
                            particles_global(ipy,p) .gt. ymin+Outflow%radius .and. &
                            particles_global(ipy,p) .lt. ymax-Outflow%radius .and. &
                            particles_global(ipz,p) .gt. zmin+Outflow%radius .and. &
                            particles_global(ipz,p) .lt. zmax-Outflow%radius))) then

                    Cell%dist(1) = xc(ip) - particles_global(ipx,p)
                    Cell%dist(2) = yc(jp) - particles_global(ipy,p)
                    Cell%dist(3) = zc(kp) - particles_global(ipz,p)

                    if (sink_grav_boundary_type .eq. 2) call pt_sinkCorrectForPeriodicBCs(Cell%dist(1), Cell%dist(2), Cell%dist(3))

                    Cell%distance = sqrt( Cell%dist(1)**2 + Cell%dist(2)**2 + Cell%dist(3)**2)

                    ! if no spin or no speed, then stop
                    if ( SP(p)%angmom .gt. 0.0 .and. Cell%distance .gt. 0.0 .and. SP(p)%v_kepler .gt. 0.0) then

                        ! angle between sink spin and cell position inside r_outflow;
                        ! compare cos(chi) with cos(theta) for efficiency reasons: saves acos() for most cells
                        Cell%cos_theta = ( Cell%dist(1)*SP(p)%l(1) + Cell%dist(2)*SP(p)%l(2) + Cell%dist(3)*SP(p)%l(3) ) &
                            / Cell%distance / SP(p)%angmom
                        Cell%cos_theta = max(-1.0e0, min(1.0e0, Cell%cos_theta)) ! in cosine range
                        Cell%cone = 0

                        ! decide which of the 2 jets the cell relates to
                        if (Cell%cos_theta .gt. Outflow%cos_theta) then
                            Cell%theta = acos(Cell%cos_theta)
                            Cell%cone = 1
                        else if (-Cell%cos_theta .gt. Outflow%cos_theta) then
                            Cell%theta = acos(-Cell%cos_theta) ! acos(-x) == pi-acos(x)
                            Cell%cone = 2
                        endif

                        ! if cell is in cone, build integral sum of mass shape
                        if ( Cell%cone .ne. 0 ) then

                            ! store old values for conservation laws
#ifdef ALFL_MSCALAR
                            Cell%m = (1.0-solnData(ALFL_MSCALAR,ip,jp,kp))*solnData(DENS_VAR,ip,jp,kp)*dVol
#else
                            Cell%m = solnData(DENS_VAR,ip,jp,kp)*dVol
#endif
                            Cell%ekin = 0.5*( solnData(VELX_VAR,ip,jp,kp)**2 + &
                                              solnData(VELY_VAR,ip,jp,kp)**2 + &
                                              solnData(VELZ_VAR,ip,jp,kp)**2 )

                            ! shape
                            Cell%massshape = MassShape(Outflow, Cell%distance, Cell%theta)
                            Cell%velshape = SP(p)%v_kepler * VelocityShape(Outflow, Cell%theta)
                            if (Outflow%amom_model .ne. 0) Cell%angmomshape = AmomShape(Outflow) &
                                * TimeShape(Outflow, time - particles_global(ipt,p))

                            ! change cell mass
                            Cell%dm = Cell%massshape / (SP(p)%massshape(1)+SP(p)%massshape(2)) &
                                * TimeShape(Outflow, time - particles_global(ipt,p))
                            if (Outflow%mass_frac .le. 1) then
                                Cell%dm = Cell%dm * Outflow%mass_frac * SP(p)%macc
                            else
                                Cell%dm = Cell%dm * Outflow%mass_frac * dt ! contains the mass outflow rate
                            endif
#ifdef ALFL_MSCALAR
                            ! in case we are using the Alfven limiter
                            alfl_dens = solnData(ALFL_MSCALAR,ip,jp,kp) * solnData(DENS_VAR,ip,jp,kp)
#endif
                            ! add outflow mass
                            solnData(DENS_VAR,ip,jp,kp) = solnData(DENS_VAR,ip,jp,kp) + Cell%dm / dVol
#ifdef ALFL_MSCALAR
                            ! update Alfven limiter
                            solnData(ALFL_MSCALAR,ip,jp,kp) = alfl_dens / solnData(DENS_VAR,ip,jp,kp)
#endif
                            ! radial momentum
                            if ( Cell%cone .eq. 1 ) then
                                Cell%dprad(1) = Cell%dm * Cell%velshape * Cell%dist(1)/Cell%distance / abs(SP(p)%delta(1))
                                Cell%dprad(2) = Cell%dm * Cell%velshape * Cell%dist(2)/Cell%distance / abs(SP(p)%delta(2))
                                Cell%dprad(3) = Cell%dm * Cell%velshape * Cell%dist(3)/Cell%distance / abs(SP(p)%delta(3))
                            else
                                Cell%dprad(1) = Cell%dm * Cell%velshape * Cell%dist(1)/Cell%distance * SP(p)%delta(1)
                                Cell%dprad(2) = Cell%dm * Cell%velshape * Cell%dist(2)/Cell%distance * SP(p)%delta(2)
                                Cell%dprad(3) = Cell%dm * Cell%velshape * Cell%dist(3)/Cell%distance * SP(p)%delta(3)
                            end if

                            ! rotational momentum (from angular momentum transfer)
                            Cell%prot(1) = SP(p)%l(2)*Cell%dist(3) - SP(p)%l(3)*Cell%dist(2)
                            Cell%prot(2) = SP(p)%l(3)*Cell%dist(1) - SP(p)%l(1)*Cell%dist(3)
                            Cell%prot(3) = SP(p)%l(1)*Cell%dist(2) - SP(p)%l(2)*Cell%dist(1)
                            if (Outflow%amom_model .ne. 0) then
                                ! transfer specified fraction of accreted angular momentum
                                hilf = sqrt(Cell%prot(1)**2 + Cell%prot(2)**2 + Cell%prot(3)**2)
                                if (hilf .gt. 0.0) hilf = Cell%massshape * Cell%angmomshape * SP(p)%amomnorm / hilf
                            else
                                ! transfer the angular momentum due to replacing mass from sink to surrounding gas
                                hilf =  Cell%dm / SP(p)%m / Cell%distance**2
                            end if
                            ! here now [dprot] = [m*v] (momentum due to rotation)
                            Cell%dprot(:) = Cell%prot(:) * hilf

                            ! total momentum (includes last term due to mass displacement)
                            Cell%dp(1) = Cell%dprad(1) + Cell%dprot(1) + Cell%dm * particles_global(ipvx,p)
                            Cell%dp(2) = Cell%dprad(2) + Cell%dprot(2) + Cell%dm * particles_global(ipvy,p)
                            Cell%dp(3) = Cell%dprad(3) + Cell%dprot(3) + Cell%dm * particles_global(ipvz,p)

                            ! change cell velocities
                            solnData(VELX_VAR,ip,jp,kp) = (Cell%m * solnData(VELX_VAR,ip,jp,kp) + Cell%dp(1)) / (Cell%m+Cell%dm)
                            solnData(VELY_VAR,ip,jp,kp) = (Cell%m * solnData(VELY_VAR,ip,jp,kp) + Cell%dp(2)) / (Cell%m+Cell%dm)
                            solnData(VELZ_VAR,ip,jp,kp) = (Cell%m * solnData(VELZ_VAR,ip,jp,kp) + Cell%dp(3)) / (Cell%m+Cell%dm)

                            ! change specific energy
#ifdef ENER_VAR
                            solnData(ENER_VAR,ip,jp,kp) = solnData(ENER_VAR,ip,jp,kp) + &
                                  0.5 * (solnData(VELX_VAR,ip,jp,kp)**2 + &
                                         solnData(VELY_VAR,ip,jp,kp)**2 + &
                                         solnData(VELZ_VAR,ip,jp,kp)**2) - Cell%ekin
#endif

                            ! mark outflow field
#ifdef OUTF_VAR
                            solnData(OUTF_VAR,ip,jp,kp) = solnData(OUTF_VAR,ip,jp,kp) + Cell%massshape
#endif
#ifdef OADV_MSCALAR
                            ! OADV_MSCALAR contains the fraction of the cell mass that was injected by the sink outflow
                            solnData(OADV_MSCALAR,ip,jp,kp) = (solnData(OADV_MSCALAR,ip,jp,kp)*Cell%m+Cell%dm)/(Cell%m+Cell%dm)
#endif
                            ! Sum up outflow properties for sink particle

                            ! mass += dm
                            SP(p)%dm = SP(p)%dm + Cell%dm

                            ! center of mass cm += dcm
                            SP(p)%dcm(1) = SP(p)%dcm(1) + Cell%dist(1)*Cell%dm
                            SP(p)%dcm(2) = SP(p)%dcm(2) + Cell%dist(2)*Cell%dm
                            SP(p)%dcm(3) = SP(p)%dcm(3) + Cell%dist(3)*Cell%dm

                            ! momentum p += dp 
                            SP(p)%dp(1) = SP(p)%dp(1) + Cell%dp(1)
                            SP(p)%dp(2) = SP(p)%dp(2) + Cell%dp(2)
                            SP(p)%dp(3) = SP(p)%dp(3) + Cell%dp(3)

                            ! angular momentum S += dl
                            SP(p)%dl(1) = SP(p)%dl(1) + Cell%dist(2)*Cell%dprot(3) - Cell%dist(3)*Cell%dprot(2)
                            SP(p)%dl(2) = SP(p)%dl(2) + Cell%dist(3)*Cell%dprot(1) - Cell%dist(1)*Cell%dprot(3)
                            SP(p)%dl(3) = SP(p)%dl(3) + Cell%dist(1)*Cell%dprot(2) - Cell%dist(2)*Cell%dprot(1)

                        endif ! inside outflow cone
                    endif ! pl gt 0 and rad gt 0
                endif ! outflow particle found
              end do ! particles for which cell is in Outflow%radius
            endif ! particle found
          enddo ! cells
         enddo ! cells
        enddo ! cells

        call Grid_releaseBlkPtr(lb, solnData)

        deallocate(xc)
        deallocate(yc)
        deallocate(zc)

        call Eos_wrapped(MODE_DENS_EI, blkLimits, lb)

    enddo ! blocks

    call Timers_stop("sinkOutflowLoop3")

    call Timers_start("sinkOutflowFinalize")

    ! Communicate (MPI_AllReduce) using particle ordering by tag
    call pt_sinkMPIAllReduceReal(SP(:)%dm, localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%dcm(1), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%dcm(2), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%dcm(3), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%dp(1), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%dp(2), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%dp(3), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%dl(1), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%dl(2), localnpf, QSindex)
    call pt_sinkMPIAllReduceReal(SP(:)%dl(3), localnpf, QSindex)

    ! Update sink particle properties
    ! (note that the 1:localnp elements are indentical in particles_global and particles_local)
    do p = 1, localnp

        ! computing angles
        hilf = sqrt(SP(p)%dl(1)**2 + SP(p)%dl(2)**2 + SP(p)%dl(3)**2)

        if (hilf .gt. 0.0) then
            ! angle(dl,l)
            hilf2 = (SP(p)%dl(1)*SP(p)%l(1) + SP(p)%dl(2)*SP(p)%l(2) + SP(p)%dl(3)*SP(p)%l(3)) / hilf / SP(p)%angmom
            hilf2 = max(-1.0e0, min(1.0e0, hilf2)) ! make sure this is in acos range
            SP(p)%angle = acos( hilf2 ) / PI * 180.
        endif

        write (SPchar,'(I9.9)') int(particles_local(iptag,p))

        if ( SP(p)%dm .ne. 0.0 ) then

            if (Outflow%mass_frac .le. 1) then
                tempvar(1) = particles_local(ipmdot,p)
            else
                tempvar(1) = Outflow%mass_frac
            endif

            ! update particle properties (conservation laws)
            ! old masses
            SP(p)%m = particles_local(ipm,p)

            ! mass conservation
            particles_local(ipm,p) = SP(p)%m - SP(p)%dm

            ! center of mass conservation
            tempvar(2) = particles_local(ipx,p)
            tempvar(3) = particles_local(ipy,p)
            tempvar(4) = particles_local(ipz,p)
            px_old = particles_local(ipx,p)
            py_old = particles_local(ipy,p)
            pz_old = particles_local(ipz,p)
            particles_local(ipx,p) = particles_local(ipx,p) - SP(p)%dcm(1) / particles_local(ipm,p)
            particles_local(ipy,p) = particles_local(ipy,p) - SP(p)%dcm(2) / particles_local(ipm,p)
            particles_local(ipz,p) = particles_local(ipz,p) - SP(p)%dcm(3) / particles_local(ipm,p)
            tempvar(2) = (particles_local(ipx,p) - tempvar(2)) / delta_at_lrefmax
            tempvar(3) = (particles_local(ipy,p) - tempvar(3)) / delta_at_lrefmax
            tempvar(4) = (particles_local(ipz,p) - tempvar(4)) / delta_at_lrefmax

            ! momentum conservation
            tempvar(5) = particles_local(ipvx,p)
            tempvar(6) = particles_local(ipvy,p)
            tempvar(7) = particles_local(ipvz,p)
            particles_local(ipvx,p) = (particles_local(ipvx,p)*SP(p)%m - SP(p)%dp(1)) / particles_local(ipm,p)
            particles_local(ipvy,p) = (particles_local(ipvy,p)*SP(p)%m - SP(p)%dp(2)) / particles_local(ipm,p)
            particles_local(ipvz,p) = (particles_local(ipvz,p)*SP(p)%m - SP(p)%dp(3)) / particles_local(ipm,p)
            tempvar(5) = particles_local(ipvx,p) / tempvar(5) - 1.0
            tempvar(6) = particles_local(ipvy,p) / tempvar(6) - 1.0
            tempvar(7) = particles_local(ipvz,p) / tempvar(7) - 1.0

            ! angular momentum conservation
            particles_local(iplx,p) = particles_local(iplx,p) - SP(p)%dl(1) - particles_local(ipm,p) * &
                                      ( (particles_local(ipy,p)-py_old)*particles_local(ipvz,p) - &
                                        (particles_local(ipz,p)-pz_old)*particles_local(ipvy,p)  )
            particles_local(iply,p) = particles_local(iply,p) - SP(p)%dl(2) - particles_local(ipm,p) * &
                                      ( (particles_local(ipz,p)-pz_old)*particles_local(ipvx,p) - &
                                        (particles_local(ipx,p)-px_old)*particles_local(ipvz,p)  )
            particles_local(iplz,p) = particles_local(iplz,p) - SP(p)%dl(3) - particles_local(ipm,p) * &
                                      ( (particles_local(ipx,p)-px_old)*particles_local(ipvy,p) - &
                                        (particles_local(ipy,p)-py_old)*particles_local(ipvx,p)  )

            ! OUTPUT outflow*.dat ! This should all be put together in one file with a proper header like sinks_evol.dat

            opened_successful = .false.
            do while (.not. opened_successful)
              open(fu_outflowdat, file='outflow'//SPchar//'.dat', action='write', iostat=iostat, position='APPEND')
              if (iostat.eq.0) then
               write(fu_outflowdat,'(1(1X,ES16.9),(1X,I10),11(1X,ES16.9),2(I10))') &
                time, int(particles_local(iptag,p)), &
                SP(p)%dm/dt, abs(SP(p)%dm/dt/tempvar(1)), &
                sqrt(SP(p)%dl(1)**2+SP(p)%dl(2)**2+SP(p)%dl(3)**2)/dt, &
                sqrt(SP(p)%dl(1)**2+SP(p)%dl(2)**2+SP(p)%dl(3)**2) &
                / (SP(p)%angmom - sqrt(particles_local(iplx_old,p)**2+ &
                                       particles_local(iply_old,p)**2+ &
                                       particles_local(iplz_old,p)**2)), &
                SP(p)%angle, & ! angle
                tempvar(2), tempvar(3), tempvar(4), & ! diff in position normalized to cell width
                tempvar(5), tempvar(6), tempvar(7), & ! diff in fraction of sink velocity
                SP(p)%outflowcells(1), SP(p)%outflowcells(2) ! number of cells in 1st and 2nd cone
               opened_successful = .true.
               close(fu_outflowdat)
              else
               write (*,'(A,I6,2A)') '[',myPE,'] Particles_sinkOutflow: could not open file for write. filename: ', &
                                     'outflow'//SPchar//'.dat'
               write (*,'(A,I6,A)') '[',myPE,'] Trying again...'
               call sleep(1) ! wait a second...
              endif
            enddo ! opened_successful

        end if ! SP(p)%dm .ne. 0.0

    enddo ! loop over all local particles

    deallocate(QSindex)

    call Timers_stop("sinkOutflowFinalize")

    ! write sink particle data to sinks_evol_after_outflow.dat
    call Timers_start("sinkOutflowDump")
    call pt_sinkDumpParticles(time, "sinks_evol_after_outflow.dat")
    call Timers_stop("sinkOutflowDump")

    if (debug .and. (MyPE == MasterPE)) print *,'[', myPE, '] Particles_sinkOutflow: exiting.'

    call Timers_stop("sinkOutflow")

    return

end subroutine Particles_sinkOutflow
