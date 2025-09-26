!!****if* source/Particles/ParticlesMain/Sink/Particles_sinkCreateAccrete
!!
!! NAME
!!
!!  Particles_sinkCreateAccrete
!!
!! SYNOPSIS
!!
!!  call Particles_sinkCreateAccrete(real, intent(IN) :: dt)
!!
!! DESCRIPTION
!!
!!  Handles creation of sink particles and accretion of gas onto sink particles.
!!  It also calls routines for sink particle merging and for dumping sink
!!  particle properties every time step (pt_sinkDumpParticles).
!!
!! ARGUMENTS
!!
!!   dt - the current simulation time step
!!
!! NOTES
!!
!!   written by Christoph Federrath, 2008-2022
!!   ported to FLASH3.3/4 by Chalence Safranek-Shrader, 2010-2012
!!   modified by Nathan Goldbaum, 2012
!!   refactored for FLASH4 by John Bachan, 2012
!!   renamed and cleaned by Christoph Federrath, 2013-2022
!!
!! Main references:
!!   Federrath et al. 2010, ApJ 713, 269 (https://ui.adsabs.harvard.edu/abs/2010ApJ...713..269F/abstract)
!!   Federrath et al. 2011, IAUS 270, 425 (https://ui.adsabs.harvard.edu/abs/2011IAUS..270..425F/abstract)
!!   Federrath et al. 2014, ApJ 790, 128 (https://ui.adsabs.harvard.edu/abs/2014ApJ...790..128F/abstract)
!!
!!***

subroutine Particles_sinkCreateAccrete(dt)

  use Particles_sinkData
  use Particles_data, ONLY: pt_posInitialized, pt_velInitialized, pt_reduceGcellFills
  use pt_sinkInterface, ONLY: pt_sinkGatherGlobal, pt_sinkMergingAfterCreation, pt_sinkFindList, &
                              pt_sinkParticleMerging, pt_sinkDumpParticles, pt_sinkCreateParticle, &
                              pt_sinkCorrectForPeriodicBCs
  use Driver_interface, ONLY : Driver_abortFlash
  use Driver_data, ONLY : dr_globalMe, dr_simTime
  use RuntimeParameters_interface, ONLY : RuntimeParameters_get
  use Grid_interface, ONLY : Grid_fillGuardCells, Grid_getCellCoords, Grid_getBlkPhysicalSize, &
                             Grid_getBlkPtr, Grid_releaseBlkPtr, Grid_getBlkIndexLimits, &
                             Grid_getListOfBlocks, Grid_getMinCellSize
  use Eos_interface, ONLY : Eos_wrapped
  use Cosmology_interface, ONLY : Cosmology_getRedshift
  use Logfile_interface, ONLY : Logfile_stamp, Logfile_stampVarMask
  use Timers_interface, ONLY : Timers_start, Timers_stop

#include "constants.h"
#include "Flash.h"
#include "Particles.h"
#include "Eos.h"
#include "GridParticles.h"

  implicit none

  real, intent(IN) :: dt

  logical, save :: first_call = .true.
  logical, save :: convergingFlowCheck, potentialMinCheck, JeansCheck, negativeEtotCheck, GasAccretionChecks
  logical, save :: gcMaskLogged = .false., gcmask1(NUNK_VARS), gcmask(NUNK_VARS)
  integer, save :: MyPE, MasterPE
  real, save :: density_thresh_factor, delta_at_lrefmax
  integer, save :: creationInfo

  integer :: blockCount
  integer, dimension(MAXBLOCKS) :: blockList

  real, pointer, dimension(:,:,:,:) :: solnData
  real, dimension(:), allocatable :: xc, yc, zc
  real, dimension(maxsinks) :: tot_mass, cm_x, cm_y, cm_z, vel_cm_x, vel_cm_y, vel_cm_z, &
                             & ang_x, ang_y, ang_z, etot, vr, radius
  integer, dimension(maxsinks) :: pindex_found
  integer             :: np_found, npf, pno_to_accrete
  real                :: egrav_gas, egrav_part, egrav, ekin, etot_min_inner_r_accr, etot_min
  logical             :: within_inner_r_accr
  real                :: size(3)
  real                :: dx_block, dy_block, dz_block, dVol, inner_r_accr
  real                :: mass, gpot, absgpot, total_sink_mass
  real                :: rad, pt, time, dvx, dvy, dvz
  real                :: x, y, z, vrad, distx, disty, distz, px_old, py_old, pz_old
  integer             :: ip, jp, kp, lp, nlp, npart

#if defined(MAGX_VAR) && defined(MAGY_VAR) && defined(MAGZ_VAR)
  real, save          :: mu_zero
  character(4), save  :: units
#endif

  logical, parameter  :: debug = .false.
  integer, parameter  :: funit_accretion_checks = 43
  integer, parameter  :: funit_accretion        = 44
  integer, parameter  :: ngc_sink_creation_check = 2
  real, parameter     :: ngc_sink_creation_check_radius_sqr = (ngc_sink_creation_check+1.0)**2
  integer             :: i1, j1, k1, ii1, jj1, kk1, ncells_in_vol
  real                :: vxcm_in_vol, vycm_in_vol, vzcm_in_vol, ekindisp_in_vol, etherm_in_vol, emag_in_vol
  real                :: mass_in_vol, maxgpot_in_vol, egravdeltapot_in_vol
  logical             :: create_sink, affected_block

  integer             :: lb, llb, llnblocks, blockID, pno, old_localnp

  integer, dimension(MAXBLOCKS) :: block_list

  integer, dimension(2,MDIM) :: blkLimits, blkLimitsGC

  integer :: size_x, size_y, size_z
  real :: dens, pres, cs2, density_thresh, dens_loc

#ifdef COSMOLOGY
  real :: redshift, a
#else
  real, parameter :: a = 1.0 ! constant scale factor
#endif
#ifdef ALFL_MSCALAR
  real :: alfl_dens
#endif

  integer, parameter :: gather_nprops_pos = 3
  integer, dimension(gather_nprops_pos), save :: gather_propinds_pos = &
    (/ integer :: POSX_PART_PROP, POSY_PART_PROP, POSZ_PART_PROP /)

  integer, parameter :: gather_nprops = 11
  integer, dimension(gather_nprops), save :: gather_propinds = &
    (/ integer :: TAG_PART_PROP, MASS_PART_PROP, &
                  POSX_PART_PROP, POSY_PART_PROP, POSZ_PART_PROP, &
                  VELX_PART_PROP, VELY_PART_PROP, VELZ_PART_PROP, &
                  X_ANG_PART_PROP, Y_ANG_PART_PROP, Z_ANG_PART_PROP /)

#define get_tag(arg1,arg2) ((arg1)*65536 + (arg2))
#define get_pno(arg1) ((arg1)/65536)
#define get_ppe(arg1) ((arg1) - get_pno(arg1)*65536)

  if (first_call) then

    MyPE = dr_globalMe
    MasterPE = MASTER_PE

    call RuntimeParameters_get("sink_creationInfo", creationInfo)
    call RuntimeParameters_get("sink_convergingFlowCheck", convergingFlowCheck)
    call RuntimeParameters_get("sink_potentialMinCheck", potentialMinCheck)
    call RuntimeParameters_get("sink_JeansCheck", JeansCheck)
    call RuntimeParameters_get("sink_negativeEtotCheck", negativeEtotCheck)
    call RuntimeParameters_get("sink_GasAccretionChecks", GasAccretionChecks)

    ! prepare density threshold: rho_Jeans = pi c_s^2 / (4 G r_sink^2),
    ! with the sound speed squared (c_s^2) evaluated cell-by-cell below from c_s^2 = gamma P/rho
    density_thresh_factor = sink_density_thresh_factor * PI / (4.0*newton*accretion_radius**2)

    call Grid_getMinCellSize(delta_at_lrefmax)

    call pt_sinkGatherGlobal()

    local_tag_number = 0
    do lp = 1, localnpf
      if (get_ppe(int(particles_global(iptag,lp))) .EQ. MyPE) then
        local_tag_number = max(local_tag_number, get_pno(int(particles_global(iptag,lp))))
      endif
    enddo

#if defined(MAGX_VAR) && defined(MAGY_VAR) && defined(MAGZ_VAR)
    call RuntimeParameters_get("UnitSystem", units)
    if ( units == "SI" .or. units == "si" ) then
      mu_zero = 4.0*PI*1.e-7
    else if ( units == "CGS" .or. units == "cgs" ) then
      mu_zero = 4.0*PI
    else
      mu_zero = 1.0
    endif
#endif

    ! set the guard cell filling mask (1); this is normally not used,
    ! because Particles_advance will have updated these variables already;
    ! these variables are defined in pt_gcMaskForAdvance, in pt_setMask
    gcmask1(:) = .FALSE.
#ifdef DENS_VAR
    gcmask1(DENS_VAR) = .TRUE.
#endif
#ifdef VELX_VAR
    gcmask1(VELX_VAR) = .TRUE.
    gcmask1(VELY_VAR) = .TRUE.
    gcmask1(VELZ_VAR) = .TRUE.
#endif
#ifdef GPOT_VAR
    gcmask1(GPOT_VAR) = .TRUE.
#endif
    ! set the guard cell filling mask for additional variables used here
    gcmask(:) = .FALSE.
#ifdef PRES_VAR
    gcmask(PRES_VAR) = .TRUE.
#endif
#ifdef GAMC_VAR
    gcmask(GAMC_VAR) = .TRUE.
#endif
#ifdef ALFL_MSCALAR
    gcmask(ALFL_MSCALAR) = .TRUE.
#endif
#ifdef MAGX_VAR
    gcmask(MAGX_VAR) = .TRUE.
    gcmask(MAGY_VAR) = .TRUE.
    gcmask(MAGZ_VAR) = .TRUE.
#endif

    first_call = .false.

  endif

  if (.not. useSinkParticles) return

  call Timers_start("sinkCreateAccrete")

  call Timers_start("sinkCreate")

  if (debug .and. (dr_globalMe .eq. MASTER_PE)) print *, "Particles_sinkCreateAccrete: entering."

  ! blockList,blockCount used to be passed in as args but not anymore
  call Grid_getListOfBlocks(LEAF,blockList,blockCount)

#ifdef COSMOLOGY
  call Cosmology_getRedshift(redshift)
  a = 1.0 / (1.0 + redshift) ! scale factor
#endif

  call Logfile_stamp(localnpf, "[SinkParticles]: localnpf now")

  ! this call to GC fill takes most of the time here. GCs are filled in
  ! Particles_advance though, but only if pt_reduceGcellFills = .false.
  ! Thus, we need to call it here on all variables (including GPOT and MAG?),
  ! if it has not been called with (CENTER, ALLDIR) already in Particles_advance
  ! Fortunately, the default is pt_reduceGcellFills = .false., so we can save time here.
  if (pt_reduceGcellFills) then
    if (.NOT.gcMaskLogged) call Logfile_stampVarMask(gcmask1, .false., '[Particles_sinkCreateAccrete]', 'gcMask1')
    call Grid_fillGuardCells(CENTER, ALLDIR, masksize=NUNK_VARS, mask=gcmask1, makeMaskConsistent=.TRUE., &
                             selectBlockType=LEAF, doLogMask=.NOT.gcMaskLogged)
  endif
  ! we need to call this to fill GCs of additional vars that have not been filled in Particles_advance
  if (.NOT.gcMaskLogged) call Logfile_stampVarMask(gcmask, .false., '[Particles_sinkCreateAccrete]', 'gcMask')
  call Grid_fillGuardCells(CENTER, ALLDIR, masksize=NUNK_VARS, mask=gcmask, makeMaskConsistent=.TRUE., &
                           selectBlockType=LEAF, doLogMask=.NOT.gcMaskLogged)
  gcMaskLogged = .TRUE.

  ! update particle's cpu info
  particles_local(ipcpu, 1:localnp) = MyPE

  call pt_sinkGatherGlobal(gather_propinds_pos, gather_nprops_pos)

  ! stops simulation (creates .dump_restart) when total sink particle mass >= sink_stop_on_total_mass
  if (MyPE .eq. MasterPE) then
    total_sink_mass = 0.0
    do lp = 1, localnpf
      total_sink_mass = total_sink_mass + particles_global(ipm,lp)
    enddo
    if (total_sink_mass .ge. sink_stop_on_total_mass) then
      print *, 'Creating .dump_restart due to total sink mass ', total_sink_mass, &
              & ' >= ', sink_stop_on_total_mass
      open(66, file='.dump_restart', action='write', position='append')
      write(66, *) 'Signalling stop due to total sink mass ', total_sink_mass, &
              & ' >= ', sink_stop_on_total_mass
      close(66)
    endif
  endif

  ! ======== START creation loop ========

  mass = 0.0

  time = dr_simTime

  llb = 0

  ! loop over leaf blocks (note that passed blockList only contains leafs)
  do lb = 1, blockCount

    blockID = blockList(lb)

    call Grid_getBlkPhysicalSize(blockID, size)
    dx_block = size(1)/real(NXB)
    dy_block = size(2)/real(NYB)
    dz_block = size(3)/real(NZB)

    ! check if this block is at the highest level of refinement; else cycle
    if (nint( min(dx_block, dy_block, dz_block) / delta_at_lrefmax ) .ne. 1) cycle

    ! cell volume for cells in this block
    dVol = dx_block*dy_block*dz_block

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

    call Grid_getBlkPtr(blockID, solnData)

    affected_block = .false.

    ! loop over cells (not including guard cells)
    do kp = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
      do jp = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
        do ip = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

#ifdef ALFL_MSCALAR
          ! in case we are using the Alfven limiter
          dens = (1.0-solnData(ALFL_MSCALAR,ip,jp,kp))*solnData(DENS_VAR,ip,jp,kp)
#else
          dens = solnData(DENS_VAR,ip,jp,kp)
#endif
#ifdef PRES_VAR
          pres = solnData(PRES_VAR,ip,jp,kp)
#else
          pres = dens ! isothermal sound-speed-one case
#endif
          ! compute sound speed (squared)
#ifdef GAMC_VAR
          cs2 = pres / dens * solnData(GAMC_VAR,ip,jp,kp)
#else
          cs2 = pres / dens
#endif
          ! get density threshold (note that rho_thresh_proper = rho_thresh_comoving)
          density_thresh = density_thresh_factor * cs2 * a**3

          ! if cell density <= Jeans density threshold (comparison in comoving), skip the following
          if (dens .le. density_thresh) cycle

          affected_block = .true. ! mark block (as it has at least one cell with dens > density_thresh)

          create_sink = .true. ! for now...

          ! Is there an existing particle in range?
          do pno = 1, localnpf ! Looping over global particles list
            distx = xc(ip) - particles_global(ipx,pno)
            disty = yc(jp) - particles_global(ipy,pno)
            distz = zc(kp) - particles_global(ipz,pno)
            if (sink_grav_boundary_type .eq. 2) call pt_sinkCorrectForPeriodicBCs(distx, disty, distz)
            rad = sqrt(distx**2 + disty**2 + distz**2)
            ! Does this position fall within 2x accretion radius of existing sink?
            if (rad .le. 2.0*accretion_radius) create_sink = .false.
          enddo

          if (.not. create_sink) cycle ! skip following

          ! check for converging flow in all surrounding cells
          if (convergingFlowCheck) then
            do k1 = -1, 1
              do j1 = -1, 1
                do i1 = -1, 1
                  rad = sqrt(real(i1*i1+j1*j1+k1*k1))
                  if (rad .gt. 0.) then
                    vrad = (i1*(solnData(VELX_VAR, ip+i1, jp+j1, kp+k1)-solnData(VELX_VAR, ip, jp, kp)) + &
                            j1*(solnData(VELY_VAR, ip+i1, jp+j1, kp+k1)-solnData(VELY_VAR, ip, jp, kp)) + &
                            k1*(solnData(VELZ_VAR, ip+i1, jp+j1, kp+k1)-solnData(VELZ_VAR, ip, jp, kp))) / rad
                    ! if surrounding cell diverges, do not create sink
                    if (vrad .gt. 1.e-5*sqrt(cs2)) create_sink = .false.
                  endif ! rad > 0
                enddo
              enddo
            enddo
            if (creationInfo .gt. 0) then
              if (create_sink) then
                write(*,'(A,3(1X,ES16.9))') '1. convergingFlowCheck successful at position: ', xc(ip), yc(jp), zc(kp)
              else
                if (creationInfo .gt. 1) &
                  & write(*,'(A,3(1X,ES16.9))') '1. convergingFlowCheck failed at position: ', xc(ip), yc(jp), zc(kp)
              endif
            endif
          endif

          if (.not. create_sink) cycle ! skip following

          ! check for potential minimum
          if (potentialMinCheck) then
            ncells_in_vol = 1
            gpot = solnData(GPOT_VAR,ip,jp,kp)
            absgpot = abs(gpot)
            do k1 = -ngc_sink_creation_check, ngc_sink_creation_check
              do j1 = -ngc_sink_creation_check, ngc_sink_creation_check
                  do i1 = -ngc_sink_creation_check, ngc_sink_creation_check
                    if (i1**2 + j1**2 + k1**2 .le. ngc_sink_creation_check_radius_sqr) then
                      if ((solnData(GPOT_VAR,ip+i1,jp+j1,kp+k1)-gpot)/absgpot .lt. -1.e-3) then
                        create_sink = .false.
                      endif
                      ncells_in_vol = ncells_in_vol + 1
                    endif
                  enddo
              enddo
            enddo
            if (creationInfo .gt. 0) then
              if (create_sink) then
                write(*,'(A,3(1X,ES16.9))') '2. potentialMinCheck successful at position: ', xc(ip), yc(jp), zc(kp)
              else
                if (creationInfo .gt. 1) &
                  & write(*,'(A,3(1X,ES16.9))') '2. potentialMinCheck failed at position: ', xc(ip), yc(jp), zc(kp)
              endif
            endif
          endif

          if (.not. create_sink) cycle ! skip following

          ! check for Jeans condition and total energies
          if (JeansCheck .or. negativeEtotCheck) then

            etherm_in_vol = 0.0
            vxcm_in_vol = 0.0
            vycm_in_vol = 0.0
            vzcm_in_vol = 0.0
            mass_in_vol = 0.0
            maxgpot_in_vol = solnData(GPOT_VAR,ip,jp,kp)
            do k1 = -ngc_sink_creation_check, ngc_sink_creation_check
              do j1 = -ngc_sink_creation_check, ngc_sink_creation_check
                do i1 = -ngc_sink_creation_check, ngc_sink_creation_check
                  if (i1**2 + j1**2 + k1**2 .le. ngc_sink_creation_check_radius_sqr) then
                    ii1 = ip+i1
                    jj1 = jp+j1
                    kk1 = kp+k1
#ifdef ALFL_MSCALAR
                    ! in case we are using the Alfven limiter
                    dens_loc = (1.0-solnData(ALFL_MSCALAR,ii1,jj1,kk1)) * solnData(DENS_VAR,ii1,jj1,kk1)
#else
                    dens_loc = solnData(DENS_VAR,ii1,jj1,kk1)
#endif
                    ! avoid use of EINT_VAR, so use only PRES_VAR;
                    ! coefficient for pressure should be 1/(gamma-1), so for gamma = 5/3,
                    ! the coefficient is 3/2 but we take 1, to allow for easier sink formation
#ifdef PRES_VAR
                    etherm_in_vol = etherm_in_vol + 1.0*solnData(PRES_VAR,ii1,jj1,kk1)
#else
                    ! isothermal sound speed = 1 case (probably never used in this context)
                    etherm_in_vol = etherm_in_vol + 1.0*solnData(DENS_VAR,ii1,jj1,kk1)
#endif
                    vxcm_in_vol = vxcm_in_vol + solnData(VELX_VAR,ii1,jj1,kk1)*dens_loc
                    vycm_in_vol = vycm_in_vol + solnData(VELY_VAR,ii1,jj1,kk1)*dens_loc
                    vzcm_in_vol = vzcm_in_vol + solnData(VELZ_VAR,ii1,jj1,kk1)*dens_loc
                    mass_in_vol = mass_in_vol + dens_loc
                    if (solnData(GPOT_VAR,ii1,jj1,kk1) .gt. maxgpot_in_vol) then
                      maxgpot_in_vol = solnData(GPOT_VAR,ii1,jj1,kk1)
                    endif
                  endif ! in local Jeans volume
                enddo
              enddo
            enddo

            etherm_in_vol = etherm_in_vol*dVol
            vxcm_in_vol = vxcm_in_vol/mass_in_vol
            vycm_in_vol = vycm_in_vol/mass_in_vol
            vzcm_in_vol = vzcm_in_vol/mass_in_vol
            mass_in_vol = mass_in_vol*dVol

            ekindisp_in_vol = 0.0
            egravdeltapot_in_vol = 0.0
            emag_in_vol = 0.0

            do k1 = -ngc_sink_creation_check, ngc_sink_creation_check
              do j1 = -ngc_sink_creation_check, ngc_sink_creation_check
                do i1 = -ngc_sink_creation_check, ngc_sink_creation_check
                  if (i1**2 + j1**2 + k1**2 .le. ngc_sink_creation_check_radius_sqr) then
                    ii1 = ip+i1
                    jj1 = jp+j1
                    kk1 = kp+k1
#ifdef ALFL_MSCALAR
                    ! in case we are using the Alfven limiter
                    dens_loc = (1.0-solnData(ALFL_MSCALAR,ii1,jj1,kk1)) * solnData(DENS_VAR,ii1,jj1,kk1)
#else
                    dens_loc = solnData(DENS_VAR,ii1,jj1,kk1)
#endif
                    ekindisp_in_vol = ekindisp_in_vol + dens_loc * &
                        ( (solnData(VELX_VAR, ii1, jj1, kk1) - vxcm_in_vol)**2 + &
                          (solnData(VELY_VAR, ii1, jj1, kk1) - vycm_in_vol)**2 + &
                          (solnData(VELZ_VAR, ii1, jj1, kk1) - vzcm_in_vol)**2  )
                    egravdeltapot_in_vol = egravdeltapot_in_vol + &
                        (solnData(GPOT_VAR, ii1, jj1, kk1) - maxgpot_in_vol)*dens_loc
#if defined(MAGX_VAR) && defined(MAGY_VAR) && defined(MAGZ_VAR)
                    emag_in_vol = emag_in_vol + solnData(MAGX_VAR, ii1, jj1, kk1)**2 + &
                                                solnData(MAGY_VAR, ii1, jj1, kk1)**2 + &
                                                solnData(MAGZ_VAR, ii1, jj1, kk1)**2
#endif
                  endif ! in local Jeans volume
                enddo
              enddo
            enddo

            ekindisp_in_vol = 0.5*ekindisp_in_vol*dVol
            egravdeltapot_in_vol = -egravdeltapot_in_vol*dVol
#if defined(MAGX_VAR) && defined(MAGY_VAR) && defined(MAGZ_VAR)
            emag_in_vol = 0.5/mu_zero*emag_in_vol*dVol
#endif
            ! Jeans mass virial argument (see e.g., Bate Bonnell Price 1995)
            if (JeansCheck) then
              if (2.0*etherm_in_vol + emag_in_vol .gt. egravdeltapot_in_vol) create_sink = .false.
              if (creationInfo .gt. 0) then
                if (create_sink) then
                  write(*,'(A,3(1X,ES16.9))') '3. JeansCheck successful at position: ', xc(ip), yc(jp), zc(kp)
                else
                  if (creationInfo .gt. 1) then
                    write(*,'(A,3(1X,ES16.9))') '3. JeansCheck failed at position: ', xc(ip), yc(jp), zc(kp)
                    write(*,'(A,4(1X,ES16.9))') 'Epot, Eth, Emag, 2*Eth+Emag: ', &
                      & egravdeltapot_in_vol, etherm_in_vol, emag_in_vol, 2.0*etherm_in_vol+emag_in_vol
                  endif
                endif
              endif
            endif

            if (.not. create_sink) cycle ! skip following

            ! total energy should be negative (see e.g., Bate Bonnell Price 1995)
            if (negativeEtotCheck) then
              if (etherm_in_vol + ekindisp_in_vol + emag_in_vol .gt. egravdeltapot_in_vol) create_sink = .false.
              if (creationInfo .gt. 0) then
                if (create_sink) then
                  write(*,'(A,3(1X,ES16.9))') '4. negativeEtotCheck successful at position: ', xc(ip), yc(jp), zc(kp)
                else
                  if (creationInfo .gt. 1) then
                    write(*,'(A,3(1X,ES16.9))') '4. negativeEtotCheck failed at position: ', xc(ip), yc(jp), zc(kp)
                    write(*,'(A,5(1X,ES16.9))') 'Epot, Eth, Ekin, Emag, Eth+Ekin+Emag: ', egravdeltapot_in_vol, &
                      & etherm_in_vol, ekindisp_in_vol, emag_in_vol, etherm_in_vol+ekindisp_in_vol+emag_in_vol
                  endif
                endif
              endif
            endif

          endif ! check for Jeans condition and total energies

          if (.not. create_sink) cycle ! skip following

          ! if here, finally create the sink in the cell centre
          x = xc(ip)
          y = yc(jp)
          z = zc(kp)
          pt = time
          pno = pt_sinkCreateParticle(x, y, z, pt, blockID, MyPE)
          write(*,'(A,4(1X,ES16.9),3I8)') "sink particle created (x, y, z, pt, blockID, MyPE, tag): ", &
              & x, y, z, pt, blockID, MyPE, int(particles_local(iptag,pno))
          ! this allows us to stop the simulation when the next sink particle forms
          if (sink_stop_on_creation) then
            print *, 'Creating .dump_restart due to sink particle creation with tag = ', &
                    & int(particles_local(iptag,pno))
            open(66, file='.dump_restart', action='write', position='append')
            write(66, *) 'Signalling stop on sink particle creation with tag = ', &
                        & int(particles_local(iptag,pno))
            close(66)
          endif

        enddo ! ip
      enddo ! jp
    enddo ! kp

    ! mark this block, so we only loop relevant blocks below
    if (affected_block) then
      llb = llb + 1
      block_list(llb) = blockID
    endif

    call Grid_releaseBlkPtr(blockID, solnData)

    deallocate(xc)
    deallocate(yc)
    deallocate(zc)

  enddo ! block loop

  ! ======== END creation loop ========

  ! call this here instead of in pt_sinkMergingAfterCreation(), in preparation for pt_sinkFindList()
  ! and accumulation below, involving particles_global;
  ! NOTE THAT ALL particle properties must be communicated here.
  call pt_sinkGatherGlobal()

  ! Merges sink particles that were created close to one another
  call pt_sinkMergingAfterCreation(delta_at_lrefmax)

  call Timers_stop("sinkCreate")

  call Timers_start("sinkAccrete")

  ! ======== START accretion loop ========

  llnblocks = llb

  old_localnp = localnp

  ! clear mass & velocity
  tot_mass(:)   = 0.0
  cm_x(:)       = 0.0
  cm_y(:)       = 0.0
  cm_z(:)       = 0.0
  vel_cm_x(:)   = 0.0
  vel_cm_y(:)   = 0.0
  vel_cm_z(:)   = 0.0
  ang_x(:)      = 0.0
  ang_y(:)      = 0.0
  ang_z(:)      = 0.0

  ! do it again, but only loop over affected blocks and add mass to particles

  do llb = 1, llnblocks

    lb = block_list(llb)

    call Grid_getBlkPhysicalSize(lb, size)
    dx_block = size(1)/real(NXB)
    dy_block = size(2)/real(NYB)
    dz_block = size(3)/real(NZB)
    dVol = dx_block*dy_block*dz_block

    call Grid_getBlkIndexLimits(lb, blkLimits, blkLimitsGC)

    size_x = blkLimitsGC(HIGH,IAXIS)-blkLimitsGC(LOW,IAXIS)+1
    size_y = blkLimitsGC(HIGH,JAXIS)-blkLimitsGC(LOW,JAXIS)+1
    size_z = blkLimitsGC(HIGH,KAXIS)-blkLimitsGC(LOW,KAXIS)+1

    allocate(xc(size_x))
    allocate(yc(size_y))
    allocate(zc(size_z))

    call Grid_getCellCoords(IAXIS, lb, CENTER, .true., xc, size_x)
    call Grid_getCellCoords(JAXIS, lb, CENTER, .true., yc, size_y)
    call Grid_getCellCoords(KAXIS, lb, CENTER, .true., zc, size_z)

    call Grid_getBlkPtr(lb, solnData)

    affected_block = .false.

    do kp = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
      do jp = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
        do ip = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

#ifdef ALFL_MSCALAR
          ! in case we are using the Alfven limiter
          dens = (1.0-solnData(ALFL_MSCALAR,ip,jp,kp))*solnData(DENS_VAR,ip,jp,kp)
#else
          dens = solnData(DENS_VAR,ip,jp,kp)
#endif
#ifdef PRES_VAR
          pres = solnData(PRES_VAR,ip,jp,kp)
#else
          pres = dens ! isothermal sound-speed-one case
#endif
          ! compute sound speed (squared)
#ifdef GAMC_VAR
          cs2 = pres / dens * solnData(GAMC_VAR,ip,jp,kp)
#else
          cs2 = pres / dens
#endif
          ! get density threshold (note that rho_thresh_proper = rho_thresh_comoving)
          density_thresh = density_thresh_factor * cs2 * a**3

          ! if cell density <= Jeans density threshold (comparison in comoving), skip the following
          if (dens .le. density_thresh) cycle

          mass = (dens - density_thresh) * dVol

          ! return a list 'pindex_found' containing all particles found within r_accr
          ! all affected particles are in local list 'particles';
          ! if necessary extend the list with a dummy particle (create_part = .true. switch)
          ! if create_part = .true., then particles_local also contains the global particles
          ! that fall within the accretion_radius

          call pt_sinkFindList(xc(ip), yc(jp), zc(kp), accretion_radius, .true., pindex_found, np_found)

          if (np_found .gt. 0) then
            ! there is a particle within accretion_radius of this cell

            do npf = 1, np_found
              ! loop over all particles within accretion_radius of this cell

              pno = pindex_found(npf)

              dvx = solnData(VELX_VAR,ip,jp,kp) - particles_local(ipvx, pno)
              dvy = solnData(VELY_VAR,ip,jp,kp) - particles_local(ipvy, pno)
              dvz = solnData(VELZ_VAR,ip,jp,kp) - particles_local(ipvz, pno)

              distx = xc(ip) - particles_local(ipx, pno)
              disty = yc(jp) - particles_local(ipy, pno)
              distz = zc(kp) - particles_local(ipz, pno)

              if (sink_grav_boundary_type .eq. 2) call pt_sinkCorrectForPeriodicBCs(distx, disty, distz)

              radius(npf) = sqrt(distx**2 + disty**2 + distz**2)

              egrav_Gas = -newton*2.0 * PI / 3.0 * density_thresh * &
                & (accretion_radius**2 - radius(npf)**2)*mass

              if (radius(npf) .gt. 0) then
                egrav_part = -newton*particles_local(ipm, pno)*mass * &
                              (1.0/radius(npf)-1./accretion_radius)
              else
                egrav_part = -1.0e99
              endif

              egrav = egrav_gas + egrav_part
              ! CTSS - factor of (1+z)^3 needed for correct comoving potential
              egrav = egrav / a**3
              ekin = 0.5 * mass * ( dvx**2 + dvy**2 + dvz**2 )

              etot(npf) = egrav + ekin

              ! calculate the radial velocity wrt each particle found
              if (radius(npf) .gt. 0.) then
                vr(npf) = ( distx*dvx + disty*dvy + distz*dvz ) / radius(npf)
              else
                vr(npf) = 0.0
              endif

            enddo ! npf

            pno_to_accrete = 0

            inner_r_accr = max( 0.2*accretion_radius, (dVol**(1.0/3.0)) )
            within_inner_r_accr = .false.
            etot_min_inner_r_accr = 1e99
            etot_min = 1e99

            do npf = 1, np_found
              if (radius(npf) .lt. inner_r_accr) then
                if (etot(npf) .lt. etot_min_inner_r_accr) then
                  pno_to_accrete = pindex_found(npf)
                  pno = npf
                  etot_min_inner_r_accr = etot(npf)
                  within_inner_r_accr = .true.
                endif
              else
                if (GasAccretionChecks) then
                  if (.not. within_inner_r_accr .and. vr(npf) .lt. 1.0e-5*sqrt(cs2) &
                      .and. etot(npf) .lt. 0. .and. etot(npf) .lt. etot_min) then
                    pno_to_accrete = pindex_found(npf)
                    pno = npf
                    etot_min = etot(npf)
                  endif
                else
                  if (.not. within_inner_r_accr .and. (etot(npf) .lt. etot_min)) then
                    pno_to_accrete = pindex_found(npf)
                    pno = npf
                    etot_min = etot(npf)
                  endif
                endif ! perform gas accretion checks?
              endif ! inner accretion?
            enddo ! potential sinks

            if (pno_to_accrete .gt. 0) then

              dvx = solnData(VELX_VAR,ip,jp,kp) - particles_local(ipvx, pno_to_accrete)
              dvy = solnData(VELY_VAR,ip,jp,kp) - particles_local(ipvy, pno_to_accrete)
              dvz = solnData(VELZ_VAR,ip,jp,kp) - particles_local(ipvz, pno_to_accrete)

              distx = xc(ip) - particles_local(ipx, pno_to_accrete)
              disty = yc(jp) - particles_local(ipy, pno_to_accrete)
              distz = zc(kp) - particles_local(ipz, pno_to_accrete)

              if (sink_grav_boundary_type .eq. 2) call pt_sinkCorrectForPeriodicBCs(distx, disty, distz)
#ifdef ALFL_MSCALAR
              ! in case we are using the Alfven limiter
              alfl_dens = solnData(ALFL_MSCALAR,ip,jp,kp) * solnData(DENS_VAR,ip,jp,kp)
              solnData(DENS_VAR,ip,jp,kp) = density_thresh + alfl_dens
              solnData(ALFL_MSCALAR,ip,jp,kp) = alfl_dens / solnData(DENS_VAR,ip,jp,kp)
#else
              solnData(DENS_VAR,ip,jp,kp) = density_thresh
#endif
              affected_block = .true.

              tot_mass(pno_to_accrete) = tot_mass(pno_to_accrete) + mass
              cm_x(pno_to_accrete)     = cm_x(pno_to_accrete) + distx*mass
              cm_y(pno_to_accrete)     = cm_y(pno_to_accrete) + disty*mass
              cm_z(pno_to_accrete)     = cm_z(pno_to_accrete) + distz*mass
              vel_cm_x(pno_to_accrete) = vel_cm_x(pno_to_accrete) + dvx*mass
              vel_cm_y(pno_to_accrete) = vel_cm_y(pno_to_accrete) + dvy*mass
              vel_cm_z(pno_to_accrete) = vel_cm_z(pno_to_accrete) + dvz*mass
              ang_x(pno_to_accrete)    = ang_x(pno_to_accrete) + &
                    & (disty*solnData(VELZ_VAR,ip,jp,kp)-distz*solnData(VELY_VAR,ip,jp,kp))*mass
              ang_y(pno_to_accrete)    = ang_y(pno_to_accrete) + &
                    & (distz*solnData(VELX_VAR,ip,jp,kp)-distx*solnData(VELZ_VAR,ip,jp,kp))*mass
              ang_z(pno_to_accrete)    = ang_z(pno_to_accrete) + &
                    & (distx*solnData(VELY_VAR,ip,jp,kp)-disty*solnData(VELX_VAR,ip,jp,kp))*mass

            endif ! pno_to_accrete

          endif ! np_found

        enddo ! ip
      enddo ! jp
    enddo ! kp

    if (affected_block) then
      call Eos_wrapped(MODE_DENS_EI, blkLimits, lb)
    end if

    call Grid_releaseBlkPtr(lb, solnData)

    deallocate(xc)
    deallocate(yc)
    deallocate(zc)

   enddo ! blocks

   ! ======== END accretion loop ========

   ! Copy grid info to dummy particle for data exchange
   do lp = old_localnp+1, localnp
      particles_local(ipm,lp) = tot_mass(lp)
      particles_local(ipx,lp) = cm_x(lp)
      particles_local(ipy,lp) = cm_y(lp)
      particles_local(ipz,lp) = cm_z(lp)
      particles_local(ipvx,lp) = vel_cm_x(lp)
      particles_local(ipvy,lp) = vel_cm_y(lp)
      particles_local(ipvz,lp) = vel_cm_z(lp)
      particles_local(iplx,lp) = ang_x(lp)
      particles_local(iply,lp) = ang_y(lp)
      particles_local(iplz,lp) = ang_z(lp)
   enddo

   ! Exchange information across CPUs
   call pt_sinkGatherGlobal(gather_propinds, gather_nprops)

   npart = localnp

   ! delete dummy (non-local) particles from list
   ! do this because pt_sinkFindList with .true. call raises localnp
   do lp = 1, localnp
    if (int(particles_local(ipcpu,lp)) .ne. MyPE) then
      npart = npart-1
    endif
   enddo
   localnp = npart

   do lp = 1, localnp

    ! check if local particle is affected by regions on other CPUs
    do nlp = localnp+1, localnpf
      if (int(particles_local(iptag,lp)) .eq. int(particles_global(iptag,nlp))) then
        tot_mass(lp) = tot_mass(lp) + particles_global(ipm,nlp)
        cm_x(lp) = cm_x(lp) + particles_global(ipx,nlp)
        cm_y(lp) = cm_y(lp) + particles_global(ipy,nlp)
        cm_z(lp) = cm_z(lp) + particles_global(ipz,nlp)
        vel_cm_x(lp) = vel_cm_x(lp) + particles_global(ipvx,nlp)
        vel_cm_y(lp) = vel_cm_y(lp) + particles_global(ipvy,nlp)
        vel_cm_z(lp) = vel_cm_z(lp) + particles_global(ipvz,nlp)
        ang_x(lp) = ang_x(lp) + particles_global(iplx,nlp)
        ang_y(lp) = ang_y(lp) + particles_global(iply,nlp)
        ang_z(lp) = ang_z(lp) + particles_global(iplz,nlp)
      endif
    enddo

    ! update particle properties (conservation laws)

    particles_local(iold_pmass,lp) = particles_local(ipm,lp)

    if (tot_mass(lp) .ne. 0.0) then

      ! mass update
      particles_local(ipm,lp) = particles_local(ipm,lp) + tot_mass(lp)

      ! position update
      px_old = particles_local(ipx,lp)
      py_old = particles_local(ipy,lp)
      pz_old = particles_local(ipz,lp)
      particles_local(ipx,lp) = particles_local(ipx,lp) + cm_x(lp)/particles_local(ipm,lp)
      particles_local(ipy,lp) = particles_local(ipy,lp) + cm_y(lp)/particles_local(ipm,lp)
      particles_local(ipz,lp) = particles_local(ipz,lp) + cm_z(lp)/particles_local(ipm,lp)

      ! velocity update
      particles_local(ipvx,lp) = particles_local(ipvx,lp) + vel_cm_x(lp)/particles_local(ipm,lp)
      particles_local(ipvy,lp) = particles_local(ipvy,lp) + vel_cm_y(lp)/particles_local(ipm,lp)
      particles_local(ipvz,lp) = particles_local(ipvz,lp) + vel_cm_z(lp)/particles_local(ipm,lp)

      ! spin update
      particles_local(iplx_old,lp) = particles_local(iplx,lp)
      particles_local(iply_old,lp) = particles_local(iply,lp)
      particles_local(iplz_old,lp) = particles_local(iplz,lp)
      particles_local(iplx,lp) = particles_local(iplx,lp) + ang_x(lp) - particles_local(ipm,lp) * &
                                ( (particles_local(ipy,lp)-py_old)*particles_local(ipvz,lp) - &
                                  (particles_local(ipz,lp)-pz_old)*particles_local(ipvy,lp)  )
      particles_local(iply,lp) = particles_local(iply,lp) + ang_y(lp) - particles_local(ipm,lp) * &
                                ( (particles_local(ipz,lp)-pz_old)*particles_local(ipvx,lp) - &
                                  (particles_local(ipx,lp)-px_old)*particles_local(ipvz,lp)  )
      particles_local(iplz,lp) = particles_local(iplz,lp) + ang_z(lp) - particles_local(ipm,lp) * &
                                ( (particles_local(ipx,lp)-px_old)*particles_local(ipvy,lp) - &
                                  (particles_local(ipy,lp)-py_old)*particles_local(ipvx,lp)  )
    endif ! mass was accreted

    particles_local(ipmdot,lp) = (particles_local(ipm,lp) - particles_local(iold_pmass,lp)) / dt

   enddo ! lp

   lp = 1
   do while (lp .le. localnp)
    if (particles_local(ipm,lp) .le. 0.0) then
      write(*,'(A,I8,3(1X,ES16.9))') "SinkParticles: deleted particle due to zero mass with tag, pos = ", &
        int(particles_local(iptag,lp)), particles_local(ipx,lp), particles_local(ipy,lp), particles_local(ipz,lp)
      particles_local(:,lp) = particles_local(:,localnp)
      particles_local(ipblk,localnp) = NONEXISTENT
      n_empty = n_empty + 1
      localnp = localnp - 1
      lp = lp - 1
    endif
    lp = lp + 1
   enddo

   call Timers_stop("sinkAccrete")

   call Timers_start("sinkMerging")
   if (sink_merging) call pt_sinkParticleMerging(dt)
   call Timers_stop("sinkMerging")

   ! write sink particle data to sinks_evol.dat
   call Timers_start("sinkDump")
   call pt_sinkDumpParticles(time, "sinks_evol.dat")
   call Timers_stop("sinkDump")

   ! This is needed to signal the particle unit that particle
   ! positions have been initialized, which is important for
   ! Particles_updateRefinement() to actually move particle to
   ! to the right blocks and processors.
   if (localnpf .gt. 0) then
    pt_posInitialized = .true.
    pt_velInitialized = .true.
   endif

   if (debug .and. (dr_globalMe .eq. MASTER_PE)) then
    print*, "Particles_sinkCreateAccrete: exiting, localnpf = ", localnpf
   endif

   call Timers_stop("sinkCreateAccrete")

   return

end subroutine Particles_sinkCreateAccrete

