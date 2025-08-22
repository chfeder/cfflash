/*

  projection.cpp

  Computes projections (e.g., column density) or slices
  of grid variables from FLASH (Uniform Grid or AMR) output files

  By Christoph Federrath, 2013-2025

*/

#include "mpi.h" /// MPI lib
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include "../Libs/HDFIO.h" // HDF5 IO
#include "../Libs/FlashGG.h" /// Flash General Grid class
#include "../Libs/FlashParticles.h" // for sink particle IO
#include "../Libs/GRID3D.h" // grid for the image
#include "../Libs/CFTools.h" // for Gaussian beam smoothing and timing

// constants
#define NDIM 3
using namespace std;
enum {X, Y, Z};
static const double pi = 3.14159265358979323846;
static const char dens_name[] = "dens";
int Verbose = 1;

// MPI stuff
int MyPE = 0, NPE = 1;

// some global stuff (inputs)
string inputfile = "";
string datasetname = "dens";
vector<int> pixel(2,512);
bool user_pixels_set = false;
bool user_proj_range = false;
vector<double> projection_range(6,0.0);
bool user_data_range = false;
double zoom_factor = 1.0;
vector<double> data_range(2,0.0);
string view = "xy";
bool slice = false;
bool density_weight = false;
bool use_output_file = false;
string output_file_name = "";
double opacity = 0.0;
bool opacity_set = false;
// spherical weighting
bool sp_weighting = false;
double sp_r_max = 0.49;
double sp_r_chr = 0.50;
/// perspective
bool perspective = false;
double viewangle = 5.;
// for moment 0, 1, 2 maps
bool moment_maps = false;
// for Gaussian beam smoothing
double gauss_smooth_fwhm = 0;
// if user wants to specify boundary conditions
string boundary_condition = "";

// for rotation
vector<double> rotation_center(3);
bool rotation_center_set = false;
vector<double> rotation_axis(3);
bool rotation_axis_set = false;
double rotation_angle = 0.;
bool rotation_angle_set = false;
int  fullrotation = 0;
bool fullrotation_set = false;

bool process_particles = true;
bool make_starfield = false;

// for cell splitting
bool split_cells = false;

// forward functions
vector<double> Rotation(const vector<double> &inp, const double phi, const vector<double> &rotaxis, const vector<double> &rc);
double PerspectiveScale(const vector<double> &inp, const double focallength, const vector<double> &projection_center, const vector<int> &view_coord);
vector<double> Perspective(const vector<double> &inp, const double focallength, const vector<double> &projection_center, const vector<int> &view_coord);
double SphericalWeight(const vector<double> &inp, const vector<double> &rotation_center, const vector<int> &view_coord);
int ParseInputs(const vector<string> Argument);
void HelpMe(void);


/// --------
///   MAIN
/// --------
int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NPE);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
    if (Verbose && MyPE==0) cout<<" === projection === using MPI num procs: "<<NPE<<endl;

    /// Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) != 0)
    {
        if (Verbose && MyPE==0) cout << endl << "Error in ParseInputs(). Exiting." << endl;
        HelpMe();
        MPI_Finalize(); return 0;
    }

    long starttime = time(NULL);

    /// FLASH file meta data
    FlashGG gg = FlashGG(inputfile, Verbose);
    int NumBlocks = gg.GetNumBlocks();
    vector<int> NB = gg.GetNumCellsInBlock();
    vector< vector<double> > MinMaxDomain = gg.GetMinMaxDomain();
    vector<double> L = gg.GetL();
    vector<vector<double> > D = gg.GetDblock();
    vector<double> Dmin = gg.GetDmin();
    vector<double> Dmax = gg.GetDmax();

    if (Verbose && MyPE==0) gg.PrintInfo();

    // initialise tools
    CFTools cft = CFTools();

    /// prepare particles
    FlashParticles Particles;
    long np = 0;
    map<string, vector<double> > sinks;
    if (process_particles) {
        Particles = FlashParticles(inputfile);
        np = Particles.GetN();
        if (MyPE==0) Particles.PrintInfo();
        if (make_starfield) {
            const char *pnchar[] = {"posx", "posy", "posz", "mass", "luminosity"};
            vector<string> prop_names(pnchar, pnchar + sizeof(pnchar)/sizeof(pnchar[0])); // horrible way of convenience
            sinks = Particles.ReadSinkData(prop_names); // get sink particle data
        }
    }

    // set view_coord
    vector<int> view_coord(3);
    if (view == "xy") { view_coord[0] = 0; view_coord[1] = 1; view_coord[2] = 2;}
    if (view == "xz") { view_coord[0] = 0; view_coord[1] = 2; view_coord[2] = 1;}
    if (view == "yz") { view_coord[0] = 1; view_coord[1] = 2; view_coord[2] = 0;}

    /// set the projection ranges
    if (!user_proj_range) {
        projection_range[2*view_coord[0]+0] = MinMaxDomain[view_coord[0]][0];
        projection_range[2*view_coord[0]+1] = MinMaxDomain[view_coord[0]][1];
        projection_range[2*view_coord[1]+0] = MinMaxDomain[view_coord[1]][0];
        projection_range[2*view_coord[1]+1] = MinMaxDomain[view_coord[1]][1];
        projection_range[2*view_coord[2]+0] = MinMaxDomain[view_coord[2]][0];
        projection_range[2*view_coord[2]+1] = MinMaxDomain[view_coord[2]][1];
        if (slice) {
            /// slice with thickness of min_cell_width through domain center
            projection_range[2*view_coord[2]+0] = MinMaxDomain[view_coord[2]][0]+L[view_coord[2]]/2.-Dmin[view_coord[2]]/2.;
            projection_range[2*view_coord[2]+1] = projection_range[2*view_coord[2]+0]+Dmin[view_coord[2]];
        } // slice
    } // !user_proj_range
    else {
        if (slice) {
            if (projection_range[2*view_coord[2]+0] == projection_range[2*view_coord[2]+1]) {
                projection_range[2*view_coord[2]+0] = projection_range[2*view_coord[2]+0] - Dmin[view_coord[2]]/2.;
                projection_range[2*view_coord[2]+1] = projection_range[2*view_coord[2]+1] + Dmin[view_coord[2]]/2.;
            }
        }
    } // user_proj_range

    // apply zoom
    if (zoom_factor != 1.0) {
        for (int dir = 0; dir < 3; dir++) {
            double mid       = 0.5 * (projection_range[2*dir+0] + projection_range[2*dir+1]);
            double Lnew_half = 0.5 * (projection_range[2*dir+1] - projection_range[2*dir+0]) / zoom_factor;
            projection_range[2*dir+0] = mid - Lnew_half;
            projection_range[2*dir+1] = mid + Lnew_half;
        }
    }

    if (!user_pixels_set) {
        // scale y pixels by pixels in x times (projection size in y) /(projection size in x)
        pixel[1] = pixel[0] * (projection_range[view_coord[1]*2+1]-projection_range[view_coord[1]*2+0]) /
                              (projection_range[view_coord[0]*2+1]-projection_range[view_coord[0]*2+0]);
    }
    if (Verbose && MyPE==0) cout << " Output grid resolution: " << pixel[0] << " x " << pixel[1] << endl;

    /// set the data range (if not provided by the user)
    if (!user_data_range) {
    data_range[0] = -1e99; // set to huge range (so all is included)
    data_range[1] = +1e99; // set to huge range (so all is included)
    } else {
        if (Verbose && MyPE==0) cout<<" Data range restricted to ["<<data_range[0]<<","<<data_range[1]<<"]"<<endl;
    }

    /// setting up opacity
    if (opacity_set && Verbose && MyPE==0) cout<<" Using opacity weighting: opacity="<<opacity<<endl;
    double proj_zmin = projection_range[2*view_coord[2]+0];
    double proj_zmax = projection_range[2*view_coord[2]+1];
    double proj_depth = proj_zmax - proj_zmin;

    /// create 2D grids (a 3D grid with Nx x Ny pixels and 1 pixel in the 3rd dimension)
    int ng = 1; // number of 2D grids
    if (density_weight) ng = 2;
    if (moment_maps) ng = 3;
    if (make_starfield) ng = 1 + Particles.GetTypeCount()["sink"]; // main output grid + every sink gets a local grid (optical depth)
    vector<GRID3D> grid2d(ng);
    for (int g = 0; g < ng; g++) {
        int npx = pixel[0], npy = pixel[1];
        if (make_starfield && g > 0) {npx = 19; npy = 19;} // small grids for each sink (should be odd number)
        grid2d[g] = GRID3D(npx, npy, 1);
    }
    /// set grid bounds
    for (int g = 0; g < ng; g++) grid2d[g].set_bnds(projection_range[2*view_coord[0]], projection_range[2*view_coord[0]+1],
                                                    projection_range[2*view_coord[1]], projection_range[2*view_coord[1]+1],
                                                    projection_range[2*view_coord[2]], projection_range[2*view_coord[2]+1]);
    if (Verbose && MyPE==0) grid2d[0].print_bnds();

    // convert projection_range to more intuitive bounding box container
    vector<vector<double> > bounds(3);
    for (int dim = 0; dim < 3; dim++) bounds[dim].resize(2);
    bounds[view_coord[0]][0] = projection_range[2*view_coord[0]+0];
    bounds[view_coord[0]][1] = projection_range[2*view_coord[0]+1];
    bounds[view_coord[1]][0] = projection_range[2*view_coord[1]+0];
    bounds[view_coord[1]][1] = projection_range[2*view_coord[1]+1];
    bounds[view_coord[2]][0] = projection_range[2*view_coord[2]+0];
    bounds[view_coord[2]][1] = projection_range[2*view_coord[2]+1];

    /// prepare roation
    if (!rotation_center_set) // set default rotation center to domain center
    {
        rotation_center[X] = MinMaxDomain[X][0] + L[X]/2.;
        rotation_center[Y] = MinMaxDomain[Y][0] + L[Y]/2.;
        rotation_center[Z] = MinMaxDomain[Z][0] + L[Z]/2.;
    }
    if (Verbose && MyPE==0) cout<<" Rotation center: "<<rotation_center[X]<<" "<<rotation_center[Y]<<" "<<rotation_center[Z]<<endl;
    // set the rotation center for the rotation of vector quantities, which is always (0,0,0)
    vector<double> rotation_center_for_vector_quantity(3);
    rotation_center_for_vector_quantity[X] = 0.0;
    rotation_center_for_vector_quantity[Y] = 0.0;
    rotation_center_for_vector_quantity[Z] = 0.0;
    // set default rotation around z-axis
    if (!rotation_axis_set) {
        rotation_axis[X] = 0.; rotation_axis[Y] = 0.; rotation_axis[Z] = 1.; }
    else { // make a unity vector out of it
        const double abs_vect = sqrt( rotation_axis[X]*rotation_axis[X] +
                                      rotation_axis[Y]*rotation_axis[Y] +
                                      rotation_axis[Z]*rotation_axis[Z]  );
        rotation_axis[X] /= abs_vect;
        rotation_axis[Y] /= abs_vect;
        rotation_axis[Z] /= abs_vect;
    }
    if (Verbose && MyPE==0) cout<<" Rotation axis: "<<rotation_axis[X]<<" "<<rotation_axis[Y]<<" "<<rotation_axis[Z]<<endl;

    if (density_weight && Verbose && MyPE==0) {
        cout<<" Using mass weighting!"<<endl;
        if (sp_weighting || perspective || opacity_set)
            cout<<" CAUTION: -mw might lead to unexpected results when used with spherical weighting or perspective view or opacity!"<<endl;
    }

    if (make_starfield && Verbose && MyPE==0) {
        if (datasetname!="dens" || slice || density_weight || !process_particles)
            cout<<" CAUTION: -make_starfield needs -dset dens, and should not be used togehter with -slice, -mw, or -no_particles!"<<endl;
    }

    if (moment_maps && Verbose && MyPE==0) cout<<" Generating 0th, 1st, and 2nd moment maps!"<<endl;

    if (sp_weighting && Verbose && MyPE==0) cout<<" Using spherical weighting: sp_r_max="<<sp_r_max<<" sp_r_chr="<<sp_r_chr<<endl;

    // setting up perspective
    double focallength = 0.5*(projection_range[2*view_coord[0]+1]-projection_range[2*view_coord[0]+0])/tan(viewangle/180.*pi);
    vector<double> projection_center(3);
    projection_center[view_coord[0]] = (projection_range[2*view_coord[0]+1]+projection_range[2*view_coord[0]])/2.;
    projection_center[view_coord[1]] = (projection_range[2*view_coord[1]+1]+projection_range[2*view_coord[1]])/2.;
    projection_center[view_coord[2]] = (projection_range[2*view_coord[2]+1]+projection_range[2*view_coord[2]])/2.;
    if (perspective && Verbose && MyPE==0) cout<< " Using perspective view with angle: "<<viewangle<<" degrees; focal length is "<<focallength<<endl;

    /// make a loop over all fullrotation images (if not set, this loop is only executed once)
    for (int irot = 0; irot <= fullrotation; irot++)
    {
        /// clear fields
        for (int g = 0; g < ng; g++) grid2d[g].clear();

        /// set rotation angle
        double angle = rotation_angle*pi/180.;
        if (fullrotation > 0)
            angle += (double)(irot)/(double)(fullrotation)*2.*pi;

        /// check if we have periodic boundary conditions
        bool periodic_boundary_conditions = false; // if user specified 'isolated'
        if (boundary_condition == "periodic") { // user specified 'periodic'
            periodic_boundary_conditions = true;
        }
        if (boundary_condition == "") { // user did not specify BCs, so we auto-detect them from the input file
            string bc = gg.GetBoundaryConditions();
            if (bc == "periodic") periodic_boundary_conditions = true;
        }
        if (Verbose && MyPE==0) cout<<" periodic_boundary_conditions = "<<periodic_boundary_conditions<<endl;

        // if PBCs, create block replicas (extend BoundingBox and NumBlocksRep)
        if (periodic_boundary_conditions) {
            if (Verbose>1 && MyPE==0) cout<<" calling AddBlockReplicasPBCs()"<<endl;
            gg.AddBlockReplicasPBCs();
        }

        // Get list of blocks that overlap with requested projection bounds (affected blocks)
        vector<int> AffectedBlocks;

        /// get block bounding box estimates after rotation and perspective of each block,
        /// so we can find the affected blocks for this projection; this is solely for optimising load balancing and speed
        vector<vector<vector<double> > > BB_for_getting_affected_blocks = gg.GetBoundingBox(); // begin with normal BB
        if (rotation_angle_set || fullrotation_set || perspective || make_starfield) {
            if (Verbose>1 && MyPE==0) cout<<" looping blocks to assign BB_for_getting_affected_blocks"<<endl;
            for (int b=0; b<gg.GetNumBlocksRep(); b++) { // loop over all blocks (may already have replicas, as per call to AddBlockReplicasPBCs)
                double LBmax = 0.0; // get max block length (in any direction) of this block
                for (unsigned int dir = X; dir <= Z; dir++) {
                    double LB = BB_for_getting_affected_blocks[b][dir][1] - BB_for_getting_affected_blocks[b][dir][0]; // block length
                    if (LB > LBmax) LBmax = LB;
                }
                /// add rotation and/or add perspective for block center coordinate
                vector<double> pos = gg.BlockCenter(b);
                /// rotation
                if (rotation_angle_set || fullrotation_set) pos = Rotation(pos, angle, rotation_axis, rotation_center);
                /// perspective
                double perspective_scale = 1.0;
                if (perspective) {
                    perspective_scale = PerspectiveScale(pos, focallength, projection_center, view_coord);
                    pos = Perspective(pos, focallength, projection_center, view_coord);
                }
                // estimate of new min/max coordinate of bounding box for getting affected blocks;
                // as long as the blocks are roughly cubic, using LBmax doesn't add too much overhead;
                // go a little over the diagonal with the sqrt(3.1) factor
                for (unsigned int dir = X; dir <= Z; dir++) {
                    BB_for_getting_affected_blocks[b][dir][0] = pos[dir] - sqrt(3.1)*LBmax/2.0*perspective_scale;
                    BB_for_getting_affected_blocks[b][dir][1] = pos[dir] + sqrt(3.1)*LBmax/2.0*perspective_scale;
                }
            }
        }

        if (!make_starfield) { // default case: call to GetAffectedBlocks
            if (Verbose>1 && MyPE==0) cout<<" calling GetAffectedBlocks()"<<endl;
            AffectedBlocks = gg.GetAffectedBlocks(bounds, BB_for_getting_affected_blocks);
        }

        // prep sink particle grids and get affected blocks for starfield
        if (make_starfield) {
            if (Verbose>1 && MyPE==0) cout<<" starfield: setting up sink particle grids and affected blocks"<<endl;
            double racc = gg.ReadRealParameters()["sink_accretion_radius"];
            vector<vector<double> > bounds_for_sink(3, vector<double>(2));
            vector<int> NodeType = gg.GetNodeType();
            /// work on each sink particle position
            vector<double> particle_pos(3);
            for (int i=0; i<Particles.GetTypeCount()["sink"]; i++) {
                particle_pos[X] = sinks["posx"][i]; //+ pbc[X]*L[X]; /// TODO: periodic BCs
                particle_pos[Y] = sinks["posy"][i]; //+ pbc[Y]*L[Y];
                particle_pos[Z] = sinks["posz"][i]; //+ pbc[Z]*L[Z];
                /// rotation
                if (rotation_angle_set || fullrotation_set) particle_pos = Rotation(particle_pos, angle, rotation_axis, rotation_center);
                /// perspective
                double perspective_scale = 1.0;
                if (perspective) {
                    perspective_scale = PerspectiveScale(particle_pos, focallength, projection_center, view_coord);
                    particle_pos = Perspective(particle_pos, focallength, projection_center, view_coord);
                }
                /// each particle gets its own local (small) grid around itself
                double LsinkGrid = 2.1*racc*perspective_scale; // length of grid set by accretion radius
                bounds_for_sink[X][0] = particle_pos[view_coord[X]]-LsinkGrid/2.0;
                bounds_for_sink[X][1] = particle_pos[view_coord[X]]+LsinkGrid/2.0;
                bounds_for_sink[Y][0] = particle_pos[view_coord[Y]]-LsinkGrid/2.0;
                bounds_for_sink[Y][1] = particle_pos[view_coord[Y]]+LsinkGrid/2.0;
                bounds_for_sink[Z][0] = bounds[view_coord[Z]][0]; // lower projection edge
                bounds_for_sink[Z][1] = particle_pos[view_coord[Z]]; // particle position along projection direction
                if (Verbose>1 && MyPE==0) {
                    cout<<" starfield: sink i = "<<i<<", bounds_for_sink = "<<bounds_for_sink[X][0]<<" "<<bounds_for_sink[X][1]<<" "
                                                                            <<bounds_for_sink[Y][0]<<" "<<bounds_for_sink[Y][1]<<" "
                                                                            <<bounds_for_sink[Z][0]<<" "<<bounds_for_sink[Z][1]<<endl;
                }
                /// set bounds of projection grid for this sink
                grid2d[i+1].set_bnds(bounds_for_sink[X][0], bounds_for_sink[X][1],
                                     bounds_for_sink[Y][0], bounds_for_sink[Y][1],
                                     bounds_for_sink[Z][0], bounds_for_sink[Z][1]);
                /// append affected blocks for this sink to total AffectedBlocks list
                for (int b=0; b<gg.GetNumBlocksRep(); b++) {
                    if (NodeType[b % NumBlocks] == 1) { // LEAF block
                        bool overlap = true;
                        for (int dir=X; dir<=Z; dir++) {
                            if (BB_for_getting_affected_blocks[b][dir][1] < bounds_for_sink[dir][0]) { overlap = false; break; }
                            if (BB_for_getting_affected_blocks[b][dir][0] > bounds_for_sink[dir][1]) { overlap = false; break; }
                        }
                        if (overlap)
                            // check if b is not yet in AffectedBlocks
                            if (find(AffectedBlocks.begin(), AffectedBlocks.end(), b) == AffectedBlocks.end())
                                AffectedBlocks.push_back(b);
                    }
                }
            } // loop over sinks
        } // starfield

        // distribute affected blocks equally amongst processors (parallelisation)
        vector<int> MyBlocks;
        if (!make_starfield) MyBlocks = gg.GetMyBlocks(MyPE, NPE, AffectedBlocks, true); /// affected leaf blocks (allow_idle_cores=true)
        vector<int> MyParts;
        if (make_starfield) { // in this case, we parallelise over the sinks, but let each core have all blocks to work on
            MyBlocks = AffectedBlocks;
            if (Verbose && MyPE==0) cout<<" Parallelising over sink particles; each core has "<<MyBlocks.size()<<" blocks."<<endl;
            MyParts = Particles.GetMyParticles(MyPE, NPE, Particles.GetTypeCount()["sink"]);
        }

        /// loop over all blocks and fill containers
        cft.TimerStart("block loop");
        cft.InitProgressBar();
        for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
        {
            /// write progress
            if (MyPE==0) cft.PrintProgressBar(ib, MyBlocks.size());

            /// get actual block index
            int b = MyBlocks[ib];
            int b_all = b;
            if (periodic_boundary_conditions) b = b_all % NumBlocks; // take care of PBCs (if present)

            /// read block data
            float *block_data = 0;
            float *block_data_x = 0;
            float *block_data_y = 0;
            float *block_data_z = 0;
            if (!moment_maps) { // standard mode
                if (rotation_angle_set || fullrotation_set) { // check if we need to rotate a vector quantity
                    if (datasetname == "velx" || datasetname == "vely" || datasetname == "velz") {
                        block_data_x = gg.ReadBlockVar(b, "velx");
                        block_data_y = gg.ReadBlockVar(b, "vely");
                        block_data_z = gg.ReadBlockVar(b, "velz");
                    }
                    if (datasetname == "magx" || datasetname == "magy" || datasetname == "magz") {
                        block_data_x = gg.ReadBlockVar(b, "magx");
                        block_data_y = gg.ReadBlockVar(b, "magy");
                        block_data_z = gg.ReadBlockVar(b, "magz");
                    }
                }
                if (block_data_x == 0) {
                    // if here, we don't have a vector quantity or we don't do a rotation
                    block_data = gg.ReadBlockVar(b, datasetname);
                }
            } else { // moment maps
                if (view == "xy") block_data = gg.ReadBlockVar(b, "velz");
                if (view == "xz") block_data = gg.ReadBlockVar(b, "vely");
                if (view == "yz") block_data = gg.ReadBlockVar(b, "velx");
            }
            float *block_dens = 0;
            if (density_weight || moment_maps) block_dens = gg.ReadBlockVar(b, dens_name);

            /// prep for loop over cells in that block
            int split_factor = 1;
            if (split_cells) {
                split_factor = round(D[b][X] / Dmin[X]); // splitting each cell by factor (must be mutiple of 2)
                if (Verbose>1) cout << " ["<<MyPE<<"] block, split_factor = " << b << ", " << split_factor << endl;
            }
            int sfm1 = split_factor - 1;

            // cell deltas of splitted cells
            vector<double> Dc(3); Dc[X] = D[b][X]/split_factor; Dc[Y] = D[b][Y]/split_factor; Dc[Z] = D[b][Z]/split_factor;
            double cell_vol = Dc[X] * Dc[Y] * Dc[Z]; // cell volume

            /// loop over cells in that block
            for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++)
            {
                long index = k*NB[X]*NB[Y] + j*NB[X] + i;
                double cell_dat = 0.0;
                if (block_data != 0) cell_dat = block_data[index];

                // mass-weighting or moment_maps, if needed
                double cell_dens = 1.0;
                if (density_weight || moment_maps) cell_dens = block_dens[index];

                // for normalisation: this is either volume or mass (if density_weight or moment_maps)
                double cell_norm = cell_vol * cell_dens;

                // check if data is within data_range; if not, cycle to the next cell
                double for_range_check = cell_dat;
                if (moment_maps) for_range_check = cell_dens;
                if ((for_range_check < data_range[0]) || (for_range_check > data_range[1])) continue;

                vector<double> cell_center_orig = gg.CellCenter(b_all, i, j, k); // cell center position
                vector<double> cell_center(3); // create new splitted cell centers from original cell in loop below

                // loop over cell splits (loop over delta/2 multiples)
                for (int kr=-sfm1; kr<=sfm1; kr+=2) for (int jr=-sfm1; jr<=sfm1; jr+=2) for (int ir=-sfm1; ir<=sfm1; ir+=2)
                {
                    cell_center[X] = cell_center_orig[X] + ir*0.5*Dc[X];
                    cell_center[Y] = cell_center_orig[Y] + jr*0.5*Dc[Y];
                    cell_center[Z] = cell_center_orig[Z] + kr*0.5*Dc[Z];

                    /// spherical weight
                    double sp_weight = 1.0;
                    if (sp_weighting)
                        sp_weight = SphericalWeight(cell_center, rotation_center, view_coord);

                    /// rotation
                    if (rotation_angle_set || fullrotation_set) {
                        cell_center = Rotation(cell_center, angle, rotation_axis, rotation_center);
                        if (block_data_x != 0) { // rotate components of vector quantity
                            vector<double> vector_quantity(3);
                            vector_quantity[X] = block_data_x[index];
                            vector_quantity[Y] = block_data_y[index];
                            vector_quantity[Z] = block_data_z[index];
                            // rotate vector quantity
                            vector_quantity = Rotation(vector_quantity, angle, rotation_axis, rotation_center_for_vector_quantity);
                            // extract relevant vector component after rotation
                            if (datasetname=="velx" || datasetname=="magx") cell_dat = vector_quantity[X];
                            if (datasetname=="vely" || datasetname=="magy") cell_dat = vector_quantity[Y];
                            if (datasetname=="velz" || datasetname=="magz") cell_dat = vector_quantity[Z];
                        }
                    }

                    /// perspective
                    double perspective_scale = 1.0;
                    if (perspective) {
                        perspective_scale = PerspectiveScale(cell_center, focallength, projection_center, view_coord);
                        cell_center = Perspective(cell_center, focallength, projection_center, view_coord);
                    }

                    /// opacity
                    double opacity_weight = 1.0;
                    if (opacity_set)
                        opacity_weight = 2/(2+opacity)*(1+opacity*(proj_zmax-cell_center[view_coord[2]])/proj_depth);

                    // check if whole or part of cell intersects with projection range
                    if ((cell_center[view_coord[0]]+0.5*Dc[view_coord[0]] >= projection_range[2*view_coord[0]  ]) &&
                        (cell_center[view_coord[0]]-0.5*Dc[view_coord[0]] <= projection_range[2*view_coord[0]+1]) &&
                        (cell_center[view_coord[1]]+0.5*Dc[view_coord[1]] >= projection_range[2*view_coord[1]  ]) &&
                        (cell_center[view_coord[1]]-0.5*Dc[view_coord[1]] <= projection_range[2*view_coord[1]+1]) &&
                        (cell_center[view_coord[2]]+0.5*Dc[view_coord[2]] >= projection_range[2*view_coord[2]  ]) &&
                        (cell_center[view_coord[2]]-0.5*Dc[view_coord[2]] <= projection_range[2*view_coord[2]+1]) )
                    {

                        for (int g = 0; g < ng; g++) {
                            // scale the cell area contribution (perspective_scale * perspective_scale)
                            double cell_data = perspective_scale * perspective_scale * sp_weight * opacity_weight;
                            if (!moment_maps) { // standard mode
                                if (g == 0) cell_data *= cell_norm * cell_dat; // data
                                if (g == 1) cell_data *= cell_norm; // normalisation
                            } else { // moment maps
                                if (g == 0) cell_data *= cell_norm; // normalisation (0th moment)
                                if (g == 1) cell_data *= cell_norm * cell_dat; // 1st moment
                                if (g == 2) cell_data *= cell_norm * cell_dat * cell_dat; // 2nd moment
                            }
                            if (make_starfield) {
                                if (g > 0) continue; // only do this once to fill up the sink grids
                                vector<double> particle_pos(3);
                                //for (int i=0; i<Particles.GetTypeCount()["sink"]; i++) { // loop over sinks to fill grids with optical depth
                                for (unsigned int ip=0; ip<MyParts.size(); ip++) { // loop over sinks to fill grids with optical depth
                                    int i = MyParts[ip];
                                    particle_pos[X] = sinks["posx"][i]; //+ pbc[X]*L[X];
                                    particle_pos[Y] = sinks["posy"][i]; //+ pbc[Y]*L[Y];
                                    particle_pos[Z] = sinks["posz"][i]; //+ pbc[Z]*L[Z];
                                    /// rotation
                                    if (rotation_angle_set || fullrotation_set) particle_pos = Rotation(particle_pos, angle, rotation_axis, rotation_center);
                                    /// perspective
                                    if (perspective) particle_pos = Perspective(particle_pos, focallength, projection_center, view_coord);
                                    double depth_weight = 0.0;
                                    double dist = particle_pos[view_coord[Z]] - cell_center[view_coord[Z]];
                                    double column_length = particle_pos[view_coord[Z]] - projection_range[2*view_coord[Z]]; // so we get this in g/cm^2
                                    double opac = 0.1; // ~ 0.1cm^2/g @ T ~ 10K (see Semenov et al. 2003); but should use Drain opacities
                                    if (dist >= 0) depth_weight = 1.0 - exp(-dist/Dmin[view_coord[X]]); // switch on if cell is in front of sink
                                    grid2d[i+1].add_coord_fields(cell_center[view_coord[0]],
                                                                 cell_center[view_coord[1]],
                                                                 cell_center[view_coord[2]],
                                                                 Dc[view_coord[0]]*perspective_scale,
                                                                 Dc[view_coord[1]]*perspective_scale,
                                                                 Dc[view_coord[2]], cell_data*column_length*opac*depth_weight); // add to optical depth
                                }
                                continue; // skip the rest, as we're not doing anything to the main output grid here
                            }
                            if (sp_weight < 1e-6) continue; /// skip if the cell contribution is very small
                            /// finally, add cell_data to output projection grid
                            grid2d[g].add_coord_fields(cell_center[view_coord[0]],
                                                       cell_center[view_coord[1]],
                                                       cell_center[view_coord[2]],
                                                       Dc[view_coord[0]]*perspective_scale,
                                                       Dc[view_coord[1]]*perspective_scale,
                                                       Dc[view_coord[2]], cell_data);
                        } // loop over grids (g)
                    } // inside projection range
                } // loop over split cells
            } // loop over actual cells

            if (block_data != 0) { delete [] block_data; block_data = 0; }
            if (block_dens != 0) { delete [] block_dens; block_dens = 0; }
            if (block_data_x != 0) { delete [] block_data_x; block_data_x = 0; }
            if (block_data_y != 0) { delete [] block_data_y; block_data_y = 0; }
            if (block_data_z != 0) { delete [] block_data_z; block_data_z = 0; }

        } // loop over blocks

        cft.TimerStop("block loop");

        if (Verbose > 1) {
            cout << " ["<<MyPE<<"]: ";
            cft.TimerReport();
        }

        // if PBCs, remove block replicas (shrink BoundingBox and NumBlocksRep)
        if (periodic_boundary_conditions) gg.RemoveBlockReplicasPBCs();

        /// sum up each CPUs contribution
        if (Verbose>1 && MyPE==0) cout << "Now reduce..."<<endl;
        for (int g = 0; g < ng; g++) {
            // prep containers
            int ntot = grid2d[g].get_ntot();
            double *tmp_red = new double[ntot]; double *tmp = new double[ntot];
            int *tmp_int_red = new int[ntot]; int *tmp_int = new int[ntot];
            // field
            for (int n=0; n<ntot; n++) tmp[n] = grid2d[g].field[n];
            MPI_Allreduce(tmp, tmp_red, ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            for (int n=0; n<ntot; n++) grid2d[g].field[n] = tmp_red[n];
            // is_set
            for (int n=0; n<ntot; n++) if (grid2d[g].is_set[n]) tmp_int[n] = 1; else tmp_int[n] = 0;
            MPI_Allreduce(tmp_int, tmp_int_red, ntot, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            for (int n=0; n<ntot; n++) if (tmp_int_red[n] > 0) grid2d[g].is_set[n] = true;
            // clean
            delete [] tmp_int; delete [] tmp_int_red;
        }
        if (Verbose>1 && MyPE==0) cout << "...reduce done."<<endl;

        /// Normalisation, beam smearing and output (only Master PE)
        if (MyPE==0)
        {
            // Gaussian beam smearing (must be done before moment maps normalisation)
            if (gauss_smooth_fwhm > 0) {
                double fwhm_in_pixels = gauss_smooth_fwhm/grid2d[0].get_Lx()*grid2d[0].get_dimx();
                if (Verbose) cout<<"Applying Gaussian beam smoothing with FWHM "<<fwhm_in_pixels<<" pixels..."<<endl;
                if (gauss_smooth_fwhm > grid2d[0].get_Lx()) {
                    if (Verbose) cout<<"=== WARNING: gauss_smooth_fwhm > size of projection (range). "
                        <<"Smoothing with a kernel FWHM greater than the image size is very inaccurate! ==="<<endl;
                }
                for (int g = 0; g < ng; g++) {
                    double * grid2d_smoothed = cft.GaussSmooth(grid2d[g].field, grid2d[g].get_dimx(), grid2d[g].get_dimy(), fwhm_in_pixels);
                    for (int n = 0; n < grid2d[g].get_ntot(); n++) {
                        // =================================================================================needs changes in handling of NAN
                        if (grid2d[g].is_set[n]) grid2d[g].field[n] = grid2d_smoothed[n];
                    }
                    delete [] grid2d_smoothed;
                }
            }

            // normalize
            if (density_weight) {
                for (int n = 0; n < grid2d[0].get_ntot(); n++)
                    if (grid2d[0].is_set[n]) grid2d[0].field[n] /= grid2d[1].field[n];
            }

            // moment maps
            if (moment_maps) {
                for (int n = 0; n < grid2d[0].get_ntot(); n++)
                    if (grid2d[0].is_set[n]) {
                        // 1st moment map
                        grid2d[1].field[n] /= grid2d[0].field[n];
                        // 2nd moment map: sqrt( mean_sq - mean^2 )
                        grid2d[2].field[n] = sqrt( grid2d[2].field[n] / grid2d[0].field[n] - grid2d[1].field[n]*grid2d[1].field[n] );
                    }
            }

            // for moment maps, multiply by LOS
            if (moment_maps) {
                double los = projection_range[2*view_coord[2]+1] - projection_range[2*view_coord[2]+0];
                for (int n = 0; n < grid2d[0].get_ntot(); n++)
                    if (grid2d[0].is_set[n]) grid2d[0].field[n] *= los;
            }

            // starfield
            if (make_starfield) {
                // loop over sinks to compute their intensity reduced by exp(-tau), as seen by the observer
                for (int i=0; i<Particles.GetTypeCount()["sink"]; i++) { // loop over sinks to fill grids with optical depth
                    int Nc[2] = {grid2d[i+1].get_dimx(), grid2d[i+1].get_dimy()};
                    double Dc[2] = {grid2d[i+1].get_dx(), grid2d[i+1].get_dy()};
                    double racc_proj = grid2d[i+1].get_Lx()/2.1; // (Lx was set above to 2.1*racc*perspective_scale)
                    double cell_norm = Dc[X]*Dc[Y]*proj_depth; // normalisation
                    cell_norm *= sinks["luminosity"][i]; // scale by sink luminosity
                    double cell_center[2], dist[2];
                    for (int iy=0; iy<Nc[Y]; iy++) for (int ix=0; ix<Nc[X]; ix++) { // loop over all x and y cells of the grid
                        long n = grid2d[i+1].get_index(ix, iy, 0);
                        if (!grid2d[i+1].is_set[n]) continue;
                        double cell_data = cell_norm * exp(-grid2d[i+1].field[n]); // optical depth factor: exp(-tau)
                        /// add Gaussian blur
                        cell_center[X] = grid2d[i+1].getX(ix); cell_center[Y] = grid2d[i+1].getY(iy);
                        dist[X] = (ix - Nc[X]/2) * Dc[X]; // distance from cell center X
                        dist[Y] = (iy - Nc[Y]/2) * Dc[Y]; // distance from cell center Y
                        double r2 = dist[X]*dist[X] + dist[Y]*dist[Y];
                        cell_data *= sqrt(max(0.0, racc_proj*racc_proj - r2)) / racc_proj; // star surface model
                        grid2d[0].add_coord_fields(cell_center[X], cell_center[Y], proj_zmin+proj_depth/2.0,
                                                   Dc[X], Dc[Y], 1e-99, cell_data); // add intensity to final output grid
                    }
                }
            }

            // set all un-set cells to NaN
            for (int n=0; n<grid2d[0].get_ntot(); n++) if (!grid2d[0].is_set[n]) {
                grid2d[0].field[n] = std::numeric_limits<double>::quiet_NaN();
                if (moment_maps) {
                    grid2d[1].field[n] = std::numeric_limits<double>::quiet_NaN();
                    grid2d[2].field[n] = std::numeric_limits<double>::quiet_NaN();
                }
            }

            /// prepare output
            string direction_string;
            if (view == "xy") direction_string = "z";
            if (view == "xz") direction_string = "y";
            if (view == "yz") direction_string = "x";

            string outvarid = datasetname+"_proj";
            if (make_starfield) outvarid = "starfield_proj";
            if (slice) outvarid = datasetname+"_slice";
            if (moment_maps) outvarid = "moment_maps";

            /// write HDF5 output
            HDFIO HDFOutput = HDFIO();
            string hdf5filename = output_file_name;
            if (!use_output_file) {
                hdf5filename = inputfile+"_"+outvarid+"_"+direction_string;
            }
            if (rotation_angle_set)
                hdf5filename += "_rotated";
            if (fullrotation_set) {
                stringstream ss; ss << setw(3) << setfill('0') << irot;
                string irotstring = ss.str(); ss.clear();
                hdf5filename += "_R"+irotstring;
            }

            hdf5filename += ".h5";
            HDFOutput.create(hdf5filename);

            /// write projection
            vector<int> picture_dims(2);
            picture_dims[0] = grid2d[0].get_dimy();
            picture_dims[1] = grid2d[0].get_dimx();
            if (!moment_maps) { // standard case
                HDFOutput.write(grid2d[0].field, outvarid, picture_dims, H5T_NATIVE_DOUBLE);
            } else { // moment maps
                HDFOutput.write(grid2d[0].field, "moment_0", picture_dims, H5T_NATIVE_DOUBLE);
                HDFOutput.write(grid2d[1].field, "moment_1", picture_dims, H5T_NATIVE_DOUBLE);
                HDFOutput.write(grid2d[2].field, "moment_2", picture_dims, H5T_NATIVE_DOUBLE);
            }

            /// write direction string
            vector<int> hdf5dims(1); hdf5dims[0] = 1;
            HDFOutput.write(direction_string.c_str(), "direction", hdf5dims, H5T_C_S1);

            /// write sim time
            double sim_time = 0.0, redshift = 0.0;
            if (gg.GetGridType() != 'E') {
                map<string, double> scl_dbl = gg.ReadRealScalars();
                sim_time = scl_dbl["time"];
                redshift = scl_dbl["redshift"];
            } else {
                HDFIO hdfio = HDFIO();
                hdfio.open(inputfile, 'r');
                hdfio.read(&sim_time, "time", H5T_NATIVE_DOUBLE);
                hdfio.read(&redshift, "redshift", H5T_NATIVE_DOUBLE);
                hdfio.close();
            }
            HDFOutput.write(&sim_time, "time", hdf5dims, H5T_NATIVE_DOUBLE);
            HDFOutput.write(&redshift, "redshift", hdf5dims, H5T_NATIVE_DOUBLE);

            /// write minmax_xyz
            hdf5dims.resize(2); hdf5dims[0] = 3; hdf5dims[1] = 2;
            double minmax_xyz[6];
            minmax_xyz[0] = projection_range[0]; minmax_xyz[1] = projection_range[1]; //min, max, x
            minmax_xyz[2] = projection_range[2]; minmax_xyz[3] = projection_range[3]; //min, max, y
            minmax_xyz[4] = projection_range[4]; minmax_xyz[5] = projection_range[5]; //min, max, z
            HDFOutput.write(minmax_xyz, "minmax_xyz", hdf5dims, H5T_NATIVE_DOUBLE);

            /// write density_weight
            hdf5dims.resize(1); hdf5dims[0] = 1;
            int dens_weight[1]; dens_weight[0] = 0; if (density_weight) dens_weight[0] = 1;
            HDFOutput.write(dens_weight, "density_weight", hdf5dims, H5T_NATIVE_INT);

            /// add particle information to output file

            /// write numpart
            hdf5dims.resize(1); hdf5dims[0] = 1;
            HDFOutput.write(&np, "numpart", hdf5dims, H5T_NATIVE_INT);

            /// check if there are sink particles in the file; if so, read the accretion radius
            bool sink_particles_in_file = false;
            double r_accretion = 0.0;
            if (np > 0) {
                map<string, double> par_dbl = gg.ReadRealParameters();
                if (par_dbl.count("sink_accretion_radius") > 0) {
                    r_accretion = par_dbl["sink_accretion_radius"];
                    sink_particles_in_file = true;
                }
            }
            hdf5dims.resize(1); hdf5dims[0] = 1;
            HDFOutput.write(&r_accretion, "r_accretion", hdf5dims, H5T_NATIVE_DOUBLE);

            if (np > 0)
            {
                if (Verbose) cout << " Number of particles in file: " << np << endl;
                double * posx = Particles.ReadVar("posx");
                double * posy = Particles.ReadVar("posy");
                double * posz = Particles.ReadVar("posz");
                double * tag  = Particles.ReadVar("tag");
                double * mass;
                double * type;
                if (sink_particles_in_file) mass = Particles.ReadVar("mass");
                else { mass = new double[np]; for (long i=0; i<np; i++) mass[i] = 0.0; }
                /// see if 'type' is in particle names
                std::vector<std::string> p_names = Particles.GetPropertyNames();
                if (count(p_names.begin(), p_names.end(), "type") > 0) {
                    type = Particles.ReadVar("type");
                    // change from type=2 to type=0 for sink particles; type=1 would be tracer particles
                    long sink_count = 0;
                    for (long i=0; i<np; i++) {
                        if (type[i] == Particles.GetType()["sink"]) {
                            type[i] = 0.0;
                            sink_count++;
                        }
                    }
                    if (Verbose) cout<<" Processing "<<sink_count<<" sink particle(s) and "<<np-sink_count<<" other particle(s)."<<endl;
                } else { // fill it up with either 0 (sinks) or 1 (tracers)
                    type = new double[np];
                    if (!sink_particles_in_file) for (long i=0; i<np; i++) type[i] = 1.0; // tracer particles
                    else for (long i=0; i<np; i++) type[i] = 0.0; // sink particles
                }

                // write particle positions
                hdf5dims.resize(2); hdf5dims[0] = 3; hdf5dims[1] = np;
                double * ppos = new double[3*np];
                for (int dir = 0; dir < 3; dir++) {
                    for (long i = 0; i < np; i++) {
                        long index = dir*np + i;
                        if (dir == 0) ppos[index] = posx[i];
                        if (dir == 1) ppos[index] = posy[i];
                        if (dir == 2) ppos[index] = posz[i];
                    }
                }
                HDFOutput.write(ppos, "p_pos", hdf5dims, H5T_NATIVE_DOUBLE);
                delete [] ppos;

                // write particle masses
                hdf5dims.resize(1); hdf5dims[0] = np;
                double * pmass = new double[np];
                double * ptag  = new double[np];
                double * ptype = new double[np];
                for (long i = 0; i < np; i++) {
                    pmass[i] = mass[i];
                     ptag[i] =  tag[i];
                    ptype[i] = type[i];
                }
                HDFOutput.write(pmass, "p_mass", hdf5dims, H5T_NATIVE_DOUBLE);
                HDFOutput.write(ptag,  "p_tag",  hdf5dims, H5T_NATIVE_DOUBLE);
                HDFOutput.write(ptype, "p_type", hdf5dims, H5T_NATIVE_DOUBLE);
                delete [] pmass;
                delete [] ptag;
                delete [] ptype;

                /// particle information in projected region
                vector<double> particle_pos(3);
                vector<double> posx_proj;
                vector<double> posy_proj;
                vector<double> posz_proj;
                vector<double> mass_proj;
                vector<double>  tag_proj;
                vector<double> type_proj;

                // take care of periodic PBCs
                int pbc[3]; int pbc_nrep[3] = {0, 0, 0}; // this is the non-PBC case (i.e., 0 replicas)
                if (periodic_boundary_conditions) for (int dir = X; dir <= Z; dir++) pbc_nrep[dir] = 1;

                for (long i = 0; i < np; i++)
                {
                    for (pbc[Z] = -pbc_nrep[Z]; pbc[Z] <= pbc_nrep[Z]; pbc[Z]++) // loop over replicas in x, y, z
                    for (pbc[Y] = -pbc_nrep[Y]; pbc[Y] <= pbc_nrep[Y]; pbc[Y]++)
                    for (pbc[X] = -pbc_nrep[X]; pbc[X] <= pbc_nrep[X]; pbc[X]++)
                    {
                        particle_pos[X] = posx[i] + pbc[X]*L[X];
                        particle_pos[Y] = posy[i] + pbc[Y]*L[Y];
                        particle_pos[Z] = posz[i] + pbc[Z]*L[Z];

                        /// spherical weight
                        double sp_weight = 1.0;
                        if (sp_weighting)
                            sp_weight = SphericalWeight(particle_pos, rotation_center, view_coord);

                        /// rotation
                        if (rotation_angle_set || fullrotation_set)
                            particle_pos = Rotation(particle_pos, angle, rotation_axis, rotation_center);

                        /// perspective
                        if (perspective)
                            particle_pos = Perspective(particle_pos, focallength, projection_center, view_coord);

                        // check projection range
                        if ( (particle_pos[view_coord[0]] >= projection_range[view_coord[0]*2]) &&
                             (particle_pos[view_coord[0]] <= projection_range[view_coord[0]*2+1]) &&
                             (particle_pos[view_coord[1]] >= projection_range[view_coord[1]*2]) &&
                             (particle_pos[view_coord[1]] <= projection_range[view_coord[1]*2+1]) &&
                             (particle_pos[view_coord[2]] >= projection_range[view_coord[2]*2]) &&
                             (particle_pos[view_coord[2]] <= projection_range[view_coord[2]*2+1]) )
                        {
                            if (sp_weight < 1) continue; // if outside spherical sp_r_max, we remove the particle completely
                            posx_proj.push_back(particle_pos[view_coord[0]]);
                            posy_proj.push_back(particle_pos[view_coord[1]]);
                            posz_proj.push_back(particle_pos[view_coord[2]]);
                            mass_proj.push_back(mass[i]);
                            tag_proj.push_back( tag[i]);
                            type_proj.push_back(type[i]);
                        }
                    } // loop over replicas
                } // loop over particles

                // write out projected particle positions and masses
                const long np_proj = posx_proj.size();
                hdf5dims.resize(1); hdf5dims[0] = 1;
                HDFOutput.write(&np_proj, "numpart_proj", hdf5dims, H5T_NATIVE_INT);
                if (np_proj > 0)
                {
                    double * ppos_proj = new double[3*np_proj];
                    double * pmass_proj = new double[np_proj];
                    double * ptag_proj = new double[np_proj];
                    double * ptype_proj = new double[np_proj];
                    for (long i = 0; i < np_proj; i++)
                    {
                        ppos_proj[0*np_proj+i] = posx_proj[i];
                        ppos_proj[1*np_proj+i] = posy_proj[i];
                        ppos_proj[2*np_proj+i] = posz_proj[i];
                        pmass_proj[i]          = mass_proj[i];
                         ptag_proj[i]          =  tag_proj[i];
                        ptype_proj[i]          = type_proj[i];
                    }
                    hdf5dims.resize(2); hdf5dims[0] = 3; hdf5dims[1] = np_proj;
                    HDFOutput.write(ppos_proj, "p_pos_proj", hdf5dims, H5T_NATIVE_DOUBLE);
                    hdf5dims.resize(1); hdf5dims[0] = np_proj;
                    HDFOutput.write(pmass_proj, "p_mass_proj", hdf5dims, H5T_NATIVE_DOUBLE);
                    hdf5dims.resize(1); hdf5dims[0] = np_proj;
                    HDFOutput.write(ptag_proj, "p_tag_proj", hdf5dims, H5T_NATIVE_DOUBLE);
                    hdf5dims.resize(1); hdf5dims[0] = np_proj;
                    HDFOutput.write(ptype_proj, "p_type_proj", hdf5dims, H5T_NATIVE_DOUBLE);
                    delete [] ppos_proj;
                    delete [] pmass_proj;
                    delete [] ptag_proj;
                    delete [] ptype_proj;
                }

                /// clean
                delete [] posx; delete [] posy; delete [] posz;
                delete [] mass; delete [] tag; delete [] type;

            } /// np > 0

            // close hdf5 file
            HDFOutput.close();
            if (Verbose) cout << " '" << hdf5filename << "' written." << endl;

        } // MyPE == 0

    } /// end roation loop

    /// print out wallclock time used
    long endtime = time(NULL);
    long duration = endtime-starttime, duration_red = 0;
    if (Verbose>1) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (Verbose && MyPE==0) cout << "****************** Global time to finish = "<<duration_red<<"s ******************" << endl;

    MPI_Finalize();
    return 0;

} // end main


/// Rotation
/// assumes that v is a unity vector pointing along the rotation axis
inline vector<double> Rotation(const vector<double> &inp, const double phi, const vector<double> &v, const vector<double> &rc)
{
    vector<double> outp(3);
    const double sinp = sin(phi);
    const double cosp = cos(phi);
    const double omcp = 1.-cosp;
    outp[X] = (cosp+v[X]*v[X]*omcp)     *(inp[X]-rc[X]) + (v[X]*v[Y]*omcp-v[Z]*sinp)*(inp[Y]-rc[Y]) + (v[X]*v[Z]*omcp+v[Y]*sinp)*(inp[Z]-rc[Z]);
    outp[Y] = (v[Y]*v[X]*omcp+v[Z]*sinp)*(inp[X]-rc[X]) + (cosp+v[Y]*v[Y]*omcp)     *(inp[Y]-rc[Y]) + (v[Y]*v[Z]*omcp-v[X]*sinp)*(inp[Z]-rc[Z]);
    outp[Z] = (v[Z]*v[X]*omcp-v[Y]*sinp)*(inp[X]-rc[X]) + (v[Z]*v[Y]*omcp+v[X]*sinp)*(inp[Y]-rc[Y]) + (cosp+v[Z]*v[Z]*omcp)     *(inp[Z]-rc[Z]);
    outp[X] += rc[X];
    outp[Y] += rc[Y];
    outp[Z] += rc[Z];
    return outp;
}

/// PerspectiveScale
inline double PerspectiveScale(const vector<double> &inp, const double focallength, const vector<double> &projection_center, const vector<int> &view_coord)
{
    return focallength/(focallength+inp[view_coord[2]]-projection_center[view_coord[2]]);
}

/// Perspective
inline vector<double> Perspective(const vector<double> &inp, const double focallength, const vector<double> &projection_center, const vector<int> &view_coord)
{
    vector<double> outp = inp;
    double perspective_scale = PerspectiveScale(inp, focallength, projection_center, view_coord);
    outp[view_coord[0]] -= projection_center[view_coord[0]];
    outp[view_coord[0]] *= perspective_scale;
    outp[view_coord[0]] += projection_center[view_coord[0]];
    outp[view_coord[1]] -= projection_center[view_coord[1]];
    outp[view_coord[1]] *= perspective_scale;
    outp[view_coord[1]] += projection_center[view_coord[1]];
    return outp;
}

// SphericalWeight
inline double SphericalWeight(const vector<double> &inp, const vector<double> &rotation_center, const vector<int> &view_coord)
{
    double dr1 = inp[view_coord[0]] - rotation_center[view_coord[0]];
    double dr2 = inp[view_coord[1]] - rotation_center[view_coord[1]];
    double dr3 = inp[view_coord[2]] - rotation_center[view_coord[2]];
    double radius = sqrt(dr1*dr1+dr2*dr2+dr3*dr3);
    double sp_weight = 1.0;
    if (radius >= sp_r_max) sp_weight = exp(-(radius-sp_r_max)/(sp_r_chr-sp_r_max));
    return sp_weight;
}

/** ------------------------- ParseInputs ----------------------------
 **  Parses the command line Arguments
 ** ------------------------------------------------------------------ */
int ParseInputs(const vector<string> Argument)
{
    stringstream dummystream;

    /// read tool specific options
    if (Argument.size() < 2)
    {
        if (MyPE==0) { cout << endl << "ParseInputs: Invalid number of arguments." << endl; }
        return -1;
    }
    inputfile = Argument[1];

    for (unsigned int i = 2; i < Argument.size(); i++)
    {
        if (Argument[i] != "" && Argument[i] == "-dset")
        {
            if (Argument.size()>i+1) datasetname = Argument[i+1]; else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-pixel")
        {
            if (Argument.size()>i+2) {
                dummystream << Argument[i+1]; dummystream >> pixel[0]; dummystream.clear();
                dummystream << Argument[i+2]; dummystream >> pixel[1]; dummystream.clear();
                user_pixels_set = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-view")
        {
            if (Argument.size()>i+1) view = Argument[i+1]; else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-range")
        {
            if (Argument.size()>i+6) {
                dummystream << Argument[i+1]; dummystream >> projection_range[0]; dummystream.clear();
                dummystream << Argument[i+2]; dummystream >> projection_range[1]; dummystream.clear();
                dummystream << Argument[i+3]; dummystream >> projection_range[2]; dummystream.clear();
                dummystream << Argument[i+4]; dummystream >> projection_range[3]; dummystream.clear();
                dummystream << Argument[i+5]; dummystream >> projection_range[4]; dummystream.clear();
                dummystream << Argument[i+6]; dummystream >> projection_range[5]; dummystream.clear();
                if ( (projection_range[1] < projection_range[0]) ||
                     (projection_range[3] < projection_range[2]) ||
                     (projection_range[5] < projection_range[4]) ) {
                        if (MyPE==0) cout<<"ParseInputs: something wrong with projection range."<<endl;
                        return -1;
                }
                user_proj_range = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-zoom")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> zoom_factor; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-slice")
        {
            slice = true;
        }
        if (Argument[i] != "" && Argument[i] == "-mw")
        {
            density_weight = true;
        }
        if (Argument[i] != "" && Argument[i] == "-data_range")
        {
            if (Argument.size()>i+2) {
                dummystream << Argument[i+1]; dummystream >> data_range[0]; dummystream.clear();
                dummystream << Argument[i+2]; dummystream >> data_range[1]; dummystream.clear();
                if ( (data_range[1] < data_range[0])) {
                        if (MyPE==0) cout<<"ParseInputs: something wrong with data range."<<endl;
                        return -1;
                }
                user_data_range = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-rotcenter")
        {
            if (Argument.size()>i+3) {
                dummystream << Argument[i+1]; dummystream >> rotation_center[0]; dummystream.clear();
                dummystream << Argument[i+2]; dummystream >> rotation_center[1]; dummystream.clear();
                dummystream << Argument[i+3]; dummystream >> rotation_center[2]; dummystream.clear();
                rotation_center_set = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-rotaxis")
        {
            if (Argument.size()>i+3) {
                dummystream << Argument[i+1]; dummystream >> rotation_axis[0]; dummystream.clear();
                dummystream << Argument[i+2]; dummystream >> rotation_axis[1]; dummystream.clear();
                dummystream << Argument[i+3]; dummystream >> rotation_axis[2]; dummystream.clear();
                rotation_axis_set = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-rotangle")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> rotation_angle; dummystream.clear();
                rotation_angle_set = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-fullrotation")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> fullrotation; dummystream.clear();
                fullrotation_set = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-opacity")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> opacity; dummystream.clear();
                opacity_set = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-sp_r_max")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> sp_r_max; dummystream.clear();
                sp_weighting = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-sp_r_chr")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> sp_r_chr; dummystream.clear();
                sp_weighting = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-viewangle")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> viewangle; dummystream.clear();
                perspective = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-moment_maps")
        {
            moment_maps = true;
        }
        if (Argument[i] != "" && Argument[i] == "-gauss_smooth_fwhm")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> gauss_smooth_fwhm; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-no_particles")
        {
            process_particles = false;
        }
        if (Argument[i] != "" && Argument[i] == "-make_starfield")
        {
            make_starfield = true;
        }
        if (Argument[i] != "" && Argument[i] == "-bc")
        {
            if (Argument.size()>i+1) boundary_condition = Argument[i+1]; else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-split_cells")
        {
            split_cells = true;
        }
        if (Argument[i] != "" && Argument[i] == "-o")
        {
            if (Argument.size()>i+1) output_file_name = Argument[i+1]; else return -1;
            use_output_file = true;
        }
        if (Argument[i] != "" && Argument[i] == "-verbose")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> Verbose; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-h")
        {
            HelpMe();
            MPI_Finalize(); exit(0);
        }

    } // loop over all args

    /// print out parsed values
    if (Verbose && MyPE==0) {
        cout << " ParseInputs: Command line arguments: ";
        for (unsigned int i = 0; i < Argument.size(); i++) cout << Argument[i] << " ";
        cout << endl;
    }
    return 0;

} // end: ParseInputs()


/** --------------------------- HelpMe -------------------------------
 **  Prints out helpful usage information to the user
 ** ------------------------------------------------------------------ */
void HelpMe(void)
{
    if (MyPE==0) {
        cout << endl
        << "Syntax:" << endl
        << " projection <filename> [<OPTIONS>]" << endl << endl
        << "   <OPTIONS>:           " << endl
        << "     -dset <datasetname>                    : datasetname to be processed (default: dens)" << endl
        << "     -pixel <Nx Ny>                         : pixels for projected image (default: 512 512)" << endl
        << "     -view <view>                           : view can be xy, xz, or yz (default: xy)" << endl
        << "     -range <xmin xmax ymin ymax zmin zmax> : projection range in sim coordinates (default is full simulation range)" << endl
        << "     -zoom <zoom_factor>                    : zoom by zoom_factor (default 1.0)" << endl
        << "     -slice                                 : make a slice (set min=max at slice position)" << endl
        << "     -mw                                    : mass weighting" << endl
        << "     -data_range <dmin dmax>                : range in data to be considered (if outside, it won't contribute to the projection)" << endl
        << "     -rotcenter <X Y Z>                     : specify rotation center (default is box center)" << endl
        << "     -rotaxis <X Y Z>                       : specify rotation axis (default: 0 0 1)" << endl
        << "     -rotangle <angle>                      : specify rotation angle in degrees (default: 0.0)" << endl
        << "     -fullrotation <number>                 : make a 360° rotation with <number> images" << endl
        << "     -opacity <opacity>                     : add linear opacity weight (default: 0.0)" << endl
        << "     -sp_r_max <radius>                     : start radius for spherical weighting (default: off)" << endl
        << "     -sp_r_chr <radius>                     : characteristic radius for spherical weighting (default: off; must be >sp_r_max)" << endl
        << "     -viewangle <angle in degrees>          : use perspective view with angle (default: off)" << endl
        << "     -moment_maps                           : make 0th, 1st, and 2nd moment maps (-dset is ignored; not to be used with -slice or -mw; -data_range acts on dens)" << endl
        << "     -gauss_smooth_fwhm <fwhm>              : Gaussian smoothing with FWHM (in coordinate units, i.e., same units as -range)" << endl
        << "     -no_particles                          : do not process particles" << endl
        << "     -make_starfield                        : special case to make (sink particle) starfield (do not use with -slice, -mw, -no_particles)" << endl
        << "     -bc <boundary_condition>               : specify the boundary conditions to be used: 'isolated' or 'periodic' (default: automatic)" << endl
        << "     -split_cells                           : split cells to finest level (default: off)" << endl
        << "     -o <filename>                          : specify output filename" << endl
        << "     -verbose <level>                       : verbose level (0, 1, 2) (default: 1)" << endl
        << "     -h                                     : print this help message" << endl
        << endl
        << "Example: projection DF_hdf5_plt_cnt_0020 -dset dens -pixel 512 512 -view xy -range 0 1 0 1 0 1"
        << endl << endl;
    }
}
