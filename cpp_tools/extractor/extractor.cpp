/*

  extractor.cpp

  Extracts the whole or a portion of grid variable(s)
  from FLASH (Uniform Grid or AMR) output files
  to a uniform grid of arbitrary size

  By Christoph Federrath, 2013-2025

*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include "mpi.h" /// MPI lib
#include "../Libs/HDFIO.h" // HDF5 IO
#include "../Libs/FlashGG.h" /// Flash General Grid class
#include "../Libs/FlashParticles.h" // for sink particle IO
#include "../Libs/GRID3D.h" // grid for the image

// constants
#define NDIM 3
using namespace std;
enum {X, Y, Z};
static const bool Debug = false;
static const char dens_name[] = "dens";

// MPI stuff
int MyPE = 0, NPE = 1;

// some global stuff (inputs)
string inputfile = "";
vector<string> datasetnames; // list of datasetnames to extract
vector<int> pixel(3,128); // number of pixels for extraction
bool user_bounds = false; // if user does not set this, then do full domain extraction
vector<vector<double> > bounds(3, vector<double>(2,0.0)); // extraction bounding box
bool user_weights = false; // if user wants to set mass weighting flags for each dataset
vector<int> mws; // mass weights for each dataset (for standard datasets, this is set automatically)
bool user_div_factor = false;
vector<int> div_factor(3, 1); // this controls in how many serial parts we divide the output domain
bool use_output_file = false;
string output_file_name = "";

// forward functions
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
    if (MyPE==0) cout<<setprecision(16)<<" === extractor === using MPI num procs: "<<NPE<<endl;

    /// Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) == -1)
    {
        if (MyPE==0) cout << endl << "Error in ParseInputs(). Exiting." << endl;
        HelpMe();
        MPI_Finalize(); return 0;
    }

    long starttime = time(NULL);

    /// FLASH file meta data
    FlashGG gg = FlashGG(inputfile, 0);
    int NumBlocks = gg.GetNumBlocks();
    vector<int> NB = gg.GetNumCellsInBlock();
    vector< vector<double> > MinMaxDomain = gg.GetMinMaxDomain();
    vector<double> L = gg.GetL();
    vector<vector<double> > D = gg.GetDblock();
    vector<double> Dmin = gg.GetDmin();
    vector<double> Dmax = gg.GetDmax();

    if (MyPE==0) gg.PrintInfo();

    /// set the extraction ranges
    if (!user_bounds) {
        for (int dim = 0; dim < 3; dim++) {
            bounds[dim][0] = MinMaxDomain[dim][0];
            bounds[dim][1] = MinMaxDomain[dim][1];
        }
    }

    // prep output filename
    string hdf5filename = output_file_name;
    if (!use_output_file) hdf5filename = inputfile+"_extracted.h5";

    /// check if we have periodic boundary conditions
    bool periodic_boundary_conditions = false;
    std::map<std::string, std::string> str_parms = gg.ReadStringParameters();
    // PBCs are currently only supported for all directions, so only check xl_boundary_type
    if (str_parms.at("xl_boundary_type") == "periodic") periodic_boundary_conditions = true;
    if (MyPE==0) cout<<" periodic_boundary_conditions = "<<periodic_boundary_conditions<<endl;

    // if PBCs, create block replicas (extend BoundingBox and NumBlocksRep)
    if (periodic_boundary_conditions) gg.AddBlockReplicasPBCs();

    // loop over datasetnames to extract
    for (unsigned int d=0; d<datasetnames.size(); d++) {

        bool mass_weighting = false;
        if (user_weights) { // the user provided a list of mass-weighting flags for each datasetname
            if (mws[d]==1) mass_weighting = true;
        } else {
            // set mass weighting for standard datasets
            if (datasetnames[d]=="velx") mass_weighting = true;
            if (datasetnames[d]=="vely") mass_weighting = true;
            if (datasetnames[d]=="velz") mass_weighting = true;
        }

        if (MyPE==0) cout<<">>> Extracting dataset '"<<datasetnames[d]
                         <<"' (mass weighting = "<<mass_weighting
                         <<") to uniform grid ("<<pixel[X]<<" "<<pixel[Y]<<" "<<pixel[Z]<<")..."<<endl;

        // automatically determine a reasonable div_factor, if the user has not provided one
        if (!user_div_factor) {
            for (int dim = 0; dim < 3; dim++) {
                div_factor[dim] = max(1, pixel[dim] / 256); // if extraction is >=512 cells, we start dividing the extraction domain
            }
        }
        if (MyPE==0) cout<<" using div_factor "<<div_factor[X]<<" "<<div_factor[Y]<<" "<<div_factor[Z]<<endl;

        // prepare to divide the output domain in div_factor[dim] parts for serial processing
        vector<vector<double> > bounds_part(3, vector<double>(2,0.0)); // bounding box for extraction parts
        vector<int> pixel_part(3); // number of pixels for extraction parts
        vector<double> Lext(3), Lpart(3);
        for (int dim = 0; dim < 3; dim++) {
            pixel_part[dim] = ceil( (double)pixel[dim] / (double)div_factor[dim] );
            Lext[dim] = bounds[dim][1] - bounds[dim][0];
            Lpart[dim] = Lext[dim] * (double)pixel_part[dim] / (double)pixel[dim];
        }

        // this starts the monster loop that goes over the serial extraction parts
        for (int kd = 0; kd < div_factor[Z]; kd++)
        for (int jd = 0; jd < div_factor[Y]; jd++)
        for (int id = 0; id < div_factor[X]; id++)
        {
            if (MyPE==0) cout << " loop over parts: "<< id+1 << "/" << div_factor[X] << ", "
                                                     << jd+1 << "/" << div_factor[Y] << ", "
                                                     << kd+1 << "/" << div_factor[Z] << endl;

            /// compute the bounds for the extraction parts
            bounds_part[X][0] = bounds[X][0] + id*Lpart[X];
            bounds_part[X][1] = bounds_part[X][0] + Lpart[X];
            bounds_part[Y][0] = bounds[Y][0] + jd*Lpart[Y];
            bounds_part[Y][1] = bounds_part[Y][0] + Lpart[Y];
            bounds_part[Z][0] = bounds[Z][0] + kd*Lpart[Z];
            bounds_part[Z][1] = bounds_part[Z][0] + Lpart[Z];
            // for the last extraction part in each x, y, z, we need to compute the rest
            if (id == div_factor[X]-1) { // last part in x
                bounds_part[X][1] = bounds[X][1];
                pixel_part[X] = (int)round(pixel[X] * (bounds_part[X][1]-bounds_part[X][0]) / Lext[X]);
            } else // back to normal pixel_part
                pixel_part[X] = ceil( (double)pixel[X] / (double)div_factor[X] );
            if (jd == div_factor[Y]-1) { // last part in y
                bounds_part[Y][1] = bounds[Y][1];
                pixel_part[Y] = (int)round(pixel[Y] * (bounds_part[Y][1]-bounds_part[Y][0]) / Lext[Y]);
            } else // back to normal pixel_part
                pixel_part[Y] = ceil( (double)pixel[Y] / (double)div_factor[Y] );
            if (kd == div_factor[Z]-1) { // last part in z
                bounds_part[Z][1] = bounds[Z][1];
                pixel_part[Z] = (int)round(pixel[Z] * (bounds_part[Z][1]-bounds_part[Z][0]) / Lext[Z]);
            } else // back to normal pixel_part
                pixel_part[Z] = ceil( (double)pixel[Z] / (double)div_factor[Z] );

            if (Debug &&  MyPE==0) cout<<" pixel_part = "<<pixel_part[X]<<" "<<pixel_part[Y]<<" "<<pixel_part[Z]<<" "<<endl;
            if (Debug &&  MyPE==0) cout<<" bounds_part = "<<bounds_part[X][0]<<" "<<bounds_part[X][1]<< "  "
                                                          <<bounds_part[Y][0]<<" "<<bounds_part[Y][1]<< "  "
                                                          <<bounds_part[Z][0]<<" "<<bounds_part[Z][1]<<endl;

            /// create grids (with Nx x Ny x Nz extraction part pixels)
            if (Debug && MyPE==0) { cout<<" creating extraction grids..."<<endl; }
            GRID3D grid      = GRID3D(pixel_part[X], pixel_part[Y], pixel_part[Z]);
            GRID3D grid_norm = GRID3D(pixel_part[X], pixel_part[Y], pixel_part[Z]);
            if (Debug && MyPE==0) { cout<<" ...extraction grids created."<<endl; }

            /// set grid bounds
            grid.set_bnds     (bounds_part[X][0], bounds_part[X][1],
                               bounds_part[Y][0], bounds_part[Y][1],
                               bounds_part[Z][0], bounds_part[Z][1]);
            grid_norm.set_bnds(bounds_part[X][0], bounds_part[X][1],
                               bounds_part[Y][0], bounds_part[Y][1],
                               bounds_part[Z][0], bounds_part[Z][1]);
            if (Debug && MyPE==0) grid.print_bnds();

            /// clear fields
            grid.clear();
            grid_norm.clear();

            // Get list of blocks overlapping with requested extraction bounds
            vector<int> AffectedBlocks = gg.GetAffectedBlocks(bounds_part);
            /// decompose domain in leaf blocks
            vector<int> MyBlocks = gg.GetMyBlocks(MyPE, NPE, AffectedBlocks);

            /// loop over all blocks and fill containers
            bool printed_progress_1 = false, printed_progress_10 = false, printed_progress_100 = false;
            for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
            {
                // write progress
                double percent_done = (double)(ib+1)/MyBlocks.size()*100;
                bool print_progress = false;
                if (percent_done >    1.0 && !printed_progress_1  ) {print_progress=true; printed_progress_1  =true;}
                if (percent_done >   10.0 && !printed_progress_10 ) {print_progress=true; printed_progress_10 =true;}
                if (percent_done == 100.0 && !printed_progress_100) {print_progress=true; printed_progress_100=true;}
                if (print_progress && MyPE==0) cout<<" ["<<MyPE<<"] "<<percent_done<<"% done..."<<endl;

                int b = MyBlocks[ib];
                int b_all = b;
                if (periodic_boundary_conditions) b = b_all % NumBlocks; // take care of PBCs (if present)

                /// read block data
                float *block_data = gg.ReadBlockVar(b, datasetnames[d]);
                float *block_dens = 0;
                if (mass_weighting) block_dens = gg.ReadBlockVar(b, dens_name);

                /// loop over cells in that block
                for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++)
                {
                    long index = k*NB[X]*NB[Y] + j*NB[X] + i;
                    double cell_vol = D[b][X]*D[b][Y]*D[b][Z];
                    double cell_dat = block_data[index];
                    double cell_dens = 1.;
                    if (mass_weighting) cell_dens = block_dens[index];
                    double cell_norm = cell_vol*cell_dens;
                    double cell_datnorm = cell_dat*cell_norm;

                    vector<double> cell_center = gg.CellCenter(b_all, i, j, k);

                    // check if whole or part of cell intersects with extraction range
                    if ((cell_center[X]+D[b][X]/2. >= bounds_part[X][0]) &&
                        (cell_center[X]-D[b][X]/2. <= bounds_part[X][1]) &&
                        (cell_center[Y]+D[b][Y]/2. >= bounds_part[Y][0]) &&
                        (cell_center[Y]-D[b][Y]/2. <= bounds_part[Y][1]) &&
                        (cell_center[Z]+D[b][Z]/2. >= bounds_part[Z][0]) &&
                        (cell_center[Z]-D[b][Z]/2. <= bounds_part[Z][1]) )
                    {
                        grid.add_coord_fields(cell_center[X],
                                              cell_center[Y],
                                              cell_center[Z],
                                              D[b][X], D[b][Y], D[b][Z], cell_datnorm);
                        grid_norm.add_coord_fields(cell_center[X],
                                                   cell_center[Y],
                                                   cell_center[Z],
                                                   D[b][X], D[b][Y], D[b][Z], cell_norm);
                    }

                } //end loop over cells

                delete [] block_data; if (mass_weighting) delete [] block_dens;

            } //end loop over blocks

            /// sum up each CPUs contribution
            int ntot = grid.get_ntot();
            double *tmp_red = new double[ntot]; double *tmp = new double[ntot];
            for (int n=0; n<ntot; n++) tmp[n] = grid.field[n];
            MPI_Allreduce(tmp, tmp_red, ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            for (int n=0; n<ntot; n++) grid.field[n] = tmp_red[n];
            for (int n=0; n<ntot; n++) tmp[n] = grid_norm.field[n];
            MPI_Allreduce(tmp, tmp_red, ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            for (int n=0; n<ntot; n++) grid_norm.field[n] = tmp_red[n];
            delete [] tmp; delete [] tmp_red;
            int *tmp_int_red = new int[ntot]; int *tmp_int = new int[ntot];
            for (int n=0; n<ntot; n++) if (grid.is_set[n]) tmp_int[n] = 1; else tmp_int[n] = 0;
            MPI_Allreduce(tmp_int, tmp_int_red, ntot, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            for (int n=0; n<ntot; n++) {
                if (tmp_int_red[n] > 0) { grid.is_set[n] = true; grid_norm.is_set[n] = true; }
                else                    { grid.is_set[n] = false; grid_norm.is_set[n] = false; }
            }
            delete [] tmp_int; delete [] tmp_int_red;

            /// normalize now
            for (int n = 0; n < grid.get_ntot(); n++)
                if (grid.is_set[n])
                    grid.field[n] /= grid_norm.field[n];

            /// OUTPUT (only Master PE writes data to disk)
            if (MyPE==0)
            {
                /// open/create HDF5 output file
                HDFIO HDFOutput = HDFIO();
                ifstream ifstream_outfile(hdf5filename.c_str());
                if (ifstream_outfile) { // overwrite datasets in file
                    HDFOutput.open(hdf5filename,'w');
                    // if (id==0 && jd==0 && kd==0 && d==0) HDFOutput.delete_datasets();
                }
                else { // create new file
                    HDFOutput.create(hdf5filename);
                }
                // create empty dataset
                if (id==0 && jd==0 && kd==0) {
                    HDFOutput.delete_dataset(datasetnames[d]); // first delete if dset already exists
                    HDFOutput.create_dataset(datasetnames[d], pixel, H5T_NATIVE_FLOAT, MPI_COMM_NULL);
                }
                /// write extracted dataset
                float *outfield = new float[grid.get_ntot()];
                for (int n = 0; n < grid.get_ntot(); n++) outfield[n] = (float)(grid.field[n]);
                hsize_t offset[3]; if (Debug) cout << " offset = ";
                for (int dim = 0; dim < 3; dim++) { // work out the offset
                    offset[2-dim] = (int)round(pixel[dim] * (bounds_part[dim][0]-bounds[dim][0]) / Lext[dim]);
                    if (Debug) cout << offset[2-dim] << " ";
                } if (Debug) cout << endl;
                hsize_t count[3] = {(hsize_t)pixel_part[Z], (hsize_t)pixel_part[Y], (hsize_t)pixel_part[X]};
                hsize_t out_offset[3] = {0, 0, 0};
                hsize_t out_count[3] = {(hsize_t)pixel_part[Z], (hsize_t)pixel_part[Y], (hsize_t)pixel_part[X]};
                HDFOutput.overwrite_slab(outfield, datasetnames[d], H5T_NATIVE_FLOAT, offset, count, 3, out_offset, out_count);
                delete [] outfield;
                // close hdf5 file
                HDFOutput.close();
            }

        } // loop over extraction parts

        if (MyPE==0) cout<<">>> Dataset '"<<datasetnames[d]<<"' in '"<<hdf5filename<<"' written."<<endl;

    } //end loop over datatsetnames (d)

    // if PBCs, remove block replicas (shrink BoundingBox and NumBlocksRep)
    if (periodic_boundary_conditions) gg.RemoveBlockReplicasPBCs();

    /// OUTPUT of meta and particle data (only Master PE writes data to disk)
    if (MyPE==0)
    {
        /// open/create HDF5 output file
        HDFIO HDFOutput = HDFIO();
        ifstream ifstream_outfile(hdf5filename.c_str());
        if (ifstream_outfile) { // overwrite datasets in file
            HDFOutput.open(hdf5filename,'w');
        }
        else { // create new file
            HDFOutput.create(hdf5filename);
        }

        vector<int> hdf5dims(1);

        /// write sim time
        hdf5dims.resize(1); hdf5dims[0] = 1;
        map<string, double> scl_dbl = gg.ReadRealScalars();
        double sim_time = scl_dbl.at("time");
        HDFOutput.delete_dataset("time");
        HDFOutput.write(&sim_time, "time", hdf5dims, H5T_NATIVE_DOUBLE);
        double redshift = scl_dbl.at("redshift");
        HDFOutput.delete_dataset("redshift");
        HDFOutput.write(&redshift, "redshift", hdf5dims, H5T_NATIVE_DOUBLE);

        /// write minmax_xyz
        hdf5dims.resize(2); hdf5dims[0] = 3; hdf5dims[1] = 2;
        double minmax_xyz[6];
        minmax_xyz[0] = bounds[X][0]; minmax_xyz[1] = bounds[X][1]; //min, max, x
        minmax_xyz[2] = bounds[Y][0]; minmax_xyz[3] = bounds[Y][1]; //min, max, y
        minmax_xyz[4] = bounds[Z][0]; minmax_xyz[5] = bounds[Z][1]; //min, max, z
        HDFOutput.delete_dataset("minmax_xyz");
        HDFOutput.write(minmax_xyz, "minmax_xyz", hdf5dims, H5T_NATIVE_DOUBLE);

        /// write extraction dimensions
        hdf5dims.resize(1); hdf5dims[0] = 3;
        int dims_xyz[3]; dims_xyz[X] = pixel[X]; dims_xyz[Y] = pixel[Y]; dims_xyz[Z] = pixel[Z];
        HDFOutput.delete_dataset("dims_xyz");
        HDFOutput.write(dims_xyz, "dims_xyz", hdf5dims, H5T_NATIVE_INT);

        /// write boundary conditions
        int pbc = 0;
        if (periodic_boundary_conditions) {
            pbc = 1;
            for (int dim = 0; dim < 3; dim++) { // if the extraction is not over the whole domain size, pbc = 0
                if (bounds[dim][1] - bounds[dim][0] != L[dim]) pbc = 0;
            }
        }
        hdf5dims.resize(1); hdf5dims[0] = 1;
        HDFOutput.delete_dataset("pbc");
        HDFOutput.write(&pbc, "pbc", hdf5dims, H5T_NATIVE_INT);

        /// read particles
        FlashParticles Particles = FlashParticles(inputfile);
        const long np = Particles.GetN();

        /// write numpart
        hdf5dims.resize(1); hdf5dims[0] = 1;
        HDFOutput.delete_dataset("numpart");
        HDFOutput.write(&np, "numpart", hdf5dims, H5T_NATIVE_INT);

        /// check if there are sink particles in the file; if so, read the accretion radius
        bool sink_particles_in_file = false;
        double r_accretion = 0.0;
        if (np > 0) {
            map<string, double> par_dbl = gg.ReadRealParameters();
            if (par_dbl.count("sink_accretion_radius") > 0) {
                r_accretion = par_dbl.at("sink_accretion_radius");
                sink_particles_in_file = true;
            }
        }
        hdf5dims.resize(1); hdf5dims[0] = 1;
        HDFOutput.delete_dataset("r_accretion");
        HDFOutput.write(&r_accretion, "r_accretion", hdf5dims, H5T_NATIVE_DOUBLE);

        if (np > 0)
        {
            cout << " Number of particles in file: " << np << endl;
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
                cout<<" Found "<<sink_count<<" sink particle(s) and "<<np-sink_count<<" tracer particle(s)."<<endl;
            } else { // fill it up with either 0 (sinks) or 1 (tracers)
                type = new double[np];
                if (!sink_particles_in_file) for (long i=0; i<np; i++) type[i] = 1.0; // tracer particles
                else for (long i=0; i<np; i++) type[i] = 0.0; // sink particles
            }

            // write particle positions
            hdf5dims.resize(2); hdf5dims[0] = 3; hdf5dims[1] = np;
            double * ppos = new double[3*np];
            for (int dir = 0; dir < 3; dir++) for (long i = 0; i < np; i++)
            {
                long index = dir*np + i;
                if (dir == 0) ppos[index] = posx[i];
                if (dir == 1) ppos[index] = posy[i];
                if (dir == 2) ppos[index] = posz[i];
            }
            HDFOutput.delete_dataset("p_pos");
            HDFOutput.write(ppos, "p_pos", hdf5dims, H5T_NATIVE_DOUBLE);
            delete [] ppos;

            // write particle masses
            hdf5dims.resize(1); hdf5dims[0] = np;
            double * pmass = new double[np];
            double * ptag  = new double[np];
            double * ptype = new double[np];
            for (long i = 0; i < np; i++) {
                pmass[i] = mass[i];
                ptag [i] = tag [i];
                ptype[i] = type[i];
            }
            HDFOutput.delete_dataset("p_mass");
            HDFOutput.delete_dataset("p_tag");
            HDFOutput.delete_dataset("p_type");
            HDFOutput.write(pmass, "p_mass", hdf5dims, H5T_NATIVE_DOUBLE);
            HDFOutput.write(ptag,  "p_tag",  hdf5dims, H5T_NATIVE_DOUBLE);
            HDFOutput.write(ptype, "p_type", hdf5dims, H5T_NATIVE_DOUBLE);
            delete [] pmass;
            delete [] ptag;
            delete [] ptype;

            vector<double> particle_pos(3);
            vector<double> posx_extr;
            vector<double> posy_extr;
            vector<double> posz_extr;
            vector<double> mass_extr;
            vector<double> tag_extr;
            vector<double> type_extr;

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

                    // check extraction range
                    if ( (particle_pos[X] >= bounds[X][0]) &&
                         (particle_pos[X] <= bounds[X][1]) &&
                         (particle_pos[Y] >= bounds[Y][0]) &&
                         (particle_pos[Y] <= bounds[Y][1]) &&
                         (particle_pos[Z] >= bounds[Z][0]) &&
                         (particle_pos[Z] <= bounds[Z][1]) )
                    {
                        posx_extr.push_back(particle_pos[X]);
                        posy_extr.push_back(particle_pos[Y]);
                        posz_extr.push_back(particle_pos[Z]);
                        mass_extr.push_back(mass[i]);
                         tag_extr.push_back(tag[i]);
                        type_extr.push_back(type[i]);
                    }
                } // loop over replicas
            } // loop over particles

            // write out extracted particle positions and masses
            const long np_extr = posx_extr.size();
            hdf5dims.resize(1); hdf5dims[0] = 1;
            HDFOutput.delete_dataset("numpart_extr");
            HDFOutput.write(&np_extr, "numpart_extr", hdf5dims, H5T_NATIVE_INT);
            if (np_extr > 0)
            {
                double * ppos_extr = new double[3*np_extr];
                double * pmass_extr = new double[np_extr];
                double * ptag_extr = new double[np_extr];
                double * ptype_extr = new double[np_extr];
                for (long i = 0; i < np_extr; i++)
                {
                    ppos_extr[0*np_extr+i] = posx_extr[i];
                    ppos_extr[1*np_extr+i] = posy_extr[i];
                    ppos_extr[2*np_extr+i] = posz_extr[i];
                    pmass_extr[i]          = mass_extr[i];
                    ptag_extr [i]          = tag_extr [i];
                    ptype_extr[i]          = type_extr[i];
                }
                hdf5dims.resize(2); hdf5dims[0] = 3; hdf5dims[1] = np_extr;
                HDFOutput.delete_dataset("p_pos_extr");
                HDFOutput.write(ppos_extr, "p_pos_extr", hdf5dims, H5T_NATIVE_DOUBLE);
                hdf5dims.resize(1); hdf5dims[0] = np_extr;
                HDFOutput.delete_dataset("p_mass_extr");
                HDFOutput.write(pmass_extr, "p_mass_extr", hdf5dims, H5T_NATIVE_DOUBLE);
                hdf5dims.resize(1); hdf5dims[0] = np_extr;
                HDFOutput.delete_dataset("p_tag_extr");
                HDFOutput.write(ptag_extr, "p_tag_extr", hdf5dims, H5T_NATIVE_DOUBLE);
                hdf5dims.resize(1); hdf5dims[0] = np_extr;
                HDFOutput.delete_dataset("p_type_extr");
                HDFOutput.write(ptype_extr, "p_type_extr", hdf5dims, H5T_NATIVE_DOUBLE);
                delete [] ppos_extr;
                delete [] pmass_extr;
                delete [] ptag_extr;
                delete [] ptype_extr;
            }

            /// clean
            delete [] posx; delete [] posy; delete [] posz;
            delete [] mass; delete [] tag; delete [] type;

        } /// np > 0

        // close hdf5 file
        HDFOutput.close();
        cout<<">>> Meta data and particles data (if present) in '"<<hdf5filename<<"' written."<<endl;

    } // MyPE == 0

    /// print out wallclock time used
    long endtime = time(NULL);
    long duration = endtime-starttime; long duration_red = 0;
    if (Debug) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    duration = duration_red;
    if (MyPE==0) cout << "****************** Global time to finish = "<<duration<<"s ******************" << endl;   

    MPI_Finalize();
    return 0;

} // end main


/** ------------------------- ParseInputs ----------------------------
 **  Parses the command line Arguments
 ** ------------------------------------------------------------------ */
int ParseInputs(const vector<string> Argument)
{
    // check for valid options
    vector<string> valid_options;
    valid_options.push_back("-dsets");
    valid_options.push_back("-pixel");
    valid_options.push_back("-range");
    valid_options.push_back("-mws");
    valid_options.push_back("-div_factor");
    valid_options.push_back("-o");
    for (unsigned int i = 0; i < Argument.size(); i++) {
        if (Argument[i].at(0) == '-') {
            if (isdigit(Argument[i].at(1))) continue; // skip check if the '-' is part of a (negative) input value (number)
            bool valid_arg = false;
            for (unsigned int j = 0; j < valid_options.size(); j++) {
                if (Argument[i] == valid_options[j]) { valid_arg = true; break; }
            }
            if (!valid_arg) {
                if (MyPE==0) { cout << endl << "ParseInputs: '"<<Argument[i]<<"' is not a valid option." << endl; }
                return -1;
            }
        }
    }

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
        if (Argument[i] != "" && Argument[i] == "-dsets")
        {
            for (unsigned int j = i+1; j < Argument.size(); j++) {
                if (Argument[j].at(0) != '-') datasetnames.push_back(Argument[j]); else break;
            }
        }
        if (Argument[i] != "" && Argument[i] == "-pixel")
        {
            unsigned int nargs = 3;
            if (Argument.size()>i+nargs) {
                for (unsigned int j = i+1; j < i+nargs+1; j++) {
                    if (Argument[j].at(0) != '-') {
                        dummystream << Argument[j]; dummystream >> pixel[j-i-1]; dummystream.clear();
                    } else return -1;
                }
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-range")
        {
            unsigned int nargs = 6;
            if (Argument.size()>i+nargs) {
                for (unsigned int j = i+1; j < i+nargs+1; j++) {
                    if (Argument[j].at(0) != ' ') {
                        int dir = (j-i-1)/2;
                        int minmax = (j-i-1)%2;
                        dummystream << Argument[j]; dummystream >> bounds[dir][minmax]; dummystream.clear();
                    } else return -1;
                }
            } else return -1;
            if ( (bounds[X][1] < bounds[X][0]) || (bounds[Y][1] < bounds[Y][0]) || (bounds[Z][1] < bounds[Z][0]) ) {
                if (MyPE==0) cout<<endl<<"ParseInputs: something wrong with extraction range."<<endl;
                return -1;
            }
            user_bounds = true;
        }
        if (Argument[i] != "" && Argument[i] == "-mws")
        {
            for (unsigned int j = i+1; j < Argument.size(); j++) {
                if (Argument[j].at(0) != '-') {
                    int mw = 0;
                    dummystream << Argument[j]; dummystream >> mw; dummystream.clear();
                    mws.push_back(mw);
                } else break;
            }
            user_weights = true;
        }
        if (Argument[i] != "" && Argument[i] == "-div_factor")
        {
            unsigned int nargs = 3;
            if (Argument.size()>i+nargs) {
                for (unsigned int j = i+1; j < i+nargs+1; j++) {
                    if (Argument[j].at(0) != '-') {
                        dummystream << Argument[j]; dummystream >> div_factor[j-i-1]; dummystream.clear();
                    } else return -1;
                }
            } else return -1;
            user_div_factor = true;
        }
        if (Argument[i] != "" && Argument[i] == "-o")
        {
            if (Argument.size()>i+1) output_file_name = Argument[i+1]; else return -1;
            use_output_file = true;
        }

    } // loop over all args

    // error checking
    if (datasetnames.size() == 0) {
        if (MyPE==0) cout<<endl<<"ParseInputs: need to specify at least one dataset to extract (e.g., -dsets dens)."<<endl;
        return -1;
    }
    if (user_weights) {
        if (mws.size() != datasetnames.size()) {
            if (MyPE==0) cout<<endl<<"ParseInputs: number of <mws> must be the same as the number of <dsets>."<<endl;
            return -1;
        }
    }

    /// print out parsed values
    if (MyPE==0) {
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
        << " extractor <filename> [<OPTIONS>]" << endl << endl
        << "   <OPTIONS>:           " << endl
        << "     -dsets <dset(s)>                       : datasetname(s) to be processed (e.g., dens velx vely velz)" << endl
        << "     -pixel <Nx Ny Nz>                      : pixels for extraction grid (default 128 128 128)" << endl
        << "     -range <xmin xmax ymin ymax zmin zmax> : extraction range in sim coordinates (default ALL)" << endl
        << "     -mws <mw(s)>                           : mass weighting flags (must be list of 0 or 1 with same length as dsets); if not provided, only vel? are mass-weighted" << endl
        << "     -div_factor <fx fy fz>                 : divide the extraction domain in fx, fy, fz parts for memory-effective processing (default: automatic)" << endl
        << "     -o <filename>                          : specify output filename (use this to add different datasets to the same file)" << endl
        << endl
        << "Example: extractor DF_hdf5_plt_cnt_0020 -dsets dens velx vely velz -pixel 128 128 128 -range 0 1 0 1 0 1 -o DF_hdf5_plt_cnt_0020_ext"
        << endl << endl;
    }
}
