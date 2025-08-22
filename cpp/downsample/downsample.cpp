/*

  downsample.cpp

  Takes a FLASH (uniform grid) input file and samples down the cells in each
  block by a given factor for a given grid variable and writes the result to file

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
#include "mpi.h" // MPI lib
#include "../Libs/HDFIO.h" // HDF5 IO
#include "../Libs/FlashGG.h" // Flash General Grid class

// constants
#define NDIM 3
using namespace std;
enum {X, Y, Z};
static const bool Debug = false;

// MPI stuff
int MyPE = 0, NPE = 1;

// some global stuff (inputs)
string inputfile = "";
int downsample_factor = 0;
vector<string> datasetnames(0);
int ncells_pseudo_blocks = 0;
string output_file_name = "";
bool use_output_file = false;

// forward functions
float * DownSample(const float * const var_in, const vector<int> dim_in,
                   const float * const dens_in, const bool massweighting, const int factor);
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
    if (MyPE==0) cout<<"=== downsample === using MPI num procs: "<<NPE<<endl;

    // Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) == -1)
    {
        if (MyPE==0) cout<<endl<<"Error in ParseInputs(). Exiting."<<endl;
        HelpMe();
        MPI_Finalize(); return 0;
    }

    long starttime = time(NULL);

    // open FLASH input file
    FlashGG gg = FlashGG(inputfile);
    char GridType = gg.GetGridType();
    if (GridType == 'A') {
        if (MyPE==0) cout<<endl<<"Error: only uniform or extracted grid input files are supported here."<<endl;
        MPI_Finalize(); return 0;
    }
    if (MyPE==0) gg.PrintInfo();

    // setup pseudo blocks
    if (MyPE == 0) {
        // print options for pseudo block cell options
        vector< vector<int> > NB_PB_options = gg.GetNumCellsInBlock_PB_options();
        cout<<"Possible values for number of cells in pseudo blocks given a downsampling factor of "
            <<downsample_factor<<":"<<endl;
        for (int dir = X; dir <= Z; dir++) {
            cout<<" in direction ["<<dir<<"] = ";
            for (unsigned int i = 0; i < NB_PB_options[dir].size(); i++) {
                if (NB_PB_options[dir][i] % downsample_factor == 0) {
                    cout<<NB_PB_options[dir][i]<<" ";
                }
            }
            cout<<endl;
        }
    }
    // set NB_PB
    vector<int> NB_PB(3);
    NB_PB[X] = ncells_pseudo_blocks;
    NB_PB[Y] = ncells_pseudo_blocks;
    NB_PB[Z] = ncells_pseudo_blocks;
    // error checking
    for (int dir = X; dir <= Z; dir++) {
        if (NB_PB[dir] % downsample_factor != 0) {
            if (MyPE==0) cout<<"Error: NB_PB["<<dir<<"] = "<<NB_PB[dir]
                            <<" is not divisible by downsample_factor = "<<downsample_factor<<"!"<<endl;
            MPI_Finalize(); return 0;
        }
    }
    // number of cells in downsampled blocks
    vector<int> NB_ds(3);
    NB_ds[X] = NB_PB[X] / downsample_factor;
    NB_ds[Y] = NB_PB[Y] / downsample_factor;
    NB_ds[Z] = NB_PB[Z] / downsample_factor;
    if (MyPE == 0) {
        cout<<"Using pseudo blocks with "<<NB_PB[X]<<" "<<NB_PB[Y]<<" "<<NB_PB[Z]<<" cells for reading."<<endl;
        cout<<"Output will have "<<NB_ds[X]<<" "<<NB_ds[Y]<<" "<<NB_ds[Z]<<" cells per block."<<endl;
    }
    gg.SetupPseudoBlocks(NB_PB);

    /// read FLASH file meta data after setting up pseudo blocks
    int NBLK = gg.GetNumBlocks_PB();
    vector< vector <vector<double> > > BB_PB = gg.GetBoundingBox_PB();
    vector<int> NumBlocksIn = gg.GetNumBlocksVector_PB();

    // check that num procs is <= total number of blocks
    if ((NPE < 2) || (NPE > NBLK)) {
        if (MyPE==0) cout<<"ERROR: Number of processors must be >= 2 and <= number of blocks."<<endl;
        MPI_Finalize(); return 0;
    }

    // create/overwrite outputfile
    if (!use_output_file) output_file_name = inputfile+"_downsampled";
    HDFIO HDFOutput = HDFIO();
    if (MyPE==0) cout<<"Creating output file '"<<output_file_name<<"'..."<<endl;
    HDFOutput.create(output_file_name, MPI_COMM_WORLD);
    vector<int> HDFDims;
    // write bounding box
    HDFDims.resize(3); HDFDims[0] = NBLK; HDFDims[1] = 3; HDFDims[2] = 2;
    float * float_data = new float[NBLK*3*2];
    for (int b=0; b<NBLK; b++) for (int dir=X; dir<=Z; dir++) {
        float_data[2*3*b+2*dir+0] = BB_PB[b][dir][0];
        float_data[2*3*b+2*dir+1] = BB_PB[b][dir][1];
    }
    HDFOutput.write(float_data, "bounding box", HDFDims, H5T_NATIVE_FLOAT, MPI_COMM_WORLD);
    if (MyPE==0) cout<<"'bounding box' written to '"<<output_file_name<<"'."<<endl;
    delete [] float_data;
    // write block size and coordinates
    HDFDims.resize(2); HDFDims[0] = NBLK; HDFDims[1] = 3;
    float_data = new float[NBLK*3];
    for (int b=0; b<NBLK; b++) for (int dir=X; dir<=Z; dir++) float_data[3*b+dir] = BB_PB[b][dir][1] - BB_PB[b][dir][0];
    HDFOutput.write(float_data, "block size", HDFDims, H5T_NATIVE_FLOAT, MPI_COMM_WORLD);
    if (MyPE==0) cout<<"'block size' written to '"<<output_file_name<<"'."<<endl;
    for (int b=0; b<NBLK; b++) for (int dir=X; dir<=Z; dir++)
        float_data[3*b+dir] = BB_PB[b][dir][0] + 0.5 * (BB_PB[b][dir][1] - BB_PB[b][dir][0]);
    HDFOutput.write(float_data, "coordinates", HDFDims, H5T_NATIVE_FLOAT, MPI_COMM_WORLD);
    if (MyPE==0) cout<<"'coordinates' written to '"<<output_file_name<<"'."<<endl;
    delete [] float_data;
    // write node type, refine level, processor number
    HDFDims.resize(1); HDFDims[0] = NBLK;
    int * int_data = new int[NBLK];
    for (int b=0; b<NBLK; b++) int_data[b] = 1;
    HDFOutput.write(int_data, "node type", HDFDims, H5T_NATIVE_INT, MPI_COMM_WORLD);
    if (MyPE==0) cout<<"'node type' written to '"<<output_file_name<<"'."<<endl;
    HDFOutput.write(int_data, "refine level", HDFDims, H5T_NATIVE_INT, MPI_COMM_WORLD);
    if (MyPE==0) cout<<"'refine level' written to '"<<output_file_name<<"'."<<endl;
    for (int b=0; b<NBLK; b++) int_data[b] = b;
    HDFOutput.write(int_data, "processor number", HDFDims, H5T_NATIVE_INT, MPI_COMM_WORLD);
    if (MyPE==0) cout<<"'processor number' written to '"<<output_file_name<<"'."<<endl;
    delete [] int_data;
    // write gid
    HDFDims.resize(2); HDFDims[0] = NBLK; HDFDims[1] = 15;
    int_data = new int[NBLK*15];
    for (int b=0; b<NBLK; b++) {
        int_data[15*b+0] = b - 1;
        int_data[15*b+1] = b + 1;
        int_data[15*b+2] = b + NumBlocksIn[X];
        int_data[15*b+3] = b - NumBlocksIn[X];
        int_data[15*b+4] = b + NumBlocksIn[X]*NumBlocksIn[Y];
        int_data[15*b+5] = b - NumBlocksIn[X]*NumBlocksIn[Y];
        if ( b    % NumBlocksIn[X] == 0) int_data[15*b+0] = -21;
        if ((b+1) % NumBlocksIn[X] == 0) int_data[15*b+1] = -21;
        if (int_data[15*b+2] >= NumBlocksIn[X]*NumBlocksIn[Y]*NumBlocksIn[Z]) int_data[15*b+2] = -21;
        if (int_data[15*b+3] < 0) int_data[15*b+3] = -21;
        if (int_data[15*b+4] >= NumBlocksIn[X]*NumBlocksIn[Y]*NumBlocksIn[Z]) int_data[15*b+4] = -21;
        if (int_data[15*b+5] < 0) int_data[15*b+5] = -21;
        // parents and children are non-existent in UG
        for (int n = 6; n < 15; n++) int_data[15*b+n] = -21;
    }
    HDFOutput.write(int_data, "gid", HDFDims, H5T_NATIVE_INT, MPI_COMM_WORLD);
    if (MyPE==0) cout<<"'gid' written to '"<<output_file_name<<"'."<<endl;
    delete [] int_data;
    // create empty output dataset(s)
    HDFDims.resize(4);
    HDFDims[0] = NBLK; HDFDims[1] = NB_ds[Z]; HDFDims[2] = NB_ds[Y]; HDFDims[3] = NB_ds[X];
    for (unsigned int ds=0; ds<datasetnames.size(); ds++) {
        HDFOutput.create_dataset(datasetnames[ds], HDFDims, H5T_NATIVE_FLOAT, MPI_COMM_WORLD);
        if (MyPE==0) cout<<"'"<<datasetnames[ds]<<"' prepared in '"<<output_file_name<<"'."<<endl;
    }
    HDFOutput.close();
    MPI_Barrier(MPI_COMM_WORLD);

    // copy meta data; only the master PE does that
    if (MyPE == 0) {
        // now copy meta data using HDF5 command line tool 'h5copy'
        string cp_dataset_list[10] = {"integer runtime parameters", "integer scalars",
                                      "real runtime parameters", "real scalars",
                                      "logical runtime parameters", "logical scalars",
                                      "string runtime parameters", "string scalars", "sim info", "unknown names"};
        for (unsigned int i=0; i<10; i++)
        {
            string cmd = "h5copy -i "+inputfile+" -o "+output_file_name+" -s '"+cp_dataset_list[i]+"' -d '"+cp_dataset_list[i]+"'";
            cout<<"Calling system command: "<<cmd<<endl;
            system(cmd.c_str());
        }
        // reopen to overwrite meta data
        HDFOutput.open(output_file_name, 'w');
        map<string, int> int_props = HDFOutput.ReadFlashIntegerScalars();
        int_props["iprocs"] = NumBlocksIn[X];
        int_props["jprocs"] = NumBlocksIn[Y];
        int_props["kprocs"] = NumBlocksIn[Z];
        int_props["nxb"] = NB_ds[X];
        int_props["nyb"] = NB_ds[Y];
        int_props["nzb"] = NB_ds[Z];
        int_props["globalnumblocks"] = NumBlocksIn[X]*NumBlocksIn[Y]*NumBlocksIn[Z];
        int_props["splitnumblocks"] = NumBlocksIn[X]*NumBlocksIn[Y]*NumBlocksIn[Z];
        HDFOutput.OverwriteFlashIntegerScalars(int_props);
        int_props = HDFOutput.ReadFlashIntegerParameters();
        int_props["iprocs"] = NumBlocksIn[X];
        int_props["jprocs"] = NumBlocksIn[Y];
        int_props["kprocs"] = NumBlocksIn[Z];
        HDFOutput.OverwriteFlashIntegerParameters(int_props);
        HDFOutput.close();
    } // MyPE == 0

    // open the output file with FlashGG to write the downsampled dataset blocks in loop below
    FlashGG gg_out = FlashGG(output_file_name, 'w');
    vector<int> N_out = gg_out.GetN();
    if (MyPE==0) cout<<">>> Output grid size (downsampled): "<<N_out[X]<<" x "<<N_out[Y]<<" x "<<N_out[Z]<<endl;

    /// overwrite "unknown names" with requested downsampled datasets
    vector<string> unknown_names;
    for (unsigned int ds = 0; ds < datasetnames.size(); ds++) unknown_names.push_back(datasetnames[ds]);
    gg_out.OverwriteUnknownNames(unknown_names);
    if (MyPE==0) cout<<"'unknown names' over-written in '"<<output_file_name<<"'."<<endl;

    // domain decomposition
    vector<int> MyBlocks = gg.GetMyBlocks();

    // loop over all datasets and (over)write actual data
    for (unsigned int ds = 0; ds < datasetnames.size(); ds++) {

        // velocity and velocity-derived datasets should be mass weighted when downsampling
        bool density_weight = false;
        if ( datasetnames[ds] == "velx" || datasetnames[ds] == "vely" || datasetnames[ds] == "velz" ||
             datasetnames[ds] == "dvvl" || datasetnames[ds] == "mvrt" || datasetnames[ds] == "divv" ) density_weight = true;

        // loop over all blocks and fill containers
        for (unsigned int ib = 0; ib < MyBlocks.size(); ib++)
        {
            int b = MyBlocks[ib]; // actual block index
            // read block data
            float *block_data = gg.ReadBlockVarPB(b, datasetnames[ds]);
            float *block_dens = 0;
            if (density_weight) block_dens = gg.ReadBlockVarPB(b, "dens");
            // downsample this block
            float * block_out = DownSample(block_data, NB_PB, block_dens, density_weight, downsample_factor);
            // write downsampled block to file; overwrite (either created above or was already present in file)
            gg_out.OverwriteBlockVar(b, datasetnames[ds], block_out);
            // clean arrays
            delete [] block_out;
            delete [] block_data; if (density_weight) delete [] block_dens;
            // print progress info
            if (MyPE==0) cout<<" ["<<setw(6)<<MyPE<<"] Block "<<setw(6)<<ib+1<<" of "<<setw(6)<<MyBlocks.size()
                             <<" (on root proc) downsampled for dataset '"<<datasetnames[ds]<<"'"<<endl;
        } //end loop over blocks

    } // end loop over variables from command input line

    MPI_Barrier(MPI_COMM_WORLD);
    if (MyPE==0) cout<<">>> Finished writing '"<<output_file_name<<"'."<<endl;

    // print out wallclock time used
    long endtime = time(NULL);
    long duration = endtime-starttime, duration_red = 0;
    if (Debug) cout<<"["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************"<<endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout<<"****************** Global time to finish = "<<duration_red<<"s ******************"<<endl;

    // done
    MPI_Finalize(); return 0;

} // end main


/** ------------------------- DownSample -----------------------------
 **  takes a cube and samples it down by a given factor with option for mass-weighting
 ** ------------------------------------------------------------------ */
float * DownSample(const float * const var_in, const vector<int> dim_in,
                    const float * const dens_in, const bool massweighting, const int factor)
{
    if (Debug) cout << "Downsample: entering" << endl;

    int grid_dimension_x_out = dim_in[X] / factor;
    int grid_dimension_y_out = dim_in[Y] / factor;
    int grid_dimension_z_out = dim_in[Z] / factor;

    long gridcellindex_in = 0, gridcellindex_out = 0;
    long grid_size_out = grid_dimension_x_out * grid_dimension_y_out * grid_dimension_z_out;

    int dim_in_XY = dim_in[X] * dim_in[Y];
    int dim_out_XY = grid_dimension_x_out * grid_dimension_y_out;

    // initialize output and density weighting array
    float * var_out = new float[grid_size_out];
    float * dens_out = new float[grid_size_out];
    for (long n = 0; n < grid_size_out; n++) {
        var_out[n] = 0.0; dens_out[n] = 0.0;
    }

    if (Debug) cout << "Downsample: GridDimensions_in : " << dim_in[X]
                    << " " << dim_in[Y] << " " << dim_in[Z] << endl;
    if (Debug) cout << "Downsample: GridDimensions_out: " << grid_dimension_x_out
                    << " " << grid_dimension_y_out << " " << grid_dimension_z_out << endl;

    if (Debug) cout << "Downsample: outer loop k = ";
    for (int k = 0; k < dim_in[Z]; k++) { if (Debug) cout << k << " ";
      for (int j = 0; j < dim_in[Y]; j++) {
        for (int i = 0; i < dim_in[X]; i++) {

          gridcellindex_in =  k * dim_in_XY + j * dim_in[X] + i;

          if ((k % factor == 0) && (j % factor == 0) && (i % factor == 0))
            gridcellindex_out = k/factor * dim_out_XY + j/factor * grid_dimension_x_out + i/factor;

          else if ((k % factor != 0) && (j % factor == 0) && (i % factor == 0))
            gridcellindex_out = (k-1)/factor * dim_out_XY + j/factor * grid_dimension_x_out + i/factor;

          else if ((k % factor == 0) && (j % factor != 0) && (i % factor == 0))
            gridcellindex_out = k/factor * dim_out_XY + (j-1)/factor * grid_dimension_x_out + i/factor;

          else if ((k % factor == 0) && (j % factor == 0) && (i % factor != 0))
            gridcellindex_out = k/factor * dim_out_XY + j/factor * grid_dimension_x_out + (i-1)/factor;

          else if ((k % factor != 0) && (j % factor != 0) && (i % factor == 0))
            gridcellindex_out = (k-1)/factor * dim_out_XY + (j-1)/factor * grid_dimension_x_out + i/factor;

          else if ((k % factor != 0) && (j % factor == 0) && (i % factor != 0))
            gridcellindex_out = (k-1)/factor * dim_out_XY + j/factor * grid_dimension_x_out + (i-1)/factor;

          else if ((k % factor == 0) && (j % factor != 0) && (i % factor != 0))
            gridcellindex_out = k/factor * dim_out_XY + (j-1)/factor * grid_dimension_x_out + (i-1)/factor;

          else if ((k % factor != 0) && (j % factor != 0) && (i % factor != 0))
            gridcellindex_out = (k-1)/factor * dim_out_XY + (j-1)/factor * grid_dimension_x_out + (i-1)/factor;

          double dens_weight = 1.0;
          if (massweighting) dens_weight = dens_in[gridcellindex_in];

          //accumulate
          var_out[gridcellindex_out] = (double)(var_out[gridcellindex_out]) + (double)(var_in[gridcellindex_in])*dens_weight;
          dens_out[gridcellindex_out] = (double)(dens_out[gridcellindex_out]) + dens_weight;

        } // for i
      } // for j
    } // for k

    // calculate the average values (works for volume and mass-weighting)
    for (long n = 0; n < grid_size_out; n++) var_out[n] = (double)(var_out[n]) / (double)(dens_out[n]);

    delete [] dens_out;

    if (Debug) cout << "Downsample: exiting" << endl;

    return var_out;
}


/** ------------------------- ParseInputs ----------------------------
 **  Parses the command line Arguments
 ** ------------------------------------------------------------------ */
int ParseInputs(const vector<string> Argument)
{
    // check for valid options
    vector<string> valid_options;
    valid_options.push_back("-dsets");
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
    if (Argument.size() < 4)
    {
        if (MyPE==0) { cout << endl << "ParseInputs: Invalid number of arguments." << endl; }
        return -1;
    }
    inputfile = Argument[1];

    /// downsampling factor
    dummystream << Argument[2]; dummystream >> downsample_factor; dummystream.clear();

    /// ncells for pseudo blocks
    dummystream << Argument[3]; dummystream >> ncells_pseudo_blocks; dummystream.clear();

    for (unsigned int i = 4; i < Argument.size(); i++)
    {
        if (Argument[i] != "" && Argument[i] == "-dsets")
        {
            for (unsigned int j = i+1; j < Argument.size(); j++) {
                if (Argument[j].at(0) != '-') datasetnames.push_back(Argument[j]); else break;
            }
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

    /// print out parsed values
    if (MyPE==0) {
        cout << "ParseInputs: Command line arguments: ";
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
        << " downsample <filename> <factor> <ncells_pb> [OPTIONS]" << endl
        << "   <filename>              : FLASH input filename" << endl
        << "   <factor>                : downsampling factor (integer; assumes cubic pseudo blocks)" << endl
        << "   <ncells_pb>             : number of cells in pseudo blocks (integer; must be divisible by <factor>)" << endl
        << " <OPTIONS>:" << endl
        << "   -dsets <dset(s)>        : datasetname(s) to be processed (e.g., dens velx vely velz)" << endl
        << "                             (note that downsampling of vel? and divv is done with a mass-weighted average; otherwise: volume-weighted)" << endl
        << "   -o <filename>           : specify output filename" << endl
        << endl
        << "Example: downsample DF_hdf5_plt_cnt_0050 2 16 -dsets dens velx vely velz"
        << endl << endl;
    }
}
