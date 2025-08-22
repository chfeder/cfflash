/*

  merge_split_flash_files.cpp

  Merge split FLASH plot files (HDF5)

  By Christoph Federrath, 2013-2025

*/

#include "mpi.h" /// MPI lib
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <hdf5.h>
#include "../Libs/HDFIO.h" // HDF5 IO
#include "../Libs/FlashGG.h" /// Flash Grid class
#include "../Libs/CFTools.h"

// constants
#define NDIM 3
using namespace std;
enum {X, Y, Z};

// MPI stuff
int MyPE = 0, NPE = 1;

// some global stuff (inputs)
string inputfile;
string outpath = "";
int splitnum = 0;
vector<string> DatasetNames(0);
int Verbose = 1;

// forward functions
int ParseInputs(const vector<string> Argument);
void HelpMe(void);
int CompareArrays(const   int * const a, const   int * const b, const long int n);
int CompareArrays(const float * const a, const float * const b, const long int n);

/// --------
///   MAIN
/// --------
int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NPE);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);

    long starttime = time(NULL);

    /// Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) == -1)
    {
        if (MyPE==0) cout << endl << "Error in ParseInputs(). Exiting." << endl;
        HelpMe();
        MPI_Finalize(); return 0;
    }

    if (Verbose && MyPE==0) cout<<"=== merge_split_flash_files === using MPI num procs: "<<NPE<<endl;

    /// determine dump number, i/o filename(s), plt or chk file type, etc.
    string dump_str = inputfile.substr(inputfile.size()-4, 4);
    vector<string> plt_or_chk; plt_or_chk.push_back("_hdf5_plt_cnt_"); plt_or_chk.push_back("_hdf5_chk_");
    string plt_or_chk_str = "";
    string flashbasename = "";
    for (unsigned int i = 0; i < plt_or_chk.size(); i++) {
        size_t pos = inputfile.find(plt_or_chk[i]);
        if (pos != string::npos) {
            plt_or_chk_str = plt_or_chk[i];
            flashbasename = (pos != string::npos) ? inputfile.substr(0, pos-6) : inputfile;
        }
    }
    string outputfile = outpath+flashbasename+plt_or_chk_str+dump_str;
    if (MyPE==0 && Verbose>1) {
        cout << "inputfile = " << inputfile << endl;
        cout << "plt_or_chk_str = " << plt_or_chk_str << endl;
        cout << "outputfile = " << outputfile << endl;
        cout << "flashbasename = " << flashbasename << endl;
        cout << "outpath = " << outpath << endl;
        cout << "dump_str = " << dump_str << endl;
        cout << "splitnum = " << splitnum << endl;
    }

    /// now copy FLASH meta data from the s0000 file to the outputfile; the unknown vars follow below
    /// only the Master PE does that
    if (MyPE==0) {
        // see if input file exisits; if so, delete any potentially existing output file, e.g.,
        // because it might have been created in an earlier merging attempt, but wasn't finished writing
        ifstream ifs_infile(inputfile.c_str());
        if (ifs_infile.good()) {
            ifstream ifs_outfile(outputfile.c_str());
            if (ifs_outfile.good()) {
                if (Verbose) cout<<"!!! Output file "<<outputfile<<" exists; removing it..."<<endl;
                string cmd = "rm "+outputfile;
                if (Verbose) cout<<"calling system command: "<<cmd<<endl;
                system(cmd.c_str());
            }
            ifs_outfile.close();
        } else {
            cout<<"!!! Input file "<<inputfile<<" does not exist. Exiting."<<endl;
            return 0;
        }
        ifs_infile.close();

        // now copy meta data using HDF5 command line tool 'h5copy'
        string cp_dataset_list[17] = {"block size", "bounding box", "coordinates", "gid", "integer runtime parameters",
                                     "integer scalars", "logical runtime parameters", "logical scalars", "node type",
                                     "processor number", "real runtime parameters", "real scalars", "refine level",
                                     "sim info", "string runtime parameters", "string scalars", "unknown names"};
        for (int i=0; i<17; i++)
        {
            string cmd = "h5copy -i "+inputfile+" -o "+outputfile+" -s '"+cp_dataset_list[i]+"' -d '"+cp_dataset_list[i]+"'";
            if (Verbose) cout<<"calling system command: "<<cmd<<endl;
            system(cmd.c_str());
        }
        if (Verbose) cout<<"Copying of meta data done."<<endl;
    }

    /// wait until everybody is here
    MPI_Barrier(MPI_COMM_WORLD);

    /// Here we create the big target datasets with zeros (in parallel).
    /// First get the dimensions of the datasets (NBLK and NB[X,Y,Z]).
    /// Then create the big target datasets (just filled with zeros) in the output file.
    FlashGG gg_inp = FlashGG(inputfile, Verbose);
    int NBLK = gg_inp.GetNumBlocks();
    vector<int> NB = gg_inp.GetNumCellsInBlock();
    if (Verbose && MyPE==0) gg_inp.PrintInfo();
    vector<int> dims(4);
    dims[0] = NBLK;
    dims[1] = NB[Z];
    dims[2] = NB[Y];
    dims[3] = NB[X];
    /// loop over all the block variables in the plot file
    HDFIO hdfio_out = HDFIO(outputfile, 'w');
    for (unsigned int ds=0; ds<DatasetNames.size(); ds++)
        hdfio_out.create_dataset(DatasetNames[ds], dims, H5T_NATIVE_FLOAT, MPI_COMM_WORLD);
    hdfio_out.close();
    if (Verbose && MyPE==0) cout << "Output datasets created..." << endl;

    /// decompose domain in blocks
    vector<int> MyBlocks = gg_inp.GetMyBlocks();

    /// for each block, find the file that contains this block
    vector<string> BlockFile(MyBlocks.size());
    for (int f=0; f<splitnum; f++) {
        if (MyPE==0 && Verbose>1) cout << "f counter = " << f << endl;
        char split_str[80]; snprintf(split_str, sizeof(split_str), "%4.4d", f);
        if (MyPE==0 && Verbose>1) cout << "split_str = " << split_str << endl;
        string filename = flashbasename+"_s"+split_str+plt_or_chk_str+dump_str;
        if (MyPE==0 && Verbose>1) cout << "filename = " << filename << endl;
        FlashGG gg_in = FlashGG(filename, Verbose);
        vector<int> NodeType = gg_in.GetNodeType();
        for (unsigned int ib=0; ib<MyBlocks.size(); ib++) {
            int b = MyBlocks[ib];
            if (NodeType[b] == 1) BlockFile[ib] = filename;
        }
    }
    if (Verbose>1) {
        for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
            cout<<" ["<<MyPE<<"] MyBlock="<<MyBlocks[ib]<<" BlockFile="<<BlockFile[ib]<<endl;
    }

    if (Verbose && MyPE==0) cout << "Entering merging loop now..." << endl;

    /// open the FLASH output file in order to merge everything into it
    FlashGG gg_out = FlashGG(outputfile, 'w', Verbose);

    /// loop over all of my blocks, open the right files and overwrite datasets
    CFTools cft = CFTools(); // initialise progress bar
    for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
    {
        /// write progress
        if (MyPE==0) cft.PrintProgressBar(ib, MyBlocks.size());

        /// get actual block index
        int b = MyBlocks[ib];
        string splitfile = BlockFile[ib];
        FlashGG gg_in = FlashGG(splitfile, Verbose);

        if (Verbose>1) cout<<" ["<<MyPE<<"] working on block "<< b << "; my file: " << splitfile << endl;

        long int array_size = 0;

        /// read block size from split file and overwrite in output file
        float *block_size = gg_in.ReadBlockSize(b, array_size);
        gg_out.OverwriteBlockSize(b, block_size);
        delete [] block_size;

        /// read block bounding box from split file and overwrite in output file
        float *block_bb = gg_in.ReadBoundingBox(b, array_size);
        gg_out.OverwriteBoundingBox(b, block_bb);
        delete [] block_bb;

        /// read block coordinates from split file and overwrite in output file
        float *block_coords = gg_in.ReadCoordinates(b, array_size);
        gg_out.OverwriteCoordinates(b, block_coords);
        delete [] block_coords;

        /// read block GID from split file and overwrite in output file
        int *block_gid = gg_in.ReadGID(b, array_size);
        gg_out.OverwriteGID(b, block_gid);
        delete [] block_gid;

        /// read block node type from split file and overwrite in output file
        int *block_nodetype = gg_in.ReadNodeType(b, array_size);
        gg_out.OverwriteNodeType(b, block_nodetype);
        delete [] block_nodetype;

        /// read block proc number from split file and overwrite in output file
        int *block_procnum = gg_in.ReadProcessorNumber(b, array_size);
        gg_out.OverwriteProcessorNumber(b, block_procnum);
        delete [] block_procnum;

        /// read block refine level from split file and overwrite in output file
        int *block_reflev = gg_in.ReadRefineLevel(b, array_size);
        gg_out.OverwriteRefineLevel(b, block_reflev);
        delete [] block_reflev;

        /// loop over all the block variables in the plot file
        for (unsigned int ds=0; ds<DatasetNames.size(); ds++) {
            /// read block data from split file and overwrite in output file
            float *block_data = gg_in.ReadBlockVar(b, DatasetNames[ds]);
            gg_out.OverwriteBlockVar(b, DatasetNames[ds], block_data);
            delete [] block_data;
        }

    } //end loop over my blocks

    MPI_Barrier(MPI_COMM_WORLD);
    if (Verbose && MyPE==0) cout << "Starting verification now..." << endl;

    // loop again and verify written data
    cft.InitProgressBar();
    for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
    {
        /// write progress
        if (MyPE==0) cft.PrintProgressBar(ib, MyBlocks.size());

        /// get actual block index
        int b = MyBlocks[ib];
        string splitfile = BlockFile[ib];
        FlashGG gg_in = FlashGG(splitfile, Verbose);

        float *in_d = 0, *out_d = 0;
        int    *in_i = 0, *out_i = 0;
        long int array_size = 0;

        /// read block size from split file and check in output file
        in_d = gg_in.ReadBlockSize(b, array_size);
        out_d = gg_out.ReadBlockSize(b, array_size);
        if (CompareArrays(in_d,out_d,array_size) != 0) cout<<"ERROR in block size file: "<<splitfile<<" block: "<<b<<endl;
        delete [] in_d;
        delete [] out_d;

        /// read block bounding box from split file and check in output file
        in_d = gg_in.ReadBoundingBox(b, array_size);
        out_d = gg_out.ReadBoundingBox(b, array_size);
        if (CompareArrays(in_d,out_d,array_size) != 0) cout<<"ERROR in bounding box. file: "<<splitfile<<" block: "<<b<<endl;
        delete [] in_d;
        delete [] out_d;

        /// read block coordinates from split file and check in output file
        in_d = gg_in.ReadCoordinates(b, array_size);
        out_d = gg_out.ReadCoordinates(b, array_size);
        if (CompareArrays(in_d,out_d,array_size) != 0) cout<<"ERROR in coordinates. file: "<<splitfile<<" block: "<<b<<endl;
        delete [] in_d;
        delete [] out_d;

        /// read block gid from split file and check in output file
        in_i = gg_in.ReadGID(b, array_size);
        out_i = gg_out.ReadGID(b, array_size);
        if (CompareArrays(in_i,out_i,array_size) != 0) cout<<"ERROR in gid file: "<<splitfile<<" block: "<<b<<endl;
        delete [] in_i;
        delete [] out_i;

        /// read block node type from split file and check in output file
        in_i = gg_in.ReadNodeType(b, array_size);
        out_i = gg_out.ReadNodeType(b, array_size);
        if (CompareArrays(in_i,out_i,array_size) != 0) cout<<"ERROR in node type file: "<<splitfile<<" block: "<<b<<endl;
        delete [] in_i;
        delete [] out_i;

        /// read block proc num from split file and check in output file
        in_i = gg_in.ReadProcessorNumber(b, array_size);
        out_i = gg_out.ReadProcessorNumber(b, array_size);
        if (CompareArrays(in_i,out_i,array_size) != 0) cout<<"ERROR in processor number file: "<<splitfile<<" block: "<<b<<endl;
        delete [] in_i;
        delete [] out_i;

        /// read block refine level from split file and check in output file
        in_i = gg_in.ReadRefineLevel(b, array_size);
        out_i = gg_out.ReadRefineLevel(b, array_size);
        if (CompareArrays(in_i,out_i,array_size) != 0) cout<<"ERROR in refine level file: "<<splitfile<<" block: "<<b<<endl;
        delete [] in_i;
        delete [] out_i;

        /// loop over all the block variables in the plot file
        for (unsigned int ds=0; ds<DatasetNames.size(); ds++) {
            /// read block data from split file and check in output file
            in_d = gg_in.ReadBlockVar(b, DatasetNames[ds], array_size);
            out_d = gg_out.ReadBlockVar(b, DatasetNames[ds], array_size);
            if (CompareArrays(in_d,out_d,array_size) != 0) cout<<"ERROR in "<<DatasetNames[ds]<<" file: "<<splitfile<<" block: "<<b<<endl;
            delete [] in_d;
            delete [] out_d;
        }

    } //end loop over my blocks

    /// print info to shell
    if (Verbose && MyPE==0) {
        cout<<"Standard FLASH arrays"<<endl;
        cout<<"block size, bounding box, coordinates, gid, node type, processor number, refine level"<<endl;
        for (unsigned int ds=0; ds<DatasetNames.size(); ds++)
            cout<<"and plot variable '"+DatasetNames[ds]+"'"<<endl;
        cout<<"in merged file '"+outputfile+"' written."<<endl;
    }

    /// print out wallclock time used
    long endtime = time(NULL);
    long duration = endtime-starttime, duration_red = 0;
    if (Verbose>1) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (Verbose && MyPE==0) cout << "****************** Global time to finish = "<<duration_red<<"s ******************" << endl;

    MPI_Finalize();
    return 0;

} // end main


int CompareArrays(const int * const a, const int * const b, const long int n)
{
    bool diff_found = false;
    for (int i=0; i<n; i++)
        if (a[i]!=b[i]) {
            cout<<"CompareArrays: diff found: i="<<i<<": "<<a[i]<<" vs. "<<b[i]<<endl;
            diff_found = true;
        }
    if (diff_found) return -1; else return 0;
}

int CompareArrays(const float * const a, const float * const b, const long int n)
{
    bool diff_found = false;
    for (int i=0; i<n; i++)
        if (a[i]!=b[i]) {
            cout<<"CompareArrays: diff found: i="<<i<<": "<<a[i]<<" vs. "<<b[i]<<endl;
            diff_found = true;
        }
    if (diff_found) return -1; else return 0;
}


/** ------------------------- ParseInputs ----------------------------
 **  Parses the command line Arguments
 ** ------------------------------------------------------------------ */
int ParseInputs(const vector<string> Argument)
{
    // check for valid options
    vector<string> valid_options;
    valid_options.push_back("-sn");
    valid_options.push_back("-dsets");
    valid_options.push_back("-o");
    valid_options.push_back("-verbose");
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

    inputfile = Argument[1];

    for (unsigned int i = 2; i < Argument.size(); i++)
    {
        if (Argument[i] != "" && Argument[i] == "-sn")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> splitnum; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-dsets")
        {
            for (unsigned int j = i+1; j < Argument.size(); j++) {
                if (Argument[j].at(0) != '-') DatasetNames.push_back(Argument[j]); else break;
            }
        }
        if (Argument[i] != "" && Argument[i] == "-o")
        {
            if (Argument.size()>i+1) outpath = Argument[i+1]; else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-verbose")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> Verbose; dummystream.clear();
            } else return -1;
        }
    }

    /// print out parsed values
    if (Verbose && MyPE==0) {
        cout << "ParseInputs: Command line arguments: ";
        for (unsigned int i = 0; i < Argument.size(); i++) cout << Argument[i] << " ";
        cout << endl;
    }

    /// error checks
    if (splitnum == 0) {
        if (MyPE==0) cout << endl << "Need to specify the number of split files with '-sn'. Exiting." << endl;
        return -1;
    }
    if (DatasetNames.size() == 0) {
        if (MyPE==0) cout << endl << "Warning: no datasets selected for merging; only meta data will be merged; use '-dsets' to select datasets (e.g., 'dens velx' ...)." << endl;
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
        << " merge_split_flash_files <inputsplitfile> [<OPTIONS>]" << endl << endl
        << "     -sn <number>            : number of split files" << endl
        << "     -dsets <datasetname(s)> : FLASH variables to be processed (e.g, dens velx ...)" << endl
        << "     -o <outputpath>         : output path (default: same as input path)" << endl
        << "     -verbose <level>        : verbose level (0, 1, 2) (default: 1)" << endl
        << endl
        << "Example: merge_split_flash_files Turb_s0000_hdf5_plt_cnt_0010 -sn 4 -dsets dens velx vely velz"
        << endl << endl;
    }
}
