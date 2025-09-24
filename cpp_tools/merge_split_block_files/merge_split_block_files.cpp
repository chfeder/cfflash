/*

  merge_split_block_files.cpp

  Merge split block files (HDF5)

  By Christoph Federrath, 2018-2025

*/

#include "mpi.h" /// MPI lib
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <unistd.h>
#include <hdf5.h>
#include "../Libs/HDFIO.h" // HDF5 IO
#include "../Libs/FlashGG.h" /// Flash Uniform Grid class
#include "../Libs/CFTools.h"

// constants
#define NDIM 3
using namespace std;
enum {X, Y, Z};
static const bool Debug = false;

// MPI stuff
int MyPE = 0, NPE = 1;

// some global stuff (inputs)
string IOfile = "";
string Datasetname = "denw";

/// --------
///   MAIN
/// --------
int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NPE);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
    if (MyPE==0) cout<<"=== merge_split_block_files === using MPI num procs: "<<NPE<<endl;
    if ((argc < 3)||(argc > 3)) {
        if (MyPE==0) cout << "Usage:   merge_split_block_files INPUTFILEBASE DATASETNAME" << endl;
        if (MyPE==0) cout << "Example: merge_split_block_files Turb_hdf5_plt_cnt_0100 vecpotx" << endl;
        if (MyPE==0) cout << "(note that for good file I/O performance, a lower number of cores is usually preferred)" << endl;
        MPI_Finalize(); return 0;
    }
    IOfile = argv[1];
    Datasetname = argv[2];

    MPI_Barrier(MPI_COMM_WORLD);
    long starttime = time(NULL);

    /// first get the block dimensions NB[X,Y,Z]
    FlashGG gg = FlashGG(IOfile, 'w');
    int NBLK = gg.GetNumBlocks();
    vector<int> NB = gg.GetNumCellsInBlock();

    /// now create the big target datasets (just filled with zeros) in the output file
    vector<int> Dimensions(4);
    Dimensions[0] = NBLK;
    Dimensions[1] = NB[Z];
    Dimensions[2] = NB[Y];
    Dimensions[3] = NB[X];
    if (MyPE==0) cout<<"Creating "<<Datasetname<<" in file "<<IOfile<<endl;
    gg.CreateDataset(Datasetname, Dimensions);
    MPI_Barrier(MPI_COMM_WORLD);

    /// decompose domain in blocks (equally; the last PE gets the rest)
    if ((NPE > NBLK) || (NBLK % NPE != 0)) {
        if (MyPE==0) cout << "ERROR. Number of cores must be multiple of total blocks. NBLK = " << NBLK << endl;
        MPI_Finalize();
        return 0;
    }
    vector<int> MyBlocks(0);
    int DivBlocks = (int)((double)(NBLK)/(double)(NPE)+0.5);
    int ModBlocks = NBLK-(NPE-1)*DivBlocks;
    if (MyPE==NPE-1) { // last PE gets the rest (ModBlocks)
        for (int ib=0; ib<ModBlocks; ib++)
            MyBlocks.push_back((NPE-1)*DivBlocks+ib);
    } else { // all others get DivBlocks blocks
        for (int ib=0; ib<DivBlocks; ib++)
            MyBlocks.push_back(MyPE*DivBlocks+ib);
    }
    if (Debug) {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(1000*(MyPE+1));
        cout<<"["<<MyPE<<"] My blocks = ";
        for (unsigned int b=0; b<MyBlocks.size(); b++)
            cout<<MyBlocks[b]<<" "; cout << endl;
    }

    if (MyPE==0) cout << "Entering merging loop now..." << endl;

    Dimensions.resize(3);
    Dimensions[0] = NB[Z];
    Dimensions[1] = NB[Y];
    Dimensions[2] = NB[X];
    const int n_cells_in_block = NB[X]*NB[Y]*NB[Z];
    float *block_data = new float[n_cells_in_block];

    /// loop over all of my blocks, open the right files and overwrite dataset
    CFTools cft = CFTools(); // initialise progress bar
    for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
    {
        /// write progress
        if (MyPE==0) cft.PrintProgressBar(ib, MyBlocks.size());
        /// get actual block index
        int b = MyBlocks[ib];
        /// read block data from individual file
        stringstream dummystream;
        dummystream << setfill('0') << setw(6) << b;
        string block_string = dummystream.str(); dummystream.clear();
        HDFIO HDFinput = HDFIO();
        HDFinput.open(IOfile+"_blocks/"+Datasetname+"_block_"+block_string, 'r');
        HDFinput.read(block_data, Datasetname, H5T_NATIVE_FLOAT);
        HDFinput.close();
        /// overwrite block in output file
        gg.OverwriteBlockVar(b, Datasetname, block_data);

    } //end loop over my blocks

    delete [] block_data;
    MPI_Barrier(MPI_COMM_WORLD);

    /// extend "unknown names" with dataset name
    vector<string> unknown_names = gg.ReadUnknownNames();
    bool datasetname_exists = false;
    for (int i = 0; i < unknown_names.size(); i++) {
        if (Debug) cout << "i, unknown_names[i] = " << i << ", " << unknown_names[i] << endl;
        if (unknown_names[i] == Datasetname) {
            datasetname_exists = true;
            if (MyPE==0) cout << "WARNING: " << Datasetname << " already in 'unknown names'" << endl;
            break;
        }
    }
    if (!datasetname_exists) unknown_names.push_back(Datasetname);
    gg.OverwriteUnknownNames(unknown_names);

    /// print out wallclock time used
    long endtime = time(NULL);
    long duration = endtime-starttime, duration_red = 0;
    if (Debug) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << "****************** Global time to finish = "<<duration_red<<"s ******************" << endl;	

    MPI_Finalize();
    return 0;

} // end main

