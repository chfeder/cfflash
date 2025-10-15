#ifndef FLASHGG_H
#define FLASHGG_H

// The detault is to use this class with MPI support (to switch off MPI, the user needs to #define NO_MPI)
#ifdef NO_MPI
#ifndef MPI_Comm
#define MPI_Comm int
#endif
#ifndef MPI_COMM_NULL
#define MPI_COMM_NULL 0
#endif
#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD 0
#endif
#else
#include <mpi.h>
#endif

#include "HDFIO.h"

#include <hdf5.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include "GRID3D.h" // for uniform grid operations

// define float or double mode of FLASH GG ('general grid'; supposed to work for both UG and AMR)
// (note that float is the default because FLASH plt files are written in single precision)
// (if FLASH GG should use double precision,
//  both FLASH_GG_REAL and FLASH_GG_H5_REAL need to be defined accordingly by the user)
#ifndef FLASH_GG_REAL
#define FLASH_GG_REAL float
#define FLASH_GG_H5_REAL H5T_NATIVE_FLOAT
#endif

namespace NameSpaceFlashGG {
    // constants
    static const double pi = 3.14159265358979323846;
    static const double k_b = 1.380649e-16; // Boltzmann constant
    static const double m_p = 1.67262192369e-24; // proton mass
    static const double g_n = 6.6743e-8; // gravitational constant
    static const double mean_particle_weight = 2.3;
}

/**
 * FlashGG class
 * handels Uniform Grid and AMR (Paramesh) FLASH v3/4 files
 *
 * @author Christoph Federrath (christoph.federrath@anu.edu.au)
 * @version 2010-2025
 *
 */

class FlashGG
{
    private:
    enum {X, Y, Z};
    std::string ClassSignature;
    std::string Inputfilename;
    std::string bounding_box_datasetname;
    std::string node_type_datasetname;
    char grid_type; // 'U' is UG, 'A' is AMR
    int NumBlocks, NumBlocksRep, NumDims, NBXY;
    std::vector<int> NB, NumBlocksIn, N;
    std::vector< std::vector< std::vector<double> > > BoundingBox;
    std::vector<int> NodeType;
    std::vector< std::vector<double> > MinMaxDomain, LBlock;
    std::vector< std::vector<double> > D;
    std::vector<double> Dmin;
    std::vector<double> Dmax, DmaxAll;
    std::vector<double> L;
    int Verbose;
    // for pseudo blocks
    int NumBlocks_PB, NBXY_PB;
    std::vector<int> NB_PB, NumBlocksIn_PB;
    std::vector< std::vector<double> > LBlock_PB;
    std::vector< std::vector< std::vector<double> > > BoundingBox_PB;
    // HDF input/output and MPI mpi_communicator
    HDFIO hdfio; MPI_Comm mpi_comm; int MyPE, NPE;
    std::vector<std::string> Datasetnames;

    /// Constructors
    public: FlashGG(void)
    {
        // empty constructor, so we can define a global FlashGG object in application code
        Verbose = 0; // avoids message from destructor
    };
    public: FlashGG(const std::string flashfile)
    {
        Constructor(flashfile, 'r', MPI_COMM_NULL, 1);
    };
    public: FlashGG(const std::string flashfile, const int verbose)
    {
        Constructor(flashfile, 'r', MPI_COMM_NULL, verbose);
    };
    public: FlashGG(const std::string flashfile, const char read_write_char)
    {
        if (read_write_char == 'w')
            Constructor(flashfile, read_write_char, MPI_COMM_WORLD, 1);
        else
            Constructor(flashfile, read_write_char, MPI_COMM_NULL, 1);
    };
    public: FlashGG(const std::string flashfile, const char read_write_char, const int verbose)
    {
        if (read_write_char == 'w')
            Constructor(flashfile, read_write_char, MPI_COMM_WORLD, verbose);
        else
            Constructor(flashfile, read_write_char, MPI_COMM_NULL, verbose);
    };

    /// Destructor
    public: ~FlashGG()
    {
      //hdfio.close();
      if (Verbose>1) std::cout<<"FlashGG: destructor called."<<std::endl;
    };

    private: void Constructor(const std::string flashfile, const char read_write_char, MPI_Comm comm, const int verbose)
    {
        ClassSignature = "FlashGG: "; // class signature, when this class is printing to stdout
        Inputfilename = flashfile;
        bounding_box_datasetname = "bounding box";
        node_type_datasetname = "node type";
        NumBlocks = 0; NumDims = 0; NBXY = 0;
        Verbose = verbose;
        mpi_comm = comm;
        MyPE = 0;
        NPE = 1;
#ifndef NO_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &NPE);
        MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
#endif
        // open file
        hdfio = HDFIO(Verbose);
        hdfio.open(Inputfilename, read_write_char, mpi_comm);
        Datasetnames = hdfio.getDatasetnames(); // read all datasetnames in the file
        // following calling sequence matters!
        if (Verbose>1) { std::cout<<FuncSig(__func__)<<"calling GetGridType..."<<std::endl;}
        grid_type = this->GetGridType();
        if (Verbose>1) { std::cout<<FuncSig(__func__)<<"Grid type is "<<grid_type<<std::endl;}
        if (grid_type == 'A' || grid_type == 'U') {
            if (Verbose>1) { std::cout<<FuncSig(__func__)<<"calling ReadNumBlocks..."<<std::endl;}
            this->ReadNumBlocks();
            if (Verbose>1) { std::cout<<FuncSig(__func__)<<"calling ReadNumCellsInBlock..."<<std::endl;}
            this->ReadNumCellsInBlock();
            if (Verbose>1) { std::cout<<FuncSig(__func__)<<"calling ReadNodeType..."<<std::endl;}
            this->ReadNodeType();
            if (Verbose>1) { std::cout<<FuncSig(__func__)<<"calling ReadBoundingBoxAndMinMaxDomain..."<<std::endl;}
            this->ReadBoundingBoxAndMinMaxDomain();
        }
        if (grid_type == 'E') { // setup extracted grid blocks
            // (user can call this from outside to set a requested number of cells per extracted grid block)
            std::vector<int> ncells_ext(3,0); // if we feed 0 for ncells_ext, SetupExtractedGridBlocks will find an automatic value
            if (Verbose>1) { std::cout<<FuncSig(__func__)<<"calling SetupExtractedGridBlocks..."<<std::endl;}
            this->SetupExtractedGridBlocks(ncells_ext);
        }
        // initialize pseudo blocks to be equal to actual blocks if UG or extracted grid
        if (grid_type == 'U' || grid_type == 'E') {
            // (user can call this from outside to set a requested number of cells per pseudo block)
            std::vector<int> ncells_pb = NB;
            if (Verbose>1) { std::cout<<FuncSig(__func__)<<"calling SetupPseudoBlocks..."<<std::endl;}
            this->SetupPseudoBlocks(ncells_pb);
        }
        if (Verbose>1) {
            std::cout<<FuncSig(__func__)<<"FlashGG object created for file "<<flashfile<<"."<<std::endl;
            this->PrintInfo();
        }
    };

    // get function signature for printing to stdout
    private: std::string FuncSig(const std::string func_name)
    { return ClassSignature+func_name+": "; };

    // function to get list of all prime factors in integer n
    public: std::vector<int> factors(int n) {
        std::vector<int> primes;
        for (int i = 2; i <= n; i++) {
            while (n % i == 0) {
                primes.push_back(i);
                n /= i;
            }
        }
        return primes;
    };

    public: void SetupExtractedGridBlocks(std::vector<int> ncells_ext)
    {
        // read extraction grid dims and setup blocks
        NumDims = 3;
        N.resize(NumDims);
        int * dims = new int[NumDims];
        hdfio.read(dims, "dims_xyz", H5T_NATIVE_INT);
        for (int dim = 0; dim < NumDims; dim++) N[dim] = dims[dim]; // total resolution
        delete [] dims;
        // determine a good number of cells per block if ncells_ext[?] = 0
        for (int dim = 0; dim < NumDims; dim++) {
            if (ncells_ext[dim] == 0) {
                ncells_ext[dim] = N[dim]; // init with N
                std::vector<int> primes = this->factors(N[dim]);
                for (unsigned int i = 0; i < primes.size(); i++) {
                    int ncells_new = ncells_ext[dim] / primes[i];
                    if (ncells_new >= 32) ncells_ext[dim] = ncells_new; // pick a minimum of 32 cells per block for automatic setting
                }
            }
        }
        // setup exstraction grid blocks
        NB.resize(NumDims);
        NumBlocksIn.resize(NumDims);
        NumBlocks = 1;
        for (int dim = 0; dim < NumDims; dim++) {
            NB[dim] = ncells_ext[dim];
            // number of cells in extraction grid blocks must always be integer multiples of total N cells in domain
            assert (N[dim] % NB[dim] == 0);
            // define number of blocks along [X,Y,Z]
            NumBlocksIn[dim] = N[dim] / NB[dim];
            NumBlocks *= NumBlocksIn[dim];
        }
        NumBlocksRep = NumBlocks;
        NBXY = NB[X]*NB[Y];
        if (Verbose > 0 && MyPE==0) {
            std::cout<<FuncSig(__func__)<<"SetupExtractedGridBlocks: N = "<<N[X]<<" "<<N[Y]<<" "<<N[Z]<<std::endl;
            std::cout<<FuncSig(__func__)<<"SetupExtractedGridBlocks: setting NB = "<<NB[X]<<" "<<NB[Y]<<" "<<NB[Z]<<std::endl;
            std::cout<<FuncSig(__func__)<<"SetupExtractedGridBlocks: NumBlocks: " << NumBlocks<<std::endl;
        }
        // read extraction grid bounds
        double * minmax_xyz = new double[2*NumDims];
        hdfio.read(minmax_xyz, "minmax_xyz", H5T_NATIVE_DOUBLE);
        MinMaxDomain.resize(NumDims);
        for (int dim = 0; dim < NumDims; dim++) {
          MinMaxDomain[dim].resize(2);
          MinMaxDomain[dim][0] = minmax_xyz[2*dim+0];
          MinMaxDomain[dim][1] = minmax_xyz[2*dim+1];
        }
        delete [] minmax_xyz;
        // set domain properties
        L.resize(NumDims);
        for (int dim = 0; dim < NumDims; dim++) L[dim] = MinMaxDomain[dim][1]-MinMaxDomain[dim][0];
        NodeType.resize(NumBlocks);
        BoundingBox.resize(NumBlocks);
        LBlock.resize(NumBlocks);
        D.resize(NumBlocks);
        Dmin.resize(NumDims);
        Dmax.resize(NumDims); DmaxAll.resize(NumDims);
        int nb[3];
        for (nb[Z] = 0; nb[Z] < NumBlocksIn[Z]; nb[Z]++) {
            for (nb[Y] = 0; nb[Y] < NumBlocksIn[Y]; nb[Y]++) {
                for (nb[X] = 0; nb[X] < NumBlocksIn[X]; nb[X]++) {
                    int block = nb[Z]*NumBlocksIn[X]*NumBlocksIn[Y] + nb[Y]*NumBlocksIn[X] + nb[X];
                    NodeType[block] = 1;
                    BoundingBox[block].resize(NumDims);
                    LBlock[block].resize(NumDims);
                    D[block].resize(NumDims);
                    for (int dim = 0; dim < NumDims; dim++) {
                        LBlock[block][dim] = L[dim] / NumBlocksIn[dim];
                        D[block][dim] = LBlock[block][dim] / (double)(NB[dim]);
                        Dmin[dim]=D[block][dim]; Dmax[dim]=Dmin[dim]; DmaxAll[dim]=Dmin[dim];
                        BoundingBox[block][dim].resize(2);
                        BoundingBox[block][dim][0] = MinMaxDomain[dim][0] + LBlock[block][dim]*(nb[dim]+0);
                        BoundingBox[block][dim][1] = MinMaxDomain[dim][0] + LBlock[block][dim]*(nb[dim]+1);
                        if (Verbose > 2 && MyPE==0) {
                            std::cout<<FuncSig(__func__)<<"SetupExtractedGridBlocks: BoundingBox[block="<<block<<"][dim="<<dim<<"][0] = "
                                     <<BoundingBox[block][dim][0]<<std::endl;
                            std::cout<<FuncSig(__func__)<<"SetupExtractedGridBlocks: BoundingBox[block="<<block<<"][dim="<<dim<<"][1] = "
                                     <<BoundingBox[block][dim][1]<<std::endl;
                        }
                    } // dim
                } // nb[X]
            } // nb[Y]
        } // nb[Z]
    };

    // get possible numbers of cells (NB_PB) for pseudo-block setup
    public: std::vector< std::vector<int> > GetNumCellsInBlock_PB_options(void)
    {
        // get integer multiples of N
        std::vector< std::vector<int> > integer_multiples(3);
        for (int dim = X; dim <= Z; dim++)
            for (int n = 1; n <= N[dim]; n++)
                if (N[dim] % n == 0) integer_multiples[dim].push_back(n);
        // print possible value of ncells_pb
        if (MyPE == 0 && Verbose > 0) std::cout<<FuncSig(__func__)<<"Possible values for number of cells in pseudo blocks "<<std::endl;
        for (int dim = X; dim <= Z; dim++) {
            if (MyPE == 0 && Verbose > 0) std::cout<<" in direction ["<<dim<<"] = ";
            for (unsigned int i = 0; i < integer_multiples[dim].size(); i++) {
                if (MyPE == 0 && Verbose > 0) std::cout<<integer_multiples[dim][i]<<" ";
            }
            if (MyPE == 0 && Verbose > 0) std::cout<<std::endl;
        }
        return integer_multiples;
    };

    // setup pseudo blocks (overloaded) to automatically select a good number
    public: void SetupPseudoBlocks(void)
    {
        // get possible options for NB_PB
        std::vector< std::vector<int> > integer_multiples = this->GetNumCellsInBlock_PB_options();
        std::vector<int> ncells_pb(3);
        for (int dim = X; dim <= Z; dim++) {
            bool done_setting = false;
            for (unsigned int i = 0; i < integer_multiples[dim].size(); i++) {
                if ((N[dim]/integer_multiples[dim][i] <= 8) && (!done_setting)) {
                    ncells_pb[dim] = integer_multiples[dim][i];
                    done_setting = true;
                }
            }
            if (MyPE == 0 && Verbose > 0) std::cout<<"  -> automatically selected ncells_pb["<<dim<<"] = "<<ncells_pb[dim]<<std::endl;
        }
        this->SetupPseudoBlocks(ncells_pb); // now set up pseudo blocks with selected ncells_pb
    };

    public: void SetupPseudoBlocks(const std::vector<int> ncells_pb)
    {
        // divide whole domain in pseudo blocks
        // argument ncells_pb is the requested number of cells per pseudo block (per dimension)
        NB_PB.resize(NumDims);
        NumBlocksIn_PB.resize(NumDims);
        NumBlocks_PB = 1;
        for (int dim = 0; dim < NumDims; dim++) {
            NB_PB[dim] = ncells_pb[dim];
            // number of cells in pseudo blocks should always be integer multiples of total N cells in domain
            assert (N[dim] % NB_PB[dim] == 0);
            // define number of PBs along [X,Y,Z]
            NumBlocksIn_PB[dim] = N[dim] / NB_PB[dim];
            NumBlocks_PB *= NumBlocksIn_PB[dim];
        }
        // set up bounding box, etc for pseudo blocks
        NBXY_PB = NB_PB[X]*NB_PB[Y];
        BoundingBox_PB.resize(NumBlocks_PB);
        LBlock_PB.resize(NumBlocks_PB);
        for (int block = 0; block < NumBlocks_PB; block++) {
            LBlock_PB[block].resize(NumDims);
            BoundingBox_PB[block].resize(NumDims);
            for (int dim = 0; dim < NumDims; dim++) {
                LBlock_PB[block][dim] = L[dim] / NumBlocksIn_PB[dim];
                BoundingBox_PB[block][dim].resize(2);
            }
            // assume 3D blocks here
            int kmodb = block % (NumBlocksIn_PB[X]*NumBlocksIn_PB[Y]);
            int kb = block / (NumBlocksIn_PB[X]*NumBlocksIn_PB[Y]);
            int jb = kmodb / NumBlocksIn_PB[X];
            int ib = kmodb % NumBlocksIn_PB[X];
            BoundingBox_PB[block][X][0] = MinMaxDomain[X][0] + ib*LBlock_PB[block][X];
            BoundingBox_PB[block][X][1] = BoundingBox_PB[block][X][0] + LBlock_PB[block][X];
            BoundingBox_PB[block][Y][0] = MinMaxDomain[Y][0] + jb*LBlock_PB[block][Y];
            BoundingBox_PB[block][Y][1] = BoundingBox_PB[block][Y][0] + LBlock_PB[block][Y];
            BoundingBox_PB[block][Z][0] = MinMaxDomain[Z][0] + kb*LBlock_PB[block][Z];
            BoundingBox_PB[block][Z][1] = BoundingBox_PB[block][Z][0] + LBlock_PB[block][Z];
        }
    };

    /// PrintInfo (overloaded)
    public: void PrintInfo(void)
    {
        this->PrintInfo(false);
    }
    /// PrintInfo
    public: void PrintInfo(bool by_block)
    {
        std::string fsig = FuncSig(__func__);
        std::cout<<fsig<<"Grid type = "<<grid_type<<std::endl;
        std::cout<<fsig<<"Number of dimensions (NumDims) = "<<NumDims<<std::endl;
        std::cout<<fsig<<"Total number of blocks (NumBlocks) = "<<NumBlocks<<std::endl;
        if (grid_type == 'A') {
            std::vector<int> LeafBlocks = this->GetLeafBlocks();
            std::cout<<fsig<<"Number of leaf blocks = "<<LeafBlocks.size()<<std::endl;
            std::cout<<fsig<<"Max effective grid resolution: "<<N[X]<<" "<<N[Y]<<" "<<N[Z]<<std::endl;
        }
        if (grid_type == 'U' || grid_type == 'E') {
            std::cout<<fsig<<"Total grid resolution (N) = "<<N[X]<<" "<<N[Y]<<" "<<N[Z]<<std::endl;
            std::cout<<fsig<<"Number of blocks in x,y,z (NumBlocksIn) = "<<NumBlocksIn[X]<<" "<<NumBlocksIn[Y]<<" "<<NumBlocksIn[Z]<<std::endl;
        }
        std::cout<<FuncSig(__func__)<<"Number of cells in block (NB) = "<<NB[X]<<" "<<NB[Y]<<" "<<NB[Z]<<std::endl;
        if (grid_type == 'U' || grid_type == 'E') {
            std::cout<<fsig<<"Number of cells in pseudo block (NB_PB) = "<<NB_PB[X]<<" "<<NB_PB[Y]<<" "<<NB_PB[Z]<<std::endl;
            std::cout<<fsig<<"Number of pseudo blocks in x,y,z (NumBlocksIn_PB) = "<<NumBlocksIn_PB[X]<<" "<<NumBlocksIn_PB[Y]<<" "<<NumBlocksIn_PB[Z]<<std::endl;
            std::cout<<fsig<<"Total number of pseudo blocks (NumBlocks_PB) = "<<NumBlocks_PB<<std::endl;
        }
        std::cout<<fsig<<"Min domain = "<<MinMaxDomain[X][0]<<" "<<MinMaxDomain[Y][0]<<" "<<MinMaxDomain[Z][0]<<std::endl;
        std::cout<<fsig<<"Max domain = "<<MinMaxDomain[X][1]<<" "<<MinMaxDomain[Y][1]<<" "<<MinMaxDomain[Z][1]<<std::endl;
        std::cout<<fsig<<"Length of domain (L) = "<<L[X]<<" "<<L[Y]<<" "<<L[Z]<<std::endl;
        if (grid_type == 'A') {
            std::cout<<fsig<<"Min cell size (leaf blocks) = "<<Dmin[X]<<" "<<Dmin[Y]<<" "<<Dmin[Z]<<std::endl;
            std::cout<<fsig<<"Max cell size (leaf blocks) = "<<Dmax[X]<<" "<<Dmax[Y]<<" "<<Dmax[Z]<<std::endl;
            std::cout<<fsig<<"Max cell size (all blocks)  = "<<DmaxAll[X]<<" "<<DmaxAll[Y]<<" "<<DmaxAll[Z]<<std::endl;
        }
        if (grid_type == 'U' || grid_type == 'E') {
            std::cout<<fsig<<"Cell size = "<<D[0][X]<<" "<<D[0][Y]<<" "<<D[0][Z]<<std::endl;
        }
        // print by-block info, if keyword set
        if (by_block) {
            for (int b = 0; b < NumBlocks; b++) {
              std::cout<<fsig<<"cell size (D) = "<<D[b][X]<<" "<<D[b][Y]<<" "<<D[b][Z]<<std::endl;
              std::cout<<fsig<<"block="<<b<<": Min BBox = "<<BoundingBox[b][X][0]<<" "<<BoundingBox[b][Y][0]<<" "<<BoundingBox[b][Z][0]<<std::endl;
              std::cout<<fsig<<"block="<<b<<": Max BBox = "<<BoundingBox[b][X][1]<<" "<<BoundingBox[b][Y][1]<<" "<<BoundingBox[b][Z][1]<<std::endl;
              std::cout<<fsig<<"block="<<b<<": LBlock = "<<LBlock[b][X]<<" "<<LBlock[b][Y]<<" "<<LBlock[b][Z]<<std::endl;
            }
        }
    };

    // Domain decomposition by blocks (overloaded).
    // Use FlashGG internal MyPE and NPE
    public: std::vector<int> GetMyBlocks(void)
    {
        return GetMyBlocks(this->MyPE, this->NPE, false);
    }
    public: std::vector<int> GetMyBlocks(const int MyPE, const int NPE)
    {
        return GetMyBlocks(MyPE, NPE, false);
    }
    // Domain decomposition by blocks (overloaded).
    // If AMR: leaf blocks; if UG: pseudo blocks (= normal blocks by default).
    // Inputs: MPI rank (MyPE), total number of MPI ranks (NPE).
    // Return indices of blocks for MyPE.
    public: std::vector<int> GetMyBlocks(const int MyPE, const int NPE, const bool allow_idle_cores)
    {
#ifndef NO_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        std::vector<int> BlockList(0);
        if (grid_type == 'A') {
            BlockList = this->GetLeafBlocks();
        }
        if (grid_type == 'U' || grid_type == 'E') {
            for (int i = 0; i < NumBlocks_PB; i++)
                BlockList.push_back(i);
        }
        return GetMyBlocks(MyPE, NPE, BlockList, allow_idle_cores);
    }
    // Domain decomposition by blocks.
    // Inputs: MPI rank (MyPE), total number of MPI ranks (NPE), total number of blocks to distribute (nB).
    // Return indices of blocks for MyPE.
    public: std::vector<int> GetMyBlocks(const int MyPE, const int NPE, const std::vector<int> BlockList)
    {
        return GetMyBlocks(MyPE, NPE, BlockList, false);
    }
    public: std::vector<int> GetMyBlocks(const int MyPE, const int NPE, const std::vector<int> BlockList, const bool allow_idle_cores)
    {
        std::vector<int> MyBlocks(0);
        int DivBlocks = ceil( (double)(BlockList.size()) / (double)(NPE) );
        int NPE_main = BlockList.size() / DivBlocks;
        int ModBlocks = BlockList.size() - NPE_main * DivBlocks;
        if (MyPE < NPE_main) { // (NPE_main) cores get DivBlocks blocks
            for (int ib = 0; ib < DivBlocks; ib++)
                MyBlocks.push_back(BlockList[MyPE*DivBlocks+ib]);
        }
        if (MyPE==0 && Verbose > 0) std::cout<<FuncSig(__func__)<<"GetMyBlocks: First "<<NPE_main<<" core(s) carry(ies) "<<DivBlocks<<" block(s) (each)."<<std::endl;
        if ((MyPE == NPE_main) && (ModBlocks > 0)) { // core (NPE_main + 1) gets the rest (ModBlocks)
            for (int ib = 0; ib < ModBlocks; ib++)
                MyBlocks.push_back(BlockList[NPE_main*DivBlocks+ib]);
            if (Verbose > 0) std::cout<<FuncSig(__func__)<<"GetMyBlocks: Core #"<<NPE_main+1<<" carries "<<ModBlocks<<" block(s)."<<std::endl;
        }
        int NPE_in_use = NPE_main; if (ModBlocks > 0) NPE_in_use += 1;
        if (NPE_in_use < NPE) {
            if (MyPE == 0) std::cout<<FuncSig(__func__)<<"GetMyBlocks: Warning: non-optimal load balancing; "<<NPE-NPE_in_use<<" core(s) remain(s) idle."<<std::endl;
            if (!allow_idle_cores) {
                if (MyPE == 0) std::cout<<FuncSig(__func__)<<"GetMyBlocks: Error: Need to adjust number of cores to avoid idle cores."<<std::endl;
                exit(-1);
            }
        }
        if (Verbose>1) {
            for (int pe = 0; pe < NPE; ++pe) {
#ifndef NO_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
                if (pe == MyPE) {
                    std::cout<<FuncSig(__func__)<<" ["<<MyPE<<"] GetMyBlocks: MyBlocks =";
                    for (unsigned int ib = 0; ib < MyBlocks.size(); ib++)
                        std::cout<<" "<<MyBlocks[ib];
                    std::cout<<std::endl;
                }
#ifndef NO_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
            }
        }
#ifndef NO_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        return MyBlocks;
    }

    /// GetHDFIO
    public: HDFIO GetHDFIO(void) { return hdfio; };

    /// GetNumDims
    public: int GetNumDims(void) { return NumDims; };
    /// GetNumBlocks
    public: int GetNumBlocks(void) { return NumBlocks; };
    /// GetNumBlocksRep
    public: int GetNumBlocksRep(void) { return NumBlocksRep; };
    /// GetNumBlocksVector
    public: std::vector<int> GetNumBlocksVector(void) { return NumBlocksIn; };
    /// GetNumCellsInBlock
    public: std::vector<int> GetNumCellsInBlock(void) { return NB; };
    /// GetMinMaxDomain
    public: std::vector< std::vector<double> > GetMinMaxDomain(void) { return MinMaxDomain; };
    /// GetN
    public: std::vector<int> GetN(void) { return N; };
    /// GetL
    public: std::vector<double> GetL(void) { return L; };
    /// GetDblock (return cell size as function of block and dim)
    public: std::vector< std::vector<double> > GetDblock(void) { return D; };
    /// GetD (return cell size as function of dim only; assumes UG)
    public: std::vector<double> GetD(void) {
        if (grid_type == 'U' || grid_type == 'E') {
            return D[0];
        } else {
            if (MyPE==0) std::cout<<FuncSig(__func__)<<"Error! This is not a uniform grid. Use GetDblock() instead."<<std::endl;
            exit(-1);
        }
    };
    /// GetDmin (return minimum cell size)
    public: std::vector<double> GetDmin(void) { return Dmin; };
    /// GetDmax (return maximum cell size)
    public: std::vector<double> GetDmax(void) { return Dmax; };
    /// GetDmaxAll (return maximum cell size, including non-leaf blocks)
    public: std::vector<double> GetDmaxAll(void) { return DmaxAll; };
    /// GetCellVolumeBlock (return cell volume as function of block)
    public: std::vector<double> GetCellVolumeBlock(void) {
        std::vector<double> CellVolumeBlock(NumBlocks);
        for (int ib=0; ib<NumBlocks; ib++) CellVolumeBlock[ib] = D[ib][X]*D[ib][Y]*D[ib][Z];
        return CellVolumeBlock;
    };
    /// GetLBlock
    public: std::vector< std::vector<double> > GetLBlock(void) { return LBlock; };
    /// GetBoundingBox
    public: std::vector< std::vector <std::vector<double> > > GetBoundingBox(void) { return BoundingBox; };
    /// GetNodeType
    public: std::vector<int> GetNodeType(void) { return NodeType; };
    /// Get LeafBlocks (get a list of all active (leaf) blocks)
    public: std::vector<int> GetLeafBlocks(void) {
        std::vector<int> LeafBlocks(0);
        for (int ib=0; ib<NumBlocks; ib++) {
            if (NodeType[ib] == 1) { // LEAF block
                LeafBlocks.push_back(ib);
            }
        }
        return LeafBlocks;
    };
    /// GetNumBlocks_PB
    public: int GetNumBlocks_PB(void) { return NumBlocks_PB; };
    /// GetNumBlocksVector_PB
    public: std::vector<int> GetNumBlocksVector_PB(void) { return NumBlocksIn_PB; };
    /// GetNumCellsInBlock_PB
    public: std::vector<int> GetNumCellsInBlock_PB(void) { return NB_PB; };
    /// GetLBlock_PB
    public: std::vector< std::vector<double> > GetLBlock_PB(void) { return LBlock_PB; };
    /// GetBoundingBox_PB
    public: std::vector< std::vector <std::vector<double> > > GetBoundingBox_PB(void) { return BoundingBox_PB; };

    /// private helper function that computes derived variables for all block reading functions below
    private: std::vector<std::string> DerivedVar(const int block, const std::string datasetname,
                                                 const long ntot, std::vector<FLASH_GG_REAL*> &data)
    {
        if (Verbose>1 && MyPE==0) std::cout<<FuncSig(__func__)<<"entering..."<<std::endl;
        std::vector<std::string> ret; // returns required dataset names for computing derived variable
        if (hdfio.dataset_exists(datasetname)) { // the datasetname is in the file (default)
            ret.push_back(datasetname);
        }
        else { // we are looking to see if a derived variable is requested
            if (datasetname == "outflow_dens") { // === outflow density ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("oadv"); // data[0]
                    ret.push_back("dens"); // data[1]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) data[0][n] *= data[1][n]; // create oadv * dens
                }
            }
            else if (datasetname == "temperature") { // === temperature (from pres and dens) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("pres"); // data[0]
                    ret.push_back("dens"); // data[1]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL)((double)data[0][n]/(double)data[1][n] *
                                        NameSpaceFlashGG::mean_particle_weight*NameSpaceFlashGG::m_p /
                                        NameSpaceFlashGG::k_b);
                }
            }
            else if (datasetname == "valf") { // === Alfven speed ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("dens"); // data[0]
                    ret.push_back("magx"); // data[1]
                    ret.push_back("magy"); // data[2]
                    ret.push_back("magz"); // data[3]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL) sqrt( ( (double)data[1][n]*(double)data[1][n] +
                                                             (double)data[2][n]*(double)data[2][n] +
                                                             (double)data[3][n]*(double)data[3][n] )
                                                              / (4.0*NameSpaceFlashGG::pi*(double)data[0][n]) );
                }
            }
            else if (datasetname == "cs") { // === sound speed (from pres and dens) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("pres"); // data[0]
                    ret.push_back("dens"); // data[1]
                } else { // compute derived dataset and return in data[0]
                    std::map<std::string, double> real_params = this->ReadRealParameters();
                    double gamma = real_params.at("gamma");
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL) sqrt( gamma*(double)data[0][n]/(double)data[1][n] );
                }
            }
            else if (datasetname == "jnum") { // === Jeans number (from pres and dens) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("pres"); // data[0]
                    ret.push_back("dens"); // data[1]
                } else { // compute derived dataset and return in data[0]
                    std::map<std::string, double> real_params = this->ReadRealParameters();
                    double gamma = real_params.at("gamma");
                    double jeans_num_pre_factor = sqrt(NameSpaceFlashGG::pi / NameSpaceFlashGG::g_n) /
                                                    std::max(D[block][X], std::max(D[block][Y], D[block][Z]));
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL) jeans_num_pre_factor *
                            sqrt( gamma*(double)data[0][n]/(double)data[1][n]/(double)data[1][n]);
                }
            }
            else if (datasetname == "machalf") { // === Alfven Mach number ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("dens"); // data[0]
                    ret.push_back("magx"); // data[1]
                    ret.push_back("magy"); // data[2]
                    ret.push_back("magz"); // data[3]
                    ret.push_back("velx"); // data[4]
                    ret.push_back("vely"); // data[5]
                    ret.push_back("velz"); // data[6]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) {
                        double vel = sqrt( (double)data[4][n]*(double)data[4][n] +
                                           (double)data[5][n]*(double)data[5][n] +
                                           (double)data[6][n]*(double)data[6][n] );
                        double vA = sqrt( ( (double)data[1][n]*(double)data[1][n] +
                                            (double)data[2][n]*(double)data[2][n] +
                                            (double)data[3][n]*(double)data[3][n] )
                                             / (4.0*NameSpaceFlashGG::pi*(double)data[0][n]) );
                        data[0][n] = (FLASH_GG_REAL) (vel / vA);
                    }
                }
            }
            else if (datasetname == "momx") { // === x-momentum density (rho*vx) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("dens"); // data[0]
                    ret.push_back("velx"); // data[1]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) data[0][n] = (FLASH_GG_REAL) ( (double)data[0][n]*(double)data[1][n] );
                }
            }
            else if (datasetname == "momy") { // === y-momentum density (rho*vy) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("dens"); // data[0]
                    ret.push_back("vely"); // data[1]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) data[0][n] = (FLASH_GG_REAL) ( (double)data[0][n]*(double)data[1][n] );
                }
            }
            else if (datasetname == "momz") { // === z-momentum density (rho*vz) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("dens"); // data[0]
                    ret.push_back("velz"); // data[1]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) data[0][n] = (FLASH_GG_REAL) ( (double)data[0][n]*(double)data[1][n] );
                }
            }
            else if (datasetname == "vel2") { // === velocity squared (v^2) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("velx"); // data[0]
                    ret.push_back("vely"); // data[1]
                    ret.push_back("velz"); // data[2]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL) ( (double)data[0][n]*(double)data[0][n] +
                                                       (double)data[1][n]*(double)data[1][n] +
                                                       (double)data[2][n]*(double)data[2][n] );
                }
            }
            else if (datasetname == "mag2") { // === magnetic field squared (B^2) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("magx"); // data[0]
                    ret.push_back("magy"); // data[1]
                    ret.push_back("magz"); // data[2]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL) ( (double)data[0][n]*(double)data[0][n] +
                                                       (double)data[1][n]*(double)data[1][n] +
                                                       (double)data[2][n]*(double)data[2][n] );
                }
            }
            else if (datasetname == "ekin") { // === kinetic energy density (1/2*rho*v^2) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("dens"); // data[0]
                    ret.push_back("velx"); // data[1]
                    ret.push_back("vely"); // data[2]
                    ret.push_back("velz"); // data[3]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL) ( ( (double)data[1][n]*(double)data[1][n] +
                                                         (double)data[2][n]*(double)data[2][n] +
                                                         (double)data[3][n]*(double)data[3][n] )
                                                            * 0.5*(double)data[0][n] );
                }
            }
            else if (datasetname == "emag") { // === magnetic energy density (B^2/8pi) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("magx"); // data[0]
                    ret.push_back("magy"); // data[1]
                    ret.push_back("magz"); // data[2]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL) ( ( (double)data[0][n]*(double)data[0][n] +
                                                         (double)data[1][n]*(double)data[1][n] +
                                                         (double)data[2][n]*(double)data[2][n] )
                                                            / (8.0*NameSpaceFlashGG::pi) );
                }
            }
            else if (datasetname == "eint") { // === specific internal energy (from pres and dens) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("pres"); // data[0]
                    ret.push_back("dens"); // data[1]
                } else { // compute derived dataset and return in data[0]
                    std::map<std::string, double> real_params = this->ReadRealParameters();
                    double gammam1 = real_params.at("gamma") - 1.0;
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL)((double)data[0][n]/(double)data[1][n]) / gammam1;
                }
            }
            else if (datasetname == "vorticity") { // === vorticity magnitude (|nabla x v|) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("vorticity_x"); // data[0]
                    ret.push_back("vorticity_y"); // data[1]
                    ret.push_back("vorticity_z"); // data[2]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++)
                        data[0][n] = (FLASH_GG_REAL) sqrt( (double)data[0][n]*(double)data[0][n] +
                                                           (double)data[1][n]*(double)data[1][n] +
                                                           (double)data[2][n]*(double)data[2][n] );
                }
            }
            else if (   datasetname == "vx_para_b" || datasetname == "vy_para_b" || datasetname == "vz_para_b" || // === x/y/z-vel component parallel to B ===
                        datasetname == "vx_perp_b" || datasetname == "vy_perp_b" || datasetname == "vz_perp_b"  ) // === x/y/z-vel component perpendicular to B ===
            {
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("velx"); // data[0]
                    ret.push_back("vely"); // data[1]
                    ret.push_back("velz"); // data[2]
                    ret.push_back("magx"); // data[3]
                    ret.push_back("magy"); // data[4]
                    ret.push_back("magz"); // data[5]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) {
                        double b_mag = sqrt((double)data[3][n]*(double)data[3][n] +
                                            (double)data[4][n]*(double)data[4][n] +
                                            (double)data[5][n]*(double)data[5][n]);
                        double v_dot_b =    (double)data[3][n]*(double)data[0][n] +
                                            (double)data[4][n]*(double)data[1][n] +
                                            (double)data[5][n]*(double)data[2][n];
                        v_dot_b /= (b_mag*b_mag); // scale by 1/|B|^2
                        if (datasetname == "vx_para_b") data[0][n] = (FLASH_GG_REAL)(v_dot_b*(double)data[3][n]);
                        if (datasetname == "vy_para_b") data[0][n] = (FLASH_GG_REAL)(v_dot_b*(double)data[4][n]);
                        if (datasetname == "vz_para_b") data[0][n] = (FLASH_GG_REAL)(v_dot_b*(double)data[5][n]);
                        if (datasetname == "vx_perp_b") data[0][n] = (FLASH_GG_REAL)(data[0][n]-v_dot_b*(double)data[3][n]);
                        if (datasetname == "vy_perp_b") data[0][n] = (FLASH_GG_REAL)(data[1][n]-v_dot_b*(double)data[4][n]);
                        if (datasetname == "vz_perp_b") data[0][n] = (FLASH_GG_REAL)(data[2][n]-v_dot_b*(double)data[5][n]);
                    }
                }
            }
            else if (datasetname == "magnetic_helicity") // === magnetic helicity A.B (A: vector potential, B: magnetic field) ===
            {
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("vecpotx"); // data[0]
                    ret.push_back("vecpoty"); // data[1]
                    ret.push_back("vecpotz"); // data[2]
                    ret.push_back("magx"); // data[3]
                    ret.push_back("magy"); // data[4]
                    ret.push_back("magz"); // data[5]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) {
                        double a_dot_b =    (double)data[3][n]*(double)data[0][n] +
                                            (double)data[4][n]*(double)data[1][n] +
                                            (double)data[5][n]*(double)data[2][n];
                        data[0][n] = (FLASH_GG_REAL)(a_dot_b);
                    }
                }
            }
            else if (datasetname == "mfx") { // === x-component of v x B (magnetic force x) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("vely"); // data[0]
                    ret.push_back("velz"); // data[1]
                    ret.push_back("magy"); // data[2]
                    ret.push_back("magz"); // data[3]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) data[0][n] = (FLASH_GG_REAL) ( (double)data[0][n]*(double)data[3][n] - (double)data[1][n]*(double)data[2][n] );
                }
            }
            else if (datasetname == "mfy") { // === y-component of v x B (magnetic force y) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("velx"); // data[0]
                    ret.push_back("velz"); // data[1]
                    ret.push_back("magx"); // data[2]
                    ret.push_back("magz"); // data[3]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) data[0][n] = (FLASH_GG_REAL) ( (double)data[1][n]*(double)data[2][n] - (double)data[0][n]*(double)data[3][n] );
                }
            }
            else if (datasetname == "mfz") { // === z-component of v x B (magnetic force z) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("velx"); // data[0]
                    ret.push_back("vely"); // data[1]
                    ret.push_back("magx"); // data[2]
                    ret.push_back("magy"); // data[3]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) data[0][n] = (FLASH_GG_REAL) ( (double)data[0][n]*(double)data[3][n] - (double)data[1][n]*(double)data[2][n] );
                }
            }
            else if (datasetname == "adotv") // === magnetic helicity A.v (A: vector potential, v: velocity field) ===
            {
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("vecpotx"); // data[0]
                    ret.push_back("vecpoty"); // data[1]
                    ret.push_back("vecpotz"); // data[2]
                    ret.push_back("velx"); // data[3]
                    ret.push_back("vely"); // data[4]
                    ret.push_back("velz"); // data[5]
                } else { // compute derived dataset and return in data[0]
                    for (int n=0; n<ntot; n++) {
                        double a_dot_v =    (double)data[0][n]*(double)data[3][n] +
                                            (double)data[1][n]*(double)data[4][n] +
                                            (double)data[2][n]*(double)data[5][n];
                        data[0][n] = (FLASH_GG_REAL)(a_dot_v);
                    }
                }
            }
            else if (datasetname.find("dens_thresh_") != std::string::npos) { // === thresholded density field (set to 1 above threshold; 0 otherwise) ===
                if (ntot == 0) { // initial call to get required datset(s)
                    ret.push_back("dens"); // data[0]
                } else { // compute derived dataset and return in data[0]
                    std::string search_str = "dens_thresh_";
                    int index = datasetname.find(search_str) + search_str.size();
                    double threshold = 0.0;
                    std::stringstream dummystream; // convert string to double
                    dummystream << datasetname.substr(index); dummystream >> threshold; dummystream.clear();
                    static bool firstcall = true;
                    if (Verbose && MyPE==0 && firstcall) {
                        firstcall = false;
                        std::cout<<FuncSig(__func__)<<"using threshold "<<threshold<<", based on requested dataset '"<<datasetname<<"'"<<std::endl;
                    }
                    for (int n=0; n<ntot; n++) {
                        if (data[0][n] > threshold) data[0][n] = 1.0;
                        else data[0][n] = 0.0;
                    }
                }
            }
            else { // not a valid dataset
                if (MyPE==0) std::cout<<FuncSig(__func__)<<"Error: '"<<datasetname<<"' is not a valid dataset."<<std::endl;
                exit(-1);
            }
        }
        return ret;
    };

    /// ReadBlockVar (overloaded) -- reads variable into normal block
    public: FLASH_GG_REAL * ReadBlockVar(const int block, const std::string datasetname, long &ntot)
    {
        ntot = NB[X]*NB[Y]*NB[Z];
        return this->ReadBlockVar(block, datasetname);
    };
    /// ReadBlockVar (reads a block variable into a normal block of size NB[X]*NB[Y]*NB[Z])
    public: FLASH_GG_REAL * ReadBlockVar(const int block, const std::string datasetname)
    {
        if (Verbose>1 && MyPE==0) std::cout<<FuncSig(__func__)<<"entering..."<<std::endl;
        long ntot = 0; // total number of cells in dataset (set to 0 in first call to DerivedVar)
        std::vector<FLASH_GG_REAL*> data; // required data pointers for derived dataset (empty in first call to DerivedVar)
         // get requested dataset names for computing derived dataset (if not a derived dataset, it returns datasetname)
        std::vector<std::string> req_dset_names = DerivedVar(block, datasetname, ntot, data);
        for (unsigned int d = 0; d < req_dset_names.size(); d++) { // read required data pointers for derived dataset
            data.push_back(ReadBlockVar_Direct(block, req_dset_names[d], ntot));
        }
        DerivedVar(block, datasetname, ntot, data); // compute derived dataset for return, located in data[0]
        for (unsigned int d = 1; d < req_dset_names.size(); d++) delete [] data[d]; // clean up rest of data
        return data[0];
    };
    /// ReadBlockVar_Direct (reads a block variable into a normal block of size NB[X]*NB[Y]*NB[Z])
    public: FLASH_GG_REAL * ReadBlockVar_Direct(const int block, const std::string datasetname, long &ntot)
    {
        // check if requested datasetname is in file
        if (!hdfio.dataset_exists(datasetname)) {
            if (MyPE==0) std::cout<<FuncSig(__func__)<<"Error: '"<<datasetname<<"' does not exist."<<std::endl;
            exit(-1);
        }
        ntot = NB[X]*NB[Y]*NB[Z];
        FLASH_GG_REAL * DataPointer = new FLASH_GG_REAL[ntot];
        hsize_t out_offset[3] = {0, 0, 0};
        hsize_t out_count[3] = {(hsize_t)NB[Z], (hsize_t)NB[Y], (hsize_t)NB[X]};
        int b = block % NumBlocks; // note that the % NumBlocks takes care of PBCs (if called with a block replica index)
        if (grid_type == 'A' || grid_type == 'U') { // AMR or UG
            hsize_t offset[4] = {(hsize_t)b, 0, 0, 0};
            hsize_t count[4] = {1, (hsize_t)NB[Z], (hsize_t)NB[Y], (hsize_t)NB[X]};
            hdfio.read_slab(DataPointer, datasetname, FLASH_GG_H5_REAL, offset, count, 3, out_offset, out_count);
        }
        if (grid_type == 'E') { // extracted grid
            int nb[3];
            int mod = b % (NumBlocksIn[X]*NumBlocksIn[Y]);
            nb[X] = mod % NumBlocksIn[X];
            nb[Y] = mod / NumBlocksIn[X];
            nb[Z] = b / (NumBlocksIn[X]*NumBlocksIn[Y]);
            hsize_t offset[3] = {(hsize_t)(nb[Z]*NB[Z]), (hsize_t)(nb[Y]*NB[Y]), (hsize_t)(nb[X]*NB[X])};
            hsize_t count[3] = {(hsize_t)NB[Z], (hsize_t)NB[Y], (hsize_t)NB[X]};
            if (Verbose>1) std::cout<<"offset = "<<offset[X]<<" "<<offset[Y]<<" "<<offset[Z]<<" "<<std::endl;
            hdfio.read_slab(DataPointer, datasetname, FLASH_GG_H5_REAL, offset, count, 3, out_offset, out_count);
        }
        return DataPointer;
    };

    /// ReadBlockVarGC (reads a block variable with added NGC guard cells on each side of the block)
    public: FLASH_GG_REAL * ReadBlockVarGC(const int block, const std::string datasetname,
                                           const int ngc, const bool mass_weighting,
                                           const bool periodic_boundary_conditions)
    {
        if (Verbose>1 && MyPE==0) std::cout<<FuncSig(__func__)<<"entering..."<<std::endl;
        long ntot = 0; // total number of cells in dataset (set to 0 in first call to DerivedVar)
        std::vector<FLASH_GG_REAL*> data; // required data pointers for derived dataset (empty in first call to DerivedVar)
         // get requested dataset names for computing derived dataset (if not a derived dataset, it returns datasetname)
        std::vector<std::string> req_dset_names = DerivedVar(block, datasetname, ntot, data);
        for (unsigned int d = 0; d < req_dset_names.size(); d++) { // read required data pointers for derived dataset
            data.push_back(ReadBlockVarGC_Direct(block, req_dset_names[d], ngc, mass_weighting, periodic_boundary_conditions, ntot));
        }
        DerivedVar(block, datasetname, ntot, data); // compute derived dataset for return, located in data[0]
        for (unsigned int d = 1; d < req_dset_names.size(); d++) delete [] data[d]; // clean up rest of data
        return data[0];
    };
    /// ReadBlockVarGC_Direct (reads a block variable with added NGC guard cells on each side of the block)
    public: FLASH_GG_REAL * ReadBlockVarGC_Direct(const int block, const std::string datasetname,
                                                  const int ngc, const bool mass_weighting,
                                                  const bool periodic_boundary_conditions, long &ntot)
    {
        // number of cells in block with added guard cells ngc
        std::vector<int> NB_GC = NB;
        std::vector<int> NGC(3); NGC[X]=0; NGC[Y]=0; NGC[Z]=0;
        // extend number of cells in block and bounding box to include guard cell layers
        for (int d = 0; d < NumDims; d++) {
            NGC[d] = ngc;
            NB_GC[d] += 2*NGC[d];
        }
        // total number of cells in output block (including GCs)
        ntot = NB_GC[X]*NB_GC[Y]*NB_GC[Z];
        // AMR case; we call the (slow) ReadBlockVarGC_Interpolated, which calls GetUniformGrid
        if (grid_type == 'A') {
            return this->ReadBlockVarGC_Interpolated(block, datasetname, ngc, mass_weighting, periodic_boundary_conditions);
        }
        // else, we are doing UG reading with GCs now...
        FLASH_GG_REAL * DataPointer = new FLASH_GG_REAL[ntot]; // create output block with GCs
        for (int n = 0; n < NB_GC[X]*NB_GC[Y]*NB_GC[Z]; n++) DataPointer[n] = 0.0; // init to 0.0, because we won't set the corners of the block
        // read middle (active) part of the block
        hsize_t offset[4] = {(hsize_t)(block%NumBlocks), 0, 0, 0}; // note that the % NumBlocks takes care of PBCs (if called with a block replica index)
        hsize_t count[4] = {1, (hsize_t)NB[Z], (hsize_t)NB[Y], (hsize_t)NB[X]};
        hsize_t out_offset[3] = {(hsize_t)NGC[Z], (hsize_t)NGC[Y], (hsize_t)NGC[X]};
        hsize_t out_count[3] = {(hsize_t)NB[Z], (hsize_t)NB[Y], (hsize_t)NB[X]};
        hsize_t total_out_count[3] = {(hsize_t)NB_GC[Z], (hsize_t)NB_GC[Y], (hsize_t)NB_GC[X]};
        hdfio.read_slab(DataPointer, datasetname, FLASH_GG_H5_REAL, offset, count, 3, out_offset, out_count, total_out_count);
        // get the 6 surrounding blocks to fill GCs
        std::vector<double> coord(3), coord_block(3);
        std::vector<int> NC(3), Noffset(3), Nout_offset(3);
        for (int d = 0; d < NumDims; d++) coord_block[d] = BoundingBox[block][d][0]+D[block][d]/2.0;
        for (int d = 0; d < NumDims; d++) {
            for (int s = 0; s < 2; s++) {
                // get GC block
                coord = coord_block; // reset coord
                coord[d] = BoundingBox[block][d][s] + (2*s-1)*D[block][d]/2.0; // lower for s=0 and upper for s=1
                if (periodic_boundary_conditions) {
                    if (s==0) if (coord[d] < MinMaxDomain[d][0]) coord[d] += L[d]; //PBCs
                    if (s==1) if (coord[d] > MinMaxDomain[d][1]) coord[d] -= L[d]; //PBCs
                }
                int block_gc = this->BlockIndex(coord[X],coord[Y],coord[Z]);
                // get number of cells and offsets
                NC = NB; // first set to number of cells of active part
                NC[d] = NGC[d];
                Noffset[X] = 0; Noffset[Y] = 0; Noffset[Z] = 0;
                Noffset[d] = (1-s)*(NB[d]-NGC[d]); // lower offset is NB-NGC, upper offset is 0
                Nout_offset = NGC;
                Nout_offset[d] = s*(NB[d]+NGC[d]); // lower offset is 0, upper offset is NB+NGC
                // read GC block part into output block
                hsize_t offset[4] = {(hsize_t)(block_gc%NumBlocks), (hsize_t)Noffset[Z], (hsize_t)Noffset[Y], (hsize_t)Noffset[X]}; // note that the % NumBlocks takes care of PBCs (if called with a block replica index)
                hsize_t count[4] = {1, (hsize_t)NC[Z], (hsize_t)NC[Y], (hsize_t)NC[X]};
                hsize_t out_offset[3] = {(hsize_t)Nout_offset[Z], (hsize_t)Nout_offset[Y], (hsize_t)Nout_offset[X]};
                hsize_t out_count[3] = {(hsize_t)NC[Z], (hsize_t)NC[Y], (hsize_t)NC[X]};
                hsize_t total_out_count[3] = {(hsize_t)NB_GC[Z], (hsize_t)NB_GC[Y], (hsize_t)NB_GC[X]};
                hdfio.read_slab(DataPointer, datasetname, FLASH_GG_H5_REAL, offset, count, 3, out_offset, out_count, total_out_count);
            } // s
        } // d
        return DataPointer;
    };
    /// ReadBlockVarGC_Interpolated (reads a block variable with added NGC guard cells on each side of the block)
    public: FLASH_GG_REAL * ReadBlockVarGC_Interpolated(const int block, const std::string datasetname,
                                                        const int ngc, const bool mass_weighting,
                                                        const bool periodic_boundary_conditions)
    {
        // number of cells in block with added guard cells ngc
        std::vector<int> NB_GC = NB; for (int d = 0; d < NumDims; d++) NB_GC[d] += 2*ngc;
        // extend bounding box to include guard cell layers
        std::vector< std::vector< std::vector<double> > > BoundingBox_GC = BoundingBox;
        for (int b = 0; b < NumBlocks; b++) {
            for (int d = 0; d < NumDims; d++) {
                BoundingBox_GC[b][d][0] -= ngc*D[b][d]; // lower side
                BoundingBox_GC[b][d][1] += ngc*D[b][d]; // upper side
            }
        }
        // get uniform grid of this block with guard cells
        return this->GetUniformGrid(NB_GC, BoundingBox_GC[block], datasetname, mass_weighting, periodic_boundary_conditions);
    };

    /// ReadBlockVarPB (overloaded) -- reads variable into pseudo block
    public: FLASH_GG_REAL * ReadBlockVarPB(const int block_pb, const std::string datasetname, long &size)
    {
        size = NB_PB[X]*NB_PB[Y]*NB_PB[Z];
        return this->ReadBlockVarPB(block_pb, datasetname);
    };
    /// ReadBlockVarPB (reads a normal block variable into a pseudo block of size NB_PB[X]*NB_PB[Y]*NB_PB[Z])
    public: FLASH_GG_REAL * ReadBlockVarPB(const int block_pb, const std::string datasetname)
    {
        if (Verbose>1 && MyPE==0) std::cout<<FuncSig(__func__)<<"entering..."<<std::endl;
        long ntot = 0; // total number of cells in dataset (set to 0 in first call to DerivedVar)
        std::vector<FLASH_GG_REAL*> data; // required data pointers for derived dataset (empty in first call to DerivedVar)
         // get requested dataset names for computing derived dataset (if not a derived dataset, it returns datasetname)
        std::vector<std::string> req_dset_names = DerivedVar(block_pb, datasetname, ntot, data);
        for (unsigned int d = 0; d < req_dset_names.size(); d++) { // read required data pointers for derived dataset
            data.push_back(ReadBlockVarPB_Direct(block_pb, req_dset_names[d], ntot));
        }
        DerivedVar(block_pb, datasetname, ntot, data); // compute derived dataset for return, located in data[0]
        for (unsigned int d = 1; d < req_dset_names.size(); d++) delete [] data[d]; // clean up rest of data
        return data[0];
    };
    /// ReadBlockVarPB_Direct (reads a normal block variable into a pseudo block of size NB_PB[X]*NB_PB[Y]*NB_PB[Z])
    public: FLASH_GG_REAL * ReadBlockVarPB_Direct(const int block_pb, const std::string datasetname, long &ntot)
    {
        if (grid_type == 'A') { // AMR
            if (MyPE==0) std::cout<<FuncSig(__func__)<<"Error: AMR not supported."<<std::endl;
            exit(-1);
        }
        // create new PB dataset pointer
        ntot = NB_PB[X]*NB_PB[Y]*NB_PB[Z];
        FLASH_GG_REAL * DataPointer = new FLASH_GG_REAL[ntot];
        if (grid_type == 'E') { // extracted grid
            int b = block_pb % NumBlocks_PB; // note that the % NumBlocks_PB takes care of PBCs (if called with a block replica index)
            int nb[3];
            int mod = b % (NumBlocksIn_PB[X]*NumBlocksIn_PB[Y]);
            nb[X] = mod % NumBlocksIn_PB[X];
            nb[Y] = mod / NumBlocksIn_PB[X];
            nb[Z] = b / (NumBlocksIn_PB[X]*NumBlocksIn_PB[Y]);
            hsize_t offset[3] = {(hsize_t)(nb[Z]*NB_PB[Z]), (hsize_t)(nb[Y]*NB_PB[Y]), (hsize_t)(nb[X]*NB_PB[X])};
            hsize_t count[3] = {(hsize_t)NB_PB[Z], (hsize_t)NB_PB[Y], (hsize_t)NB_PB[X]};
            hsize_t out_offset[3] = {0, 0, 0};
            hsize_t out_count[3] = {(hsize_t)NB_PB[Z], (hsize_t)NB_PB[Y], (hsize_t)NB_PB[X]};
            if (Verbose>1) std::cout<<FuncSig(__func__)<<"offset = "<<offset[X]<<" "<<offset[Y]<<" "<<offset[Z]<<" "<<std::endl;
            hdfio.read_slab(DataPointer, datasetname, FLASH_GG_H5_REAL, offset, count, 3, out_offset, out_count);
        } // grid_type = 'E'
        if (grid_type == 'U') { // UG
            // find all file blocks that overlap this PB
            std::vector<int> AffectedBlocks = this->GetAffectedBlocks(BoundingBox_PB[block_pb]);
            if (Verbose > 2) std::cout<<FuncSig(__func__)<<"BoundingBox_PB[block_pb][X] = " <<BoundingBox_PB[block_pb][X][0]<<" "
                                                                                            <<BoundingBox_PB[block_pb][X][1]<<std::endl;
            if (Verbose > 2) std::cout<<FuncSig(__func__)<<"AffectedBlocks.size() = "<<AffectedBlocks.size()<<std::endl;
            for (unsigned int ib = 0; ib < AffectedBlocks.size(); ib++)
            {
                int block = AffectedBlocks[ib];
                std::vector<int> cb_offset(NumDims); // file block cell offset
                std::vector<int> cb_count(NumDims); // file block cell count
                std::vector<int> cpb_offset(NumDims); // pseudo block cell offset
                std::vector<int> cpb_count(NumDims); // pseudo block cell count
                for (int dim = 0; dim < NumDims; dim++) {
                    // if left edge of PB is inside current file block
                    if ( (BoundingBox_PB[block_pb][dim][0] > BoundingBox[block][dim][0]) &&
                         (BoundingBox_PB[block_pb][dim][0] < BoundingBox[block][dim][1] ) ) {
                        cb_offset[dim] = (int)((BoundingBox_PB[block_pb][dim][0]+D[block][dim]/2-BoundingBox[block][dim][0])/LBlock[block][dim]*NB[dim]);
                        cpb_offset[dim] = 0;
                    } else {
                        cb_offset[dim] = 0;
                        cpb_offset[dim] = (int)((BoundingBox[block][dim][0]+D[block][dim]/2-BoundingBox_PB[block_pb][dim][0])/LBlock_PB[block_pb][dim]*NB_PB[dim]);
                    }
                    // if right edge of PB is inside current file block
                    if ( (BoundingBox_PB[block_pb][dim][1] > BoundingBox[block][dim][0]) &&
                         (BoundingBox_PB[block_pb][dim][1] < BoundingBox[block][dim][1] ) ) {
                        cb_count[dim] = (int)((BoundingBox_PB[block_pb][dim][1]+D[block][dim]/2-BoundingBox[block][dim][0])/LBlock[block][dim]*NB[dim]) - cb_offset[dim];
                    } else {
                        cb_count[dim] = NB[dim] - cb_offset[dim];
                    }
                }
                // HDF5 offsets and count for slab selections
                hsize_t offset[4] = {(hsize_t)block, (hsize_t)cb_offset[Z], (hsize_t)cb_offset[Y], (hsize_t)cb_offset[X]};
                hsize_t count[4] = {1, (hsize_t)cb_count[Z], (hsize_t)cb_count[Y], (hsize_t)cb_count[X]};
                hsize_t out_offset[3] = {(hsize_t)cpb_offset[Z], (hsize_t)cpb_offset[Y], (hsize_t)cpb_offset[X]};
                hsize_t out_count[3] = {(hsize_t)cb_count[Z], (hsize_t)cb_count[Y], (hsize_t)cb_count[X]};
                hsize_t total_out_count[3] = {(hsize_t)NB_PB[Z], (hsize_t)NB_PB[Y], (hsize_t)NB_PB[X]};
                if (Verbose > 2) {
                    std::cout<<FuncSig(__func__)<<"block = "<<block<<std::endl;
                    for (int dim = 0; dim < NumDims; dim++) {
                        std::cout<<FuncSig(__func__)<<"cb_count["<<dim<<"] = "<<cb_count[dim]
                                <<", cb_offset["<<dim<<"] = "<<cb_offset[dim]
                                <<", cpb_offset["<<dim<<"] = "<<cpb_offset[dim]<<std::endl;
                    }
                }
                // HDFIO for reading slab
                hdfio.read_slab(DataPointer, datasetname, FLASH_GG_H5_REAL, offset, count, 3, out_offset, out_count, total_out_count);
            } // affected main blocks
        } // grid_type = 'U'
        return DataPointer;
    };
    /// ReadBlockVarPB_Interpolated
    public: FLASH_GG_REAL * ReadBlockVarPB_Interpolated(const int block_pb, const std::string datasetname, const bool mass_weighting)
    {
        // get uniform grid for this PB from file blocks
        return this->GetUniformGrid(NB_PB, BoundingBox_PB[block_pb], datasetname, mass_weighting);
    };

    // GetUniformGrid (overloaded: without PBCs)
    public: FLASH_GG_REAL * GetUniformGrid(const std::vector<int> np,
                                           const std::vector< std::vector<double> > bounds,
                                           const std::string datasetname, const bool mass_weighting)
    {
        return this->GetUniformGrid(np, bounds, datasetname, mass_weighting, false);
    }
    // GetUniformGrid (returns a uniform grid; automatically interpolated if need be)
    public: FLASH_GG_REAL * GetUniformGrid(const std::vector<int> np, // target number of grid cells
                                           const std::vector< std::vector<double> > bounds, // target grid bounding box coordinates
                                           const std::string datasetname, const bool mass_weighting,
                                           const bool periodic_boundary_conditions)
    {
        if (Verbose>1) std::cout<<FuncSig(__func__)<<"entering."<<std::endl;
        // create GRID3D uniform grid
        assert (np.size() == 3); // this function is currently only implemented for 3D
        assert (bounds.size() == 3); // this function is currently only implemented for 3D
        GRID3D grid_data = GRID3D(np[X], np[Y], np[Z]);
        grid_data.set_bnds(bounds[X][0], bounds[X][1], bounds[Y][0], bounds[Y][1], bounds[Z][0], bounds[Z][1]);
        grid_data.clear();
        // GRID3D for density in case of mass-weighting
        GRID3D grid_dens;
        if (mass_weighting) {
            grid_dens = GRID3D(np[X], np[Y], np[Z]);
            grid_dens.set_bnds(bounds[X][0], bounds[X][1], bounds[Y][0], bounds[Y][1], bounds[Z][0], bounds[Z][1]);
            grid_dens.clear();
        }
        // if PBCs, create block replicas (extend BoundingBox and NumBlocksRep)
        if (periodic_boundary_conditions) this->AddBlockReplicasPBCs();
        // find affected blocks in file
        if (Verbose>1) std::cout<<FuncSig(__func__)<<"finding affected blocks..."<<std::endl;
        // returns all affected block indices (including block replicas if AddBlockReplicasPBCs was called earlier)
        std::vector<int> AffectedBlocks = this->GetAffectedBlocks(bounds);
        // loop over affected blocks
        if (Verbose>1) std::cout<<FuncSig(__func__)<<"looping..."<<std::endl;
        for (unsigned int ib = 0; ib < AffectedBlocks.size(); ib++)
        {
            int b_all = AffectedBlocks[ib];
            int b = b_all % NumBlocks; // take care of PBCs (if present)
            FLASH_GG_REAL * block_data = this->ReadBlockVar(b, datasetname);
            FLASH_GG_REAL * block_dens = 0;
            if (mass_weighting) block_dens = this->ReadBlockVar(b, "dens");
            // loop over cells in this block and assign data to GRID3D
            for (int k = 0; k < NB[Z]; k++)
                for (int j = 0; j < NB[Y]; j++)
                    for (int i = 0; i < NB[X]; i++) {
                        long index = k*NB[X]*NB[Y] + j*NB[X] + i;
                        double dvol = D[b][X]*D[b][Y]*D[b][Z];
                        double data = block_data[index]*dvol;
                        if (mass_weighting) data *= block_dens[index];
                        std::vector<double> cc = this->CellCenter(b_all, i, j, k); /// this function can also take block replica indices
                        grid_data.add_coord_fields(cc[X], cc[Y], cc[Z], D[b][X], D[b][Y], D[b][Z], data);
                        if (mass_weighting) grid_dens.add_coord_fields(cc[X], cc[Y], cc[Z], D[b][X], D[b][Y], D[b][Z], block_dens[index]*dvol);
                    }

            delete [] block_data;
            if (block_dens) delete [] block_dens;
        }
        // if PBCs, remove block replicas (shrink BoundingBox and NumBlocksRep)
        if (periodic_boundary_conditions) this->RemoveBlockReplicasPBCs();
        // prepare output
        if (Verbose>1) std::cout<<FuncSig(__func__)<<"copying to output..."<<std::endl;
        int ntot = grid_data.get_ntot();
        FLASH_GG_REAL * grid_out = new FLASH_GG_REAL[ntot];
        for (int n = 0; n < ntot; n++) {
            if (mass_weighting) {
                if (grid_data.is_set[n]) {
                    if (grid_dens.field[n] > 0) grid_out[n] = (FLASH_GG_REAL)(grid_data.field[n]/grid_dens.field[n]);
                    else grid_out[n] = 0.0;
                } else grid_out[n] = 0.0;
            }
            else {
                if (grid_data.is_set[n]) grid_out[n] = (FLASH_GG_REAL)(grid_data.field[n]);
                else grid_out[n] = 0.0;
            }
        }
        if (Verbose>1) std::cout<<FuncSig(__func__)<<"exiting."<<std::endl;
        return grid_out;
    }

    /// GetAffectedBlocks
    public: std::vector<int> GetAffectedBlocks(const std::vector< std::vector<double> > bounds)
    {
        return GetAffectedBlocks(bounds, BoundingBox); // use default BoundingBox
    }
    /// GetAffectedBlocks (overloaded to take user block bounding box)
    public: std::vector<int> GetAffectedBlocks(const std::vector< std::vector<double> > bounds, const std::vector< std::vector< std::vector<double> > > BB)
    {
        assert ((int)bounds.size() == NumDims); // quick check on caller
        std::vector<int> AffectedBlocks(NumBlocksRep); // return vector
        unsigned int block_count = 0;
        for (int b=0; b<NumBlocksRep; b++) {
            if (NodeType[b % NumBlocks] == 1) { // LEAF block
                bool overlap = true;
                for (int dim = 0; dim < NumDims; dim++) {
                    if (BB[b][dim][1] <= bounds[dim][0]) { overlap = false; break; }
                    if (BB[b][dim][0] >= bounds[dim][1]) { overlap = false; break; }
                }
                if (overlap) AffectedBlocks[block_count++] = b;
            }
        }
        AffectedBlocks.resize(block_count);
        return AffectedBlocks;
    }

    // AddBlockReplicasPBCs (extend BoundingBox to allow for PBCs)
    public: void AddBlockReplicasPBCs(void)
    {
        // resize (extend) BoundingBox to carry block replicas (always appended after the active blocks, starting at index NumBlocks)
        NumBlocksRep = pow(3, NumDims) * NumBlocks; // reset total number of blocks (now including replicas)
        BoundingBox.resize(NumBlocksRep);
        for (int b_rep = 0; b_rep < NumBlocksRep; b_rep++) {
            BoundingBox[b_rep].resize(NumDims);
            for (int dim = 0; dim < NumDims; dim++) {
                BoundingBox[b_rep][dim].resize(2);
            }
        }
        // generate block replicas for PBCs by setting the BoundingBox'es of the block replicas
        int b_rep_factor = 0;
        int pbc_x_nrep = 1;
        int pbc_y_nrep = 0; if (NumDims > 1) pbc_y_nrep = 1;
        int pbc_z_nrep = 0; if (NumDims > 2) pbc_z_nrep = 1;
        for (int pbc_z = -pbc_z_nrep; pbc_z <= pbc_z_nrep; pbc_z++) { // loop over replicas per dim
            for (int pbc_y = -pbc_y_nrep; pbc_y <= pbc_y_nrep; pbc_y++) { // loop over replicas per dim
                for (int pbc_x = -pbc_x_nrep; pbc_x <= pbc_x_nrep; pbc_x++) { // loop over replicas per dim
                    if ((pbc_x == 0) && (pbc_y == 0) && (pbc_z == 0)) continue; // skip the centre (it's already the original set of blocks)
                    b_rep_factor++;
                    // loop over all active blocks
                    for (int b = 0; b < NumBlocks; b++) {
                        int b_rep = b_rep_factor*NumBlocks + b; // block replica index into BoundingBox (original block index b = b_rep % NumBlocks)
                        assert(b_rep < NumBlocksRep);
                        for (int minmax = 0; minmax < 2; minmax++) {
                            BoundingBox[b_rep][X][minmax] = BoundingBox[b][X][minmax] + pbc_x*L[X];
                            if (NumDims > 1) BoundingBox[b_rep][Y][minmax] = BoundingBox[b][Y][minmax] + pbc_y*L[Y];
                            if (NumDims > 2) BoundingBox[b_rep][Z][minmax] = BoundingBox[b][Z][minmax] + pbc_z*L[Z];
                        } // minmax
                    } // b
                } // pbc_x
            } // pbc_y
        } // pbc_z
    }
    // RemoveBlockReplicasPBCs (shrink BoundingBox to original size)
    public: void RemoveBlockReplicasPBCs(void)
    {
        BoundingBox.resize(NumBlocks);
        NumBlocksRep = NumBlocks;
    }

    /// CreateDataset
    public: void CreateDataset(const std::string datasetname, std::vector<int> Dimensions)
    {
        bool datasetname_exists = false;
        int n_datasets = hdfio.getNumberOfDatasets();
        for (int n=0; n<n_datasets; n++)
        {
            std::string datasetname_in_file = hdfio.getDatasetname(n);
            if (datasetname_in_file == datasetname)
            {
                datasetname_exists = true;
            }
        }
        if (datasetname_exists)
        {
            if (Verbose > 0 && MyPE==0) std::cout<<FuncSig(__func__)<<"Datasetname '"<<datasetname<<"' already exists in file. SKIPPING creation!"<<std::endl;
        }
        else
        {
            hdfio.create_dataset(datasetname, Dimensions, FLASH_GG_H5_REAL, mpi_comm);
        }
    };

    /// OverwriteBlockVar (overloaded for typical blocks of this GG)
    public: void OverwriteBlockVar(const int &block, const std::string datasetname, FLASH_GG_REAL * const DataPointer)
    {
        this->OverwriteBlockVar(block, NB, datasetname, DataPointer);
    };
    /// OverwriteBlockVar (takes the number of cells of that block variable,
    /// in case we created a new block var with different cell dimensions)
    public: void OverwriteBlockVar(const int &block, const std::vector<int> myNB, 
                                   const std::string datasetname, FLASH_GG_REAL * const DataPointer)
    {
        hsize_t offset[4] = {(hsize_t)block, 0, 0, 0};
        hsize_t count[4] = {1, (hsize_t)myNB[Z], (hsize_t)myNB[Y], (hsize_t)myNB[X]};
        hsize_t out_offset[3] = {0, 0, 0};
        hsize_t out_count[3] = {(hsize_t)myNB[Z], (hsize_t)myNB[Y], (hsize_t)myNB[X]};
        hdfio.overwrite_slab(DataPointer, datasetname, FLASH_GG_H5_REAL, offset, count, 3, out_offset, out_count);
    };

    /// public ReadDatasetNames
    public: std::vector<std::string> ReadDatasetNames(void)
    {
        std::vector<std::string> ret = hdfio.getDatasetnames();
        return ret;
    };

    /// ReadBlockSize
    public: FLASH_GG_REAL * ReadBlockSize(const int &block, long &size)
    {
        size = NumDims;
        FLASH_GG_REAL * DataPointer = new FLASH_GG_REAL[size];
        hsize_t offset[2] = {(hsize_t)block, 0};
        hsize_t count[2] = {1, (hsize_t)size};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {(hsize_t)size};
        hdfio.read_slab(DataPointer, "block size", FLASH_GG_H5_REAL, offset, count, 1, out_offset, out_count);
        return DataPointer;
    };

    /// OverwriteBlockSize
    public: void OverwriteBlockSize(const int &block, FLASH_GG_REAL * const DataPointer)
    {
        hsize_t offset[2] = {(hsize_t)block, 0};
        hsize_t count[2] = {1, (hsize_t)NumDims};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {(hsize_t)NumDims};
        hdfio.overwrite_slab(DataPointer, "block size", FLASH_GG_H5_REAL, offset, count, 1, out_offset, out_count);
    };

    /// ReadBoundingBox
    public: FLASH_GG_REAL * ReadBoundingBox(const int &block, long &size)
    {
        size = NumDims*2;
        FLASH_GG_REAL * DataPointer = new FLASH_GG_REAL[size];
        hsize_t offset[3] = {(hsize_t)block, 0, 0};
        hsize_t count[3] = {1, (hsize_t)NumDims, 2};
        hsize_t out_offset[2] = {0, 0};
        hsize_t out_count[2] = {(hsize_t)NumDims, 2};
        hdfio.read_slab(DataPointer, bounding_box_datasetname, FLASH_GG_H5_REAL, offset, count, 2, out_offset, out_count);
        return DataPointer;
    };

    /// OverwriteBoundingBox
    public: void OverwriteBoundingBox(const int &block, FLASH_GG_REAL * const DataPointer)
    {
        hsize_t offset[3] = {(hsize_t)block, 0, 0};
        hsize_t count[3] = {1, (hsize_t)NumDims, 2};
        hsize_t out_offset[2] = {0, 0};
        hsize_t out_count[2] = {(hsize_t)NumDims, 2};
        hdfio.overwrite_slab(DataPointer, bounding_box_datasetname, FLASH_GG_H5_REAL, offset, count, 2, out_offset, out_count);
    };

    /// ReadCoordinates
    public: FLASH_GG_REAL * ReadCoordinates(const int &block, long &size)
    {
        size = NumDims;
        FLASH_GG_REAL * DataPointer = new FLASH_GG_REAL[size];
        hsize_t offset[2] = {(hsize_t)block, 0};
        hsize_t count[2] = {1, (hsize_t)size};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {(hsize_t)size};
        hdfio.read_slab(DataPointer, "coordinates", FLASH_GG_H5_REAL, offset, count, 1, out_offset, out_count);
        return DataPointer;
    };

    /// OverwriteCoordinates
    public: void OverwriteCoordinates(const int &block, FLASH_GG_REAL * const DataPointer)
    {
        hsize_t offset[2] = {(hsize_t)block, 0};
        hsize_t count[2] = {1, (hsize_t)NumDims};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {(hsize_t)NumDims};
        hdfio.overwrite_slab(DataPointer, "coordinates", FLASH_GG_H5_REAL, offset, count, 1, out_offset, out_count);
    };

    /// ReadGID
    public: int * ReadGID(const int &block, long &size)
    {
        size = 15;
        int * DataPointer = new int[size];
        hsize_t offset[2] = {(hsize_t)block, 0};
        hsize_t count[2] = {1, (hsize_t)size};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {(hsize_t)size};
        hdfio.read_slab(DataPointer, "gid", H5T_NATIVE_INT, offset, count, 1, out_offset, out_count);
        return DataPointer;
    };

    /// OverwriteGID
    public: void OverwriteGID(const int &block, int * const DataPointer)
    {
        hsize_t offset[2] = {(hsize_t)block, 0};
        hsize_t count[2] = {1, 15};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {15};
        hdfio.overwrite_slab(DataPointer, "gid", H5T_NATIVE_INT, offset, count, 1, out_offset, out_count);
    };

    /// public ReadNodeType
    public: int * ReadNodeType(const int &block, long &size)
    {
        size = 1;
        int * DataPointer = new int[size];
        hsize_t offset[1] = {(hsize_t)block};
        hsize_t count[1] = {(hsize_t)size};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {(hsize_t)size};
        hdfio.read_slab(DataPointer, node_type_datasetname, H5T_NATIVE_INT, offset, count, 1, out_offset, out_count);
        return DataPointer;
    };

    /// OverwriteNodeType
    public: void OverwriteNodeType(const int &block, int * const DataPointer)
    {
        hsize_t offset[1] = {(hsize_t)block};
        hsize_t count[1] = {1};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {1};
        hdfio.overwrite_slab(DataPointer, node_type_datasetname, H5T_NATIVE_INT, offset, count, 1, out_offset, out_count);
    };

    /// public ReadProcessorNumber
    public: int * ReadProcessorNumber(const int &block, long &size)
    {
        size = 1;
        int * DataPointer = new int[size];
        hsize_t offset[1] = {(hsize_t)block};
        hsize_t count[1] = {(hsize_t)size};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {(hsize_t)size};
        hdfio.read_slab(DataPointer, "processor number", H5T_NATIVE_INT, offset, count, 1, out_offset, out_count);
        return DataPointer;
    };

    /// OverwriteProcessorNumber
    public: void OverwriteProcessorNumber(const int &block, int * const DataPointer)
    {
        hsize_t offset[1] = {(hsize_t)block};
        hsize_t count[1] = {1};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {1};
        hdfio.overwrite_slab(DataPointer, "processor number", H5T_NATIVE_INT, offset, count, 1, out_offset, out_count);
    };

    /// public ReadRefineLevel
    public: int * ReadRefineLevel(const int &block, long &size)
    {
        size = 1;
        int * DataPointer = new int[size];
        hsize_t offset[1] = {(hsize_t)block};
        hsize_t count[1] = {(hsize_t)size};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {(hsize_t)size};
        hdfio.read_slab(DataPointer, "refine level", H5T_NATIVE_INT, offset, count, 1, out_offset, out_count);
        return DataPointer;
    };

    /// OverwriteRefineLevel
    public: void OverwriteRefineLevel(const int &block, int * const DataPointer)
    {
        hsize_t offset[1] = {(hsize_t)block};
        hsize_t count[1] = {1};
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {1};
        hdfio.overwrite_slab(DataPointer, "refine level", H5T_NATIVE_INT, offset, count, 1, out_offset, out_count);
    };

    /// ReadUnknownNames
    public: std::vector<std::string> ReadUnknownNames(void)
    {
        hid_t File_id = hdfio.getFileID();
        hid_t dataset = H5Dopen(File_id, "unknown names", H5P_DEFAULT);
        hid_t dataspace = H5Dget_space(dataset);
        const int rank = H5Sget_simple_extent_ndims(dataspace);
        std::vector<hsize_t> dimens_2d(rank);
        H5Sget_simple_extent_dims(dataspace, dimens_2d.data(), NULL);
        const int n_names = dimens_2d[0];
        // mallocate output
        char ** unk_labels = (char **) malloc (n_names * sizeof (char *));
        // determine whether this is a string of fixed or variable size
        hid_t string_type = H5Tcopy(H5T_C_S1);
        hid_t filetype = H5Dget_type(dataset);
        htri_t variable_length_string = H5Tis_variable_str(filetype);
        // in case it's a string of fixed length:
        if (variable_length_string == 0) {
            size_t string_size = H5Tget_size(filetype); string_size++; /* Make room for null terminator */
            // some horrible additional pointer stuff allocation
            unk_labels[0] = (char *) malloc (n_names * string_size * sizeof (char));
            for (int i = 1; i < n_names; i++) unk_labels[i] = unk_labels[0] + i * string_size;
            H5Tset_size(string_type, string_size);
            H5Dread(dataset, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, unk_labels[0]);
        }
        // in case it's a string of variable length:
        if (variable_length_string == 1) {
            H5Tset_size(string_type, H5T_VARIABLE);
            H5Dread(dataset, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, unk_labels);
        }
        // copy c-strings into std::strings
        std::vector<std::string> ret(n_names);
        for (int i = 0; i < n_names; i++) {
            ret[i] = unk_labels[i];
        }
        // garbage collection
        free (unk_labels[0]);
        free (unk_labels);
        H5Tclose(string_type);
        H5Sclose(dataspace);
        H5Dclose(dataset);
        return ret;
    };

    /// OverwriteUnknownNames (note that this overwrites STRSIZE 4 with STRSIZE 40)
    public: void OverwriteUnknownNames(const std::vector<std::string> unknown_names)
    {
        // copy input strings -> c-strings (of fixed size)
        const int string_size = 40;
        const int n_names = unknown_names.size();
        std::vector<char> unk_labels_flat(n_names * string_size);
        for (int i = 0; i < n_names; i++) std::strcpy(&unk_labels_flat[i * string_size], unknown_names[i].c_str());
        // get file ID, delete previous unknown names and write new one
        hid_t File_id = hdfio.getFileID();
        hid_t string_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(string_type, string_size);
        H5Ldelete(File_id, "unknown names", H5P_DEFAULT); // delete dataset
        const int rank = 2; hsize_t dimens_2d[rank]; dimens_2d[0] = n_names; dimens_2d[1] = 1;
        hid_t dataspace = H5Screate_simple(rank, dimens_2d, NULL);
        hid_t dataset = H5Dcreate(File_id, "unknown names", string_type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        /// create property list for collective dataset i/o
        hid_t plist_id = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
        if (mpi_comm != MPI_COMM_NULL) {
            plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        }
#endif
        H5Dwrite(dataset, string_type, H5S_ALL, H5S_ALL, plist_id, unk_labels_flat.data());
#ifdef H5_HAVE_PARALLEL
        if (mpi_comm != MPI_COMM_NULL) {
            herr_t HDF5_status = H5Pclose(plist_id);
            assert( HDF5_status != -1 );
        }
#endif
        H5Tclose(string_type);
        H5Sclose(dataspace);
        H5Dclose(dataset);
    };

    /// BlockIndex (x, y, z)
    public: inline int BlockIndex(const double &x, const double &y, const double &z)
    {
        int ix = (int)((x-MinMaxDomain[X][0])/L[X]*NumBlocksIn[X]);
        int iy = (int)((y-MinMaxDomain[Y][0])/L[Y]*NumBlocksIn[Y]);
        int iz = (int)((z-MinMaxDomain[Z][0])/L[Z]*NumBlocksIn[Z]);
        int block_index = iz*NumBlocksIn[X]*NumBlocksIn[Y] + iy*NumBlocksIn[X] + ix;
        return block_index;
    };
    /// BlockIndex_PB (x, y, z)
    public: inline int BlockIndex_PB(const double &x, const double &y, const double &z)
    {
        int ix = (int)((x-MinMaxDomain[X][0])/L[X]*NumBlocksIn_PB[X]);
        int iy = (int)((y-MinMaxDomain[Y][0])/L[Y]*NumBlocksIn_PB[Y]);
        int iz = (int)((z-MinMaxDomain[Z][0])/L[Z]*NumBlocksIn_PB[Z]);
        int block_index = iz*NumBlocksIn_PB[X]*NumBlocksIn_PB[Y] + iy*NumBlocksIn_PB[X] + ix;
        return block_index;
    };

    /// CellIndexBlock (block, x, y, z)
    public: inline std::vector<int> CellIndexBlock(const int &block, const double &x, const double &y, const double &z)
    {
        std::vector<int> cell_index(3);
        int b = block % NumBlocks; // take care of PBCs if present (here only for LBlock, access orig block ind)
        cell_index[X] = (int)((x-BoundingBox[block][X][0])/LBlock[b][X]*NB[X]);
        cell_index[Y] = (int)((y-BoundingBox[block][Y][0])/LBlock[b][Y]*NB[Y]);
        cell_index[Z] = (int)((z-BoundingBox[block][Z][0])/LBlock[b][Z]*NB[Z]);
        return cell_index;
    };
    /// CellIndexBlock_PB (block, x, y, z)
    public: inline std::vector<int> CellIndexBlock_PB(const int &block, const double &x, const double &y, const double &z)
    {
        std::vector<int> cell_index(3);
        cell_index[X] = (int)((x-BoundingBox_PB[block][X][0])/LBlock_PB[block][X]*NB_PB[X]);
        cell_index[Y] = (int)((y-BoundingBox_PB[block][Y][0])/LBlock_PB[block][Y]*NB_PB[Y]);
        cell_index[Z] = (int)((z-BoundingBox_PB[block][Z][0])/LBlock_PB[block][Z]*NB_PB[Z]);
        return cell_index;
    };

    /// CellIndexDomain (x, y, z)
    public: inline std::vector<int> CellIndexDomain(const double &x, const double &y, const double &z)
    {
        std::vector<int> cell_index(3);
        cell_index[X] = (int)((x-MinMaxDomain[X][0])/L[X]*N[X]);
        cell_index[Y] = (int)((y-MinMaxDomain[Y][0])/L[Y]*N[Y]);
        cell_index[Z] = (int)((z-MinMaxDomain[Z][0])/L[Z]*N[Z]);
        return cell_index;
    };

    /// CellCenter (block index, cell index i, j, k)
    public: inline std::vector<double> CellCenter(const int &block, const int &i, const int &j, const int &k)
    {
        std::vector<double> cell_center(3);
        int b = block % NumBlocks; // take care of PBCs if present (here only for D, access orig block ind)
        cell_center[X] = BoundingBox[block][X][0]+((double)(i)+0.5)*D[b][X];
        cell_center[Y] = BoundingBox[block][Y][0]+((double)(j)+0.5)*D[b][Y];
        cell_center[Z] = BoundingBox[block][Z][0]+((double)(k)+0.5)*D[b][Z];
        return cell_center;
    };
    /// CellCenter_PB (block index, cell index i, j, k)
    public: inline std::vector<double> CellCenter_PB(const int &block, const int &i, const int &j, const int &k)
    {
        std::vector<double> cell_center(3);
        cell_center[X] = BoundingBox_PB[block][X][0]+((double)(i)+0.5)*D[0][X];
        cell_center[Y] = BoundingBox_PB[block][Y][0]+((double)(j)+0.5)*D[0][Y];
        cell_center[Z] = BoundingBox_PB[block][Z][0]+((double)(k)+0.5)*D[0][Z];
        return cell_center;
    };

    /// CellCenter (block index, cell index)
    public: inline std::vector<double> CellCenter(const int &block, const long &cellindex)
    {
        std::vector<double> cell_center(3);
        int kmod = cellindex % NBXY;
        int k = cellindex / NBXY;
        int j = kmod / NB[X];
        int i = kmod % NB[X];
        int b = block % NumBlocks; // take care of PBCs if present (here only for D, access orig block ind)
        cell_center[X] = BoundingBox[block][X][0]+((double)(i)+0.5)*D[b][X];
        cell_center[Y] = BoundingBox[block][Y][0]+((double)(j)+0.5)*D[b][Y];
        cell_center[Z] = BoundingBox[block][Z][0]+((double)(k)+0.5)*D[b][Z];
        return cell_center;
    };
    /// CellCenter_PB (block index, cell index)
    public: inline std::vector<double> CellCenter_PB(const int &block, const long &cellindex)
    {
        std::vector<double> cell_center(3);
        int kmod = cellindex % NBXY_PB;
        int k = cellindex / NBXY_PB;
        int j = kmod / NB_PB[X];
        int i = kmod % NB_PB[X];
        cell_center[X] = BoundingBox_PB[block][X][0]+((double)(i)+0.5)*D[0][X];
        cell_center[Y] = BoundingBox_PB[block][Y][0]+((double)(j)+0.5)*D[0][Y];
        cell_center[Z] = BoundingBox_PB[block][Z][0]+((double)(k)+0.5)*D[0][Z];
        return cell_center;
    };

    /// BlockCenter (block index)
    public: inline std::vector<double> BlockCenter(const int &block)
    {
        std::vector<double> block_center(3);
        int b = block % NumBlocks; // take care of PBCs if present (here only for LBlock, access orig block ind)
        block_center[X] = BoundingBox[block][X][0]+LBlock[b][X]/2.0;
        block_center[Y] = BoundingBox[block][Y][0]+LBlock[b][Y]/2.0;
        block_center[Z] = BoundingBox[block][Z][0]+LBlock[b][Z]/2.0;
        return block_center;
    };

    /// GetGridType
    public: char GetGridType(void)
    {
        char grid_type = 'U'; // Flash UG
        // check for whether this is an extracted uniform grid (obtained with extractor_mpi)
        std::vector<std::string> datasetnames = hdfio.getDatasetnames();
        for (unsigned int i = 0; i < datasetnames.size(); i++)
            if (datasetnames[i] == "minmax_xyz") {
                grid_type = 'E';
                return grid_type;
            }
        // check for whether this is a Flash AMR grid
        std::map<std::string, int> integer_params = this->ReadIntegerParameters();
        if (integer_params.count("lrefine_max") == 1) { grid_type = 'A'; }
        return grid_type;
    };

    /// ReadNumBlocks
    private: void ReadNumBlocks(void)
    {
        std::vector<int>Dim(3);
        Dim = hdfio.getDims(bounding_box_datasetname);
        NumBlocks    = Dim[0];
        NumBlocksRep = NumBlocks; // default is no block replicas
        NumDims      = Dim[1];
        assert(Dim[2] == 2); // min, max
    };

    /// ReadNumCellsInBlock
    private: void ReadNumCellsInBlock(void)
    {
        NB.resize(NumDims);
        std::map<std::string, int> integer_scalars = ReadIntegerScalars();
        NB[X] = integer_scalars.at("nxb");
        NB[Y] = integer_scalars.at("nyb");
        NB[Z] = integer_scalars.at("nzb");
        NBXY = NB[X]*NB[Y];
    };

    /// WriteNumCellsInBlock
    public: void WriteNumCellsInBlock(std::vector<int> NB_target)
    {
        // only Master PE writes nxb, nyb, nzb
        if (MyPE==0) {
            std::map<std::string, int> int_props = hdfio.ReadFlashIntegerScalars();
            int_props["nxb"] = NB_target[X];
            int_props["nyb"] = NB_target[Y];
            int_props["nzb"] = NB_target[Z];
            hdfio.OverwriteFlashIntegerScalars(int_props);
        } // MyPE == 0
    };

    /// ReadNodeType
    private: void ReadNodeType(void)
    {
        std::vector<int>Dim(1);
        Dim = hdfio.getDims(node_type_datasetname);
        NumBlocks = Dim[0];
        int * NodeTypePointer = new int[NumBlocks];
        hdfio.read(NodeTypePointer, node_type_datasetname, H5T_NATIVE_INT);
        NodeType.resize(NumBlocks);
        for (int block = 0; block < NumBlocks; block++)
            NodeType[block] = NodeTypePointer[block];
        delete [] NodeTypePointer;
    };

    /// ReadBoundingBoxAndMinMaxDomain
    private: void ReadBoundingBoxAndMinMaxDomain(void)
    {
        std::vector<int>Dim(3);
        Dim = hdfio.getDims(bounding_box_datasetname);
        NumBlocks = Dim[0];
        NumDims   = Dim[1];
        assert(Dim[2] == 2); // min, max
        FLASH_GG_REAL * BoundingBoxPointer = new FLASH_GG_REAL[NumBlocks*NumDims*2];
        hdfio.read(BoundingBoxPointer, bounding_box_datasetname, FLASH_GG_H5_REAL);
        if (Verbose>1) {
            for (int ind = 0; ind < NumBlocks*NumDims*2; ind++)
                std::cout<<FuncSig(__func__)<<"BoundingBoxPointer["<<ind<<"] = "<<BoundingBoxPointer[ind]<<std::endl;
        }
        MinMaxDomain.resize(NumDims);
        for (int dim = 0; dim < NumDims; dim++) {
          MinMaxDomain[dim].resize(2);
          MinMaxDomain[dim][0] = BoundingBoxPointer[2*dim+0]; //init
          MinMaxDomain[dim][1] = BoundingBoxPointer[2*dim+1]; //init
        }
        BoundingBox.resize(NumBlocks);
        LBlock.resize(NumBlocks);
        D.resize(NumBlocks);
        Dmin.resize(NumDims); Dmin[X] = +1e99; Dmin[Y] = +1e99; Dmin[Z] = +1e99;
        Dmax.resize(NumDims); Dmax[X] = -1e99; Dmax[Y] = -1e99; Dmax[Z] = -1e99;
        DmaxAll.resize(NumDims); DmaxAll[X] = -1e99; DmaxAll[Y] = -1e99; DmaxAll[Z] = -1e99;
        if (Verbose>1) std::cout<<FuncSig(__func__)<<"NumBlocks: "<<NumBlocks<<std::endl;
        for (int block = 0; block < NumBlocks; block++) {
          BoundingBox[block].resize(NumDims);
          LBlock[block].resize(NumDims);
          D[block].resize(NumDims);
          for (int dim = 0; dim < NumDims; dim++) {
            BoundingBox[block][dim].resize(2);
            for (int minmax = 0; minmax < 2; minmax++) {
                int index = NumDims*2*block + 2*dim + minmax;
                if (Verbose > 2) {
                    std::cout<<FuncSig(__func__)<<"BoundingBoxPointer["<<index<<"] (block="<<block
                        <<" dim="<<dim<<" minmax="<<minmax<<") = "<<BoundingBoxPointer[index]<<std::endl;
                }
                BoundingBox[block][dim][minmax] = BoundingBoxPointer[index];
                if (BoundingBox[block][dim][minmax] < MinMaxDomain[dim][0])
                  MinMaxDomain[dim][0] = BoundingBox[block][dim][minmax];
                if (BoundingBox[block][dim][minmax] > MinMaxDomain[dim][1])
                  MinMaxDomain[dim][1] = BoundingBox[block][dim][minmax];
            }
            LBlock[block][dim] = BoundingBox[block][dim][1]-BoundingBox[block][dim][0];
            D[block][dim] = LBlock[block][dim]/(double)(NB[dim]);
            if (D[block][dim] > DmaxAll[dim]) DmaxAll[dim] = D[block][dim];
            if (NodeType[block] == 1) { // for leaf blocks
                if (D[block][dim] < Dmin[dim]) Dmin[dim] = D[block][dim];
                if (D[block][dim] > Dmax[dim]) Dmax[dim] = D[block][dim];
            }
          }
        }
        // special case of uniform grid (which can also treat split FLASH files)
        if (grid_type == 'U') {
            std::map<std::string, double> real_params = this->ReadRealParameters();
            MinMaxDomain[X][0] = real_params.at("xmin");
            MinMaxDomain[X][1] = real_params.at("xmax");
            MinMaxDomain[Y][0] = real_params.at("ymin");
            MinMaxDomain[Y][1] = real_params.at("ymax");
            MinMaxDomain[Z][0] = real_params.at("zmin");
            MinMaxDomain[Z][1] = real_params.at("zmax");
            std::map<std::string, int> integer_scalars = this->ReadIntegerScalars();
            NumBlocksIn.resize(NumDims);
            NumBlocksIn[X] = integer_scalars.at("iprocs");
            NumBlocksIn[Y] = integer_scalars.at("jprocs");
            NumBlocksIn[Z] = integer_scalars.at("kprocs");
            N.resize(NumDims); L.resize(NumDims);
            for (int dim = 0; dim < NumDims; dim++) {
                N[dim] = NumBlocksIn[dim] * NB[dim];
                L[dim] = MinMaxDomain[dim][1] - MinMaxDomain[dim][0];
                Dmin[dim] = L[dim] / N[dim];
                Dmax[dim] = Dmin[dim];
                DmaxAll[dim] = Dmax[dim];
            }
            for (int block = 0; block < NumBlocks; block++) {
                for (int dim = 0; dim < NumDims; dim++) {
                    LBlock[block][dim] = L[dim] / NumBlocksIn[dim];
                    D[block][dim] = DmaxAll[dim];
                }
                // assume 3D blocks here
                int kmodb = block % (NumBlocksIn[X]*NumBlocksIn[Y]);
                int kb = block / (NumBlocksIn[X]*NumBlocksIn[Y]);
                int jb = kmodb / NumBlocksIn[X];
                int ib = kmodb % NumBlocksIn[X];
                BoundingBox[block][X][0] = MinMaxDomain[X][0] + ib*LBlock[block][X];
                BoundingBox[block][X][1] = BoundingBox[block][X][0] + LBlock[block][X];
                BoundingBox[block][Y][0] = MinMaxDomain[Y][0] + jb*LBlock[block][Y];
                BoundingBox[block][Y][1] = BoundingBox[block][Y][0] + LBlock[block][Y];
                BoundingBox[block][Z][0] = MinMaxDomain[Z][0] + kb*LBlock[block][Z];
                BoundingBox[block][Z][1] = BoundingBox[block][Z][0] + LBlock[block][Z];
            }
        }
        // Check whether we are dealing with a 1D or 2D simulation,
        // in which case the bounding box min and max are the same.
        // In such cases, we reset the bounding box to a finite size,
        // but the number of cells remains 1 in the respective dimension(s)
        for (int block = 0; block < NumBlocks; block++) {
          for (int dim = 0; dim < NumDims; dim++) {
            if (BoundingBox[block][dim][0]==BoundingBox[block][dim][1])
            {
                BoundingBox[block][dim][0] = 0.0;
                BoundingBox[block][dim][1] = 1.0;
                MinMaxDomain[dim][0] = 0.0;
                MinMaxDomain[dim][1] = 1.0;
                LBlock[block][dim] = 1.0;
                D[block][dim] = 1.0;
                Dmin[dim] = 1.0;
                Dmax[dim] = 1.0;
                DmaxAll[dim] = 1.0;
            }
          }
        }
        L.resize(NumDims);
        NumBlocksIn.resize(NumDims);
        N.resize(NumDims);
        for (int dim = 0; dim < NumDims; dim++) {
          L[dim] = MinMaxDomain[dim][1]-MinMaxDomain[dim][0];
          NumBlocksIn[dim] = (int)(L[dim]/LBlock[0][dim]+0.1); // blocks have same size in UG
          N[dim] = (int)(L[dim]/Dmin[dim]+0.1); // effective maximum resolution
        }
        delete [] BoundingBoxPointer;
    };

    /// GetBoundaryConditions
    public: std::string GetBoundaryConditions(void)
    {
        std::string bc = "isolated"; // currently, we only distinguish "isolated" and "periodic"
        if (grid_type == 'A' || grid_type == 'U') {
            /// check if we have periodic boundary conditions
            std::map<std::string, std::string> str_parms = this->ReadStringParameters();
            // PBCs are currently only supported for all directions, so only check xl_boundary_type
            if (str_parms.at("xl_boundary_type") == "periodic") bc = "periodic";
        }
        if (grid_type == 'E') {
            if (MyPE==0) std::cout<<"GetBoundaryConditions: not implemented; returns 'isolated'"<<std::endl;
        }
        return bc;
    };

    /// ReadIntegerScalars
    public: std::map<std::string, int> ReadIntegerScalars(void)
    {   return hdfio.ReadFlashIntegerScalars(); };
    /// ReadIntegerParameters
    public: std::map<std::string, int> ReadIntegerParameters(void)
    {   return hdfio.ReadFlashIntegerParameters(); };
    /// ReadRealScalars
    public: std::map<std::string, double> ReadRealScalars(void)
    {   return hdfio.ReadFlashRealScalars(); };
    /// ReadRealParameters
    public: std::map<std::string, double> ReadRealParameters(void)
    {   return hdfio.ReadFlashRealParameters(); };
    /// ReadLogicalScalars
    public: std::map<std::string, bool> ReadLogicalScalars(void)
    {   return hdfio.ReadFlashLogicalScalars(); };
    /// ReadLogicalParameters
    public: std::map<std::string, bool> ReadLogicalParameters(void)
    {   return hdfio.ReadFlashLogicalParameters(); };
    /// ReadStringScalars
    public: std::map<std::string, std::string> ReadStringScalars(void)
    {   return hdfio.ReadFlashStringScalars(); };
    /// ReadStringParameters
    public: std::map<std::string, std::string> ReadStringParameters(void)
    {   return hdfio.ReadFlashStringParameters(); };

}; // end: FlashGG
#endif
