/*

  sf_cf.cpp

  Computes structure functions or correlation functions of
  grid variables from FLASH output files

  By Christoph Federrath, 2013-2025

*/

#include <algorithm>
#include "mpi.h" /// MPI lib
#include "stdlib.h"
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <limits> /// numeric limits
#include "../Libs/FlashGG.h" /// Flash General Grid class

// constants
#define NDIM 3
using namespace std;
enum {X, Y, Z};
static const bool Debug = false;
static const int MAX_NUM_BINS = 10048;
static const double pi = 3.14159265358979323846;
static const double k_b = 1.380649e-16;
static const double m_p = 1.67262192369e-24;
static const double mu = 2.0;

// MPI stuff
int MyPE = 0, NPE = 1;

// some global stuff (inputs)
FlashGG gg; // global FlashGG object
char GridType;
bool compute_sf_not_cf = true;
string inputfile = "";
double n_samples = 0.0;
int ncells_pseudo_blocks = 0;
bool ncells_pseudo_blocks_set = false;
int MaxStructureFunctionOrder = 10;
bool take_abs = false;
string OutputPath = "./";
vector<string> DsetName(3, ""); // for DSETTYPE (type 0); user supplies custom dataset via -dset (max 3 components)

// structure/correlation function types
static const int MaxNumTypes = 16;
// v, B, sonic-Mach, magneto-sonic-Mach, Alfven-Mach, Alfven-speed, rho^(1/2)*v, rho^(1/3)*v, rho*v, rho, rho^(1/2), ln(rho), GaltierBanerjee2011-flux, GB11-s
enum {DSETTYPE, VELS, MAGS, MACH, MSMACH, AMACH, VALF, SQRTRHO, RHO3, RHOV, RHO, RHOTOHALF, LNRHO, GB11FLUX, GB11S, VELDSETSQ};
string OutfileStringForType[MaxNumTypes] = {"dset", "vels", "mags", "mach", "msmach", "amach", "valf", "sqrtrho", "rho3", "rhov",
                                            "rho", "rhotohalf", "lnrho", "gb11flux", "gb11s", "veldsetsq"};
vector<int> RequestedTypes;
vector<string> RequiredDatasets;

// FLASH datasets that can be used here
static const int MaxNumDsets = 12;
enum {DSETX, DSETY, DSETZ, DENS, VELX, VELY, VELZ, MAGX, MAGY, MAGZ, TEMP, DIVV};
string DsetNames[MaxNumDsets] = {"", "", "", "dens", "velx", "vely", "velz", "magx", "magy", "magz", "temp", "divv"};

// for output
vector<string> OutputFileHeader;
vector< vector<double> > WriteOutTable;

// forward functions
void SetupTypes(void);
void ComputeSFsOrCFsFunctions(void);
float * ReadBlock(const int block, const string datasetname);
void WriteOutAnalysedData(const string OutputFilename);
int ParseInputs(const vector<string> Argument);
void HelpMe(void);


/// --------
///   MAIN
/// --------
int main(int argc, char * argv[])
{
    /*
    // random number test
    int n = 10000;
    double x[n], y[n], z[n];
    mt19937 seed1(0);
    for (int i = 0; i < n; i++) x[i] = random_number(seed1);
    mt19937 seed2(10);
    for (int i = 0; i < n; i++) y[i] = random_number(seed2);
    mt19937 seed3(20);
    for (int i = 0; i < n; i++) z[i] = random_number(seed3);
    for (int i = 0; i < n; i++) cout << x[i] << " " << y[i] << " " << z[i] << " " << endl;
    exit(0);
    */

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NPE);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
    long starttime = time(NULL);

    if (MyPE==0) cout<<"=== sf_cf === using MPI num procs: "<<NPE<<endl;

    /// Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) == -1)
    {
        if (MyPE==0) cout << endl << "Error in ParseInputs(). Exiting." << endl;
        HelpMe();
        MPI_Finalize(); return 0;
    }

    /// setup structure/correlation function types
    SetupTypes();

    /// read data, compute SF, and output
    ComputeSFsOrCFsFunctions();

    /// print out wallclock time used
    long endtime = time(NULL);
    long duration = endtime-starttime;
    long duration_red = 0;
    if (Debug) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    duration = duration_red;
    if (MyPE==0) cout << "****************** Global time to finish = "<<duration<<"s ******************" << endl;

    MPI_Finalize();
    return 0;

} // end main


/** --------------------------- SetupTypes ------------------------------------
 **  setup structure/correlation function types and dataset requirements
 ** --------------------------------------------------------------------------- */
void SetupTypes(void)
{
    /// setup general structure/correlation function type requirements (on reading datasets) and output file strings
    vector< vector<string> > RequiredDsetsForType(MaxNumTypes);
    for (int t = 0; t < MaxNumTypes; t++) {
        if (t == DSETTYPE) {
            for (int i=0; i<3; i++) {
                if (DsetName[i] != "") RequiredDsetsForType[t].push_back(DsetName[i]);
                DsetNames[i] = DsetName[i];
            }
        }
        if (t == VELS) {
            for (int i=0; i<3; i++) RequiredDsetsForType[t].push_back(DsetNames[VELX+i]);
        }
        if (t == MAGS) {
            for (int i=0; i<3; i++) RequiredDsetsForType[t].push_back(DsetNames[MAGX+i]);
        }
        if (t == MACH) {
            for (int i=0; i<3; i++) RequiredDsetsForType[t].push_back(DsetNames[VELX+i]);
            RequiredDsetsForType[t].push_back(DsetNames[TEMP]);
        }
        if ( (t == MSMACH) || (t == AMACH) || (t == VALF) ) {
            RequiredDsetsForType[t].push_back(DsetNames[DENS]);
            for (int i=0; i<3; i++) RequiredDsetsForType[t].push_back(DsetNames[VELX+i]);
            for (int i=0; i<3; i++) RequiredDsetsForType[t].push_back(DsetNames[MAGX+i]);
        }
        if ((t == SQRTRHO) || (t == RHO3) || (t == RHOV) || (t == GB11FLUX)) {
            RequiredDsetsForType[t].push_back(DsetNames[DENS]);
            for (int i=0; i<3; i++) RequiredDsetsForType[t].push_back(DsetNames[VELX+i]);
        }
        if ((t == RHO) || (t == RHOTOHALF) || (t == LNRHO)) {
            RequiredDsetsForType[t].push_back(DsetNames[DENS]);
        }
        if (t == GB11S) {
            RequiredDsetsForType[t].push_back(DsetNames[DENS]);
            for (int i=0; i<3; i++) RequiredDsetsForType[t].push_back(DsetNames[VELX+i]);
            RequiredDsetsForType[t].push_back(DsetNames[DIVV]);
        }
        if (t == VELDSETSQ) {
            for (int i=0; i<3; i++) RequiredDsetsForType[t].push_back(DsetNames[VELX+i]);
            for (int i=0; i<3; i++) {
                if (DsetName[i] != "") RequiredDsetsForType[t].push_back(DsetName[i]);
                DsetNames[i] = DsetName[i];
            }
        }
    }

    /// based on the user's requested types, determine which datsets are required (need to be read from file later)
    for (unsigned int t = 0; t < RequestedTypes.size(); t++) {
        int tid = RequestedTypes[t];
        for (unsigned int d = 0; d < RequiredDsetsForType[tid].size(); d++) {
            string dsetname = RequiredDsetsForType[tid][d];
            // check if this dsetname is not already in RequiredDatasets
            if (find(RequiredDatasets.begin(), RequiredDatasets.end(), dsetname) == RequiredDatasets.end())
                RequiredDatasets.push_back(dsetname); // add dsetname to RequiredDatasets
        }
    }

    if (MyPE==0 && Debug)
        for (unsigned int i = 0; i < RequiredDatasets.size(); i++)
            cout << "RequiredDatasets = "<<RequiredDatasets[i]<<endl;

} /// =======================================================================

/** ------------------------- ScalarProduct --------------------------
 ** computes scalar product of two vectors a, b
 ** ------------------------------------------------------------------ */
inline double ScalarProduct(vector<double> a, vector<double> b) {
    double ret = a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
    return ret;
}
/** ----------------- GetLongComp and GetTrsvComp --------------------
 ** computes longitudinal and transverse components of a vector v w.r.t. dru (unit vector)
 ** ------------------------------------------------------------------ */
inline vector<double> GetLongComp(vector<double> v, vector<double> dru) {
    double long_component = ScalarProduct(v, dru);
    // compute longitudinal vector component
    vector<double> ret(3);
    for (unsigned int i=0; i<3; i++) ret[i] = long_component * dru[i];
    return ret;
};
inline vector<double> GetTrsvComp(vector<double> v, vector<double> dru) {
    // first get longitudinal vector component
    vector<double> long_comp = GetLongComp(v, dru);
    // compute transverse vector component
    vector<double> ret(3);
    for (unsigned int i=0; i<3; i++) ret[i] = v[i] - long_comp[i];
    return ret;
};

/** ---------------- ComputeSFsOrCFsFunctions -----------------------
 ** computes longitudinal and transverse structure functions up to
 ** the 10th order (this assumes a cubic box !) or correlation functions
 ** ------------------------------------------------------------------ */
void ComputeSFsOrCFsFunctions(void)
{
    if (MyPE==0 && Debug) cout<<"ComputeSFsOrCFsFunctions: entering."<<endl;

    /// FLASH file meta data
    gg = FlashGG(inputfile);
    GridType = gg.GetGridType();
    vector< vector<double> > MinMaxDomain = gg.GetMinMaxDomain();
    vector<double> L = gg.GetL();
    vector<double> LHalf(3); for (int dir=X; dir<=Z; dir++) LHalf[dir] = L[dir]/2.;
    vector<double> Dmin = gg.GetDmin();
    vector<double> Dmax = gg.GetDmax();
    vector< vector<double> > D = gg.GetDblock();
    vector<int> N = gg.GetN();
    assert (N[X] == N[Y] && N[X] == N[Z]); // make sure this is a cube where N[X] = N[Y] = N[Z]
    vector<int> LeafBlocks = gg.GetLeafBlocks();
    int NBLK = LeafBlocks.size();
    vector<int> NB = gg.GetNumCellsInBlock();
    vector< vector <vector<double> > > BB = gg.GetBoundingBox();
    vector< vector<double> > LBlock = gg.GetLBlock();

    if (GridType == 'U' || GridType == 'E') { // uniform or extracted grid
        // automatically select a good value for ncells_pb, if not set by user
        if (!ncells_pseudo_blocks_set) {
            gg.SetupPseudoBlocks();
        } else {
            // setup pseudo blocks
            vector<int> NB_PB(3);
            NB_PB[X] = ncells_pseudo_blocks;
            NB_PB[Y] = ncells_pseudo_blocks;
            NB_PB[Z] = ncells_pseudo_blocks;
            if (MyPE == 0) cout << "Using pseudo blocks with "<<ncells_pseudo_blocks<<" (cubed) cells."<< endl;
            gg.SetupPseudoBlocks(NB_PB);
        }
        // specific to pseudo block setup
        NBLK = gg.GetNumBlocks_PB();
        NB = gg.GetNumCellsInBlock_PB();
        BB = gg.GetBoundingBox_PB();
        LBlock = gg.GetLBlock_PB();
        LeafBlocks = vector<int>(NBLK); for (int ib=0; ib<NBLK; ib++) { LeafBlocks[ib] = ib; }
        D = vector< vector<double> >(NBLK, Dmin); // overwrite D with UG or Extracted Grid version
    }

    double LBlockMin = 1e99;
    for (int ib=0; ib<NBLK; ib++) {
        int ibl = LeafBlocks[ib];
        if (LBlock[ibl][X] < LBlockMin) LBlockMin = LBlock[ibl][X];
    }

    // set more meta data and print info
    long NBXYZ = NB[X]*NB[Y]*NB[Z];
    if (MyPE==0) gg.PrintInfo();

    /// decompose domain in blocks
    vector<int> MyBlocks = gg.GetMyBlocks();

    if (Debug) {
        cout<<" ["<<MyPE<<"] MyBlocks =";
        for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
            cout<<" "<<MyBlocks[ib];
        cout<<endl;
    }

    // set the number of samples if the user has not supplied a number
    if (n_samples == 0.0) {
        n_samples = (10.0*NBLK)*(10.0*NBLK);
        if (MyPE==0) {
            cout<<"ComputeSFsOrCFsFunctions: Number of samples was not set; calculating a guess of -n "<<n_samples<<"."<<endl;
            cout<<"  Suggest to increase manually with the -n option, and check for statistical convergence."<<endl;
        }
    }
    if (MyPE==0) cout << "ComputeSFsOrCFsFunctions: Using about "<<n_samples<<" total samples."<<endl;
    if (MyPE==0) cout << "ComputeSFsOrCFsFunctions: === Start looping ==="<<endl;

    /// initialization
    const int NumTypes = RequestedTypes.size();
    const double onethird = static_cast<double>(1.0/3.0);

    /// construct bins grid
    int NumberOfBins = 0;
    double grid     [MAX_NUM_BINS]; grid     [0] = 0.0;
    double grid_stag[MAX_NUM_BINS]; grid_stag[0] = Dmax[X]/2.;
    const double MaxLength = 0.5*L[X]; /// max length between two different points in a cubic box
    double length = Dmax[X];
    while (grid[NumberOfBins] < 0.5*sqrt(3.0)*L[X]) /// max distance
    {
        NumberOfBins++;
        grid     [NumberOfBins] = length;
        grid_stag[NumberOfBins] = length + Dmax[X]/2.;
        length += Dmax[X];
        if (NumberOfBins > MAX_NUM_BINS) {
            if (MyPE==0) cout << "ERROR. NumberOfBins exceeds MaximumNumberofBins!" << endl;
            MPI_Finalize();
        }
    }
    NumberOfBins++;

    /// initialize main buffers
    const double numeric_epsilon = 10*numeric_limits<double>::epsilon();
    const int    NumBufferIndices = NumTypes*MaxStructureFunctionOrder*NumberOfBins;

    /// initialize additional buffers
    double     *buf1_bin_counter_long           = new double    [NumBufferIndices];
    double     *buf1_bin_counter_trsv           = new double    [NumBufferIndices];
    double     *buf1_binsum_long                = new double    [NumBufferIndices];
    double     *buf1_binsum_trsv                = new double    [NumBufferIndices];
    long       *buf1_numeric_error_counter_long = new long      [NumBufferIndices];
    long       *buf1_numeric_error_counter_trsv = new long      [NumBufferIndices];
    double     *buf2_bin_counter_long           = new double    [NumBufferIndices];
    double     *buf2_bin_counter_trsv           = new double    [NumBufferIndices];
    double     *buf2_binsum_long                = new double    [NumBufferIndices];
    double     *buf2_binsum_trsv                = new double    [NumBufferIndices];
    for (int BufIndex = 0; BufIndex < NumBufferIndices; BufIndex++)
    {
        buf1_bin_counter_long          [BufIndex] = 0;
        buf1_bin_counter_trsv          [BufIndex] = 0;
        buf1_binsum_long               [BufIndex] = numeric_epsilon;
        buf1_binsum_trsv               [BufIndex] = numeric_epsilon;
        buf1_numeric_error_counter_long[BufIndex] = 0;
        buf1_numeric_error_counter_trsv[BufIndex] = 0;
        buf2_bin_counter_long          [BufIndex] = 0;
        buf2_bin_counter_trsv          [BufIndex] = 0;
        buf2_binsum_long               [BufIndex] = numeric_epsilon;
        buf2_binsum_trsv               [BufIndex] = numeric_epsilon;
    }

    double incr_long[MaxNumTypes] = {0}; double incr_trsv[MaxNumTypes] = {0};
    double incr_long_pow[MaxNumTypes] = {0}; double incr_trsv_pow[MaxNumTypes] = {0};

    // for random numbers (Mersenne-Twister)
    uniform_real_distribution<double> random_number(0.0, 1.0);
    // different random seed for each PE
    mt19937 seed(MyPE);

    /// loop over my blocks (b1)
    for (unsigned int ib1=0; ib1<MyBlocks.size(); ib1++)
    {
        int b1 = MyBlocks[ib1];
        if (MyPE==0) cout<<" ["<<setw(6)<<MyPE<<"] working on block1 = "<<setw(6)<<ib1+1<<" of "<<MyBlocks.size()<<endl;

        // block center 1
        vector<double> bc1(3);
        bc1[X] = (BB[b1][X][0]+BB[b1][X][1])/2.;
        bc1[Y] = (BB[b1][Y][0]+BB[b1][Y][1])/2.;
        bc1[Z] = (BB[b1][Z][0]+BB[b1][Z][1])/2.;
        double block_diagonal = sqrt( (BB[b1][X][1]-BB[b1][X][0])*(BB[b1][X][1]-BB[b1][X][0]) +
                                      (BB[b1][Y][1]-BB[b1][Y][0])*(BB[b1][Y][1]-BB[b1][Y][0]) +
                                      (BB[b1][Z][1]-BB[b1][Z][0])*(BB[b1][Z][1]-BB[b1][Z][0]) );

        /// read block data 1
        if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block1..."<<endl;
        float *data1[MaxNumDsets]; for (int i = 0; i < MaxNumDsets; i++) data1[i] = 0; // init pointers to NULL
        for (int i = 0; i < MaxNumDsets; i++) // find requested datasets and read them
            if (find(RequiredDatasets.begin(), RequiredDatasets.end(), DsetNames[i]) != RequiredDatasets.end()) {
                data1[i] = ReadBlock(b1, DsetNames[i]);
                if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block1 "+DsetNames[i]+" done."<<endl;
            }
        if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block1 done."<<endl;

        /// loop over all blocks (b2)
        bool printed_progress_1_b2 = false, printed_progress_10_b2 = false, printed_progress_100_b2 = false;
        for (int ib2=0; ib2<NBLK; ib2++)
        {
            int b2 = LeafBlocks[ib2];

            // write progress
            double percent_done = (double)(b2+1)/NBLK*100;
            bool print_progress = false;
            if (percent_done >    1.0 && !printed_progress_1_b2  ) {print_progress=true; printed_progress_1_b2  =true;}
            if (percent_done >   10.0 && !printed_progress_10_b2 ) {print_progress=true; printed_progress_10_b2 =true;}
            if (percent_done == 100.0 && !printed_progress_100_b2) {print_progress=true; printed_progress_100_b2=true;}
            if (print_progress && MyPE==0) cout<<"   ..."<<percent_done<<"% done..."<<endl;

            // total volume (1 for UG or Extracted grid; or sum of cell1 and cell2 vol for AMR grid)
            double TotVol = 1.0;
            if (GridType == 'A') TotVol = D[b1][X]*D[b1][Y]*D[b1][Z] + D[b2][X]*D[b2][Y]*D[b2][Z];

            // block center 2
            vector<double> bc2(3);
            bc2[X] = (BB[b2][X][0]+BB[b2][X][1])/2.;
            bc2[Y] = (BB[b2][Y][0]+BB[b2][Y][1])/2.;
            bc2[Z] = (BB[b2][Z][0]+BB[b2][Z][1])/2.;

            // distance between block center 1 and 2
            vector<double> dbc(3);
            dbc[X] = bc2[X]-bc1[X]; if (dbc[X]>LHalf[X]) dbc[X]-=L[X]; if (dbc[X]<-LHalf[X]) dbc[X]+=L[X];
            dbc[Y] = bc2[Y]-bc1[Y]; if (dbc[Y]>LHalf[Y]) dbc[Y]-=L[Y]; if (dbc[Y]<-LHalf[Y]) dbc[Y]+=L[Y];
            dbc[Z] = bc2[Z]-bc1[Z]; if (dbc[Z]>LHalf[Z]) dbc[Z]-=L[Z]; if (dbc[Z]<-LHalf[Z]) dbc[Z]+=L[Z];

            double block_distance = sqrt(dbc[X]*dbc[X]+dbc[Y]*dbc[Y]+dbc[Z]*dbc[Z]);

            if (Debug) cout << block_distance << " " << dbc[X] << " " << dbc[Y] << " " << dbc[Z] << endl;

            /// loop parameters for samples in block 1
            long n_samples1 = max(1.0,round(sqrt((double)(n_samples))/(double)(NBLK)));
            if (n_samples1 < 10 && MyPE == 0) cout << "Warning: n_samples1 = "<<n_samples1<<
                " < 10 !!! Consider increasing n samples; suggesting n > " << (10.0*NBLK)*(10.0*NBLK) << endl;

            /// loop parameters for samples in block 2
            long n_samples2 = min((double)(n_samples1),(double)NBXYZ);
            if (Debug) cout << "n_samples2 = " << n_samples2 << endl;

            double rmin = LBlockMin, rmax = MaxLength;
            double n_samples2_norm = (double)(n_samples2)/(rmax/rmin-0.9);
            long n_samples2_block = n_samples2_norm*pow(block_distance/rmax,-2.0);
            if (block_distance == 0) n_samples2_block = n_samples2_norm*n_samples2;
            if (Debug) {
                cout << "rmin = " << rmin << endl;
                cout << "rmax = " << rmax << endl;
                cout << "(rmax/rmin-1.0) = " << (rmax/rmin-1.0) << endl;
                cout << "n_samples2_norm = " << n_samples2_norm << endl;
                cout << "n_samples2_block = " << n_samples2_block << endl;
            }

            double NBS = (double)n_samples2_block; if (NBS==0) NBS = 1.0;
            int incr = max((double)(NBXYZ)/NBS,1.0);
            if (incr == 1 && MyPE == 0) cout << "Warning: looping over ALL cells in block2." << endl;
            if (n_samples1*n_samples2_block > 1e9)
            {
                if (MyPE == 0) cout << "Warning: n_samples1*n_samples2_block > 1e9. Resetting to 1e9." << endl;
                incr = (int)max((double)(NBXYZ)/(1e9/n_samples1),1.);
            }

            if (Debug) cout << block_distance << " " << MaxLength+block_diagonal << " " << n_samples2_block << " " << incr << " " << NBXYZ << endl;

            /// continue with next block if too far away or no samples
            if ((block_distance > MaxLength+block_diagonal) || (n_samples2_block < 1) || incr >= NBXYZ) continue;

            /// read block data 2
            if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block2..."<<endl;
            float *data2[MaxNumDsets];
            for (int i = 0; i < MaxNumDsets; i++)
                if (data1[i] == 0) data2[i] = 0; // init pointers to NULL, if not required
                else { // this was already required for data1, so we also need it for data2
                    data2[i] = ReadBlock(b2, DsetNames[i]);
                    if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block2 "+DsetNames[i]+" done."<<endl;
                }
            if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block2 done. block1 = "<<setw(6)<<ib1+1<<" of "<<MyBlocks.size()<<
                         "; block2 count on root processor = "<<setw(6)<<b2+1<<" of "<<NBLK<<" now computing SFs..."<<endl;

            if (Debug && MyPE==0) cout<<"block_distance="<<setw(16)<<block_distance<<" n_samples1="<<setw(16)<<n_samples1
                             <<" n_samples2_block="<<setw(16)<<(int)((double)NBXYZ/(double)incr)<<endl;

            for (long n1=0; n1<n_samples1; n1++)
            {
                int i1(NB[X]*(random_number(seed)));
                int j1(NB[Y]*(random_number(seed)));
                int k1(NB[Z]*(random_number(seed)));

                long cellindex1 = k1*NB[Y]*NB[X] + j1*NB[X] + i1;
                vector<double> cc1;
                if (GridType == 'U' || GridType == 'E')
                    cc1 = gg.CellCenter_PB(b1, i1, j1, k1);
                else // AMR
                    cc1 = gg.CellCenter(b1, i1, j1, k1);
                if (Debug) cout<<">>> cc1 = "<<cc1[X]<<" "<<cc1[Y]<<" "<<cc1[Z]<<endl;

                for (long cellindex2=0; cellindex2<NBXYZ; cellindex2+=incr)
                {
                    vector<double> cc2;
                    if (GridType == 'U' || GridType == 'E')
                        cc2 = gg.CellCenter_PB(b2, cellindex2);
                    else // AMR
                        cc2 = gg.CellCenter(b2, cellindex2);
                    if (Debug) cout<<"cc2 = "<<cc2[X]<<" "<<cc2[Y]<<" "<<cc2[Z]<<endl;

                    /// distance and length increments (mind PBCs)
                    double di=cc2[X]-cc1[X]; if (di>LHalf[X]) di-=L[X]; if (di<-LHalf[X]) di+=L[X];
                    double dj=cc2[Y]-cc1[Y]; if (dj>LHalf[Y]) dj-=L[Y]; if (dj<-LHalf[Y]) dj+=L[Y];
                    double dk=cc2[Z]-cc1[Z]; if (dk>LHalf[Z]) dk-=L[Z]; if (dk<-LHalf[Z]) dk+=L[Z];
                    double distance = sqrt(di*di+dj*dj+dk*dk);
                    vector<double> dru(3);
                    dru[X]=di/distance; dru[Y]=dj/distance; dru[Z]=dk/distance; // unit vector spatial increment

                    /// skip self-comparison and cells > MaxLength
                    if (distance <= 0 || distance > MaxLength) continue;

                    // variable vectors at pos 1 and 2 for correlation function computation
                    vector<double> dv(3), v1(3), v2(3);

                    /// ========= FILL SF containers =============
                    for (int t = 0; t < NumTypes; t++)
                    {
                        for (unsigned int dir=X; dir<=Z; dir++) {
                            dv[dir] = 0.0; v1[dir] = 0.0; v2[dir] = 0.0;
                        }

                        switch(RequestedTypes[t])
                        {
                            case DSETTYPE: /// dset (any dataset name supplied by the user via -dset and type=0)
                            {
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    if (data1[DSETX+dir] != 0) {
                                        v1[dir] = (double)data1[DSETX+dir][cellindex1];
                                        v2[dir] = (double)data2[DSETX+dir][cellindex2];
                                    }
                                }
                                break;
                            }
                            case VELS: /// v (velocity)
                            {
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = (double)data1[VELX+dir][cellindex1];
                                    v2[dir] = (double)data2[VELX+dir][cellindex2];
                                }
                                break;
                            }
                            case MAGS: /// B (magnetic field)
                            {
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = (double)data1[MAGX+dir][cellindex1];
                                    v2[dir] = (double)data2[MAGX+dir][cellindex2];
                                }
                                break;
                            }
                            case MACH: /// Mach number (needs temp)
                            {
                                double cs1 = sqrt((double)data1[TEMP][cellindex1]*k_b/mu/m_p);
                                double cs2 = sqrt((double)data2[TEMP][cellindex2]*k_b/mu/m_p);
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = (double)data1[VELX+dir][cellindex1]/cs1;
                                    v2[dir] = (double)data2[VELX+dir][cellindex2]/cs2;
                                }
                                break;
                            }
                            case MSMACH: /// magneto-sonic Mach number = v / sqrt(c_s^2 + 0.5*v_A^2),
                                         /// with sound speed c_s and Alfven speed v_A
                            {
                                double cssq1 = 1.0, cssq2 = 1.0; // sound speed squared
                                if (data1[TEMP] && data2[TEMP]) {
                                    cssq1 = (double)data1[TEMP][cellindex1]*k_b/mu/m_p;
                                    cssq2 = (double)data2[TEMP][cellindex2]*k_b/mu/m_p;
                                }
                                double rho1 = (double)data1[DENS][cellindex1];
                                double rho2 = (double)data2[DENS][cellindex2];
                                double b1[3] = { (double)data1[MAGX][cellindex1], (double)data1[MAGY][cellindex1], (double)data1[MAGZ][cellindex1] };
                                double b2[3] = { (double)data2[MAGX][cellindex2], (double)data2[MAGY][cellindex2], (double)data2[MAGZ][cellindex2] };
                                double vAsq1 = (b1[X]*b1[X] + b1[Y]*b1[Y] + b1[Z]*b1[Z]) / (4.0*pi*rho1); // Alfven speed squared
                                double vAsq2 = (b2[X]*b2[X] + b2[Y]*b2[Y] + b2[Z]*b2[Z]) / (4.0*pi*rho2); // Alfven speed squared
                                double ms1 = sqrt(cssq1 + 0.5*vAsq1); // magneto-sonic speed
                                double ms2 = sqrt(cssq2 + 0.5*vAsq2); // magneto-sonic speed
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = (double)data1[VELX+dir][cellindex1]/ms1;
                                    v2[dir] = (double)data2[VELX+dir][cellindex2]/ms2;
                                }
                                break;
                            }
                            case AMACH: /// Alfven Mach number, Ma = v/v_A = v*sqrt(4*pi*rho)/B
                            {
                                double rho1 = (double)data1[DENS][cellindex1];
                                double rho2 = (double)data2[DENS][cellindex2];
                                double b1[3] = { (double)data1[MAGX][cellindex1], (double)data1[MAGY][cellindex1], (double)data1[MAGZ][cellindex1] };
                                double b2[3] = { (double)data2[MAGX][cellindex2], (double)data2[MAGY][cellindex2], (double)data2[MAGZ][cellindex2] };
                                double vA1 = sqrt( (b1[X]*b1[X] + b1[Y]*b1[Y] + b1[Z]*b1[Z]) / (4.0*pi*rho1) ); // Alfven speed
                                double vA2 = sqrt( (b2[X]*b2[X] + b2[Y]*b2[Y] + b2[Z]*b2[Z]) / (4.0*pi*rho2) ); // Alfven speed
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = (double)data1[VELX+dir][cellindex1]/vA1;
                                    v2[dir] = (double)data2[VELX+dir][cellindex2]/vA2;
                                }
                                break;
                            }
                            case VALF: /// Alfven velocity v_A = B / sqrt(4*pi*rho)
                            {
                                double denom1 = sqrt(4.0*pi*(double)data1[DENS][cellindex1]);
                                double denom2 = sqrt(4.0*pi*(double)data2[DENS][cellindex2]);
                                double b1[3] = { (double)data1[MAGX][cellindex1], (double)data1[MAGY][cellindex1], (double)data1[MAGZ][cellindex1] };
                                double b2[3] = { (double)data2[MAGX][cellindex2], (double)data2[MAGY][cellindex2], (double)data2[MAGZ][cellindex2] };
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = b1[dir]/denom1;
                                    v2[dir] = b2[dir]/denom2;
                                }
                                break;
                            }
                            case SQRTRHO: /// sqrt(rho)*v
                            {
                                double sqrtrho1 = sqrt((double)data1[DENS][cellindex1]);
                                double sqrtrho2 = sqrt((double)data2[DENS][cellindex2]);
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = sqrtrho1*(double)data1[VELX+dir][cellindex1];
                                    v2[dir] = sqrtrho2*(double)data2[VELX+dir][cellindex2];
                                }
                                break;
                            }
                            case RHO3: /// rho^(1/3)*v
                            {
                                double pow3rho1 = pow((double)data1[DENS][cellindex1],onethird);
                                double pow3rho2 = pow((double)data2[DENS][cellindex2],onethird);
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = pow3rho1*(double)data1[VELX+dir][cellindex1];
                                    v2[dir] = pow3rho2*(double)data2[VELX+dir][cellindex2];
                                }
                                break;
                            }
                            case RHOV: /// rho*v
                            {
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = (double)data1[DENS][cellindex1]*(double)data1[VELX+dir][cellindex1];
                                    v2[dir] = (double)data2[DENS][cellindex2]*(double)data2[VELX+dir][cellindex2];
                                }
                                break;
                            }
                            case RHO: /// rho
                            {
                                v1[X] = (double)data1[DENS][cellindex1];
                                v2[X] = (double)data2[DENS][cellindex2];
                                break;
                            }
                            case RHOTOHALF: /// sqrt(rho)='rhotohalf'
                            {
                                v1[X] = sqrt((double)data1[DENS][cellindex1]);
                                v2[X] = sqrt((double)data2[DENS][cellindex2]);
                                break;
                            }
                            case LNRHO: /// ln(rho)
                            {
                                v1[X] = log((double)data1[DENS][cellindex1]);
                                v2[X] = log((double)data2[DENS][cellindex2]);
                                break;
                            }
                            case GB11FLUX: /// gb11 Flux; Exact equation (11) for F(r):
                                           /// -2*eps = S(r) + nabla_r(F(r)) in Galtier & Banerjee (2011)
                                           ///  (using the same containers here, so the transverse part is supposed to vanish, see below)
                            {
                                double rho       = (double)data1[DENS][cellindex1];
                                double rho_prim  = (double)data2[DENS][cellindex2];
                                double ux        = (double)data1[VELX][cellindex1];
                                double uy        = (double)data1[VELY][cellindex1];
                                double uz        = (double)data1[VELZ][cellindex1];
                                double ux_prim   = (double)data2[VELX][cellindex2];
                                double uy_prim   = (double)data2[VELY][cellindex2];
                                double uz_prim   = (double)data2[VELZ][cellindex2];
                                double e         = log(rho); // assumes that c_s=1 and <rho>=1 !
                                double e_prim    = log(rho_prim);
                                double d_ux      = ux_prim - ux;
                                double d_uy      = uy_prim - uy;
                                double d_uz      = uz_prim - uz;
                                double d_rhoux   = rho_prim*ux_prim - rho*ux;
                                double d_rhouy   = rho_prim*uy_prim - rho*uy;
                                double d_rhouz   = rho_prim*uz_prim - rho*uz;
                                double d_rho     = rho_prim - rho;
                                double d_e       = e_prim - e;
                                double d_rho_bar = 0.5 * (rho + rho_prim);
                                double d_e_bar   = 0.5 * (e + e_prim);
                                double square_brackets = 0.5*(d_rhoux*d_ux+d_rhouy*d_uy+d_rhouz*d_uz) + d_rho*d_e - d_rho_bar;
                                dv[X] = square_brackets * d_ux + d_e_bar * d_rhoux;
                                dv[Y] = square_brackets * d_uy + d_e_bar * d_rhouy;
                                dv[Z] = square_brackets * d_uz + d_e_bar * d_rhouz;
                                break;
                            }
                            case GB11S: /// gb11 S(r); Exact equation (11) for S(r):
                                        /// -2*eps = S(r) + nabla_r(F(r)) in Galtier & Banerjee (2011)
                            {
                                double rho       = data1[DENS][cellindex1];
                                double rho_prim  = data2[DENS][cellindex2];
                                double ux        = data1[VELX][cellindex1];
                                double uy        = data1[VELY][cellindex1];
                                double uz        = data1[VELZ][cellindex1];
                                double ux_prim   = data2[VELX][cellindex2];
                                double uy_prim   = data2[VELY][cellindex2];
                                double uz_prim   = data2[VELZ][cellindex2];
                                double divu      = data1[DIVV][cellindex1];
                                double divu_prim = data2[DIVV][cellindex2];
                                double e         = log(rho); // assumes that c_s=1 and <rho>=1 !
                                double e_prim    = log(rho_prim);
                                double E         = rho      * ( 0.5*(ux*ux+uy*uy+uz*uz) + e );
                                double E_prim    = rho_prim * ( 0.5*(ux_prim*ux_prim+uy_prim*uy_prim+uz_prim*uz_prim) + e_prim );
                                double uu_prim   = ux*ux_prim+uy*uy_prim+uz*uz_prim;
                                double R         = rho      * ( 0.5*uu_prim + e_prim );
                                double R_tilde   = rho_prim * ( 0.5*uu_prim + e );
                                dv[X] = divu_prim * (R - E) + divu * (R_tilde - E_prim);
                                break;
                            }
                            case VELDSETSQ: /// d(vel)*d(dset)^2 (any dataset name supplied by the user via -dset)
                            {
                                // dset increment(s)
                                double ddsetx = 0.0, ddsety = 0.0, ddsetz = 0.0;
                                if (data1[DSETX] != 0) ddsetx = (double)data2[DSETX][cellindex2] - (double)data1[DSETX][cellindex1];
                                if (data1[DSETY] != 0) ddsety = (double)data2[DSETY][cellindex2] - (double)data1[DSETY][cellindex1];
                                if (data1[DSETZ] != 0) ddsetz = (double)data2[DSETZ][cellindex2] - (double)data1[DSETZ][cellindex1];
                                // square of dset increment (note that we add up all the components of dset in sq; other combinations may be possible though)
                                double ddsetsq = ddsetx*ddsetx + ddsety*ddsety + ddsetz*ddsetz;
                                // combine d(vel) * d(dset)^2
                                for (unsigned int dir=X; dir<=Z; dir++) {
                                    v1[dir] = ddsetsq*(double)data1[VELX+dir][cellindex1];
                                    v2[dir] = ddsetsq*(double)data2[VELX+dir][cellindex2];
                                }
                                break;
                            }
                            default:
                            {
                                if (MyPE==0) cout << "ComputeSFsOrCFsFunctions:  something is wrong with the structure/correlaton function type! Exiting." << endl;
                                MPI_Finalize();
                                break;
                            }
                        } // end: switch(RequestedTypes[t])

                        if (compute_sf_not_cf) { // for structure functions
                            // compute data increment dv for all types, except some special types that have dv already filled above
                            if ((RequestedTypes[t] != GB11FLUX) && (RequestedTypes[t] != GB11S)) {
                                for (unsigned int dir=X; dir<=Z; dir++) dv[dir] = v2[dir] - v1[dir];
                            }
                            if ((dv[Y] != 0.0) || (dv[Z] != 0.0)) { /// decomposition into transverse and longitudinal parts
                                incr_long[t] = ScalarProduct(dv, dru); // longitudinal component (signed quantity)
                                incr_trsv[t] = sqrt((dv[X]*dv[X] + dv[Y]*dv[Y] + dv[Z]*dv[Z]) - incr_long[t]*incr_long[t]); /// Pythagoras (trsv SF component is always > 0)
                            }
                            else { /// no decomposition
                                incr_long[t] = dv[X];
                                incr_trsv[t] = 0.0;
                            }
                        }
                        else { // for correlation functions
                            if ((v1[Y] != 0.0) || (v1[Z] != 0.0) || (v2[Y] != 0.0) || (v2[Z] != 0.0)) { /// decomposition into transverse and longitudinal parts
                                vector<double> long_comp_1 = GetLongComp(v1, dru); // get longitudial vector component of v1
                                vector<double> long_comp_2 = GetLongComp(v2, dru); // get longitudial vector component of v2
                                vector<double> trsv_comp_1 = GetTrsvComp(v1, dru); // get transverse vector component of v1
                                vector<double> trsv_comp_2 = GetTrsvComp(v2, dru); // get transverse vector component of v2
                                incr_long[t] = ScalarProduct(long_comp_1, long_comp_2); // multiply (scalar product) for long (signed quantity)
                                incr_trsv[t] = ScalarProduct(trsv_comp_1, trsv_comp_2); // multiply (scalar product) for trsv (signed quantity)
                            }
                            else { /// no decomposition
                                incr_long[t] = v1[X]*v2[X];
                                incr_trsv[t] = 0.0;
                            }
                        }
                        if (take_abs) {
                            incr_long[t] = abs(incr_long[t]);
                            incr_trsv[t] = abs(incr_trsv[t]);
                        }

                    } // end: types

                    /// add to the appropriate bin (nested intervals)
                    int bin1 = 0; int bin2 = NumberOfBins; int bin = 0;
                    while ((bin2 - bin1) > 1)
                    {
                        bin = bin1 + (bin2 - bin1)/2;
                        if (distance < grid[bin])
                            bin2 = bin;
                        else
                            bin1 = bin;
                    }
                    bin = bin1;

                    /// compute higher order structure functions (be cautious with numerics);
                    /// in case of correlation functions, we only do order 1, so the code below
                    /// works for both 'sf' and 'cf'
                    for (int i = 0; i < MaxStructureFunctionOrder; i++)
                      for (int t = 0; t < NumTypes; t++)
                      {
                        incr_long_pow[t] = pow(incr_long[t],(double)(i+1));
                        incr_trsv_pow[t] = pow(incr_trsv[t],(double)(i+1));
                        int BufIndex = bin*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
                        if (TotVol*abs(incr_long_pow[t]/buf2_binsum_long[BufIndex]) > numeric_epsilon)
                        {
                            buf2_bin_counter_long[BufIndex] += TotVol;
                            buf2_binsum_long     [BufIndex] += TotVol*incr_long_pow[t];
                        }
                        else // immediately reduce to buf1 and clear buf2
                        {
                            buf1_bin_counter_long          [BufIndex] += buf2_bin_counter_long[BufIndex];
                            buf1_binsum_long               [BufIndex] += buf2_binsum_long     [BufIndex];
                            buf1_numeric_error_counter_long[BufIndex]++;
                            buf2_bin_counter_long          [BufIndex] = TotVol;
                            buf2_binsum_long               [BufIndex] = TotVol*incr_long_pow[t];
                        }
                        if (TotVol*abs(incr_trsv_pow[t]/buf2_binsum_trsv[BufIndex]) > numeric_epsilon)
                        {
                            buf2_bin_counter_trsv[BufIndex] += TotVol;
                            buf2_binsum_trsv     [BufIndex] += TotVol*incr_trsv_pow[t];
                        }
                        else // immediately reduce to buf1 and clear buf2
                        {
                            buf1_bin_counter_trsv          [BufIndex] += buf2_bin_counter_trsv[BufIndex];
                            buf1_binsum_trsv               [BufIndex] += buf2_binsum_trsv     [BufIndex];
                            buf1_numeric_error_counter_trsv[BufIndex]++;
                            buf2_bin_counter_trsv          [BufIndex] = TotVol;
                            buf2_binsum_trsv               [BufIndex] = TotVol*incr_trsv_pow[t];
                        }
                      }
                    /// ==========================================

                } // end: loop over cells in 2

            } // end: loop over cells in 1

            // clean
            for (int i = 0; i < MaxNumDsets; i++) if (data2[i]) delete [] data2[i];

        } // end: loop over all blocks

        // clean
        for (int i = 0; i < MaxNumDsets; i++) if (data1[i]) delete [] data1[i];

    } // end: loop over my blocks

    /// reduce the rest of buf2 to buf1
    for (int BufIndex = 0; BufIndex < NumBufferIndices; BufIndex++)
    {
        buf1_bin_counter_long[BufIndex] += buf2_bin_counter_long[BufIndex];
        buf1_binsum_long     [BufIndex] += buf2_binsum_long     [BufIndex];
        buf1_bin_counter_trsv[BufIndex] += buf2_bin_counter_trsv[BufIndex];
        buf1_binsum_trsv     [BufIndex] += buf2_binsum_trsv     [BufIndex];
        // set all empty bins to 0
        if (abs(buf1_bin_counter_long[BufIndex]) <= 2*numeric_epsilon) buf1_bin_counter_long[BufIndex] = 0;
        if (abs(buf1_binsum_long     [BufIndex]) <= 2*numeric_epsilon) buf1_binsum_long     [BufIndex] = 0;
        if (abs(buf1_bin_counter_trsv[BufIndex]) <= 2*numeric_epsilon) buf1_bin_counter_trsv[BufIndex] = 0;
        if (abs(buf1_binsum_trsv     [BufIndex]) <= 2*numeric_epsilon) buf1_binsum_trsv     [BufIndex] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (MyPE==0) cout << "ComputeSFsOrCFsFunctions: done. Now reducing and output..." << endl;

    /// init MPI reduction buffers
    double *bin_counter_long           = new double[NumBufferIndices];
    double *bin_counter_trsv           = new double[NumBufferIndices];
    double *struc_funct_binsum_long    = new double[NumBufferIndices];
    double *struc_funct_binsum_trsv    = new double[NumBufferIndices];
    long   *numeric_error_counter_long = new long  [NumBufferIndices];
    long   *numeric_error_counter_trsv = new long  [NumBufferIndices];
    for (int BufIndex = 0; BufIndex < NumBufferIndices; BufIndex++)
    {
        bin_counter_long          [BufIndex] = 0;
        bin_counter_trsv          [BufIndex] = 0;
        struc_funct_binsum_long   [BufIndex] = 0;
        struc_funct_binsum_trsv   [BufIndex] = 0;
        numeric_error_counter_long[BufIndex] = 0;
        numeric_error_counter_trsv[BufIndex] = 0;
    }

    /// Sum up CPU contributions
    MPI_Allreduce(buf1_bin_counter_long, bin_counter_long, NumBufferIndices, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_bin_counter_trsv, bin_counter_trsv, NumBufferIndices, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_binsum_long, struc_funct_binsum_long, NumBufferIndices, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_binsum_trsv, struc_funct_binsum_trsv, NumBufferIndices, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_numeric_error_counter_long, numeric_error_counter_long, NumBufferIndices, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_numeric_error_counter_trsv, numeric_error_counter_trsv, NumBufferIndices, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    /// clean-up
    delete[] buf1_bin_counter_long; buf1_bin_counter_long = 0;
    delete[] buf1_bin_counter_trsv; buf1_bin_counter_trsv = 0;
    delete[] buf1_binsum_long; buf1_binsum_long = 0;
    delete[] buf1_binsum_trsv; buf1_binsum_trsv = 0;
    delete[] buf1_numeric_error_counter_long; buf1_numeric_error_counter_long = 0;
    delete[] buf1_numeric_error_counter_trsv; buf1_numeric_error_counter_trsv = 0;
    delete[] buf2_bin_counter_long; buf2_bin_counter_long = 0;
    delete[] buf2_bin_counter_trsv; buf2_bin_counter_trsv = 0;
    delete[] buf2_binsum_long; buf2_binsum_long = 0;
    delete[] buf2_binsum_trsv; buf2_binsum_trsv = 0;

    /// divide binsum by bin_counter
    for (int BufIndex = 0; BufIndex < NumBufferIndices; BufIndex++)
    {
        if (bin_counter_long[BufIndex] > 0)
            struc_funct_binsum_long[BufIndex] /= bin_counter_long[BufIndex];
        else
            struc_funct_binsum_long[BufIndex] = 0.0;
        if (bin_counter_trsv[BufIndex] > 0)
            struc_funct_binsum_trsv[BufIndex] /= bin_counter_trsv[BufIndex];
        else
            struc_funct_binsum_trsv[BufIndex] = 0.0;
    }

    /// print out useful information and output
    if (MyPE==0)
    {
        vector< vector<double> > ReSumCounter_long(NumTypes, vector<double>(MaxStructureFunctionOrder));
        vector< vector<double> > ReSumCounter_trsv(NumTypes, vector<double>(MaxStructureFunctionOrder));
        for (int i = 0; i < MaxStructureFunctionOrder; i++)
          for (int t = 0; t < NumTypes; t++)
          {
            ReSumCounter_long[t][i] = 0;
            ReSumCounter_trsv[t][i] = 0;
          }
        for (int t = 0; t < NumTypes; t++)
        {
         cout << endl << "ComputeSFsOrCFsFunctions:  *********************** statistics and summary for >>> " <<
                OutfileStringForType[RequestedTypes[t]] << " <<< structure/correlation functions. ********************" << endl;
         for (int b = 0; b < NumberOfBins; b++)
         {
          cout << "bin_counter_long   [" << setw(3) << b << "]=";
          for (int i = 0; i < MaxStructureFunctionOrder; i++)
          {
            int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
            cout << setw(14) << bin_counter_long[BufIndex];
            ReSumCounter_long[t][i] += bin_counter_long[BufIndex];
          }
          cout << endl;
          cout << "error_counter_long [" << setw(3) << b << "]=";
          for (int i = 0; i < MaxStructureFunctionOrder; i++)
          {
            int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
            cout << setw(14) << numeric_error_counter_long[BufIndex];
          }
          cout << endl;
          cout << "bin_counter_trsv   [" << setw(3) << b << "]=";
          for (int i = 0; i < MaxStructureFunctionOrder; i++)
          {
            int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
            cout << setw(14) << bin_counter_trsv[BufIndex];
            ReSumCounter_trsv[t][i] += bin_counter_trsv[BufIndex];
          }
          cout << endl;
          cout << "error_counter_trsv [" << setw(3) << b << "]=";
          for (int i = 0; i < MaxStructureFunctionOrder; i++)
          {
            int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
            cout << setw(14) << numeric_error_counter_trsv[BufIndex];
          }
          cout << endl;
         }
         for (int i = 0; i < MaxStructureFunctionOrder; i++)
         {
            cout << "Resummed total number of samples for longitudinal structure/correlation functions [order="<<setw(2)<<i+1<<"] = " << ReSumCounter_long[t][i] << endl;
            cout << "Resummed total number of samples for transverse   structure/correlation functions [order="<<setw(2)<<i+1<<"] = " << ReSumCounter_trsv[t][i] << endl;
         }
        }

        /// prepare OutputFileHeader
        string fn_typ = "_SF";
        if (!compute_sf_not_cf) fn_typ = "_CF";
        OutputFileHeader.resize(0);
        stringstream dummystream, sscol, ssorder;
        dummystream << setw(30) << left << "#00_BinIndex";
        dummystream << setw(30) << left << "#01_GridStag";
        dummystream << setw(30) << left << "#02_Grid";
        for (int i = 0; i < MaxStructureFunctionOrder; i++)
        {
          ssorder=stringstream(); ssorder<<setw(2)<<setfill('0')<<(i+1);
          sscol=stringstream(); sscol<<setw(2)<<setfill('0')<<(3+4*i+0);
          dummystream<<setw(30)<<left<<"#"+sscol.str()+"_NP(long,order="+ssorder.str()+")";
          sscol=stringstream(); sscol<<setw(2)<<setfill('0')<<(3+4*i+1);
          dummystream<<setw(30)<<left<<"#"+sscol.str()+fn_typ+"(long,order="+ssorder.str()+")";
          sscol=stringstream(); sscol<<setw(2)<<setfill('0')<<(3+4*i+2);
          dummystream<<setw(30)<<left<<"#"+sscol.str()+"_NP(trsv,order="+ssorder.str()+")";
          sscol=stringstream(); sscol<<setw(2)<<setfill('0')<<(3+4*i+3);
          dummystream<<setw(30)<<left<<"#"+sscol.str()+fn_typ+"(trsv,order="+ssorder.str()+")";
        }
        OutputFileHeader.push_back(dummystream.str());
        dummystream.clear();
        dummystream.str("");
        /// OUTPUT in files (needs to be here because of the different types)
        string OutputFilename = "";
        for (int t = 0; t < NumTypes; t++)
        {
            /// resize and fill WriteOutTable
            WriteOutTable.resize(NumberOfBins); /// structure/correlation function output has NumberOfBins lines
            for (unsigned int i = 0; i < WriteOutTable.size(); i++)
                WriteOutTable[i].resize(3+4*MaxStructureFunctionOrder); /// structure/correlation function output has ... columns
            for (int b = 0; b < NumberOfBins; b++)
            {
                WriteOutTable[b][0] = b;
                WriteOutTable[b][1] = grid_stag[b];
                WriteOutTable[b][2] = grid     [b];
                for (int i = 0; i < MaxStructureFunctionOrder; i++)
                {
                    int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
                    WriteOutTable[b][3+4*i+0] = bin_counter_long       [BufIndex];
                    WriteOutTable[b][3+4*i+1] = struc_funct_binsum_long[BufIndex];
                    WriteOutTable[b][3+4*i+2] = bin_counter_trsv       [BufIndex];
                    WriteOutTable[b][3+4*i+3] = struc_funct_binsum_trsv[BufIndex];
                }
            }
            string outfilestringfortype = OutfileStringForType[RequestedTypes[t]];
            if ((outfilestringfortype == "dset") || (outfilestringfortype == "veldsetsq")) {
                outfilestringfortype += "_"+DsetName[0];
                for (int i=1; i<3; i++)
                    if (DsetName[i] != "") outfilestringfortype += "_"+DsetName[i];
            }
            string fn_typ = "_sf_";
            if (!compute_sf_not_cf) fn_typ = "_cf_";
            OutputFilename = OutputPath+"/"+inputfile+fn_typ+outfilestringfortype+".dat";
            WriteOutAnalysedData(OutputFilename);

        } // end: loop over types

    } // end: MyPE==0

    /// clean-up
    delete[] bin_counter_long; bin_counter_long = 0;
    delete[] bin_counter_trsv; bin_counter_trsv = 0;
    delete[] struc_funct_binsum_long; struc_funct_binsum_long = 0;
    delete[] struc_funct_binsum_trsv; struc_funct_binsum_trsv = 0;
    delete[] numeric_error_counter_long; numeric_error_counter_long = 0;
    delete[] numeric_error_counter_trsv; numeric_error_counter_trsv = 0;

    if (MyPE==0) cout << "ComputeSFsOrCFsFunctions: exiting." << endl;
}


// function call to gg.ReadBlockVar ot gg.ReadBlockVarPB, depending on whether grid is uniform, extracted, or AMR
float * ReadBlock(const int block, const string datasetname)
{
    float *tmp = 0;
    if (GridType == 'U' || GridType == 'E')
        tmp = gg.ReadBlockVarPB(block, datasetname);
    else
        tmp = gg.ReadBlockVar(block, datasetname);
    return tmp;
}


/** -------------------- WriteOutAnalysedData ---------------------------------
 **  Writes out a variable table of data and a FileHeader to a specified file
 ** --------------------------------------------------------------------------- */
void WriteOutAnalysedData(const string OutputFilename)
{
    /// open output file
    ofstream Outputfile(OutputFilename.c_str());

    /// check for file
    if (!Outputfile)
    {
        cout << "WriteOutAnalysedData:  File system error. Could not create '" << OutputFilename.c_str() << "'."<< endl;
        MPI_Finalize();
    }
    /// write data to output file
    else
    {
        cout << "WriteOutAnalysedData:  Writing output file '" << OutputFilename.c_str() << "' ..." << endl;

        for (unsigned int row = 0; row < OutputFileHeader.size(); row++)
        {
            Outputfile << setw(61) << left << OutputFileHeader[row] << endl;      /// header
            if (Debug) cout << setw(61) << left << OutputFileHeader[row] << endl;
        }
        for (unsigned int row = 0; row < WriteOutTable.size(); row++)                  /// data
        {
            for (unsigned int col = 0; col < WriteOutTable[row].size(); col++)
            {
                Outputfile << scientific << setw(30) << left << setprecision(8) << WriteOutTable[row][col];
                if (Debug) cout << scientific << setw(30) << left << setprecision(8) << WriteOutTable[row][col];
            }
            Outputfile << endl; if (Debug) cout << endl;
        }

        Outputfile.close();
        Outputfile.clear();

        cout << "WriteOutAnalysedData:  done!" << endl;
    }
} /// =======================================================================


/** ------------------------- ParseInputs ----------------------------
 **  Parses the command line Arguments
 ** ------------------------------------------------------------------ */
int ParseInputs(const vector<string> Argument)
{
    // check for valid options
    vector<string> valid_options;
    valid_options.push_back("-n");
    valid_options.push_back("-types");
    valid_options.push_back("-dsets");
    valid_options.push_back("-ncells_pb");
    valid_options.push_back("-max_sf_order");
    valid_options.push_back("-take_abs");
    valid_options.push_back("-opath");
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
    /// read tool specific options
    if (Argument.size() < 3)
    {
        if (MyPE==0) { cout << endl << "ParseInputs: Invalid number of arguments." << endl; }
        return -1;
    }
    /// determine whether we compute structure function (sf) or correlation function (cf)
    string sf_or_cf_str = Argument[1];
    if (sf_or_cf_str == "sf") {
        compute_sf_not_cf = true;
    } else if (sf_or_cf_str == "cf") {
        compute_sf_not_cf = false;
    } else {
        if (MyPE==0) cout << endl << "ParseInputs: Error: need to specify 'sf' or 'cf' to either compute structure functions (sf) or correlation functions (cf)." << endl;
        return -1;
    }
    /// input file
    inputfile = Argument[2];
    /// parse optional arguments
    stringstream dummystream;
    for (unsigned int i = 3; i < Argument.size(); i++)
    {
        if (Argument[i] != "" && Argument[i] == "-n")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> n_samples; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-types")
        {
            for (unsigned int j = i+1; j < Argument.size(); j++) {
                if (Argument[j].at(0) != '-') {
                    int req_type;
                    dummystream << Argument[j]; dummystream >> req_type; dummystream.clear();
                    RequestedTypes.push_back(req_type);
                } else break;
            }
        }
        if (Argument[i] != "" && Argument[i] == "-dsets")
        {
            int ncomp = 0;
            for (unsigned int j = i+1; j < Argument.size(); j++) {
                if (Argument[j].at(0) != '-') DsetName[ncomp++] = Argument[j]; else break;
            }
        }
        if (Argument[i] != "" && Argument[i] == "-ncells_pb")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> ncells_pseudo_blocks; dummystream.clear();
                ncells_pseudo_blocks_set = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-max_sf_order")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> MaxStructureFunctionOrder; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-take_abs")
        {
            take_abs = true;
        }
        if (Argument[i] != "" && Argument[i] == "-opath")
        {
            if (Argument.size()>i+1) OutputPath = Argument[i+1]; else return -1;
        }

    } // loop over all args

    // correlation functions special case
    if (!compute_sf_not_cf) MaxStructureFunctionOrder = 1;

    // if user did not specify any types, we set VELS as the (only) default
    if (RequestedTypes.size() == 0) RequestedTypes.push_back(VELS);

    // print requested types and error check on DSETTYPE type
    if (MyPE==0) {
        cout << "ParseInputs: Requested structure/correlation function types:" << endl;
        for (unsigned int i = 0; i < RequestedTypes.size(); i++){
            cout << "  -> "<<setw(2)<<RequestedTypes[i]<<": "<<OutfileStringForType[RequestedTypes[i]]<<endl;
            if ((RequestedTypes[i] == DSETTYPE || RequestedTypes[i] == VELDSETSQ) && DsetName[0] == "") {
                cout << "ParseInputs: Error. If using -types 0 or 15, you must specify -dset ..." << endl;
                return -1;
            }
        }
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
        << " sf_cf <sf,cf> <filename> [<OPTIONS>]" << endl << endl
        << "   <sf, cf> : specify either 'sf' for structure function or 'cf' for correlation function." << endl
        << "   <OPTIONS>:           " << endl
        << "     -n <num_samples>        : total number of sampling pairs" << endl
        << "     -types <type(s)>        : requested structure/correlation function type(s) by sequence of numbers; available types:" << endl;
        for (int i = 0; i < MaxNumTypes; i++) cout << "                               "<<setw(2)<<i<<" : "+OutfileStringForType[i]<<endl;
        cout << "     -dsets <datasetname(s)> : for type 0, specify dataset(s) to process (up to 3 components)" << endl
        << "     -ncells_pb <num_cells>  : number of cells in pseudo blocks (default: as in file)" << endl
        << "     -max_sf_order <max_o>   : maximum structure function order; only applies for 'sf' (default: "<<MaxStructureFunctionOrder<<")" << endl
        << "     -take_abs               : take absolute value of data increment prior to accumulation" << endl
        << "     -opath <path>           : specify output path" << endl
        << endl
        << "Example: sf_cf sf DF_hdf5_plt_cnt_0020 -n 1e6 -types 1 8"
        << endl << endl;
    }
}
