///
///  Fourier spectra MPI version (single-precision) for FlashGG
///
///  written by Christoph Federrath, 2012-2025

#include "mpi.h" /// MPI lib
#include <iostream>
#include <iomanip> /// for io manipulations
#include <sstream> /// stringstream
#include <fstream> /// for filestream operation
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fftw3-mpi.h> /// Fast Fourier Transforms MPI version
#include "../Libs/FlashGG.h" /// Flash General Grid class

// constants
int NDIM = 3;
using namespace std;
enum {X, Y, Z};
static const bool Debug = false;
static const int FAILURE = 0;
static const int MAX_NUM_BINS = 10048;
static const double pi = 3.14159265358979323846;

// MPI stuff
int MyPE = 0, NPE = 1;

// for FFTW
fftwf_complex *fft_data_x, *fft_data_y, *fft_data_z, *fft_data_wq;
fftwf_plan fft_plan_x, fft_plan_y, fft_plan_z, fft_plan_wq;

// arguments
string inputfile = "";
double dset_scale = 1.0;
double dens_scale = 1.0;
double vels_scale = 1.0;
double mags_scale = 1.0;
bool weighting = false;
string weight_name = "";

FlashGG gg; // grid class handler

vector<string> DsetName(3, ""); // for DSETTYPE (type 0); user supplies custom dataset via -dset (max 3 components)

// spectra types
static const int MaxNumTypes = 14;
enum {DSETTYPE, VELS, MAGS, MACH, MSMACH, AMACH, VALF, SQRTRHO, RHO3, RHOV, VARRHO, VARLNRHO, RHO, LNRHO};
int DecomposedType[MaxNumTypes] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
string OutfileStringForType[MaxNumTypes] = {"dset", "vels", "mags", "mach", "msmach", "amach", "valf", "sqrtrho", "rho3", "rhov",
                                            "varrho", "varlnrho", "rho", "lnrho"};
vector<int> RequestedTypes;
vector<string> RequiredDatasets;

// FLASH datasets that can be used here
static const int MaxNumDsets = 11;
enum {DSETX, DSETY, DSETZ, DENS, VELX, VELY, VELZ, MAGX, MAGY, MAGZ, TEMP};
string DsetNames[MaxNumDsets] = {"", "", "", "dens", "velx", "vely", "velz", "magx", "magy", "magz", "temp"};

// for output
vector<string> OutputFileHeader;
vector< vector<double> > WriteOutTable;

/// forward function
void SetupTypes(void);
vector<int> InitFFTW(const vector<int> nCells);
float * ReadParallel(const string datasetname, vector<int> MyInds);
void SwapMemOrder(float * const data, const vector<int> N);
vector<int> InitFFTW(const vector<int> nCells);
void AssignDataToFFTWContainer(const int type, const vector<int> N, const long ntot_local, vector<float *> data_ptrs, vector<string> data_nams);
void ComputeSpectrum(const vector<int> Dim, const vector<int> MyInds, const bool decomposition);
void WriteOutAnalysedData(const string OutputFilename);
void Normalize(float * const data_array, const long n, const double norm);
double Mean(const float * const data, const long size);
void Window(float* const data_array, const int nx, const int ny, const int nz);
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
    if (MyPE==0) cout<<"=== spectra === MPI num procs: "<<NPE<<endl;

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

    /// get the dataset dimensions
    gg = FlashGG(inputfile);
    vector<int> N = gg.GetN();

    // signal dimensionality of the data
    if (N[Z]==1) NDIM = 2;

    /// allocate FFTW containers and create FTTW plan
    vector<int> MyInds = InitFFTW(N);
    long ntot_local = MyInds[1]*N[Y]*N[Z];
    if (Debug) cout<<"["<<MyPE<<"] MyInds: "<<MyInds[0]<<" "<<MyInds[1]<<" ntot_local="<<ntot_local<<endl;

    /// parallelisation / decomposition check
    int wrong_decompostion = 0, wrong_decompostion_red = 0;
    if (MyInds[1] != 0) { if (N[X] % MyInds[1] != 0) wrong_decompostion = 1;
    } else { wrong_decompostion = 1; }
    MPI_Allreduce(&wrong_decompostion, &wrong_decompostion_red, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (wrong_decompostion_red > 0) {
        if (MyPE==0) cout<<"Error: Number of cores is not multiple of N[X]."<<endl;
        MPI_Finalize();
        return 0;
    }

    // setup pseudo blocks to match FFTW data structure
    vector<int> ncells_pb(3);
    ncells_pb[X] = MyInds[1]; // parallel direction
    ncells_pb[Y] = N[Y];
    ncells_pb[Z] = N[Z];
    gg.SetupPseudoBlocks(ncells_pb);
    if (MyPE==0) gg.PrintInfo();

    /// setup spectra types
    SetupTypes();

    /// read all required datasets
    if (MyPE==0) cout<<"reading data from disk..."<<endl;
    vector<float *> data_ptrs;
    vector<string> data_nams;
    for (unsigned int i=0; i<RequiredDatasets.size(); i++) {
        data_ptrs.push_back(ReadParallel(RequiredDatasets[i], MyInds));
        data_nams.push_back(RequiredDatasets[i]);
        if (MyPE==0) cout<<data_nams[i]+" read."<<endl;
    }

    /// read weighting dataset
    if (weighting) {
        if (find(RequiredDatasets.begin(), RequiredDatasets.end(), weight_name) == RequiredDatasets.end()) {
            data_ptrs.push_back(ReadParallel(weight_name, MyInds));
            data_nams.push_back(weight_name);
            if (MyPE==0) cout<<data_nams.back()+" read."<<endl;
        }
        if (MyPE==0) cout<<">>>>>>>>>>>> Using '"<<weight_name<<"' for weighting. <<<<<<<<<<<<"<<endl;
    }

    /// normalizations
    if (dset_scale != 1.0) {
        if (MyPE==0) cout << "applying dset scale = " << dset_scale << endl;
        for (int i=0; i<3; i++) {
            int ind = -1;
            if (DsetNames[DSETX+i] != "") ind = find(data_nams.begin(), data_nams.end(), DsetNames[DSETX+i]) - data_nams.begin();
            if (ind > -1) Normalize(data_ptrs[ind], ntot_local, dset_scale);
        }
    }
    if (dens_scale != 1.0) {
        if (MyPE==0) cout << "applying density scale = " << dens_scale << endl;
        int ind = find(data_nams.begin(), data_nams.end(), DsetNames[DENS]) - data_nams.begin();
        Normalize(data_ptrs[ind], ntot_local, dens_scale);
    }
    if (vels_scale != 1.0) {
        if (MyPE==0) cout << "applying velocity scale = " << vels_scale << endl;
        int ind = find(data_nams.begin(), data_nams.end(), DsetNames[VELX]) - data_nams.begin();
        for (int i=0; i<3; i++) Normalize(data_ptrs[ind+i], ntot_local, vels_scale);
    }
    if (mags_scale != 1.0) {
        if (MyPE==0) cout << "applying magnetic field scale = " << mags_scale << endl;
        int ind = find(data_nams.begin(), data_nams.end(), DsetNames[MAGX]) - data_nams.begin();
        for (int i=0; i<3; i++) Normalize(data_ptrs[ind+i], ntot_local, mags_scale);
    }

    long endtime = time(NULL);
    int duration = endtime-starttime, duration_red = 0;
    if (Debug) cout << "["<<MyPE<<"] ****************** Local time for startup, allocation, and reading data = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << "****************** Global time for startup, allocation, and reading data = "<<duration_red<<"s ******************" << endl;

    string outfilename = "";
    string weight_str = "";

    if (weighting) {
        // fill weighting container
        AssignDataToFFTWContainer(-1, N, ntot_local, data_ptrs, data_nams);
        weight_str = "_wq_"+weight_name;
    }

    // for each requested spectrum type: assign data to FFTW containers, compute spectra, and write result to file
    for (unsigned int t=0; t<RequestedTypes.size(); t++) {
        AssignDataToFFTWContainer(RequestedTypes[t], N, ntot_local, data_ptrs, data_nams);
        ComputeSpectrum(N, MyInds, DecomposedType[RequestedTypes[t]]);
        if (MyPE==0) {
            string outfilestringfortype = OutfileStringForType[RequestedTypes[t]];
            if (outfilestringfortype == "dset") {
                outfilestringfortype += "_"+DsetName[0];
                for (int i=1; i<3; i++) if (DsetName[i] != "") outfilestringfortype += "_"+DsetName[i];
            }
            outfilename = inputfile+"_spect_"+outfilestringfortype;
            WriteOutAnalysedData(outfilename+weight_str+".dat");
        }
    }

    /// clean
    for (unsigned int i=0; i<data_ptrs.size(); i++) delete [] data_ptrs[i];

    fftwf_free(fft_data_x);
    fftwf_free(fft_data_y);
    if (NDIM==3) fftwf_free(fft_data_z);

    fftwf_destroy_plan(fft_plan_x);
    fftwf_destroy_plan(fft_plan_y);
    if (NDIM==3) fftwf_destroy_plan(fft_plan_z);
    fftwf_mpi_cleanup();

    endtime = time(NULL);
    duration = endtime-starttime; duration_red = 0;
    if (Debug) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << "****************** Global time to finish = "<<duration_red<<"s ******************" << endl;

    MPI_Finalize();
    return 0;

} // end main ==========================================================


/** --------------------------- SetupTypes ------------------------------------
 **  setup spectra types and dataset requirements
 ** --------------------------------------------------------------------------- */
void SetupTypes(void)
{
    /// setup general spectra type requirements (on reading datasets) and output file strings
    vector< vector<string> > RequiredDsetsForType(MaxNumTypes);
    for (int t = 0; t < MaxNumTypes; t++) {
        if (t == DSETTYPE) {
            for (int i=0; i<3; i++) {
                if (DsetName[i] != "") RequiredDsetsForType[t].push_back(DsetName[i]);
                DsetNames[i] = DsetName[i];
            }
            if (DsetName[1]!="" || DsetName[2]!="") DecomposedType[0] = 1;
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
        if ((t == SQRTRHO) || (t == RHO3) || (t == RHOV)) {
            RequiredDsetsForType[t].push_back(DsetNames[DENS]);
            for (int i=0; i<3; i++) RequiredDsetsForType[t].push_back(DsetNames[VELX+i]);
        }
        if ((t == VARRHO) || (t == VARLNRHO) || (t == RHO) || (t == LNRHO)) {
            RequiredDsetsForType[t].push_back(DsetNames[DENS]);
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


/// InitFFTW ==========================================================
vector<int> InitFFTW(const vector<int> nCells)
{
    const bool Debug = false;

    const ptrdiff_t N[3] = {nCells[X], nCells[Y], nCells[Z]};
    ptrdiff_t alloc_local = 0, local_n0 = 0, local_0_start = 0;

    fftwf_mpi_init();

    // get local data size and allocate
    if (NDIM==3) alloc_local = fftwf_mpi_local_size_3d(N[X], N[Y], N[Z], MPI_COMM_WORLD, &local_n0, &local_0_start);
    if (NDIM==2) alloc_local = fftwf_mpi_local_size_2d(N[X], N[Y], MPI_COMM_WORLD, &local_n0, &local_0_start);
    /// ALLOCATE
    if (Debug) cout<<"["<<MyPE<<"] Allocating fft_data_x..."<<endl;
    fft_data_x = fftwf_alloc_complex(alloc_local);
    if (Debug) cout<<"["<<MyPE<<"] Allocating fft_data_y..."<<endl;
    fft_data_y = fftwf_alloc_complex(alloc_local);
    if (NDIM==3) {
        if (Debug) cout<<"["<<MyPE<<"] Allocating fft_data_z..."<<endl;
        fft_data_z = fftwf_alloc_complex(alloc_local);
    }
    if (weighting) {
        if (Debug) cout<<"["<<MyPE<<"] Allocating fft_data_wq..."<<endl;
        fft_data_wq = fftwf_alloc_complex(alloc_local);
    }
    if (Debug) cout<<"["<<MyPE<<"] ...alloc done."<<endl;

    /// PLAN
    if (NDIM==3) {
        if (Debug) cout<<"["<<MyPE<<"] fft_plan_x..."<<endl;
        fft_plan_x = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_x, fft_data_x, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        if (Debug) cout<<"["<<MyPE<<"] fft_plan_y..."<<endl;
        fft_plan_y = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_y, fft_data_y, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        if (Debug) cout<<"["<<MyPE<<"] fft_plan_z..."<<endl;
        fft_plan_z = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_z, fft_data_z, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        if (weighting) {
            if (Debug) cout<<"["<<MyPE<<"] fft_plan_wq..."<<endl;
            fft_plan_wq = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_wq, fft_data_wq, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        }
    }
    if (NDIM==2) {
        if (Debug) cout<<"["<<MyPE<<"] fft_plan_x..."<<endl;
        fft_plan_x = fftwf_mpi_plan_dft_2d(N[X], N[Y], fft_data_x, fft_data_x, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        if (Debug) cout<<"["<<MyPE<<"] fft_plan_y..."<<endl;
        fft_plan_y = fftwf_mpi_plan_dft_2d(N[X], N[Y], fft_data_y, fft_data_y, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        if (weighting) {
            if (Debug) cout<<"["<<MyPE<<"] fft_plan_wq..."<<endl;
            fft_plan_wq = fftwf_mpi_plan_dft_2d(N[X], N[Y], fft_data_wq, fft_data_wq, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        }
    }
    if (Debug) cout<<"["<<MyPE<<"] ...plans done."<<endl;

    vector<int> ReturnVector(2);
    ReturnVector[0] = local_0_start;
    ReturnVector[1] = local_n0;

    return ReturnVector;

} /// ================================================================


/// ReadParallel ===============================================================
float * ReadParallel(const string datasetname, vector<int> MyInds)
{
    const bool Debug = false;

    vector<int> N = gg.GetN();

    /// decompose domain into pseudo blocks (PB)
    /// note that SetupPseudoBlocks() was called above,
    /// such that there is exactly 1 block per core
    vector<int> MyBlocks = gg.GetMyBlocks();

    if (Debug) {
        MPI_Barrier(MPI_COMM_WORLD);
        cout<<"["<<MyPE<<"] My blocks = ";
        for (unsigned int b=0; b<MyBlocks.size(); b++) cout<<MyBlocks[b]<<" ";
        cout<<endl<<"["<<MyPE<<"] My number of blocks = "<<MyBlocks.size()<<endl;
    }

    assert(MyBlocks.size() == 1);
    float * data_ptr = gg.ReadBlockVarPB(MyBlocks[0], datasetname); // read the 1 PB that contains this core's data

    if (Debug) {
        double sum = 0.0, sum_red = 0.0;
        for (long n = 0; n < MyInds[1]*N[Y]*N[Z]; n++) sum += data_ptr[n];
        MPI_Allreduce(&sum, &sum_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (MyPE==0) cout<<"before SwapMemOrder: sum(data_ptr of "+datasetname+")="<<sum_red<<endl;
    }

    /// SwapMemOrder
    vector<int> Dims(3); Dims[0] = MyInds[1]; Dims[1] = N[Y]; Dims[2] = N[Z];
    SwapMemOrder(data_ptr, Dims);

    if (Debug) {
        double sum = 0.0, sum_red = 0.0;
        for (long n = 0; n < MyInds[1]*N[Y]*N[Z]; n++) sum += data_ptr[n];
        MPI_Allreduce(&sum, &sum_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (MyPE==0) cout<<"after SwapMemOrder: sum(data_ptr of "+datasetname+")="<<sum_red<<endl;
    }

    return data_ptr;

} /// ==========================================================================


/// SwapMemOrder =====================================================
void SwapMemOrder(float * const data, const vector<int> N)
{
    const long ntot = N[0]*N[1]*N[2];
    float *tmp = new float[ntot];
    for (int i=0; i<N[0]; i++) for (int j=0; j<N[1]; j++) for (int k=0; k<N[2]; k++) {
        long ind1 = i*N[1]*N[2] + j*N[2] + k;
        long ind2 = k*N[1]*N[0] + j*N[0] + i;
        tmp[ind1] = data[ind2];
    }
    for (long i=0; i<ntot; i++) data[i] = tmp[i];
    delete [] tmp;

} /// ==============================================================


void AssignDataToFFTWContainer(const int type, const vector<int> N, const long ntot_local,
                               vector<float *> data_ptrs, vector<string> data_nams)
{

    // set everything to 0 at the start
    for (long n=0; n<ntot_local; n++) {
        fft_data_x[n][0] = 0.0; /// Real part x
        fft_data_x[n][1] = 0.0; /// Imaginary part x
        fft_data_y[n][0] = 0.0; /// Real part y
        fft_data_y[n][1] = 0.0; /// Imaginary part y
        if (NDIM==3) {
            fft_data_z[n][0] = 0.0; /// Real part z
            fft_data_z[n][1] = 0.0; /// Imaginary part z
        }
    }

    // assign weight
    if (type == -1) {
        int ind = find(data_nams.begin(), data_nams.end(), weight_name) - data_nams.begin();
        double mean_weig = Mean(data_ptrs[ind], ntot_local); // note that dens contains weig here; see call from MAIN
        for (long n=0; n<ntot_local; n++) {
            fft_data_wq[n][0] = data_ptrs[ind][n]/mean_weig - 1.0; /// Real part of weight
            fft_data_wq[n][1] = 0.0; /// Imaginary part of weight
        }
    }

    if (type == DSETTYPE) {
        int ind[3];
        for (int i=0; i<3; i++) {
            if (DsetNames[DSETX+i] != "") ind[i] = find(data_nams.begin(), data_nams.end(), DsetNames[DSETX+i]) - data_nams.begin();
            else ind[i] = -1;
        }
        for (long n=0; n<ntot_local; n++) {
            if (ind[0]!=-1) fft_data_x[n][0] = data_ptrs[ind[0]][n]; /// Real part x
            if (ind[1]!=-1) fft_data_y[n][0] = data_ptrs[ind[1]][n]; /// Real part y
            if ((NDIM==3) && (ind[2]!=-1)) fft_data_z[n][0] = data_ptrs[ind[2]][n]; /// Real part z
        }
    }

    if ((type == VELS) || (type == MAGS)) {
        int ind = -1;
        if (type == VELS)    ind = find(data_nams.begin(), data_nams.end(), DsetNames[VELX]) - data_nams.begin();
        if (type == MAGS)    ind = find(data_nams.begin(), data_nams.end(), DsetNames[MAGX]) - data_nams.begin();
        for (long n=0; n<ntot_local; n++) {
            fft_data_x[n][0] = data_ptrs[ind+0][n]; /// Real part x
            fft_data_y[n][0] = data_ptrs[ind+1][n]; /// Real part y
            if (NDIM==3) fft_data_z[n][0] = data_ptrs[ind+2][n]; /// Real part z
        }
    }

    if ((type == VARRHO) || (type == VARLNRHO) || (type == RHO) || (type == LNRHO)) {
        int ind = find(data_nams.begin(), data_nams.end(), DsetNames[DENS]) - data_nams.begin();
        if (type == VARRHO) {
            double mean_dens = Mean(data_ptrs[ind], ntot_local);
            if (MyPE==0) cout<<"AssignDataToFFTWContainer: mean dens = "<<mean_dens<<endl;
            for (long n=0; n<ntot_local; n++) fft_data_x[n][0] = data_ptrs[ind][n]-mean_dens; /// Real part x
        }
        if (type == VARLNRHO) {
            for (long n=0; n<ntot_local; n++) data_ptrs[ind][n] = log(data_ptrs[ind][n]); // turn into ln(rho/<rho>)
            double mean_lndens = Mean(data_ptrs[ind], ntot_local); // compute mean of ln(rho)
            if (MyPE==0) cout<<"AssignDataToFFTWContainer: mean log(dens) = "<<mean_lndens<<endl;
            for (long n=0; n<ntot_local; n++) data_ptrs[ind][n] = exp(data_ptrs[ind][n]); // turn back to normal rho (so we don't mess up any later calculations using dens)
            for (long n=0; n<ntot_local; n++) fft_data_x[n][0] = log(data_ptrs[ind][n])-mean_lndens; /// Real part x
        }
        if (type == RHO) {
            for (long n=0; n<ntot_local; n++) fft_data_x[n][0] = sqrt(data_ptrs[ind][n]); /// Real part x
        }
        if (type == LNRHO) {
            double mean_dens = Mean(data_ptrs[ind], ntot_local);
            if (MyPE==0) cout<<"AssignDataToFFTWContainer: mean dens = "<<mean_dens<<endl;
            for (long n=0; n<ntot_local; n++) fft_data_x[n][0] = log(data_ptrs[ind][n]/mean_dens); /// Real part x
        }
    }

    if ((type == RHO3) || (type == SQRTRHO) || (type == RHOV)) {
        int ind_dens = find(data_nams.begin(), data_nams.end(), DsetNames[DENS]) - data_nams.begin();
        int ind_velx = find(data_nams.begin(), data_nams.end(), DsetNames[VELX]) - data_nams.begin();
        double dens_weight = 0.0;
        for (long n=0; n<ntot_local; n++) {
            if (type == RHO3)    dens_weight = pow(data_ptrs[ind_dens][n], 1./3.);
            if (type == SQRTRHO) dens_weight = sqrt(data_ptrs[ind_dens][n]);
            if (type == RHOV)    dens_weight = data_ptrs[ind_dens][n];
            fft_data_x[n][0] = dens_weight * data_ptrs[ind_velx+0][n]; /// Real part x
            fft_data_y[n][0] = dens_weight * data_ptrs[ind_velx+1][n]; /// Real part y
            if (NDIM==3) fft_data_z[n][0] = dens_weight * data_ptrs[ind_velx+2][n]; /// Real part z
        }
    }

    /// error check
    if (type != -1) {
        double sum = 0.0, sum_red = 0.0;
        double ntot = (double)(N[X])*(double)(N[Y])*(double)(N[Z]);
        if (NDIM==3)
            for (long n = 0; n < ntot_local; n++)
                sum += fft_data_x[n][0]*fft_data_x[n][0]+fft_data_y[n][0]*fft_data_y[n][0]+fft_data_z[n][0]*fft_data_z[n][0];
        if (NDIM==2)
            for (long n = 0; n < ntot_local; n++)
                sum += fft_data_x[n][0]*fft_data_x[n][0]+fft_data_y[n][0]*fft_data_y[n][0];
        if (Debug) cout << "["<<MyPE<<"] Local sum in physical space = " << sum/ntot << endl;
        MPI_Allreduce(&sum, &sum_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (MyPE==0) cout << "Global sum in physical space before weighting ("<<type<<") = " << sum_red/ntot << endl;
    }

} /// ==============================================================


/** --------------------- ComputeSpectrum ----------------------------
 **  computes total, transversal, longitudinal spectrum functions
 ** ------------------------------------------------------------------ */
void ComputeSpectrum(const vector<int> Dim, const vector<int> MyInds, const bool decomposition)
{
    const bool Debug = false;
    long starttime = time(NULL);

    if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: entering." << endl;

    if (NDIM==3) {
        if ((Dim[X]!=Dim[Y])||(Dim[X]!=Dim[Z])||(Dim[Y]!=Dim[Z])) {
            cout << "Spectra can only be obtained from cubic datasets (Nx=Ny=Nz)." << endl;
            exit(FAILURE);
        }
    }
    if (NDIM==2) {
        if (Dim[X]!=Dim[Y]) {
            cout << "Spectra can only be obtained from quadratic datasets (Nx=Ny)." << endl;
            exit(FAILURE);
        }
    }

    /////////// EXECUTE PLAN
    if (decomposition) {
        fftwf_execute(fft_plan_x);
        fftwf_execute(fft_plan_y);
        if (NDIM==3) fftwf_execute(fft_plan_z);
    } else {
        fftwf_execute(fft_plan_x);
    }
    if (weighting) {
        fftwf_execute(fft_plan_wq);
    }

    /// general constants
    const long N = Dim[X]; /// assume a cubic (square in 2D) box !
    long LocalNumberOfDataPoints = 0;
    long TotalNumberOfDataPoints = 0;
    if (NDIM==3) {
        LocalNumberOfDataPoints = N*N*MyInds[1];
        TotalNumberOfDataPoints = N*N*N;
    }
    if (NDIM==2) {
        LocalNumberOfDataPoints = N*MyInds[1];
        TotalNumberOfDataPoints = N*N;
    }
    const double TotalNumberOfDataPoints_squared = (double)(TotalNumberOfDataPoints)*(double)(TotalNumberOfDataPoints);

    /// allocate containers
    if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Allocating energy_spect..." << endl;
    double *energy_spect = new double[LocalNumberOfDataPoints];
    if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Allocating energy_lt_spect..." << endl;
    double *energy_lt_spect = new double[LocalNumberOfDataPoints];
    double *energy_spect_wq = 0;
    if (weighting) {
        if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Allocating energy_spect_wq..." << endl;
        energy_spect_wq = new double[LocalNumberOfDataPoints];
    }
    if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: ...allocating done." << endl;
    for (long n = 0; n < LocalNumberOfDataPoints; n++) {
        energy_spect   [n] = 0.0;
        energy_lt_spect[n] = 0.0;
        if (weighting) energy_spect_wq[n] = 0.0;
    }

    /// FFTW normalization
    if (decomposition) {
        if (NDIM==3) {
          for (long n = 0; n < LocalNumberOfDataPoints; n++)
          {
            energy_spect[n] += ( fft_data_x[n][0]*fft_data_x[n][0]+fft_data_x[n][1]*fft_data_x[n][1] +
                                 fft_data_y[n][0]*fft_data_y[n][0]+fft_data_y[n][1]*fft_data_y[n][1] +
                                 fft_data_z[n][0]*fft_data_z[n][0]+fft_data_z[n][1]*fft_data_z[n][1]  )
                                 / TotalNumberOfDataPoints_squared;
          }
        }
        if (NDIM==2) {
          for (long n = 0; n < LocalNumberOfDataPoints; n++)
          {
            energy_spect[n] += ( fft_data_x[n][0]*fft_data_x[n][0]+fft_data_x[n][1]*fft_data_x[n][1] +
                                 fft_data_y[n][0]*fft_data_y[n][0]+fft_data_y[n][1]*fft_data_y[n][1] )
                                 / TotalNumberOfDataPoints_squared;
          }
        }
    }
    else {
        for (long n = 0; n < LocalNumberOfDataPoints; n++)
        {
            energy_spect[n] += (fft_data_x[n][0]*fft_data_x[n][0]+fft_data_x[n][1]*fft_data_x[n][1])
                                / TotalNumberOfDataPoints_squared;
        }
    }
    if (weighting) {
        for (long n = 0; n < LocalNumberOfDataPoints; n++)
        {
            energy_spect_wq[n] += (fft_data_wq[n][0]*fft_data_wq[n][0]+fft_data_wq[n][1]*fft_data_wq[n][1])
                                   / TotalNumberOfDataPoints_squared;
        }
    }

    //////////////////////////////////////////////////////////////////////////////

    double tot_energy_spect = 0.0, tot_energy_spect_red = 0.0;
    for (long n = 0; n < LocalNumberOfDataPoints; n++)
        tot_energy_spect += energy_spect[n];
    if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Local sum in spectral space = " << tot_energy_spect << endl;
    MPI_Allreduce(&tot_energy_spect, &tot_energy_spect_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    tot_energy_spect = tot_energy_spect_red;
    if (MyPE==0) cout << "ComputeSpectrum: Global sum in spectral space before weighting = " << tot_energy_spect << endl;

    /// compute longitudinal spectrum (remember how FFTW sorts the k-values)
    double tot_energy_lt_spect = 0.0, tot_energy_lt_spect_red = 0.0;
    double dec_lt_0 = 0.0; double dec_lt_1 = 0.0;
    if (decomposition)
    {
        int k1 = 0; int k2 = 0; int k3 = 0;
        for (int j = MyInds[0]; j < MyInds[0]+MyInds[1]; j++) // parallelized bit (only loop local part)
        {
          if (j <= Dim[X]/2.) k1 = j; else k1 = j-Dim[X];
          for (int l = 0; l < Dim[Y]; l++)
          {
            if (l <= Dim[Y]/2.) k2 = l; else k2 = l-Dim[Y];
            for (int m = 0; m < Dim[Z]; m++)
            {
              if (m <= Dim[Z]/2.) k3 = m; else k3 = m-Dim[Z];
              double k_sqr_index = k1*k1 + k2*k2 + k3*k3;
              long index = (j-MyInds[0])*Dim[Y]*Dim[Z] + l*Dim[Z] + m; // row-major
              if (NDIM==3) {
                dec_lt_0 = k1*fft_data_x[index][0] + k2*fft_data_y[index][0] + k3*fft_data_z[index][0]; // scalar product (real part)
                dec_lt_1 = k1*fft_data_x[index][1] + k2*fft_data_y[index][1] + k3*fft_data_z[index][1]; // scalar product (imag part)
#if 0
                // longitudinal projection
                fft_data_lt_x[index][0] = dec_lt_0 * k1 / k_sqr_index;
                fft_data_lt_y[index][0] = dec_lt_0 * k2 / k_sqr_index;
                fft_data_lt_z[index][0] = dec_lt_0 * k3 / k_sqr_index;
                fft_data_lt_x[index][1] = dec_lt_1 * k1 / k_sqr_index;
                fft_data_lt_y[index][1] = dec_lt_1 * k2 / k_sqr_index;
                fft_data_lt_z[index][1] = dec_lt_1 * k3 / k_sqr_index;
#endif
              }
              if (NDIM==2) {
                dec_lt_0 = k1*fft_data_x[index][0] + k2*fft_data_y[index][0];
                dec_lt_1 = k1*fft_data_x[index][1] + k2*fft_data_y[index][1];
              }
              if (k_sqr_index > 0)
                energy_lt_spect[index] = (dec_lt_0*dec_lt_0+dec_lt_1*dec_lt_1)/k_sqr_index/TotalNumberOfDataPoints_squared;
            }
          }
        }

        for (long n = 0; n < LocalNumberOfDataPoints; n++) tot_energy_lt_spect += energy_lt_spect[n];
        if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Local sum of longitudinal part in spectral space = " << tot_energy_lt_spect << endl;
        MPI_Allreduce(&tot_energy_lt_spect, &tot_energy_lt_spect_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tot_energy_lt_spect = tot_energy_lt_spect_red;
        if (MyPE==0) cout << "ComputeSpectrum: Global sum of longitudinal part in spectral space before weighting = " << tot_energy_lt_spect << endl;

    } // decomposition

    /// compute the maximum k and construct the spect_grid i.e. the k_axis as well as a staggered grid
    const int    k_cut = N/2;
    const double log_increment_k = 0.01;
    const double increase_k_factor = pow(10.0, 2.0*log_increment_k);
    double       spect_grid_stag[MAX_NUM_BINS]; spect_grid_stag[0] = 0.0;
    double       spect_grid     [MAX_NUM_BINS]; spect_grid[0]      = 0.0;
    double       k_sqr = 1.0;
    int          bin_index = 0;
    while (k_sqr <= (k_cut+1.0)*(k_cut+1.0))
    {
        bin_index++;
        if (k_sqr <= (k_cut+1.0)*(k_cut+1.0))
          k_sqr = bin_index*bin_index;
        else
          k_sqr = k_sqr * increase_k_factor;
        spect_grid_stag[bin_index] = k_sqr;
        if (bin_index >= MAX_NUM_BINS)
        {
          cout << "["<<MyPE<<"] ComputeSpectrum: ERROR. Number of spectral bins exceeds maximum number." << endl;
          exit(FAILURE);
        }
    }
    const int numbins = bin_index;

    /// construct the spectral grid
    for (int bin = 1; bin <= numbins; bin++)
        spect_grid[bin-1] = pow(sqrt(spect_grid_stag[bin-1])+(sqrt(spect_grid_stag[bin])-sqrt(spect_grid_stag[bin-1]))/2.0, 2.0);

    /// calculate spectral densities
    if (MyPE==0 && Debug) cout << "ComputeSpectrum: Calculating spectral densities ..." << endl;
    vector< vector<double> > spect_binsum      (3, vector<double>(numbins)); /// contains longit., transv., total spectral densities
    vector< vector<double> > spect_binsum_sqr  (3, vector<double>(numbins)); /// this is used to compute the RMS and finally the sigma
    vector< vector<double> > sigma_spect_binsum(3, vector<double>(numbins)); /// contains sigmas
    vector< vector<double> > spect_funct       (3, vector<double>(numbins)); /// contains longit., transv., total spectrum functions
    vector< vector<double> > sigma_spect_funct (3, vector<double>(numbins)); /// contains sigmas
    vector<double> comp_lt_spect_funct        (numbins); /// Kolmogorov compensated spectrum function
    vector<double> sigma_comp_lt_spect_funct  (numbins); /// contains sigmas
    vector<double> comp_trsv_spect_funct      (numbins); /// Kolmogorov compensated spectrum function
    vector<double> sigma_comp_trsv_spect_funct(numbins); /// contains sigmas
    vector<double> diss_spect_funct           (numbins); /// dissipative spectrum function
    vector<double> sigma_diss_spect_funct     (numbins); /// contains sigmas
    vector<double> spect_binsum_lin           (numbins); /// contains spectral densities of the non-decomposed dataset
    vector<double> spect_binsum_lin_sqr       (numbins); /// this is used to compute the RMS and finally the sigma
    vector<double> sigma_spect_binsum_lin     (numbins); /// contains sigmas
    vector<double> spect_funct_lin            (numbins); /// contains lin spectrum function
    vector<double> sigma_spect_funct_lin      (numbins); // contains sigmas
    vector<double> weight_binsum              (numbins); /// contains spectrum of weighting quantity
    vector<long> n_cells                      (numbins); /// the number of cells inside a spherical shell in k-space

    for (int bin = 0; bin < numbins; bin++) /// set containers to zero
    {
      if (decomposition)
      {
        for (int type = 0; type < 3; type++) // type means long, trans, total
        {
          spect_binsum      [type][bin] = 0.0;
          spect_binsum_sqr  [type][bin] = 0.0;
          sigma_spect_binsum[type][bin] = 0.0;
          spect_funct       [type][bin] = 0.0;
          sigma_spect_funct [type][bin] = 0.0;
        }
        comp_lt_spect_funct        [bin] = 0.0;
        sigma_comp_lt_spect_funct  [bin] = 0.0;
        comp_trsv_spect_funct      [bin] = 0.0;
        sigma_comp_trsv_spect_funct[bin] = 0.0;
        diss_spect_funct           [bin] = 0.0;
        sigma_diss_spect_funct     [bin] = 0.0;
      }
      else // no decomposition
      {
        spect_binsum_lin      [bin] = 0.0;
        spect_binsum_lin_sqr  [bin] = 0.0;
        sigma_spect_binsum_lin[bin] = 0.0;
        spect_funct_lin       [bin] = 0.0;
        sigma_spect_funct_lin [bin] = 0.0;
      }
      weight_binsum[bin] = 0.0;
      n_cells[bin] = 0;
    }

    int k1 = 0; int k2 = 0; int k3 = 0; // these are the time consuming loops (start optimization here)
    for (int j = MyInds[0]; j < MyInds[0]+MyInds[1]; j++) // the parallel bit
    {
      if (j <= Dim[X]/2.) k1 = j; else k1 = j-Dim[X];
      for (int l = 0; l < Dim[Y]; l++)
      {
        if (l <= Dim[Y]/2.) k2 = l; else k2 = l-Dim[Y];
        for (int m = 0; m < Dim[Z]; m++)
        {
          if (m <= Dim[Z]/2.) k3 = m; else k3 = m-Dim[Z];
          long k_sqr_index = k1*k1 + k2*k2 + k3*k3;
          int interval_l = 0; int interval_r = numbins-1; int bin_id = 0;
          while ((interval_r - interval_l) > 1) /// nested intervals
          {
            bin_id = interval_l + (interval_r - interval_l)/2;
            if (spect_grid[bin_id] > k_sqr_index) interval_r = bin_id;
            else                                  interval_l = bin_id;
          }
          bin_id = interval_r;
          if ((bin_id <= 0) || (bin_id > numbins-1))
          {
            cout << "["<<MyPE<<"] ComputeSpectrum: ERROR. illegal bin index." << endl;
            exit(FAILURE);
          }
          long index = (j-MyInds[0])*Dim[Y]*Dim[Z] + l*Dim[Z] + m; // row-major
          {
            double weight = 1.0;
            if (weighting) {
                weight = energy_spect_wq[index];
            }
            if (decomposition)
            {
              double energy_trsv_spect = energy_spect[index] - energy_lt_spect[index];
              spect_binsum    [0][bin_id] += weight*energy_lt_spect[index];
              spect_binsum    [1][bin_id] += weight*energy_trsv_spect;
              spect_binsum    [2][bin_id] += weight*energy_spect[index];
              spect_binsum_sqr[0][bin_id] += weight*energy_lt_spect[index]*energy_lt_spect[index];
              spect_binsum_sqr[1][bin_id] += weight*energy_trsv_spect*energy_trsv_spect;
              spect_binsum_sqr[2][bin_id] += weight*energy_spect[index]*energy_spect[index];
            }
            else // no decomposition
            {
              spect_binsum_lin    [bin_id] += weight*energy_spect[index];
              spect_binsum_lin_sqr[bin_id] += weight*energy_spect[index]*energy_spect[index];
            }
            weight_binsum[bin_id] += weight;
            n_cells[bin_id]++;
          }
        } // j
      } // l
    } // m

    /// resum the number of cells and total energy in k-space for error checking
    long n_cells_tot = 0, n_cells_tot_red = 0;
    tot_energy_spect = 0.0; tot_energy_lt_spect = 0.0;
    for (int bin = 0; bin < numbins; bin++)
    {
        n_cells_tot += n_cells[bin];
        if (decomposition)  {
            tot_energy_spect    += spect_binsum[2][bin];
            tot_energy_lt_spect += spect_binsum[0][bin];
        }
        if (!decomposition) tot_energy_spect += spect_binsum_lin[bin];
    }
    if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Local ReSummed total number of cells   = " << n_cells_tot << endl;
    MPI_Allreduce(&n_cells_tot, &n_cells_tot_red, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    n_cells_tot = n_cells_tot_red;
    if (MyPE==0) cout << "ComputeSpectrum: Global ReSummed total number of cells = " << n_cells_tot << endl;
    if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: Local ReSummed total in spectral space = " << tot_energy_spect << endl;
    MPI_Allreduce(&tot_energy_spect, &tot_energy_spect_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    tot_energy_spect = tot_energy_spect_red;
    if (MyPE==0) cout << "ComputeSpectrum: Global ReSummed energy in spectral space = " << tot_energy_spect << endl;
    if (decomposition)  {
        MPI_Allreduce(&tot_energy_lt_spect, &tot_energy_lt_spect_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tot_energy_lt_spect = tot_energy_lt_spect_red;
        if (MyPE==0) cout << "ComputeSpectrum: Global ReSummed longitudinal energy in spectral space = " << tot_energy_lt_spect << endl;
    }

    /// MPI Allreduce of the bin containers
    int *tmp_i_red = 0; double *tmp_d_red = 0;
    int *tmp_i = 0; double *tmp_d = 0;

    // reduce n_cells
    tmp_i_red = new int[numbins]; tmp_i = new int[numbins];
    for (int n=0; n<numbins; n++) tmp_i[n] = n_cells[n];
    MPI_Allreduce(tmp_i, tmp_i_red, numbins, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    for (int n=0; n<numbins; n++) n_cells[n]=tmp_i_red[n];
    delete [] tmp_i; delete [] tmp_i_red;
 
    // reduce decomp spect
    tmp_d_red = new double[3*numbins]; tmp_d = new double[3*numbins];
    for (int n=0; n<numbins; n++) for (int dir=0; dir<3; dir++) tmp_d[3*n+dir] = spect_binsum[dir][n];
    MPI_Allreduce(tmp_d, tmp_d_red, 3*numbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int n=0; n<numbins; n++) for (int dir=0; dir<3; dir++) spect_binsum[dir][n]=tmp_d_red[3*n+dir];
    for (int n=0; n<numbins; n++) for (int dir=0; dir<3; dir++) tmp_d[3*n+dir] = spect_binsum_sqr[dir][n];
    MPI_Allreduce(tmp_d, tmp_d_red, 3*numbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int n=0; n<numbins; n++) for (int dir=0; dir<3; dir++) spect_binsum_sqr[dir][n]=tmp_d_red[3*n+dir];
    delete [] tmp_d; delete [] tmp_d_red;

    // reduce lin spect
    tmp_d_red = new double[numbins]; tmp_d = new double[numbins];
    for (int n=0; n<numbins; n++) tmp_d[n] = spect_binsum_lin[n];
    MPI_Allreduce(tmp_d, tmp_d_red, numbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int n=0; n<numbins; n++) spect_binsum_lin[n]=tmp_d_red[n];
    for (int n=0; n<numbins; n++) tmp_d[n] = spect_binsum_lin_sqr[n];
    MPI_Allreduce(tmp_d, tmp_d_red, numbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int n=0; n<numbins; n++) spect_binsum_lin_sqr[n]=tmp_d_red[n];
    // reduce weight
    for (int n=0; n<numbins; n++) tmp_d[n] = weight_binsum[n];
    MPI_Allreduce(tmp_d, tmp_d_red, numbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int n=0; n<numbins; n++) weight_binsum[n]=tmp_d_red[n];
    delete [] tmp_d; delete [] tmp_d_red;


    /// write out (MASTER CPU only)
    if (MyPE==0)
    {
        /// calculate spectral densities and functions (normalization)
        for (int bin = 0; bin < numbins; bin++)
        {
            if (n_cells[bin] > 0)
            {
                if (decomposition)
                {
                    for (int dir = 0; dir < 3; dir++) /// long., transv., total
                    {
                        spect_binsum      [dir][bin] /= weight_binsum[bin];
                        spect_binsum_sqr  [dir][bin] /= weight_binsum[bin];
                        sigma_spect_binsum[dir][bin]  = sqrt(spect_binsum_sqr[dir][bin] - spect_binsum[dir][bin]*spect_binsum[dir][bin]);
                        if (NDIM==3) {
                          spect_funct       [dir][bin]  = 4*pi*spect_grid_stag[bin]*spect_binsum      [dir][bin];
                          sigma_spect_funct [dir][bin]  = 4*pi*spect_grid_stag[bin]*sigma_spect_binsum[dir][bin];
                        }
                        if (NDIM==2) {
                          spect_funct       [dir][bin]  = 2*pi*sqrt(spect_grid_stag[bin])*spect_binsum      [dir][bin];
                          sigma_spect_funct [dir][bin]  = 2*pi*sqrt(spect_grid_stag[bin])*sigma_spect_binsum[dir][bin];
                        }
                    }
                    comp_lt_spect_funct        [bin] = pow(spect_grid_stag[bin], 2.0/2.0) * spect_funct      [0][bin];
                    sigma_comp_lt_spect_funct  [bin] = pow(spect_grid_stag[bin], 2.0/2.0) * sigma_spect_funct[0][bin];
                    comp_trsv_spect_funct      [bin] = pow(spect_grid_stag[bin], 5.0/6.0) * spect_funct      [1][bin];
                    sigma_comp_trsv_spect_funct[bin] = pow(spect_grid_stag[bin], 5.0/6.0) * sigma_spect_funct[1][bin];
                    diss_spect_funct           [bin] = spect_grid_stag[bin] * spect_funct      [2][bin];
                    sigma_diss_spect_funct     [bin] = spect_grid_stag[bin] * sigma_spect_funct[2][bin];
                }
                else // no decomposition
                {
                    spect_binsum_lin      [bin] /= weight_binsum[bin];
                    spect_binsum_lin_sqr  [bin] /= weight_binsum[bin];
                    sigma_spect_binsum_lin[bin]  = sqrt(spect_binsum_lin_sqr[bin] - spect_binsum_lin[bin]*spect_binsum_lin[bin]);
                    if (NDIM==3) {
                      spect_funct_lin       [bin]  = 4*pi*spect_grid_stag[bin]*spect_binsum_lin[bin];
                      sigma_spect_funct_lin [bin]  = 4*pi*spect_grid_stag[bin]*sigma_spect_binsum_lin[bin];
                    }
                    if (NDIM==2) {
                      spect_funct_lin       [bin]  = 2*pi*sqrt(spect_grid_stag[bin])*spect_binsum_lin[bin];
                      sigma_spect_funct_lin [bin]  = 2*pi*sqrt(spect_grid_stag[bin])*sigma_spect_binsum_lin[bin];
                    }
                }
            }
        }

        /// prepare OutputFileHeader
        OutputFileHeader.resize(0);
        stringstream dummystream;
        if (decomposition)
        {
            dummystream.precision(8);
            dummystream << "P_tot = " << endl;
            dummystream << scientific << tot_energy_spect << endl;
            dummystream << "P_lgt = " << endl;
            dummystream << scientific << tot_energy_lt_spect << endl << endl;
            dummystream << setw(30) << left << "#00_BinIndex";
            dummystream << setw(30) << left << "#01_KStag"              << setw(30) << left << "#02_K";
            dummystream << setw(30) << left << "#03_DK"                 << setw(30) << left << "#04_NCells";
            dummystream << setw(30) << left << "#05_SpectDensLgt"       << setw(30) << left << "#06_SpectDensLgtSigma";
            dummystream << setw(30) << left << "#07_SpectDensTrv"       << setw(30) << left << "#08_SpectDensTrvSigma";
            dummystream << setw(30) << left << "#09_SpectDensTot"       << setw(30) << left << "#10_SpectDensTotSigma";
            dummystream << setw(30) << left << "#11_SpectFunctLgt"      << setw(30) << left << "#12_SpectFunctLgtSigma";
            dummystream << setw(30) << left << "#13_SpectFunctTrv"      << setw(30) << left << "#14_SpectFunctTrvSigma";
            dummystream << setw(30) << left << "#15_SpectFunctTot"      << setw(30) << left << "#16_SpectFunctTotSigma";
            dummystream << setw(30) << left << "#17_CompSpectFunctLgt"  << setw(30) << left << "#18_CompSpectFunctLgtSigma";
            dummystream << setw(30) << left << "#19_CompSpectFunctTrv"  << setw(30) << left << "#20_CompSpectFunctTrvSigma";
            dummystream << setw(30) << left << "#21_DissSpectFunct"     << setw(30) << left << "#22_DissSpectFunctSigma";
            OutputFileHeader.push_back(dummystream.str()); dummystream.clear(); dummystream.str("");
        }
        else // no decomposition
        {
            dummystream << setw(30) << left << "#00_BinIndex";
            dummystream << setw(30) << left << "#01_KStag"           << setw(30) << left << "#02_K";
            dummystream << setw(30) << left << "#03_DK"              << setw(30) << left << "#04_NCells";
            dummystream << setw(30) << left << "#05_SpectDens"       << setw(30) << left << "#06_SpectDensSigma";
            dummystream << setw(30) << left << "#07_SpectFunct"      << setw(30) << left << "#08_SpectFunctSigma";
            OutputFileHeader.push_back(dummystream.str()); dummystream.clear(); dummystream.str("");
        }

        if (decomposition)
        {
            /// resize and fill WriteOutTable
            WriteOutTable.resize(numbins-2); /// spectrum output has numbins-2 lines
            for (unsigned int i = 0; i < WriteOutTable.size(); i++)
                WriteOutTable[i].resize(23); /// dec energy spectrum output has 23 columns
            for (int bin = 1; bin < numbins-1; bin++)
            {
                int wob = bin-1;
                WriteOutTable[wob][ 0] = bin;
                WriteOutTable[wob][ 1] = sqrt(spect_grid_stag[bin]); /// k (staggered)
                WriteOutTable[wob][ 2] = sqrt(spect_grid     [bin]); /// k
                WriteOutTable[wob][ 3] = sqrt(spect_grid[bin])-sqrt(spect_grid[bin-1]); /// delta k
                WriteOutTable[wob][ 4] = n_cells                    [bin]; /// the number of cells in bin
                WriteOutTable[wob][ 5] = spect_binsum           [0] [bin]; /// longitudinal spectral density
                WriteOutTable[wob][ 6] = sigma_spect_binsum     [0] [bin]; /// sigma
                WriteOutTable[wob][ 7] = spect_binsum           [1] [bin]; /// transversal spectral density
                WriteOutTable[wob][ 8] = sigma_spect_binsum     [1] [bin]; /// sigma
                WriteOutTable[wob][ 9] = spect_binsum           [2] [bin]; /// total spectral density
                WriteOutTable[wob][10] = sigma_spect_binsum     [2] [bin]; /// sigma
                WriteOutTable[wob][11] = spect_funct            [0] [bin]; /// longitudinal spectrum function
                WriteOutTable[wob][12] = sigma_spect_funct      [0] [bin]; /// sigma
                WriteOutTable[wob][13] = spect_funct            [1] [bin]; /// transversal spectrum function
                WriteOutTable[wob][14] = sigma_spect_funct      [1] [bin]; /// sigma
                WriteOutTable[wob][15] = spect_funct            [2] [bin]; /// total spectrum function
                WriteOutTable[wob][16] = sigma_spect_funct      [2] [bin]; /// sigma
                WriteOutTable[wob][17] = comp_lt_spect_funct        [bin]; /// compensated longitudinal spectrum function
                WriteOutTable[wob][18] = sigma_comp_lt_spect_funct  [bin]; /// sigma
                WriteOutTable[wob][19] = comp_trsv_spect_funct      [bin]; /// compensated tranversal spectrum function
                WriteOutTable[wob][20] = sigma_comp_trsv_spect_funct[bin]; /// sigma
                WriteOutTable[wob][21] = diss_spect_funct           [bin]; /// dissipative spectrum function
                WriteOutTable[wob][22] = sigma_diss_spect_funct     [bin]; /// sigma
            }
        }
        else // no decomposition
        {
            /// resize and fill WriteOutTable
            WriteOutTable.resize(numbins-2); /// spectrum output has numbins-2 lines
            for (unsigned int i = 0; i < WriteOutTable.size(); i++)
                WriteOutTable[i].resize(9); /// density spectrum output has 9 columns
            for (int bin = 1; bin < numbins-1; bin++)
            {
                int wob = bin-1;
                WriteOutTable[wob][0] = bin;
                WriteOutTable[wob][1] = sqrt(spect_grid_stag[bin]); /// k (staggered)
                WriteOutTable[wob][2] = sqrt(spect_grid     [bin]); /// k
                WriteOutTable[wob][3] = sqrt(spect_grid[bin])-sqrt(spect_grid[bin-1]); /// delta k
                WriteOutTable[wob][4] = n_cells                   [bin]; /// the number of cells in bin
                WriteOutTable[wob][5] = spect_binsum_lin          [bin]; /// spectral density of non-decomposed dataset
                WriteOutTable[wob][6] = sigma_spect_binsum_lin    [bin]; /// sigma
                WriteOutTable[wob][7] = spect_funct_lin           [bin]; /// spectrum function of non-decomposed dataset
                WriteOutTable[wob][8] = sigma_spect_funct_lin     [bin]; /// sigma
            }
        }

    } // MyPE==0

    /// clean up
    delete [] energy_spect;
    delete [] energy_lt_spect;
    if (weighting) delete [] energy_spect_wq;

    long endtime = time(NULL);
    int duration = endtime-starttime, duration_red = 0;
    if (Debug) cout << "["<<MyPE<<"] ****************** Local elapsed time for spectrum function computation = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << "****************** Global elapsed time for spectrum function computation = "<<duration_red<<"s ******************" << endl;
    if (Debug) cout << "["<<MyPE<<"] ComputeSpectrum: exiting." << endl;
} /// =======================================================================


/** --------------------- Normalize -----------------------------------------------
 ** divide by norm
 ** ------------------------------------------------------------------------------- */
void Normalize(float * const data_array, const long n, const double norm)
{
    for (long i = 0; i < n; i++) data_array[i] /= norm;
} /// =======================================================================


/** ----------------------------- Mean -------------------------------
 **  computes the mean of a pointer-array
 ** ------------------------------------------------------------------ */
double Mean(const float * const data, const long size)
{
    long local_size = size;
    long global_size = 0;
    double value = 0.0, value_red = 0.0;
    for (long n = 0; n < local_size; n++) value += data[n];
    MPI_Allreduce(&value, &value_red, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    value_red /= static_cast<double>(global_size);
    return value_red;
} /// =======================================================================


/** ------------------------ Window -----------------------------------------------
 ** apply window function to data ("Hann" or "Hanning" window in 3D)
 ** ------------------------------------------------------------------------------- */
void Window(float* const data_array, const int nx, const int ny, const int nz)
{
    if ((nx!=ny)||(nx!=nz))
    {
        cout << "Window: only works for nx=ny=nz." << endl;
        exit(FAILURE);
    }
    const long nxy = nx*ny;
    const double twopi = 2.*pi;
    const double L = (double)(nx);

    for (int k = 0; k < nz; k++)
    {
      for (int j = 0; j < ny; j++)
      {
        for (int i = 0; i < nx; i++)
        {
            double dx = (double)(i)+0.5-((double)(nx)/2.);
            double dy = (double)(j)+0.5-((double)(ny)/2.);
            double dz = (double)(k)+0.5-((double)(nz)/2.);
            double r = sqrt(dx*dx+dy*dy+dz*dz);
            long index = k*nxy+j*nx+i;
            if (r < L/2.)
                data_array[index] *= 0.5*(1.0+cos((twopi*r)/L));
            else
                data_array[index] = 0.;
        } //i
      } //j
    } //k
} /// =======================================================================


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
        exit (FAILURE);
    }
    /// write data to output file
    else
    {
        cout << "WriteOutAnalysedData:  Writing output file '" << OutputFilename.c_str() << "' ..." << endl;

        for (unsigned int row = 0; row < OutputFileHeader.size(); row++)
        {
        Outputfile << setw(61) << left << OutputFileHeader[row] << endl;      /// header
        if (false && Debug) cout << setw(61) << left << OutputFileHeader[row] << endl;
        }
        for (unsigned int row = 0; row < WriteOutTable.size(); row++)                  /// data
        {
        for (unsigned int col = 0; col < WriteOutTable[row].size(); col++)
        {
            Outputfile << scientific << setw(30) << left << setprecision(8) << WriteOutTable[row][col];
            if (false && Debug) cout << scientific << setw(30) << left << setprecision(8) << WriteOutTable[row][col];
        }
        Outputfile << endl; if (false && Debug) cout << endl;
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
    valid_options.push_back("-types");
    valid_options.push_back("-dsets");
    valid_options.push_back("-dset_scale");
    valid_options.push_back("-dens_scale");
    valid_options.push_back("-vels_scale");
    valid_options.push_back("-mags_scale");
    valid_options.push_back("-wq");
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
        if (Argument[i] != "" && Argument[i] == "-dset_scale")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> dset_scale; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-dens_scale")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> dens_scale; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-vels_scale")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> vels_scale; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-mags_scale")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> mags_scale; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-wq")
        {
            if (Argument.size()>i+1) weight_name = Argument[i+1]; else return -1;
            weighting = true;
        }
    } // loop over all args

    // if user did not specify any types, we set VELS as the (only) default
    if (RequestedTypes.size() == 0) RequestedTypes.push_back(VELS);

    // print requested types and error check on DSETTYPE type
    if (MyPE==0) {
        cout << "ParseInputs: Requested spectrum types:" << endl;
        for (unsigned int i = 0; i < RequestedTypes.size(); i++){
            cout << "  -> "<<setw(2)<<RequestedTypes[i]<<": "<<OutfileStringForType[RequestedTypes[i]]<<endl;
            if (RequestedTypes[i] == DSETTYPE && DsetName[0] == "") {
                cout << "ParseInputs: Error. If using -types 0 or 15, you must specify -dset ..." << endl;
                return -1;
            }
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
            << " spectra <filename> [<OPTIONS>]" << endl << endl
            << "   <OPTIONS>:           " << endl
            << "     -types <type(s)>        : requested spectrum type(s) by sequence of numbers; available types:" << endl;
            for (int i = 0; i < MaxNumTypes; i++) cout << "                               "<<setw(2)<<i<<" : "+OutfileStringForType[i]<<endl;
            cout << "     -dsets <datasetname(s)> : for type 0, specify dataset(s) to process (up to 3 components)" << endl
            << "     -dset_scale <double>    : divide data by dset_scale during read (default 1.0); applies for type 0 (dset) only" << endl
            << "     -dens_scale <double>    : divide dens by dens_scale during read (default 1.0)" << endl
            << "     -vels_scale <double>    : divide vel(x,y,z) by vels_scale during read (default 1.0)" << endl
            << "     -mags_scale <double>    : divide mag(x,y,z) by mags_scale during read (default 1.0)" << endl
            << "     -wq <weight_dset>       : compute spectra weighted by <weight_dset>; e.g., for mass weighting use '-wq dens'" << endl
            << endl
            << "Example: spectra DF_hdf5_plt_cnt_0020 -types 1 10" << endl
            << endl << endl;
    }
}
