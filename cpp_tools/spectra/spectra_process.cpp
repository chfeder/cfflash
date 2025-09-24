///
///  Fourier spectra MPI version (single-precision) for FlashGG
///  This program does Read->FFTW->Process->FFTW->Write
///
///  written by Christoph Federrath, 2012-2023
///

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
#include <unistd.h>
#include <sys/stat.h>

// constants
int NDIM = 3;
using namespace std;
enum {X, Y, Z};

// MPI stuff
int MyPE = 0, NPE = 1;

// for FFTW
fftwf_complex *fft_data_x, *fft_data_y, *fft_data_z;
fftwf_plan fft_plan_x_fwd, fft_plan_y_fwd, fft_plan_z_fwd;
fftwf_plan fft_plan_x_inv, fft_plan_y_inv, fft_plan_z_inv;

// arguments
bool vector_pot = false;
string decompose = "none";
string inputfile = "";
string filter = "none";
double filter_k = 0.0;
double filter_fwhm = 0.0;
string write_parallel = "blocks";

FlashGG gg; // grid class handler

/// forward function
int VectorPot(const vector<int> Dim);
int Decompose(const vector<int> Dim);
int Filter(const string datatsetname, const vector<int> Dim);
vector<int> InitFFTW(const vector<int> Dim, const bool vector_dataset);
int SetupParallelisation(const vector<int> Dim, const vector<int> MyInds);
void FinaliseFFTW(const bool vector_dataset);
float * ReadParallel(const string datasetname, vector<int> MyInds);
void WriteParallelBlocks(const string inputfile, const string datasetname, float * data_ptr, vector<int> MyInds);
void WriteParallelSlabs (const string inputfile, const string datasetname, float * data_ptr, vector<int> MyInds);
void SwapMemOrder(float * const data, const vector<int> N);
void Normalize(float * const data_array, const long n, const double norm);
double ComputeMean(float * const data_array, const long n);
int ParseInputs(const vector<string> Argument);
void HelpMe(void);


/// --------
///   MAIN
/// --------
int main(int argc, char * argv[])
{
    const string FuncName = "Main: ";
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NPE);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
    if (MyPE==0) cout<<FuncName+"spectra_process === MPI num procs: "<<NPE<<endl;

    /// Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) == -1)
    {
        if (MyPE==0) cout << endl << FuncName+"Error in ParseInputs(). Exiting." << endl;
        HelpMe(); MPI_Finalize(); return 0;
    }

    long starttime = time(NULL);

    // create FlashGG object
    if (write_parallel ==  "slabs") gg = FlashGG(inputfile, 'w');
    if (write_parallel == "blocks") gg = FlashGG(inputfile);

    /// get the dataset dimensions
    vector<int> N = gg.GetN();
    if (MyPE==0) gg.PrintInfo();

    // signal dimensionality of the simulation
    if (N[Z]==1) NDIM = 2;
    if (NDIM==3) {
        if ((N[X]!=N[Y])||(N[X]!=N[Z])||(N[Y]!=N[Z])) {
            cout << FuncName+"Spectra can only be obtained from cubic datasets (Nx=Ny=Nz)." << endl;
            MPI_Finalize(); return 0;
        }
    }

    // processes
    if (vector_pot) VectorPot(N); // Compute vector potential and write to file
    if (decompose != "none") Decompose(N); // Helmholtz decomposition and write longitudinal vector component to file
    if (filter != "none") Filter("dens", N); // Filter in Fourier space and write filtered, backwards-FFTed field to file

    long endtime = time(NULL);
    int duration = endtime-starttime, duration_red = 0;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << FuncName+"Time to finish = "<<duration_red<<"s" << endl;	

    MPI_Finalize();
    return 0;

} // end main ==========================================================


/** ------------------------- VectorPot -----------------------------
 **  read, FFTW (forward), compute vector potential, FFTW (inverse), write
 ** ----------------------------------------------------------------- */
int VectorPot(const vector<int> Dim)
{
    const bool Debug = false;
    const string FuncName = "VectorPot: ";

    if ((MyPE==0) && Debug) cout << FuncName+"entering." << endl;

    /// allocate FFTW containers and create FTTW plans
    vector<int> MyInds = InitFFTW(Dim, true); // 2nd argument=true: uses vector dataset
    if (MyInds[0]==-1 && MyInds[1]==-1) return -1; // error

    // setup pseudo blocks and check parallelisation
    SetupParallelisation(Dim, MyInds);

    /// read data
    long starttime_read = time(NULL);
    if (MyPE==0) cout<<FuncName+"Start reading data from disk..."<<endl;
    const long ntot_local = (long)MyInds[1]*(long)Dim[Y]*(long)Dim[Z];
    float *datx = ReadParallel("magx", MyInds); if (MyPE==0) cout<<FuncName+"magx"+" read."<<endl;
    float *daty = ReadParallel("magy", MyInds); if (MyPE==0) cout<<FuncName+"magy"+" read."<<endl;
    float *datz = ReadParallel("magz", MyInds); if (MyPE==0) cout<<FuncName+"magz"+" read."<<endl;
    long endtime_read = time(NULL);
    int duration_read = endtime_read-starttime_read, duration_read_red = 0;
    MPI_Allreduce(&duration_read, &duration_read_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << FuncName+"Time for reading data = "<<duration_read_red<<"s" << endl;

    // assign data to FFTW container
    for (long n=0; n<ntot_local; n++) {
        fft_data_x[n][0] = datx[n]; /// Real part
        fft_data_y[n][0] = daty[n]; /// Real part
        fft_data_z[n][0] = datz[n]; /// Real part
        fft_data_x[n][1] = 0.0; /// Imaginary part
        fft_data_y[n][1] = 0.0; /// Imaginary part
        fft_data_z[n][1] = 0.0; /// Imaginary part
    }

    /// general constants
    const long TotalNumberOfDataPoints = (long)Dim[X]*(long)Dim[Y]*(long)Dim[Z];
    const double TotalNumberOfDataPointsDouble = (double)(TotalNumberOfDataPoints);

    // FORWARD FFTW
    fftwf_execute(fft_plan_x_fwd);
    fftwf_execute(fft_plan_y_fwd);
    fftwf_execute(fft_plan_z_fwd);

    // compute vector potential
    int k1 = 0; int k2 = 0; int k3 = 0; // loop over all cells to get k
    for (int j = MyInds[0]; j < MyInds[0]+MyInds[1]; j++) // the parallel bit
    {
        if (j <= Dim[X]/2.) k1 = j; else k1 = j-Dim[X];
        for (int l = 0; l < Dim[Y]; l++)
        {
            if (l <= Dim[Y]/2.) k2 = l; else k2 = l-Dim[Y];
            for (int m = 0; m < Dim[Z]; m++)
            {
                if (m <= Dim[Z]/2.) k3 = m; else k3 = m-Dim[Z];

                long n = (j-MyInds[0])*Dim[Y]*Dim[Z] + l*Dim[Z] + m; // row-major

                long k_sqr = (k1*k1 + k2*k2 + k3*k3) * 2*M_PI;

                if (k_sqr > 0) {
                    // k cross B-ffted
                    double axr = k2*fft_data_z[n][0] - k3*fft_data_y[n][0]; // x-component of k cross B-ffted (real part)
                    double axi = k2*fft_data_z[n][1] - k3*fft_data_y[n][1]; // x-component of k cross B-ffted (imag part)
                    double ayr = k3*fft_data_x[n][0] - k1*fft_data_z[n][0]; // y-component of k cross B-ffted (real part)
                    double ayi = k3*fft_data_x[n][1] - k1*fft_data_z[n][1]; // y-component of k cross B-ffted (imag part)
                    double azr = k1*fft_data_y[n][0] - k2*fft_data_x[n][0]; // z-component of k cross B-ffted (real part)
                    double azi = k1*fft_data_y[n][1] - k2*fft_data_x[n][1]; // z-component of k cross B-ffted (imag part)
                    // i * (k cross B) / k^2
                    fft_data_x[n][0] = -axi / k_sqr / TotalNumberOfDataPointsDouble; // new real part (<- negative imag part)
                    fft_data_x[n][1] = +axr / k_sqr / TotalNumberOfDataPointsDouble; // new imag part (<- positive real part)
                    fft_data_y[n][0] = -ayi / k_sqr / TotalNumberOfDataPointsDouble; // new real part (<- negative imag part)
                    fft_data_y[n][1] = +ayr / k_sqr / TotalNumberOfDataPointsDouble; // new imag part (<- positive real part)
                    fft_data_z[n][0] = -azi / k_sqr / TotalNumberOfDataPointsDouble; // new real part (<- negative imag part)
                    fft_data_z[n][1] = +azr / k_sqr / TotalNumberOfDataPointsDouble; // new imag part (<- positive real part)
                }
                else { // k = 0
                    fft_data_x[n][0] /= TotalNumberOfDataPointsDouble; // real part
                    fft_data_y[n][0] /= TotalNumberOfDataPointsDouble; // real part
                    fft_data_z[n][0] /= TotalNumberOfDataPointsDouble; // real part
                    fft_data_x[n][1] /= TotalNumberOfDataPointsDouble; // imag part
                    fft_data_y[n][1] /= TotalNumberOfDataPointsDouble; // imag part
                    fft_data_z[n][1] /= TotalNumberOfDataPointsDouble; // imag part
                }
            } // j
        } // l
    } // m

    // INVERSE FFTW
    fftwf_execute(fft_plan_x_inv);
    fftwf_execute(fft_plan_y_inv);
    fftwf_execute(fft_plan_z_inv);

    // overwrite data
    for (long n=0; n<ntot_local; n++) {
        datx[n] = fft_data_x[n][0]; /// only Real part
        daty[n] = fft_data_y[n][0]; /// only Real part
        datz[n] = fft_data_z[n][0]; /// only Real part
    }

    /// write out
    long starttime_write = time(NULL);
    ostringstream ss; ss << filter_k;
    string datasetname_out_x = "vecpotx";
    string datasetname_out_y = "vecpoty";
    string datasetname_out_z = "vecpotz";
    if (write_parallel ==  "slabs") {
        WriteParallelSlabs (inputfile, datasetname_out_x, datx, MyInds);
        WriteParallelSlabs (inputfile, datasetname_out_y, daty, MyInds);
        WriteParallelSlabs (inputfile, datasetname_out_z, datz, MyInds);
    }
    if (write_parallel == "blocks") {
        WriteParallelBlocks(inputfile, datasetname_out_x, datx, MyInds);
        WriteParallelBlocks(inputfile, datasetname_out_y, daty, MyInds);
        WriteParallelBlocks(inputfile, datasetname_out_z, datz, MyInds);
    }
    /// deallocate and clean
    delete [] datx; ///// DEALLOCATE
    delete [] daty; ///// DEALLOCATE
    delete [] datz; ///// DEALLOCATE
    FinaliseFFTW(true);

    long endtime_write = time(NULL);
    int duration_write = endtime_write-starttime_write, duration_write_red = 0;
    MPI_Allreduce(&duration_write, &duration_write_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout<<FuncName+"Time for writing data = "<<duration_write_red<<"s" << endl;

    if ((MyPE==0) && Debug) cout << FuncName+"exiting." << endl;

    return 0;

} /// =======================================================================


/** ------------------------- Decompose -----------------------------
 **  read, FFTW (forward), decompose, FFTW (inverse), write
 ** ----------------------------------------------------------------- */
int Decompose(const vector<int> Dim)
{
    const bool Debug = false;
    const string FuncName = "Decompose: ";

    if ((MyPE==0) && Debug) cout << FuncName+"entering." << endl;

    /// allocate FFTW containers and create FTTW plans
    vector<int> MyInds = InitFFTW(Dim, true); // 2nd argument=true: uses vector dataset
    if (MyInds[0]==-1 && MyInds[1]==-1) return -1; // error

    // setup pseudo blocks and check parallelisation
    SetupParallelisation(Dim, MyInds);

    /// read data
    long starttime_read = time(NULL);
    if (MyPE==0) cout<<FuncName+"Start reading data from disk..."<<endl;
    const long ntot_local = (long)MyInds[1]*(long)Dim[Y]*(long)Dim[Z];
    float *datx = ReadParallel("velx", MyInds); if (MyPE==0) cout<<FuncName+"velx"+" read."<<endl;
    float *daty = ReadParallel("vely", MyInds); if (MyPE==0) cout<<FuncName+"vely"+" read."<<endl;
    float *datz = ReadParallel("velz", MyInds); if (MyPE==0) cout<<FuncName+"velz"+" read."<<endl;
    long endtime_read = time(NULL);
    int duration_read = endtime_read-starttime_read, duration_read_red = 0;
    MPI_Allreduce(&duration_read, &duration_read_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << FuncName+"Time for reading data = "<<duration_read_red<<"s" << endl;

    // assign data to FFTW container
    for (long n=0; n<ntot_local; n++) {
        fft_data_x[n][0] = datx[n]; /// Real part
        fft_data_y[n][0] = daty[n]; /// Real part
        fft_data_z[n][0] = datz[n]; /// Real part
        fft_data_x[n][1] = 0.0; /// Imaginary part
        fft_data_y[n][1] = 0.0; /// Imaginary part
        fft_data_z[n][1] = 0.0; /// Imaginary part
    }

    /// general constants
    const long TotalNumberOfDataPoints = (long)Dim[X]*(long)Dim[Y]*(long)Dim[Z];
    const double TotalNumberOfDataPointsDouble = (double)(TotalNumberOfDataPoints);

    // FORWARD FFTW
    fftwf_execute(fft_plan_x_fwd);
    fftwf_execute(fft_plan_y_fwd);
    fftwf_execute(fft_plan_z_fwd);

    // decompose
    int k1 = 0; int k2 = 0; int k3 = 0; // loop over all cells to get k
    for (int j = MyInds[0]; j < MyInds[0]+MyInds[1]; j++) // the parallel bit
    {
        if (j <= Dim[X]/2.) k1 = j; else k1 = j-Dim[X];
        for (int l = 0; l < Dim[Y]; l++)
        {
            if (l <= Dim[Y]/2.) k2 = l; else k2 = l-Dim[Y];
            for (int m = 0; m < Dim[Z]; m++)
            {
                if (m <= Dim[Z]/2.) k3 = m; else k3 = m-Dim[Z];

                long n = (j-MyInds[0])*Dim[Y]*Dim[Z] + l*Dim[Z] + m; // row-major

                long k_sqr = k1*k1 + k2*k2 + k3*k3;

                if (k_sqr > 0) {
                    // longitudinal projection
                    double dec_lt_0 = k1*fft_data_x[n][0] + k2*fft_data_y[n][0] + k3*fft_data_z[n][0]; // scalar product (real part)
                    double dec_lt_1 = k1*fft_data_x[n][1] + k2*fft_data_y[n][1] + k3*fft_data_z[n][1]; // scalar product (imag part)
                    fft_data_x[n][0] = dec_lt_0 * k1 / k_sqr / TotalNumberOfDataPointsDouble; // real part
                    fft_data_y[n][0] = dec_lt_0 * k2 / k_sqr / TotalNumberOfDataPointsDouble; // real part
                    fft_data_z[n][0] = dec_lt_0 * k3 / k_sqr / TotalNumberOfDataPointsDouble; // real part
                    fft_data_x[n][1] = dec_lt_1 * k1 / k_sqr / TotalNumberOfDataPointsDouble; // imag part
                    fft_data_y[n][1] = dec_lt_1 * k2 / k_sqr / TotalNumberOfDataPointsDouble; // imag part
                    fft_data_z[n][1] = dec_lt_1 * k3 / k_sqr / TotalNumberOfDataPointsDouble; // imag part
                }
                else { // k = 0
                    fft_data_x[n][0] /= TotalNumberOfDataPointsDouble; // real part
                    fft_data_y[n][0] /= TotalNumberOfDataPointsDouble; // real part
                    fft_data_z[n][0] /= TotalNumberOfDataPointsDouble; // real part
                    fft_data_x[n][1] /= TotalNumberOfDataPointsDouble; // imag part
                    fft_data_y[n][1] /= TotalNumberOfDataPointsDouble; // imag part
                    fft_data_z[n][1] /= TotalNumberOfDataPointsDouble; // imag part
                }

            } // j
        } // l
    } // m

    // INVERSE FFTW
    fftwf_execute(fft_plan_x_inv);
    fftwf_execute(fft_plan_y_inv);
    fftwf_execute(fft_plan_z_inv);

    // overwrite data
    for (long n=0; n<ntot_local; n++) {
        datx[n] = fft_data_x[n][0]; /// only Real part
        daty[n] = fft_data_y[n][0]; /// only Real part
        datz[n] = fft_data_z[n][0]; /// only Real part
    }

    /// write out data
    long starttime_write = time(NULL);
    ostringstream ss; ss << filter_k;
    string datasetname_out_x = "velx_lt";
    string datasetname_out_y = "vely_lt";
    string datasetname_out_z = "velz_lt";
    if (write_parallel ==  "slabs") {
        WriteParallelSlabs (inputfile, datasetname_out_x, datx, MyInds);
        WriteParallelSlabs (inputfile, datasetname_out_y, daty, MyInds);
        WriteParallelSlabs (inputfile, datasetname_out_z, datz, MyInds);
    }
    if (write_parallel == "blocks") {
        WriteParallelBlocks(inputfile, datasetname_out_x, datx, MyInds);
        WriteParallelBlocks(inputfile, datasetname_out_y, daty, MyInds);
        WriteParallelBlocks(inputfile, datasetname_out_z, datz, MyInds);
    }
    /// deallocate and clean
    delete [] datx; ///// DEALLOCATE
    delete [] daty; ///// DEALLOCATE
    delete [] datz; ///// DEALLOCATE
    FinaliseFFTW(true);

    long endtime_write = time(NULL);
    int duration_write = endtime_write-starttime_write, duration_write_red = 0;
    MPI_Allreduce(&duration_write, &duration_write_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout<<FuncName+"Time for writing data = "<<duration_write_red<<"s" << endl;

    if ((MyPE==0) && Debug) cout << FuncName+"exiting." << endl;

    return 0;

} /// =======================================================================


/** -------------------------- Filter -------------------------------
 **  read, FFTW (forward), filter, FFTW (inverse), write
 ** ----------------------------------------------------------------- */
int Filter(const string datatsetname, const vector<int> Dim)
{
    const bool Debug = false;
    const string FuncName = "Filter: ";

    if ((MyPE==0) && Debug) cout << FuncName+"entering." << endl;

    /// allocate FFTW containers and create FTTW plans
    vector<int> MyInds = InitFFTW(Dim, false); // 2nd argument=false: uses scalar dataset
    if (MyInds[0]==-1 && MyInds[1]==-1) return -1; // error

    // setup pseudo blocks and check parallelisation
    SetupParallelisation(Dim, MyInds);

    /// read data
    long starttime_read = time(NULL);
    if (MyPE==0) cout<<FuncName+"Start reading data from disk..."<<endl;
    const long ntot_local = (long)MyInds[1]*(long)Dim[Y]*(long)Dim[Z];
    float *data = ReadParallel(datatsetname, MyInds); if (MyPE==0) cout<<FuncName+datatsetname+" read."<<endl;
    long endtime_read = time(NULL);
    int duration_read = endtime_read-starttime_read, duration_read_red = 0;
    MPI_Allreduce(&duration_read, &duration_read_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << FuncName+"Time for reading data = "<<duration_read_red<<"s" << endl;	

    // assign data to FFTW container
    for (long n=0; n<ntot_local; n++) {
        fft_data_x[n][0] = data[n]; /// Real part
        fft_data_x[n][1] = 0.0; /// Imaginary part
    }
    if (datatsetname == "dens") { // take natural log to keep dens > 0
        for (long n=0; n<ntot_local; n++) fft_data_x[n][0] = log(fft_data_x[n][0]);
    }

    /// general constants
    const long TotalNumberOfDataPoints = (long)Dim[X]*(long)Dim[Y]*(long)Dim[Z];
    const double TotalNumberOfDataPointsDouble = (double)(TotalNumberOfDataPoints);

    // FORWARD FFTW
    fftwf_execute(fft_plan_x_fwd);

    double log_filter_k = 0.0;
    double sigma_log_filter_k = 0.0;
    if (filter == "Gaussian_k_eq") {
        // wavenumber position and sigma in log space
        // USER CHANGES filter_k (e.g., k=81 is 1/81th of the box size)
        log_filter_k = log10(filter_k); 
        // FWHM (e.g. 1.7^2=2.9) -> Gaussian filter (this is the full range of the sonic scale uncertainty in terms of a factor)
        sigma_log_filter_k = log10(filter_fwhm)/2.355;
    }

    // apply filter
    int k1 = 0; int k2 = 0; int k3 = 0; // loop over all cells to get k and apply filter
    for (int j = MyInds[0]; j < MyInds[0]+MyInds[1]; j++) // the parallel bit
    {
        if (j <= Dim[X]/2.) k1 = j; else k1 = j-Dim[X];
        for (int l = 0; l < Dim[Y]; l++)
        {
            if (l <= Dim[Y]/2.) k2 = l; else k2 = l-Dim[Y];
            for (int m = 0; m < Dim[Z]; m++)
            {
                if (m <= Dim[Z]/2.) k3 = m; else k3 = m-Dim[Z];

                long k_sqr = k1*k1 + k2*k2 + k3*k3;
                double filter_factor = 1.0;

                if (filter == "TopHat_keep_k_lt") {
                    double abs_k = sqrt(k_sqr);
                    if (abs_k > filter_k) filter_factor = 0.0;
                }
                if (filter == "TopHat_keep_k_gt") {
                    double abs_k = sqrt(k_sqr);
                    if (abs_k < filter_k) filter_factor = 0.0;
                }
                if (filter == "Gaussian_k_eq") {
                    double log_k = log10(sqrt(k_sqr));
                    filter_factor = exp( -0.5 * (log_k-log_filter_k)*(log_k-log_filter_k) / (sigma_log_filter_k*sigma_log_filter_k) );
                }

                long n = (j-MyInds[0])*Dim[Y]*Dim[Z] + l*Dim[Z] + m; // row-major
                fft_data_x[n][0] = fft_data_x[n][0] * filter_factor / TotalNumberOfDataPointsDouble; // real part
                fft_data_x[n][1] = fft_data_x[n][1] * filter_factor / TotalNumberOfDataPointsDouble; // imag part

            } // j
        } // l
    } // m

    // INVERSE FFTW
    fftwf_execute(fft_plan_x_inv);

    // overwrite data with fitered data
    for (long n=0; n<ntot_local; n++) {
        data[n] = fft_data_x[n][0]; /// only Real part 
    }
    if (datatsetname == "dens") { // undo earlier log(dens)
        for (long n=0; n<ntot_local; n++) data[n] = exp(data[n]);
    }

    /// write out filtered data
    long starttime_write = time(NULL);
    ostringstream ss; ss << filter_k;
    string datasetname_out = "dens_"+filter+"_"+ss.str(); ss.clear(); ss.str("");
    if (filter == "Gaussian_k_eq") {
        ss << filter_fwhm; datasetname_out += "_FWHM_"+ss.str(); ss.clear(); ss.str("");
    }
    if (write_parallel ==  "slabs") WriteParallelSlabs (inputfile, datasetname_out, data, MyInds); 
    if (write_parallel == "blocks") WriteParallelBlocks(inputfile, datasetname_out, data, MyInds);

    /// deallocate and clean
    delete [] data; ///// DEALLOCATE
    FinaliseFFTW(false);

    long endtime_write = time(NULL);
    int duration_write = endtime_write-starttime_write, duration_write_red = 0;
    MPI_Allreduce(&duration_write, &duration_write_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout<<FuncName+"Time for writing data = "<<duration_write_red<<"s" << endl;	

    if ((MyPE==0) && Debug) cout << FuncName+"exiting." << endl;

    return 0;

} /// =======================================================================


/// InitFFTW ==========================================================
vector<int> InitFFTW(const vector<int> Dim, const bool vector_dataset)
{
    const string FuncName = "InitFFTW: ";

    long starttime = time(NULL);

    const ptrdiff_t N[3] = {Dim[X], Dim[Y], Dim[Z]};
    ptrdiff_t alloc_local = 0, local_n0 = 0, local_0_start = 0;

    fftwf_mpi_init();

    // get local data size and allocate
    alloc_local = fftwf_mpi_local_size_3d(N[X], N[Y], N[Z], MPI_COMM_WORLD, &local_n0, &local_0_start);

    /// ALLOCATE
    if (MyPE==0) cout<<FuncName+"Allocating fft_data..."<<endl;
    fft_data_x = fftwf_alloc_complex(alloc_local);
    if (vector_dataset) {
        fft_data_y = fftwf_alloc_complex(alloc_local);
        fft_data_z = fftwf_alloc_complex(alloc_local);
    }

    /// PLAN
    if (MyPE==0) cout<<FuncName+"fft_plan_fwd..."<<endl;
    fft_plan_x_fwd = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_x, fft_data_x, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    if (vector_dataset) {
        fft_plan_y_fwd = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_y, fft_data_y, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
        fft_plan_z_fwd = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_z, fft_data_z, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    }
    if (MyPE==0) cout<<FuncName+"fft_plan_inv..."<<endl;
    fft_plan_x_inv = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_x, fft_data_x, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    if (vector_dataset) {
        fft_plan_y_inv = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_y, fft_data_y, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
        fft_plan_z_inv = fftwf_mpi_plan_dft_3d(N[X], N[Y], N[Z], fft_data_z, fft_data_z, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    }

    vector<int> MyInds(2);
    MyInds[0] = local_0_start;
    MyInds[1] = local_n0;

    /// parallelisation / decomposition check
    int wrong_decompostion = 0, wrong_decompostion_red = 0;
    if (MyInds[1] != 0) {
        if (N[X] % MyInds[1] != 0) wrong_decompostion = 1;
    } else wrong_decompostion = 1;
    MPI_Allreduce(&wrong_decompostion, &wrong_decompostion_red, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (wrong_decompostion_red > 0) {
        if (MyPE==0) cout<<FuncName+"Error: Number of cores is not multiple of N[X]."<<endl;
        MyInds[0] = -1; MyInds[1] = -1; // return -1
    }

    // time for InitFFTW
    long endtime = time(NULL);
    int duration = endtime-starttime, duration_red = 0;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << FuncName+"Time for initilization = "<<duration_red<<"s" << endl;	

    return MyInds;

} /// ================================================================

/// SetupParallelisation ===============================================
int SetupParallelisation(const vector<int> Dim, const vector<int> MyInds)
{
    /// parallelisation / decomposition check
    int wrong_decompostion = 0, wrong_decompostion_red = 0;
    if (MyInds[1] != 0) { if (Dim[X] % MyInds[1] != 0) wrong_decompostion = 1;
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
    ncells_pb[Y] = Dim[Y];
    ncells_pb[Z] = Dim[Z];
    gg.SetupPseudoBlocks(ncells_pb);
    if (MyPE==0) gg.PrintInfo();

    return 0;
} /// ==================================================================

/// FinaliseFFTW =====================================================
void FinaliseFFTW(const bool vector_dataset)
{
    fftwf_free(fft_data_x);
    fftwf_destroy_plan(fft_plan_x_fwd);
    fftwf_destroy_plan(fft_plan_x_inv);
    if (vector_dataset) {
        fftwf_free(fft_data_y);
        fftwf_free(fft_data_z);
        fftwf_destroy_plan(fft_plan_y_fwd);
        fftwf_destroy_plan(fft_plan_z_fwd);
        fftwf_destroy_plan(fft_plan_y_inv);
        fftwf_destroy_plan(fft_plan_z_inv);
    }
    fftwf_mpi_cleanup();
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


/// WriteParallelBlocks (Block version) =====================================
/// this version defines only a subset of cores (MPI group) to write whole blocks to several files;
/// all-reduce is done before, to gather data from different slabs into each block before the write
void WriteParallelBlocks(const string inputfile, const string datasetname, float * data_ptr, vector<int> MyInds)
{
    const bool Debug = false;

    int NBLK = gg.GetNumBlocks();
    vector<int> NB = gg.GetNumCellsInBlock();
    vector< vector <vector<double> > > BB = gg.GetBoundingBox();
    vector< vector<double> > MinMaxDomain = gg.GetMinMaxDomain();
    vector<double> L = gg.GetL();
    vector<double> D = gg.GetD();
    vector<int> N = gg.GetN();

    /// find local affected blocks for this CPU
    vector<int> MyBlocks(0);
    double xl = (double)(MyInds[0])/(double)(N[X])*L[X]+MinMaxDomain[X][0];
    double xr = (double)(MyInds[0]+MyInds[1])/(double)(N[X])*L[X]+MinMaxDomain[X][0];
    //if (Debug) cout<<"["<<MyPE<<"] xl= "<<xl<<" xr="<<xr<<endl;
    for (int b=0; b<NBLK; b++)
    {
        if ( ((xl <= BB[b][X][0]) && (BB[b][X][0] <  xr)) ||
             ((xl <  BB[b][X][1]) && (BB[b][X][1] <= xr)) ||
             ((xl >= BB[b][X][0]) && (BB[b][X][1] >= xr)) )
                 MyBlocks.push_back(b);
    }
    if (Debug) {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(1000*(MyPE+1));
        cout<<"["<<MyPE<<"] WriteParallelBlocks: My blocks = ";
        for (unsigned int b=0; b<MyBlocks.size(); b++) cout<<MyBlocks[b]<<" ";
        cout << endl;
    }

    /// create dataset, if it does not exist already
    vector<int> Dimensions(3);
    Dimensions[0] = NB[Z];
    Dimensions[1] = NB[Y];
    Dimensions[2] = NB[X];

    /// SwapMemOrder
    if (MyPE==0 && Debug) cout << "WriteParallelBlocks: Swapping mem order of FFTW data..." << endl;
    vector<int> Dims(3); Dims[0] = N[Z]; Dims[1] = N[Y]; Dims[2] = MyInds[1];
    SwapMemOrder(data_ptr, Dims);

    // number of blocks in X and per core
    const int NBinX = N[X] / NB[X];
    // decomposition check
    bool wrong_decomposition_for_writing = false;
    if (NPE <= NBinX) if (NBinX % NPE != 0) wrong_decomposition_for_writing = true;
    if (NPE >  NBinX) if (NPE % NBinX != 0) wrong_decomposition_for_writing = true;
    if (wrong_decomposition_for_writing) {
        if (MyPE==0) cout << "WriteParallelBlocks: ERROR. Number of cores (or blocks in X) must be integer multiple of blocks in X (or cores)." << endl;
        MPI_Finalize();
        exit(0);
    }
    const double BPCX = (double)NBinX / (double)NPE;
    // cores per block
    const int CPB = round ( 1.0 / BPCX );
    if (MyPE==0) cout << "WriteParallelBlocks: Number of blocks in X direction (and per core): " << NBinX << " (" << BPCX << ")" << endl;
    if (MyPE==0) cout << "WriteParallelBlocks: Cores per block: " << CPB << endl;

    // create write MPI comm group
    vector<int> MasterPE_write(0);
    MasterPE_write.push_back(0); // the Master PE is always one of the writers
    for (int b=0; b<NBLK; b++) {
        const int MasterPE_for_block = floor ( (double)( b % NBinX ) / BPCX );
        bool in_list = false;
        for (unsigned int n = 0; n < MasterPE_write.size(); n++)
        if (MasterPE_write[n] == MasterPE_for_block) {
            in_list = true; break;
        }
        if (!in_list) MasterPE_write.push_back(MasterPE_for_block);
    }
    const int NPE_writers = MasterPE_write.size();
    if (Debug) cout<<"["<<MyPE<<"] WriteParallelBlocks: NPE_writers = " << NPE_writers << endl;
    vector<int> MasterPE_write_list(NPE_writers);
    for (int n = 0; n < NPE_writers; n++)
        MasterPE_write_list[n] = MasterPE_write[n];

    if (Debug) {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(1000*(MyPE+1));
        cout<<"["<<MyPE<<"] WriteParallelBlocks: MasterPE_write_list = ";
        for (int n = 0; n < NPE_writers; n++) cout << " " << MasterPE_write_list[n];
        cout << endl;
    }

    // create subdirectory for all the block files to be dumped into
    string output_dir_name = inputfile+"_blocks/";
    if (MyPE==0) {
        if (mkdir(output_dir_name.c_str(), 0777) == -1)
            cerr << "Warning about '"+output_dir_name+"': " << strerror(errno) << endl;
        else
            cout << "Directory created: '"+output_dir_name+"'" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    const int n_cells_in_block = NB[X]*NB[Y]*NB[Z];
    float *block_data = new float[n_cells_in_block];

    bool printed_progress_1 = false, printed_progress_10 = false, printed_progress_100 = false;
    for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
    {
        // init/reset block data to zero
        for (int n = 0; n < n_cells_in_block; n++) block_data[n] = 0.0;
        
        const int b = MyBlocks[ib];
        vector<int> ind_xl = gg.CellIndexBlock(b, xl+D[X]/2., BB[b][Y][0]+D[Y]/2., BB[b][Z][0]+D[Z]/2.);
        vector<int> ind_xr = gg.CellIndexBlock(b, xr-D[X]/2., BB[b][Y][1]-D[Y]/2., BB[b][Z][1]-D[Z]/2.);
        vector<int> ind_bl = gg.CellIndexBlock(b, BB[b][X][0]+D[X]/2., BB[b][Y][0]+D[Y]/2., BB[b][Z][0]+D[Z]/2.);
        vector<int> ind_br = gg.CellIndexBlock(b, BB[b][X][1]-D[X]/2., BB[b][Y][1]-D[Y]/2., BB[b][Z][1]-D[Z]/2.);
        vector<int> ind_dl = gg.CellIndexDomain(BB[b][X][0]+D[X]/2., BB[b][Y][0]+D[Y]/2., BB[b][Z][0]+D[Z]/2.);
        //if (Debug) cout<<"ind_dl="<<ind_dl[X]<<" "<<ind_dl[Y]<<" "<<ind_dl[Z]<<endl;

        int is = ind_bl[X];
        if (ind_xl[X] > is) is = ind_xl[X];
        int ie = ind_br[X];
        if (ind_xr[X] < ie) ie = ind_xr[X];

        for (int k=0; k<NB[Z]; k++)
            for (int j=0; j<NB[Y]; j++)
                for (int i=is; i<=ie; i++) {
                    long data_index = (ind_dl[Z]+k)*N[Y]*MyInds[1] + (ind_dl[Y]+j)*MyInds[1] + (ind_dl[X]-MyInds[0]+i);
                    long block_index = k*NB[X]*NB[Y] + j*NB[X] + i;
                    block_data[block_index] = data_ptr[data_index];
        }

        const int MasterPE_for_block = floor ( (double)( b % NBinX ) / BPCX );
        //cout<<"["<<MyPE<<"] WriteParallel: MasterPE_for_block ("<<b<<") = " << MasterPE_for_block << endl;
        vector<int> PEs_for_block(CPB); for (int n = 0; n < CPB; n++) PEs_for_block[n] = n + MasterPE_for_block;
        if (Debug) {
            MPI_Barrier(MPI_COMM_WORLD);
            usleep(1000*(MyPE+1)*(b+1));
            cout<<"["<<MyPE<<"] WriteParallelBlocks: PEs_for_block ("<<b<<") = ";
            for (int n = 0; n < CPB; n++) cout << " " << PEs_for_block[n];
            cout << endl;
        }
        
        // all-reduce this block across all processors that are PE_for_(this)_block
        // Construct a group containing all of the reduce PEs from world_group
        MPI_Group reduce_group;
        MPI_Group_incl(world_group, CPB, PEs_for_block.data(), &reduce_group);
        // Create a new communicator based on the group
        MPI_Comm reduce_comm;
        MPI_Comm_create_group(MPI_COMM_WORLD, reduce_group, 1, &reduce_comm);
        // all reduce the full block with each proc's contribution
        if (reduce_comm != MPI_COMM_NULL) {
            MPI_Allreduce(MPI_IN_PLACE, block_data, n_cells_in_block, MPI_FLOAT, MPI_SUM, reduce_comm);
            // free reduce group
            MPI_Group_free(&reduce_group);
            MPI_Comm_free(&reduce_comm);
        }

        // write file for each block
        if (MyPE == MasterPE_for_block) {
            if (Debug) cout<<"["<<MyPE<<"] WriteParallelBlocks: MasterPE_for_block ("<<b<<") = " << MasterPE_for_block << endl;
            stringstream dummystream;
            dummystream << setfill('0') << setw(6) << b;
            string block_string = dummystream.str(); dummystream.clear();
            HDFIO HDFoutput = HDFIO();
            string output_filename = output_dir_name+datasetname+"_block_"+block_string;
            cout<<"["<<MyPE<<"] WriteParallelBlocks: writing to file '"<<output_filename<<"'"<<endl;
            HDFoutput.create(output_filename, MPI_COMM_NULL);
            HDFoutput.write(block_data, datasetname, Dimensions, H5T_NATIVE_FLOAT);
            HDFoutput.close();
        }

        // write progress
        double percent_done = (double)(ib+1)/MyBlocks.size()*100;
        bool print_progress = false;
        if (percent_done >    1.0 && !printed_progress_1  ) {print_progress=true; printed_progress_1  =true;}
        if (percent_done >   10.0 && !printed_progress_10 ) {print_progress=true; printed_progress_10 =true;}
        if (percent_done == 100.0 && !printed_progress_100) {print_progress=true; printed_progress_100=true;}
        if (print_progress && MyPE==0) cout<<"["<<MyPE<<"] WriteParallelBlocks: "<<percent_done<<"% done."<<endl;

    } // ib
    
    if (Debug) {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(1000*(MyPE+1));
        cout<<"["<<MyPE<<"] WriteParallelBlocks: freeing write_comm now..."<<endl;
    }

    MPI_Group_free(&world_group);

    delete [] block_data;

    /// setting "unknown names" will have to be done by block-merging tool

} /// ==========================================================================


/// WriteParallelSlabs (slab version) ===============================================
/// this turned out to be slow when run on many processors; I think this is because of slow file I/O
/// in case of many MPI cores trying to write small(er) chunks in parallel to the same file
void WriteParallelSlabs(const string inputfile, const string datasetname, float * data_ptr, vector<int> MyInds)
{
    const bool Debug = false;
	
    int NBLK = gg.GetNumBlocks();
    vector<int> NB = gg.GetNumCellsInBlock();
    vector< vector <vector<double> > > BB = gg.GetBoundingBox();
    vector< vector<double> > MinMaxDomain = gg.GetMinMaxDomain();
    vector<double> L = gg.GetL();
    vector<double> D = gg.GetD();
    vector<int> N = gg.GetN();

    /// find local affected blocks for this CPU
    vector<int> MyBlocks(0);
    double xl = (double)(MyInds[0])/(double)(N[X])*L[X]+MinMaxDomain[X][0];
    double xr = (double)(MyInds[0]+MyInds[1])/(double)(N[X])*L[X]+MinMaxDomain[X][0];
    if (Debug) cout<<"["<<MyPE<<"] WriteParallelSlabs: xl= "<<xl<<" xr="<<xr<<endl;
    for (int b=0; b<NBLK; b++)
    {
        if ( ((xl <= BB[b][X][0]) && (BB[b][X][0] <  xr)) ||
             ((xl <  BB[b][X][1]) && (BB[b][X][1] <= xr)) ||
             ((xl >= BB[b][X][0]) && (BB[b][X][1] >= xr)) )
                 MyBlocks.push_back(b);
    }
    if (Debug) {
        cout<<"["<<MyPE<<"] WriteParallelSlabs: My blocks = ";
        for (unsigned int b=0; b<MyBlocks.size(); b++) cout<<MyBlocks[b]<<" ";
        cout << endl;
    }

    /// create dataset, if it does not exist already
    vector<int> Dimensions; Dimensions.resize(4);
    Dimensions[0] = NBLK;
    Dimensions[1] = NB[Z];
    Dimensions[2] = NB[Y];
    Dimensions[3] = NB[X];
    if (MyPE==0) cout << "WriteParallelSlabs: Creating block var '"<<datasetname<<"'..." << endl;
    gg.CreateDataset(datasetname, Dimensions);
    MPI_Barrier(MPI_COMM_WORLD);
    if (MyPE==0) cout << "WriteParallelSlabs: ...done creating." << endl;

    /// SwapMemOrder
    if (MyPE==0 && Debug) cout << "WriteParallelSlabs: Swapping mem order of FFTW data..." << endl;
    vector<int> Dims(3); Dims[0] = N[Z]; Dims[1] = N[Y]; Dims[2] = MyInds[1];
    SwapMemOrder(data_ptr, Dims);

    bool printed_progress_1 = false, printed_progress_10 = false, printed_progress_100 = false;
    for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
    {
        int b = MyBlocks[ib];
        vector<int> ind_xl = gg.CellIndexBlock(b, xl+D[X]/2., BB[b][Y][0]+D[Y]/2., BB[b][Z][0]+D[Z]/2.);
        vector<int> ind_xr = gg.CellIndexBlock(b, xr-D[X]/2., BB[b][Y][1]-D[Y]/2., BB[b][Z][1]-D[Z]/2.);
        vector<int> ind_bl = gg.CellIndexBlock(b, BB[b][X][0]+D[X]/2., BB[b][Y][0]+D[Y]/2., BB[b][Z][0]+D[Z]/2.);
        vector<int> ind_br = gg.CellIndexBlock(b, BB[b][X][1]-D[X]/2., BB[b][Y][1]-D[Y]/2., BB[b][Z][1]-D[Z]/2.);
        vector<int> ind_dl = gg.CellIndexDomain(BB[b][X][0]+D[X]/2., BB[b][Y][0]+D[Y]/2., BB[b][Z][0]+D[Z]/2.);
        if (Debug) cout<<"WriteParallelSlabs: ind_dl="<<ind_dl[X]<<" "<<ind_dl[Y]<<" "<<ind_dl[Z]<<endl;

        int is = ind_bl[X];
        if (ind_xl[X] > is) is = ind_xl[X];
        int ie = ind_br[X];
        if (ind_xr[X] < ie) ie = ind_xr[X];

        int count_x = ie-is+1;
        float * slab_data = new float[count_x*NB[Y]*NB[Z]];

        for (int k=0; k<NB[Z]; k++)
            for (int j=0; j<NB[Y]; j++)
                for (int i=is; i<=ie; i++) {
                    long data_index = (ind_dl[Z]+k)*N[Y]*MyInds[1] + (ind_dl[Y]+j)*MyInds[1] + (ind_dl[X]-MyInds[0]+i);
                    long slab_index = k*count_x*NB[Y] + j*count_x + i-is;
                    slab_data[slab_index] = data_ptr[data_index];
        }

        hsize_t offset[4] = {(hsize_t)b, 0, 0, (hsize_t)is};
        hsize_t count[4] = {1, (hsize_t)NB[Z], (hsize_t)NB[Y], (hsize_t)count_x};
        hsize_t out_offset[3] = {0, 0, 0};
        hsize_t out_count[3] = {(hsize_t)NB[Z], (hsize_t)NB[Y], (hsize_t)count_x};
        gg.GetHDFIO().overwrite_slab(slab_data, datasetname, H5T_NATIVE_FLOAT, offset, count, 3, out_offset, out_count, MPI_COMM_WORLD);

        delete [] slab_data;

        // write progress
        double percent_done = (double)(ib+1)/MyBlocks.size()*100;
        bool print_progress = false;
        if (percent_done >    1.0 && !printed_progress_1  ) {print_progress=true; printed_progress_1  =true;}
        if (percent_done >   10.0 && !printed_progress_10 ) {print_progress=true; printed_progress_10 =true;}
        if (percent_done == 100.0 && !printed_progress_100) {print_progress=true; printed_progress_100=true;}
        if (print_progress && MyPE==0) cout<<"["<<MyPE<<"] WriteParallelSlabs: "<<percent_done<<"% done."<<endl;

    } // ib

    if (MyPE==0) cout<<"WriteParallelSlabs: "+datasetname+" written in "+inputfile<<endl;

    /// extend "unknown names" with filtered dataset name
    vector<string> unknown_names = gg.ReadUnknownNames();
    bool datasetname_exists = false;
    for (unsigned int i = 0; i < unknown_names.size(); i++)
    if (unknown_names[i] == datasetname) {
        datasetname_exists = true;
        if (MyPE==0) cout << "WriteParallelSlabs: WARNING: '" << datasetname << "' already in 'unknown names'" << endl;
        break;
    }
    if (!datasetname_exists) unknown_names.push_back(datasetname);
    gg.OverwriteUnknownNames(unknown_names);

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


/** --------------------- Normalize -----------------------------------------------
** divide by norm
** ------------------------------------------------------------------------------- */
void Normalize(float * const data_array, const long n, const double norm)
{
    for (long i = 0; i < n; i++) data_array[i] /= norm;
} /// =======================================================================


/** --------------------- ComputeMean -----------------------------------------------
** compute the mean
** ------------------------------------------------------------------------------- */
double ComputeMean(float * const data_array, const long n)
{
    double mean_loc = 0.0, mean_glob = 0.0;
    long n_loc = 0, n_glob = 0;
    for (long i = 0; i < n; i++) mean_loc += data_array[i];
    MPI_Allreduce(&mean_loc, &mean_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    n_loc += n;
    MPI_Allreduce(&n_loc, &n_glob, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    mean_glob /= (double)n_glob;
    return mean_glob;
} /// =======================================================================


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
        if (Argument[i] != "" && Argument[i] == "-vector_potential")
        {
            vector_pot = true;
        }
        if (Argument[i] != "" && Argument[i] == "-decompose")
        {
            if (Argument.size()>i+1) decompose = Argument[i+1]; else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-filter")
        {
            if (Argument.size()>i+1) filter = Argument[i+1]; else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-filter_k")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> filter_k; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-filter_fwhm")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> filter_fwhm; dummystream.clear();
            } else return -1;
            if (filter == "Gaussian_k_eq") {
                if (MyPE==0) { cout << "ParseInputs: using filter_fwhm " << filter_fwhm << endl; }
            } else {
                if (MyPE==0) { cout << endl << "ParseInputs: filter must be 'Gaussian_k_eq' when using filter_fwhm." << endl; }
                return -1;
            }
        }
        if (Argument[i] != "" && Argument[i] == "-write_parallel")
        {
            if (Argument.size()>i+1) write_parallel = Argument[i+1]; else return -1;
            if (write_parallel == "slabs" || write_parallel == "blocks") {
                if (MyPE==0) { cout << "ParseInputs: using write_parallel " << write_parallel << endl; }
            } else {
                if (MyPE==0) { cout << endl << "ParseInputs: Invalid write_parallel type." << endl; }
                return -1;
            }
        }
    } // loop over all args

    // check that a valid filter and/or decomposition type was set
    if (filter == "none" || filter == "TopHat_keep_k_gt" || filter == "TopHat_keep_k_lt" || filter == "Gaussian_k_eq") {
        if (MyPE==0) { cout << "ParseInputs: using filter " << filter << endl; }
    } else return -1;
    if (decompose == "none" || decompose == "vel") {
        if (MyPE==0) { cout << "ParseInputs: using decompose " << decompose << endl; }
    } else return -1;

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
            << " spectra_process <filename> [<OPTIONS>]" << endl << endl
            << "   <OPTIONS>:           " << endl
            << "     -vector_potential        : Compute the vector potential A from the B field using the Coulomb gauge (reminder: B = curl(A))"  << endl
            << "     -decompose <string>      : ['vel'] Helmholtz decomposition (computes and writes back longitudinal vector component)"  << endl
            << "     -filter <string>         : ['TopHat_keep_k_gt', 'TopHat_keep_k_lt', 'Gaussian_k_eq'] Apply Fourier filter" << endl
            << "     -filter_k <double>       : Wave number position of the filter (default 0)" << endl
            << "     -filter_fwhm <double>    : FWHM of the Gaussian filter; only applies if filter 'Gaussian_k_eq' (default 0)" << endl
            << "     -write_parallel <string> : ['blocks', 'slabs'] (default 'blocks'; note that in case of many MPI tasks, write_parallel 'blocks' is preferred)" << endl
            << endl
            << "Example: spectra_process DF_hdf5_plt_cnt_0020 -filter Gaussian_k_eq -filter_k 81 -filter_fwhm 1.7" << endl
            << "(note that for best performance it is recommended to use N[X] cores and '-write_parallel blocks', followed by merge_split_block_files)"
            << endl << endl;
    }
}

