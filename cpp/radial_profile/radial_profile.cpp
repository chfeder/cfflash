/*

  radial_profile.cpp

  Computes radial profiles of a grid variable from FLASH (AMR or UG) output files

  By Christoph Federrath, 2020

*/

#include "mpi.h" /// MPI lib
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm> /// min and max finding
#include "../Libs/FlashGG.h" /// Flash General Grid class

// constants
using namespace std;
enum {X, Y, Z};
static const bool Debug = false;
static const int MAX_NUM_BINS = 4096;
static bool print_info = true;

// MPI stuff
int MyPE = 0, NPE = 1;

// some global stuff (inputs)
vector<int> N;
string inputfile = "";
string datasetname = "dens";
double BinWidth = 0.0, MinRad = 0.0, MaxRad = 0.0;
int    n_bins = 0;
vector<double> Center(3);
bool   CenterSet = false;
string weight_name = "";
bool   WeightingFlagSet = false;
bool   LogFlagSet = false;
bool   LgDataFlagSet = false;
bool   DataScaleSet = false;
double DataScale = 1.0;
bool   use_output_file = false;
string OutputFilename = "";
string OutputPath = "";

vector<string> OutputFileHeader;
vector< vector<double> > WriteOutTable;

// global profile containers
double dat1_binsum[MAX_NUM_BINS]; // binned data
double dat2_binsum[MAX_NUM_BINS]; // binned data^2
double norm_binsum[MAX_NUM_BINS]; // binned volume

// forward functions
void ScaleData(float* const data_array, const long size, const double scale_value);
void FillWriteOutTable(const double bin_width, const double min_rad, const double max_rad, const bool log_flag);
void WriteOutAnalysedData(const string OutputFilename);
int ParseInputs(const vector<string> Argument);
void HelpMe(void);


/// --------
///   MAIN
/// --------
int main(int argc, char * argv[])
{
    /// start MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NPE);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
    if (MyPE==0) cout<<"=== radial_profile === using MPI num procs: "<<NPE<<endl;

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

    /// set flags
    if (weight_name != "") WeightingFlagSet = true;
    if (WeightingFlagSet)
        if (MyPE==0) cout << "Computing weighted profile; weighting quantity: " << weight_name << endl;
    if (DataScaleSet)
        if (MyPE==0) cout << "Dividing by data scale = " << DataScale << endl;
    if (LgDataFlagSet)
        if (MyPE==0) cout << "Computing base-10 logarithm of " << datasetname << endl;

    /// set output filename
    if (!use_output_file)
    {
        string logbins_str = "";
        if (LogFlagSet) logbins_str = "_lgbins";
        string weight_str = "";
        if (WeightingFlagSet) weight_str = "_weighted";
        string lgdata_str = "";
        if (LgDataFlagSet) lgdata_str = "_lg";
        OutputFilename = OutputPath + inputfile + "_" + datasetname + lgdata_str + weight_str + ".profile" + logbins_str;
    } // end: set OutputFilename

    /// clear radial profile containers
    for (int j = 0; j < MAX_NUM_BINS; j++) {
        dat1_binsum[j] = 0.0;
        dat2_binsum[j] = 0.0;
        norm_binsum[j] = 0.0;
    }

    /// FLASH file meta data
    FlashGG gg = FlashGG(inputfile);
    int NumDims = gg.GetNumDims();
    vector<int> NB = gg.GetNumCellsInBlock();
    vector< vector<double> > D = gg.GetDblock();

    if (MyPE==0) gg.PrintInfo();

    /// decompose domain in blocks
    vector<int> MyBlocks = gg.GetMyBlocks();

    /// show center
    if (MyPE==0) cout << "Center coordinates for radial binning =";
    for (int dim = 0; dim < NumDims; dim++) {
        if (!CenterSet) Center[dim] = 0.0;
        if (MyPE==0) cout << " " << Center[dim];
    }
    if (MyPE==0) cout << endl;

    /// loop over all blocks and fill PDF containers with local data
    for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
    {
        int b = MyBlocks[ib];

        // number of cells in block
        long number_of_cells_in_block = 1;
        for (int dim = 0; dim < NumDims; dim++) number_of_cells_in_block *= NB[dim];

        // get cell volume for this block
        double cell_volume = 1.0;
        for (int dim = 0; dim < NumDims; dim++) cell_volume *= D[b][dim];

        float *block_data = 0;
        float *block_weig = 0;

        /// read block data
        block_data = gg.ReadBlockVar(b, datasetname);

        if (WeightingFlagSet) block_weig = gg.ReadBlockVar(b, weight_name);

        if (DataScaleSet) ScaleData(block_data, number_of_cells_in_block, DataScale);

        if (LgDataFlagSet)
        {
            for (long n = 0; n < number_of_cells_in_block; n++)
                block_data[n] = log10((double)block_data[n]); //base-10 logarithm
        }

        /// === prepare radial binning ===
        double log_min_rad = 0.0;
        /// logarithmic binning
        if (LogFlagSet)
        {
            if (MyPE==0 && print_info) cout << "Doing logarithmic binning..." << endl;
            n_bins = static_cast<int>(log10(MaxRad/MinRad)/BinWidth);
            log_min_rad = log10(MinRad);
            if (MyPE==0 && print_info) {
                cout << "BinWidth (log)                      = " << BinWidth << endl;
                cout << "MaxRad / MinRad                     = " << MaxRad / MinRad << endl;
                cout << "log10( MaxRad / MinRad )            = " << log10( MaxRad / MinRad) << endl;
                cout << "log10( MaxRad / MinRad ) / BinWidth = " << log10( MaxRad / MinRad) / BinWidth << endl;
                cout << "log10( MinRad )                     = " << log_min_rad << endl;
                cout << "number of bins                      = " << n_bins << endl;
            }
        }
        else // linear binning
        {
            if (MyPE==0 && print_info) cout << "Doing linear binning..." << endl;
            n_bins = static_cast<int>((MaxRad - MinRad) / BinWidth);
            if (MyPE==0 && print_info) {
                cout << "BinWidth (linear)                   = " << BinWidth << endl;
                cout << "number of bins                      = " << n_bins << endl;
            }
        }
        assert (n_bins < MAX_NUM_BINS);

        /// === do radial binning ===
        for (long n = 0; n < number_of_cells_in_block; n++)
        {
            // get cell center coordinates
            vector<double> cell_center = gg.CellCenter(b,n);

            // compute radius
            double radius = 0.0;
            for (int dim = 0; dim < NumDims; dim++) radius += (cell_center[dim]-Center[dim])*(cell_center[dim]-Center[dim]);
            radius = sqrt(radius);

            // compute bin index from radius
            int bin_index = 0;
            if (LogFlagSet) // logarithmic binning
                bin_index = static_cast<int>((log10(radius) - log_min_rad) / BinWidth) + 1;
            else
                bin_index = static_cast<int>((radius - MinRad) / BinWidth) + 1;

            // find bin
            if (bin_index > n_bins) continue; // skip loop cycle
            if (bin_index < 1) dat1_binsum[0] += 1; // error bin
            else // actually do the right stuff
            {
                double data = (double)block_data[n];
                double norm = cell_volume;
                if (WeightingFlagSet) norm *= (double)block_weig[n]; // weighted profile
                dat1_binsum[bin_index] += data     *norm;
                dat2_binsum[bin_index] += data*data*norm;
                norm_binsum[bin_index] += norm;
            }
        }

        print_info = false;

        /// clean up
        delete [] block_data; if (WeightingFlagSet) delete [] block_weig;

    } //end loop over blocks

    /// sum up each CPUs contribution
    double tmp[MAX_NUM_BINS];
    for (int n = 0; n < MAX_NUM_BINS; n++) tmp[n] = dat1_binsum[n];
    MPI_Allreduce(tmp, dat1_binsum, MAX_NUM_BINS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int n = 0; n < MAX_NUM_BINS; n++) tmp[n] = dat2_binsum[n];
    MPI_Allreduce(tmp, dat2_binsum, MAX_NUM_BINS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int n = 0; n < MAX_NUM_BINS; n++) tmp[n] = norm_binsum[n];
    MPI_Allreduce(tmp, norm_binsum, MAX_NUM_BINS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /// write data (Master PE only)
    if (MyPE==0)
    {
        FillWriteOutTable(BinWidth, MinRad, MaxRad, LogFlagSet);
        WriteOutAnalysedData(OutputFilename);
    }

    /// print out wallclock time used
    long endtime = time(NULL);
    int duration = endtime-starttime, duration_red = 0;
    if (Debug) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << "****************** Global time to finish = "<<duration_red<<"s ******************" << endl;

    MPI_Finalize();
    return 0;

} // end main


void FillWriteOutTable(const double bin_width, const double min_rad, const double max_rad, const bool log_flag)
{
    // prepare output table size
    WriteOutTable.resize(n_bins); /// output has n_bins lines
    for (unsigned int i = 0; i < WriteOutTable.size(); i++)
        WriteOutTable[i].resize(3); /// output has 3 columns

    // loop over all bins
    for (int i = 1; i <= n_bins; i++)
    {
        /// recompute radial bins
        double radius = 0.0;
        if (log_flag) { /// logarithmic binning
            radius = max_rad * pow(10.0, static_cast<double>((-bin_width)*(n_bins - i + 0.5)));
        }
        else { /// linear binning
            radius = min_rad + ( i - 0.5 ) * bin_width;
        }
        // compute data value and error
        double mean1 = dat1_binsum[i] / norm_binsum[i]; // mean in bin
        double mean2 = dat2_binsum[i] / norm_binsum[i]; // mean squared in bin
        double stddev = sqrt ( mean2 - mean1*mean1 ); // standard deviation in bin
        // fill outout table
        WriteOutTable[i-1][0] = radius;
        WriteOutTable[i-1][1] = mean1;
        WriteOutTable[i-1][2] = stddev;
    }

    /// write header
    OutputFileHeader.clear();
    stringstream dummystream; dummystream.precision(8);
    dummystream.str(""); // clear content of stringstream
    dummystream << setw(30) << left << "#00_radius" << setw(30) << left << "#01_mean" << setw(30) << left << "#02_stddev";
    OutputFileHeader.push_back(dummystream.str()); dummystream.clear();

} // end: FillWriteOutTable


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
        exit (0);
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
        for (unsigned int row = 0; row < WriteOutTable.size(); row++)             /// data
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
} // end: WriteOutAnalysedData


/** ------------------------- ScaleData ------------------------------
 **  Scales the data in data_array to a certain scale_value
 ** ------------------------------------------------------------------ */
void ScaleData(float* const data_array, const long size, const double scale_value)
{
    for (long n = 0; n < size; n++)
        data_array[n] /= scale_value;
}


/** ------------------------- ParseInputs ----------------------------
 **  Parses the command line Arguments
 ** ------------------------------------------------------------------ */
int ParseInputs(const vector<string> Argument)
{
    stringstream dummystream;

    /// read tool specific options
    if (Argument.size() < 8)
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
        if (Argument[i] != "" && Argument[i] == "-bw")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> BinWidth; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-rmin")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> MinRad; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-rmax")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> MaxRad; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-center")
        {
            if (Argument.size()>i+3) {
                dummystream << Argument[i+1]; dummystream >> Center[0]; dummystream.clear();
                dummystream << Argument[i+2]; dummystream >> Center[1]; dummystream.clear();
                dummystream << Argument[i+3]; dummystream >> Center[2]; dummystream.clear();
                CenterSet = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-scl")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> DataScale; dummystream.clear();
                DataScaleSet = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-lgdata")
        {
            LgDataFlagSet = true;
        }
        if (Argument[i] != "" && Argument[i] == "-log")
        {
            LogFlagSet = true;
        }
        if (Argument[i] != "" && Argument[i] == "-wq")
        {
            if (Argument.size()>i+1) weight_name = Argument[i+1]; else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-o")
        {
            if (Argument.size()>i+1) OutputFilename = Argument[i+1]; else return -1;
            use_output_file = true;
        }

    } // loop over all args

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
        << " radial_profile <filename> -bw <bw> -rmin <rmin> -rmax <rmax> [<OPTIONS>]" << endl << endl
        << "     -bw <BinWidth>          : bin width for radial binning (if -log then -bw is in log)" << endl
        << "     -rmin <MinRad>          : minimum radius for binning" << endl
        << "     -rmax <MaxRad>          : maximum radius for binning" << endl << endl
        << "   <OPTIONS>:           " << endl
        << "     -dset <datasetname>     : datasetname to be processed (default: dens)" << endl
        << "     -center <cx cy cz>      : center coordinates (default 0 0 0)" << endl
        << "     -log                    : use logarithmic binning (default: linear binning)" << endl
        << "     -lgdata                 : take base 10 logarithm of the data" << endl
        << "     -wq <weight_dset>       : compute weighted profile, weighted by <weight_dset>; for mass weighting use '-wq dens'" << endl
        << "     -scl <scale value>      : data conversion through division by <scale value>" << endl
        << "     -o <filename>           : specify output filename" << endl
        << endl
        << "Example: radial_profile MCT_hdf5_plt_cnt_0020 -dset dens -rmin 1e15 -rmax 1e20 -bw 0.1 -log"
        << endl << endl;
    }
}
