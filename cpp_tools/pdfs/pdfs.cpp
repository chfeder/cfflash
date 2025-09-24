/*

  pdfs.cpp

  Computes PDFs of a grid variable from FLASH (AMR or UG) output files

  By Christoph Federrath, 2013-2025

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
static const bool ShowMinMax = true;
static const int MAX_NUM_BINS = 4096;
static bool print_info = true;

// MPI stuff
int MyPE = 0, NPE = 1;

// some global stuff (inputs)
vector<int> N;
string inputfile = "";
string datasetname = "dens";
double Gamma = -66.;
double BinWidth = 0.0, MinValue = 0.0, MaxValue = 0.0;
int    n_bins = 0;
string weight_name = "";
bool   WeightingFlagSet = false;
bool   LogFlagSet = false;
bool   LgDataFlagSet = false;
bool   LnDataFlagSet = false;
bool   DataScaleSet = false;
double DataScale = 1.0;
bool   use_output_file = false;
string OutputFilename = "";
string OutputPath = "";

vector<string> OutputFileHeader;
vector< vector<double> > WriteOutTable;

// global PDF containers
double data_binsum[MAX_NUM_BINS];

double ave_compare = 0.0;
double rms_compare = 0.0;
double skew_compare = 0.0;
double kurt_compare = 0.0;
double sigma_compare = 0.0;

// forward functions
void InitPDFAndMoments(void);
void AddToPDF(const float* const data_array, const float* const weighting_array,
              const long number_of_datapoints, const double cell_volume,
              const double bin_width, const double min_val, const double max_val,
              const bool log_flag, const bool weight_flag);
void ScaleData(float* const data_array, const long size, const double scale_value);
double ComputeValMin(const float* const data_array, const long size);
double ComputeValMax(const float* const data_array, const long size);
void FillWriteOutTable(const double bin_width, const double min_val, const double max_val, const bool log_flag);
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
    if (MyPE==0) cout<<"=== pdfs === using MPI num procs: "<<NPE<<endl;

    /// Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) == -1)
    {
        if (MyPE==0) cout << endl << "Error in ParseInputs(). Exiting." << endl;
        HelpMe();
        MPI_Finalize(); return 0;
    }

    if ((datasetname == "cs") || (datasetname == "mach"))
    {
        if (Gamma == -66.) // then the value for Gamma has not been specified by the user; so exit and issue error
        {
            if (MyPE==0) cout << endl << "Gamma must be specified if making PDF of cs or mach. Exiting." << endl;
            HelpMe();
            MPI_Finalize(); return 0;
        }
        if (MyPE==0) cout << endl << "Using polytropic Gamma = " << Gamma << endl;
    }

    long starttime = time(NULL);

    /// set flags
    if (weight_name != "") WeightingFlagSet = true;
    if (WeightingFlagSet)
        if (MyPE==0) cout << "Computing weighted PDF; weighting quantity: " << weight_name << endl;
    if (DataScaleSet)
        if (MyPE==0) cout << "Dividing by data scale = " << DataScale << endl;
    if (LgDataFlagSet)
        if (MyPE==0) cout << "Computing base-10 logarithm of " << datasetname << endl;
    if (LnDataFlagSet)
        if (MyPE==0) cout << "Computing natural logarithm of " << datasetname << endl;

    /// set output filename
    if (!use_output_file)
    {
        if (LogFlagSet && !LgDataFlagSet && !LnDataFlagSet)
        {
            if (WeightingFlagSet)
                OutputFilename = OutputPath + inputfile + "_" + datasetname + ".pdf_data_log_w";
            else
                OutputFilename = OutputPath + inputfile + "_" + datasetname + ".pdf_data_log";
        }
        if (!LogFlagSet && !LgDataFlagSet && !LnDataFlagSet)
        {
            if (WeightingFlagSet)
                OutputFilename = OutputPath + inputfile + "_" + datasetname + ".pdf_data_w";
            else
                OutputFilename = OutputPath + inputfile + "_" + datasetname + ".pdf_data";
        }
        if (!LogFlagSet && LgDataFlagSet && !LnDataFlagSet)
        {
            if (WeightingFlagSet)
                OutputFilename = OutputPath + inputfile + "_" + datasetname + ".pdf_lg_data_w";
            else
                OutputFilename = OutputPath + inputfile + "_" + datasetname + ".pdf_lg_data";
        }
        if (!LogFlagSet && !LgDataFlagSet && LnDataFlagSet)
        {
            if (WeightingFlagSet)
                OutputFilename = OutputPath + inputfile + "_" + datasetname + ".pdf_ln_data_w";
            else
                OutputFilename = OutputPath + inputfile + "_" + datasetname + ".pdf_ln_data";
        }
    } // end: set OutputFilename

    /// FLASH file meta data
    FlashGG gg = FlashGG(inputfile);
    vector<int> NB = gg.GetNumCellsInBlock();
    vector< vector<double> > D = gg.GetDblock();

    if (MyPE==0) gg.PrintInfo();

    /// decompose domain in blocks
    vector<int> MyBlocks = gg.GetMyBlocks();

    /// clear PDF containers
    InitPDFAndMoments();

    /// prepare min/max
    double min_loc =  1e99, min_glob =  1e99;
    double max_loc = -1e99, max_glob = -1e99;

    /// loop over all blocks and fill PDF containers with local data
    for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
    {
        int b = MyBlocks[ib];

        long number_of_cells_in_block = NB[X]*NB[Y]*NB[Z];

        float *block_data = 0;
        float *block_weig = 0;

        /// read block data
        if (datasetname == "cs") // local sound speed
        {
            block_data = gg.ReadBlockVar(b, "pres");
            float *dens = gg.ReadBlockVar(b, "dens");
            for (long n = 0; n < number_of_cells_in_block; n++)
            {
                block_data[n] = sqrt( Gamma * (double)block_data[n] / (double)dens[n] );
            }
            delete [] dens;
        }
        else if (datasetname == "mach") // local Mach number
        {
            block_data = gg.ReadBlockVar(b, "pres");
            float *dens = gg.ReadBlockVar(b, "dens");
            float *velx = gg.ReadBlockVar(b, "velx");
            float *vely = gg.ReadBlockVar(b, "vely");
            float *velz = gg.ReadBlockVar(b, "velz");
            for (long n = 0; n < number_of_cells_in_block; n++)
            {
                block_data[n] = sqrt( ( (double)velx[n]*(double)velx[n] +
                                        (double)vely[n]*(double)vely[n] +
                                        (double)velz[n]*(double)velz[n] ) /
                                       ( Gamma * (double)block_data[n] / (double)dens[n] ) );
            }
            delete [] dens; delete [] velx; delete [] vely; delete [] velz;
        }
        else
        {
            block_data = gg.ReadBlockVar(b, datasetname);
        }

        if (WeightingFlagSet) block_weig = gg.ReadBlockVar(b, weight_name);

        if (DataScaleSet) ScaleData(block_data, number_of_cells_in_block, DataScale);

        if (LgDataFlagSet)
        {
            for (long n = 0; n < number_of_cells_in_block; n++)
                block_data[n] = log10((double)block_data[n]); //base-10 logarithm
        }
        if (LnDataFlagSet)
        {
            for (long n = 0; n < number_of_cells_in_block; n++)
                block_data[n] = log((double)block_data[n]); //natural logarithm
        }

        // get min/max value in this block
        if (ShowMinMax)
        {
            double min_blk = ComputeValMin(block_data, number_of_cells_in_block);
            double max_blk = ComputeValMax(block_data, number_of_cells_in_block);
            if (min_blk < min_loc) { min_loc = min_blk; }
            if (max_blk > max_loc) { max_loc = max_blk; }
        }

        // get cell volume for this block
        double cell_volume = D[b][X]*D[b][Y]*D[b][Z];

        // call to making PDF
        AddToPDF(block_data, block_weig, number_of_cells_in_block, cell_volume,
                 BinWidth, MinValue, MaxValue, LogFlagSet, WeightingFlagSet);

        delete [] block_data; if (WeightingFlagSet) delete [] block_weig;

    } //end loop over blocks

    // reduce min/max and print to screen
    if (ShowMinMax)
    {
        MPI_Allreduce(&min_loc, &min_glob, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&max_loc, &max_glob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (MyPE==0) cout << "Min (data) = " << scientific << min_glob << ", Max (data) = " << scientific << max_glob << endl;
    }

    /// sum up each CPUs contribution
    double tmp[MAX_NUM_BINS];
    for (int n = 0; n < MAX_NUM_BINS; n++) tmp[n] = data_binsum[n];
    MPI_Allreduce(tmp, data_binsum, MAX_NUM_BINS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /// write data (Master PE only)
    if (MyPE==0)
    {
        FillWriteOutTable(BinWidth, MinValue, MaxValue, LogFlagSet);
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


void InitPDFAndMoments(void)
{
    /// init PDF containers
    for (int j = 0; j < MAX_NUM_BINS; j++)
        data_binsum[j] = 0.0;

    /// init moments
    ave_compare = 0.0;
    rms_compare = 0.0;
    skew_compare = 0.0;
    kurt_compare = 0.0;
    sigma_compare = 0.0;

} // end: InitPDFAndMoments


/** -------------------------- AddToPDF ----------------------------
 **  Calculation of probability distribution functions
 ** ------------------------------------------------------------------ */
void AddToPDF(const float* const data_array, const float* const weighting_array,
              const long number_of_datapoints, const double cell_volume,
              const double bin_width, const double min_val, const double max_val,
              const bool log_flag, const bool weight_flag)
{

    double log_min_val = 0.0;

    /// logarithmic data analysis
    if (log_flag)
    {
        if (MyPE==0 && print_info) cout << "AddToPDF:  Logarithmic binning..." << endl;
        n_bins = static_cast<int>(log10(max_val/min_val)/bin_width);
        log_min_val = log10(min_val);
        if (MyPE==0 && print_info) {
            cout << "AddToPDF:  bin_width                             = " << bin_width << endl;
            cout << "AddToPDF:  max_val / min_val                     = " << max_val / min_val << endl;
            cout << "AddToPDF:  log10( max_val / min_val)             = " << log10( max_val / min_val) << endl;
            cout << "AddToPDF:  log10( max_val / min_val) / bin_width = " << log10( max_val / min_val) / bin_width << endl;
            cout << "AddToPDF:  log_min_val                           = " << log_min_val << endl;
            cout << "AddToPDF:  number of bins                        = " << n_bins << endl;
        }
    }
    else // linear data analysis
    {
        if (MyPE==0 && print_info) cout << "AddToPDF:  Linear binning..." << endl;
        n_bins = static_cast<int>((max_val - min_val) / bin_width);
        if (MyPE==0 && print_info) {
            cout << "AddToPDF:  bin_width                             = " << bin_width << endl;
            cout << "AddToPDF:  number of bins                        = " << n_bins << endl;
        }
    }

    assert (n_bins < MAX_NUM_BINS);

    int bin_index = 0;

    for (long n = 0; n < number_of_datapoints; n++)
    {
        if (log_flag)
            bin_index = static_cast<int>((log10((double)data_array[n]) - log_min_val) / bin_width) + 1;
        else
            bin_index = static_cast<int>(((double)data_array[n] - min_val) / bin_width) + 1;

        if (bin_index > n_bins) continue; // skip loop cycle
        if (bin_index < 1) data_binsum[0] += cell_volume; // error bin
        else // actually do the right stuff
        {
            if (weight_flag) // weighted PDF
                data_binsum[bin_index] += (double)weighting_array[n]*cell_volume;
            else
                data_binsum[bin_index] += cell_volume;
        }
    }

    print_info = false;

} // end: AddToPDF()


void FillWriteOutTable(const double bin_width, const double min_val, const double max_val, const bool log_flag)
{
    // prepare output table size
    WriteOutTable.resize(n_bins); /// PDF output has n_bins lines
    for (unsigned int i = 0; i < WriteOutTable.size(); i++)
        WriteOutTable[i].resize(4); /// PDF output has 4 columns

    double bw = 0.0, data_value = 0.0;
    double pdf = 0.0, cdf = 0.0;
    double ave = 0.0, rms = 0.0;

    // compute PDF normalisation
    double norm = 0.0;
    for (int i = 1; i <= n_bins; i++)
        norm += data_binsum[i];

    // loop over all bins
    for (int i = 1; i <= n_bins; i++)
    {
        if (log_flag) /// logarithmic binning
        {
            /// normalize data to bin width
            double interval_lb = max_val * pow(10.0, static_cast<double>((-bin_width)*(n_bins - i + 1)));
            double interval_rb = max_val * pow(10.0, static_cast<double>((-bin_width)*(n_bins - i + 0)));
            bw = interval_rb - interval_lb;
            if (Debug)
            {
                cout << "FillWriteOutTable:  interval_lb = " << interval_lb << endl;
                cout << "FillWriteOutTable:  interval_rb = " << interval_rb << endl;
                cout << "FillWriteOutTable:  bw          = " << bw << endl;
            }
            // data value
            data_value = max_val * pow(10.0, static_cast<double>((-bin_width)*(n_bins - i + 0.5)));
        }
        else /// linear binning
        {
            /// normalize data to bin width
            bw = bin_width;
            // data value
            data_value = min_val + ( i - 0.5 ) * bin_width;
        }
        // define PDF and CDF
        pdf = data_binsum[i] / norm / bw;
        cdf = cdf + pdf*bw;
        // fill outout table
        WriteOutTable[i-1][0] = data_value;
        WriteOutTable[i-1][1] = 0.0;
        WriteOutTable[i-1][2] = pdf;
        WriteOutTable[i-1][3] = cdf;
        // average nd RMS from PDF data
        ave  = ave  +            data_value*pdf*bw;
        rms  = rms  + data_value*data_value*pdf*bw;
    }

    /// compute skewness and kurtosis of the PDF
    double skew = 0.0, kurt = 0.0;
    for (int i = 1; i <= n_bins; i++)
    {
        /// writing logarithmic data output to file
        if (log_flag)
        {
            /// normalize data to bin width
            double interval_lb = max_val * pow(10.0, static_cast<double>((-bin_width)*(n_bins - i + 1)));
            double interval_rb = max_val * pow(10.0, static_cast<double>((-bin_width)*(n_bins - i + 0)));
            bw = interval_rb - interval_lb;
        }
        else
        {
            bw = bin_width;
        }
        data_value = WriteOutTable[i-1][0]-ave; ///subtract the mean
        pdf        = WriteOutTable[i-1][2];
        skew = skew + pow(data_value, 3.0)*pdf*bw;
        kurt = kurt + pow(data_value, 4.0)*pdf*bw;
    }

    double rms_pdf = sqrt(rms);
    double sigma_pdf = sqrt( rms - pow(ave, 2.0) );
    double skew_pdf = skew / pow( sigma_pdf, 3.0 );
    double kurt_pdf = kurt / pow( sigma_pdf, 4.0 ) - 3.0;

    cout << "FillWriteOutTable:  Mean                 = " << ave << endl;
    cout << "FillWriteOutTable:  Root Mean Squared    = " << rms_pdf << endl;
    cout << "FillWriteOutTable:  Skewness             = " << skew_pdf << endl;
    cout << "FillWriteOutTable:  Kurtosis             = " << kurt_pdf << endl;
    cout << "FillWriteOutTable:  Sigma                = " << sigma_pdf << endl;

    //cout << "ComputePDF:  Mean (compare...!full dataset!) = " <<   ave_compare << endl;
    //cout << "ComputePDF:  Root Mean Squared               = " <<   rms_compare << endl;
    //cout << "ComputePDF:  Skewness                        = " <<  skew_compare << endl;
    //cout << "ComputePDF:  Kurtosis                        = " <<  kurt_compare << endl;
    //cout << "ComputePDF:  Sigma(!full dataset!...compare) = " << sigma_compare << endl;

    int   ave_check = floor((ave       /   ave_compare)*100.0);
    int   rms_check = floor((rms_pdf   /   rms_compare)*100.0);
    int  skew_check = floor((skew_pdf  /  skew_compare)*100.0);
    int  kurt_check = floor((kurt_pdf  /  kurt_compare)*100.0);
    int sigma_check = floor((sigma_pdf / sigma_compare)*100.0);
    if (    ave_check < 90 ||   ave_check > 110 ||
            rms_check < 90 ||   rms_check > 110 ||
           skew_check < 90 ||  skew_check > 110 ||
           kurt_check < 90 ||  kurt_check > 110 ||
          sigma_check < 90 || sigma_check > 110    ) // cout << "ComputePDF:  WARNING: check statistical values!" << endl;

    /// write a PDF file header containing mean, rms, skewness and standard deviation
    OutputFileHeader.clear();
    stringstream dummystream; dummystream.precision(8); string dummystring = ""; string dummystring2 = "";
    OutputFileHeader.push_back("mean  (PDF, dataset)"); dummystream << scientific << ave; dummystream >> dummystring; dummystream.clear();
    dummystream << scientific << ave_compare; dummystream >> dummystring2; OutputFileHeader.push_back(dummystring+"  "+dummystring2); dummystream.clear();
    OutputFileHeader.push_back("rms   (PDF, dataset)"); dummystream << scientific << rms_pdf; dummystream >> dummystring; dummystream.clear();
    dummystream << scientific << rms_compare; dummystream >> dummystring2; OutputFileHeader.push_back(dummystring+"  "+dummystring2); dummystream.clear();
    OutputFileHeader.push_back("skew  (PDF, dataset)"); dummystream << scientific << skew_pdf; dummystream >> dummystring; dummystream.clear();
    dummystream << scientific << skew_compare; dummystream >> dummystring2; OutputFileHeader.push_back(dummystring+"  "+dummystring2); dummystream.clear();
    OutputFileHeader.push_back("kurt  (PDF, dataset)"); dummystream << scientific << kurt_pdf; dummystream >> dummystring; dummystream.clear();
    dummystream << scientific << kurt_compare; dummystream >> dummystring2; OutputFileHeader.push_back(dummystring+"  "+dummystring2); dummystream.clear();
    OutputFileHeader.push_back("sigma (PDF, dataset)"); dummystream << scientific << sigma_pdf; dummystream >> dummystring; dummystream.clear();
    dummystream << scientific << sigma_compare; dummystream >> dummystring2; OutputFileHeader.push_back(dummystring+"  "+dummystring2); dummystream.clear();
    dummystream.str(""); // clear content of stringstream
    dummystream << setw(30) << left << "#00_GridVariable" << setw(30) << left << "#01_GridVariableErr" << setw(30) << left << "#02_PDF" << setw(30) << left << "#03_CDF";
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

/** ------------------------ ComputeValMin ---------------------------
 **  Computes minimum value in pointer data_array
 ** ------------------------------------------------------------------ */
double ComputeValMin(const float* const data_array, const long size)
{
    //cout << "ComputeValMin:  Searching for minimum data value... ";
    double ReturnValMin = *min_element(data_array, data_array + size);
    //cout << "ComputeValMin:  min data value: " << ReturnValMin << endl;
    return ReturnValMin;
}
/** ------------------------ ComputeValMax ---------------------------
 **  Computes maximum value in pointer data_array
 ** ------------------------------------------------------------------ */
double ComputeValMax(const float* const data_array, const long size)
{
    //cout << "ComputeValMax:  Searching for maximum data value... ";
    double ReturnValMax = *max_element(data_array, data_array + size);
    //cout << "ComputeValMax:  max data value: " << ReturnValMax << endl;
    return ReturnValMax;
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
        if (Argument[i] != "" && Argument[i] == "-vmin")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> MinValue; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-vmax")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> MaxValue; dummystream.clear();
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
        if (Argument[i] != "" && Argument[i] == "-lndata")
        {
            LnDataFlagSet = true;
        }
        if (Argument[i] != "" && Argument[i] == "-log")
        {
            LogFlagSet = true;
        }
        if (Argument[i] != "" && Argument[i] == "-wq")
        {
            if (Argument.size()>i+1) weight_name = Argument[i+1]; else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-gamma")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> Gamma; dummystream.clear();
            } else return -1;
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
        << " pdfs <filename> -bw <bw> -vmin <vmin> -vmax <vmax> [<OPTIONS>]" << endl << endl
        << "     -bw <BinWidth>          : bin width for PDF of dset" << endl
        << "     -vmin <MinValue>        : minimum for the PDF of dset" << endl
        << "     -vmax <MaxValue>        : maximum for the PDF of dset" << endl << endl
        << "   <OPTIONS>:           " << endl
        << "     -dset <datasetname>     : datasetname to be processed (default: dens)" << endl
        << "     -scl <scale value>      : data conversion through division by <scale value>" << endl
        << "     -lgdata                 : take base 10 logarithm of the data" << endl
        << "     -lndata                 : take natural logarithm of the data" << endl
        << "     -log                    : use logarithmic binning (default: linear binning)" << endl
        << "     -wq <weight_dset>       : compute weighted PDFs weighted by <weight_dset>; for mass weighting use '-wq dens'" << endl
        << "     -gamma <Gamma>          : polytropic Gamma exponent for computing sound speed (only if '-dset cs' and 'pres' must be in file)" << endl
        << "     -o <filename>           : specify output filename" << endl
        << endl
        << "Example: pdfs DF_hdf5_plt_cnt_0020 -dset dens -bw 1. -vmin 0. -vmax 1e3 -log"
        << endl << endl;
    }
}
