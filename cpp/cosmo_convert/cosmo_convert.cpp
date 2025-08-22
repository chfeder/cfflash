///
///  Scale cosmo hdf5 file coords and variables from comoving to proper, MPI version
///
///  written by Christoph Federrath (2021)
///
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
#include "../Libs/FlashGG.h" /// Flash General Grid class

// constants
#define NDIM 3
using namespace std;
enum {X, Y, Z};
static const bool Debug = false;

// MPI stuff
int MyPE = 0, NPE = 1;

// arguments
string inputfile = "";
double redshift = 15.0;

/// forward functions
double ConvertToProperVelocity(double x_comoving, double v_comoving, double redshift);
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
    if (MyPE==0) cout<<"=== cosmo_convert === MPI num procs: "<<NPE<<endl;

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

    string outputfile = inputfile+"_converted_to_proper";

    /// copy input FLASH file to output file; only the Master PE does that
    if (MyPE==0) {
        // see if input file exisits; if so, delete any potentially existing output file, e.g.,
        // because it might have been created in an earlier merging attempt, but wasn't finished writing
        ifstream ifs_infile(inputfile.c_str());
        if (ifs_infile.good()) {
            ifstream ifs_outfile(outputfile.c_str());
            if (ifs_outfile.good()) {
                cout<<"!!! Output file "<<outputfile<<" exists; removing it..."<<endl;
                string cmd = "rm "+outputfile;
                cout<<"calling system command: "<<cmd<<endl;
                system(cmd.c_str());
            }
            ifs_outfile.close();
        } else {
            cout<<"!!! Input file "<<inputfile<<" does not exist. Exiting."<<endl;
            return 0;
        }
        ifs_infile.close();

        // now copy file
        string cmd = "cp "+inputfile+" "+outputfile;
        cout<<"calling system command: "<<cmd<<endl;
        system(cmd.c_str());
        cout<<"Copying done."<<endl;
    }
    /// wait until everybody is here
    MPI_Barrier(MPI_COMM_WORLD);

    /// FLASH file meta data
    FlashGG gg = FlashGG(outputfile, 'w');
    vector<int> NB = gg.GetNumCellsInBlock();
    long int ncells_in_block = NB[X]*NB[Y]*NB[Z];
    //vector< vector <vector<double> > > BB = gg.GetBoundingBox();
    //vector< vector<double> > MinMaxDomain = gg.GetMinMaxDomain();
    //vector<double> L = gg.GetL();

    if (MyPE==0) gg.PrintInfo();

    /// decompose domain in blocks
    vector<int> AllBlocks(0);
    for (int ib=0; ib<gg.GetNumBlocks(); ib++) AllBlocks.push_back(ib);
    vector<int> MyBlocks = gg.GetMyBlocks(MyPE, NPE, AllBlocks);

    MPI_Barrier(MPI_COMM_WORLD);

    /// loop over all my blocks
    int blocks_done = 0, nrep = 1;
    for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
    {
        /// centre block
        int b = MyBlocks[ib];

        //cout << "["<<MyPE<<"] start working on block #" << b << endl;

        float *array = 0;
        long int array_size = 0;
        double scale = 1.0;
        string dsetname = "";

        // BlockSize
        scale = 1.0 / (1.0 + redshift);
        array = gg.ReadBlockSize(b, array_size);
        for (long int i=0; i<array_size; i++) array[i] *= scale; // scale
        gg.OverwriteBlockSize(b, array);
        delete [] array;

        // BoundingBox
        scale = 1.0 / (1.0 + redshift);
        array = gg.ReadBoundingBox(b, array_size);
        for (long int i=0; i<array_size; i++) array[i] *= scale; // scale
        gg.OverwriteBoundingBox(b, array);
        delete [] array;

        // Coordinates
        scale = 1.0 / (1.0 + redshift);
        array = gg.ReadCoordinates(b, array_size);
        for (long int i=0; i<array_size; i++) array[i] *= scale; // scale
        gg.OverwriteCoordinates(b, array);
        delete [] array;

        // dens
        dsetname = "dens";
        scale = pow((1.0 + redshift), 3.0);
        array = gg.ReadBlockVar(b, dsetname);
        for (long int i=0; i<ncells_in_block; i++) array[i] *= scale; // scale
        gg.OverwriteBlockVar(b, dsetname, array);
        delete [] array;

        // velx
        dsetname = "velx";
        array = gg.ReadBlockVar(b, dsetname);
        for (long int i=0; i<ncells_in_block; i++) {
            vector<double> cell_center = gg.CellCenter(b, i); // in comoving coords
            array[i] = ConvertToProperVelocity(cell_center[X], array[i], redshift);
        }
        gg.OverwriteBlockVar(b, dsetname, array);
        delete [] array;
        // vely
        dsetname = "vely";
        array = gg.ReadBlockVar(b, dsetname);
        for (long int i=0; i<ncells_in_block; i++) {
            vector<double> cell_center = gg.CellCenter(b, i); // in comoving coords
            array[i] = ConvertToProperVelocity(cell_center[Y], array[i], redshift);
        }
        gg.OverwriteBlockVar(b, dsetname, array);
        delete [] array;
        // velz
        dsetname = "velz";
        array = gg.ReadBlockVar(b, dsetname);
        for (long int i=0; i<ncells_in_block; i++) {
            vector<double> cell_center = gg.CellCenter(b, i); // in comoving coords
            array[i] = ConvertToProperVelocity(cell_center[Z], array[i], redshift);
        }
        gg.OverwriteBlockVar(b, dsetname, array);
        delete [] array;

        // pres
        dsetname = "pres";
        scale = 1.0 + redshift;
        array = gg.ReadBlockVar(b, dsetname);
        for (long int i=0; i<ncells_in_block; i++) array[i] *= scale; // scale
        gg.OverwriteBlockVar(b, dsetname, array);
        delete [] array;

        // eint
        dsetname = "eint";
        scale = pow((1.0 + redshift), -2.0);
        array = gg.ReadBlockVar(b, dsetname);
        for (long int i=0; i<ncells_in_block; i++) array[i] *= scale; // scale
        gg.OverwriteBlockVar(b, dsetname, array);
        delete [] array;

        // temp
        dsetname = "temp";
        scale = pow((1.0 + redshift), -2.0);
        array = gg.ReadBlockVar(b, dsetname);
        for (long int i=0; i<ncells_in_block; i++) array[i] *= scale; // scale
        gg.OverwriteBlockVar(b, dsetname, array);
        delete [] array;

        // ener (compute as eint + 0.5*v^2 after eint and velx, vely, velz have been converted to proper)
        array = gg.ReadBlockVar(b, "eint");
        float* velx = gg.ReadBlockVar(b, "velx");
        float* vely = gg.ReadBlockVar(b, "vely");
        float* velz = gg.ReadBlockVar(b, "velz");
        for (long int i=0; i<ncells_in_block; i++) {
            array[i] += 0.5*((double)velx[i]*(double)velx[i] + (double)vely[i]*(double)vely[i] + (double)velz[i]*(double)velz[i]);
        }
        gg.OverwriteBlockVar(b, "ener", array);
        delete [] array; delete [] velx; delete [] vely; delete [] velz;

        /// report local progress
        blocks_done++;
        double frac_done = (double)(blocks_done)/(double)(MyBlocks.size());
        if (frac_done >= 0.05*nrep) {
            nrep++;
            cout<<" ["<<setw(5)<<right<<MyPE<<"] blocks_done = "<<setw(6)<<right<<blocks_done<<"; total: "<<setw(3)<<right<<(int)(frac_done*100+0.5)<<"% done."<<endl;
        }

    } //end loop over blocks

    MPI_Barrier(MPI_COMM_WORLD);

    long endtime = time(NULL);
    long duration = endtime-starttime; long duration_red = 0;
    if (Debug) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (MyPE==0) cout << "****************** Global time to finish = "<<duration_red<<"s ******************" << endl;

    if (MyPE==0) cout<<"Output file '"<<outputfile<<"' written."<<endl;

    MPI_Finalize();
    return 0;

} // end main ==========================================================


/** --------------- Convert_to_proper_velocity ----------------------
**  Convert comoving velocity to proper velocity
**  args: x_comoving in cm, v_comoving in cm/s
** ------------------------------------------------------------------ */
double ConvertToProperVelocity(double x_comoving, double v_comoving, double redshift)
{
    double a = 1.0 / (1.0 + redshift);
    double H_0 = 70.3; // km/s/Mpc
    double omega_m = 0.3089;
    double omega_lambda = 0.6911;
    double H_a = H_0 * sqrt(omega_lambda + omega_m/a/a/a); // don't need other terms in the sqrt as long as omega_m+omega_lambda=1.0
    H_a = H_a * 1e5 / 3.086e24; // convert H_a to units of s^-1
    double term1 = H_a * a * x_comoving;
    double term2 = a * v_comoving;
    double v_proper = term1 + term2;
    return v_proper;
}

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
            << " cosmo_convert <filename> [<OPTIONS>]" << endl << endl
            << "   <OPTIONS>:           " << endl
            //<< "     -z : redshift" << endl
            << endl
            << "Example: cosmo_convert CosmoTest_hdf5_chk_0031" << endl
            << endl << endl;
    }
}
