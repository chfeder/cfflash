///
///  Computes divergence of velocity, vorticity, current of magnetic field,
///           viscous dissipation rate, enstrophy source terms
///
///  Enstrophy source terms are calculated according to Wittor & Gaspari (2020, arxiv:2009.03344).
///  Viscous dissipation rate based on Offner et al. (2009).
///
///  Written by Christoph Federrath, James Beattie, Rajsekhar Mohapatra
///  Current version dated: 2022-2025
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
#include "../Libs/CFTools.h" /// tools

// constants
#define NDIM 3
using namespace std;
enum {X, Y, Z};

// MPI stuff
int MyPE = 0, NPE = 1;

// arguments
string inputfile = "";
bool compute_divv = false;
bool compute_divb = false;
bool compute_vort = false;
bool compute_divrhov = false;
bool compute_current = false;
bool compute_mag_from_vecpot = false;
bool compute_MHD_scales = false;
bool compute_induction = false;
bool compute_dissipation = false;
bool compute_enstrophy_sources = false;
bool compute_grav_accel = false;
bool mw = true; // default is mass-weighted interpolation to get guard cell info (only relevant for AMR)
bool pbc = true; // periodic boundary conditions to get guard cell info
int Verbose = 1;


/// forward functions
void ProcessBlock(const string term, const int ib, const int b, const int ndo, const int ndi, string dno[], string dni[],
                  const vector<int> Dimensions, const vector<int> NB, const int NGC, const vector<int> NBGC,
                  const vector<vector<double> > D, FlashGG gg, vector<string> & dsetnames_for_extending_unknown_names,
                  const double multiplier, const double added_constant);
void ProcessBlock(const string term, const int ib, const int b, const int ndo, const int ndi, string dno[], string dni[],
                  const vector<int> Dimensions, const vector<int> NB, const int NGC, const vector<int> NBGC,
                  const vector<vector<double> > D, FlashGG gg, vector<string> & dsetnames_for_extending_unknown_names);
inline void GetIndices(const int i, const int j, const int k,
                       const vector<int> NB, const vector<int> NBGC, const int NGC,
                       long & index, long & index_gc, long il[3], long ir[3]);
void ComputeTerm(const string term, std::vector<float*>& output, std::vector<float*>& input,
                 const vector<int> NB, const vector<int> NBGC, const vector<double> D, const int NGC);
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

    /// Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) == -1)
    {
        if (MyPE==0) cout << endl << "Error in ParseInputs(). Exiting." << endl;
        HelpMe();
        MPI_Finalize(); return 0;
    }

    if (Verbose && MyPE==0) cout << "=== derivative_var === MPI num procs: " << NPE <<endl;

    long starttime = time(NULL);

    /// FLASH file meta data
    FlashGG gg = FlashGG(inputfile, 'w');
    int NBLK = gg.GetNumBlocks();
    vector<int> NB = gg.GetNumCellsInBlock();
    vector<vector<double> > D = gg.GetDblock();

    /// check to see if vorticity_x, y, z already exist in file if we do enstrophy_sources
    if (compute_enstrophy_sources) {
        vector<string> unknown_names = gg.ReadUnknownNames();
        bool datasetname_exists = false;
        for (unsigned int i = 0; i < unknown_names.size(); i++) {
            if (unknown_names[i] == "vorticity_x") { datasetname_exists = true; break; } // found
        }
        if (!datasetname_exists) {
            if (MyPE==0) cout << "ERROR: vorticity_[x, y, z] do not exist in " << inputfile << ". "
                              << "Run with '-vort' only to create vorticty first, then do -enstrophy_sources." << endl;
            MPI_Finalize(); return 0;
        }
    }

    /// check to see if the current field exists
    if (compute_MHD_scales) {
        vector<string> unknown_names = gg.ReadUnknownNames();
        bool datasetname_exists = false;
        for (unsigned int i = 0; i < unknown_names.size(); i++) {
            if (unknown_names[i] == "curx") { datasetname_exists = true; break; } // found
        }
        if (!datasetname_exists) {
            if (MyPE==0) cout << "ERROR: cur_[x, y, z] do not exist in " << inputfile << ". "
                              << "Run with -current' only to create the current first, then do -MHD_scales." << endl;
            MPI_Finalize(); return 0;
        }
    }

    if (Verbose && MyPE==0) gg.PrintInfo();
    /// decompose domain in blocks
    vector<int> MyBlocks = gg.GetMyBlocks();

    // define block sizes with GCs
    int NGC = 1; // number of ghost cells
    vector<int> NBGC(3);
    NBGC[X] = NB[X]+2*NGC;
    NBGC[Y] = NB[Y]+2*NGC;
    NBGC[Z] = NB[Z]+2*NGC;

    /// dimensions of output dataset
    vector<int> Dimensions; Dimensions.resize(4);
    Dimensions[0] = NBLK;
    Dimensions[1] = NB[Z];
    Dimensions[2] = NB[Y];
    Dimensions[3] = NB[X];

    vector<string> dsetnames_for_extending_unknown_names;

    /// loop over all my blocks
    CFTools cft = CFTools(); // initialise progress bar
    for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
    {
        /// write progress
        if (MyPE==0) cft.PrintProgressBar(ib, MyBlocks.size());

        /// get actual block index
        int b = MyBlocks[ib];

        if (Verbose > 1) cout << "["<<MyPE<<"] start working on block #" << b << endl;

        /// compute divergence of velocity
        if (compute_divv) {
            const int ndo = 1; string dno[ndo] = {"divv"}; // output
            const int ndi = 3; string dni[ndi] = {"velx", "vely", "velz"}; // input
            ProcessBlock("Div", ib, b, ndo, ndi, dno, dni, Dimensions,
                         NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
        }

        /// compute divergence of magnetic field
        if (compute_divb) {
            const int ndo = 1; string dno[ndo] = {"divb"}; // output
            const int ndi = 3; string dni[ndi] = {"magx", "magy", "magz"}; // input
            ProcessBlock("Div", ib, b, ndo, ndi, dno, dni, Dimensions,
                         NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
        }

        /// compute vorticity components
        if (compute_vort) {
            const int ndo = 3; string dno[ndo] = {"vorticity_x", "vorticity_y", "vorticity_z"}; // output
            const int ndi = 3; string dni[ndi] = {"velx", "vely", "velz"}; // input
            ProcessBlock("Curl", ib, b, ndo, ndi, dno, dni, Dimensions,
                         NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
        }

        /// compute divergence of momentum density (density*velocity)
        if (compute_divrhov) {
            const int ndo = 1; string dno[ndo] = {"divrhov"}; // output
            const int ndi = 3; string dni[ndi] = {"momx", "momy", "momz"}; // input
            ProcessBlock("Div", ib, b, ndo, ndi, dno, dni, Dimensions,
                         NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
        }

        /// compute current (curl B) components
        if (compute_current) {
            const int ndo = 3; string dno[ndo] = {"curx", "cury", "curz"}; // output
            const int ndi = 3; string dni[ndi] = {"magx", "magy", "magz"}; // input
            ProcessBlock("Curl", ib, b, ndo, ndi, dno, dni, Dimensions,
                         NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
        }

        /// compute B from vector potential (B = curl A) components
        if (compute_mag_from_vecpot) {
            const int ndo = 3; string dno[ndo] = {"magx_from_vecpot", "magy_from_vecpot", "magz_from_vecpot"}; // output
            const int ndi = 3; string dni[ndi] = {"vecpotx", "vecpoty", "vecpotz"}; // input
            ProcessBlock("Curl", ib, b, ndo, ndi, dno, dni, Dimensions,
                         NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
        }

        /// compute B cross J, B dot J, and the magnetic tension for computing the geometry of the structures
        /// in MHD turbulence (see, e.g., Galishnikova et al. 2022)
        if (compute_MHD_scales) {
            { // compute B cross J
                const int ndo = 3; string dno[ndo] = {"mXcx", "mXcy", "mXcz"}; // output
                const int ndi = 6; string dni[ndi] = {"magx", "magy", "magz","curx", "cury", "curz"}; // input
                ProcessBlock("MagCrossCurrent", ib, b, ndo, ndi, dno, dni, Dimensions,
                            NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
            }
            { // compute B dot J
                const int ndo = 1; string dno[ndo] = {"mdc"}; // output
                const int ndi = 6; string dni[ndi] = {"magx", "magy", "magz","curx", "cury", "curz"}; // input
                ProcessBlock("MagDotCurrent", ib, b, ndo, ndi, dno, dni, Dimensions,
                            NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);

            }
            { // compute magnetic tension
                const int ndo = 3; string dno[ndo] = {"tenx", "teny", "tenz"}; // output
                const int ndi = 3; string dni[ndi] = {"magx", "magy", "magz"}; // input
                ProcessBlock("Tension", ib, b, ndo, ndi, dno, dni, Dimensions,
                            NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
            }
        }

        /// compute curl(v cross B), induction term of the MHD induction equation
        if (compute_induction) {
            const int ndo = 3; string dno[ndo] = {"curlvcbx", "curlvcby", "curlvcbz"}; // output
            const int ndi = 3; string dni[ndi] = {"velx", "vely", "velz"}; // input
            ProcessBlock("Curl", ib, b, ndo, ndi, dno, dni, Dimensions,
                         NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
        }

        /// compute viscous dissipation rate (assuming a viscosity that Re=1 on the scale of one grid-cell length)
        if (compute_dissipation) {
            const int ndo = 1; string dno[ndo] = {"diss_rate"}; // output
            const int ndi = 4; string dni[ndi] = {"velx", "vely", "velz", "dens"}; // input
            ProcessBlock("DissipationRate", ib, b, ndo, ndi, dno, dni, Dimensions,
                         NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
        }

        /// compute enstrophy source terms
        if (compute_enstrophy_sources) {
            {   /// enstrophy advection term
                const int ndo = 1; string dno[ndo] = {"enstrophy_advection"}; // output
                const int ndi = 6; string dni[ndi] = {"velx", "vely", "velz", "vorticity_x", "vorticity_y", "vorticity_z"}; // input
                ProcessBlock("EnstrophyAdvection", ib, b, ndo, ndi, dno, dni, Dimensions,
                             NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
            }
            {   /// vortex stretching term
                const int ndo = 1; string dno[ndo] = {"enstrophy_vortex_stretching"}; // output
                const int ndi = 6; string dni[ndi] = {"velx", "vely", "velz", "vorticity_x", "vorticity_y", "vorticity_z"}; // input
                ProcessBlock("EnstrophyVortexStretching", ib, b, ndo, ndi, dno, dni, Dimensions,
                             NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
            }
            {   /// compressive source term
                const int ndo = 1; string dno[ndo] = {"enstrophy_compressive_term"}; // output
                const int ndi = 6; string dni[ndi] = {"velx", "vely", "velz", "vorticity_x", "vorticity_y", "vorticity_z"}; // input
                ProcessBlock("EnstrophyCompressiveTerm", ib, b, ndo, ndi, dno, dni, Dimensions,
                             NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
            }
            {   /// baroclinic source term
                const int ndo = 1; string dno[ndo] = {"enstrophy_baroclinic_term"}; // output
                const int ndi = 5; string dni[ndi] = {"vorticity_x", "vorticity_y", "vorticity_z", "dens", "pres"}; // input
                ProcessBlock("EnstrophyBaroclinicTerm", ib, b, ndo, ndi, dno, dni, Dimensions,
                             NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names);
            }
        }

        /// compute gravitational acceleration components from potential (GPOT)
        if (compute_grav_accel) {
            const int ndo = 3; string dno[ndo] = {"grax", "gray", "graz"}; // output
            const int ndi = 1; string dni[ndi] = {"gpot"}; // input
            ProcessBlock("Grad", ib, b, ndo, ndi, dno, dni, Dimensions,
                            NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names, -1.0, 0.0); // multiply by -1
        }

    } //end loop over blocks

    MPI_Barrier(MPI_COMM_WORLD);

    /// extend "unknown names" with dataset name
    vector<string> unknown_names = gg.ReadUnknownNames();
    bool datasetname_exists = false;
    for (unsigned int di = 0; di < dsetnames_for_extending_unknown_names.size(); di++) {
        for (unsigned int i = 0; i < unknown_names.size(); i++) {
            if (Verbose > 1) cout << "i, unknown_names[i] = " << i << ", " << unknown_names[i] << endl;
            if (unknown_names[i] == dsetnames_for_extending_unknown_names[di]) {
                datasetname_exists = true;
                if (MyPE==0) cout << "WARNING: '" << dsetnames_for_extending_unknown_names[di] << "' already in 'unknown names'" << endl;
                break;
            }
        } /// end loop over dataset names
        if (!datasetname_exists) {
            unknown_names.push_back(dsetnames_for_extending_unknown_names[di]);
            if (Verbose && MyPE==0) cout << "adding '" << dsetnames_for_extending_unknown_names[di] << "' to file '" << inputfile << "'..." << endl;
        }
    } // end loop over added dataset names
    gg.OverwriteUnknownNames(unknown_names);

    long endtime = time(NULL);
    long duration = endtime-starttime; long duration_red = 0;
    if (Verbose > 1) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (Verbose && MyPE==0) cout << "****************** Global time to finish = "<<duration_red<<"s ******************" << endl;

    MPI_Finalize();
    return 0;

} // end main ==========================================================


/// Helper function to work on a block;
/// Creates the output dataset, calls ComputeTerm, and overwrites this block with the output data.
void ProcessBlock(const string term, const int ib, const int b, const int ndo, const int ndi, string dno[], string dni[],
    const vector<int> Dimensions, const vector<int> NB, const int NGC, const vector<int> NBGC,
    const vector<vector<double> > D, FlashGG gg, vector<string> & dsetnames_for_extending_unknown_names)
{
    // overloaded call with multiplier=1 and added_constant=0
    ProcessBlock(term, ib, b, ndo, ndi, dno, dni, Dimensions, NB, NGC, NBGC, D, gg, dsetnames_for_extending_unknown_names, 1.0, 0.0);
}

void ProcessBlock(const string term, const int ib, const int b, const int ndo, const int ndi, string dno[], string dni[],
                  const vector<int> Dimensions, const vector<int> NB, const int NGC, const vector<int> NBGC,
                  const vector<vector<double> > D, FlashGG gg, vector<string> & dsetnames_for_extending_unknown_names,
                  const double multiplier, const double added_constant)
{
    if (ib==0) { // only do this once to create the dataset(s)
        for (int d = 0; d < ndo; d++) {
            if (Verbose && MyPE==0) cout << "Creating block var '"+dno[d]+"'..." << endl;
            gg.CreateDataset(dno[d], Dimensions);
            dsetnames_for_extending_unknown_names.push_back(dno[d]);
        }
    }
    std::vector<float*> output(ndo); std::vector<float*> input(ndi);
    for (int d = 0; d < ndo; d++) output[d] = new float[NB[X]*NB[Y]*NB[Z]]; // output
    for (int d = 0; d < ndi; d++) input[d] = gg.ReadBlockVarGC(b, dni[d], NGC, mw, pbc); // input
    ComputeTerm(term, output, input, NB, NBGC, D[b], NGC);
    for (int d = 0; d < ndo; d++) {
        // scale by multiplier and offset by added_constant if requested
        if ((multiplier != 1.0) || (added_constant != 0.0)) {
            for (long index=0; index<NB[X]*NB[Y]*NB[Z]; index++)
                output[d][index] = multiplier * output[d][index] + added_constant;
        }
        gg.OverwriteBlockVar(b, dno[d], output[d]); // overwrite (either created above or was already present in file)
        if (Verbose > 1) cout << "["<<MyPE<<"] '"+dno[d]+"' written for block #" << b << endl;
        delete [] output[d]; /// clean
    }
    for (int d = 0; d < ndi; d++) delete [] input[d];
}

/// get indices into active cell and guard cell and (left, right) of active cell
inline void GetIndices(const int i, const int j, const int k,
                       const vector<int> NB, const vector<int> NBGC, const int NGC,
                       long & index, long & index_gc, long il[3], long ir[3])
{
    /// define the left and right indexes for the x,y,z, coordinates
    index = k*NB[X]*NB[Y] + j*NB[X] + i;
    index_gc = (k+NGC)*NBGC[X]*NBGC[Y] + (j+NGC)*NBGC[X] + (i+NGC);
    il[X] = (k+NGC  )*NBGC[X]*NBGC[Y] + (j+NGC  )*NBGC[X] + (i+NGC-1);
    ir[X] = (k+NGC  )*NBGC[X]*NBGC[Y] + (j+NGC  )*NBGC[X] + (i+NGC+1);
    il[Y] = (k+NGC  )*NBGC[X]*NBGC[Y] + (j+NGC-1)*NBGC[X] + (i+NGC  );
    ir[Y] = (k+NGC  )*NBGC[X]*NBGC[Y] + (j+NGC+1)*NBGC[X] + (i+NGC  );
    il[Z] = (k+NGC-1)*NBGC[X]*NBGC[Y] + (j+NGC  )*NBGC[X] + (i+NGC  );
    ir[Z] = (k+NGC+1)*NBGC[X]*NBGC[Y] + (j+NGC  )*NBGC[X] + (i+NGC  );
}

/// generalised function to compute a term based on derivatives of v, depending on what's requested
void ComputeTerm(const string term, std::vector<float*>& output, std::vector<float*>& input,
                 const vector<int> NB, const vector<int> NBGC, const vector<double> D, const int NGC)
{
    /// double cell size
    double DD[] = {0.5/D[X], 0.5/D[Y], 0.5/D[Z]};
    /// indices used below; filled by call to GetIndices
    long index, index_gc, il[3], ir[3];

    /// computes the divergence of an input vector field (e.g., div v or div B)
    if (term == "Div") { // outputs: 0: div(vec); inputs: 0: vecx, 1: vecy, 2: vecz
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            output[0][index] = ((double)input[0][ir[X]]-(double)input[0][il[X]]) * DD[X] +
                               ((double)input[1][ir[Y]]-(double)input[1][il[Y]]) * DD[Y] +
                               ((double)input[2][ir[Z]]-(double)input[2][il[Z]]) * DD[Z];
        } // loop over cells
    }

    /// computes vorticity or current or mag_from_vecpot components (x, y, z)
    if (term == "Curl") {
        // for vorticity: outputs: 0: vort(x), 1: vort(y), 2: vort(z); inputs: 0: velx, 1: vely, 2: velz
        // for current: outputs: 0: current(x), 1: current(y), 2: current(z); inputs: 0: magx, 1: magy, 2: magz
        // for B = curl(A): outputs: 0: mag(x), 1: mag(y), 2: mag(z); inputs: 0: vecpotx, 1: vecpoty, 2: vecpotz
        // for curl(v x B): outputs: 0: curlvxbx(x), 1: curlvxby(y), 2: curlvxbz(z); inputs: 0: mfx, 1: mfy, 2: mfz
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            /// vorticity x - component: [v_z(x,y+dy,z) - v_z(x,y-dy,z)] / 2dy - [v_y(x,y,z+dz) - v_y(x,y,z-dz)] / 2dz
            ///   current x - component: [B_z(x,y+dy,z) - B_z(x,y-dy,z)] / 2dy - [B_y(x,y,z+dz) - B_y(x,y,z-dz)] / 2dz
            /// curl(v x B) x - component: [mf_z(x,y+dy,z) - mf_z(x,y-dy,z)] / 2dy - [mf_y(x,y,z+dz) - mf_y(x,y,z-dz)] / 2dz
            output[0][index] = ((double)input[2][ir[Y]]-(double)input[2][il[Y]]) * DD[Y] -
                               ((double)input[1][ir[Z]]-(double)input[1][il[Z]]) * DD[Z];
            /// vorticity y - component: [v_x(x,y,z+dz) - v_x(x,y,z-dz)] / 2dz - [v_z(x+dx,y,z) - v_z(x+dx,y,z)] / 2dx
            ///   current y - component: [B_x(x,y,z+dz) - B_x(x,y,z-dz)] / 2dz - [B_z(x+dx,y,z) - B_z(x+dx,y,z)] / 2dx
            /// curl(v x B) y - component: [mf_x(x,y,z+dz) - mf_x(x,y,z-dz)] / 2dz - [mf_z(x+dx,y,z) - mf_z(x+dx,y,z)] / 2dx
            output[1][index] = ((double)input[0][ir[Z]]-(double)input[0][il[Z]]) * DD[Z] -
                               ((double)input[2][ir[X]]-(double)input[2][il[X]]) * DD[X];
            /// vorticity z - component: [v_y(x+dx,y,z) - v_y(x-dx,y,z)] / 2dx - [v_x(x,y+dy,z) - v_x(x,y-dy,z)] / 2dy
            ///   current z - component: [B_y(x+dx,y,z) - B_y(x-dx,y,z)] / 2dx - [B_x(x,y+dy,z) - B_x(x,y-dy,z)] / 2dy
            /// curl(v x B) z - component: [mf_y(x+dx,y,z) - mf_y(x-dx,y,z)] / 2dx - [mf_x(x,y+dy,z) - mf_x(x,y-dy,z)] / 2dy
            output[2][index] = ((double)input[1][ir[X]]-(double)input[1][il[X]]) * DD[X] -
                               ((double)input[0][ir[Y]]-(double)input[0][il[Y]]) * DD[Y];
        } // loop over cells
    }

    /// computes the gradient of input and puts it into output(x, y, z)
    if (term == "Grad") {
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            for (int dir = X; dir <= Z; dir++)
                output[dir][index] = ((double)input[0][ir[dir]] - (double)input[0][il[dir]]) * DD[dir];
        } // loop over cells
    }

    if (term == "MagCrossCurrent") {
        // outputs: 0: magcrosscurrentx 1: magcrosscurrenty 2: magcrosscurrentz;
        // inputs: 0: magx, 1: magy, 2: magz, 3: curx, 4: cury, 5: curz;
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            // B_y J_z - B_z J_y
            // v_y B_z - v_z B_y
            output[0][index] =  (double)input[1][index_gc]*(double)input[5][index_gc] -
                                (double)input[2][index_gc]*(double)input[4][index_gc];
            // B_z J_x - B_x J_z
            // v_z B_x - v_x B_z
            output[1][index] =  (double)input[2][index_gc]*(double)input[3][index_gc] -
                                (double)input[0][index_gc]*(double)input[5][index_gc];
            // B_x J_y - B_y J_x
            // v_x B_y - v_y B_x
            output[2][index] =  (double)input[0][index_gc]*(double)input[4][index_gc] -
                                (double)input[1][index_gc]*(double)input[3][index_gc];
        }
    }

    if (term == "MagDotCurrent") {
        // outputs: 0: magdotcurrent;
        // inputs: 0: magx, 1: magy, 2: magz, 3: curx, 4: cury, 5: curz
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            // B_x J_x + B_y J_y + B_z J_z
            output[0][index] =  (double)input[0][index_gc]*(double)input[3][index_gc] +
                                (double)input[1][index_gc]*(double)input[4][index_gc] +
                                (double)input[2][index_gc]*(double)input[5][index_gc];
        }
    }

    if (term == "Tension") {
        // outputs: 0: tenx, outputs: 1: teny, outputs: 2: tenz;
        // inputs: 0: magx, 1: magy, 2: magz
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            // B_x \partial_x B_x + B_y \partial_y B_x + B_z \partial_z B_x
            output[0][index] =  (double)input[0][index_gc] * ((double)input[0][ir[X]]-(double)input[0][il[X]]) * DD[X] +
                                (double)input[1][index_gc] * ((double)input[0][ir[Y]]-(double)input[0][il[Y]]) * DD[Y] +
                                (double)input[2][index_gc] * ((double)input[0][ir[Z]]-(double)input[0][il[Z]]) * DD[Z];
            // B_x \partial_x B_y + B_y \partial_y B_y + B_z \partial_z B_y
            output[1][index] =  (double)input[0][index_gc] * ((double)input[1][ir[X]]-(double)input[1][il[X]]) * DD[X] +
                                (double)input[1][index_gc] * ((double)input[1][ir[Y]]-(double)input[1][il[Y]]) * DD[Y] +
                                (double)input[2][index_gc] * ((double)input[1][ir[Z]]-(double)input[1][il[Z]]) * DD[Z];
            // B_x \partial_x B_z + B_y \partial_y B_z + B_z \partial_z B_z
            output[2][index] =  (double)input[0][index_gc] * ((double)input[2][ir[X]]-(double)input[2][il[X]]) * DD[X] +
                                (double)input[1][index_gc] * ((double)input[2][ir[Y]]-(double)input[2][il[Y]]) * DD[Y] +
                                (double)input[2][index_gc] * ((double)input[2][ir[Z]]-(double)input[2][il[Z]]) * DD[Z];
        }
    }

    if (term == "DissipationRate") { // outputs: 0: viscous dissipation rate; inputs: 0: velx, 1: vely, 2: velz, 3: dens
        /*
            This function computes the viscous dissipation rate based on Offner et al. (2009).
            The model is:
                \dot{e}_{diss} = - \mu (\sigma \cdot \nabla) \cdot v with units of g/cm/s^3,
            where \sigma is the viscous stress tensor,
                \sigma = \nu ( S - 2/3 I \nabla \cdot v),
            and v is the velocity. S is the strain-rate tensor,
                S = \nabla v - ( \nabla v )^T.
            The dynamic viscosity is given by
                \mu = \rho |v| \delta_x / Re_g,
            where delta_x is the grid differential and Re_g ~ 1 is the Reynolds number at
            the grid scale. The largest uncertainty is in the Re_g.
            To compute \mu we assume that on the grid scale the Reynolds number is approximately 1, which is the only
            assumption for this model. Clearly this is not universally true, but regardless, this only scales the
            dissipation rate by a proportionality factor, hence the overall dissipation structure is correct.
        */

        double dx = D[X]; // the differential in x for the \nu calculation
        double Re_g = 1.0; // use Re_g ~ 1 for the \nu calculation

        // viscous stress tensor components (rank 2 tensor)
        double sigma_xx, sigma_xy, sigma_xz;
        double sigma_yx, sigma_yy, sigma_yz;
        double sigma_zx, sigma_zy, sigma_zz;

        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices

            // first construct mu
            double vel_mag = sqrt( (double)input[0][index_gc]*(double)input[0][index_gc] +
                                   (double)input[1][index_gc]*(double)input[1][index_gc] +
                                   (double)input[2][index_gc]*(double)input[2][index_gc] );
            double mu = (double)input[3][index_gc] * vel_mag * dx / Re_g;

            // x components
            // sigma_xx = - mu 2/3 div(v)
            sigma_xx = -2.0/3.0 * ( ((double)input[0][ir[X]]-(double)input[0][il[X]]) * DD[X] +
                                    ((double)input[1][ir[Y]]-(double)input[1][il[Y]]) * DD[Y] +
                                    ((double)input[2][ir[Z]]-(double)input[2][il[Z]]) * DD[Z]);
            // sigma_xy = mu(\partial_y v_x - \partial_x v_y)
            sigma_xy =  ((double)input[0][ir[Y]]-(double)input[0][il[Y]]) * DD[Y] -
                        ((double)input[1][ir[X]]-(double)input[1][il[X]]) * DD[X];
            // sigma_xz = mu(\partial_z v_x - \partial_x v_z)
            sigma_xz =  ((double)input[0][ir[Z]]-(double)input[0][il[Z]]) * DD[Z] -
                        ((double)input[2][ir[X]]-(double)input[2][il[X]]) * DD[X];

            // y components
            // sigma_yx = mu(\partial_x v_y - \partial_y v_x)
            sigma_yx =  ((double)input[1][ir[X]]-(double)input[1][il[X]]) * DD[X] -
                        ((double)input[0][ir[Y]]-(double)input[0][il[Y]]) * DD[Y];
            // sigma_yy = - mu 2/3 div(v)
            sigma_yy = sigma_xx;
            // sigma_yz = mu(\partial_z v_y - \partial_y v_z)
            sigma_yz =  ((double)input[1][ir[Z]]-(double)input[1][il[Z]]) * DD[Z] -
                        ((double)input[2][ir[Y]]-(double)input[2][il[Y]]) * DD[Y];

            // z components
            // sigma_zx = mu(\partial_x v_z - \partial_z v_x)
            sigma_zx =  ((double)input[2][ir[X]]-(double)input[2][il[X]]) * DD[X] -
                        ((double)input[0][ir[Z]]-(double)input[0][il[Z]]) * DD[Z];
            // sigma_zy = mu(\partial_y v_z - \partial_z v_y)
            sigma_zy =  ((double)input[2][ir[Y]]-(double)input[2][il[Y]]) * DD[Y] -
                        ((double)input[1][ir[Z]]-(double)input[1][il[Z]]) * DD[Z];
            // sigma_zz = - mu 2/3 div(v)
            sigma_zz = sigma_xx;

            /*
                \dot{e}_viscous = sigma_ij partial_j v_i =
                    -(  [sigma_xx partial_x v_x + sigma_yx partial_x v_y + sigma_zx partial_x v_z] +
                        [sigma_xy partial_y v_x + sigma_yy partial_y v_y + sigma_zy partial_y v_z] +
                        [sigma_xz partial_z v_x + sigma_yz partial_z v_y + sigma_zz partial_z v_z]  )
            */
            output[0][index] = -mu *( ( sigma_xx * ( ((double)input[0][ir[X]]-(double)input[0][il[X]]) * DD[X] ) +
                                        sigma_yx * ( ((double)input[1][ir[X]]-(double)input[1][il[X]]) * DD[X] ) +
                                        sigma_zx * ( ((double)input[2][ir[X]]-(double)input[2][il[X]]) * DD[X] ) ) +
                                      ( sigma_xy * ( ((double)input[0][ir[Y]]-(double)input[0][il[Y]]) * DD[Y] ) +
                                        sigma_yy * ( ((double)input[1][ir[Y]]-(double)input[1][il[Y]]) * DD[Y] ) +
                                        sigma_zy * ( ((double)input[2][ir[Y]]-(double)input[2][il[Y]]) * DD[Y] ) ) +
                                      ( sigma_xz * ( ((double)input[0][ir[Z]]-(double)input[0][il[Z]]) * DD[Z] ) +
                                        sigma_yz * ( ((double)input[1][ir[Z]]-(double)input[1][il[Z]]) * DD[Z] ) +
                                        sigma_zz * ( ((double)input[2][ir[Z]]-(double)input[2][il[Z]]) * DD[Z] ) ) );

        } // loop over cells
    }

    /// computes enstrophy source 'advection' (see Wittor and Gaspari, 2020; arxiv:2009.03344)
    if (term == "EnstrophyAdvection") { // outputs: 0: advection term; inputs: 0: velx, 1: vely, 2: velz, 3: vorticity_x, 4: vorticity_y, 5: vorticity_z
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            // eps = vort^2/2
            double eps_l[3], eps_r[3]; // left, right
            for (int dir = X; dir <= Z; dir++) {
                eps_l[dir] = 0.5 * ( (double)input[3][il[dir]]*(double)input[3][il[dir]] +
                                     (double)input[4][il[dir]]*(double)input[4][il[dir]] +
                                     (double)input[5][il[dir]]*(double)input[5][il[dir]]  );
                eps_r[dir] = 0.5 * ( (double)input[3][ir[dir]]*(double)input[3][ir[dir]] +
                                     (double)input[4][ir[dir]]*(double)input[4][ir[dir]] +
                                     (double)input[5][ir[dir]]*(double)input[5][ir[dir]]  );
            }
            // advection = -div(vel*vort^2/2)
            output[0][index] = -( ((double)input[0][ir[X]]*eps_r[X]-(double)input[0][il[X]]*eps_l[X]) * DD[X] +
                                  ((double)input[1][ir[Y]]*eps_r[Y]-(double)input[1][il[Y]]*eps_l[Y]) * DD[Y] +
                                  ((double)input[2][ir[Z]]*eps_r[Z]-(double)input[2][il[Z]]*eps_l[Z]) * DD[Z]  );

        } // loop over cells
    }

    /// computes enstrophy source 'vortex stretching' (see Wittor and Gaspari, 2020; arxiv:2009.03344)
    if (term == "EnstrophyVortexStretching") { // outputs: 0: vortex term; inputs: 0: velx, 1: vely, 2: velz, 3: vorticity_x, 4: vorticity_y, 5: vorticity_z
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            // vort_stretch = (vort.nabla) v.vort
            output[0][index] = ((double)input[3][index_gc]*(double)input[4][index_gc]*
                              (((double)input[0][ir[Y]]-(double)input[0][il[Y]]) * DD[Y]  +
                               ((double)input[1][ir[X]]-(double)input[1][il[X]]) * DD[X]))+
                               ((double)input[4][index_gc]*(double)input[5][index_gc]*
                              (((double)input[1][ir[Z]]-(double)input[1][il[Z]]) * DD[Z]  +
                               ((double)input[2][ir[Y]]-(double)input[2][il[Y]]) * DD[Y]))+
                               ((double)input[5][index_gc]*(double)input[3][index_gc]*
                              (((double)input[2][ir[X]]-(double)input[2][il[X]]) * DD[X]  +
                               ((double)input[0][ir[Z]]-(double)input[0][il[Z]]) * DD[Z]))+
                               ((double)input[3][index_gc]*(double)input[3][index_gc]*
                              (((double)input[0][ir[X]]-(double)input[0][il[X]]) * DD[X]))+
                               ((double)input[4][index_gc]*(double)input[4][index_gc]*
                              (((double)input[1][ir[Y]]-(double)input[1][il[Y]]) * DD[Y]))+
                               ((double)input[5][index_gc]*(double)input[5][index_gc]*
                              (((double)input[2][ir[Z]]-(double)input[2][il[Z]]) * DD[Z]));
        } // loop over cells
    }

    /// computes enstrophy source 'compressive (div v)' (see Wittor and Gaspari, 2020; arxiv:2009.03344)
    if (term == "EnstrophyCompressiveTerm") { // outputs: 0: compressive term; inputs: 0: velx, 1: vely, 2: velz, 3: vorticity_x, 4: vorticity_y, 5: vorticity_z
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            // comp_vort = -(vort^2/2)*div(v)
            output[0][index] = -0.5*((double)input[3][index_gc]*(double)input[3][index_gc] +
                                     (double)input[4][index_gc]*(double)input[4][index_gc] +
                                     (double)input[5][index_gc]*(double)input[5][index_gc])*
                                   (((double)input[0][ir[X]]-(double)input[0][il[X]]) * DD[X] +
                                    ((double)input[1][ir[Y]]-(double)input[1][il[Y]]) * DD[Y] +
                                    ((double)input[2][ir[Z]]-(double)input[2][il[Z]]) * DD[Z]);
        } // loop over cells
    }

    /// computes enstrophy source 'baroclinic' (see Wittor and Gaspari, 2020; arxiv:2009.03344)
    if (term == "EnstrophyBaroclinicTerm") { // outputs: 0: baroclinic term; inputs: 0: vorticity_x, 1: vorticity_y, 2: vorticity_z, 3: dens, 4: pres
        for (int k=0; k<NB[Z]; k++) for (int j=0; j<NB[Y]; j++) for (int i=0; i<NB[X]; i++) {
            GetIndices(i, j, k, NB, NBGC, NGC, index, index_gc, il, ir); /// get indices
            // baroclinic = 1/dens^2*vort.(grad(dens) X grad(pres))
            double grad_dens[3], grad_pres[3];
            for (int dir = X; dir <= Z; dir++) {
                grad_dens[dir] = ((double)input[3][ir[dir]] - (double)input[3][il[dir]]) * DD[dir];
                grad_pres[dir] = ((double)input[4][ir[dir]] - (double)input[4][il[dir]]) * DD[dir];
            }
            output[0][index] = 1.0/(input[3][index_gc]*input[3][index_gc])*(
                                (double)input[0][index_gc]*(grad_dens[Y]*grad_pres[Z] - grad_dens[Z]*grad_pres[Y])+
                                (double)input[1][index_gc]*(grad_dens[Z]*grad_pres[X] - grad_dens[X]*grad_pres[Z])+
                                (double)input[2][index_gc]*(grad_dens[X]*grad_pres[Y] - grad_dens[Y]*grad_pres[X]));
        } // loop over cells
    }

} // end: ComputeTerm


/** ------------------------- ParseInputs ----------------------------
**  Parses the command line Arguments
** ------------------------------------------------------------------ */
int ParseInputs(const vector<string> Argument)
{
    // check for valid options
    vector<string> valid_options;
    valid_options.push_back("-divv");
    valid_options.push_back("-divb");
    valid_options.push_back("-vort");
    valid_options.push_back("-divrhov");
    valid_options.push_back("-current");
    valid_options.push_back("-MHD_scales");
    valid_options.push_back("-induction");
    valid_options.push_back("-dissipation");
    valid_options.push_back("-enstrophy_sources");
    valid_options.push_back("-grav_accel");
    valid_options.push_back("-non-periodic");
    valid_options.push_back("-no-mw");
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

    /// read tool specific options
    if (Argument.size() < 2)
    {
        if (MyPE==0) { cout << endl << "ParseInputs: Invalid number of arguments." << endl; }
        return -1;
    }
    inputfile = Argument[1];

    for (unsigned int i = 2; i < Argument.size(); i++)
    {
        if (Argument[i] != "" && Argument[i] == "-divv") compute_divv = true;
        if (Argument[i] != "" && Argument[i] == "-divb") compute_divb = true;
        if (Argument[i] != "" && Argument[i] == "-vort") compute_vort = true;
        if (Argument[i] != "" && Argument[i] == "-divrhov") compute_divrhov = true;
        if (Argument[i] != "" && Argument[i] == "-current") compute_current = true;
        if (Argument[i] != "" && Argument[i] == "-mag_from_vecpot") compute_mag_from_vecpot = true;
        if (Argument[i] != "" && Argument[i] == "-MHD_scales") compute_MHD_scales = true;
        if (Argument[i] != "" && Argument[i] == "-induction") compute_induction = true;
        if (Argument[i] != "" && Argument[i] == "-dissipation") compute_dissipation = true;
        if (Argument[i] != "" && Argument[i] == "-enstrophy_sources") compute_enstrophy_sources = true;
        if (Argument[i] != "" && Argument[i] == "-grav_accel") compute_grav_accel = true;
        if (Argument[i] != "" && Argument[i] == "-non-periodic") pbc = false;
        if (Argument[i] != "" && Argument[i] == "-no-mw") mw = false;
        if (Argument[i] != "" && Argument[i] == "-verbose")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> Verbose; dummystream.clear();
            } else return -1;
        }

    } // loop over all args

    /// print out parsed values
    if (Verbose && MyPE==0) {
        cout << " ParseInputs: Command line arguments: ";
        for (unsigned int i = 0; i < Argument.size(); i++) cout << Argument[i] << " ";
        cout << endl;
    }

    if (!compute_divv && !compute_divb && !compute_vort && !compute_divrhov && !compute_current && !compute_mag_from_vecpot && !compute_MHD_scales && !compute_induction && !compute_dissipation && !compute_enstrophy_sources && !compute_grav_accel) {
        if (MyPE==0) cout << endl << "Need to specify at least one of '-divv', '-divb', '-vort', '-divrhov', '-current', '-mag_from_vecpot', '-MHD_scales', '-induction', '-dissipation', '-enstrophy_sources, '-compute_grav_accel'. Exiting." << endl;
        return -1;
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
            << " derivative_var <filename> [<OPTIONS>]" << endl << endl
            << "   <OPTIONS>:           " << endl
            << "     -divv              : compute divergence of velocity" << endl
            << "     -divb              : compute divergence of magnetic field" << endl
            << "     -vort              : compute vorticity components" << endl
            << "     -divrhov           : compute divergence of momentum density (rho*v)" << endl
            << "     -current           : compute current components (J = curl of B)" << endl
            << "     -mag_from_vecpot   : compute magnetic field components from vector potential (B = curl of A)" << endl
            << "     -MHD_scales        : compute BxJ, B.J and the magnetic tension" << endl
            << "     -induction         : compute curl(v x B), the induction term" << endl
            << "     -dissipation       : compute the viscous dissipation rate (assumes Re=1 on dx)" << endl
            << "     -enstrophy_sources : compute enstrophy source terms" << endl
            << "     -grav_accel        : compute gravitational acceleration from potential (GPOT)" << endl
            << "     -non-periodic      : do not use periodic boundary conditions (default is to assume periodic BCs)" << endl
            << "     -no-mw             : do not use mass weighting and instead use volume weighting (only relevant for AMR)" << endl
            << "     -verbose <level>   : verbose level (0, 1, 2) (default: 1)" << endl
            << endl
            << "Example: derivative_var DF_hdf5_plt_cnt_0020 -vort -divv" << endl
            << endl << endl;
    }
}
