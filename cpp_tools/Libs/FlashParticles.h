#ifndef FLASHPARTICLES_H
#define FLASHPARTICLES_H

#include "HDFIO.h"

#include <hdf5.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <limits> /// numeric limits

// define float or double output mode of FLASH Particles
#ifndef FLASH_PARTICLES_REAL
#define FLASH_PARTICLES_REAL double
#define FLASH_PARTICLES_H5_REAL H5T_NATIVE_DOUBLE
#endif

/**
 * FlashParticles class
 *
 * @author Christoph Federrath (christoph.federrath@anu.edu.au)
 * @version 2010-2025
 *
 */

class FlashParticles
{
    private:
        std::string ClassSignature;
        std::string Filename;
        long numParticles;
        std::vector<std::string> PropertyNames;
        std::map<std::string, int> PartType;
        std::map<std::string, int> PartTypeCount;
        int Verbose;
        // HDF input/output and MPI mpi_communicator
        HDFIO hdfio;// MPI_Comm mpi_comm; int MyPE;

    /// Constructors
    public: FlashParticles(void)
    {
        // empty constructor, so we can define a global class object in application code
        Verbose = 0; // avoids message from destructor
    }
    public: FlashParticles(const std::string input_filename)
    {
        this->Constructor(input_filename, 1);
    }
    public: FlashParticles(const std::string input_filename, const int verbose)
    {
        this->Constructor(input_filename, verbose);
    }
        /// Destructor
    public: ~FlashParticles()
    {
      if (Verbose > 1) std::cout<<__func__<<": destructor called."<<std::endl;
    }

    // actual constructor
    private: void Constructor(const std::string input_filename, const int verbose)
    {
        ClassSignature = "FlashParticles: ";
        Filename = "";
        numParticles = 0;
        PropertyNames.clear();
        PartType.clear();
        PartTypeCount.clear();
        Verbose = verbose;

        this->SetParticleFilename(input_filename);
        if (Filename != "") {
            if (Verbose > 1) std::cout << ClassSignature << "Using '"<<Filename<<"' as particle file." << std::endl;
            // open file
            hdfio = HDFIO();
            hdfio.open(Filename, 'r'); // currently only reading is supported here (no need for MPI then)
            this->ReadMetaData();
            if (numParticles > 0) {
                this->SetParticleTypes();
            }
        }
    }

    // get function signature for printing
    private: std::string FuncSig(const std::string funcname)
    {
        return ClassSignature+funcname+": ";
    }

    // clear whitespace and/or terminators at the end of a string
    private: std::string Trim(const std::string input)
    {
        std::string ret = input;
        return ret.erase(input.find_last_not_of(" \n\r\t")+1);
    }

    /// Determine the FLASH particle filename; for example, if a plt file is supplied
    private: void SetParticleFilename(std::string input_filename)
    {
        Filename = ""; // clear Filename
        // if input_filename is a chk or a part file, simply use it
        if (input_filename.find("_hdf5_chk_") != std::string::npos) Filename = input_filename;
        if (input_filename.find("_hdf5_part_") != std::string::npos) Filename = input_filename;
        // if input_filename is a plt file, determine the respective particle file
        if (input_filename.find("_hdf5_plt_cnt_") != std::string::npos) {
            // see if part file exists with same dump number, which is assumed to be the plt<->part matching pair
            std::string partfile = input_filename; partfile.replace(partfile.length()-12,7,"part");
            std::ifstream ifs_file(partfile.c_str());
            if (ifs_file.good()) Filename = partfile;
            else if (Verbose > 1) std::cout<<FuncSig(__func__)<<"WARNING. Could not find matching particle file."<<std::endl;
        } // is plt file
    }

    /**
      * ReadMetaData
      * read number of particles and particle property names from particle file
      */
    private: void ReadMetaData()
    {
        /// read number of particles
        std::vector<int> dims = hdfio.getDims("tracer particles");
        if (Verbose > 1) std::cout<<FuncSig(__func__)<<"dims = "<<dims[0]<<" "<<dims[1]<<std::endl;
        numParticles = dims[0];
        if (Verbose > 1) std::cout<<FuncSig(__func__)<<Filename<<" contains total of "<<numParticles<<" particles."<<std::endl;
        /// read particle property names
        const unsigned int string_size = 24;
        hid_t string_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(string_type, string_size);
        int nnames = hdfio.getDims("particle names")[0];
        if (Verbose > 1) std::cout<<FuncSig(__func__)<<"nnames = "<<nnames<< std::endl;
        char * cdata = new char[nnames*string_size];
        hdfio.read(cdata, "particle names", string_type);
        PropertyNames.resize(nnames);
        for (unsigned int i = 0; i < PropertyNames.size(); i++) {
            std::string tmp_str; tmp_str.resize(0);
            for (unsigned int j = 0; j < string_size; j++) tmp_str.push_back(cdata[i*string_size+j]);
            PropertyNames[i] = tmp_str;
            PropertyNames[i].erase(PropertyNames[i].find_last_not_of(" \n\r\t")+1); // clear whitespace
            if (Verbose > 1) std::cout<<FuncSig(__func__)<<"PropertyNames = '"<<PropertyNames[i]<<"'"<< std::endl;
        }
        delete [] cdata;
    }

    /**
      * GetParticleTypes
      * determine particle types
      */
    private: void SetParticleTypes()
    {
        /// see if 'type' is in particle names
        bool have_types = false;
        if (std::count(PropertyNames.begin(), PropertyNames.end(), "type") > 0) have_types = true;
        if (Verbose > 1) std::cout<<FuncSig(__func__)<<"have_types = "<<have_types<<std::endl;

        // if 'accr_rate' in particle names, we have sink particles
        bool have_sinks = false;
        if (std::count(PropertyNames.begin(), PropertyNames.end(), "accr_rate") > 0) have_sinks = true;
        if (Verbose > 1) std::cout<<FuncSig(__func__)<<"have_sinks = "<<have_sinks<<std::endl;

        /// if there are different types in file, count how many and which
        std::vector<int> types;
        std::vector<int> types_count;
        if (have_types) {
            // read in chunks of 16 million particles = 128 MB for a single (8-byte) variable (here "type")
            long chunk_size = 16000000;
            long start_index = -chunk_size; // init for incrementing in while loop below
            bool finished = false;
            while (!finished) {
                // set start_index and size for reading the next chunk of particle data
                start_index += chunk_size;
                if (start_index+chunk_size >= numParticles) {
                    chunk_size = numParticles - start_index; // last chunk
                    finished = true; // end while condition
                }
                FLASH_PARTICLES_REAL * type = this->ReadVar("type", start_index, chunk_size);
                for (long i=0; i<chunk_size; i++) {
                    unsigned int index = std::find(types.begin(), types.end(), type[i]) - types.begin();
                    if (index == types.size()) { // not in types yet -> new type
                        types.push_back(type[i]);
                        types_count.push_back(1);
                    } else { // type existed, so increase particle count for this type
                        types_count[index]++;
                    }
                }
                delete [] type;
            }
            if (Verbose > 1) {
                std::cout<<FuncSig(__func__)<<"Number of particles by type:";
                for (unsigned int t=0; t<types.size(); t++) std::cout<<"  np="<<types_count[t]<<" (type:"<<types[t]<<")";
                std::cout<<std::endl;
            }
        } else {
            types.push_back(1); // only single type
            types_count.push_back(numParticles);
        }

        /// assign particle types to PartType map
        if (have_sinks) {
            if (have_types) {
                PartType["tracer"] = 1;
                PartType["sink"] = 2;
            } else {
                PartType["sink"] = 1;
            }
        } else {
            PartType["tracer"] = 1;
        }

        /// fill PartTypeCount
        if (Verbose > 1) std::cout<<FuncSig(__func__)<<"Number of particles by type:";
        for (std::map<std::string, int>::iterator it = PartType.begin(); it != PartType.end(); it++) {
            for (unsigned int t=0; t<types.size(); t++)
                if (PartType[it->first] == types[t]) PartTypeCount[it->first] = types_count[t];
            if (Verbose > 1) std::cout<<"  np="<<PartTypeCount[it->first]<<" (type:"<<it->first<<",id:"<<PartType[it->first]<<")";
        }
        if (Verbose > 1) std::cout<<std::endl;
    }

    public: void PrintInfo(void)
    {
        if (Filename != "") {
            std::cout<<FuncSig(__func__)<<"Using '"<<Filename<<"' as particle file."<<std::endl;
            std::cout<<FuncSig(__func__)<<"Total number of particles: "<<numParticles<<std::endl;
            std::cout<<FuncSig(__func__)<<"Number of particles by type:";
            for (std::map<std::string, int>::iterator it = PartType.begin(); it != PartType.end(); it++) {
                std::cout<<"  np="<<PartTypeCount[it->first]<<" (type:"<<it->first<<",id:"<<PartType[it->first]<<")";
            }
            std::cout<<std::endl;
        }
    }

    /// ReadVar (overloaded to read all particle types and full length of particle array)
    public: FLASH_PARTICLES_REAL * ReadVar(const std::string varname)
    {
        return ReadVarAll(varname, 0, numParticles);
    }

    /// ReadVar (overloaded to read all particle types, starting from start_index with length 'size')
    public: FLASH_PARTICLES_REAL * ReadVar(const std::string varname,
                                           const long start_index, const long size)
    {
        return ReadVarAll(varname, start_index, size);
    }

    /// ReadVar (overloaded to read requested particle type and full length of particle array)
    public: FLASH_PARTICLES_REAL * ReadVar(const std::string varname, const std::string type_req)
    {
        long size = numParticles;
        return ReadVar(varname, type_req, 0, size);
    }

    /// ReadVar (read particle variable from file, given a type request, starting from start_index with length 'size')
    public: FLASH_PARTICLES_REAL * ReadVar(const std::string varname, const std::string type_req,
                                           const long start_index, long &size)
    {
        /// check that requested type is valid
        if (PartType.find(type_req) == PartType.end()) { // request type does not exist
            std::cout<<FuncSig(__func__)<<"ERROR: requested type '"<<type_req<<"' not in file."<<std::endl;
            return NULL;
        }
        // if there is only this single particle type, we return it right away
        if (PartType.size() == 1) return ReadVarAll(varname, start_index, size);
        /// read full varname and type; and fill up return array
        FLASH_PARTICLES_REAL * var = ReadVarAll(varname, start_index, size);
        if (var==NULL) return NULL; // return if empty
        FLASH_PARTICLES_REAL * type = ReadVarAll("type", start_index, size);
        FLASH_PARTICLES_REAL * tmp = new FLASH_PARTICLES_REAL[size];
        long index = 0; // loop to match requested type for filling up return array
        for (long i=0; i<size; i++) if (type[i]==PartType[type_req]) tmp[index++] = var[i];
        size = index; // change the size for return purposes (so the caller knows how many actually matched in type)
        FLASH_PARTICLES_REAL * ret = new FLASH_PARTICLES_REAL[size];
        for (long i=0; i<size; i++) ret[i] = tmp[i]; // copy only up to index=size, i.e., only the type-matched data
        delete [] var; delete [] type; delete [] tmp;
        return ret;
    }

    /// ReadVarAll (overloaded to read full length of particle array; all particle types included)
    public: FLASH_PARTICLES_REAL * ReadVarAll(const std::string varname)
    {
        return ReadVarAll(varname, 0, numParticles);
    }

    /// ReadVarAll (read particle variable from file; all particle types included, starting from start_index with length 'size')
    public: FLASH_PARTICLES_REAL * ReadVarAll(const std::string varname,
                                              const long start_index, const long size)
    {
        // find the particle name index
        unsigned int p_name_index = std::find(PropertyNames.begin(), PropertyNames.end(), varname) - PropertyNames.begin();
        if (p_name_index == PropertyNames.size()) {
            std::cout<<FuncSig(__func__)<<"WARNING: particle property name '"<<varname<<"' not found; skipping."<<std::endl;
            return NULL;
        }
        assert (p_name_index < PropertyNames.size());
        // read into FLASH_PARTICLES_REAL container
        FLASH_PARTICLES_REAL * DataPointer = new FLASH_PARTICLES_REAL[size];
        hsize_t out_offset[1] = {0};
        hsize_t out_count[1] = {(hsize_t)size};
        hsize_t offset[2] = {(hsize_t)start_index, (hsize_t)p_name_index};
        hsize_t count[2] = {(hsize_t)size, 1};
        hdfio.read_slab(DataPointer, "tracer particles", FLASH_PARTICLES_H5_REAL, offset, count, 1, out_offset, out_count);
        return DataPointer;
    }

    /// ReadSinkData into map of vectors (input: string prop_name)
    public: std::map<std::string, std::vector<FLASH_PARTICLES_REAL> > ReadSinkData(const std::string prop_name) {
        std::vector<std::string> prop_names; prop_names.push_back(prop_name);
        return ReadSinkData(prop_names);
    }

    /// ReadSinkData into map of vectors (input: vector<string> prop_names)
    public: std::map<std::string, std::vector<FLASH_PARTICLES_REAL> > ReadSinkData(const std::vector<std::string> prop_names) {
        std::map<std::string, std::vector<FLASH_PARTICLES_REAL> > ret;
        for (unsigned int pn=0; pn<prop_names.size(); pn++) {
            FLASH_PARTICLES_REAL * tmp = ReadVar(prop_names[pn], "sink"); if (tmp==NULL) continue; // skip if empty
            for (long p=0; p<PartTypeCount["sink"]; p++) ret[prop_names[pn]].push_back(tmp[p]);
            delete [] tmp;
        }
        return ret;
    }

    // Domain decomposition.
    // Inputs: MPI rank (MyPE), total number of MPI ranks (NPE), total number of particles to distribute (np).
    // Return indices of particles for MyPE.
    public: std::vector<int> GetMyParticles(const int MyPE, const int NPE, const long np)
    {
        std::vector<int> MyParticles(0);
        unsigned int Div = ceil( (double)(np) / (double)(NPE) );
        int NPE_main = np / Div;
        unsigned int Mod = np - NPE_main * Div;
        if (MyPE < NPE_main) // (NPE_main) cores get Div particles
            for (unsigned int i = 0; i < Div; i++) MyParticles.push_back(MyPE*Div+i);
        if (MyPE==0 && Verbose > 0) std::cout<<FuncSig(__func__)<<"First "<<NPE_main<<" core(s) carry(ies) "<<Div<<" particle(s) (each)."<<std::endl;
        if ((MyPE == NPE_main) && (Mod > 0)) { // core (NPE_main + 1) gets the rest (Mod)
            for (unsigned int i = 0; i < Mod; i++)
                MyParticles.push_back(NPE_main*Div+i);
            if (Verbose > 0) std::cout<<FuncSig(__func__)<<"Core #"<<NPE_main+1<<" carries "<<Mod<<" particle(s)."<<std::endl;
        }
        int NPE_in_use = NPE_main; if (Mod > 0) NPE_in_use += 1;
        if ((MyPE == 0 && Verbose > 0) && (NPE_in_use < NPE))
            std::cout<<FuncSig(__func__)<<"Warning: non-optimal load balancing; "<<NPE-NPE_in_use<<" core(s) remain(s) idle."<<std::endl;
        return MyParticles;
    }

    /// Return particle filename
    public: std::string GetFilename()
    { return Filename; }
    /// Return number of particles
    public: long GetN()
    { return numParticles; }
    /// Return particle names
    public: std::vector<std::string> GetPropertyNames()
    { return PropertyNames; }
    public: std::map<std::string, int> GetType()
    { return PartType; }
    public: std::map<std::string, int> GetTypeCount()
    { return PartTypeCount; }

}; // end: FlashParticles
#endif
