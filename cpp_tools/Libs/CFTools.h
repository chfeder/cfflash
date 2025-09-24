#ifndef CFTOOLS_H
#define CFTOOLS_H

#include <mpi.h>
#include <map>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cmath>

// define float or double mode
#ifndef CFTOOLS_REAL
#define CFTOOLS_REAL double
#endif

/**
 * CFTools class
 * has tools
 *
 * @author Christoph Federrath (christoph.federrath@anu.edu.au)
 * @version 2025
 *
 */

class CFTools
{
    private:
    enum {X, Y, Z};
    std::string ClassSignature;
    std::string CFTools_env;
    std::map<std::string, long> Timers_start; // in millisec
    std::map<std::string, double> Timers_duration; // in sec
    std::chrono::time_point<std::chrono::steady_clock> ProgressBar_startTime;
    int Verbose;

    /**
      * Default constructor.
      */
    public: CFTools(void)
    {
        Constructor(1); // Verbose = 1
    };

    public: CFTools(const int verbose)
    {
        Constructor(verbose);
    };

    private: void Constructor(const int verbose)
    {
        ClassSignature = "CFTools: "; // class signature, when this class is printing to stdout
        Verbose = verbose;
        const char* env_var = std::getenv("CFTOOLS"); // read the CFTOOLS environment variable for external control
        CFTools_env = (env_var != NULL) ? env_var : ""; // assign empty string if env variable is missing
        ProgressBar_startTime = std::chrono::steady_clock::now();
        if (Verbose > 1) { std::cout<<FuncSig(__func__)<<"CFTools object created."<<std::endl; }
    };

    // get function signature for printing to stdout
    private: std::string FuncSig(const std::string func_name)
    { return ClassSignature+func_name+": "; };

    // starts a named timer to time code segments (create the timer if not already exists)
    public: void TimerStart(std::string name)
    {
        if (Timers_duration.count(name) == 0) Timers_duration[name] = 0; // create timer
        // get current time
        long time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        Timers_start[name] = time;
    };
    // stops the timer and adds to the time duration
    public: void TimerStop(std::string name)
    {
        long time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        Timers_duration[name] += 1e-3 * static_cast<double>(time - Timers_start[name]); // fill duration from now since timer start
    };
    // report all timers
    public: void TimerReport(void)
    {
        for (std::map<std::string, double>::iterator it = Timers_duration.begin(); it != Timers_duration.end(); ++it) {
            TimerReport(it->first);
        }
    };
    // overloaded: report specific named timer
    public: void TimerReport(std::string name)
    {
        // print timer info to screen
        std::cout << "Time (s) for '" << name << "': " << Timers_duration[name] << std::endl;
    };

    // for progress bar
    void InitProgressBar(void) {
        ProgressBar_startTime = std::chrono::steady_clock::now();
    };
    void PrintProgressBar(double loop_index, int total) {
        this->PrintProgressBar(loop_index, total, 100, 60); // overloaded call with maxUpdates=100, bar_width=60
    };
    void PrintProgressBar(double loop_index, int total, int maxUpdates) {
        this->PrintProgressBar(loop_index, total, maxUpdates, 60); // overloaded call with bar_width=60
    };
    void PrintProgressBar(double loop_index, int total, int maxUpdates, int bar_width) {
        static int lastPrinted = -1; // track last printed step
        // calculate current update step (out of maxUpdates)
        int step = static_cast<int>(round((loop_index * maxUpdates) / static_cast<double>(total-1)));
        // limit progress bar update based on current step and also print in the very last step
        if ((step != lastPrinted) || (loop_index == total-1)) {
            lastPrinted = step; // save current step in lastPrinted
            // get elapsed time since init
            std::chrono::time_point<std::chrono::steady_clock> current_time = std::chrono::steady_clock::now();
            double time_elapsed = std::chrono::duration<double>(current_time - ProgressBar_startTime).count();
            int pos = bar_width;
            std::string eta_str = "";
            double percent_done = 100;
            if (total > 1) {
                pos = (loop_index * bar_width) / (total-1); // bar position
                // estimate ETA
                double ETA = (loop_index > 0) ? (time_elapsed/loop_index) * (total-1-loop_index) : -1;
                std::stringstream eta_oss; eta_oss << std::fixed << std::setprecision(1) << ETA;
                if (ETA > 0) eta_str = "| ETA: " + eta_oss.str() + "s";
                percent_done = loop_index * 100 / (total-1);
            }
            // check whether we should write a new line for each progress bar update (e.g., in file redirect mode)
            bool redirect_mode = false;
            if ((CFTools_env.find("redir") != std::string::npos) || (CFTools_env.find("job") != std::string::npos)) redirect_mode = true;
            if (!redirect_mode) std::cout << "\r"; // only do carriage return if not in redirect mode
            std::cout << "[";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos)
                    std::cout << "=";
                else if (i == pos)
                    std::cout << ">";
                else
                    std::cout << " ";
            }
            std::cout << "] " << std::fixed << std::setprecision(1) << percent_done << "% | "
                        << "Elapsed: " << std::setprecision(1) << time_elapsed << "s " << eta_str << "   ";
            // end line in redirect mode or when at 100%, end the line
            if (redirect_mode || pos == bar_width) std::cout << "        " << std::endl;
            std::cout.flush();
        }
    };

    // function for Gaussian beam smoothing (input: standard deviation, number of boxes)
    private: std::vector<int> boxesForGauss(double sigma, int n)
    {
        double wIdeal = sqrt((12*sigma*sigma/n)+1);  // ideal averaging filter width
        int wl = floor(wIdeal);
        if (wl % 2 == 0) wl--;
        int wu = wl + 2;
        double mIdeal = (12*sigma*sigma - n*wl*wl - 4*n*wl - 3*n)/(-4*wl - 4);
        int m = round(mIdeal);
        std::vector<int> sizes(n);
        for (int i = 0; i < n; i++) sizes[i] = i < m ? wl : wu;
        return sizes;
    };
    private: void boxBlurH_4(std::vector<double>& scl, std::vector<double>& tcl, int w, int h, int r) {
        double iarr = 1.0 / (r+r+1);
        for (int i=0; i<h; i++) {
            int ti = i*w, li = ti, ri = ti+r;
            double fv = scl[ti], lv = scl[ti+w-1], val = (r+1)*fv;
            for (int j=0; j<r; j++) val += scl[ti+j];
            for (int j=0  ; j<=r ; j++) { val += scl[ri++] - fv       ;   tcl[ti++] = val*iarr; }
            for (int j=r+1; j<w-r; j++) { val += scl[ri++] - scl[li++];   tcl[ti++] = val*iarr; }
            for (int j=w-r; j<w  ; j++) { val += lv        - scl[li++];   tcl[ti++] = val*iarr; }
        }
    };
    private: void boxBlurT_4(std::vector<double>& scl, std::vector<double>& tcl, int w, int h, int r) {
        double iarr = 1.0 / (r+r+1);
        for (int i=0; i<w; i++) {
            int ti = i, li = ti, ri = ti+r*w;
            double fv = scl[ti], lv = scl[ti+w*(h-1)], val = (r+1)*fv;
            for (int j=0; j<r; j++) val += scl[ti+j*w];
            for (int j=0  ; j<=r ; j++) { val += scl[ri] - fv     ;  tcl[ti] = val*iarr;  ri+=w; ti+=w; }
            for (int j=r+1; j<h-r; j++) { val += scl[ri] - scl[li];  tcl[ti] = val*iarr;  li+=w; ri+=w; ti+=w; }
            for (int j=h-r; j<h  ; j++) { val += lv      - scl[li];  tcl[ti] = val*iarr;  li+=w; ti+=w; }
        }
    };
    private: void boxBlur_4(std::vector<double>& scl, std::vector<double>& tcl, int w, int h, int r) {
        for (unsigned int i = 0; i < scl.size(); i++) tcl[i] = scl[i];
        boxBlurH_4(tcl, scl, w, h, r);
        boxBlurT_4(scl, tcl, w, h, r);
    };
    private: void gaussBlur_4(std::vector<double>& scl, std::vector<double>& tcl, int w, int h, double radius) {
        std::vector<int> bxs = boxesForGauss(radius, 3);
        boxBlur_4(scl, tcl, w, h, (bxs[0] - 1) / 2);
        boxBlur_4(tcl, scl, w, h, (bxs[1] - 1) / 2);
        boxBlur_4(scl, tcl, w, h, (bxs[2] - 1) / 2);
    };
    private: void gaussBlur_1(std::vector<double>& scl, std::vector<double>& tcl, int w, int h, double radius) {
        // significant radius to truncate the Gaussian kernel
        int s = ceil(radius * 2.57);
        // loop over all pixels
        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                double val = 0.0, wsum = 0.0;
                // loop over Gaussian kernel
                for (int jk = j-s; jk < j+s+1 ; jk++) {
                    for (int ik = i-s; ik < i+s+1; ik++) {
                        int ii = std::min(w-1, std::max(0, ik));
                        int jj = std::min(h-1, std::max(0, jk));
                        double dsq = (ii-i)*(ii-i)+(jj-j)*(jj-j);
                        double wght = exp(-dsq/(2.0*radius*radius));
                        val += scl[jj*w+ii] * wght;
                        wsum += wght;
                    }
                }
                tcl[j*w+i] = val/wsum;
            }
        }
    };
    // Gaussian smoothing of a 2D grid data_in
    public: CFTOOLS_REAL * GaussSmooth(CFTOOLS_REAL * data_in, const int ni, const int nj, const double fwhm)
    {
        // warning in case the FWHM is too large
        if ((fwhm > ni) || (fwhm > nj)) {
            std::cout<<"=== GaussSmooth: WARNING: fwhm > ni or nj. "
                <<"Smoothing with a kernel FWHM greater than the image size is very inaccurate! ==="<<std::endl;
        }
        // get Gaussian sigma from FWHM
        double sigma = fwhm / 2.35482;
        // vectors for private functions
        std::vector<double> source(ni*nj);
        std::vector<double> target(ni*nj);
        // copy into vector
        for (int i = 0; i < ni*nj; i++) source[i] = data_in[i];
        // if sigma is small we do direct gaussBlur, otherwise we do boxBlur
        if (sigma < 1.8) {
            gaussBlur_1(source, target, ni, nj, sigma);
        }
        else {
            gaussBlur_4(source, target, ni, nj, sigma);
        }
        // prep output container
        CFTOOLS_REAL * data_out = new CFTOOLS_REAL[ni*nj];
        // copy vector to pointer array
        for (int i = 0; i < ni*nj; i++) data_out[i] = target[i];
        return data_out;
    };

}; // end: CFTools
#endif
