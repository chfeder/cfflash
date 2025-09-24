#ifndef _GRID3D_H_
#define _GRID3D_H_

#include <iostream>
#include <fstream>
#include <cstdlib>

/// written by Christoph Federrath, 2010-2025
/// based on a 2D class written by Philipp Girichidis

class GRID3D
{
  public:
    double * field;
    bool * is_set;

  private:
    int dimx, dimy, dimz;
    long dimxy, ntot;
    double xl, xh, yl, yh, zl, zh;
    double Lx, Ly, Lz;
    double minval, maxval;
    double cell_dx, cell_dy, cell_dz;
    static const bool Debug = false;

  //////////////////////////////////////////////////////////////
  // CONSTRUCTOR & SET GLOBAL PROPERTIES
  //////////////////////////////////////////////////////////////

  public: GRID3D()
  {
    minval = 0.0;
    maxval = 0.0;
    dimx = 0;
    dimy = 0;
    dimz = 0;
    dimxy = dimx*dimy;
    ntot = dimxy*dimz;
    field = 0;
    is_set = 0;
    if (Debug) std::cout<<"GRID3D: default constructor called."<<std::endl;
  };
  public: GRID3D(int N)
  {
    minval = 0.0;
    maxval = 0.0;
    dimx = N;
    dimy = N;
    dimz = N;
    dimxy = dimx*dimy;
    ntot = dimxy*dimz;
    field = new double[ntot];
    is_set = new bool[ntot];
    clear();
    if (Debug) std::cout<<"GRID3D: constructor(N) called."<<std::endl;
  };
  public: GRID3D(int Nx, int Ny, int Nz)
  {
    minval = 0.0;
    maxval = 0.0;
    dimx = Nx;
    dimy = Ny;
    dimz = Nz;
    dimxy = dimx*dimy;
    ntot = dimxy*dimz;
    field = new double[ntot];
    is_set = new bool[ntot];
    clear();
    if (Debug) std::cout<<"GRID3D: constructor(Nx,Ny,Nz) called."<<std::endl;
  };

  /// copy constructor (to make a deep copy of a given GRID3D)
  GRID3D(const GRID3D &t)
  {
    ntot = t.ntot;
    dimx = t.dimx;
    dimy = t.dimy;
    dimz = t.dimz;
    dimxy = t.dimxy;
    minval = t.minval;
    maxval = t.maxval;
    if (t.field != 0 && t.is_set != 0)
    {
      field = new double[ntot];
      is_set = new bool[ntot];
      for (long n=0; n<ntot; n++)
      {
        field[n] = t.field[n];
        is_set[n] = t.is_set[n];
      }
    }
    else { field = 0; is_set = 0; }
  };

  /// assigment operator
  GRID3D &operator=(const GRID3D &t)
  {
    if (Debug) std::cout<<"GRID3D: operator= called."<<std::endl;
    if (&t == this) return *this;
    else
    {
      ntot = t.ntot;
      dimx = t.dimx;
      dimy = t.dimy;
      dimz = t.dimz;
      dimxy = t.dimxy;
      minval = t.minval;
      maxval = t.maxval;
      if (field != 0 && is_set != 0) // delete old fields
      {
        delete [] field; delete [] is_set; field = 0; is_set = 0;
      }
      if (t.field != 0 && t.is_set != 0) // assign
      {
        field = new double[ntot];
        is_set = new bool[ntot];
        for (long n=0; n<ntot; n++)
        {
          field[n] = t.field[n];
          is_set[n] = t.is_set[n];
        }
      }
      return *this;
    }
  };

  public: ~GRID3D()
  {
    if (field != 0) { delete [] field; field = 0; if (Debug) std::cout<<"GRID3D: field destroyed."<<std::endl;}
    if (is_set != 0) { delete [] is_set; is_set = 0; if (Debug) std::cout<<"GRID3D: is_set destroyed."<<std::endl;}
    if (Debug) std::cout<<"GRID3D: destructor called."<<std::endl;
  };

  public: inline void set_bnds(double low, double high)
  {
    set_bnds(low, high, low, high, low, high);
  }
  public: inline void set_bnds(double xlow, double xhigh, double ylow, double yhigh, double zlow, double zhigh)
  {
    if(xhigh-xlow <= 0.0)
    {
      std::cout << " xlow > xhigh !! swapping these values !!" << std::endl;
      double temp = xlow; xlow = xhigh; xhigh = temp;
    }
    if(yhigh-ylow <= 0.0)
    {
      std::cout << " ylow > yhigh !! swapping these values !!" << std::endl;
      double temp = ylow; ylow = yhigh; yhigh = temp;
    }
    if(zhigh-zlow <= 0.0)
    {
      std::cout << " zlow > zhigh !! swapping these values !!" << std::endl;
      double temp = zlow; zlow = zhigh; zhigh = temp;
    }
    xl = xlow; xh = xhigh;
    yl = ylow; yh = yhigh;
    zl = zlow; zh = zhigh;
    Lx = xh-xl;
    Ly = yh-yl;
    Lz = zh-zl;
    cell_dx = (xhigh-xlow)/(double)dimx;
    cell_dy = (yhigh-ylow)/(double)dimy;
    cell_dz = (zhigh-zlow)/(double)dimz;
  }


  //////////////////////////////////////////////////////////////
  // GLOBAL OBJECT INFORMATION
  //////////////////////////////////////////////////////////////

  public: inline int get_dimx() { return dimx; }
  public: inline int get_dimy() { return dimy; }
  public: inline int get_dimz() { return dimz; }
  public: inline long get_ntot() { return ntot; }

  public: inline double get_xmin() { return xl; }
  public: inline double get_xmax() { return xh; }
  public: inline double get_ymin() { return yl; }
  public: inline double get_ymax() { return yh; }
  public: inline double get_zmin() { return zl; }
  public: inline double get_zmax() { return zh; }

  public: inline double get_dx() { return cell_dx; }
  public: inline double get_dy() { return cell_dy; }
  public: inline double get_dz() { return cell_dz; }

  public: inline double get_Lx() { return Lx; }
  public: inline double get_Ly() { return Ly; }
  public: inline double get_Lz() { return Lz; }

  public: inline double get_dV() { return cell_dx*cell_dy*cell_dz; }

  public: inline void print_bnds(std::ostream & out = std::cout)
  {
    out << " grid bounds"
        << " [" << xl << ":" << xh << "]"
        << " [" << yl << ":" << yh << "]"
        << " [" << zl << ":" << zh << "]" << std::endl;
  }
  public: inline void print_dims(std::ostream & out = std::cout)
  {
    out << " grid dimensions [" << dimx << ", "  << dimy<< ", " << dimz << "]" << std::endl;;
  }
  public: inline void print_minmax(std::ostream & out = std::cout)
  {
    double MIN, MAX;
    minmax(MIN, MAX);
    out << " min, max value : " << minval << ", " << maxval << std::endl;
  }
  public: inline void print_sum(std::ostream & out = std::cout)
  {
    double val = sum();
    out << " sum    : " << val << std::endl;
  }
  public: inline void print_sum_vw(std::ostream & out = std::cout)
  {
    double val_vw = sum_vw();
    out << " sum_vw : " << val_vw << std::endl;
  }

  public: double first_set()
  {
    // find first data point that is set and use it as starting point for min and max
    long counter = 0;
    while(counter < ntot)
    {
      if (is_set[counter]) return field[counter];
      counter++;
    }
    // no field is set so far!!
    std::cout << " NO DATA POINT IS SET IN YOUR FIELD, ABORT!!" << std::endl;
    exit(1);
    return 0.0;
  }

  public: double min()
  {
    double MIN = first_set();
    for (long n=0; n<ntot; n++)
      if (is_set[n] && (field[n] < MIN)) MIN = field[n];
    minval = MIN;
    return minval;
  }
  public: double max()
  {
    double MAX = first_set();
    for (long n=0; n<ntot; n++)
      if (is_set[n] && (field[n] > MAX)) MAX = field[n];
    maxval = MAX;
    return maxval;
  }
  public: void minmax(double &tmin, double &tmax)
  {
    tmin = first_set();
    tmax = tmin;
    for (long n=0; n<ntot; n++)
    {
      if(is_set[n] && (field[n] < tmin)) tmin = field[n];
      if(is_set[n] && (field[n] > tmax)) tmax = field[n];
    }
    minval = tmin;
    maxval = tmax;
  }
  public: double sum()
  {
    double ret = 0.;
    for (long n=0; n<ntot; n++)
      if(is_set[n]) ret += field[n];
    return ret;
  }
  public: double sum_vw()
  {
    double ret = sum()*Lx*Ly*Lz/(double)dimx/(double)dimy/(double)dimz;
    return ret;
  }


  //////////////////////////////////////////////////////////////
  // GET SINGLE FIELD INFORMATION
  //////////////////////////////////////////////////////////////

  public: inline double getX(int i)
  {
    return cell_center_x(i);
  }
  public: inline double getY(int i)
  {
    return cell_center_y(i);
  }
  public: inline double getZ(int i)
  {
    return cell_center_z(i);
  }
  public: inline long get_index(int ix, int iy, int iz)
  {
    return index(ix, iy, iz);
  }
  public: inline double get_val(int ix, int iy, int iz)
  {
    if((ix>=0) && (ix<dimx) && (iy>=0) && (iy<dimy) && (iz>=0) && (iz<dimz))
    {
      long n = index(ix, iy, iz);
      return field[n];
    }
    else return 0.0;
  }
  /// returns the index of the cell in which (x,y,z) is located
  private: inline void get_cell_index(double x, double y, double z, int &ix, int &iy, int &iz)
  {
    ix = std::max(0, std::min((int)((x-xl)/cell_dx), dimx-1));
    iy = std::max(0, std::min((int)((y-yl)/cell_dy), dimy-1));
    iz = std::max(0, std::min((int)((z-zl)/cell_dz), dimz-1));
  }
  private: inline long index(int ix, int iy, int iz)
  {
    return iz*dimxy + iy*dimx + ix;
  }
  private: inline double cell_center_x(int i)
  {
    return xl + ((double)i + 0.5)*cell_dx;
  }
  private: inline double cell_center_y(int i)
  {
    return yl + ((double)i + 0.5)*cell_dy;
  }
  private: inline double cell_center_z(int i)
  {
    return zl + ((double)i + 0.5)*cell_dz;
  }
  private: inline double get_cell_xlow(int ix)
  {
    return xl + ((double)ix)*cell_dx;
  }
  private: inline double get_cell_xhigh(int ix)
  {
    return xl + ((double)(ix+1))*cell_dx;
  }
  private: inline double get_cell_ylow(int iy)
  {
    return yl + ((double)iy)*cell_dy;
  }
  private: inline double get_cell_yhigh(int iy)
  {
    return yl + ((double)(iy+1))*cell_dy;
  }
  private: inline double get_cell_zlow(int iz)
  {
    return zl + ((double)iz)*cell_dz;
  }
  private: inline double get_cell_zhigh(int iz)
  {
    return zl + ((double)(iz+1))*cell_dz;
  }
  private: inline void cell_corner_x(int i, double &xlow, double &xhigh)
  {
    xlow  = xl + ((double)i)*cell_dx;
    xhigh = xl + ((double)(i+1))*cell_dx;
  }
  private: inline void cell_corner_y(int i, double &ylow, double &yhigh)
  {
    ylow  = yl + ((double)i)*cell_dy;
    yhigh = yl + ((double)(i+1))*cell_dy;
  }
  private: inline void cell_corner_z(int i, double &zlow, double &zhigh)
  {
    zlow  = zl + ((double)i)*cell_dz;
    zhigh = zl + ((double)(i+1))*cell_dz;
  }
  private: inline void cell_corners(int ix, int iy, int iz,
                                    double &xlow, double &xhigh,
                                    double &ylow, double &yhigh,
                                    double &zlow, double &zhigh)
  {
    xlow  = xl + ((double)ix)*cell_dx;
    xhigh = xl + ((double)(ix+1))*cell_dx;
    ylow  = yl + ((double)iy)*cell_dy;
    yhigh = yl + ((double)(iy+1))*cell_dy;
    zlow  = zl + ((double)iz)*cell_dz;
    zhigh = zl + ((double)(iz+1))*cell_dz;
  }

  /// this subroutine adds cell-centered data values given at (x,y,z) to the appropriate cell of the GRID3D
  /// taking into account all overlapping fractions of the original data
  /// This routine does not use interpolation (it is thus conserving the integral over all datavalues)
  public: inline void add_coord_fields(double x, double y, double z, double dx, double dy, double dz, double val)
  {
    double xlow  = x-0.5*dx;
    double xhigh = x+0.5*dx;
    double ylow  = y-0.5*dy;
    double yhigh = y+0.5*dy;
    double zlow  = z-0.5*dz;
    double zhigh = z+0.5*dz;
    double cell_vol = cell_dx*cell_dy*cell_dz;
    double data_vol = dx*dy*dz;
    int ix0, ix1, iy0, iy1, iz0, iz1;
    get_cell_index(xlow, ylow, zlow, ix0, iy0, iz0);
    get_cell_index(xhigh, yhigh, zhigh, ix1, iy1, iz1);
    double dxm = std::min(cell_dx, dx);
    double dym = std::min(cell_dy, dy);
    double dzm = std::min(cell_dz, dz);
    // specific based value to be scaled by a resepctive volume fraction below
    double spec_val = val / data_vol / cell_vol;
    // check whether data cell entirely overlaps with grid cell
    if ((ix0==ix1) && (iy0==iy1) && (iz0==iz1))
    {
      if (Debug) std::cout << "data cell overlaps entirely with grid cell" << std::endl;
      double delx = std::min(dxm, std::min(std::max(0.0,(xhigh-get_cell_xlow(ix0))), std::max(0.0,(get_cell_xhigh(ix0)-xlow))));
      double dely = std::min(dym, std::min(std::max(0.0,(yhigh-get_cell_ylow(iy0))), std::max(0.0,(get_cell_yhigh(iy0)-ylow))));
      double delz = std::min(dzm, std::min(std::max(0.0,(zhigh-get_cell_zlow(iz0))), std::max(0.0,(get_cell_zhigh(iz0)-zlow))));
      add(ix0, iy0, iz0, delx*dely*delz*spec_val);
      return;
    }
    // data cell occupies more than one cell in x direction
    if ((ix0!=ix1) && (iy0==iy1) && (iz0==iz1))
    {
      if (Debug) std::cout << "data cell occupies more than one cell in x direction" << std::endl;
      double dely = std::min(dym, std::min(std::max(0.0,(yhigh-get_cell_ylow(iy0))), std::max(0.0,(get_cell_yhigh(iy0)-ylow))));
      double delz = std::min(dzm, std::min(std::max(0.0,(zhigh-get_cell_zlow(iz0))), std::max(0.0,(get_cell_zhigh(iz0)-zlow))));
      for (int ix=ix0; ix<=ix1; ix++)
      {
        double delx = std::min(dxm, std::min(std::max(0.0,(xhigh-get_cell_xlow(ix))), std::max(0.0,(get_cell_xhigh(ix)-xlow))));
        add(ix, iy0, iz0, delx*dely*delz*spec_val);
      }
      return;
    }
    // data cell occupies more than one cell in y direction
    if ((ix0==ix1) && (iy0!=iy1) && (iz0==iz1))
    {
      if (Debug) std::cout << "data cell occupies more than one cell in y direction" << std::endl;
      double delx = std::min(dxm, std::min(std::max(0.0,(xhigh-get_cell_xlow(ix0))), std::max(0.0,(get_cell_xhigh(ix0)-xlow))));
      double delz = std::min(dzm, std::min(std::max(0.0,(zhigh-get_cell_zlow(iz0))), std::max(0.0,(get_cell_zhigh(iz0)-zlow))));
      for (int iy=iy0; iy<=iy1; iy++)
      {
        double dely = std::min(dym, std::min(std::max(0.0,(yhigh-get_cell_ylow(iy))), std::max(0.0,(get_cell_yhigh(iy)-ylow))));
        add(ix0, iy, iz0, delx*dely*delz*spec_val);
      }
      return;
    }
    // data cell occupies more than one cell in z direction
    if ((ix0==ix1) && (iy0==iy1) && (iz0!=iz1))
    {
      if (Debug) std::cout << "data cell occupies more than one cell in z direction" << std::endl;
      double delx = std::min(dxm, std::min(std::max(0.0,(xhigh-get_cell_xlow(ix0))), std::max(0.0,(get_cell_xhigh(ix0)-xlow))));
      double dely = std::min(dym, std::min(std::max(0.0,(yhigh-get_cell_ylow(iy0))), std::max(0.0,(get_cell_yhigh(iy0)-ylow))));
      for (int iz=iz0; iz<=iz1; iz++)
      {
        double delz = std::min(dzm, std::min(std::max(0.0,(zhigh-get_cell_zlow(iz))), std::max(0.0,(get_cell_zhigh(iz)-zlow))));
        add(ix0, iy0, iz, delx*dely*delz*spec_val);
      }
      return;
    }
    // data cell occupies more than one cell in x and y direction
    if ((ix0!=ix1) && (iy0!=iy1) && (iz0==iz1))
    {
      if (Debug) std::cout << "data cell occupies more than one cell in x and y direction" << std::endl;
      double delz = std::min(dzm, std::min(std::max(0.0,(zhigh-get_cell_zlow(iz0))), std::max(0.0,(get_cell_zhigh(iz0)-zlow))));
      for (int iy=iy0; iy<=iy1; iy++)
        for (int ix=ix0; ix<=ix1; ix++)
        {
          double delx = std::min(dxm, std::min(std::max(0.0,(xhigh-get_cell_xlow(ix))), std::max(0.0,(get_cell_xhigh(ix)-xlow))));
          double dely = std::min(dym, std::min(std::max(0.0,(yhigh-get_cell_ylow(iy))), std::max(0.0,(get_cell_yhigh(iy)-ylow))));
          add(ix, iy, iz0, delx*dely*delz*spec_val);
        }
      return;
    }
    // data cell occupies more than one cell in x and z direction
    if ((ix0!=ix1) && (iy0==iy1) && (iz0!=iz1))
    {
      if (Debug) std::cout << "data cell occupies more than one cell in x and z direction" << std::endl;
      double dely = std::min(dym, std::min(std::max(0.0,(yhigh-get_cell_ylow(iy0))), std::max(0.0,(get_cell_yhigh(iy0)-ylow))));
      for (int iz=iz0; iz<=iz1; iz++)
        for (int ix=ix0; ix<=ix1; ix++)
        {
          double delx = std::min(dxm, std::min(std::max(0.0,(xhigh-get_cell_xlow(ix))), std::max(0.0,(get_cell_xhigh(ix)-xlow))));
          double delz = std::min(dzm, std::min(std::max(0.0,(zhigh-get_cell_zlow(iz))), std::max(0.0,(get_cell_zhigh(iz)-zlow))));
          add(ix, iy0, iz, delx*dely*delz*spec_val);
        }
      return;
    }
    // data cell occupies more than one cell in y and z direction
    if ((ix0==ix1) && (iy0!=iy1) && (iz0!=iz1))
    {
      if (Debug) std::cout << "data cell occupies more than one cell in y and z direction" << std::endl;
      double delx = std::min(dxm, std::min(std::max(0.0,(xhigh-get_cell_xlow(ix0))), std::max(0.0,(get_cell_xhigh(ix0)-xlow))));
      for (int iz=iz0; iz<=iz1; iz++)
        for (int iy=iy0; iy<=iy1; iy++)
        {
          double dely = std::min(dym, std::min(std::max(0.0,(yhigh-get_cell_ylow(iy))), std::max(0.0,(get_cell_yhigh(iy)-ylow))));
          double delz = std::min(dzm, std::min(std::max(0.0,(zhigh-get_cell_zlow(iz))), std::max(0.0,(get_cell_zhigh(iz)-zlow))));
          add(ix0, iy, iz, delx*dely*delz*spec_val);
        }
      return;
    }
    // data cell occupies a larger volume
    if ((ix0!=ix1) && (iy0!=iy1) && (iz0!=iz1))
    {
      if (Debug) std::cout << "data cell occupies a larger volume, ix0, ix1, iy0, iy1, iz0, iz1: " <<ix0<<", "<<ix1<<", "<<iy0<<", "<<iy1<<", "<<iz0<<", "<<iz1<<std::endl;
      for (int iz=iz0; iz<=iz1; iz++)
        for (int iy=iy0; iy<=iy1; iy++)
          for (int ix=ix0; ix<=ix1; ix++)
          {
            double delx = std::min(dxm, std::min(std::max(0.0,(xhigh-get_cell_xlow(ix))), std::max(0.0,(get_cell_xhigh(ix)-xlow))));
            double dely = std::min(dym, std::min(std::max(0.0,(yhigh-get_cell_ylow(iy))), std::max(0.0,(get_cell_yhigh(iy)-ylow))));
            double delz = std::min(dzm, std::min(std::max(0.0,(zhigh-get_cell_zlow(iz))), std::max(0.0,(get_cell_zhigh(iz)-zlow))));
            add(ix, iy, iz, delx*dely*delz*spec_val);
          }
      return;
    } // data cell occupies a larger volume
  } // end of add_coord_fields(...)

  public: inline void add_coord_fields_simpler_but_slower(double x, double y, double z, double dx, double dy, double dz, double val)
  {
    double xlow  = x-0.5*dx;
    double xhigh = x+0.5*dx;
    double ylow  = y-0.5*dy;
    double yhigh = y+0.5*dy;
    double zlow  = z-0.5*dz;
    double zhigh = z+0.5*dz;
    double cell_vol = cell_dx*cell_dy*cell_dz;
    double data_vol = dx*dy*dz;
    int ix0, ix1, iy0, iy1, iz0, iz1;
    get_cell_index(xlow, ylow, zlow, ix0, iy0, iz0);
    get_cell_index(xhigh, yhigh, zhigh, ix1, iy1, iz1);
    double dxm = std::min(cell_dx, dx);
    double dym = std::min(cell_dy, dy);
    double dzm = std::min(cell_dz, dz);
    double spec_val = val / data_vol / cell_vol;
    for (int iz=iz0; iz<=iz1; iz++)
      for (int iy=iy0; iy<=iy1; iy++)
        for (int ix=ix0; ix<=ix1; ix++)
        {
          double delx = std::min(dxm, std::min(std::max(0.0,(xhigh-get_cell_xlow(ix))), std::max(0.0,(get_cell_xhigh(ix)-xlow))));
          double dely = std::min(dym, std::min(std::max(0.0,(yhigh-get_cell_ylow(iy))), std::max(0.0,(get_cell_yhigh(iy)-ylow))));
          double delz = std::min(dzm, std::min(std::max(0.0,(zhigh-get_cell_zlow(iz))), std::max(0.0,(get_cell_zhigh(iz)-zlow))));
          add(ix, iy, iz, delx*dely*delz*spec_val);
        }
  } // end of add_coord_fields_simpler_but_slower(...)

  private: inline void add(int ix, int iy, int iz, double val)
  {
    long n = index(ix, iy, iz);
    field[n] += val;
    is_set[n] = true;
  }

  //////////////////////////////////////////////////////////////
  // MANIPULATE ENTIRE FIELD
  //////////////////////////////////////////////////////////////

  public: inline void clear()
  {
    for (long n=0; n<ntot; n++)
    {
      field[n] = 0.0;
      is_set[n] = false;
    }
  }

  public: inline void init(double val)
  {
    for (long n=0; n<ntot; n++)
    {
      field[n] = val;
      is_set[n] = true;
    }
  }

  //////////////////////////////////////////////////////////////
  // WRITE OUTPUT / DATA / PICTURE
  //////////////////////////////////////////////////////////////

  public: void print_field()
  {
    for (int k=0; k<dimz; k++)
    {
      for (int j=0; j<dimy; j++)
      {
        std::cout<<std::endl;
        for (int i=0; i<dimx; i++)
        {
          long n = index(i, j, k);
          std::cout<<" (i,j,k)=("<<i<<","<<j<<","<<k<<")f="<<field[n];
        }
      }
    }
    std::cout<<std::endl;
  }

};
#endif
