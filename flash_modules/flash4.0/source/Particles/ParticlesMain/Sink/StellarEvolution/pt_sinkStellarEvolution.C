#include "SinkStellarEvolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "mangle_names.h"

// === creates a new StarParticle object that contains one sink particle and updates evolution state
// written by C. Federrath 2016
extern "C" void FTOC(update_stellar_evolution_c)(double* m_in, double* mdot_in, double* mdeut_in_out, 
                                                 double* r_in_out, double* burnState_in_out,
                                                 double* luminosity_out, double* dt_in)
{
    // make a new sp
    StarParticle sp = StarParticle(*m_in, *mdot_in, *mdeut_in_out, *r_in_out, (int)*burnState_in_out, *dt_in);
    // update evolution state
    sp.updateState();
    // return updated values
    *mdeut_in_out = sp.mdeut;
    *r_in_out = sp.r;
    *burnState_in_out = (double)sp.burnState;
    // return luminosity
    *luminosity_out = sp.luminosity();
}

/****************************/
/* Constructor / Destructor */
/****************************/

StarParticle::StarParticle(double m_in, double mdot_in, double mdeut_in, 
                           double r_in, int burnState_in, double dt_in)
{
  // set main variables
  m = m_in;
  mdot = mdot_in;
  mdeut = mdeut_in;
  r = r_in;
  dt = dt_in;
  // parse burnState
  if (burnState_in == Uninitialized) {burnState = Uninitialized; n = 0.0;}
  if (burnState_in == None) {burnState = None; n = nInit(mdot);}
  if (burnState_in == VariableCoreDeuterium) {burnState = VariableCoreDeuterium; n = 1.5;}
  if (burnState_in == SteadyCoreDeuterium) {burnState = SteadyCoreDeuterium; n = 1.5;}
  if (burnState_in == ShellDeuterium) {burnState = ShellDeuterium; n = 3.0;}
  if (burnState_in == ZAMS) {burnState = ZAMS; n = 3.0;}
}

StarParticle::~StarParticle() {}

/***************************/
/* Initialization routines */
/***************************/

#define MSUN     1.99e33
#define RSUN     6.96e10
#define YRSEC    3.16e7  /* Seconds in a year */

inline
double
StarParticle::nInit(double mdotInit)
{
  if (mdotInit == 0.0) return(1.5);
  double aGinit = 1.475 + 0.07*log10(mdotInit*YRSEC/MSUN);
  double nval = 5.0 - 3.0/aGinit;
  if (nval < 1.5) nval = 1.5;
  if (nval > 3.0) nval = 3.0;
  return( nval );
}

inline
double
StarParticle::radInit(double mdotInit)
{
  return ( RSUN * fmax(2.5*pow(mdotInit*YRSEC/MSUN*1.0e5, 0.2), 2.0) );
}

#undef RSUN
#undef MSUN
#undef YRSEC

/*****************************/
/* Polytropic model routines */
/*****************************/

/* Physical constants */
#define PI     3.1415927
#define G      6.67e-8
#define A      7.56e-15
#define KB     1.38e-16
#define MH     1.67e-24
#define SIGMA  5.67e-5
#define MU     0.613  /* Mean molecular weight of a fully ionized gas of solar composition */

// For a polytrope, the gravitational energy is aG G M^2 / R, aG = -3/(5-n)
inline
double
StarParticle::aG()
{ 
  return ( 3.0/(5.0-n) );
}

// The central density in a polytropic model, found by table lookup.
// See Kippenhahn & Weigert.
inline
double
StarParticle::rhoc(double mass)
{
  if ((n>=1.5) && (n<=3.0)) {}
  else {std::cout<<"rhoc: SOMETHING WRONG!! Exiting."<<std::endl; exit(EXIT_FAILURE);};
  // Table of values of rho_mean / rho_c for n=1.5 to 3.1 in intervals of 0.1
  static double rhofactab[] = {
    0.166931, 0.14742, 0.129933, 0.114265, 0.100242,
    0.0877, 0.0764968, 0.0665109, 0.0576198, 0.0497216,
    0.0427224, 0.0365357, 0.0310837, 0.0262952, 0.0221057,
    0.0184553, 0.01529
  };
  int itab = (int) floor((n-1.5)/0.1);
  double wgt = (n - (1.5 + 0.1*itab)) / 0.1;
  double rhofac = rhofactab[itab]*(1.0-wgt) + rhofactab[itab+1]*wgt;
  return ( mass / (4./3.*PI*r*r*r) / rhofac );
}

// The central pressure in a polytropic model, found by table lookup.
// See Kippenhahn & Weigert.
inline
double
StarParticle::Pc(double mass)
{
  static double pfactab[] = {
    0.770087, 0.889001, 1.02979, 1.19731, 1.39753,
    1.63818, 1.92909, 2.2825, 2.71504, 3.24792, 3.90921,
    4.73657, 5.78067, 7.11088, 8.82286, 11.0515, 13.9885
  };
  int itab = (int) floor((n-1.5)/0.1);
  double wgt = (n - (1.5 + 0.1*itab)) / 0.1;
  double pfac = pfactab[itab]*(1.0-wgt) + pfactab[itab+1]*wgt;
  return ( pfac * G * mass*mass/(r*r*r*r) );
}

// The central temperature in a protostar, found by using a bisection
// method to solve Pc = rho_c k Tc / (mu mH) + 1/3 a Tc^4.
double
StarParticle::Tc(double mass, double rhoc1, double Pc1)
{
  if (rhoc1 == -1.0) rhoc1 = rhoc(mass);
  if (Pc1 == -1.0) Pc1 = Pc(mass);
#define JMAX 40
#define TOL 1.0e-7
  double Tgas, Trad;
  int j;
  double dx, f, fmid, xmid, rtb;
  double x1, x2;
  char errstr[256];

  x1 = 0.0;
  Tgas = Pc1*MU*MH/(KB*rhoc1);
  Trad = pow(3*Pc1/A, 0.25);
  x2 = (Trad > Tgas) ? 2*Trad : 2*Tgas;
  f = Pc1 - rhoc1*KB*x1/(MU*MH) - A*pow(x1,4)/3.0;
  fmid=Pc1 - rhoc1*KB*x2/(MU*MH) - A*pow(x2,4)/3.0;
  rtb = f < 0.0 ? (dx=x2-x1,x1) : (dx=x1-x2,x2);
  for (j=1;j<=JMAX;j++) {
    xmid=rtb+(dx *= 0.5);
    fmid = Pc1 - rhoc1*KB*xmid/(MU*MH) - A*pow(xmid,4)/3.0;
    if (fmid <= 0.0) rtb=xmid;
    if (fabs(dx) < TOL*fabs(xmid) || fmid == 0.0) return rtb;
  }
  sprintf(errstr,
      "Tc(): bisection solve didn't converge, P_c = %e, rho_c = %e, mass = %e Tgas = %e Trad = %e rad = %e ",
      Pc1, rhoc1, mass, Tgas, Trad, r);
  std::cout<<errstr<<std::endl;
  return (-1);
#undef JMAX
#undef TOL
}

inline
double
StarParticle::betac(double mass, double rhoc1, double Pc1, double Tc1)
{
  if (rhoc1 == -1.0) rhoc1 = rhoc(mass);
  if (Pc1 == -1.0) Pc1 = Pc(mass);
  if (Tc1 == -1.0) Tc1 = Tc(mass, rhoc1, Pc1);
  return ( rhoc1*KB*Tc1/(MU*MH) / Pc1 );
}

double
StarParticle::beta(double mass, double rhoc1, double Pc1)
{
  if (n==3.0) {
    // In this case we solve the Eddington quartic,
    // P_c^3 = (3/a) (k / (mu mH))^4 (1 - beta) / beta^4 rho_c^4
    // for beta
#define JMAX 40
#define BETAMIN 1.0e-4
#define BETAMAX 1.0
#define TOL 1.0e-7
    int j;
    double dx, f, fmid, xmid, rtb;
    double x1, x2;
    double coef;

    if (rhoc1 == -1.0) rhoc1 = rhoc(mass);
    if (Pc1 == -1.0) Pc1 = Pc(mass);
    coef = 3/A*pow(KB*rhoc1/(MU*MH),4);
    x1=BETAMIN;
    x2=BETAMAX;
    f = pow(Pc1,3) - coef * (1.0-x1)/pow(x1,4);
    fmid = pow(Pc1,3) - coef * (1.0-x2)/pow(x2,4);
    rtb = f < 0.0 ? (dx=x2-x1,x1) : (dx=x1-x2,x2);
    for (j=1;j<=JMAX;j++) {
      xmid=rtb+(dx *= 0.5);
      fmid = pow(Pc1,3) - coef * (1.0-xmid)/pow(xmid,4);
      if (fmid <= 0.0) rtb=xmid;
      if (fabs(dx) < TOL*fabs(xmid) || fmid == 0.0) return rtb;
    }
    std::cout<<"beta(): bisection solve failed to converge"<<std::endl;
    return(-1);
#undef JMAX
#undef BETAMIN
#undef BETAMAX
#undef TOL
  } else {
    // For n != 3, we use a table lookup. The values of beta have been
    // pre-computed with mathematica. The table goes from M=5 to 50 solar
    // masses in steps of 2.5 M_sun, and from n=1.5 to n=3 in steps of 0.5.
    // We should never call this routine with M > 50 Msun, since by then
    // the star should be fully on the main sequence.
#define MSUN     1.99e33
#define MTABMIN  (5.0*MSUN)
#define MTABMAX  (50.0*MSUN)
#define MTABSTEP (2.5*MSUN)
#define NTABMIN  1.5
#define NTABMAX  3.0
#define NTABSTEP 0.5
    if (mass < MTABMIN) return(1.0);  // Set beta = 1 for M < 5 Msun
    if ((mass >= MTABMAX) || (n >= NTABMAX)) {
        std::cout<<"beta(): off interpolation table"<<std::endl;
      return(-1.0);
    }
    static double betatab[19][4] = {
      {0.98785, 0.988928, 0.98947, 0.989634}, 
      {0.97438, 0.976428, 0.977462, 0.977774}, 
      {0.957927, 0.960895, 0.962397, 0.962846}, 
      {0.939787, 0.943497, 0.945369, 0.945922}, 
      {0.92091, 0.925151, 0.927276, 0.927896}, 
      {0.901932, 0.906512, 0.908785, 0.909436}, 
      {0.883254, 0.888017, 0.890353, 0.891013}, 
      {0.865111, 0.86994, 0.872277, 0.872927}, 
      {0.847635, 0.852445, 0.854739, 0.855367}, 
      {0.830886, 0.835619, 0.837842, 0.838441}, 
      {0.814885, 0.8195, 0.821635, 0.822201}, 
      {0.799625, 0.804095, 0.806133, 0.806664}, 
      {0.785082, 0.789394, 0.791328, 0.791825}, 
      {0.771226, 0.775371, 0.777202, 0.777665}, 
      {0.758022, 0.761997, 0.763726, 0.764156}, 
      {0.745433, 0.749238, 0.750869, 0.751268}, 
      {0.733423, 0.73706, 0.738596, 0.738966}, 
      {0.721954, 0.725429, 0.726874, 0.727216}, 
      {0.710993, 0.714311, 0.715671, 0.715987}
    };

    // Locate ourselves on the table and do a linear interpolation
    int midx = (int) floor((mass-MTABMIN)/MTABSTEP);
    double mwgt = (mass-(MTABMIN+midx*MTABSTEP)) / MTABSTEP;
    int nidx = (int) floor((n-NTABMIN)/NTABSTEP);
    double nwgt = (n-(NTABMIN+nidx*NTABSTEP)) / NTABSTEP;
    return ( betatab[midx][nidx]*(1.0-mwgt)*(1.0-nwgt) +
             betatab[midx+1][nidx]*mwgt*(1.0-nwgt) +
             betatab[midx][nidx+1]*(1.0-mwgt)*nwgt +
             betatab[midx+1][nidx+1]*mwgt*nwgt );
  }
#undef MSUN
#undef MTABMIN
#undef MTABMAX
#undef MTABSTEP
#undef NTABMIN
#undef NTABMAX
#undef NTABSTEP
}

#define DM (0.01*m)
double
StarParticle::dlogBetaOverBetac_dlogM(double beta_1)
{
  // If n==3, beta = beta_c independent of M, so return 0
  if (n==3) return(0.0);

  // Otherwise take a numerical derivative
  double beta1;
  if (beta_1==-1.0) beta1 = beta(m);
  else beta1 = beta_1;
  double beta2 = beta(m+DM);
  double betac1 = betac(m);
  double betac2 = betac(m+DM);
  return ( m/(beta1/betac1) * ((beta2/betac2) - (beta1/betac1)) / DM );
}

double
StarParticle::dlogBeta_dlogM(double beta_1)
{
  // Take a numerical derivative
  double beta1;
  if (beta_1==-1.0) beta1 = beta(m);
  else beta1 = beta_1;
  double beta2 = beta(m+DM);
  return ( m/beta1 * (beta2-beta1) / DM );
}
#undef DM

#undef PI
#undef G
#undef A
#undef KB
#undef MH
#undef MU


/*******************/
/* Model functions */
/*******************/

#define PI     3.1415927
#define G      6.67e-8
#define NAVOG  6.022e23
#define ERGEV  1.6e-12    /* Number of ergs per eV */
#define FACC   0.5        /* Fraction of accreted energy that comes out as
                             radiation, rather than being advected into the
                             stellar interior or used to drive a wind */
#define FK     0.5        /* Fraction of energy that falls into the sink
                             particle but is radiated away from the
                             inner disk before reaching the stellar surface */
#define FRAD   0.33       /* A radiative barrier forms when L_deuterium <=
                             FRAD * L_ZAMS. See McKee & Tan 2002 */
#define SHELLFAC 2.1      /* Radius increases by SHELLFAC when shell burning starts */
#define THAY   3000.0     /* Hayashi temperature */
#define TDEUT  1.5e6      /* Temperature when deuterium burning starts */
#define PSIION (16.0*ERGEV*NAVOG) /* Energy per gram needed to dissociate
                                     and ionize a molecular gas with solar abundances */
#define PSID   (100*ERGEV*NAVOG)  /* Energy per gram released by burning
                                     the deuterium in a gas with solar abundances */
#define MSUN   1.99e33    /* Solar mass */
#define MRADMIN (0.01*MSUN)   /* Minimum mass at which we use the model */

double 
StarParticle::luminosity()
{
  if (burnState == Uninitialized) return (0.0);
  double lum =  lStar() + lDisk();
  return ( lum );
}

double
StarParticle::lStar()
{
  double lstar = lZAMS() + lAcc();
  double Teff = pow(lstar / (4. * PI * r*r * SIGMA), 0.25);
  if (Teff > THAY) {
    return ( lstar );
  } else {
    return ( 4.*PI*r*r*SIGMA*pow(THAY, 4) );
  }
}

inline
double
StarParticle::lAcc()
{
  return ( FACC * FK * G * m * mdot / r );
}

double
StarParticle::lDisk()
{
  return ( (1.0 - FK) * G * m * mdot / r );
}

double
StarParticle::lDeut(double beta1)
{
  switch (burnState) {
  case Uninitialized: return(0.0);
  case None: return(0.0);
  case VariableCoreDeuterium: {
    if (beta1 == -1.0) beta1=beta(m);
    return ( lStar() + eDotIon() + G*m*mdot/r *
	    (1.0 - FK - aG()*beta1/2.0 *
	     (1.0 + dlogBetaOverBetac_dlogM(beta1))) );
  }
  case SteadyCoreDeuterium: return( mdot * PSID );
  case ShellDeuterium: return( mdot * PSID );
  default: std::cout<<"lDeut(): bad value of burnState"<<std::endl;
  }
  return (-1.0); // Never get here
}
  
inline
double
StarParticle::eDotIon()
{
  return ( mdot * PSIION );
}

inline
double
StarParticle::dlogR_dlogM(double beta1)
{
  if (beta1==-1.0) beta1 = beta(m);
  return ( 2.0 - 2.0/(aG()*beta1) * (1.0 - FK) +
	  dlogBeta_dlogM(beta1) - 2.0*r/(aG()*beta1*G*m*mdot) * 
	  (lStar() + eDotIon() - lDeut(beta1)) );
}

void
StarParticle::updateState()
{
  // update Deuterium mass
  mdeut += mdot*dt;

  // Do nothing if we are below the minimum mass or we have just been
  // created and thus don't have a valid mdot. Otherwise, if we're not
  // initilized, then initialize here
  if (burnState == Uninitialized) {
    if ((m < MRADMIN) || (mdot == 0.0)) {
      return;
    }
    n = nInit(mdot);
    r = radInit(mdot);
    burnState = None;
    if (true) {
      std::cout<< "Initializing star with m = " << m << std::endl;
    }
  }

  // update n
  //if (burnState == VariableCoreDeuterium) n = 1.5;
  //if (burnState == SteadyCoreDeuterium) n = 1.5;
  //if (burnState == ShellDeuterium) n = 3.0;

  // Update the radius as long as not on the ZAMS yet
  if (burnState != ZAMS)
  {
    double beta1 = beta(m);
    double dr = (2.0*mdot/m*r*(FK/(aG()*beta1)+1.0-1.0/(aG()*beta1))
                 + beta1/m * dlogBeta_dlogM(beta1) * mdot * r / beta1
                 - 2.0/(beta1*aG())*r*r/(G*m*m)*(lStar()+eDotIon()-lDeut(beta1)));
    double rdottime = fabs(r/dr  )/100.0;
    double mdottime = fabs(m/mdot)/100.0;
    
    if (rdottime < dt)
    {
        int rdotfac = ceil(dt/rdottime);
        double dtprime = dt/rdotfac;
        for(int rdotloop = 0; rdotloop < rdotfac; rdotloop++)
        {
            beta1 = beta(m);
            dr = (2.0*mdot/m*r*(FK/(aG()*beta1)+1.0-1.0/(aG()*beta1))
                  + beta1/m * dlogBeta_dlogM(beta1) * mdot * r / beta1
                  - 2.0/(beta1*aG())*r*r/(G*m*m)*(lStar()+eDotIon()-lDeut(beta1)));
            r += dtprime * dr;
        }

    }
    else if (mdottime < dt)
    {
        int mdotfac = ceil(dt/mdottime);
        double mdotfacr = mdotfac;
        double dtprime = dt/mdotfacr;

        for(int mdotloop = 0; mdotloop < mdotfac; mdotloop++)
        {
            beta1=beta(m);
            dr = (2.0*mdot/m*r*(FK/(aG()*beta1)+1.0-1.0/(aG()*beta1))
                  + beta1/m * dlogBeta_dlogM(beta1) * mdot * r / beta1
                  - 2.0/(beta1*aG())*r*r/(G*m*m)*(lStar()+eDotIon()-lDeut(beta1)));
            r += dtprime * dr;
        }
    }
    else
    {
        beta1=beta(m);
        dr = (2.0*mdot/m*r*(FK/(aG()*beta1)+1.0-1.0/(aG()*beta1))
              + beta1/m * dlogBeta_dlogM(beta1) * mdot * r / beta1
              - 2.0/(beta1*aG())*r*r/(G*m*m)*(lStar()+eDotIon()-lDeut(beta1)));
        r += dt * dr;
    }
    if (r < 0.0e0)
    {
      r = 0.2*6.96e10; // worst case and we do get a neg radius. reset it
      std::cout<< "Stellar evolution radius update: Found negative radius. Resetting to 0.2 R_sun" << std::endl;
    }
  } // burnState != ZAMS

  // Update the burning state and things associated with it
  switch (burnState)
  {
      case Uninitialized: {break;}

      case None:
      {
        // No burning yet, so check for the onset of D burning in the core
        n = nInit(mdot);
        if (Tc(m) > TDEUT)
        {
          burnState = VariableCoreDeuterium;
          n = 1.5; // Star becomes convective
        }
        break;
      }

      case VariableCoreDeuterium:
      {
        // We are burning deuterium at a variable rate to keep the core
        // temperature constant. Check to make sure we haven't exhausted
        // our supply of D, in which case we change to steady core burning.
        mdeut -= lDeut()*dt/PSID;
        if (mdeut <= mdot*dt)
        {
          burnState = SteadyCoreDeuterium;
          mdeut = 0.0;
        }
        break;
      }

      case SteadyCoreDeuterium:
      {
        // We are burning deuterium in the core at the rate it comes in. Check
        // to see if a radiative barrier forms, which stops convection, shuts
        // off core deuterium burning, and starts shell burning.
        mdeut = 0.0;
        if (lDeut() <= FRAD*lZAMS())
        {
          burnState = ShellDeuterium;
          n = 3.0;
          r *= SHELLFAC;
        }
        break;
      }

      case ShellDeuterium:
      {
        // We are burning deuterium in a shell. Check if the radius has
        // decreased to the ZAMS radius, in which case we stay on the ZAMS from now on.
        mdeut = 0.0;
        if (r <= rZAMS())
        {
          burnState = ZAMS;
          r = rZAMS();
        }
        break;
      }

      case ZAMS:
      {
        mdeut = 0.0;
        break;
      }
  } // end switch case burnState

} //end updateState

#undef PI
#undef G
#undef NAVOG
#undef ERGEV
#undef FACC
#undef FK
#undef FRAD
#undef SHELLFAC
#undef THAY
#undef TDEUT
#undef PSIION
#undef PSID
#undef MSUN
#undef LSUN
#undef YRSEC

/**********************/
/* Main sequence fits */
/**********************/

// Parameters for the main sequence luminosity and radius fitting formulae
// from Tout et al (1996)
#define ALPHA    0.39704170
#define BETA     8.52762600
#define GAMMA    0.00025546
#define DELTA    5.43288900
#define EPSILON  5.56357900
#define ZETA     0.78866060
#define ETA      0.00586685 
#define THETA    1.71535900
#define IOTA     6.59778800
#define KAPPA   10.08855000
#define LAMBDA   1.01249500
#define MU       0.07490166
#define NU       0.01077422
#define XI       3.08223400
#define UPSILON 17.84778000
#define PI       0.00022582

// Conversions from solar to CGS units
#define MSUN     1.99e33
#define LSUN     3.90e33
#define RSUN     6.96e10

inline
double
StarParticle::lZAMS()
{
  double msol = m/MSUN;
  double lsol = (ALPHA*pow(msol,5.5) + BETA*pow(msol,11)) /
    (GAMMA+pow(msol,3)+DELTA*pow(msol,5)+EPSILON*pow(msol,7)+
     ZETA*pow(msol,8)+ETA*pow(msol,9.5));
  return ( lsol*LSUN );
}

inline
double
StarParticle::rZAMS()
{
  double msol = m/MSUN;
  double rsol = (THETA*pow(msol,2.5)+IOTA*pow(msol,6.5)+KAPPA*pow(msol,11)+
	       LAMBDA*pow(msol,19)+MU*pow(msol,19.5)) /
    (NU+XI*pow(msol,2)+UPSILON*pow(msol,8.5)+pow(msol,18.5)+
     PI*pow(msol,19.5));
  return ( rsol*RSUN );
}

#undef ALPHA
#undef BETA
#undef GAMMA
#undef DELTA
#undef EPSILON
#undef ZETA
#undef ETA
#undef THETA
#undef IOTA
#undef KAPPA
#undef LAMBDA
#undef MU
#undef NU
#undef XI
#undef UPSILON
#undef PI
#undef MSUN
#undef LSUN
#undef RSUN

/***********************/
/* Friend IO operators */
/***********************/

#define IOPREC 20

std::ostream &operator<<(std::ostream &os, const StarParticle& sp)
{
  // Set precision to IOPREC decimal places
  int oldprec = os.precision(IOPREC);
  // Output order is mass, position, momentum
  os << sp.m << " ";
  int burnVal = sp.burnState;
  os << sp.r << " " << sp.mdeut
     << " " << sp.n << " " << sp.mdot << " "
     << burnVal;
 
  // Restore old precision
  os.precision(oldprec);

  return(os);
}

std::istream &operator>>(std::istream &is, StarParticle& sp)
{
  // Set precision to IOPREC decimal places
  int oldprec = is.precision(IOPREC);

  // Input order is mass, position, momentum
  is >> sp.m;
  int burnVal = sp.burnState;
  is >> sp.r >> sp.mdeut >> sp.n
     >> sp.mdot >> burnVal;
  sp.burnState = (StarParticle::burningState) burnVal;

  // Restore old precision
  is.precision(oldprec);

  return(is);
}
