#ifndef _SinkStellarEvolution_h_
#define _SinkStellarEvolution_h_

#include <iostream>

/* A class for holding particles that are stars and embody a protostellar model */
/* adopted from Offner et al. (2009, ApJ 703, 131) */

class StarParticle
{

  friend std::ostream &operator<<(std::ostream &os, const StarParticle& sp);
  friend std::istream &operator>>(std::istream &is, StarParticle& sp);

public:

  /* Constructor */
  StarParticle(double mass, double mdot, double mdeut, double r, int burnState, double dt);

  /* Destructor */
  ~StarParticle();

  /* Routine to update the star particle's internal state one time step */
  void updateState();

  /* Routine to return the luminosity of the protostar and its surrounding inner disk */
  double luminosity();
  double lStar(); /* Luminosity of the protostar, without the disk contribution */
  double lDisk(); /* Energy radiated away from the inner disk */

  /* Data contained in the model */
  double m;              /* Mass */
  double mdot;           /* Current mass accretion rate */
  double mdeut;          /* Mass of gas that still contains deuterium */
  double r;              /* Radius */
  double n;              /* Polytropic index */
  double dt;             /* Current time step */

  enum burningState { Uninitialized, None, VariableCoreDeuterium,
          SteadyCoreDeuterium, ShellDeuterium, ZAMS } burnState;

private:

  /* Routines that are part of the protostellar model */

  /* Initialization */
  double radInit(double mdot);
  double nInit(double mdot);

  /* Utility functions for polytropic models */
  double aG(); // For a polytrope, the gravitational energy is aG GM^2/R
  double rhoc(double mass); // Central density
  double Pc(double mass); // Central pressure
  double Tc(double mass, double rhoc=-1.0, double Pc=-1.0); // Central temperature
  double betac(double mass, double rhoc=-1.0, double Pc=-1.0, double Tc=-1.0); // Central beta
  double beta(double mass, double rhoc = -1.0, double Pc = -1.0); // Mean beta
  double dlogBetaOverBetac_dlogM(double beta=-1); //d(log(beta/beta_c))/d(logM)
  double dlogBeta_dlogM(double beta=-1); // d(log beta) / d(log M)

  /* Main sequence fits */
  double lZAMS(); /* Luminosity of a ZAMS star of this mass */
  double rZAMS(); /* Radius of a ZAMS star of this mass */

  /* Model functions */
  double lAcc(); /* Accretion luminosity */
  double lDeut(double beta=-1.0); /* Luminosity from deuterium burning */
  double eDotIon(); /* Energy needed to ionize the incoming material */
  double dlogR_dlogM(double beta=-1); /* d(log R) / d(log M) */

}; // end class

#endif // _SinkStellarEvolution_h_
