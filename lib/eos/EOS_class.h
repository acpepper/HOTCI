
//SJL: 10/15
/*
Class structure for a planet for the HERCULES code
*/

//Added for compilation on oddysey
#include <cstdlib>
#include <sstream>

//include io
#include <iostream>
#include <fstream>
//cmath include maths functions
#include <cmath>
//Vector
#include <vector>
//Allows some numeric functions on vectors
#include <numeric>
//Strings
#include <string>

//Legendre polynomials
#include <boost/math/special_functions/legendre.hpp>

//Header for my self defined vector namespaces
#include "vector_operations.h"



class EOS
{
public:
  //Name of input file
  std::string fname;
  //vectors for the pressure, density, temperature in EOS
  std::vector<double> p, rho, T, S;
  //length of vectors
  int Ndata;

  //type of EOS (0=standard linear, 1=standard log, 2=Hubbard) 
  int EOS_type;


  //Functions
  //EOS();
    
  void read_EOSfile(std::string file_name);

  void write_binary(std::string file_name);
  void read_binary(std::string file_name);

  double calc_p_from_T(double temp);
  double calc_rho(double press);
  double calc_rho_from_T(double temp);
  double calc_T(double press);
};
