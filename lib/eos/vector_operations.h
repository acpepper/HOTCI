//SJL 10/15
/*
Header to the namespace for vector operatiions
 */


#ifndef VECNAMESPACES_H
#define VECNAMESPACES_H

//namespace for operating on vectors of doubles
namespace VecDoub{
  
  //see vector_operators.cc for details of functions
  std::vector<double> VecMultiply(std::vector<double>& v1, std::vector<double>& v2);
  std::vector<double> VecDivide(std::vector<double>& v1, std::vector<double>& v2);
  std::vector<double> ScalMultiply(double c1, std::vector<double>& v1);
  std::vector<double> VecExp(std::vector<double>& v1);
  double VecSum(std::vector<double>& v1);
  std::vector<double> VecPow(std::vector<double>& v1, double c1);
  std::vector<double> VecAdd(std::vector<double>& v1, std::vector<double>& v2);
  std::vector<double> VecSubtract(std::vector<double>& v1, std::vector<double>& v2);
  std::vector<double> VecExponent(double c1, std::vector<double>& v1);
  std::vector<double> ScalAdd(double c1, std::vector<double>& v1);

  std::vector<double> VecGrad2(std::vector<double>& y, std::vector<double>& x);
 
};

#endif
