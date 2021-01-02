//SJL: 10/15 
/*
Various functions to do various operations of vectors of doubles
 */
#include <vector>
#include <algorithm> // for transform
#include <functional> // for plus

//Allows some numeric functions on vectors
#include <numeric>

//cmath include maths functions
#include <cmath>

//Header for my self defined vector namespaces
#include "vector_operations.h"

#include <iostream>

////////////////////////////////////////////////////
//Function to multiply two vectors of doubles
std::vector<double> VecDoub::VecMultiply(std::vector<double>& v1, std::vector<double>& v2)
{
std::vector<double> answer;
answer.reserve(v1.size());
std::transform( v1.begin(), v1.end(),
		  v2.begin(), std::back_inserter(answer), std::multiplies<double>());

  return answer;
}

////////////////////////////////////////////////////
//Function to divide two vectors of doubles
std::vector<double> VecDoub::VecDivide(std::vector<double>& v1, std::vector<double>& v2)
{
std::vector<double> answer;
answer.reserve(v1.size());
std::transform( v1.begin(), v1.end(),
		  v2.begin(), std::back_inserter(answer), std::divides<double>());

  return answer;
}


////////////////////////////////////////////////////
//Function to multiply a scalar and a  vector of doubles
std::vector<double> VecDoub::ScalMultiply(double c1, std::vector<double>& v1)
{
std::vector<double> answer;
answer.resize(v1.size());
std::transform(v1.begin(), v1.end(), answer.begin(), std::bind2nd(std::multiplies<double>(),c1));

  return answer;
}


////////////////////////////////////////////////////
//Function to take an array of doubles to the exponent
std::vector<double> VecDoub::VecExp(std::vector<double>& v1)
{
std::vector<double> answer;
answer.resize(v1.size());
 std::transform(v1.begin(), v1.end(), answer.begin(), exp);

  return answer;
}


////////////////////////////////////////////////////
//Function to sum over an array of doubles
double VecDoub::VecSum(std::vector<double>& v1)
{
return std::accumulate(v1.begin(), v1.end(), 0.0);
}


////////////////////////////////////////////////////
//Function to take a vector of double to a power
std::vector<double> VecDoub::VecPow(std::vector<double>& v1, double c1)
{
std::vector<double> answer;
answer.resize(v1.size());
for (int i=0; i<v1.size(); ++i){
answer[i]=pow(v1[i],c1);
}

  return answer;
}

//////////////////////////////////////////////////
//Function to sum two vectors of doubles
std::vector<double> VecDoub::VecAdd(std::vector<double>& v1, std::vector<double>& v2)
{
  std::vector<double> answer;
  answer.reserve(v1.size());
  std::transform( v1.begin(), v1.end(),
		  v2.begin(), std::back_inserter(answer), std::plus<double>());

  return answer;

}


//////////////////////////////////////////////////
//Function to subtract the second from the first of  two vectors of doubles
std::vector<double> VecDoub::VecSubtract(std::vector<double>& v1, std::vector<double>& v2)
{
  std::vector<double> answer;
  answer.reserve(v1.size());
  std::transform( v1.begin(), v1.end(),
		  v2.begin(), std::back_inserter(answer), std::minus<double>());

  return answer;

}

////////////////////////////////////////////////////
//Function to take a vector of doubles and put it to the exponent of a constant
std::vector<double> VecDoub::VecExponent(double c1, std::vector<double>& v1)
{
  std::vector<double> answer;
  answer.resize(v1.size());
  for (int i=0; i<v1.size(); ++i){
    answer[i]=pow(c1, v1[i]);
  }
  return answer;
}

////////////////////////////////////////////////////
//Function to multiply a scalar and a  vector of doubles
std::vector<double> VecDoub::ScalAdd(double c1, std::vector<double>& v1)
{
std::vector<double> answer;
answer.resize(v1.size());
std::transform(v1.begin(), v1.end(), answer.begin(), std::bind2nd(std::plus<double>(),c1));

  return answer;
}

////////////////////////////////////////////////////
//Function to calculate the gradient of a vector y with respect to x
//This procedure is second order accuracy in dx
std::vector<double> VecDoub::VecGrad2(std::vector<double>& y, std::vector<double>& x)
{
  std::vector<double> dydx;
  int Ndata;
  Ndata=y.size();
  dydx.resize(Ndata);

  //Find the first term by forward difference
  dydx[0]=(pow((x[2]-x[0]),2.0)*(y[1]-y[0])+pow((x[1]-x[0]),2.0)*(y[0]-y[2]))
    /((x[2]-x[0])*(x[1]-x[0])*(x[2]-x[1]));

  //find the last term by backwards difference
  dydx[Ndata-1]=(pow((x[Ndata-3]-x[Ndata-1]),2.0)*(y[Ndata-2]-y[Ndata-1])+
		 pow((x[Ndata-2]-x[Ndata-1]),2.0)*(y[Ndata-1]-y[Ndata-3]))
    /((x[Ndata-3]-x[Ndata-1])*(x[Ndata-2]-x[Ndata-1])*(x[Ndata-3]-x[Ndata-2]));

  //Find all the intermediate terms
  for (int i=1; i<Ndata-1; ++i){
    dydx[i]=(pow((x[i-1]-x[i]),2.0)*(y[i+1]-y[i])+pow((x[i+1]-x[i]),2.0)*(y[i]-y[i-1]))
      /((x[i-1]-x[i])*(x[i+1]-x[i])*(x[i-1]-x[i+1]));
  }

  return dydx;
}



