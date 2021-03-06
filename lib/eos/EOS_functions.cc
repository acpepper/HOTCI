//SJL: 10/15
/*
Function for the class structure for an EOS for the HERCULES code
*/


#include "EOS_class.h"
//include read/write files
#include <fstream>



////////////////////////////////////////////////////////////
/*Function to read in an EOS file and process the data 
to the format used in HERCULES */
void EOS::read_EOSfile(std::string file_name) 
{

  //Define the EOS file name
  fname=file_name;

  //Read in the file
  //EDIT
  std::ifstream EOSfile(fname.c_str());

  //EDIT (4/4/21 by A.C. Pepper)
  // must throw exception for file read errors
  if (!EOSfile) {
    throw std::runtime_error("File not found: "+fname);
  }
  
  //ignore the header and read the data type
  std::string temp("temp");
  std::string delimiter("#");

  while (delimiter.compare(temp.substr(0,1)) != 0){
    EOSfile >> temp;
  }
  EOSfile >> EOS_type;
  getline(EOSfile, temp);
  getline(EOSfile, temp);

  
  //Process depending on read type
  if (EOS_type==0 || EOS_type==1){
    //read in the columns of data
    double rhoread, pread, Tread, Sread;
    while (EOSfile >> rhoread >> pread >> Tread >> Sread)
      {
	rho.push_back(rhoread);
	p.push_back(pread);
	T.push_back(Tread);
	S.push_back(Sread);
      }
    
    //Find length of arrays
    Ndata=rho.size();

  }
  else if (EOS_type==2){

    //read in the columns of data
    double rhoread, pread, Tread;
    while (EOSfile >> rhoread >> pread >> Tread)
      {
	rho.push_back(rhoread);
	p.push_back(pread);
	T.push_back(Tread);
      }
    
    //Find length of arrays
    Ndata=rho.size();
    
    //for Hubbard need to convert to SI and log
    std::vector<double> temp_vec;
    
    temp_vec=VecDoub::VecExponent(10.0, rho);
    rho=VecDoub::ScalMultiply(1.0E3,temp_vec);
    
    temp_vec=VecDoub::VecExponent(10.0, p);
    p=VecDoub::ScalMultiply(0.1,temp_vec);
    
    temp_vec=VecDoub::VecExponent(10.0, T);
    T=temp_vec;
    
  }
  else {
    std::cerr << "Unrecognised EOS read type" << std::endl;
    std::exit(1);
  }
  
  //Close the file
  EOSfile.close();
  
}


///////////////////////////////////////////////////////////////
//Function to write the name of the EOS file
void EOS::write_binary(std::string file_name)
{
  std::ofstream file;
  file.open(file_name.c_str(), std::ofstream::out);
  
  //Set standard string length in characters
  int string_length=200;
  char temp_char[string_length];

  std::string temp_string=fname;
  temp_string.insert(temp_string.end(), string_length - temp_string.size(), ' ');
  std::strcpy(temp_char, temp_string.c_str());
  file.write (temp_char, string_length*sizeof(char));

  file.write ((char*)&EOS_type, sizeof(int));
}



///////////////////////////////////////////////////////////////
//Function to read EOS from binary
void EOS::read_binary(std::string file_name)
{
  std::ifstream file;
  file.open(file_name.c_str(), std::ifstream::out);
    
  //set string length
  int string_length = 200;
  char temp_char[string_length+1];
  temp_char[string_length]=0; //this stops the string from reading beyond char

  file.read(temp_char, string_length*sizeof(char));
  std::string temp_string=temp_char;
  temp_string.erase(remove_if(temp_string.begin(), temp_string.end(), isspace), temp_string.end()); //remove white space
  fname=temp_string;
  /*
  Ndata=1;
  p.resize(Ndata);
  p[0]=0.0;
  rho.resize(Ndata);
  rho[0]=0.0;
  T.resize(Ndata);
  T[0]=0.0;
  */

  file.read ((char*)&EOS_type, sizeof(int));
  
}

///////////////////////////////////////////////////////////////
//Function to calculate the pressure given the temperature
double EOS::calc_p_from_T(double temp)
{
  double density;
  std::vector<double>::iterator pos;
  int ind;
  
  //Search the temperature vector to find the nearest temperature
  pos=std::lower_bound (T.begin(), T.end(), temp);
  pos=pos-1;
  ind=std::distance(T.begin(), pos);

  //if EOS_type is log then do a mixed log/linear interpolation
  if (EOS_type==1){
    //if low pressure do a log log interpolation
    if (p[ind]<1e3) {
      density=pow(10.0, (log10(p[ind])+(log10(temp)-log10(T[ind]))*(log10(p[ind+1])-log10(p[ind]))/(log10(T[ind+1]) - log10(T[ind]))));
      return density;
    }
    //else do linear interpolation
    else {
      density=p[ind]+(temp-T[ind])*(p[ind+1]-p[ind])/(T[ind+1] - T[ind]);
      return density;
    }
  }

  else {
    //linearly interpolate to find the density
    density=p[ind]+(temp-T[ind])*(p[ind+1]-p[ind])/(T[ind+1] - T[ind]);
    return density;
  }

  return density;
}

///////////////////////////////////////////////////////////////
//Function to calculate the density given the pressure
double EOS::calc_rho(double press)
{
  double density;				
  std::vector<double>::iterator pos;
  int ind;
  
  //Search the pressure vector to find the nearest pressure
  pos=std::lower_bound (p.begin(), p.end(), press);
  pos=pos-1;
  ind=std::distance(p.begin(), pos);

  //if EOS_type is log then do a mixed log/linear interpolation
  if (EOS_type==1){
    //if low density do a log log interpolation
    if (rho[ind]<3.0E3) {
      density=pow(10.0, (log10(rho[ind])+(log10(press)-log10(p[ind]))*(log10(rho[ind+1])-log10(rho[ind]))/(log10(p[ind+1]) - log10(p[ind]))));
      return density;
    }
    //else do linear interpolation
    else {
      density=rho[ind]+(press-p[ind])*(rho[ind+1]-rho[ind])/(p[ind+1] - p[ind]);
      return density;
    }
  }

  else {
    //linearly interpolate to find the density
    density=rho[ind]+(press-p[ind])*(rho[ind+1]-rho[ind])/(p[ind+1] - p[ind]);
    return density;
  }

  return density;
}

///////////////////////////////////////////////////////////////
//Function to calculate the density given the temperature
double EOS::calc_rho_from_T(double temp)
{
  double density;
  std::vector<double>::iterator pos;
  int ind;
  
  //Search the temperature vector to find the nearest temperature
  pos=std::lower_bound (T.begin(), T.end(), temp);
  pos=pos-1;
  ind=std::distance(T.begin(), pos);

  //if EOS_type is log then do a mixed log/linear interpolation
  if (EOS_type==1){
    //if low density do a log log interpolation
    if (rho[ind]<3.0E3) {
      density=pow(10.0, (log10(rho[ind])+(log10(temp)-log10(T[ind]))*(log10(rho[ind+1])-log10(rho[ind]))/(log10(T[ind+1]) - log10(T[ind]))));
      return density;
    }
    //else do linear interpolation
    else {
      density=rho[ind]+(temp-T[ind])*(rho[ind+1]-rho[ind])/(T[ind+1] - T[ind]);
      return density;
    }
  }

  else {
    //linearly interpolate to find the density
    density=rho[ind]+(temp-T[ind])*(rho[ind+1]-rho[ind])/(T[ind+1] - T[ind]);
    return density;
  }

  return density;
}

///////////////////////////////////////////////////////////////
//Function to calculate the temperature given the pressure
double EOS::calc_T(double press)
{
  double temp;				
  std::vector<double>::iterator pos;
  int ind;
  
  //Search the pressure vector to find the nearest pressure
  pos=std::lower_bound (p.begin(), p.end(), press);
  pos=pos-1;
  ind=std::distance(p.begin(), pos);

  //if EOS_type is log then do a mixed log/linear interpolation
  if (EOS_type==1){
    //if low temperature do a log log interpolation
    //
    // ACP NOTE: This cut-off temperature is approximately room temp
    if (T[ind]<3.0E2) {
      temp = pow(10.0, ( log10(T[ind])
			 + (log10(press) - log10(p[ind]))
			   *(log10(T[ind+1])-log10(T[ind]))
			   /(log10(p[ind+1]) - log10(p[ind]))
			)
		 );
      return temp;
    } 
    //else do linear interpolation
    else {
      temp = T[ind]+( press - p[ind])*(T[ind+1] - T[ind] )/(p[ind+1] - p[ind]);
      return temp;
    }
  }

  else {
    //linearly interpolate to find the density
    temp = T[ind]+( press - p[ind])*(T[ind+1] - T[ind] )/(p[ind+1] - p[ind]);
    return temp;
  }

  return temp;
}

