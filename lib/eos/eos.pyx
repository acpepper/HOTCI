# IMPORTANT NOTE: the following lines must be
#                 included in the 'first comment block' (which
#                 I assume means before any code)

# distutils: language = c++
# distutils: sources = [EOS_functions.cc, vector_operators.cc]

from libcpp.string cimport string
from libcpp.vector cimport vector


# The following block declares the C++ attributes and
# methods we plan to use in Cython and thus use to extend Python
# (see next block)
#
# NOTE: Even if a constructor is not defined in your C++ source code
#       (i.e. your using the default constructor) you must define one here
#      
# Cython stuff:
# - 'cdef' tells Cython the following code is to be translated into C.
#   The code inside a cdef block can't be imported by Python (because
#   it's basically C). You must also declare types when writing in a
#   cdef block.
# - 'except +' allows exeptions raised by the underlying C++
#   constructor to be handled by Python.
cdef extern from "EOS_class.h":
    cdef cppclass EOS:
        EOS() except +
        string fname
        vector[double] p, rho, T, S
        int Ndata
        int EOS_type
        void read_EOSfile(string file_name)
        void write_binary(string file_name)
        void read_binary(string file_name)
        double calc_p_from_T(double temp)
        double calc_rho(double press)
        double calc_rho_from_T(double temp)
        double calc_T(double press)


# This declares a new class whose sole purpose is
# to pass arguments to the C++ version of the EOS class
#
# NOTE: the Python class is nameed 'PyEOS' while the
#       C++ class is simply 'EOS' this is kinda obvious
#       but important to remember nonetheless.
# NOTE: we're also assuming the C++ class implements a nullary constructor
#       in our definition of '__cinit__()'
#       if this is not the case, '__cinit__()' needs to be modified and
#       '__dealloc__()' must be declared
cdef class PyEOS:
    cpdef EOS C_eos             # hold a C++ instance which we're wrapping
    cdef public string fname
    cdef public vector[double] p, rho, T, S
    cdef public int Ndata
    cdef public int EOS_type

    def __cinit__(self):
        self.C_eos = EOS()

    def update_attribs(self):
        self.fname = self.C_eos.fname
        self.p = self.C_eos.p
        self.rho = self.C_eos.rho
        self.T = self.C_eos.T
        self.S = self.C_eos.S
        self.Ndata = self.C_eos.Ndata
        self.EOS_type = self.C_eos.EOS_type

    def read_EOSfile(self, file_name):
        self.C_eos.read_EOSfile(file_name)
        self.update_attribs()
	
    def write_binary(self, file_name):
        self.C_eos.write_binary(file_name)
	
    def read_binary(self, file_name):
        self.C_eos.read_binary(file_name)
        self.update_attribs()
	
    def calc_p_from_T(self, temp):
        return self.C_eos.calc_p_from_T(temp)

    def calc_rho(self, press):
        return self.C_eos.calc_rho(press)

    def calc_rho_from_T(self, temp):
        return self.C_eos.calc_rho_from_T(temp)

    def calc_T(self, press):
        return self.C_eos.calc_T(press)
