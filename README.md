# HOTCI

**H**ERCULES **O**utput **T**o **C**TH **I**nput

## Introduction
<details>
    <summary>
HOTCI is a tool designed to aid in the study giant impacts (or impacts between planet-sized bodies) by initializing a rotating body in the Eulerian shock physics code, CTH. This is achieved by converting the bodies generated by Simon Lock's HERCULES into a collection of polygons that can then be used as initial conditions for CTH.
</summary>

### What is HERCULES?
HERCULES (Highly Eccentric Rotating Concentric U [Potential] Layers Equilibrium Structure) is a program written by Simon Lock to solve for the equilibrium structure of a self-gravitating fluid. The algorithm used by HERCULES was originally found by Hubbard (2012, 2013) in order to study Jupiter. The algorithm has since been extended by Kong et al. (2013) and Hubbard et al. (2014) to accomodate bodies with large rotational distortion. HERCULES is an open-source manifestation of this algorithm, written in C++.

### What is CTH?
CTH is a large shock-physics code that has been over-seen by many employees of Sandia National Laboratory. It is fundamentally an Eulerian method though at each time-step it solves the Lagrangian equations and remaps the solution to the Eulerian grid via a van Leer scheme that is accurate to second order (van Leer, 1977; McGlaun, 1982). CTH implements two major features that make it popular for simulating giant impacts. Firstly, it implements self-gravity, which is critical for studying any process in the large length regime. Secondly, it implements adaptive mesh refinement, wherein the Eulerian mesh is recursively subdivided to increase resolution locally, this saves computational resources when large regions of the simulation domain are occupied by the void of space.

### Flow of Data in HOTCI
HOTCI is a small script written in Python3 and C++ that can be used to simulate rapidly rotating bodies in the shock physics code CTH. This is accomplished through the following multistep process. First, a rotating body is generated using HERCULES. The output of HERCULES (which is a custom binary format) is read and analyzed by HOTCI. During this step HOTCI attempts to match data-points in HERCULES directly to points on the surface of a polygon in CTH. However, HOTCI might unresolve the body if the resolution in HERCULES is too high. since HERCULES performs without calculating temperature, HOTCI may also calculate the temperature if the user desires. Next, the data that defines the body is converted into a string format that can be read by CTH. A surrogate CTH input file must supplied to HOTCI (the surrogate CTH input file may be any valid CTH input file). HOTCI then searches this input file for the initial conditions section and overwrites it with the rotating body's string representation. Finally, HOTCI creates a new file such that the surrogate input file is undisturbed.

</details>

## Graphical Overview
<details>
    <summary>Click to view image</summary>
<img src=images/HOTCI_graphic.png width=600 height=600>

The image above illustrates how HOTCI work. Only the topmost pictures, **A** and **E**, contain data generated by HERCULES and CTH, respectively; the rest of the images have been rendered solely for illustrative purposes. 
* **A**: In this step HOTCI reads a HERCULES output file and converts it into a CTH input file.
* **B**: CTH reads the input file and processes the body one layer at a time. Each layer is homogeneous in density, pressure, and temperature.
* **C**: The layer is incorporated into the Eulerian mesh. In this step CTH gives each cell of the mesh a velocity, volume fraction for each material, and any necessary thermodynamic variables.
* **D**: This panel is included to illustratculties one has when representing a spherical object in a rectangular grid, the resolution is exaggerated.
* **E**: A cross section of an example body in CTH.
    
</details>


## List of included files
<details>
    <summary>Click to view list</summary>

### HOTCI.py
A Python file containing HOTCI’s main function. There are several variables defined at the top of HOTCI.py that are intended to be edited by the user. These variables determine HOTCI’s reading and writing behavior. They are:
* HERCULES_OUT_DIR: A string containing the directory where HERCULES dumps its output files.
* HERCULES_OUT_FNAMES: A list of strings containing the names of the HERCULES output files that will be read. The user is able to include any number of file names however the length of the HERCULES_OUT_FNAMES list must be equal to that of … In order for HOTCI to run properly, the user is responsibility for ensuring that this condition is met.
* CTH_IN_DIR: A string containing the directory where HOTCI searches for and saves all CTH input files.
* CTH_BASE_FNAME: HOTCI requires a partial CTH input file to work from, this is a sting containing the name of such a file.
* CTH_IN_FNAME: A string containing the name of the CTH input file that HOTCI generates.
* MATERIAL_FNAMES: A list of strings containing the locations of 

### lib/HERCULES_structures.py
A Python file containing classes for analyzing the binary output of HERCULES.

### lib/eosFunctions.py
A Python file containing a collection of functions for analyzing tabular eos files.

### lib/eostable.py
A Python file containing a collection of classes to represent tabular eos files.

### lib/eos/EOS_class.h
A C++ file containing the EOS class definition. This class is used to calculate the temperature of the HERCULES layers.

### lib/eos/EOS_functions.cc
A C++ file containing function definitions for the EOS class.

### lib/eos/setup.py
A Python file that determines how the C++ files in the eos directory will be compiled into a Python library.

### lib/eos/vector_operations.h
A C++ header containing the VecDoub class definition.

### lib/eos/vector_operations.cc
A C++ file containing the function definitions for the VecDoub class.

</details>

## Compiling and Running
<details>
    <summary>All compiling instructions are for a Linux operating system. However, if this is not your operating system of choice, the instructions should be straightforward to translate since HOTCI is very lightweight.</summary>

### Dependencies
HOTCI requires CTH and HERCULES to be installed and running. To work properly CTH should implement self-gravity, which is included in the latest version. HOTCI requires a Python 3.0 interpreter or later.
The only element of HOTCI which must be compiled is the eos.so library, which is a wrapping of a C++ library that was written for HERCULES. Thus a C++ compiler will also be needed, the default is g++ but this may be changed to accomodate your prefered C++ compiler. There are many ways to create a python library by wrapping C++ source code. The method detailed here used the distutils and Cython libraries. These libraries are included in many of the most popular python distributions, including Anaconda and Sage, so they will likely be installed with the Python 3.0 interpreter. To check if the distutils and Cython libraries were included in your Python distribution run the following from the command line.
```
$ python
>>> from distutils.core import setup
>>> from Cython.Build import cythonize
```
If this does not produce an error than you are ready to start compiling HOTCI.

### Wrapping eos.so
From the HOTCI directory, enter the eos subdirectory and run setup.py in “build” mode.
```
HOTCI$ cd eos
eos$ python setup.py build
```
This should create a new file called eos.so in the build subdirectory entitled lib.[your OS]. Copy the newly created eos.so file into the HOTCI parent directory.
```
eos$ cp build/lib.linux-x86_64-2.7/eos.so ../
```
If the eos.so file was not created but distutils and Cython were properly installed, then the issue probably occurred when trying to link Python.h. To fix this error open the setup.py file and modify the include_dirs list to contain the directory where your Python.h file is located. On my machine this is the /opt/local/include/ directory.

3.3 Running
    Once the necessary libraries have been downloaded and compiled, the HOTCI.py file must be modified to match your work environment. This is accomplished by opening HOTCI.py file. Lines 17-38 contain all the variables a user might want to modify. They appear as follows.

HERCULES_OUT_DIR = "../Output/"
HERCULES_OUT_FNAMES = ["M96L1.5_L1.48725_N200_Nm800_k12_f020_p10000_l1_0_1.5_fi\
nal", "M12omega2e-4_L1.983_N100_Nm400_k12_f020_p10000_l1_0_1.5_final"]

CTH_IN_DIR = "CTH_in/"
CTH_BASE_FNAME = "CTH_ANEOS_test_impact.in"
CTH_IN_FNAME = "test_M91_m12_L1.5.in"

MATERIAL_FNAMES = ['../EOS_files/HERCULES_EOS_forsterite_S3.20c_log.txt', '../EOS_files/HERCULES_EOS_Iron_T4kK_P135GPa_Rho7-15.txt']
# PD_FLAG key:
# 1: pressure and temperature
# 2: density and temperature
# 3: pressure and density
PDT_FLG = 2




# NOTE: These are in CGS units
CENTERS = [[0, 0, 0], [7.056e8, 7.056e8, 0]]
VELOCITIES = [[0, 0, 0], [-8.795e5, 0, 0]]

# CTH limits the number of vertices in its input files so when the HERCULES
# resolution is too fine the shape cannot be transferred in a 1-to-1 fashion.
# When this occurs, we unresolve the HERCULES structure following a cubic
# spline interpolation of the original points. The new number of points is
# defined by NUM_NEW_MU.
NUM_NEW_MU = 600

INDENT = " "

Each variable’s usage is detailed in section 2.1. It is particularly important that the user updates their file names and directories.
    
    After the variables have been updated HOTCI can be run by simply typing the following into the command line.

HOTCI$ python HOTCI.py

</details>
