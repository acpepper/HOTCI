from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize



# Define the extension.
# (The first field defines the extension's name)
extensions = [
    Extension(
        "eos",
        sources = [             # our Cython source
            "eos.pyx",
            "EOS_functions.cc"
        ],                
        include_dirs = [        # Path to Python.h
            '/share/apps/bio3user/anaconda3/include/python3.7m/'
        ],                        
        language="C++",         # generate C++ code
    ),
]


# because we call setup(), arguments are needed to define how
# the extension is created when we run this file in Python
# 
# Cython stuff:
# - The cythonize() function will generate and compile C++ sources
#   (defined in the 'language' field of 'Extension') from the
#   .pyx file (also defined in 'Extension')
setup(
    name = "EOS",
    ext_modules = cythonize(extensions),
)
