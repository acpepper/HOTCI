# Copyright Andrew Pepper, 2018
#
import sys
sys.path.append("/home/acpepper/HOTCI_stuff/HERCULES/HOTCI")

import eos
from HERCULES_structures import *
from scipy import interpolate
from scipy.optimize import fsolve
import numpy as np
import re
import sys
import matplotlib.pyplot as plt



# Causes the program to print extra information about the structure and
# make plots of it
DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt

CTH_IN_DIR = "CTH_in/"
CTH_BASE_FNAME = "impact_master.in"
CTH_IN_FNAME = "M1.05_m0.05_L2.7"

HERCULES_OUT_DIR = "../Output/"
HERCULES_OUT_FNAMES = ['M1.05L2.69_hiIronS_L2.66714_N200_Nm800_k12_f021_p10000_l1_0_1.5_final', 'M0.05L0_L0_N200_Nm800_k12_f021_p10000_l1_0_1.5_final']
# ['M1.05L2.69_hiIronS_L2.66714_N200_Nm800_k12_f021_p10000_l1_0_1.5_final', 'M0.05L0_L0_N200_Nm800_k12_f021_p10000_l1_0_1.5_final']
# ["M0.9L0_L0_N200_Nm800_k12_f021_p10000_l1_0_1.5_final", "M0.13L0_L0_N200_Nm800_k12_f021_p10000_l1_0_1.5_final"]
# ["M0.75L0_L0_N200_Nm800_k12_f021_p10000_l1_0_1.5_final", "M0.3L0_L0_N200_Nm800_k12_f021_p10000_l1_0_1.5_final"]
# ["M0.57L0_L0_N200_Nm800_k12_f021_p10000_l1_0_1.5_final", "M0.47L0_L0_N200_Nm800_k12_f021_p10000_l1_0_1.5_final"]

# [body1 mantle, body1 core, body2 mantle, body2 core, ...]
MATERIAL_FNAMES = [['../EOS_files/HERCULES_EOS_S3.03_Forsterite-ANEOS-SLVTv1.0G1.txt', '../EOS_files/HERCULES_EOS_S1.81_Fe85Si15-ANEOS-SLVTv0.2G1.txt'],
                   ['../EOS_files/HERCULES_EOS_S3.03_Forsterite-ANEOS-SLVTv1.0G1.txt', '../EOS_files/HERCULES_EOS_S1.8_Fe85Si15-ANEOS-SLVTv0.2G1.txt']]


# NOTE: These are in CGS units
DX = 1.19842e9 # cm
# 1.19842e9 # cm
# 8.4996e8 # cm
# 9.9483e8 # cm
# 1.20030e9 # cm

DY = -3.3316e8 # cm
# -3.3316e8 # cm
# 7.1020e8 # cm
# 3.0516e8 # cm
# 5.8834e8 # cm

V_IMP = 2.00002e6 # cm/s
# 2.00002e6 # cm/s
# 9.1995e5 # cm/s
# 1.13298e6 # cm/s
# 9.7017e5 # cm/s

# PD_FLAG key: 
# 1: pressure and temperature
# 2: density and temperature
# 3: pressure and density
PDT_FLG = 2

# The number of layers over which we smooth the temperature. This is done to
# prevent spurious vaporazation. We enforce that the last layer is T = 300K
I_SM = 0
# CTH limits the number of vertexes in its input files so when the HERCULES
# resolution is too fine the shape cannot be transfered in a 1-to-1 fashion.
# When this occurs, we unresolve the HERCULES structure following a cubic
# spline interpolation of the original points. The new number of points is
# defined by NUM_NEW_MU.
NUM_NEW_MU = 600

INDENT = " "



class Diatom_Data:
    def __init__(self):
        self.material = 0
        self.rho = 0
        self.T = 0
        self.p = 0
        self.xs = np.zeros(0)
        self.zs = np.zeros(0)



#           _._______.________._______.________.________._
#   _ _____| .___    . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#          |_._______.________._______.________.________._|
def new_mu(diatom_data, new_num_mu):
    # Iterate over the layers
    for i, layer in enumerate(diatom_data):
        # Get a cubic spline describing the shape of the layer
        spl = interpolate.splrep(layer.zs, layer.xs)
        
        # Generate new mu points
        new_zs = np.linspace(0,
                             layer.zs[-1], NUM_NEW_MU,
                             endpoint=True)
        
        new_xs = interpolate.splev(new_zs, spl)

        diatom_data[i].xs = new_xs
        diatom_data[i].zs = new_zs



#          _.________.________._______.________.________._
#   _ ____| .___     . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |_.________.________._______.________.________._|
def get_hercules_data(hero_fname):
    # Define blank planet and parameter classes
    params = HERCULES_parameters()
    planet = HERCULES_planet()

    # Read in HERCULES output
    with open(HERCULES_OUT_DIR+hero_fname, "rb") as f:
        params.read_binary(f)
        planet.read_binary(f)

    return params, planet



#          _.________.________._______.________.________._
#   _ ____| .___     . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |_.________.________._______.________.________._|
def get_diatom_data(params, planet, mat_fnames):
    diatom_data = np.zeros(planet.Nlayer, dtype = object)

    # initialize EOS files in order to convert pressure to temperature
    eos_files = np.asarray([])
    for i, mat_fname in enumerate(mat_fnames):
        eos_files = np.append(eos_files, eos.PyEOS())
        eos_files[i].read_EOSfile(mat_fname)

    if DEBUG:
        Xs = []
        Rho1s = []
        P1s = []
        T1s = []
        Rho2s = []
        P2s = []
        T2s = []

    # Get the material, density, and mu values describing each layer
    # (note that mu is proportional to the z-component of the points)
    for i, layer in enumerate(planet.layers):
        diatom_data[i] = Diatom_Data()
        diatom_data[i].material = planet.flag_material[i] + 1
        
        # make the outer layer extra cold to avoid spurrious vaporization
        if i <= I_SM - 1:
            if i == 0:
                diatom_data[i].T = .025
            else:
                # Use the EOS class to calculate T
                # and convert K -> ev
                diatom_data[i].T = eos_files[planet.flag_material[i]].calc_T(planet.press[i])/11604.0
                diatom_data[i].T = diatom_data[i].T/2 + diatom_data[i-1].T/2
                # Now get the density
                # Convert kg m^-3 -> g cm^-3
                diatom_data[i].rho = eos_files[planet.flag_material[i]].calc_rho_from_T(diatom_data[i].T*11604.0)/1000.0
                # Now get the pressure
                # Convert kg s^-2 m^-1 -> g s^-2 cm^-1
                diatom_data[i].p = eos_files[planet.flag_material[i]].calc_p_from_T(diatom_data[i].T*11604.0)*10.0
            if DEBUG:
                Rho1s.append(planet.real_rho[i]/1000.0)
                P1s.append(planet.press[i]*10.0)
                T1s.append(eos_files[planet.flag_material[i]].calc_T(planet.press[i])/11604.0)
                Rho2s.append(diatom_data[i].rho)
                P2s.append(diatom_data[i].p)
                T2s.append(diatom_data[i].T)
        else:
            # Convert kg m^-3 -> g cm^-3
            diatom_data[i].rho = planet.real_rho[i]/1000.0
            # Convert kg s^-2 m^-1 -> g s^-2 cm^-1
            diatom_data[i].p = planet.press[i]*10.0
            # Use the EOS class to calculate T
            # and convert K -> ev
            diatom_data[i].T = eos_files[planet.flag_material[i]].calc_T(planet.press[i])/11604.0

            if DEBUG:
                Rho1s.append(diatom_data[i].rho)
                P1s.append(diatom_data[i].p)
                T1s.append(diatom_data[i].T)
                Rho2s.append(diatom_data[i].rho)
                P2s.append(diatom_data[i].p)
                T2s.append(diatom_data[i].T)

        for j, (mu, xi) in enumerate(zip(layer.mu, layer.xi)):
            sin_theta_sqrd = 1.0 - mu**2.0
            if sin_theta_sqrd < 0.0:
                sin_theta_sqrd = 0.0
                
            diatom_data[i].xs = np.append(diatom_data[i].xs, xi*(layer.a*100.0)*np.sqrt(sin_theta_sqrd))
            diatom_data[i].zs = np.append(diatom_data[i].zs, xi*(layer.a*100.0)*mu)
            
        if DEBUG:
            Xs.append(diatom_data[i].xs[0])
            # Plot the mu points in each layer
            # plt.scatter(diatom_data[i].xs, diatom_data[i].zs)
            print('Layer {}:'.format(i))
            print('material: {:d}'.format(diatom_data[i].material))
            print('density: {:1.3e}'.format(diatom_data[i].rho))
            print('pressure: {:1.3e}'.format(diatom_data[i].p))
            print('temperature: {:1.3e}'.format(diatom_data[i].T))
            print('==========================')

    if DEBUG:
        plt.plot(Xs, Rho1s, 'r-')
        plt.plot(Xs, Rho2s, 'b--')
        plt.show()

    rhos = np.asarray([])
    Ts = []
    for data in diatom_data:
        rhos = np.append(rhos, data.rho)
        Ts.append(data.T)

    if DEBUG:
        plt.plot(np.arange(0, len(diatom_data), 1), rhos)
        plt.show()

        plt.plot(np.arange(0, len(diatom_data), 1), Ts)
        plt.show()

    return diatom_data



#          __._______.________._______.________.________._
#   _ ____|  .___    . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |__._______.________._______.________.________._|
def get_r2dp_insert(xs_outer, ys_outer, xs_inner, ys_inner, center, ax):
    r2dp_insert = INDENT + '  insert r2dp\n'
    indent = "   " + INDENT
    r2dp_insert += indent + 'ce1 '+str(center[0])+', '+str(center[1])+', '+str(center[2])+'\n'
    r2dp_insert += indent + 'ce2 '+str(center[0])+', '+str(center[1])+', '+str(center[2] + 1.0)+'\n'
    r2dp_insert += indent + 'ce3 '+str(center[0] + 1.0)+', '+str(center[1])+', '+str(center[2])+'\n'
    r2dp_insert += indent + 'twist = 1\n'
    r2dp_insert += indent + 'pitch = 0, 0\n'

    x2plot = []
    y2plot = []
    
    i = 1
    # First the outer edge along the 'bottom' of the layer
    for j, (x, y) in enumerate(reversed(list(zip(xs_outer, ys_outer)))):
        if j == 0:
            x2plot.append(0.0)
            y2plot.append(-y)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(0)+ ', ' + str(-y) + '\n'
        elif j == len(xs_outer) - 1:
            x2plot.append(x)
            y2plot.append(0.0)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(x)+ ', ' + str(0) + '\n'
        else:
            x2plot.append(x)
            y2plot.append(-y)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(x)+ ', ' + str(-y) + '\n'
            
        i += 1

    # Now the outer edge along the 'top'
    for j, (x, y) in enumerate(zip(xs_outer, ys_outer)):
        if j == 0:
            continue
        elif j == len(xs_outer) - 1:
            x2plot.append(0)
            y2plot.append(y)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(0)+ ', ' + str(y) + '\n'
        else:
            x2plot.append(x)
            y2plot.append(y)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(x)+ ', ' + str(y) + '\n'

        i += 1

    # Next, the 'top' of the inner edge
    for j, (x, y) in enumerate(reversed(list(zip(xs_inner, ys_inner)))):
        if j == 0:
            x2plot.append(0.0)
            y2plot.append(y)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(0)+ ', ' + str(y) + '\n'
        elif j == len(xs_outer) - 1:
            x2plot.append(x)
            y2plot.append(0.0)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(x)+ ', ' + str(0) + '\n'
        else:
            x2plot.append(x)
            y2plot.append(y)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(x)+ ', ' + str(y) + '\n'

        i += 1

    # Finally, the 'bottom' of the inner edge
    for j, (x, y) in enumerate(zip(xs_inner, ys_inner)):
        if j == 0:
            continue
        elif j == len(xs_outer) - 1:
            x2plot.append(0.0)
            y2plot.append(-y)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(0)+ ', ' + str(-y) + '\n'
        else:
            x2plot.append(x)
            y2plot.append(-y)
            r2dp_insert += indent + 'p'+ str(i) + ' = ' + str(x)+ ', ' + str(-y) + '\n'

        i += 1

    ax.plot(x2plot, y2plot, linewidth = 1.5, linestyle = '--')
        
    r2dp_insert += INDENT + '  endi\n'
    return r2dp_insert, ax



#          _.________.________._______.________.________._
#   _ ____| .___     . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |_.________.________._______.________.________._|
# This function writes the planet data to an 'assemblage' in DIATOM
def getAssemblage(diatom_data, run_name, rot_vel, pdt_flg, center, velocity, planetNum):
    assemblage = INDENT + 'assemblage \''+run_name+'\'\n'
    assemblage += INDENT + ' center '+str(center[0])+', '+str(center[1])+', '+str(center[2])+'\n'
    assemblage += INDENT + ' avelocity 0, 0, '+str(rot_vel)+'\n'
    assemblage += INDENT + ' velocity '+str(velocity[0])+', '+str(velocity[1])+', '+str(velocity[2])+'\n'

    fig, ax = plt.subplots()
    
    # Insert a layer into the diatom file
    for i, diatom_layer in enumerate(diatom_data):
        assemblage += INDENT+' package \'layer'+str(i)+'\'\n'
        assemblage += INDENT+'  mat '+str(diatom_layer.material)+'\n'
        assemblage += INDENT+'  m'+str(diatom_layer.material)+'id '+str(planetNum)+'\n'

        if pdt_flg == 1:
            assemblage += INDENT+'  pressure '+str(diatom_layer.p)+'\n'
            assemblage += INDENT+'  temperature '+str(diatom_layer.T)+'\n'
        elif pdt_flg == 2:
            assemblage += INDENT+'  density '+str(diatom_layer.rho)+'\n'
            assemblage += INDENT+'  temperature '+str(diatom_layer.T)+'\n'
        elif pdt_flg == 3:
            assemblage += INDENT+'  density '+str(diatom_layer.rho)+'\n'
            assemblage += INDENT+'  pressure '+str(diatom_layer.p)+'\n'
        else:
            error('Invalid PDT_FLG')
        
        if i == len(diatom_data) - 1:
            r2dp_insert, ax = get_r2dp_insert(diatom_layer.xs[:], diatom_layer.zs[:], [0.0], [0.0], center, ax)
            assemblage += r2dp_insert
        else:
            r2dp_insert, ax = get_r2dp_insert(diatom_layer.xs[:], diatom_layer.zs[:], diatom_data[i + 1].xs[:], diatom_data[i + 1].zs[:], center, ax)
            assemblage += r2dp_insert

        assemblage += INDENT+' endpackage\n'

    assemblage += INDENT + 'endassemblage\n'

    if DEBUG:
        ax.set_title('Inserted layer boundaries')
        ax.set_xlim(0, max(diatom_data[0].xs[:]))
        ax.set_ylim(0, max(diatom_data[0].zs[:]))
        plt.show()

    return assemblage



#           _._______.________._______.________.________._
#   _ _____| .___    . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#          |_._______.________._______.________.________._|
def replace_diatom(diatom):
    # open 'blank' CTH input deck
    CTH_base = open(CTH_IN_DIR+CTH_BASE_FNAME, 'r')
    CTH_in = open(CTH_IN_DIR+CTH_IN_FNAME, 'w')
    print("Writing to {}".format(CTH_IN_FNAME))

    CTH_in_str = re.sub(r'(?<=\s)diatom.*?enddiatom', '\ndiatom\n'+diatom+'enddiatom\n', CTH_base.read(), flags = re.DOTALL)

    # Remove the (old) CTH input file contents
    CTH_in.seek(0)
    CTH_in.truncate()

    # Now fill the CTH file with the modified content
    CTH_in.write(CTH_in_str)

    
#          __._______.________._______.________.________._
#   _ ____|  .___    . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |__._______.________._______.________.________._|
def coordTransf(m1, m2):
    print("Arranging bodies in center of mass coordinates")
    print("b = {}".format(DY/pow(DX**2 + DY**2, 0.5)))
    
    mTot = m1 + m2
    mf1 = m1/mTot
    mf2 = m2/mTot
    # First shift the planets so that the center of mass is at the origin
    x1 = -mf2*DX
    y1 = -mf2*DY
    x2 = mf1*DX
    y2 = mf1*DY
    
    # Now we use a mass-averaged velocity for each planet so that the
    # linear momentum is zero
    vx1 = mf2*V_IMP
    vx2 = -mf1*V_IMP

    return [[x1, y1, 0], [x2, y2, 0]], [[vx1, 0, 0], [vx2, 0, 0]]


#          __._______.________._______.________.________._
#   _ ____|  .___    . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |__._______.________._______.________.________._|
if __name__ == '__main__':
    for last_arg, arg in zip(sys.argv[:-1], sys.argv[1:]):
        if last_arg == "-H":
            HERCULES_OUT_DIR = "./"
            HERCULES_OUT_FNAMES = [arg]
        elif last_arg == "-Ci":
            CTH_IN_DIR = "./"
            CTH_IN_FNAME = arg
        elif last_arg == "-Cb":
            CTH_IN_DIR = "./"
            CTH_BASE_FNAME = arg
        elif last_arg == "-pdt":
            PDT_FLG = int(arg)
        elif last_arg == "-dx":
            DX = float(arg)
        elif last_arg == "-dy":
            DY = float(arg)
        elif last_arg == "-v":
            V_IMP = float(arg)

    print("Running ...")
    print("HERCULES_OUT_FNAMES = {}".format(HERCULES_OUT_FNAMES))
    print("CTH_BASE_FNAME = {}".format(CTH_BASE_FNAME))
    print("CTH_IN_FNAME = {}".format(CTH_IN_FNAME))
    print("PDT_FLG = {}".format(PDT_FLG))

    assemblage = ""

    # Get the HERCULES data
    params1, planet1 = get_hercules_data(HERCULES_OUT_FNAMES[0])
    print("omega_rot of planet 1: {}".format(planet1.omega_rot))
    print("aspect ratio of planet 1: {}".format(planet1.aspect))
    try:
        params2, planet2 = get_hercules_data(HERCULES_OUT_FNAMES[1])
        print("omega_rot of planet 2: {}".format(planet2.omega_rot))
        print("aspect ratio of planet 2: {}".format(planet2.aspect))
    except:
        print("no planet 2")
        
    # extract the relavent data for DIATOM
    diatom_data1 = get_diatom_data(params1, planet1, MATERIAL_FNAMES[0])    
    mu_per_poly1 = 4*len(diatom_data1[0].xs)
    print("Number of mu points per polygon in planet 1: "+str(mu_per_poly1))
    # If the number of mu points is too high CTH will complain so the we
    # de-resolve
    if mu_per_poly1 > 2400:
        print("Number of polygon vertices too high. Deresolving to {}".format(NUM_NEW_MU*4))
        new_mu(diatom_data1, NUM_NEW_MU)

    try:
        diatom_data2 = get_diatom_data(params2, planet2, MATERIAL_FNAMES[1])
        mu_per_poly2 = 4*len(diatom_data2[0].xs)
        print("Number of mu points per polygon in planet 2: "+str(mu_per_poly2))
        # If the number of mu points is too high CTH will complain so the we
        # de-resolve
        if mu_per_poly2 > 2400:
            print("Number of polygon vertices too high. Deresolving to {}".format(NUM_NEW_MU*4))
            new_mu(diatom_data2, NUM_NEW_MU)
    except:
        print("no planet 2")

    try:
        centers, velocities = coordTransf(planet1.Mtot, planet2.Mtot)
    except NameError:
        centers = [[0, 0, 0]]
        velocities = [[0, 0, 0]]
        
    # write the DIATOM data to a text file in DIATOM syntax
    assemblage += getAssemblage(diatom_data1, params1.run_name, planet1.omega_rot/2.0/np.pi, PDT_FLG, centers[0], velocities[0], 0)
    try:
        assemblage += getAssemblage(diatom_data2, params2.run_name, planet2.omega_rot/2.0/np.pi, PDT_FLG, centers[1], velocities[1], 1)
    except:
        print("no planet 2")
        
    # Replace the DIATOM block in the CTH input-deck
    replace_diatom(assemblage)


    '''
    for i, (hero_fname, mat_fnames) in enumerate(zip(HERCULES_OUT_FNAMES, MATERIAL_FNAMES)):
        # Get the HERCULES data
        params, planet = get_hercules_data(hero_fname)    
    
        print 'omega_rot: ', planet.omega_rot
        print 'aspect: ', planet.aspect

        # extract the relavent data for DIATOM
        diatom_data = get_diatom_data(params, planet, mat_fnames)

        mu_per_poly = 4*len(diatom_data[0].xs)
        print "Number of mu points per polygon: "+str(mu_per_poly)
        
        # If the number of mu points is too high CTH will complain so the we
        # de-resolve
        if mu_per_poly > 2400:
            print "Number of polygon vertices too high. Deresolving to {}".format(NUM_NEW_MU*4)
            new_mu(diatom_data, NUM_NEW_MU)

        
            
        # write the DIATOM data to a text file in DIATOM syntax
        assemblage += get_assemblage(diatom_data, params.run_name, planet.omega_rot/2.0/np.pi, PDT_FLG, center, velocity, i)
    '''
