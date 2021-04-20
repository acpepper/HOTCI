'''
Copyright Andrew Pepper, 2018
'''
import sys
sys.path.append("/home/acpepper/HOTCI_stuff/HERCULES/HOTCI")

import lib.eos as eos
import lib.HERCULES_structures as hstr
import json
from scipy import interpolate
from scipy.optimize import fsolve
import numpy as np
import re
import sys
import matplotlib.pyplot as plt



class IO_Data:
    def __init__(self, json_Fname):
        try:
            with open(json_Fname, 'r') as inFile:
                inData=inFile.read()

            print("initializing IO_Data object from '{}'".format(json_Fname))

            # parse file
            inObj = json.loads(inData)
        except:
            print("initializing empty IO_Data object")
        
        try:
            self.Verbose = inObj["Verbose"]
        except:
            self.Verbose = False
        try:
            self.CTH_base_Fname = inObj["CTH_base_Fname"]
        except:
            self.CTH_base_Fname = "CTH_base/example.in"
        try:
            self.CTH_in_Fname = inObj["CTH_in_Fname"]
        except:
            self.CTH_in_Fname = "../CTH_in/example_with_initCond.in"
            
        self.HERCULES_out_Fnames = []
        self.mat_Fnames = []
        self.N_planets = 0
        try:
            for i, planet in enumerate(inObj["HERCULES_bodies"]):
                self.HERCULES_out_Fnames.append(planet["HERCULES_out_Fname"])
                self.mat_Fnames.append([])
                try:
                    for material in planet["mat_Fnames"]:
                        self.mat_Fnames[i].append(material)
                except:
                    self.mat_Fnames = [["../EOS_files/default_mantle1.txt",
		                        "../EOS_files/default_core1.txt"],
                                       ["../EOS_files/default_mantle2.txt",
		                        "../EOS_files/default_core2.txt"]]
                self.N_planets += 1
        except:
            self.HERCULES_out_Fnames = ["../HERCULES_out/default_target",
                                        "../HERCULES_out/default_impactor"]

        # determine what has been used to describe the impact angle and distance
        initCond_varNames = ["dx", "dy", "r_imp", "b", "theta"]
        initCond_func_args = [None, None, None, None, None]
        for i, initCond_varName in enumerate(initCond_varNames):
            try:
                initCond_func_args[i] = inObj[initCond_varName]
            except:
                continue
        
        # solve a system of equations for the un-included variable names
        def initCond_func(x, dx=None, dy=None, r_imp=None, b=None, theta=None):
            if dx:
                x[0] = dx
            if dy:
                x[1] = dy
            if r_imp:
                x[2] = r_imp
            if b:
                x[3] = b
            if theta:
                x[4] = theta

            return [np.cos(x[4]) - x[0]/x[2],
                    np.sin(x[4]) - x[1]/x[2],
                    np.tan(x[4]) - x[1]/x[0],
                    x[2] - np.sqrt(x[0]**2 + x[1]**2),
                    x[3] - x[1]/x[2]]

        initCond = fsolve(initCond_func,
                          [1e9, 1e9, 1.4e9, 1e9, 0.785],
                          args=tuple(initCond_func_args))
    
        self.dx = initCond[0]
        self.dy = initCond[1]
        self.r_imp = initCond[2]
        self.b = initCond[3]
        self.theta = initCond[4]

        try:
            self.v_imp = inObj["v_imp"]
        except:
            self.v_imp = 0
        try:
            self.pdt_flg = inObj["pdt_flg"]
        except:
            self.pdt_flg = 2
        try:
            self.i_sm = inObj["i_sm"]
        except:
            self.i_sm = 0
        try:
            self.N_new_mu = inObj["N_new_mu"]
        except:
            self.N_new_mu = 600
        try:
            self.indent = inObj["indent"]
        except:
            self.indent = " "



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
def new_mu(planet_diatom, N_new_mu):
    # Iterate over the layers
    for i, layer in enumerate(planet_diatom):
        # Get a cubic spline describing the shape of the layer
        spl = interpolate.splrep(layer.zs, layer.xs)
        
        # Generate new mu points
        new_zs = np.linspace(0,
                             layer.zs[-1], N_new_mu,
                             endpoint=True)
        
        new_xs = interpolate.splev(new_zs, spl)

        planet_diatom[i].xs = new_xs
        planet_diatom[i].zs = new_zs



#          _.________.________._______.________.________._
#   _ ____| .___     . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |_.________.________._______.________.________._|
def get_hercules_data(hero_fname):
    # Define blank planet and parameter classes
    params = hstr.HERCULES_parameters()
    planet = hstr.HERCULES_planet()

    # Read in HERCULES output
    with open(hero_fname, "rb") as f:
        params.read_binary(f)
        planet.read_binary(f)

    return params, planet



#          _.________.________._______.________.________._
#   _ ____| .___     . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |_.________.________._______.________.________._|
def get_planet_diatoms(params, planet, mat_fnames, ioData):
    planet_diatom = np.zeros(planet.Nlayer, dtype=object)

    # initialize EOS files in order to convert pressure to temperature
    eos_files = np.asarray([])
    for i, mat_fname in enumerate(mat_fnames):
        eos_files = np.append(eos_files, eos.PyEOS())
        eos_files[i].read_EOSfile(mat_fname.encode("utf-8"))

    if ioData.Verbose:
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
        planet_diatom[i] = Diatom_Data()
        planet_diatom[i].material = planet.flag_material[i] + 1
        
        # make the outer layer extra cold to avoid spurrious vaporization
        if i <= ioData.i_sm - 1:
            if i == 0:
                planet_diatom[i].T = .025
            else:
                # Use the EOS class to calculate T
                # and convert K -> ev
                planet_diatom[i].T = eos_files[planet.flag_material[i]].calc_T(planet.press[i])/11604.0
                planet_diatom[i].T = planet_diatom[i].T/2 + planet_diatom[i-1].T/2
                # Now get the density
                # Convert kg m^-3 -> g cm^-3
                planet_diatom[i].rho = eos_files[planet.flag_material[i]].calc_rho_from_T(planet_diatom[i].T*11604.0)/1000.0
                # Now get the pressure
                # Convert kg s^-2 m^-1 -> g s^-2 cm^-1
                planet_diatom[i].p = eos_files[planet.flag_material[i]].calc_p_from_T(planet_diatom[i].T*11604.0)*10.0
            if ioData.Verbose:
                Rho1s.append(planet.real_rho[i]/1000.0)
                P1s.append(planet.press[i]*10.0)
                T1s.append(eos_files[planet.flag_material[i]].calc_T(planet.press[i])/11604.0)
                Rho2s.append(planet_diatom[i].rho)
                P2s.append(planet_diatom[i].p)
                T2s.append(planet_diatom[i].T)
        else:
            # Convert kg m^-3 -> g cm^-3
            planet_diatom[i].rho = planet.real_rho[i]/1000.0
            # Convert kg s^-2 m^-1 -> g s^-2 cm^-1
            planet_diatom[i].p = planet.press[i]*10.0
            # Use the EOS class to calculate T
            # and convert K -> ev
            planet_diatom[i].T = eos_files[planet.flag_material[i]].calc_T(planet.press[i])/11604.0

            if ioData.Verbose:
                Rho1s.append(planet_diatom[i].rho)
                P1s.append(planet_diatom[i].p)
                T1s.append(planet_diatom[i].T)
                Rho2s.append(planet_diatom[i].rho)
                P2s.append(planet_diatom[i].p)
                T2s.append(planet_diatom[i].T)

        for j, (mu, xi) in enumerate(zip(layer.mu, layer.xi)):
            sin_theta_sqrd = 1.0 - mu**2.0
            if sin_theta_sqrd < 0.0:
                sin_theta_sqrd = 0.0
                
            planet_diatom[i].xs = np.append(planet_diatom[i].xs, xi*(layer.a*100.0)*np.sqrt(sin_theta_sqrd))
            planet_diatom[i].zs = np.append(planet_diatom[i].zs, xi*(layer.a*100.0)*mu)
            
        if ioData.Verbose:
            Xs.append(planet_diatom[i].xs[0])
            # Plot the mu points in each layer
            # plt.scatter(planet_diatom[i].xs, planet_diatom[i].zs)
            print('Layer {}:'.format(i))
            print('material: {:d}'.format(planet_diatom[i].material))
            print('density: {:1.3e}'.format(planet_diatom[i].rho))
            print('pressure: {:1.3e}'.format(planet_diatom[i].p))
            print('temperature: {:1.3e}'.format(planet_diatom[i].T))
            print('==========================')

    if ioData.Verbose:
        plt.plot(Xs, Rho1s, 'r-')
        plt.plot(Xs, Rho2s, 'b--')
        plt.show()

    rhos = np.asarray([])
    Ts = []
    for data in planet_diatom:
        rhos = np.append(rhos, data.rho)
        Ts.append(data.T)

    if ioData.Verbose:
        plt.plot(np.arange(0, len(planet_diatom), 1), rhos)
        plt.show()

        plt.plot(np.arange(0, len(planet_diatom), 1), Ts)
        plt.show()

    return planet_diatom



#          __._______.________._______.________.________._
#   _ ____|  .___    . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |__._______.________._______.________.________._|
def get_r2dp_insert(xs_outer, ys_outer, xs_inner, ys_inner, ioData, center, ax):
    r2dp_insert = (3*ioData.indent
                   +'insert r2dp\n')
    r2dp_insert += (4*ioData.indent
                    +'ce1 '
                    +str(center[0])+', '
                    +str(center[1])+', '
                    +str(center[2])+'\n')
    r2dp_insert += (4*ioData.indent
                    +'ce2 '
                    +str(center[0])+', '
                    +str(center[1])+', '
                    +str(center[2] + 1.0)+'\n')
    r2dp_insert += (4*ioData.indent
                    +'ce3 '
                    +str(center[0] + 1.0)+', '
                    +str(center[1])+', '
                    +str(center[2])+'\n')
    r2dp_insert += 4*ioData.indent+'twist = 1\n'
    r2dp_insert += 4*ioData.indent+'pitch = 0, 0\n'

    x2plot = []
    y2plot = []
    
    i = 1
    # First the outer edge along the 'bottom' of the layer
    for j, (x, y) in enumerate(reversed(list(zip(xs_outer, ys_outer)))):
        if j == 0:
            x2plot.append(0.0)
            y2plot.append(-y)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(0)+', '+str(-y)+'\n')
        elif j == len(xs_outer) - 1:
            x2plot.append(x)
            y2plot.append(0.0)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(x)+', '+str(0)+'\n')
        else:
            x2plot.append(x)
            y2plot.append(-y)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(x)+', '+str(-y)+'\n')
            
        i += 1

    # Now the outer edge along the 'top'
    for j, (x, y) in enumerate(zip(xs_outer, ys_outer)):
        if j == 0:
            continue
        elif j == len(xs_outer) - 1:
            x2plot.append(0)
            y2plot.append(y)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(0)+', '+str(y)+'\n')
        else:
            x2plot.append(x)
            y2plot.append(y)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(x)+', '+str(y)+'\n')

        i += 1

    # Next, the 'top' of the inner edge
    for j, (x, y) in enumerate(reversed(list(zip(xs_inner, ys_inner)))):
        if j == 0:
            x2plot.append(0.0)
            y2plot.append(y)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(0)+', '+str(y)+'\n')
        elif j == len(xs_outer) - 1:
            x2plot.append(x)
            y2plot.append(0.0)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(x)+', '+str(0)+'\n')
        else:
            x2plot.append(x)
            y2plot.append(y)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(x)+', '+str(y)+'\n')

        i += 1

    # Finally, the 'bottom' of the inner edge
    for j, (x, y) in enumerate(zip(xs_inner, ys_inner)):
        if j == 0:
            continue
        elif j == len(xs_outer) - 1:
            x2plot.append(0.0)
            y2plot.append(-y)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(0)+', '+str(-y)+'\n')
        else:
            x2plot.append(x)
            y2plot.append(-y)
            r2dp_insert += (4*ioData.indent
                            +'p'+str(i)+' = '+str(x)+', '+str(-y)+'\n')

        i += 1

    ax.plot(x2plot, y2plot, linewidth = 1.5, linestyle = '--')
        
    r2dp_insert += 3*ioData.indent+'endi\n'
    return r2dp_insert, ax



#          _.________.________._______.________.________._
#   _ ____| .___     . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |_.________.________._______.________.________._|
# This function writes the planet data to an 'assemblage' in DIATOM
def getAssemblage(planet_diatom, run_name, rot_vel, ioData, center, velocity, planetNum):
    assemblage = ioData.indent+'assemblage \''+run_name+'\'\n'
    assemblage += (2*ioData.indent
                   +'center '
                   +str(center[0])+', '
                   +str(center[1])+', '
                   +str(center[2])+'\n')
    assemblage += ioData.indent+' avelocity 0, 0, '+str(rot_vel)+'\n'
    assemblage += (2*ioData.indent
                   +'velocity '
                   +str(velocity[0])+', '
                   +str(velocity[1])+', '
                   +str(velocity[2])+'\n')

    fig, ax = plt.subplots()
    
    # Insert a layer into the diatom file
    for i, diatom_layer in enumerate(planet_diatom):
        assemblage += (2*ioData.indent
                       +'package \'layer'+str(i)+'\'\n')
        assemblage += (3*ioData.indent
                       +'mat '+str(diatom_layer.material)+'\n')
        assemblage += (3*ioData.indent
                       +'m'+str(diatom_layer.material)
                       +'id '+str(planetNum)+'\n')

        if ioData.pdt_flg == 1:
            assemblage += (3*ioData.indent
                           +'pressure '+str(diatom_layer.p)+'\n')
            assemblage += (3*ioData.indent
                           +'temperature '+str(diatom_layer.T)+'\n')
        elif ioData.pdt_flg == 2:
            assemblage += (3*ioData.indent
                           +'density '+str(diatom_layer.rho)+'\n')
            assemblage += (3*ioData.indent
                           +'temperature '+str(diatom_layer.T)+'\n')
        elif ioData.pdt_flg == 3:
            assemblage += (3*ioData.indent
                           +'density '+str(diatom_layer.rho)+'\n')
            assemblage += (3*ioData.indent
                           +'pressure '+str(diatom_layer.p)+'\n')
        else:
            error('Invalid PDT_FLG')
        
        if i == len(planet_diatom) - 1:
            r2dp_insert, ax = get_r2dp_insert(diatom_layer.xs[:],
                                              diatom_layer.zs[:],
                                              [0.0],
                                              [0.0],
                                              ioData, center, ax)
            assemblage += r2dp_insert
        else:
            r2dp_insert, ax = get_r2dp_insert(diatom_layer.xs[:],
                                              diatom_layer.zs[:],
                                              planet_diatom[i + 1].xs[:],
                                              planet_diatom[i + 1].zs[:],
                                              ioData, center, ax)
            assemblage += r2dp_insert

        assemblage += 2*ioData.indent+'endpackage\n'

    assemblage += ioData.indent+'endassemblage\n'

    if ioData.Verbose:
        ax.set_title('Inserted layer boundaries')
        ax.set_xlim(0, max(planet_diatom[0].xs[:]))
        ax.set_ylim(0, max(planet_diatom[0].zs[:]))
        plt.show()

    return assemblage



#           _._______.________._______.________.________._
#   _ _____| .___    . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#          |_._______.________._______.________.________._|
def replace_diatom(diatom, ioData):
    # open 'blank' CTH input deck
    CTH_base = open(ioData.CTH_base_Fname, 'r')
    CTH_in = open(ioData.CTH_in_Fname, 'w')
    print("Writing to {}".format(ioData.CTH_in_Fname))

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
def coordTransf(ioData, m1, m2):
    print("Arranging bodies in center of mass coordinates")
    print("b = {} ({})".format(ioData.dy/pow(ioData.dx**2 + ioData.dy**2, 0.5),
                               ioData.b))
    
    mTot = m1 + m2
    mf1 = m1/mTot
    mf2 = m2/mTot
    # First shift the planets so that the center of mass is at the origin
    x1 = -mf2*ioData.dx
    y1 = -mf2*ioData.dy
    x2 = mf1*ioData.dx
    y2 = mf1*ioData.dy
    
    # Now we use a mass-averaged velocity for each planet so that the
    # linear momentum is zero
    vx1 = mf2*ioData.v_imp
    vx2 = -mf1*ioData.v_imp

    return [[x1, y1, 0], [x2, y2, 0]], [[vx1, 0, 0], [vx2, 0, 0]]
    


#          __._______.________._______.________.________._
#   _ ____|  .___    . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |__._______.________._______.________.________._|



#          __._______.________._______.________.________._
#   _ ____|  .___    . ___    .  ___  .    ___ .     ___. |____ _
#  / V | \\\/ | \\\\ ./ | \\\ . / | \ . /// | \. //// | \/// | V \
#  \_\_|_///\_|_//// .\_|_/// . \_|_/ . \\\_|_/. \\\\_|_/\\\_|_/_/
#         |__._______.________._______.________.________._|
if __name__ == '__main__':
    ioData = IO_Data("place-holder-name")
    
    # parse commandline arguments
    # NOTE: commands will be overwritten by the last argument
    #       e.g. "-v 5e5 -v 6e5" will produce a set of initial conditions with
    #       an impact velocity of 6e5 cm/s
    for last_arg, arg in zip(sys.argv[:-1], sys.argv[1:]):
        if last_arg == "-f":
            # overwrite ioData with a new one from the HOTCI.json file
            # (or whatever you choose to name your json file)
            ioData = IO_Data(arg)
        else:
            try:
                ioData.setattr(last_arg[1:], arg)
            except:
                if last_arg[0] == "-":
                    print("Warning: no attribute named {} in IO_Data obj".format(last_arg[1:]))

    print("Running ...")
    print("HERCULES_OUT_FNAMES = {}".format(ioData.HERCULES_out_Fnames))
    print("CTH_BASE_FNAME = {}".format(ioData.CTH_base_Fname))
    print("CTH_IN_FNAME = {}".format(ioData.CTH_in_Fname))
    print("PDT_FLG = {}".format(ioData.pdt_flg))

    
    params = np.zeros(ioData.N_planets, dtype=object)
    planets = np.zeros(ioData.N_planets, dtype=object)
    planet_diatoms = np.zeros(ioData.N_planets, dtype=object)
    mu_per_poly = np.zeros(ioData.N_planets)
    for p_ind, hero_Fname in enumerate(ioData.HERCULES_out_Fnames):
        # Get the HERCULES data
        (params[p_ind], planets[p_ind]) = get_hercules_data(ioData.HERCULES_out_Fnames[0])
        print("omega_rot of planet {}    : {} rads/s".format(p_ind,
                                                             planets[p_ind].omega_rot))
        print("aspect ratio of planet {} : {}".format(p_ind,
                                                      planets[p_ind].aspect))
        
        # extract the relavent data for DIATOM
        planet_diatoms[p_ind] = get_planet_diatoms(params[p_ind],
                                                   planets[p_ind],
                                                   ioData.mat_Fnames[p_ind],
                                                   ioData)
        mu_per_poly[p_ind] = 4*len(planet_diatoms[p_ind][0].xs)
        print("mu pnts per polygon {}    : {}".format(p_ind,
                                                      mu_per_poly[p_ind]))
        # If the number of mu points is too high
        # CTH will complain so the we de-resolve
        if mu_per_poly[p_ind] > 2400:
            print("No. polygon vertices too high.")
            print("new No. of mu pnts        : {}".format(ioData.N_new_mu*4))
            new_mu(planet_diatoms[p_ind], ioData.N_new_mu)        

    # count the number of bodies in the desired impact and transform
    # the coordinate system accordingly
    if len(planets) == 2:
        centers, velocities = coordTransf(ioData,
                                          planets[0].Mtot,
                                          planets[1].Mtot)
    elif len(planets) > 2:
        exit("Behavior for more than 2 planets is not yet defined")
    else:
        centers = [[0, 0, 0]]
        velocities = [[0, 0, 0]]

    # write the DIATOM data to a text file in DIATOM syntax
    assemblage = ""
    for p_ind, planet_diatom in enumerate(planet_diatoms):
        assemblage += getAssemblage(planet_diatom,
                                    params[p_ind].run_name.decode("utf-8"),
                                    planets[p_ind].omega_rot/2.0/np.pi,
                                    ioData,
                                    centers[p_ind],
                                    velocities[p_ind],
                                    p_ind)
        
    # Replace the DIATOM block in the CTH input-deck
    replace_diatom(assemblage, ioData)
