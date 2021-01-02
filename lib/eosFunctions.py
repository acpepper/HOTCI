'''
eosFunctions.py

Collection of functions to use with eostable.py

Created 05/05/20 by R. Citron
Edited by A. C. Pepper May 2020
'''
try:
    from . import eostable
except:
    import eostable
import matplotlib.pyplot as plt
import numpy as np


  
def get_S(eos,rho,T):
    # get entropy using bilinear interpolation
    iT = np.where(eos.T<T)[0][-1]
    iR = np.where(eos.rho<rho)[0][-1]
    SR_T1 = (eos.rho[iR+1]-rho)/(eos.rho[iR+1]-eos.rho[iR])*eos.S[iT,iR]\
            +(rho-eos.rho[iR])/(eos.rho[iR+1]-eos.rho[iR])*eos.S[iT,iR+1]
    SR_T2 = (eos.rho[iR+1]-rho)/(eos.rho[iR+1]-eos.rho[iR])*eos.S[iT+1,iR]\
            +(rho-eos.rho[iR])/(eos.rho[iR+1]-eos.rho[iR])*eos.S[iT+1,iR+1]
    S = (eos.T[iT+1]-T)/(eos.T[iT+1]-eos.T[iT])*SR_T1\
            +(T-eos.T[iT])/(eos.T[iT+1]-eos.T[iT])*SR_T2
    return S
    
def get_P(eos,rho,T):
    # get pressure using bilinear interpolation
    iT = np.where(eos.T<T)[0][-1]
    iR = np.where(eos.rho<rho)[0][-1]
    PR_T1 = (eos.rho[iR+1]-rho)/(eos.rho[iR+1]-eos.rho[iR])*eos.P[iT,iR]\
            +(rho-eos.rho[iR])/(eos.rho[iR+1]-eos.rho[iR])*eos.P[iT,iR+1]
    PR_T2 = (eos.rho[iR+1]-rho)/(eos.rho[iR+1]-eos.rho[iR])*eos.P[iT+1,iR]\
            +(rho-eos.rho[iR])/(eos.rho[iR+1]-eos.rho[iR])*eos.P[iT+1,iR+1]
    P = (eos.T[iT+1]-T)/(eos.T[iT+1]-eos.T[iT])*PR_T1\
            +(T-eos.T[iT])/(eos.T[iT+1]-eos.T[iT])*PR_T2
    return P

def get_phase(eos,rho,T,P,minT=0.):
    '''  
    check to determine phase
    determine phase (solid,liquid,vapor,supercritical fluid, or mixed)
    inputs: eos (eos class), rho (gcc), Temperature (K), Pressure (GPa)
            minT - if defined, everything below minT treated as 'bad'
    return: phase ('s','l','v','scf','sv','sl','lv','bad'),phase mass fraction (xs,xl,xv)
            'bad' phase is anything outside of eos density limits, with T < minT or 
                lowest vapor curve T, or anything in the Tension region.
    '''


    # initialize mass fraction solid, liquid, vapor, superfluid
    xs = 0.
    xl = 0.
    xv = 0.
    errflag = 0.
    S = 0.
    phase = ''

    # check for bad cells:
    # make sure density is within eos bounds
    if rho < eos.rho[0] or rho > eos.rho[-1]:
        phase = 'bad'
    # make sure T is above min T of vapor curve
    elif T <= eos.vc.T[-1] or T <= minT:
        phase = 'bad'

    # if not a bad cell then determine the phase
    if phase != 'bad':
        # compute the entropy
        S = get_S(eos,rho,T)

        #perform linear interpolation to get correct density/entropy on phase curve
        #that corresponds to the cell/tracer temperature
        if T>=eos.tp.T and T < eos.mc.T[-1]:
            im = np.where(eos.mc.T-T<0)[0][-1]
            fracm = (eos.mc.T[im]-T)/(eos.mc.T[im]-eos.mc.T[im+1])
            mc_rs = eos.mc.rs[im]+fracm*(eos.mc.rs[im+1]-eos.mc.rs[im])
            mc_rl = eos.mc.rl[im]+fracm*(eos.mc.rl[im+1]-eos.mc.rl[im])
            mc_Ss = eos.mc.Ss[im]+fracm*(eos.mc.Ss[im+1]-eos.mc.Ss[im])
            mc_Sl = eos.mc.Sl[im]+fracm*(eos.mc.Sl[im+1]-eos.mc.Sl[im])

        if T <= eos.vc.T[0]:
            iv = np.where(eos.vc.T-T>0)[0][-1]
            fracv = (eos.vc.T[iv]-T)/(eos.vc.T[iv]-eos.vc.T[iv+1])
            vc_rv = eos.vc.rv[iv]+fracv*(eos.vc.rv[iv+1]-eos.vc.rv[iv])
            vc_rl = eos.vc.rl[iv]+fracv*(eos.vc.rl[iv+1]-eos.vc.rl[iv])
            vc_Sv = eos.vc.Sv[iv]+fracv*(eos.vc.Sv[iv+1]-eos.vc.Sv[iv])
            vc_Sl = eos.vc.Sl[iv]+fracv*(eos.vc.Sl[iv+1]-eos.vc.Sl[iv])
        

        # if T > max T melting curve, then it is vapor or scf
        if T > eos.mc.T[-1]:
            # if P < critical pressure, then still vapor
            if P <= eos.cp.P:
                phase = 'v'
                xv = 1.0
            # otherwise, supercritical fluid
            else:
                phase = 'scf'
        # above triple point
        elif T >= eos.tp.T:
            # if rho > mc.rs then it is a solid
            if rho > mc_rs:
                phase = 's'
                xs = 1.0
            # if mc.rl < rho < mc.rs then it is mixed liquid solid
            elif rho > mc_rl:
                phase = 'sl'
                # check to make sure that S is with phase boundary S values
                # if it isn't, set it to the closest phase boundary value (treat as pure phase)
                # this happens due to discrete table of S values and discrete phase curve points
                if S < mc_Ss or S > mc_Sl:
                    print('WARNING: in mixed liquid-solid state but S out of bounds:  rho=%g  T=%g' % (rho,T))
                    errflag = 1
                    if S < mc_Ss:
                        print('\ttreating as solid: S = %g  Sl = %g' % (S,mc_Ss))
                        S = mc_Ss
                        phase = 's'
                    if S > mc_Sl:
                        print('\ttreating as liquid: S = %g  Sl = %g' % (S,mc_Sl))
                        S = mc_Sl
                        phase = 'l'
                xl = (S-mc_Ss)/(mc_Sl-mc_Ss)
                xs = 1.0 - xl
            else:
                # if below critical point:
                if T <= eos.vc.T[0]:
                    # if vc.rl < rho < mc.rl then liquid
                    if rho > vc_rl:
                        phase = 'l'
                        xl = 1.0
                    # if vc.rv < rho < vc.rl then mixed vapor liquid
                    elif rho > vc_rv:
                        phase = 'lv'
                        if S < vc_Sl or S > vc_Sv:
                            errflag = 1
                            print('WARNING: in mixed liquid-vapor state but S out of bounds:  rho=%g  T=%g' % (rho,T))
                            if S < vc_Sl and T < eos.tp.T+5:
                                # if within 5K of triple point can get some errors from 
                                # bilinear S value extrapolating from points below TP
                                # so recompute S just using next highest S value (in T)
                                print('Adjusting S so not extrapolating from below TP')
                                iT = np.where(eos.T<T)[0][-1]
                                iR = np.where(eos.rho<rho)[0][-1]
                                S = eos.S[iT+1,iR]

                            if S < vc_Sl:
                                print('\ttreating as liquid: S = %g  Sl = %g' % (S,vc_Sl))
                                S = vc_Sl
                                phase = 'l'
                                
                            if S > eos.vc.Sv[iv]:
                                print('\ttreating as vapor: S = %g  Sl = %g' % (S,vc_Sv))
                                S = vc_Sv
                                phase = 'v'
                        xv = (S-vc_Sl)/(vc_Sv-vc_Sl)
                        xl = 1.0 - xv
                    # if rho < vc.rv then vapor
                    else:
                        phase = 'v'
                        xv = 1.0
                # if above critical point
                else:
                    # if P < critical pressure, then still vapor
                    if P <= eos.cp.P:
                        phase = 'v'
                        xv = 1.0
                    # otherwise, supercritical fluid
                    else:
                        phase = 'scf'
        # below the triple point
        else:
            # solid if rho greater than solid curve (
            if rho > vc_rl:
                phase = 's'
                xs = 1.0
            # if between vapor and solid curve then mixed vapor solid (tension region)
            elif rho > vc_rv:
                phase = 'sv'
                P = get_P(eos,rho,T)
                # compute phase unless it is in tension region
                if P < 0:
                    phase = 'bad'
                elif S < vc_Sl or S > vc_Sv:
                    print('WARNING: in mixed vapor-solid state but S out of bounds:  rho=%g  T=%g' % (rho,T))
                    errflag = 1
                    if S < vc_Sl and T > 1000.:
                        # if its the curve point right below the triple point adjust
                        # the entropy to avoid error
                        if eos.vc.T[iv] == eos.tp.T:
                            print('resetting vapor-solid S to tp value')
                            print('old vc_Sl = ',vc_Sl)
                            vc_Sl = eos.tp.Sim+fracv*(eos.vc.Sl[iv+1]-eos.tp.Sim)
                            print('new vc_Sl = ',vc_Sl)
                            print('S value = ',S,'S > vc_Sl',S>vc_Sl)
                    if S < vc_Sl and T < eos.vc.T[-3]:
                        # on cold part of tail of curve, do quadratic fit instead of linear
                        Slx = eos.vc.Sl[-3:]
                        Svx = eos.vc.Sv[-3:]
                        Tx = eos.vc.T[-3:]
                        pl = np.polyfit(Tx,Slx,2)
                        pv = np.polyfit(Tx,Svx,2)
                        Slfit = pl[0]*T**2+pl[1]*T+pl[2]
                        Svfit = pv[0]*T**2+pv[1]*T+pv[2]
                        if S >= Slfit:
                            print('cold part of curve, using quadratic fit:')
                            print('T = ',T,'S = ',S,'vc_Sl = ',vc_Sl,'Slfit = ',Slfit)
                            vc_Sl = Slfit
                            vc_Sv = Svfit
                    if S < vc_Sl:
                        print('\ttreating as solid: S = %g  Sl = %g' % (S,vc_Sl))
                        S = vc_Sl
                        phase = 's'
                    if S > vc_Sv:
                        print('\ttreating as vapor: S = %g  Sl = %g' % (S,vc_Sv))
                        S = vc_Sv
                        phase = 'v'
                xv = (S-vc_Sl)/(vc_Sv-vc_Sl)
                xs = 1.0 - xv
            # if rho < eos.vc.rv then it is pure vapor
            else:
                phase = 'v'
                xv = 1.0


    return phase,xs,xl,xv,S,errflag



def compute_phase(eospath,rho,T,P,minT=0.):
    '''
    compute phase data for array of rho,T,P
    -----------------------
    INPUTS:
    eospath - path to eos directory (e.g. '/home/ric/research/aneos/aneos-forsterite-2019/')
    rho [kg/m^3], T [K], P [Pa] - density, temp,  and pressure of material
    minT: if defined, anything below minT is treated as 'bad' by getPhase
    -----------------------
    if rho,T,P are input as 2D arrays (e.g., multiple timesteps for each cell/tracer)
        then they are reshaped and the computation is carried out on 1D arrays, 
        which are then reshaped back to the original 2D format
    -----------------------
    Example:
    eospath = '/home/ric/research/aneos/aneos-forsterite-2019/'
    eos,phase,xs,xl,xv,S = compute_phase(eospath,rho2,T2,P2)
    
    ^ this returns eos class, phase, and mass fraction of solid, liquid, gas
      and entropy for each cell
    '''
    aneosinfile = eospath+'ANEOS.INPUT'
    aneosoutfile = eospath+'ANEOS.OUTPUT'
    sesamepathEXT = eospath+'NEW-SESAME-EXT.TXT' # original gridstyle1 format
    sesamepathSTD = eospath+'NEW-SESAME-STD.TXT' # original gridstyle1 format
    eos = eostable.extEOStable()
    eos.loadaneos(aneosinfname=aneosinfile,aneosoutfname=aneosoutfile,includetension=True,setrefs=True)
    eos.loadextsesame(sesamepathEXT)
    eos.loadstdsesame(sesamepathSTD)

    # compute shapeSl = 0.00329225
    shape = np.shape(P)
    size = np.product(shape)

    # reshape input arrays
    rho=rho.reshape(size,)
    T=T.reshape(size,)
    P=P.reshape(size,)
    error = np.zeros(size)
    

    # create necessary arrays
    phase = np.array(['   ']*size)
    xs = np.zeros(size)
    xl = np.zeros(size)
    xv = np.zeros(size) # mass fraction vapor
    S = np.zeros(size)


    Nerrs = 0
    for i in range(size):
        if rho[i] > 0.:
            # get phase data (convert rho to gcc and pressure to GPa for input)
            phase[i],xs[i],xl[i],xv[i],entropy,errflag = get_phase(eos,rho[i]/1.e3,T[i],P[i]/1.e9,minT=minT)
            S[i]=entropy
            error[i] = errflag

    print('Nerrs: ',np.sum(error))
    # reshape arrays for return
    phase = phase.reshape(shape)
    xs = xs.reshape(shape)
    xl = xl.reshape(shape)
    xv = xv.reshape(shape)
    S = S.reshape(shape)
    return eos,phase,xs,xl,xv,S
