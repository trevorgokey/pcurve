#!/usr/bin/env python3
import numpy as np
import projection as pro
import update as up
from multiprocessing import Pool
from math import ceil
import scipy.interpolate as si
import sys
import pickle
import os

VERBOSE = False

def savexyz(X,filename,mode='a',atom="C"):
    with open(filename,mode) as f:
        f.write(("{:d}\n\n").format(X.shape[0]))
        for r in X:
            f.write(("{:4s}" + " {: 8.6f}"*X.shape[1] + "\n").format(atom,*r))
        f.flush()
        f.close()

def infoprint(s,end="\n"):
    if(VERBOSE):
        print(s,end=end)

def rescale(f,N=None,targetL=None,freezeends=False, interval=None):
    if (targetL is None):
        targetL = curveEuc(f,0,f.shape[0])

    if(interval != None):
        N = int(targetL / interval)
    elif (N is None):
        N = f.shape[0]
    if(N == 1):
        el = targetL
    else:
        el = targetL/(N-1)
        # redistribute points along path to unit speed (evenly spaced)

    infoprint("\rReparameterizing curve...       ",end="")
    f1 = f.copy()
    pt=f[0]
    j=1
    f = np.empty([N] + list(f.shape[1:]),dtype=np.float64)
    end = f.shape[0]
    f[0] = f1[0]
    if(freezeends):
        end -= 1
        f[-1] = f1[-1]
    for i in range(1,end):
        l = 0
        k = 0
        w = 1.0
        while(j < f.shape[0]):
            k = euc(pt,f1[j])
            if(l+k > el):
                break
            l += k
            pt = f1[j]
            j += 1
        if(j == f.shape[0]):
            j = f.shape[0] - 1
        if(k == 0):
            w = 1.0
        else:
            w = (el-l)/k 
        pt = pt +  w * (f1[j] - pt)
        f[i] = pt
    return f

def clip_driver(f, N=None, freezeends=False, exe=None, procs=1, interval=None):
    # TODO
    #shutdown = False
    #
    #if( exe is None and procs > 1):
    #    exe = Pool(processes=procs)
    #    shutdown = True
    #fn = clip.start
    #fn = clip
    #if(fn is clip.start):
    #    chunk=500
    #    if(procs * chunk > f.shape[0]):
    #        chunk = ceil(f.shape[0] / procs)
    #    work = [exe.apply_async(fn,(f,chunk)) for j in range(0,p.shape[0],chunk)]# for j in range(X.shape[0])]
    #    out = np.array([result.get() for result in work])
    #    out = np.vstack([ck for ck in out])
    #elif(procs > 1):
    #    work = [exe.apply_async(fn,(f,)) for j in range(f.shape[0])]
#   #     work = [exe.submit(fn,tf,w,X,E,f,j,scale=scale) for j in range(f.shape[0])]
    #    out = np.array([result.get() for result in work])
#   #     f = np.array([fut.result() for fut in concurrent.futures.as_completed(work)])
    #else:
    #    out = np.array([fn(f) for j in range(f.shape[0])])
    #s = np.argsort(out[:,0])
    #f = out[s][:,1:]
    #if(shutdown):
    #    exe.close()
    #    exe.join()
    #return f
    return None
    



def clip(f, N=None, freezeends=False, exe=None, procs=1, interval=None):

    clipped = True
    it = 0
    start=1
    end=f.shape[0]-2
    maxclip = 0.0
    quick = False
    dx = 1
    costheta = np.cos(np.pi*3/4)
    if quick:
        dx = 2
    while(clipped == True and it < 10000):
        clipped = False
        it += 1
        maxclip = 0.0
        f0 = f.copy()
        for i in range(1,f.shape[0]-1,dx):
            a = euc(f[i-1],f[i ])
            b = euc(f[i+1],f[i ])
            if(a == 0.0 or b == 0.0):
                continue
            p = (((f[i-1] - f[i])/a) * (f[i+1] - f[i])/b).sum()
            #if(p < a or  euc(f[i+1],f[i ]) < a):
            # 
            if(p > costheta ):
                maxclip = max(maxclip, abs(p))
                f0[i] = (f[i-1] + f[i+1]) / 2.0
                clipped = True
        f = f0.copy()
        if quick:
            for i in range(2,f.shape[0]-1,2):
                a = euc(f[i-1],f[i ])
                b = euc(f[i+1],f[i ])
                if(a == 0.0 or b == 0.0):
                    continue
                p = (((f[i-1] - f[i])/a) * (f[i+1] - f[i])/b).sum()
                #if(p < a or  euc(f[i+1],f[i ]) < a):
                # 
                if(p > costheta ):
                    maxclip = max(maxclip, abs(p))
                    f0[i] = (f[i-1] + f[i+1]) / 2.0
                    clipped = True
            f = f0.copy()
        if True and maxclip > 0.0:
            print("clip: {:d} {:20.15e}".format(it, maxclip), end="\r")
        if maxclip == 0.0:
            break
    print()
    #print("clip: {:d} {:20.15e}".format(it, maxclip), end="\r")

    #targetL = curveEuc(f, 0, f.shape[0])
    #p = rescale(f, N=None, targetL=targetL, 
    #    freezeends=freezeends, interval=None)
    #p2 = rescale(f[::-1], N=None, targetL=targetL, 
    #    freezeends=freezeends, interval=None)
    #f = (p + p2[::-1])/2.0
    #f = rescale(f, N=None, targetL=targetL, 
    #    freezeends=freezeends, interval=None)
    return f


def update(tf, f, X, E, tp, p, w, exe=None, procs=1, scale=1.0, maxstep=1.0, targetL=None, freezeends=False):
    shutdown = False
    
    if( exe is None and procs > 1):
        exe = Pool(processes=procs)
        shutdown = True
    ptA,ptB = None,None
    srt = 0
    end = f.shape[0]
    if(freezeends):
        ptA = p[0]
        ptB = p[-1]
        srt += 1
        end -= 1
    fn = up.start
    #fn = update_curve
    if(fn is up.start):
        chunk=500
        if(procs * chunk > p.shape[0]):
            chunk = ceil(p.shape[0] / procs)
        work = [exe.apply_async(fn,(tf,f,X,E,tp,p,w,j,scale,maxstep,chunk)) for j in range(0,p.shape[0],chunk)]# for j in range(X.shape[0])]
        out = np.array([result.get() for result in work])
        out = np.vstack([ck for ck in out])
    elif(procs > 1):
        work = [exe.apply_async(fn,(tf,w,X,E,f,j,scale,maxstep)) for j in range(f.shape[0])]
#        work = [exe.submit(fn,tf,w,X,E,f,j,scale=scale) for j in range(f.shape[0])]
        out = np.array([result.get() for result in work])
#        f = np.array([fut.result() for fut in concurrent.futures.as_completed(work)])
    else:
        out = np.array([fn(tf,w,X,E,f,j,scale=scale,maxstep=maxstep) for j in range(f.shape[0])])
    s = np.argsort(out[:,0])
    p = out[s][:,1:]
    if(shutdown):
        exe.close()
        exe.join()
    
    if(freezeends):
        p[0] = ptA
        p[-1] = ptB

    return p
    return clip(p, freezeends=freezeends)

def project( f0, X, exe=None, procs=1, eps=1e-14):
    """f0 must be sorted st the first point is beginning of path"""
#    import concurrent.futures
    shutdown = False
    L = 0.0
    if( exe is None and procs > 1):
        exe = Pool( processes=procs)
#        exe = concurrent.futures.ProcessPoolExecutor(max_workers=procs)
        shutdown = True
    tf = np.empty( (X.shape[0],), np.float64)
    f  = np.empty( (X.shape[0], X.shape[1]), np.float64)

    fn = pro.start
    #fn = projectionIDX
    if( fn is pro.start):
        chunk=100
        if( procs * chunk > X.shape[0]):
            chunk = ceil( X.shape[0] / procs)
        work = [exe.apply_async( fn, (f0, X, j, chunk, eps)) for j in range( 0, X.shape[0], chunk)]# for j in range(X.shape[0])]
        out = np.array( [result.get() for result in work])
        out = np.vstack( [ck for ck in out])
    elif( procs > 1):
        work = [ exe.apply_async( fn, (f0, X, j, eps)) for j in range( X.shape[0])]
        out = np.array( [result.get() for result in work])
    else:
        out = np.array( [fn(f0,X,j,eps) for j in range( X.shape[0])])
    infoprint("\rDone. Sorting results..       ",end="")
    s = np.argsort(out[:,0])
    out = out[s]
    infoprint("\rAssembling path..       ",end="")
    L = out[:,2].mean()
    tf[:] = out[:,1]
    tf *= 1.0/L
    f[:] = out[:,3:]
    if(shutdown):
        exe.close()
        exe.join()
    return tf,f,L


def projectionIDX(f,X,ID,eps=1e-14):
    x = X[ID]
    prevF = f[0]
    fnew = prevF
    minD = euc(x,prevF)
    
    minL = 0.0
    
    L = 0.0
    for j in range(1,f.shape[0]):
        l = euc(prevF,f[j])
        if(abs(l) < eps):
            prevF = f[j]
            continue
        ev = (f[j] - prevF) / l
        p = np.dot((x - prevF).T, ev) 
        d = np.inf
        # if the projection is shorter than the segment, the projection is
        # the segment min
        dl = 0.0
        if(p == 0.0):
            d = euc(prevF,x)
        elif (p > 0.0):
            dl = p
            if(p < l):
               # pythag to find the distance to proj point
               z = euc(prevF,x)
               d = (z*z - p*p)**.5
            else:
                d = euc(x,f[j])
        elif (j > 1):
            prevF = f[j]
            L += l
            continue

        if(d < minD and abs(d - minD) > eps):
            minD = d
            minL = (L + dl)
            if(p > l):
                fnew = f[j]
            else:
                fnew = prevF + dl/l * (f[j] - prevF)

        prevF = f[j]
        L += l
    
    print("\r" + "Project     {:10d} {: 7.3f} %      ".format(ID,float(ID)/X.shape[0]*100),end="")
    Xf = np.concatenate(([ID,minL,L],fnew))
    #print("ret ", ID, minL, L)
    return Xf


def update_curve(tf,w,X,E,f,j,scale=1.0,maxstep=1.0):
    keep = int(X.shape[0]*w)
    if(keep < 1):
        keep = 1
    left = False
    right = False
    dl = 0.0
    dr = 0.0
    i = 0
    pt = np.zeros_like(X[0])
    wsum = 0.0
    kept = 0
    di = -1
    if(j == tf.shape[0]-1):
        right = True
    if(j == 0):
        left = True
    nbp = []
    nbw = []
    nbl = []
    D = 0.0
    mini = i
    maxi = i
    ehood = []
    nbhood = []
    while(not (left and right)):
        ji = j + i
        l = abs(tf[j] - tf[ji])
        L = 0.0
        #print("l is", l)
        if(l > 0.0):
            if (i >= 0):
                if(kept >= keep):
                    right = True
                    i = -i - 1
                    #print("j=", j, "right=True")
                    continue
                dr += l
                L = dr
            else:
                if(kept >= keep):
                    left = True
                    i = -i
                    #print("j=", j, "left=True")
                    continue
                dl += l
                L = dl
            #print("j=", j, "dl=", dl, "dr=", dr, "L=", L)
        D = max(L,D)
        kept += 1
        nbl.append(L)
        #print("j=",j," pushed ", L, "D=",D)
        mini = min(mini,i)
        maxi = max(maxi,i)
        
        
        if (left == True):
            i += 1
            if(j+i == tf.shape[0]):
                right = True
        elif (right == True):
            i -= 1
            if(j+i < 0):
                left = True
        else:
            if(ji == tf.shape[0]-1):
                right = True
            elif(ji == 0):
                left = True
            if (i >= 0):
                i = -i - 1
            else:
                i = -i
    nbhood = X[j+mini:j+maxi+1]
    ehood = E[j+mini:j+maxi+1]
    #print("NBL=",nbl)
    #print("D=",D)
    if(D > 0.0):
        nbw = (1 - (np.array(nbl)/D)**3)**3
    else:
        nbw = np.full(kept,1.0/kept)
    #print("NBW=",nbw)
    nbw = (nbw * ehood) / (ehood * nbw).sum()
    #print("j=",j,"dot", nbw.reshape(1,-1)*nbhood)
    npt = np.dot(nbw.reshape(1,-1),nbhood)[0]
    #print("j=",j,"NPT=",npt)
    
    norm = scale*euc(npt,f[j])
    if(norm > maxstep):
        scale = maxstep/norm
    pt = f[j] + scale*(npt - f[j])
    #print("j=",j,"PT=",pt)
    pt = np.hstack((j,pt))
    infoprint("\r" + "Expectation {:10d} {: 7.3f} %      ".format(j,float(j)/tf.shape[0]*100),end="")
    return pt

#def plot_pcurve3D(X,fig=None,c='blue',line=False):
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D
#    from mpl_toolkits.mplot3d import proj3d
##    for orthogonal
#    proj3d.persp_transformation = orthogonal_proj
#    if(fig is None):
#        fig = plt.figure()
#        ax = fig.add_subplot(111,projection='3d')
#    else:
#        ax = fig.axes[0]
#    if(line):
#        ax.plot(*X,c=c)
##        ax.scatter(*X,c=c)
#    else:
#        ax.scatter(*X,c=c,s=10)
#    return fig
#
#def plot_mlab_pcurve3D(X,fig=None,c='blue',line=False):
#    import mayavi.mlab as mlab
#    return mlab.point3d(*X,mode=sphere,color=c)
    

def calc_W_mat(t,f,X,I,tc,fc,C,K,L,U,F=None,distance="L", FORCE_SWITCH=0):
    import itertools as itr
    import time
    from datetime import timedelta
    N_c = I.max()+1 # because of 0 based indexing
    N_k = np.zeros(N_c,np.int32)

    for k in range(N_c):
        N_k[k] = (I[I == k]).shape[0]
    maxN_k = N_k.max()
    G_k = np.zeros((N_c,maxN_k),np.int32)
    for k in range(N_c):
        G_k[k][:N_k[k]] = np.arange(N_k[k], dtype=np.int32)


    import pymbar
    from pymbar import timeseries
    ###
    #
    # Need to subsample the datasets and load them into mbar
    # then need to calc p(e) for the entire set
    # get a center and stiffness for each pt
    #for k in range(N_c):
    #    try:
    #        g = timeseries.statisticalInefficiency(t[I==k])
    #        idx = timeseries.subsampleCorrelatedData(t[I==k], g=g)
    #        N_k[k] = len(idx)
    #        G_k[k][:N_k[k]] = idx
    #    except pymbar.utils.ParameterError:
    #        pass

    maxG_k = N_k.max()
    if(VERBOSE):
        print("\nSubsampling reduced max from", maxN_k, "to", maxG_k)
    maxN_k = maxG_k

    if(VERBOSE):
        print("\nCreating weight matrix requires",N_c**2*maxN_k*8/1e6,"MB")
    W_kln = np.zeros((N_c,N_c,maxN_k),np.float64)
    unit = L
    if(distance == "tf"):
        unit = 1.0


    kT = .001987*310.

    infoprint("\rBuilding MBAR weights...        ",end="")
    for k in range(N_c):
        #N_k[k] = (I[I == k]).shape[0]
        _tk = t[I == k][G_k[k]]
        _Xk = X[I == k][G_k[k]]
        _fk = f[I == k][G_k[k]]
        for n in range(N_k[k]):
            # this is snap n from sim k
            # need to eval this to center l
#            W_kln[k,:,n] = K/2.0 * ((tc - _tk[n])*unit)**2
            if(FORCE_SWITCH == 0):
                dis = np.array([(_tk[n] - cc) for cc in tc])
            elif(FORCE_SWITCH == 1):
                dis = np.array([abs(_tk[n] - cc) \
                        + euc(_Xk[n], _fk[n])/L \
                        + euc(cr, fcr)/L \
                        for cc,cr,fcr in zip(tc,C,fc)])
            elif(FORCE_SWITCH == 2):
                dis = np.linalg.norm(C - _Xk[n],axis=1)/L
            elif(FORCE_SWITCH == 3):
                dis = np.array([abs(_tk[n] - cc) \
                        - euc(_Xk[n], _fk[n])/L \
                        + euc(cr, fcr)/L \
                        for cc,cr,fcr in zip(tc,C,fc)])
#            dis = _tk[n] - tc
#            dis = (_tk[n] - tc) + np.array([(euc(_Xk[n],_fk[n]) + euc(tcc,cc))/L for tcc,cc in zip(tc,C)])
            W_kln[k,:,n] = K/2.0 * (dis*unit)**2 + U[I == k][n]
    #                        W_kln[k,:,n] = K/2.0 * ((_tk - tc[k])*L)**2 

    initial_F = F
    if((np.abs(F) < 1e-7).all()):
        infoprint("\rCalculating MBAR guess...        ",end="")
        initial_F = (W_kln[np.diag_indices(len(N_k), ndim=2)]).sum(axis=1)/N_k 
        #initial_F[K == 0.0] = 0.0

    infoprint("\rCalculating MBAR...        ",end="")
    repeat = False
    mbar_i = 1
    mbar_i_max = 10
    mbar = None
    Wnk = None
    while(mbar_i <= mbar_i_max):
        try:
            tm = time.time()
            mbar = pymbar.MBAR(W_kln, N_k, initial_f_k=initial_F,verbose = False)
            F = mbar.f_k
            Wnk = mbar.W_nk
            #F[K == 0.0] = 0.0
            F -= F.min()
            # timing
            tm2 = time.time()
            tmd = tm2 - tm
            d = timedelta(seconds=tmd)
            tm = tm2
            timestr = str(d)
            infoprint("\nMBAR step: {: 4d}/{: 4d} min= {:8.6f} max= {:8.6f} mean= {:8.6f} stddev= {:8.6f} time= {:s}".format(
                mbar_i, mbar_i_max, F.min(), F.max(), F.mean(), F.std(), timestr), end="")
            initial_F = F
            valid = pymbar.utils.check_w_normalized(Wnk, N_k)
            infoprint("\n", end="")
            break
        except pymbar.utils.ParameterError as e:
            mbar_i += 1
    #try:
    #    valid = pymbar.utils.check_w_normalized(Wnk, N_k)
    #except pymbar.utils.ParameterError as e:
    #    if( hasattr(e, 'message')):
    #        print(e.message)
    #    else:
    #        print(e)
    #    print("MBAR weights still unnormalized! Proceed with caution.")
    #F -= F.min()
    ene = np.zeros(t.shape[0])
#    ene,_ = mbar.computeExpectations(t,compute_uncertainty=False)
    c = 0.0
    f0 = 0.0
    #F,_ = mbar.computeExpectations(t,compute_uncertainty=False)
    #return ene,F
    infoprint("\nEvaluating p(X) from MBAR...        ",end="")

    #z = np.zeros((K.shape[0],U.shape[0]))
    z = np.zeros(U.shape[0])
    for k in range(K.shape[0]):
        if(FORCE_SWITCH == 0):
            dis = t - tc[k] 
        elif(FORCE_SWITCH == 1):
            dis = np.abs(t - tc[k]) + \
                ( np.linalg.norm(f-X,axis=1) + euc(fc[k],C[k]) )/L
        elif(FORCE_SWITCH == 2):
            dis = ( np.linalg.norm(C[k]-X, axis=1) )/L
        elif(FORCE_SWITCH == 3):
            dis = np.abs(t - tc[k]) + \
                ( -np.linalg.norm(f-X,axis=1) + euc(fc[k],C[k]) )/L
        bias = (K[k])/2.0 * (dis*unit)**2 + U
        #z[k] = (N_k[k] * np.exp(F[k] - bias/kT))
        z += (N_k[k] * np.exp(F[k] - bias/kT))
    #z = np.sort(z.flat).sum()
    c = np.sum(np.exp(-U/kT)/z)
    ene = np.exp(-U/kT)/(c*z)
    infoprint("integ p(X) = {:12.8e} <c> = {:12.8e} <z> = {:16.8g} ".format( ene.sum(), c.mean(), z.mean()), end="\n")
    #infoprint("integ p(X) = {:12.8e} <c> = {:12.8e} <z> = {:16.8g} ".format( ene.sum(), c, z), end="\n")

    return ene,F,mbar

def bias_spring(refcrd, refidx, idx, k, crd, refalign=False):
    if(refalign):
        pass # TODO: align crd to refcrd
    bias = k/2.0 * (crd[idx] - refcrd[refidx].mean(axis=0))**2


################################################################################
#                   The function to rule them all
################################################################################
def pcurve3D_MBAR(X,I,C,K,U=None,E=None,procs=1,
    eps=0.0005,eps_ene=.001,N=[100],W=[.1],init=None,checkpoint=None,
    scale_list=[1.0],maxstep=1.0,mbar=(-1,-1),
    freezeends=False,freezerange=None,interval=.1,
    FORCE_SWITCH=0,use_ene_indices=[],adaptive=False,savechk=False):
    """
        X is the dataset positions (Nx3)
        I is the membership of each pt in X to C (Nx1; values are [0,K))
        C is the spring center (Kx3)
        K is the force constants (Kx1)
    """
    procs = int(procs)
    from sklearn.decomposition import PCA
    import time
    from datetime import timedelta
    ii = 0
    MBAR = None
    calc_init = True
    
    if(checkpoint and os.path.exists(checkpoint)):
        print("Loading checkpoint from",checkpoint)
        chk = np.load(checkpoint)
        X = chk['X']
        I = chk['I']
        C = chk['C']
        K = chk['K']
        p = chk['p']
        tp = chk['p']
        scale_list = chk['scale_list']
        F = [chk['F'],chk['F']]
        tf = [chk['tf'],None]
        f = [chk['f'],None]
        ene = [chk['E'],chk['E']]
        tfC = [chk['tfC'],None]
        fC  = [chk['fC'],None]
        if('FORCE_SWITCH' in chk):
            FORCE_SWITCH = chk['FORCE_SWITCH']
#        N = chk['N']
#        W = chk['W']
        ii = int(chk['ii']) + 1
        U = chk['U']
        ORDER = chk['ORDER']
        ORDERC = chk['ORDERC']
    else:
        if(not np.isfinite(X).all()):
            print("X not finite! rows:")
            print(np.arange(X.shape[0])[~np.isfinite(X).any(axis=1)])
            return
        if(not np.isfinite(C).all()):
            print("C not finite! rows:")
            print(np.arange(C.shape[0])[~np.isfinite(C).any(axis=1)])
            return
        #T = np.vstack((X,C)).mean(axis=0)
        T = X.mean(axis=0)
        ORDER = np.arange(X.shape[0])
        ORDERC = np.arange(C.shape[0])
        X = X - T.T
        C = C - T.T
        ene = [E,E]
        if(ene[0] is None):
            ene[0] = np.ones(X.shape[0],np.float64)/X.shape[0]
        if(ene[1] is None):
            ene[1] = np.ones(X.shape[0],np.float64)/X.shape[0]
        if(init is None):
            #initalize f as first eigenval
            # fe is dx1
            #fe = PCA(n_components=1).fit(np.vstack((X,C))).components_[0].reshape(-1,1)
            fe = PCA(n_components=1).fit(X).components_[0].reshape(-1,1)
            # project is Nx1
            projection = np.dot(X,fe).reshape(-1,1)
            #projection = np.linspace(projection.min(),projection.max(),X.shape[0]).reshape(-1,1)
            prjC = np.dot(C,fe).reshape(-1,1)
            f = [np.dot(projection,fe.T),None]
            #if(f[0][0][-1] < 0.0):
            #    f[0] = f[0][::-1]
            #    projection = projection[::-1]
            #    fe = -fe
            if(freezeends and freezerange is not None):
                if(isinstance(freezerange, list)):
                    prA = freezerange[0]#/fe[2]
                    prB = freezerange[1]#/fe[2]
                    projection = np.linspace(prA, prB, X.shape[0]).reshape(-1,1)
                else:
                    freezerange = float(freezerange)
                    prA = min(projection)*freezerange
                    prB = max(projection)*freezerange
                    projection = np.linspace(prA, prB, X.shape[0]).reshape(-1,1)
            # get displacements, want project (Nx1) * fe.T (1xd) = Nxd 
            idx = np.argsort(projection.T[0])
            idxC = np.argsort(prjC.T[0])
            projection = projection[idx]
            prjC = prjC[idxC]
            tf = projection.copy().reshape(-1)
            tfC = prjC.copy().reshape(-1)
            tfC -= tf.min()
            tf -= tf.min()
            tfC = [tfC / tf.max(), tfC/ tf.max() ]
            tf = [tf / tf.max(),tf / tf.max() ]
            fC = [np.dot(prjC,fe.T) ,np.dot(prjC,fe.T) ]
            f = [np.dot(projection,fe.T) ,np.dot(projection,fe.T)]
            F = [np.zeros_like(K),np.zeros_like(K)]
        else:
            tf = init[:,0]
            idx = np.argsort(tf)
            idxC = np.argsort(tfC)
            tf = [tf[idx],tf[idx]]
            f = [init[:,1:4][idx], init[:,1:4][idx]]
            F = [np.zeros_like(K),np.zeros_like(K)]
        X = X + T.T
        C = C + T.T
        f[0] += T.T
        fC[0] += T.T 
        #f[0] -= f[0].mean(axis=0) - T.T
        #fC[0] -= fC[0].mean(axis=0) - T.T 
        if U is None:
            U = np.zeros(X.shape[0])


        X = X[idx]
        ORDER = idx.argsort()
        C = C[idxC]
        ORDERC = idxC.argsort()
        K = K[idxC]
        ene[0] = ene[0][idx]
        F[0] = F[0][idxC]
        I = I[idx]
        if( not isinstance(N, list) ):
            N = [N]
        if( not isinstance(W, list) ):
            W = [W]
        
    executor = None
    if(procs > X.shape[0]):
        procs = 1#X.shape[0]
    else:
        executor = Pool(processes=procs)
    tm = time.time()
    tottime = time.time()
    print("Step = ",N,"D = ",X.shape[0],"K = ",I.max()+1, "eps = ",eps,"Procs = ",procs, "Force =", FORCE_SWITCH)
    bestf = None
    besttf = None
    mindelta = np.inf
    beststep = [None,None]
    bestene = ene[0].copy()
    if(K is None):
        bestF = None
    else:
        np.zeros_like(K)
    bestorder = ORDER.copy()
    bestorderc = ORDERC.copy()
    bestK = K.copy()
    bestX = X.copy()
    bestC = C.copy()
    besttfC = tfC[0].copy()
    winner="N"
    bestL = np.inf
    memory=1
    Lmemory = np.full((memory,),-1.0)
    minmax = 0.0
    curmax = 0.0
    maxscale = 1.0#scale
    rmsd = np.full((memory,),-1.0)
    pathmem = np.empty([memory] + list(f[0].shape),dtype=np.float)
    steplimit = 1.0
    steplimitreached = False
    maxscale_start = maxscale
    maxstep_start = maxstep
    scalesteps = 1
    oldmeanFD = 0.0
    oldmeanfD = 0.0
    bestrms = [np.inf,np.inf,np.inf]
    bestpth = [np.inf,np.inf,np.inf]
    bestfre = [np.inf,np.inf,np.inf] 
    bestp = np.zeros_like(f[0])
    besttp = np.zeros_like(tf[0])
    rms = [[0,0,0],[0,0,0],[np.inf,np.inf,np.inf]] # mean, min, max for step cur,prev. last is delta
    pth = [[0,0,0],[0,0,0],[np.inf,np.inf,np.inf]]
    fre = [[0,0,0],[0,0,0],[np.inf,np.inf,np.inf]]
    converged = False
    targetL = 0.0

    def rotate(x):
        x[0] = x[1]
        return x
    def printinfoline(name, d, end="\n"):
        def P(b,a):
            if( a == 0 ):
                return b-a
            return np.nan if (a == np.nan or b == np.nan) else (b-a)/a
        #print(d) # DELETE
        print(">>> {:6s}| Mean= {: 9.6e} MeanD= {: 9.6e} Min= {: 9.6e} MinD= {: 9.6e} Max= {: 9.6e} MaxD= {: 9.6e}".format(
            name,
            d[1][0], P(d[1][0],d[0][0]),
            d[1][1], P(d[1][1],d[0][1]),
            d[1][2], P(d[1][2],d[0][2])),
            end=end)

    def stats(a,b,w=None):
        N = a.shape[0]
        if(w is None):
            w = np.ones((a.shape[0],),dtype=np.float64)/N
        axis = 1 if (len(a.shape) > 1) else 0
        d = (((b - a)**2).sum(axis=axis)**.5)
        return (d*w).sum(),d.min(),d.max()

    def delta(d):
        return [y-x if x == 0 else (y-x)/x for x,y in zip(d[0],d[1])]
    dis = euc(f[0][0],f[0][1])
    savexyz(X,"data.xyz",mode='w')
    savexyz(C,"center.xyz",mode='w')
    
    Nint = None
    if(interval != None):
        Nint = int(curveEuc(f[0])/interval)
        interval = None
    else:
        Nint = f[0].shape[0]
    p = rescale(f[0],N=None,freezeends=freezeends,interval=None)
    tp = np.linspace(0.,1.,p.shape[0])
    progress_fname="progress.xyz"
    if((checkpoint is None) or (not os.path.exists(checkpoint))):
        savexyz(X,"data.xyz",mode='w')
        savexyz(f[0]*1.5/dis,progress_fname,mode='w')
        savexyz(f[0],progress_fname,mode='a')
        savexyz(f[0]*1.5/dis,"f.xyz",mode='w')
        savexyz(f[0],"f.xyz",mode='a')
        savexyz(f[0]*1.5/dis,"best.xyz",mode='w')
        savexyz(f[0],"best.xyz",mode='a')
    else:
        savexyz(f[0],progress_fname,mode='a')
        savexyz(f[0],"f.xyz",mode='a')
        savexyz(f[0],"best.xyz",mode='a')
        savexyz(X,"data.xyz",mode='a')
    if(checkpoint is None):
        checkpoint="checkpoint.npz"
    
    drop_state = None
    pathmem[0][:] = f[0]
    memory_idx = 1
    pathmem_N = 1
    # MAIN LOOP
    firstbad = True
    rmsd[:] = -1.0
    needmbar = mbar[0] > 0 or mbar[1] >= 0
    bestp = p.copy()
    besttp = tp.copy()
    for w,n,scale in zip(W,N,scale_list):
        if(not (w > 0.0 and w < 1.0)):
            print("ERROR: span =",w,"is not acceptable")
            continue
        force_mbar = False
        drop_found = False
        deadend=False
        maxscale = scale
        maxscale_start = scale
        scalelow_param = eps
        scalelow = scalelow_param
        scalehigh = maxscale
        scalehigh_param = maxscale
        scale_argmax = False
        scale = scalehigh_param 
        if(calc_init):
            needmbar = mbar[0] > 0 or mbar[1] >= 0
            calc_init = False
            infoprint("\rCalculating RMS...              ",end="\n")
            bestrms = [np.inf,np.inf,np.inf]
            rms = [[0,0,0],[0,0,0],[np.inf,np.inf,np.inf]] # mean, min, max for step cur,prev. last is delta
            pth = [[0,0,0],[0,0,0],[np.inf,np.inf,np.inf]]
            fre = [[0,0,0],[0,0,0],[np.inf,np.inf,np.inf]]
            rms[1] = stats(X,    f[0], w=ene[0])
            rms[2] = delta(rms)
            if(rms[1][0] < bestrms[0]):
            #if(False):
                bestrms[0] = rms[1][0]
                bestrms[1] = rms[1][1]
                bestrms[2] = rms[1][2]
                bestf = f[0].copy()
                bestp = f[0].copy()
                besttp = tf[0].copy()
                bestF = F[0].copy()
                besttf = tf[0].copy()
                bestene = ene[0].copy()
                beststep = [w,0,0]
                bestL = curveEuc(f[0],0,f[0].shape[0])
                bestorder = ORDER.copy()
                bestorderc = ORDERC.copy()
                bestK = K.copy()
                bestX = X.copy()
                bestC = C.copy()
                besttfC = tfC[0].copy()
            L = curveEuc(f[0],0,f[0].shape[0])
            if(savechk):
                np.savez(checkpoint, ORDER=ORDER, ORDERC=ORDERC, X=X, C=C, K=K, I=I, E=ene[0], F=F[0],
                        tf=tf[0], f=f[0], tfC=tfC[0], fC=fC[0],
                        W=W,N=N,L=L,ii=ii-1,scale_list=scale_list,p=p,tp=tp,
                        use_ene_indices=use_ene_indices) 
            if True:
                rms = rotate(rms)
        winner = '!'
        print("\rSpan {: 5.2f} Step {:4d} {:1s} Scale {: 10.8e} StepMax {: 4.2f} L {: 10.8e}".format(
            w*100.,ii, winner, scale, maxstep, L) ,end="\n")
        printinfoline("RMS",rms)
        print()
        sys.stdout.flush()
        jj = 0
        bestjj = 0
        for i in range(n):
            if i == 0:
                adaptive = False
            else:
                adaptive = True
            #infoprint("\rSaving new iteration...               ",end="")
            #L = curveEuc(f[0],0,f[0].shape[0])
            #ret = np.insert(f[0],0,tf[0],axis=1)
            #ret = np.append(ret,X,axis=1)
            #dat = [[ret,L,ene[0],C,F[0],I]]
            #out = {}
            # for a,b in zip(W,dat):
            #         out["w"+str(a)] = b
            # np.savez("progress.npz",**out)
#            savexyz(p,"progress_"+str(ii) + ".xyz",mode='w')
            
            infoprint("\rProjecting...               ",end="")
            tf[1],f[1],L = project(p, X, exe=executor, procs=procs, eps=1e-14)
            
            X0 = X.copy()
            I0 = I.copy()
            U0 = U.copy()
            ORDER0 = ORDER.copy()
            idx = np.argsort(tf[1])
            X = X[idx].copy()
            ORDER = idx.argsort()[ORDER]
            I = I[idx]
            U = U[idx]
            tf[1] = tf[1][idx]
            f[1] = f[1][idx]
            #Lmemory[ii % memory] = L
            #avgL = Lmemory[Lmemory > 0].mean()
            # tf gives the distance along path, so since the new tf things can
            # swapped, put it back in order.
            
            tfC[1],fC[1],_ = project(p, C, exe=executor, procs=procs)
            idxC = np.argsort(tfC[1])
            tfC[1] = tfC[1][idxC]
            fC[1] = fC[1][idxC]
            C0 = C.copy()
            K0 = K.copy()
            ORDERC0 = ORDERC.copy()
            C = C[idxC]
            K = K[idxC]
            ORDERC = idxC.argsort()[ORDERC]

            ene_updated = False
            conv_ene = False if mbar[1] >= 0 else True
            case1 = (mbar[1] == 0 and converged)
            case2 = (mbar[0] == (ii+1))
            case3 = (mbar[1] > 0 and (mbar[0] > ii and 
                        ((ii - mbar[0] +1 ) % mbar[1] == 0) ))
            case4 = (deadend and not conv_ene)
            case5 = force_mbar and (case3 or mbar[1] == 0)
            dombar = case1 or case2 or case3 or case4 or case5
            if(dombar):
                needmbar = False
                force_mbar = False
                infoprint("\rMBAR estimation of energies...        ",end="")
                _U = None
                if(len(use_ene_indices) == 0):
                    _U = np.zeros_like(tf[0])
                else:
                    _U = U[:, use_ene_indices]
                    if(len(_U.shape) > 1):
                        _U = _U.sum(axis=1)
                ene[1],F[1],MBAR = calc_W_mat(tf[1],f[1],X,I,
                    tfC[1],fC[1],C,K,L,U=_U-_U.mean(),F=F[0], 
                    FORCE_SWITCH=FORCE_SWITCH)
                #fre[1] = stats(F[0], F[1])
                fre[1] = [F[1].mean(), F[1].min(), F[1].max()]
                fre[2] = delta(fre)
                ene_updated=True
                scalesteps=1
                scale=0
                scalelow = scalelow_param
                scalehigh = scalehigh_param
                #scale = maxscale * float(n - ii) / n
                # rms[2] = np.inf
                # if(deadend):
                #     scalehigh = scalehigh_param
                #     scalelow = scalelow_param
                #     scale = (scalehigh - scalelow) / 2.0
                #     deadend = False
                if(np.abs(fre[2][0]) < eps_ene or mbar[1] < 0):
                    conv_ene = True
                    pathmem_N = 0
                    memory_idx = 0
            else:
                ene[0],F[0] = ene[1],F[1]
            #print(conv_ene)
            #print(ene[1])
            #print(F[1])
            # find the expected point on the curve given the points that project
            # there
            p1 = p.copy()
            kT = .001987*310

            scale_used = scale
            # tf has the modified projections on the line, compared to p
            # now need to move p
            infoprint("\rUpdating curve..       ",end="")
            p      = update(tf[1], f[1], X, (ene[1]),tp,p,w,
                            scale=scale, maxstep=maxstep, targetL=L,
                            freezeends=freezeends, exe=executor, procs=procs)
            #p = f[1].copy()
            #tp = tf[1].copy()
            scalesteps += 1
            #if(scalesteps % 200 == 0):
            #    scale *= .9
            #f[1][:] = p

            if False:
                tf_tmp,f_tmp,L = project(f[1], p, exe=executor, procs=procs, eps=1e-14)
                idx = np.argsort(tf_tmp)
                X = X[idx]
                I = I[idx]
                U = U[idx]
                ORDER = ORDER[idx]
                tf[1] = tf_tmp[idx]
                f[1] = f[1][idx]

            # done! check if we are stationary and get timing
            infoprint("\rPreparing next step...              ",end="")
            
    
            # timing
            tm2 = time.time()
            tmd = tm2 - tm
            d = timedelta(seconds=tmd)
            tm = tm2
            timestr = str(d)
            
            # save if best
            winner = "?"
            newrms = stats(X,  p, w=ene[1])
            rms[1] = stats(X,  p, w=ene[1])
            pth[1] = stats(p1, p, w=ene[1])
            rms[2] = delta(rms)
            pth[2] = delta(pth)
            # fre[2] = delta(fre)
            #infoprint("\n" + str(rms))
            ii += 1
            savexyz(p,progress_fname)
            savexyz(p,'p.xyz')
            #savexyz(X,"data.xyz")
            savexyz(f[1],"f.xyz")
            doswap = False
            jj += 1
            if(dombar):
                beststep = [w,ii,rms[1][0]]
                winner = 'E'
                scalehigh = scalehigh_param
                scalelow = scalelow_param
                scale = scalehigh
                converged = False
                deadend = False
                drop_found = False
                jj = 0
                if(conv_ene):
                    if savechk:
                        try:
                            os.remove(os.path.splitext(checkpoint)[0]+".mbar.pickle")
                        except FileNotFoundError:
                            pass
                        try:
                            np.savez(checkpoint, ORDER=ORDER, ORDERC=ORDERC, X=X, C=C, K=K, I=I,
                                    E=ene[1], U=U, F=F[1],
                                    tf=tf[1], f=f[1], tfC=tfC[0], fC=fC[0],
                                    W=W,N=N,L=L,ii=ii-1,scale_list=scale_list, p=p, tp=tp, 
                                    mbar_pickle=pickle.dumps(MBAR),
                                    FORCE_SWITCH=FORCE_SWITCH,
                                    use_ene_indices=use_ene_indices)
                        except MemoryError:
                            print("MemoryError! Saving mbar data into separate file")
                            with open(os.path.splitext(checkpoint)[0]+".mbar.pickle",
                                    'wb') as pfile:
                                pickle.dump(MBAR, pfile)
                            if(os.path.exists(checkpoint)):
                                os.remove(checkpoint)
                            np.savez(checkpoint, ORDER=ORDER, ORDERC=ORDERC, X=X, C=C, K=K, I=I,
                                    E=ene[1], U=U, F=F[1],
                                    tf=tf[1], f=f[1], tfC=tfC[0], fC=fC[0],
                                    W=W,N=N,L=L,ii=ii-1,scale_list=scale_list, p=p, tp=tp, 
                                    FORCE_SWITCH=FORCE_SWITCH,
                                    use_ene_indices=use_ene_indices)
                    rmsd[ii%memory] = np.abs(rms[2][0])
                    if(np.abs(rms[2][0]) < eps):
                        converged = True
                    #elif((rmsd > 0).any()):
                    #    if(np.abs(rmsd[rmsd > 0.0]).mean() < eps):
                    #        converged = True
                #else:
                    # want to set this rms from reeval to current
                    #bestjj = jj
                    #bestrms = rms[1]
                    #bestfre = fre[1]
                    #bestpth = pth[1]
                    #bestp = p.copy()
                    #besttp = tp.copy()
                    #bestf = f[1].copy()
                    #besttf = tf[1].copy()
                    #bestene = ene[1].copy()
                    #bestF = F[1].copy()
                    #beststep = [w,ii,bestrms[0]]
                    #bestorder = ORDER.copy()
                    #bestorderc = ORDERC.copy()
                    #bestK = K.copy()
                    #bestX = X.copy()
                    #bestC = C.copy()
                    #besttfC = tfC.copy()
                    #bestL = L
                doswap = True
                bestrms = [np.inf,np.inf,np.inf]
                bestpth = [np.inf,np.inf,np.inf]
                bestfre = [np.inf,np.inf,np.inf] 
            elif( adaptive and rms[2][0] >= 0):
                #print("Case 1: raised target")
                # we increased, so reduce the scale
                winner = 'X'
                scalehigh = scale
                scale = scale - (scalehigh -scalelow)/2.0
                converged = False

                p = p1.copy()
                X = X0.copy()
                U = U0.copy()
                I = I0.copy()
                K = K0.copy()
                C = C0.copy()
                ORDER = ORDER0.copy()
                ORDERC = ORDERC0.copy()

                if scalehigh == scalelow or abs(scale-scalelow_param) < eps or np.abs(rms[2][0]) < eps:
                    deadend=True
                    winner = 'D'
                    if(conv_ene):
                        converged = True
                    else:
                        if(needmbar):
                            conv_ene = False
                            force_mbar = True
                        elif(np.abs(fre[2][0]) < eps_ene or mbar[1] < 0):
                            conv_ene = True
                            force_mbar = False
                        else:
                            conv_ene = False
                            force_mbar = True
            if((not adaptive) or ((not dombar) and (rms[2][0] < 0 or (deadend and drop_found)))):
                if((not adaptive) or rms[2][0] < 0):
                    scalelow = scale
                    scale = scale + (scalehigh - scalelow)/2.0
                    drop_found = True
                    deadend = False
                    winner = '-'

                    if((not adaptive) or (rms[1][0] < bestrms[0])):
                        winner = '+'
                        #  this is the best point on the search, keep it.
                        bestjj = jj
                        bestrms = rms[1]
                        bestfre = fre[1]
                        bestpth = pth[1]
                        bestp = p.copy()
                        besttp = tp.copy()
                        bestf = f[1].copy()
                        besttf = tf[1].copy()
                        bestene = ene[1].copy()
                        bestF = F[1].copy()
                        beststep = [w,ii,bestrms[0]]
                        bestorder = ORDER.copy()
                        bestorderc = ORDERC.copy()
                        bestK = K.copy()
                        bestX = X.copy()
                        bestC = C.copy()
                        besttfC = tfC[1].copy()
                        bestL = L
                        #print("SAVED")
                        #savexyz(p,"p.xyz",mode='w')
                    
                if(adaptive and ((not deadend) or (not drop_found)) and
                    abs(scalehigh - scale) >= scalelow_param):
                    # we found a low point, and can increase the scale to
                    # perhaps find a larger displacement

                    # print("Case 2: lowered target with more room to wiggle",scalehigh, scale, scalehigh-scale)
                    converged = False
                    p = p1.copy()
                    X = X0.copy()
                    U = U0.copy()
                    I = I0.copy()
                    K = K0.copy()
                    C = C0.copy()
                    ORDER = ORDER0.copy()
                    ORDERC = ORDERC0.copy()
                else:
                    # we found a low point, and cannot increase the scale
                    # 
                    #   We already determined the best, so now set it to that,
                    #+ and clear the counters to start a new macro.
                    #
                    #   Also save the state to disk since it is part of the
                    #+ path
                    # print("Case 3: lowered target and nowhere to wiggle." +
                    #    " Best jj is", bestjj)
                    infoprint("\rSaving new best iteration...            ",end="\r")

                    scalelow = scalelow_param
                    scalehigh = scalehigh_param
                    scale = scalehigh_param
                    
                    jj = bestjj
                    rms[1] = bestrms
                    pth[1] = bestpth
                    fre[1] = bestfre
                    f[1] = bestf.copy()
                    tf[1] = besttf.copy()
                    p = bestp.copy()
                    tp = besttp.copy()
                    ene[1] = bestene.copy()
                    F[1] = bestF.copy()
                    ORDER = bestorder.copy()
                    ORDERC = bestorderc.copy()
                    K = bestK.copy()
                    X = bestX.copy()
                    C = bestC.copy()
                    tfC[1] = tfC.copy()
                    L = bestL 
                    rms[2] = delta(rms)
                    pth[2] = delta(pth)
                    fre[2] = delta(fre)

                    winner = 'Y'
                    if(mbar[1] == 0):
                        needmbar = True
                    #print("SAVED")
                    savexyz(p,"best.xyz",mode='a')
                    savexyz(p,"p.xyz",mode='w')
                    if(savechk):
                        try:
                            os.remove(os.path.splitext(checkpoint)[0]+".mbar.pickle")
                        except FileNotFoundError:
                            pass
                        try:
                            np.savez(checkpoint, ORDER=ORDER, ORDERC=ORDERC, X=X, C=C, K=K, I=I,
                                    E=ene[1], U=U, F=F[1],
                                    tf=tf[1], f=f[1], tfC=tfC[1], fC=fC[1],
                                    W=W,N=N,L=L,ii=ii-1,scale_list=scale_list, p=p, tp=tp, 
                                    mbar_pickle=pickle.dumps(MBAR),
                                    FORCE_SWITCH=FORCE_SWITCH,
                                    use_ene_indices=use_ene_indices)
                        except MemoryError:
                            with open(os.path.splitext(checkpoint)[0]+".mbar.pickle",
                                    'wb') as pfile:
                                pickle.dump(MBAR, pfile)
                            if(os.path.exists(checkpoint)):
                                os.remove(checkpoint)
                            np.savez(checkpoint, ORDER=ORDER, ORDERC=ORDERC, X=X, C=C, K=K, I=I,
                                    E=ene[1], U=U, F=F[1],
                                    tf=tf[1], f=f[1], tfC=tfC[1], fC=fC[1],
                                    W=W,N=N,L=L,ii=ii-1,scale_list=scale_list, p=p, tp=tp, 
                                    FORCE_SWITCH=FORCE_SWITCH,
                                    use_ene_indices=use_ene_indices)
                        except RecursionError:
                            print("RECURSION ERROR")
                    
                    #pathmem[memory_idx % memory][:] = p
                    #pathmem_N += 1
                    #memory_idx += 1
                    #if(pathmem_N > memory):
                    #    pathmem_N = memory
                    #p = pathmem[:pathmem_N].mean(axis=0)
                    
                    rmsd[ii%memory] = np.abs(rms[2][0])
                    converged = False
                    if(np.abs(rms[2][0]) < eps):
                        converged = True
                    #elif((rmsd > 0).any()):
                    #    if(np.abs(rmsd[rmsd > 0.0]).mean() < eps):
                    #        converged = True
                    doswap = True
                    deadend = False
                    drop_found = False
                    if(converged):
                        conv_ene = False if mbar[1] >= 0 else True
            else:
                # winner is still ? so just bail
                #converged = True
                #deadend = True
                pass
            #print(rms)
            # 
            if(converged):
                print("\rSpan {: 5.2f} Step {:4d} Search {:4d} {:1s} Scale {: 10.8e} StepMax {: 4.2e} L {: 10.8e} Time {:16s} {:s}".format(
                            w*100.,ii, jj, winner, scale_used, maxstep, L, timestr, 
                            "**** CONVERGED") ,end="\n")
                printinfoline("RMS", rms)
                printinfoline("PTH", pth)
                printinfoline("ENE*" if ene_updated else "ENE", fre)
                print()
                sys.stdout.flush()
                rmsd[:] = -1.0
                pathmem_N = 0
                if(conv_ene):
                    break
                bestrms = [np.inf,np.inf,np.inf]
                bestpth = [np.inf,np.inf,np.inf]
                bestfre = [np.inf,np.inf,np.inf] 
            else:
                print("\rSpan {: 5.2f} Step {:4d} Search {:4d} {:1s} Scale {: 10.8e} StepMax {: 4.2e} L {: 10.8e} Time {:16s}".format(
                            w*100.,ii, jj, winner, scale_used, maxstep, L, timestr) ,end="\n")
                printinfoline("RMS", rms)
                printinfoline("PTH", pth)
                printinfoline("ENE*" if ene_updated else "ENE", fre)
                print()
                sys.stdout.flush()

            # rotate data to start new iter
            dombar = False
            if(converged and conv_ene):
                print("Convergence achieved.")
                break
            if(deadend and not drop_found and not force_mbar):
                if(jj > 1):
                    print("Converged, but not at a minimum.")
                else:
                    print("Convergence achieved.")
                break

            # doswap means we want to compare to this new step (make current the ref)
            if doswap:
                deadend = False
                drop_found = False
                jj = 0
                (rms,pth,f,tf,fC,tfC) = map(rotate, (rms,pth,f,tf,fC,tfC))
                (ene,F,fre) = map(rotate, (ene,F,fre))
            


        
    print("Best fit found using span =",beststep[0],"on step",beststep[1],", RMSD =",beststep[2])
    print("Total elapsed: " + str(timedelta(seconds=tm-tottime)))
    s = np.argsort(besttp)
    bestp = bestp[s]
    X = X[s]
    #bestp = clip(bestp)
    tf ,f,L = project(bestp, X, exe=executor, procs=procs, eps=1e-14)
    if(executor is not None):
        executor.close()
        executor.join()
    return tf, f, X
    return tf[1][ORDER], f[1][ORDER],X[ORDER]
    return besttf[bestorder], bestf[bestorder],X[bestorder]
    #return besttf[ORDER], bestf[ORDER], X[ORDER]

def curveLen(f):
    L = 0.0
    pt = f[0]
    for j in f[1:]:
        L += euc(pt,j)
        pt = j
    return L

def curveEuc(f,i=0,j=-1):
    s = i
    e = j
    if(i > j):
        s = j
        e = i
    if( e >= f.shape[0] or e < 0):
        e = f.shape[0] - 1
#    print("START IS ",s,"END IS",e)
    return np.sum([euc(f[k-1],f[k]) for k in range(s+1,e+1)])

def euc(r1,r2):
    return (((r2 - r1)**2).sum())**.5

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,a,b],
                        [0,0,-1e-5,zback]])

