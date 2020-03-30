from __future__ import print_function
import numpy as np
cimport numpy as cnp
#from numpy.math cimport INFINITY
#from libc.math cimport abs as cabs
#from cython.view cimport array as cvarray
#from libc.stdlib cimport abort, malloc, free
#cimport cython.parallel as cpar
cimport cython
from libc.math cimport sqrt
from libcpp cimport bool
from libc.stdio cimport printf
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort as stdsort

# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

from libc.stdlib cimport malloc, free
 
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct IndexedElement:
    cnp.ulong_t index
    cnp.float64_t value

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _compare(const_void *a, const_void *b) nogil:
    cdef cnp.float64_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return -1
    if v >= 0: return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void argsort(vector[double] data, int* order, int n) nogil:
    cdef cnp.ulong_t i
    
    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
    
    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for i in range(n):
        order[i] = order_struct[i].index
        
    # Free index tracking array.
    free(order_struct)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double cabs(double x) nogil:
    if(x < 0.0):
        return -x
    return x
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int cimax(int x, int y) nogil:
    if(x > y):
        return x
    return y
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double cfmax(double x, double y) nogil:
    if(x > y):
        return x
    return y
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int cimin(int x, int y) nogil:
    if(x < y):
        return x
    return y

cdef inline bool less(double x, double y):
        return x < y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_curve(
        double[::1]   tf, 
        double         w, 
        double[:,::1]  X, 
        double[::1]    E, 
        double[:,::1]  f, 
        double[::1]   tp, 
        double[:,::1]  p, 
        cnp.intp_t     j,
        double     scale, 
        double     maxstep, 
        double[:] ret) nogil:
    cdef double tfshape = tf.shape[0]
    cdef int n_dims = f.shape[1]
    cdef int keep = (int)(tfshape*w)
    if(keep < 1):
        keep = 1
    cdef bint left = False
    cdef bint right = False
    cdef double dl = 0.0
    cdef double dr = 0.0
#    idx = [j]
    cdef cnp.intp_t i = 0, k
    cdef int kept = 0
#    print()
    if(j == tp.shape[0]-1):
        right = True
    if(j == 0):
        left = True
#    cdef vector[double] nbp
#    cdef vector[double] nbw
    cdef vector[double] nbl
    cdef double D = 0.0
    cdef cnp.intp_t mini = i
    cdef cnp.intp_t maxi = i
    cdef cnp.intp_t ji
    cdef double L = 0.0
    cdef double norm = 0.0
    cdef double weight = 0.0
    cdef double step = 0.0
    cdef cnp.intp_t jj = j
    ret[:] = 0.0
    ret[0] = j
    cdef double reft = tp[j]
    # need to set j to the closest index tf to tp[j]
    # binary search to find closest index
    cdef cnp.intp_t bi, bmini, bmaxi 
    bmini = 0
    bmaxi = tf.shape[0]
    while bmini != bmaxi:
        bi = (bmaxi + bmini) / 2
        if(bmaxi - bmini == 1):
            if(2*tf[bi] < tf[bmini]+tf[bmaxi]):
                bi = bmini
            else:
                bi = bmaxi
            break
        if(tf[bi] < reft):
            bmini = bi
        elif(tf[bi] > reft):
            bmaxi = bi
        else:
            j = bi
            break

    while(not (left and right)):
        ji = j + i
        l = cabs(reft - tf[ji])
        L = 0.0
        #printf("l is %f\n", l)
        if(l > 0.0):
            if (i >= 0):
                if(kept >= keep):
                    right = True
                    i = -i - 1
                    #printf("j=%d right=True\n", jj)
                    continue
                dr += l
                L = dr
            else:
                if(kept >= keep):
                    left = True
                    i = -i
                    #printf("j=%d left=True\n", jj)
                    continue
                dl += l
                L = dl
            #printf("j=%d dl=%f dr=%f L=%f\n",jj,dl,dr,L)
        D = cfmax(L,D)
        kept += 1
        nbl.push_back(L)
        #printf("j=%d pushed %f, D=%f\n", jj, L, D)
        
        mini = cimin(mini,i)
        maxi = cimax(maxi,i)
        
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
    #printf("j=%d D= %f\n",jj,D)
    
    if(D > 0.0):
        for i in range(j+mini,j+maxi+1):
            weight = nbl[i-(j+mini)]/D
            weight = 1.0 - weight * weight * weight
            weight = weight * weight * weight
            weight = weight * E[i]
            #printf("j=%d i=%d E= %f weight= %f\n", jj, i, E[i], weight)
            nbl[i-(j+mini)] = weight
    else:
        # All points collected project exactly on this point -> regular average
        for i in range(j+mini,j+maxi+1):
            weight = E[i]
            nbl[i-(j+mini)] = weight


    norm = 0.0
    #stdsort(nbl.begin(), nbl.end(), less)
    # need to argsort nbl, then argsort X
    cdef int* order = <int*>malloc(nbl.size()*sizeof(int))
    argsort(nbl, order, nbl.size())
    for i in range(nbl.size()):
        norm += nbl[order[i]]
    #printf("j=%d NORM= %f\n",jj,norm)

    
    for i in range(nbl.size()):
        for k in range(n_dims):
            ret[k+1] += (X[j+mini + order[i]][k] * nbl[order[i]])/norm
            #printf("j=%d k=%d adding= %f\n",jj, k, (X[j+mini + order[i]][k] * nbl[order[i]])/norm)
    free(order)
    #printf("j=%d NPT= %f %f %f\n",jj,ret[1], ret[2], ret[3])
    
    #for i in range(nbl.size()):
    #    norm += nbl[i]
    #
    #for i in range(j+mini,j+maxi+1):
    #    for k in range(3):
    #        ret[k+1] += (X[i][k] * nbl[i-j-mini])/norm

    step = 0.0
    norm = 0.0
    for k in range(n_dims):
        step = (ret[k+1] - p[jj][k]) # changed p to f
        norm += step * step
    norm = scale*sqrt(norm) ## this is the distance between pts
    if(norm > maxstep):
        scale = maxstep/norm
#    scale *= scale
    for k in range(n_dims):
        ret[k+1] = p[jj][k] + scale*(ret[k+1] - p[jj][k])
    #printf("j=%d PT= %f %f %f\n",jj,ret[1], ret[2], ret[3])

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef void update_curve_old(
#        double[::1]   tf, 
#        double         w, 
#        double[:,::1]  X, 
#        double[::1]    E, 
#        double[:,::1]  f, 
#        double[::1]   tp, 
#        double[:,::1]  p, 
#        cnp.intp_t     j,
#        double     scale, 
#        double     maxstep, 
#        double[:] ret) nogil:
#    cdef double tfshape = tf.shape[0]
#    cdef int keep = (int)(tfshape*w)
#    if(keep < 1):
#        keep = 1
#    cdef bint left = False
#    cdef bint right = False
#    cdef double dl = 0.0
#    cdef double dr = 0.0
##    idx = [j]
#    cdef cnp.intp_t i = 0, k
#    cdef int kept = 0
##    print()
#    if(j == tf.shape[0]-1):
#        right = True
#    if(j == 0):
#        left = True
##    cdef vector[double] nbp
##    cdef vector[double] nbw
#    cdef vector[double] nbl
#    cdef vector[double] nbl_dis
#    cdef double D = 0.0
#    cdef cnp.intp_t mini = i
#    cdef cnp.intp_t maxi = i
#    cdef cnp.intp_t ji
#    cdef double L = 0
#    cdef double norm = 0.0
#    cdef double weight = 0.0
#    cdef double step = 0.0
#    ret[0] = j
#    ret[1] = 0.0
#    ret[2] = 0.0
#    ret[3] = 0.0
#    while(not (left and right)):
#        ji = j + i
#        l = cabs(tf[j] - tf[ji])
#        L = 0.0
#        if(l > 0.0):
#            if (i >= 0):
#                if(kept >= keep):
#                    right = True
#                    i = -i - 1
#                    continue
#                dr += l
#                L = dr
#            else:
#                if(kept >= keep):
#                    left = True
#                    i = -i
#                    continue
#                dl += l
#                L = dl
#        D = cfmax(L,D)
#        kept += 1
#        nbl.push_back(L)
#        
#        mini = cimin(mini,i)
#        maxi = cimax(maxi,i)
#        
#        if (left == True):
#            i += 1
#            if(j+i == tf.shape[0]):
#                right = True
#        elif (right == True):
#            i -= 1
#            if(j+i < 0):
#                left = True
#        else:
#            if(ji == tf.shape[0]-1):
#                right = True
#            elif(ji == 0):
#                left = True
#            if (i >= 0):
#                i = -i - 1
#            else:
#                i = -i
#    
#    if(D > 0.0):
#        for i in range(j+mini,j+maxi+1):
#            weight = nbl[i-(j+mini)]/D
#            weight = 1.0 - weight * weight * weight
#            weight = weight * weight * weight
#            weight = weight * E[i]
#            nbl[i-(j+mini)] = weight
#    else:
#        # All points collected project exactly on this point -> regular average
#        for i in range(j+mini,j+maxi+1):
#            weight = E[i]
#            nbl[i-(j+mini)] = weight
#
#    stdsort(nbl.begin(), nbl.end(), less)
#    for i in range(nbl.size()):
#        norm += nbl[i]
#    
#    for i in range(j+mini,j+maxi+1):
#        for k in range(3):
#            ret[k+1] += (X[i][k] * nbl[i-j-mini])/norm
#
#    step = 0.0
#    norm = 0.0
#    for k in range(3):
#        step = (ret[k+1] - f[j][k])
#        norm += step * step
#    norm = sqrt(norm) ## this is the distance between pts
#    if(scale*norm > maxstep):
#        scale = maxstep/norm
##    scale *= scale
#    for k in range(3):
#        ret[k+1] = f[j][k] + scale*(ret[k+1] - f[j][k])

@cython.boundscheck(False)
@cython.wraparound(False)
def start(tf,f,X,E, double [::1] tp, double[:,::1] p,w,ID,scale=1.0,maxstep=1.0,chunk=1):
    #l,L,x,y,z = np.array((5,),np.float64)
#    ret = np.zeros((6,),np.float64)
    cdef double[::1] ctf = tf
    cdef double[:,::1] cf = f
    cdef double[:,::1] cX = X
    cdef double[::1] cE = E
    cdef int cID = ID
    cdef int C
    cdef int c
    cdef int n_dims
    cdef double cw = w
    cdef double cmaxstep = maxstep
    cdef double cscale = scale
    cdef n_samples = tf.shape[0]
    cdef float IDf 
    if(ID + chunk >= n_samples):
        chunk = n_samples - ID
    C = chunk
    if(ID + chunk + 1 > n_samples):
        IDf = 100.0
    else:
        IDf = (ID + chunk + 1)/n_samples*100.0
    
    n_dims = X.shape[1]
    ret = np.zeros((chunk,n_dims+1),np.float64)
    cdef double[:,:] cret = ret
    
    with nogil:
        for c in range(C):
            update_curve(ctf,cw,cX,cE,cf,tp,p,cID+c,cscale,cmaxstep,cret[c])

#        printf("\rExpectation %10d % 7.3f %%      ", cID + c + 1,IDf)
            
#    print("\r" + "Project     {:10d} {: 7.3f} %      ".format(
#                ID, float(ID)/X.shape[0]*100) ,end="")
#    ret[:] = cret
    return ret


# GET TO WORK
#def rescale(f,N=None,targetL=None,freezeends=False):
#    if (targetL is None):
#        targetL = curveEuc(f,0,f.shape[0])
#    if (N is None):
#        N = f.shape[0]
#    el = targetL/(N-1)
#        # redistribute points along path to unit speed (evenly spaced)
#
#    infoprint("\rReparameterizing curve...       ",end="")
#    f1 = f.copy()
#    pt=f[0]
#    j=1
#    f = np.empty([N] + list(f.shape[1:]),dtype=np.float64)
#    end = f.shape[0]
#    f[0] = f1[0]
#    if(freezeends):
#        end -= 1
#        f[-1] = f1[-1]
#    for i in range(1,end):
#        l = 0
#        k = 0
#        w = 1.0
#        while(j < f.shape[0]):
#            k = euc(pt,f1[j])
#            if(l+k > el):
#                break
#            l += k
#            pt = f1[j]
#            j += 1
#        if(j == f.shape[0]):
#            j = f.shape[0] - 1
#        if(k == 0):
#            w = 1.0
#        else:
#            w = (el-l)/k 
##        print("point",i,"go between",j-1,j,"w,l+k,el=",w,l+k,el)
##            if(el - l > k):
##                w = 1.0
##            else:
##        f[i] = f[i] + scale * ((f[j-1] + w * (f[j] - f[j-1])) - f[i])
##            w = scale
###            norm = euc(f[j-1],f2[j])
##            d = f1[j-1] +  w * (f1[j] - f1[j-1])
##            norm = euc(f[j-1],d)
##            w = 1.0
##            if(norm > maxstep):
##                w = maxstep/norm
##            w = w*w
##            f[i] = f[i] + w * d
#        pt = pt +  w * (f1[j] - pt)
#        f[i] = pt
#
#    return f
#
# need to make this for each element
# looks like we need to sync after each loop -> while loop in python
# 
#def clip(f,freezeends=False):
#    clipped = True
#    it = 0
#    f0 = f.copy()
#    while(clipped == True and it < 100):
#        clipped = False
#        it += 1
#        for i in range(1,f.shape[0]-2):
#            a = euc(f[i-1],f[i ])
#            if(a == 0.0):
#                continue
#            p = ((f[i] - f[i-1]) * (f[i+1] - f[i-1])).sum() / a
#            if(p < a):
#                f0[i] = (f[i-1] + f[i+1]) / 2.0
#                clipped = True
#        targetL = curveEuc(f0, 0, f0.shape[0])
#        p = rescale(f0, N=None, targetL=targetL, freezeends=freezeends)
#        p2 = rescale(f0[::-1], N=None, targetL=targetL, freezeends=freezeends)
#        f0 = (p + p2[::-1])/2.0
#        f = f0.copy()
#    return f
