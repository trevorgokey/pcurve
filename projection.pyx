
from __future__ import print_function
import numpy as np
cimport numpy as cnp
from numpy.math cimport INFINITY
from libc.math cimport sqrt
#from libc.math cimport abs as cabs
from cython.view cimport array as cvarray
cimport cython
#from libc.stdlib cimport abort, malloc, free
#import pyximport; pyximport.install()
from libc.stdio cimport printf
cimport cython.parallel as cpar

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cabs(double x) nogil:
    if(x < 0.0):
        return -x
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double euc(
        double[3] X,
        double[3] Y) nogil:

#    cdef int n_dims
    cdef double tmp, d
    cdef cnp.intp_t j

#    n_dims = 3
    d = 0

    for j in range(3):
        tmp = X[j] - Y[j]
        d += tmp * tmp

    return sqrt(d)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projectionIDX(
        double[:,::1] f, 
        double[:,::1] X, 
        cnp.intp_t ID,
        const double eps, 
        double[:] ret) nogil:
    
    cdef cnp.intp_t i,j, n_pts, n_dims
    cdef double L, minL, l, p, d
    cdef double minD, dl
    cdef float IDf = ID
    n_dims = 3 #f.shape[1]
    cdef double delta = 0.0
    cdef double[3] x
    cdef double[3] prevF, curF
    cdef double[3] fnew
    
    n_pts = f.shape[0]
    for i in range(n_dims):
        ret[3+i] = f[0][i]
        curF[i] = f[0][i]
        x[i] = X[ID][i]
    L = 0.0 
    minL = 0.0
    l = 0.0
    minD = euc(x,curF)
    d = 0.0
    
    ret[0] = ID
    ret[1] = minL
    ret[2] = L
    
    for j in range(1,n_pts):
        for i in range(n_dims):
            prevF[i] =  curF[i]
            curF[i]  =  f[j][i]
        d = INFINITY
        dl = 0.0
        p = 0.0

        l = euc(prevF,curF)
        if(l == 0.0):
            continue
        
        for i in range(n_dims):
            p += (x[i] - prevF[i]) * ((curF[i] - prevF[i]) /l) 
        if(p <= 0.0):
            d = euc(x,prevF)
            dl = 0.0
        else:
            # pythag to find the distance to proj point
            dl = p
            if(p < l):
                z = euc(x,prevF)
                d = z*z - p*p
                d = sqrt(d)
            else:
                d = euc(x,curF)
        if(d < minD):# and cabs(d-minD) > eps):
            minD = d
            minL = (L + dl)
            if(p <= 0.0):
                for i in range(n_dims):
                    ret[i+3] = prevF[i]
            elif (p > l):
                for i in range(n_dims):
                    ret[i+3] = curF[i]
            else:
                for i in range(n_dims):
                    ret[i+3] = prevF[i] + dl/l * (curF[i] - prevF[i])

        L += l
    ret[1] = minL
    ret[2] = L
    return


@cython.boundscheck(False)
@cython.wraparound(False)
def start(f,X,ID,chunk,eps):
    #l,L,x,y,z = np.array((5,),np.float64)
#    ret = np.zeros((6,),np.float64)
    cdef double[:,::1] cf = f
    cdef double[:,::1] cX = X
    cdef int cID = ID
    cdef int C
    cdef int c
    cdef double ceps = eps 
    cdef n_samples = X.shape[0]
    cdef double L, minL, l, p, d
    cdef float IDf 
    if(ID + chunk >= n_samples):
        chunk = n_samples - ID
    C = chunk
    IDf = (ID + chunk + 1)/n_samples*100.0
    ret = np.ones((chunk,6),np.float64)
    cdef double[:,:] cret = ret
    with nogil:
        for c in range(C):
            projectionIDX(cf,cX,cID+c,ceps,cret[c])
#            printf("%f %f %f\n", cret[c][3],cret[c][4],cret[c][5])
#        printf("\rProject     %10d % 7.3f %%      ", cID + c+ 1, IDf)
            
#    print("\r" + "Project     {:10d} {: 7.3f} %      ".format(
#                ID, float(ID)/X.shape[0]*100) ,end="")
#    ret[:] = cret
    return ret
    #return ID,ret

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef startOMP(double[:,::1] f, double[:,::1] X, double[:,::1] Xf):
#    ret = np.zeros((6,),np.float64)
    cdef cnp.intp_t N = X.shape[0]
    cdef cnp.intp_t ID
    cdef cnp.intp_t tid 
    ret = np.array((6,),np.float64)
    cdef double ceps = 1e-14
    cdef double[:] cret = ret
    cdef cnp.intp_t j
    with nogil,cpar.parallel(num_threads=24):
        tid = cpar.threadid()
        for ID in range(tid,N,24):
            projectionIDX(f,X,ID,ceps,cret)
            for j in range(6):
                Xf[ID][j] = cret[j]
#    for ID in cpar.prange(N,nogil=True):
#        projectionIDX(f,X,ID,cret)
#        for j in range(6):
#            Xf[ID][j] = cret[j]

    #l,L,x,y,z = np.array((5,),np.float64)
    #return ID,ret


#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef void projectionIDX(
#        double[:,::1] f, 
#        double[:,::1] X, 
#        cnp.intp_t ID,
#        const double eps, 
#        double[:] ret) nogil:
#    
##    cdef double eps = 1e-4
#
#    cdef cnp.intp_t i,j, n_pts, n_dims
#    cdef double L, minL, l, p, d
#    cdef double minD, dl
#    cdef float IDf = ID
#    n_dims = 3 #f.shape[1]
#    cdef double delta = 0.0
#
#    cdef double[3] x
#    cdef double[3] prevF, curF
#    cdef double[3] fnew
#    
#
#    n_pts = f.shape[0]
#    for i in range(n_dims):
#        ret[3+i] = f[0][i]
#        curF[i] = f[0][i]
#        x[i] = X[ID][i]
#    L = 0.0 
#    minL = 0.0
#    l = 0.0
#    minD = euc(x,curF)
#    d = 0.0
#    
#    ret[0] = ID
#    ret[1] = minL
#    ret[2] = L
#    
#    for j in range(1,n_pts):
#        for i in range(n_dims):
#            prevF[i] =  curF[i]
#            curF[i]  =  f[j][i]
##        printf("\nID=%zu A= %f %f %f B= %f %f %f\n",ID,
##                prevF[0],prevF[1],prevF[2],
##                curF[0],curF[1],curF[2]
##                )
#        d = INFINITY
#        dl = 0.0
#        p = 0.0
#
#        l = euc(prevF,curF)
#        if(l == 0.0):
##            printf("l is 0 when j=%4d\n",j)
#            continue
#        
#        for i in range(n_dims):
#            p += (x[i] - prevF[i]) * ((curF[i] - prevF[i]) /l) 
##        printf("j=%4d p=%8.4f\n",j,p)
##        printf("ID %d: p is %16.8f\n",ID,p)
##        if(p < -eps):
##            L += l
##            continue
##        if(cabs(p) < eps):
##            d = euc(x,prevF)
###            printf("ID=%zu case 1 d=%f\n",ID,d)
##        elif (p < l and cabs(p-l) > eps):
##            # pythag to find the distance to proj point
##            dl = p
##            z = euc(x,prevF)
##            d = z*z - p*p
##            d = sqrt(d)
###            printf("ID=%zu case 2 d=%f z=%f\n",ID,d,z)
##        elif (j == n_pts - 1):
##            dl = l
##            d = euc(x,curF)
###            if(d < minD and cabs(d-minD) > eps):
##            minD = d
##            minL = (L + dl)
###            printf("ID=%zu case 3 d=%f\n",ID,d)
##            for i in range(n_dims):
##                ret[i+3] = curF[i]
##            continue
##        elif (j > 1):
##            L += l
###            printf("ID=%zu case 4\n",ID)
##            continue
#        if(p <= 0.0):
#            d = euc(x,prevF)
#            dl = 0.0
##            printf("ID=%zu case 1 d=%f\n",ID,d)
#        else:
#            # pythag to find the distance to proj point
#            dl = p
#            if(p < l):
#                z = euc(x,prevF)
#                d = z*z - p*p
#                d = sqrt(d)
#            else:
#                d = euc(x,curF)
##            printf("ID=%zu case 2 d=%f z=%f\n",ID,d,z)
###            printf("ID=%zu case 4\n",ID)
##            continue
##        printf("ID=%zu p=%f dl=%f l=%f d/l=%f d=%f mind=%f\n",ID,p,dl,l,dl/l,d,minD)
##        if(minD - d > eps):
#        if(d < minD):# and cabs(d-minD) > eps):
##            printf("d=%8.6f when j=%4d\n",d,j)
#            minD = d
#            minL = (L + dl)
##            printf("ID=%zu j=%zu new min p=%f dl=%f l=%f dl/l=%f d=%f %16.16e\n",ID,j,p,dl,l,dl/l,d,cabs(dl-l))
##            if(cabs(l-dl) < eps or cabs(p-l) > eps):
##            if((p-l > eps and j != n_pts-1) or cabs(l-dl) < eps or dl < eps ):
##            if(cabs(dl) < eps or cabs(dl-l) < eps):
###                if(p != 0.0):
###                    printf("DISASTER\n")
###                    printf("ID=%zu j=%d p=%f dl=%f l=%f dl/l=%f d=%f\n",ID,j,p,dl,l,dl/l,d)
###                printf("ID=%zu j=%zu endpoint\n",ID,j)
##                for i in range(n_dims):
##                    ret[i+3] = prevF[i]
##            elif(cabs(dl/l - 1.0) > eps):
###                printf("DISASTER\n")
###                printf("ID=%zu j=%d p=%f dl=%f l=%f dl/l=%f d=%f\n",ID,j,p,dl,l,dl/l,d)
###                printf("ID=%zu j=%zu in middle\n",ID,j)
##                for i in range(n_dims):
##                    ret[i+3] = prevF[i] + dl/l * (curF[i] - prevF[i])
##            else:
##                for i in range(n_dims):
##                    ret[i+3] = curF[i]
#
#            if(p <= 0.0):
#                for i in range(n_dims):
#                    ret[i+3] = prevF[i]
#            elif (p > l):
#                for i in range(n_dims):
#                    ret[i+3] = curF[i]
#            else:
#                for i in range(n_dims):
#                    ret[i+3] = prevF[i] + dl/l * (curF[i] - prevF[i])
#
#        L += l
#        
##    printf("j=%d minD (d) now %f with minL %f fnew %f %f %f\n",ID,minD,minL,ret[3],ret[4],ret[5])
##    printf("\nID=%zu p=%f %f\n\n",ID,pp,minD)
#
#    ret[1] = minL
#    ret[2] = L
