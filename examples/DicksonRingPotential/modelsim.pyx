#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, atan2, sin


ctypedef np.float64_t f8
ctypedef np.float32_t f4
ctypedef np.int64_t i8
ctypedef np.int32_t i4

cdef extern from "randomkit.h": 
    ctypedef struct rk_state: 
        unsigned long key[624] 
        int pos 
        int has_gauss 
        double gauss 
    void rk_seed(unsigned long seed, rk_state *state) 
    double rk_gauss(rk_state *state)

cdef rk_state *rng_state = NULL

# Model parameters
DEF alpha = 3.0
DEF gamma = 3.0
DEF chi1 = 2.25
DEF chi2 = 4.5
DEF mass = 1.0
DEF xi = 1.5
DEF dt = 0.05
DEF beta = 1.0

def dfunc(f4[:] p, f4[:,::1] centers):
    cdef unsigned int k
    cdef int ncenters = centers.shape[0]
    cdef np.ndarray[f4, ndim=1] d = np.empty((ncenters,), dtype=np.float32)

    for k in xrange(ncenters):
        d[k] = sqrt((p[0] - centers[k,0])**2 + (p[1] - centers[k,1])**2)

    return d


cdef void propagate(f4[:,::1] coord, i8 nsteps):
    cdef:
        unsigned int k
        double x, y, r, invr, fx, fy
        double A, att, B1, B2
        double sigma, fp

    x = coord[0,0]
    y = coord[0,1]

    sigma = sqrt(2.0*dt/(mass*beta*xi))
    fp = dt/(mass*xi)


    for k in xrange(nsteps):
        r = sqrt(x*x + y*y)
        invr = 1.0 / r
        A = 2.0 * alpha * (1.0 - gamma*invr)
        att = atan2(y, x)

        B1 = chi1 * sin(2.0*att)*invr*invr
        B2 = chi2 * sin(4.0*att)*invr*invr

        fx = -1.0*(x*A - 2.0*y*B1 - 4.0*y*B2)
        fy = -1.0*(y*A - 2.0*x*B1 + 4.0*x*B2)

        x = x + fp*fx + sigma*rk_gauss(rng_state)
        y = y + fp*fy + sigma*rk_gauss(rng_state)

    coord[1,0] = x
    coord[1,1] = y


def propagate_segments(object segments, i8 nsteps):
    cdef:
        unsigned int k, nsegs
        object segment
        np.ndarray[f4, ndim=2] coords = np.empty((2,2), dtype=np.float32)

    nsegs = len(segments)

    for k in xrange(nsegs):
        segment = segments[k]

        # Initial pcoord
        coords[0,0] = segment.pcoord[0,0]
        coords[0,1] = segment.pcoord[0,1]

        # Propagate
        propagate(coords, nsteps)

        segment.pcoord[:] = coords[:]
        segment.status = segment.SEG_STATUS_COMPLETE


def init_rng(unsigned long seed):
    global rng_state
    if rng_state == NULL:
        rng_state = <rk_state*>malloc(sizeof(rk_state)) 
    rk_seed(seed, rng_state)

def free_rng():
    global rng_state
    free(rng_state)

