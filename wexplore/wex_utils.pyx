import numpy
cimport numpy
import cython

from numpy import uint16, float32
from numpy cimport uint16_t, float32_t

ctypedef numpy.uint8_t bool_t
ctypedef float32_t coord_t
ctypedef uint16_t index_t

coord_dtype = numpy.float32

@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef apply_down_argmin_across(func,
                               args,
                               kwargs,
                               func_output_len,
                               numpy.ndarray[coord_t, ndim=2] coords,
                               numpy.ndarray[bool_t, ndim=1, cast=True] mask,
                               index_t[:] output,
                               coord_t[:] max_dist):
    '''Apply func(coord, *args, **kwargs) to each input coordinate tuple,
    skipping any for which mask is false and writing results to output.'''
    cdef:
        Py_ssize_t icoord, iout, ncoord, nout,
        coord_t _min, _max
        index_t _argmin
        numpy.ndarray[coord_t, ndim=1] func_output
    
    nout = func_output_len
    func_output = numpy.empty((func_output_len,), dtype=coord_dtype)

    ncoord = len(coords)
    for icoord from 0 <= icoord < ncoord:
        if mask[icoord]:
            func_output = func(coords[icoord], *args, **kwargs)
            if len(func_output) != func_output_len:
                raise TypeError('function returned a vector of length {} (expected length {})'
                                .format(len(func_output), func_output_len))

            # find minimum and maximum values
            _min = func_output[0]
            _max = func_output[0]
            _argmin = 0
            for iout from 1 <= iout < nout:
                if func_output[iout] < _min:
                    _min = func_output[iout]
                    _argmin = iout

                if func_output[iout] > _max:
                    _max = func_output[iout]

            output[icoord] = _argmin
            max_dist[icoord] = _max

