from __future__ import division, print_function
import time
import os
import numpy as np

import west
from west.propagators import WESTPropagator
from west import Segment, WESTSystem
from westpa.binning import FuncBinMapper

import modelsim
import wexplore

import logging
log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

pcoord_dtype = np.float32


class SimpleLangevinPropagator(WESTPropagator):

    def __init__(self, rc=None):
        super(SimpleLangevinPropagator, self).__init__(rc)

        modelsim.init_rng(np.random.randint(2**32 - 1))

    def get_pcoord(self, state):
        pcoord = None
        if state.label == 'initA':
            pcoord = [-3.0, 0.0]
        elif state.label == 'initI1':
            pcoord = [0.0, 3.0]
        elif state.label == 'initB':
            pcoord = [3.0, 0.0]
        elif state.label == 'initI2':
            pcoord = [0.0, -3.0]

        state.pcoord = pcoord

    def propagate(self, segments):
        modelsim.propagate_segments(segments, 10)

        return segments


class System(WESTSystem):

    def initialize(self):
        self.pcoord_ndim = 2
        self.pcoord_len = 2
        self.pcoord_dtype = pcoord_dtype

        # Set up hierarchical bin space
        self.bin_mapper = wexplore.WExploreBinMapper(n_regions=[4, 6, 18],
                d_cut=[4.0, 1.5, 0.3], dfunc=modelsim.dfunc)
        

        # Array to hold bin centers
        self.bin_mapper.centers = [[-3.0, 0.0]]
        self.bin_mapper.add_bin(None, 0)

        # Need to define the max number of replicas
        self.max_replicas = 600
        self.bin_target_counts = self.bin_mapper.balance_replicas(self.max_replicas,
                np.array([0,], np.int_))



