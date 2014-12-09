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
        self.hbs = wexplore.HierarchicalBinSpace(nregions=[5, 5, 5])

        # Array to hold bin centers
        self.bin_centers = [[-3.0, 0.0]]
        self.hbs.add_bin(None, 0)

        self.max_replicas = 100
        self.bin_target_counts = self.hbs.balance_replicas(self.max_replicas,
                np.array([2,], np.int_))

        self.bin_mapper = FuncBinMapper(wexplor, 1, 


