from __future__ import division; __metaclass__ = type
import logging
log = logging.getLogger(__name__)

import numpy as np
import time

import westpa, west
from westpa import extloader
from westpa.yamlcfg import check_bool, ConfigItemMissing

class WExploreDriver(object):
    def __init__(self, sim_manager, plugin_config):
        super(WExploreDriver, self).__init__()

        if not sim_manager.work_manager.is_master:
                return

        self.sim_manager = sim_manager
        self.data_manager = sim_manager.data_manager
        self.system = sim_manager.system
        self.we_driver = sim_manager.we_driver
        self.priority = plugin_config.get('priority', 0)

        self.init_from_data = check_bool(plugin_config.get('init_from_data', True))

        self.max_replicas = self.system.max_replicas

        # Initialize bin mapper from data in h5 file if available
        bin_mapper = self.init_bin_mapper()

        if bin_mapper:
            self.system.bin_mapper = bin_mapper

        # Register callback
        sim_manager.register_callback(sim_manager.pre_we, self.pre_we, self.priority)

    def init_bin_mapper(self):
        self.data_manager.open_backing()

        with self.data_manager.lock:
            n_iter = max(self.data_manager.current_iteration - 1, 1)
            iter_group = self.data_manager.get_iter_group(n_iter)

            # First attempt to initialize binmapper from data rather than system
            bin_mapper = None
            if self.init_from_data:
                log.info('Attempting to initialize WEX bin mapper from data')

                try:
                    binhash = iter_group.attrs['binhash']
                    bin_mapper = self.data_manager.get_bin_mapper(binhash)

                except:
                    log.warning('Initializing bins from data failed; Using definition in system instead.')
                    centers = self.system.bin_mapper.centers
            else:
                log.info('Initializing bin mapper from system definition')

        self.data_manager.close_backing()

        return bin_mapper

    def pre_we(self):
        starttime = time.time()
        segments = self.sim_manager.segments.values()
        bin_mapper = self.system.bin_mapper

        final_pcoords = np.empty((len(segments), self.system.pcoord_ndim), dtype=self.system.pcoord_dtype)

        for iseg, segment in enumerate(segments):
            final_pcoords[iseg] = segment.pcoord[-1,:]

        hash_init = bin_mapper.last_graph
        assignments = bin_mapper.assign(final_pcoords, add_bins=True)

        # Re-assign segments to new bins if bin_mapper changes
        if bin_mapper.last_graph != hash_init:

            # Reset we_driver internal data structures
            self.we_driver.new_iteration()

            # Re-assign segments
            self.we_driver.assign(segments)

        # Get assignments. Should use cached assignments
        assignments = bin_mapper.assign(final_pcoords)

        # Balance replicas - update bin_target_count
        target_counts = bin_mapper.balance_replicas(self.max_replicas, assignments)

        self.system.bin_target_counts = target_counts
        self.we_driver.bin_target_counts = target_counts

        endtime = time.time()

        # Report level statistics
        s = 1
        westpa.rc.pstatus('--wexplore-stats--------------------')
        westpa.rc.pstatus('wallclock time: {:.3f} s'.format(endtime - starttime))
        westpa.rc.pstatus('')
        for li, level in enumerate(bin_mapper.level_indices):
            s *= bin_mapper.n_regions[li]
            westpa.rc.pstatus('Level {}: {} cells ({} max)'.format(li, len(level), s))
        westpa.rc.pstatus('------------------------------------')
        westpa.rc.pflush()
