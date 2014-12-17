from __future__ import division; __metaclass__ = type
import logging
log = logging.getLogger(__name__)

import numpy as np

import westpa, west
from westpa import extloader
from westpa.yamlcfg import check_bool, ConfigItemMissing

class WExploreDriver(object):
    def __init__(self, sim_manager, plugin_config):
        super(ExploreDriver, self).__init__()

        if not sim_manager.work_manager.is_master:
                return

        self.sim_manager = sim_manager
        self.data_manager = sim_manager.data_manager
        self.system = sim_manager.system
        self.priority = plugin_config.get('priority', 0)

        self.init_from_data = check_bool(plugin_config.get('init_from_data', True))

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

        # Determine if any bins need to be added
        # - Traverse bin space. Only check level if more bins can be added to leaf

        # Re-assign segments to new bins if bin_mapper changes

        # Balance replicas - update bin_target_count
