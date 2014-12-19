import numpy as np
import networkx as nx
import pandas as pd
import heapq
import cPickle as pickle
import hashlib
import logging
import itertools

from westpa.binning.assign import BinMapper
from wex_utils import apply_down_argmin_across, argmin_exceed_threshold

index_dtype = np.uint16
coord_dtype = np.float32

log = logging.getLogger(__name__)

class WExploreBinMapper(BinMapper):
    def __init__(self, n_regions, d_cut, dfunc, dfargs=None, dfkwargs=None):

        assert len(n_regions) == len(d_cut)

        # Max number of bins per level (list of integers)
        self.n_regions = n_regions

        # Distance cut-off for each level
        self.d_cut = d_cut

        # bin centers. Note: this does not directly map to node ids
        self.centers = []

        # List of bin indices for each level
        self.level_indices = [[] for k in xrange(self.n_levels)]

        # Directed graph containing that defines the connectivity
        # of the hierarchical bin space.
        self.bin_graph = nx.DiGraph()

        self.next_bin_index = 0

        self.dfunc = dfunc
        self.dfargs = dfargs or ()
        self.dfkwargs = dfkwargs or {}

        # Variables to cache assignment calculation
        self.last_coords = None
        self.last_mask = None
        self.last_assignment = None
        self.last_graph = self._hash(self.bin_graph)

    @property
    def max_nbins(self):
        '''Max number of bins in the lowest level of the
        bin hierarchy'''
        return np.prod(self.n_regions)

    @property
    def nbins(self):
        '''Current number of bins in the lowest level of
        the hierarchy'''
        return len(self.level_indices[-1])

    @property
    def n_levels(self):
        return len(self.n_regions)

    @property
    def labels(self):
        return self.level_indices[-1]

    def _hash(self, obj):
        '''Return the hash digest of the object, `obj`'''
        pkldat = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        return self.hashfunc(pkldat).hexdigest()

    def fetch_centers(self, nodes):
        '''Return the coordinates associated with the nodes in self.bin_graph, `nodes`.
        This method returns a 2-dimensional numpy array with each row
        representing the bin center of a node. 
        '''
        centers = []
        for nix in nodes:
            cix = self.bin_graph.node[nix]['center_ix']
            centers.append(self.centers[cix])

        centers = np.asarray(centers, dtype=coord_dtype)
        if centers.ndim < 2:
            new_centers = np.empty((1,centers.shape[0]), dtype=coord_dtype)
            new_centers[0,:] = centers[:]
            centers = new_centers

        return centers

    def _assign_level(self, coords, centers, mask, output, min_dist):
        '''The assign method from a standard VoronoiBinMapper'''
        if coords.ndim != 2:
            raise TypeError('coords must be 2-dimensional')

        apply_down_argmin_across(self.dfunc, (centers,) + self.dfargs, 
                                self.dfkwargs, centers.shape[0], coords, mask, output, min_dist)

        return output

    def dump_graph(self):
        print ''
        for li, nodes in enumerate(self.level_indices):
            print 'Level: ', li
            for nix in nodes:
                print nix, self.bin_graph.node[nix]

    def _distribute_to_children(self, G, output, coord_indices, node_indices):
        s = pd.Series(output, copy=False)
        observed_nodes = []
        # Groupby assignments at the previous level
        for cix, grp in s.groupby(s.values, sort=False):
            # map the 0-based centers index back to a bin index
            nix = node_indices[cix]
            observed_nodes.append(nix)
            G.node[nix]['coord_ix'] = coord_indices[grp.index]

        return observed_nodes

    def _prune_violations(self):
        '''Remove any nodes and their sucessors that would no longer be mapped to 
        the same parent'''
        G = self.bin_graph
        nodes_remove = []

        for li, level_indices in enumerate(self.level_indices):
            if li == 0:
                continue

            prev_level_nodes = self.level_indices[li - 1]
            pcenters = self.fetch_centers(prev_level_nodes)

            centers = self.fetch_centers(level_indices)
            ncenters = len(centers)

            parent_nodes = [G.predecessors(ix)[0] for ix in level_indices]

            mask = np.ones((ncenters,), dtype=np.bool_)
            output = np.empty((ncenters,), dtype=index_dtype)
            min_dist = np.empty((ncenters,), dtype=coord_dtype)

            self._assign_level(centers, pcenters, mask, output, min_dist)

            for k in xrange(ncenters):
                if output[k] != prev_level_nodes.index(parent_nodes[k]):
                    nix = level_indices[k]
                    succ = nx.algorithms.traversal.dfs_successors(G, nix).values()
                    nodes_remove.extend(list(itertools.chain.from_iterable(succ)) + [nix])

        if len(nodes_remove):
            G.remove_nodes_from(nodes_remove)
            for k in xrange(self.n_levels):
                self.level_indices[k] = [nix for nix in self.level_indices[k] if nix not in nodes_remove]

    def assign(self, coords, mask=None, output=None, add_bins=False):
        '''Hierarchically assign coordinates to bins'''
        try:
            passed_coord_dtype = coords.dtype
        except AttributeError:
            coords = np.require(coords, dtype=coord_dtype)
        else:
            if passed_coord_dtype != coord_dtype:
                coords = np.require(coords, dtype=coord_dtype)

        if coords.ndim != 2:
            raise TypeError('coords must be 2-dimensional')
        if mask is None:
            mask = np.ones((len(coords),), dtype=np.bool_)
        elif len(mask) != len(coords):
            raise TypeError('mask [shape {}] has different length than coords [shape {}]'.format(mask.shape, coords.shape))

        if output is None:
            output = np.empty((len(coords),), dtype=index_dtype)
        elif len(output) != len(coords):
            raise TypeError('output has different length than coords')

        min_dist = np.empty((len(coords),), dtype=coord_dtype)

        G = self.bin_graph

        graph_hash = self._hash(G)

        # Check if we can return cached assigments instead of recalculating
        if self.last_graph == graph_hash and np.array_equal(coords, self.last_coords) and np.array_equal(mask, self.last_mask):
            log.debug('assign() using cached assigments')
            return self.last_assignment

        # List of coordinate indices assigned to each node
        nx.set_node_attributes(G, 'coord_ix', None)

        # List of new bins to potentially add at end of the routine.
        new_bins = []

        # List of coord indices used to create new bins. This avoids adding
        # A bin originating from the same coord_ix more than once.
        new_bin_coord_ix = []

        # Traverse the bin graph
        node_indices = self.level_indices[0]
        centers = self.fetch_centers(node_indices)
        self._assign_level(coords, centers, mask, output, min_dist)

        obs_nodes = self._distribute_to_children(G, output, np.arange(coords.shape[0]), node_indices)

        if add_bins:
            # Get the position that exceeds the threshold by the smallest amount
            coord_ix = argmin_exceed_threshold(min_dist, self.d_cut[0])
            if coord_ix >= 0:
                new_bins.append((0, None, coord_ix))

        for lid in xrange(1, self.n_levels):
            next_obs_nodes = []
            for nix in obs_nodes:
                successors = G.successors(nix)
                centers = self.fetch_centers(successors)
                grp_coord_ix = G.node[nix]['coord_ix']
                grp_coords = coords.take(grp_coord_ix, axis=0)
                grp_mask = mask.take(grp_coord_ix)
                grp_output = np.empty((grp_coords.shape[0],), dtype=index_dtype)
                grp_min_dist = np.empty((grp_coords.shape[0],), dtype=coord_dtype)
                self._assign_level(grp_coords, centers, grp_mask, grp_output, grp_min_dist)
                _obs_nodes = self._distribute_to_children(G, grp_output, grp_coord_ix, successors)
                next_obs_nodes.extend(_obs_nodes)

                if add_bins:
                    md_ix = argmin_exceed_threshold(grp_min_dist, self.d_cut[lid])
                    if md_ix >= 0:
                        coord_ix = grp_coord_ix[md_ix]
                        new_bins.append((lid, nix, coord_ix))

            obs_nodes = next_obs_nodes

        # Transfer assignments into final output
        bin_map = np.arange(max(self.level_indices[-1]) + 1, dtype=np.int)
        for ci, nix in enumerate(self.level_indices[-1]):
            bin_map[nix] = ci

        for nix in obs_nodes:
            cix = G.node[nix]['coord_ix']
            output[cix] = bin_map[nix]

        #nx.set_node_attributes(G, 'coord_ix', None)

        self.last_coords = coords
        self.last_mask = mask
        self.last_assignment = output
        self.last_graph = self._hash(self.bin_graph)

        if add_bins:
            total_bins = self.nbins
            for new_bin in new_bins:
                parent_ix = new_bin[1]
                coord_ix = new_bin[2]

                if coord_ix not in new_bin_coord_ix:
                    # Attempt to add new bin
                    cix = len(self.centers)
                    self.add_bin(parent_ix, cix)

                    if self.nbins > total_bins:
                        # Add coords to centers
                        self.centers.append(coords[coord_ix])
                        total_bins = self.nbins
                        new_bin_coord_ix.append(coord_ix)

            self._prune_violations()


        return output

    def check_distance(self, parent_ix, coords):
        '''Check to ensure that the distance between the coordinates `coords`
        and the parent_ix is less than the distance to any other center in the parent level'''
        if parent_ix is None:
            return True

        try:
            passed_coord_dtype = coords.dtype
        except AttributeError:
            coords = np.require(coords, dtype=coord_dtype)
        else:
            if passed_coord_dtype != coord_dtype:
                coords = np.require(coords, dtype=coord_dtype)

        coords = coords.reshape((1, -1))

        assert len(coords) == 1
        parent_level = self.bin_graph.node[parent_ix]['level']
        level_indices = self.level_indices[parent_level]
        parent_centers = self.fetch_centers(level_indices)

        mask = np.ones((1,), dtype=np.bool_)
        output = np.empty((1,), dtype=index_dtype)
        min_dist = np.empty((1,), dtype=coord_dtype)
        self._assign_level(coords, parent_centers, mask, output, min_dist)

        res = output[0] == level_indices.index(parent_ix)

        return res

    def add_bin(self, parent, center_ix):
        '''Add a bin that subdivides the bin `parent`, where `parent` is a 
        node index. 
        If `parent`is None, then it is assumed that this bin is at the top of the bin
        hiearchy. The `center_ix` argument defines an association of 
        coordinate data with the bin. 

        A bin with the same `center_ix`, but a unique bin index will
        be recursively added to all levels below the parent, `parent` in 
        the hiearchy. 
        '''

        level = 0 if parent is None else self.bin_graph.node[parent]['level'] + 1

        # Attempt to add to a non-existent level
        if level >= self.n_levels:
            return

        # Check to make sure max number of bins per leaf is not exceeded
        if level == 0 and len(self.level_indices[0]) + 1 > self.n_regions[0]:
            return
        elif level > 0:
            if len(self.bin_graph.successors(parent)) + 1 > self.n_regions[level]:
                return

        bin_index = self.next_bin_index
        self.level_indices[level].append(bin_index)
        self.bin_graph.add_node(bin_index, 
                {'center_ix': center_ix,
                 'level': level})

        self.bin_graph.add_edge(parent, bin_index)

        self.next_bin_index += 1
        self.add_bin(bin_index, center_ix)

    def balance_replicas(self, max_replicas, assignments):
        '''Given a set of assignments to the lowest level bins in the
        hierarchy, return an array containing the target number of replicas
        per bin after rebalancing such that the total number of replicas is 
        `max_replicas`.'''

        target_count = np.zeros((self.nbins), dtype=np.int_)

        # Determine bins occupied by at least one replica
        occupied_bins = np.unique(assignments)

        # Create graph to accumulate replica counts
        G = self.bin_graph.subgraph(self.bin_graph.nodes())

        # Set min number of replicas for lowest level bin in the hierarchy
        nx.set_node_attributes(G, 'nreplicas', 0)
        for bi in occupied_bins:
            G.node[self.level_indices[-1][bi]]['nreplicas'] = 1

        # Accumulate minimum number up the tree
        for top_node in self.level_indices[0]:
            for nix in nx.algorithms.traversal.dfs_postorder_nodes(G, top_node):
                try:
                    pred = G.pred[nix].keys()[0]   # parent node
                    G.node[pred]['nreplicas'] += G.node[nix]['nreplicas']
                except IndexError:
                    pass

        # Distribute replicas across the hierarchy
        for li, nodes in enumerate(self.level_indices):

            # Get number of replicas available to a level
            if li == 0:
                level_max_replicas = max_replicas
            else:
                level_max_replicas = sum([G.node[nix]['nreplicas'] for nix in self.level_indices[li - 1]])

            pq = [(G.node[nix]['nreplicas'], nix) for nix in nodes if G.node[nix]['nreplicas'] > 0]
            heapq.heapify(pq)

            available_replicas = level_max_replicas - sum(p[0] for p in pq)

            while available_replicas:
                if len(pq) == 1:
                    nr = available_replicas
                else:
                    nr = min(available_replicas, pq[1][0] - pq[0][0] + 1)

                heapq.heapreplace(pq, (pq[0][0] + nr, pq[0][1]))
                available_replicas -= nr

            if li < self.n_levels - 1:
                for nr, nix in pq:
                    G.node[nix]['nreplicas'] = nr
            else:
                bin_map = np.arange(max(self.level_indices[-1]) + 1, dtype=np.int)
                for ci, nix in enumerate(self.level_indices[-1]):
                    bin_map[nix] = ci

                for nr, nix in pq:
                    bix = bin_map[nix]
                    target_count[bix] = nr

        # Check for problems
        orig_occupied = np.zeros((self.nbins), dtype=np.int_)
        orig_occupied[occupied_bins] = 1

        assert (~((orig_occupied > 0) & (target_count == 0))).all(), 'Populated bin given zero replicas.'
        assert (~((orig_occupied == 0) & (target_count > 0))).all(), 'Unpopulated bin assigned replicas.'

        return target_count






