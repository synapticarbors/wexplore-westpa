import numpy as np
import networkx as nx
import pandas as pd
import heapq
import cPickle as pickle
import hashlib

from westpa.binning.assign import BinMapper
from wex_utils import apply_down_argmin_across

index_dtype = np.uint16
coord_dtype = np.float32


class WExploreBinMapper(BinMapper):
    def __init__(self, n_regions, d_cut, dfargs=None, dfkwargs=None):

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

    def dfunc(self):
        '''User-implemented distance function of the form dfunc(p, centers, *arg, **kwargs) that returns
        the distance between a test point `p` in the progress coordinate space and each bin center in `centers`.
        '''
        raise NotImplementedError

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

    def _assign_level(self, coords, centers, mask, output, max_dist):
        '''The assign method from a standard VoronoiBinMapper'''
        if coords.ndim != 2:
            raise TypeError('coords must be 2-dimensional')

        apply_down_argmin_across(self.dfunc, (centers,) + self.dfargs, 
                                self.dfkwargs, centers.shape[0], coords, mask, output, max_dist)

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

        max_dist = np.empty((len(coords),), dtype=coord_dtype)

        G = self.bin_graph
        graph_hash = self._hash(G)

        # Check if we can return cached assigments instead of recalculating
        if self.last_graph == graph_hash and np.array_equal(coords, self.last_coords) and np.array_equal(mask, self.last_mask):
            return self.last_assignment


        # List of new bins to potentially add at end of the routine.
        new_bins = []

        # List of coord indices used to create new bins. This avoids adding
        # A bin originating from the same coord_ix more than once.
        new_bin_coord_ix = []

        # List of coordinate indices assigned to each node
        nx.set_node_attributes(G, 'coord_ix', None)

        # Traverse the bin graph
        node_indices = self.level_indices[0]
        centers = self.fetch_centers(node_indices)
        self._assign_level(coords, centers, mask, output, max_dist)

        obs_nodes = self._distribute_to_children(G, output, np.arange(coords.shape[0]), node_indices)

        if add_bins:
            coord_ix = np.argmax(max_dist)
            if max_dist[coord_ix] > self.d_cut[0]:
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
                grp_max_dist = np.empty((grp_coords.shape[0],), dtype=coord_dtype)
                self._assign_level(grp_coords, centers, grp_mask, grp_output, grp_max_dist)
                _obs_nodes = self._distribute_to_children(G, grp_output, grp_coord_ix, successors)
                next_obs_nodes.extend(_obs_nodes)

                if add_bins:
                    md_ix = np.argmax(grp_max_dist)
                    coord_ix = grp_coord_ix[md_ix]
                    if grp_max_dist[md_ix] > self.d_cut[lid]:
                        new_bins.append((lid, nix, coord_ix))

            obs_nodes = next_obs_nodes

        # Transfer assignments into final output
        bin_map = np.arange(max(self.level_indices[-1]) + 1, dtype=np.int)
        for ci, nix in enumerate(self.level_indices[-1]):
            bin_map[nix] = ci

        for nix in obs_nodes:
            cix = G.node[nix]['coord_ix']
            output[cix] = bin_map[nix]

        self.last_coords = coords
        self.last_mask = mask
        self.last_assignment = output
        self.last_graph = self._hash(self.bin_graph)

        if add_bins:
            total_bins = self.nbins
            for new_bin in new_bins:
                coord_ix = new_bin[2]
                if coord_ix not in new_bin_coord_ix:
                    # Attempt to add new bin
                    cix = len(self.centers)
                    self.add_bin(new_bin[1], cix)

                    if self.nbins > total_bins:
                        # Add coords to centers
                        self.centers.append(coords[coord_ix])
                        total_bins = self.nbins
                        new_bin_coord_ix.append(coord_ix)

        return output

    def add_bin(self, parent, center_ix):
        '''Add a bin that subdivides the bin `parent`, where `parent` is a 
        networkx Node. 
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






