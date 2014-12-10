import numpy as np

from scipy.spatial.distance import cdist

import nose
import nose.tools

from wexplore import WExploreBinMapper

coord_dtype = np.float32


class WEBM(WExploreBinMapper):
    def dfunc(self, coordvec, centers):
        if coordvec.ndim < 2:
            new_coordvec = np.empty((1,coordvec.shape[0]), dtype=coord_dtype)
            new_coordvec[0,:] = coordvec[:]
            coordvec = new_coordvec 
        distmat = np.require(cdist(coordvec, centers), dtype=coord_dtype)

        return distmat[0,:]

    def fetch_centers(self, nodes):
        centers = []
        for nix in nodes:
            centers.append(self.bin_graph.node[nix]['coord_metadata'])

        centers = np.asarray(centers, dtype=coord_dtype)
        if centers.ndim < 2:
            new_centers = np.empty((1,centers.shape[0]), dtype=coord_dtype)
            new_centers[0,:] = centers[:]
            centers = new_centers

        return centers


class TestWExploreBinMapper:

    def test_create_mapper(self):
        bin_mapper = WEBM([2, 3, 3], [4.0, 2.0, 1.0])

        assert bin_mapper.n_levels == 3
        assert bin_mapper.nbins == 0
        assert bin_mapper.max_nbins == 18

    def test_add_bin(self):
        bin_mapper = WEBM([2, 2, 2], [4.0, 2.0, 1.0])

        bin_mapper.add_bin(None, [0.0, 0.0])

        assert bin_mapper.nbins == 1
        assert bin_mapper.next_bin_index == 3
        for k in xrange(3):
            assert len(bin_mapper.level_indices[k]) == 1

        bin_mapper.add_bin(None, [-3.0, 0.0])

        assert bin_mapper.nbins == 2
        for k in xrange(3):
            assert len(bin_mapper.level_indices[k]) == 2

        node = bin_mapper.level_indices[1][0]
        bin_mapper.add_bin(node, [-0.5, 0.0])

        assert bin_mapper.nbins == 3
        assert len(bin_mapper.level_indices[0]) == 2
        assert len(bin_mapper.level_indices[1]) == 2
        assert len(bin_mapper.level_indices[2]) == 3

        # Should not add bins when adding a bin with a parent
        # in the lowest level
        node = bin_mapper.level_indices[2][0]
        bin_mapper.add_bin(node, [-0.5, 0.0])

        assert bin_mapper.nbins == 3
        assert len(bin_mapper.level_indices[0]) == 2
        assert len(bin_mapper.level_indices[1]) == 2
        assert len(bin_mapper.level_indices[2]) == 3




