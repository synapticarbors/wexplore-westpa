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

    def test_fetch_centers(self):
        bin_mapper = WEBM([3, 3, 3], [4.0, 2.0, 1.0])

        coords = np.arange(20).reshape((10,2))

        for k in xrange(3):
            bin_mapper.add_bin(None, coords[k])

        centers = bin_mapper.fetch_centers(bin_mapper.level_indices[0])
        assert np.allclose(centers, coords[:3,:])

        centers = bin_mapper.fetch_centers([0, 1, 2])
        assert np.allclose(centers, 
                np.vstack((coords[0], coords[0], coords[0])))

        centers = bin_mapper.fetch_centers([0])
        assert np.allclose(centers, coords[0])

    def test_balance_replicas(self):
        bin_mapper = WEBM([2, 2, 2], [4.0, 2.0, 1.0])

        for k in xrange(2):
            bin_mapper.add_bin(None, None)

        for node in bin_mapper.level_indices[0]:
            bin_mapper.add_bin(node, None)

        for node in bin_mapper.level_indices[1]:
            bin_mapper.add_bin(node, None)

        assignments = np.array([1, 0, 0, 0, 0, 0, 0, 2, 2, 3, 4, 5, 6, 6, 6, 7])
        target_count = bin_mapper.balance_replicas(16, assignments)

        print target_count
        assert np.alltrue(target_count == 2*np.ones(16, dtype=np.int))




        





