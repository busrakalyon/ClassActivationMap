import unittest

import numpy as np

from edge_finer_research.proposals.sop import proposal_edge, select_object_proposal


class SOPTests(unittest.TestCase):
    def test_selects_proposal_with_matching_edge(self):
        proposals = np.zeros((50, 9, 9), dtype=bool)
        proposals[7, 3:6, 3:6] = True
        heatmap = proposal_edge(proposals)[7].astype(float)
        result = select_object_proposal(heatmap, proposals)
        self.assertEqual(result.proposal_index, 7)
        np.testing.assert_array_equal(result.mask, proposals[7])

    def test_requires_exactly_fifty_proposals(self):
        with self.assertRaises(ValueError):
            select_object_proposal(np.zeros((5, 5)), np.zeros((49, 5, 5), dtype=bool))


if __name__ == "__main__":
    unittest.main()
