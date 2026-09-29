import unittest
from library.phase_retrieval_universal import illumination_group_labels


class IlluminationGroupsTests(unittest.TestCase):
    def test_grouping_preserves_polarization_pairs(self):
        states=['a','a','b','b','a','a']
        energies=[1,1,1,1,2,2]
        for variation,expected in [
            ('common',[0,0,0,0,0,0]), ('energy',[0,0,0,0,1,1]),
            ('state',[0,0,1,1,0,0]), ('energy_state',[0,0,1,1,2,2])]:
            self.assertEqual(illumination_group_labels(states,energies,variation),
                             [f'beam{i}' for i in expected])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): illumination_group_labels(['a'],[])
        with self.assertRaises(ValueError): illumination_group_labels(['a'],[1],'polarization')
