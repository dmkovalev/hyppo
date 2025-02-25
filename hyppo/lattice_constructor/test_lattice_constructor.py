import unittest
from unittest.mock import Mock, patch
import networkx as nx
from collections import defaultdict

from hyppo.lattice_constructor._base import HypothesisLattice
from hyppo.core._base import virtual_experiment_onto
from hyppo.coa._base import Structure, Equation

class TestHypothesisLattice(unittest.TestCase):
    def setUp(self):
        # Mock workflow and hypotheses
        self.workflow = Mock()
        self.hypothesis1 = Mock()
        self.hypothesis2 = Mock()
        self.hypotheses = [self.hypothesis1, self.hypothesis2]
        
        # Setup mock structures
        self.structure1 = Mock(spec=Structure)
        self.structure2 = Mock(spec=Structure)
        self.hypothesis1.structure = self.structure1
        self.hypothesis2.structure = self.structure2

    def test_is_correct_with_valid_workflow(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = self.hypotheses
        
        # Create lattice
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        
        # Test
        self.assertTrue(lattice._is_correct())

    def test_is_correct_with_invalid_workflow(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = [self.hypothesis1]  # Missing hypothesis2
        
        # Create lattice
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        
        # Test
        self.assertFalse(lattice._is_correct())

    def test_build_lattice_with_complete_structures(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = self.hypotheses
        self.workflow.get_remaining.return_value = []
        
        united_structure = Mock(spec=Structure)
        united_structure.is_complete.return_value = True
        united_structure.build_transitive_closure.return_value = {('var1', 'var2')}
        
        self.structure1.union.return_value = united_structure
        
        # Create lattice
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        
        # Test
        self.assertIsInstance(lattice.lattice, nx.DiGraph)
        
    def test_build_lattice_with_incomplete_structures(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = self.hypotheses
        self.workflow.get_remaining.return_value = []
        
        united_structure = Mock(spec=Structure)
        united_structure.is_complete.return_value = False
        
        self.structure1.union.return_value = united_structure
        
        # Create lattice
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        
        # Test
        self.assertEqual(len(lattice.lattice.edges()), 0)

    def test_derived_by_with_existing_hypothesis(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = self.hypotheses
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        # Mock the lattice structure
        lattice.lattice.add_edge(self.hypothesis1, self.hypothesis2)
        
        # Test
        derived = lattice.derived_by(self.hypothesis2)
        self.assertEqual(derived, {self.hypothesis1})

    def test_derived_by_with_nonexistent_hypothesis(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = self.hypotheses
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        nonexistent_hypothesis = Mock()
        
        # Test
        derived = lattice.derived_by(nonexistent_hypothesis)
        self.assertEqual(derived, set())

    def test_competes_with_competing_hypotheses(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = self.hypotheses
        hypothesis3 = Mock()
        self.hypotheses.append(hypothesis3)
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        
        # Create a competition scenario
        lattice.lattice.add_edge(self.hypothesis1, self.hypothesis2)
        lattice.lattice.add_edge(self.hypothesis1, hypothesis3)
        
        # Test
        competitors = lattice.competes(self.hypothesis2)
        self.assertEqual(competitors, {hypothesis3})

    def test_impacts_with_impacted_hypotheses(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = self.hypotheses
        hypothesis3 = Mock()
        self.hypotheses.append(hypothesis3)
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        
        # Create impact chain
        lattice.lattice.add_edge(self.hypothesis1, self.hypothesis2)
        lattice.lattice.add_edge(self.hypothesis2, hypothesis3)
        
        # Test
        impacted = lattice.impacts(self.hypothesis1)
        self.assertEqual(impacted, {self.hypothesis2, hypothesis3})

    def test_remove_hypothesis(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = self.hypotheses
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        lattice.lattice.add_edge(self.hypothesis1, self.hypothesis2)
        
        # Test
        lattice.remove_hypothesis(self.hypothesis1)
        self.assertNotIn(self.hypothesis1, lattice.hypotheses)
        self.assertNotIn(self.hypothesis1, lattice.lattice.nodes())

    def test_build_hypothesis_var_mapping(self):
        # Setup
        self.workflow.get_tasks.return_value = Mock()
        self.workflow.get_tasks().get_current.return_value = self.hypotheses
        lattice = HypothesisLattice(self.hypotheses, self.workflow)
        
        # Create mock transitive closure data
        transitive_closure = defaultdict(set)
        transitive_closure[self.hypothesis1] = {('var1', 'var2')}
        transitive_closure[self.hypothesis2] = {('var1', 'var2'), ('var2', 'var3')}
        
        # Test
        dependencies = lattice._build_hypothesis_var_mapping(transitive_closure)
        expected = [(self.hypothesis1, self.hypothesis2)]  # h1 depends on h2 as its relations are subset
        self.assertEqual(dependencies, expected)


if __name__ == '__main__':
    unittest.main()