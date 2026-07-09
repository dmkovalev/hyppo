"""OWL ontology for the virtual-experiment domain (Definition 1, Chapter 2).

Defines the ``virtual_experiment_onto`` OWL ontology (owlready2) that carries
the formal artefacts of a virtual experiment: hypotheses, models, the
workflow, and the causal structure (equations/variables) that a hypothesis
is built from. ``derived_by``/``impacts`` encode the lattice edges consumed
by :class:`hyppo.lattice_constructor._base.HypothesisLattice` (Algorithm 1).
"""

import datetime

from owlready2 import (
    AllDisjoint,
    DataProperty,
    FunctionalProperty,
    ObjectProperty,
    SymmetricProperty,
    Thing,
    TransitiveProperty,
    get_ontology,
)

virtual_experiment_onto = get_ontology(
    "http://synthesis.ipi.ac.ru/virtual_experiment.owl"
)
hcp_brain_onto = get_ontology("http://synthesis.ipi.ac.ru/hcp_brain_onto.owl")
virtual_experiment_onto.imported_ontologies.append(hcp_brain_onto)

with virtual_experiment_onto:
    # define base class and its properties
    class Artefact(Thing):
        """Root OWL class for every documented artefact (Definition 1).

        Base class for ``Hypothesis``, ``Model``, ``VirtualExperiment`` and
        the structural classes below; carries the mandatory identity/
        provenance properties (id, name, description, authors, timestamps).
        """

    # class Specification(Thing): pass
    class has_for_id(Artefact >> int, DataProperty, FunctionalProperty):
        """Functional data property: unique integer identifier of an artefact."""

        python_name = "id"

    class has_for_name(Artefact >> str, DataProperty, FunctionalProperty):
        """Functional data property: human-readable name of an artefact."""

        python_name = "name"

    class has_for_description(Artefact >> str, DataProperty, FunctionalProperty):
        """Functional data property: free-text description of an artefact."""

        python_name = "description"

    class has_for_authors(Artefact >> str, DataProperty):
        """Non-functional data property: one or more author names (min 1)."""

        python_name = "authors"

    class has_for_createdate(
        Artefact >> datetime.datetime, DataProperty, FunctionalProperty
    ):
        """Functional data property: artefact creation timestamp."""

        python_name = "create_date"

    class has_for_lastupdate(
        Artefact >> datetime.datetime, DataProperty, FunctionalProperty
    ):
        """Functional data property: artefact last-modification timestamp."""

        python_name = "last_update"

    # class has_for_specification(Artefact >> Specification): pass

    Artefact.is_a.extend(
        [
            has_for_authors.min(1),
            has_for_name.exactly(1),
            has_for_description.exactly(1),
            has_for_id.exactly(1),
            has_for_lastupdate.exactly(1),
            has_for_createdate.exactly(1),
        ]
    )

    class Hypothesis(Artefact):
        """OWL individual for a hypothesis h in H (Definition 1).

        Node type of the hypothesis lattice built by Algorithm 1
        (:class:`hyppo.lattice_constructor._base.HypothesisLattice`); linked
        to its implementing ``Model`` via ``is_implemented_by_model`` and to
        competing/derived hypotheses via ``competes``/``derived_by``.
        """

    class Model(Artefact):
        """OWL individual for the model that implements a ``Hypothesis``.

        Paired 1:1 with its hypothesis through the mutually-inverse,
        functional properties ``is_implemented_by_model`` / ``refers_to_hypothesis``
        (see the Theorem 1 axiomatic support note below).
        """

    # class Mapping(Artefact): pass
    # class Relation(Artefact): pass

    # TODO probability > 0.0 and < 1.0
    class has_for_probability(Hypothesis >> float, DataProperty, FunctionalProperty):
        """Functional data property: prior/posterior probability of a hypothesis."""

        python_name = "probability"

    class is_implemented_by_model(Hypothesis >> Model):
        """Object property: hypothesis -> the model implementing it (some Model)."""

        class_property_type = ["some"]

    class refers_to_hypothesis(ObjectProperty):
        """Object property: model -> the hypothesis it implements (inverse of
        ``is_implemented_by_model``); both are declared functional below for
        the Theorem 1 uniqueness proof (paired 1:1 correspondence)."""

        domain = [Model]
        range = [Hypothesis]
        inverse_property = is_implemented_by_model
        class_property_type = ["only"]

    class competes(Hypothesis >> Hypothesis, SymmetricProperty):
        """Symmetric object property: two hypotheses compete over the same
        phenomenon (used by :func:`hyppo.core._epistemic.evaluate_status` via
        the runner's Delta-AIC SUPERSEDED check)."""

    class derived_by(Hypothesis >> Hypothesis, TransitiveProperty):
        """Transitive object property: lattice edge h_j derived_by h_i, i.e.
        h_j depends on h_i's output (Definition 1); the edge set is computed
        by Algorithm 1 and incrementally maintained by Algorithm 2."""

    # Note: AsymmetricProperty and IrreflexiveProperty removed because
    # OWL 2 DL simplicity constraint forbids them on transitive properties.
    # Acyclicity is enforced by Algorithm 3 (consistency check), not OWL axioms.
    class impacts(ObjectProperty, TransitiveProperty):
        """Transitive object property, inverse of ``derived_by``: h_i impacts
        h_j means h_j is derived_by h_i. Acyclicity of this relation (no
        hypothesis impacts itself transitively) is enforced procedurally by
        Algorithm 3's consistency check, not by an OWL axiom (see note above)."""

        domain = [Hypothesis]
        range = [Hypothesis]
        inverse_property = derived_by

    class VirtualExperiment(Artefact):
        """OWL individual for a virtual experiment: bundles a set of
        hypotheses, their models, a ``Workflow`` and a ``Configuration``."""

    class Configuration(Artefact):
        """OWL individual holding the run configuration of a virtual experiment."""

    class Workflow(Artefact):
        """OWL individual for the task DAG of a virtual experiment; the
        Python-level counterpart with execution semantics is
        :class:`hyppo.core._workflow.Workflow`."""

    # class Task(Thing): pass

    class has_for_hypothesis(VirtualExperiment >> Hypothesis):
        """Object property: virtual experiment -> its hypotheses (some Hypothesis)."""

        class_property_type = ["some"]

    class has_for_model(VirtualExperiment >> Model):
        """Object property: virtual experiment -> its models (some Model)."""

        class_property_type = ["some"]

    class has_for_workflow(VirtualExperiment >> Workflow):
        """Object property: virtual experiment -> its task workflow (only Workflow)."""

        class_property_type = ["only"]

    class has_for_configuration(VirtualExperiment >> Configuration):
        """Object property: virtual experiment -> its run configuration
        (some Configuration)."""

        class_property_type = ["some"]

    # class has_for_task(Workflow >> Task): class_property_type = ["some"]

    class Structure(Artefact):
        """OWL counterpart of a causal-ordering structure: a set of equations
        over a set of variables, as manipulated by the pure-Python core
        :class:`hyppo.coa._base.Structure` (Algorithm 1 input)."""

    class FullStructure(Structure):
        """A ``Structure`` that is complete (Hall's condition holds): every
        equation can be matched to a distinct output variable, so a causal
        mapping (``FullCausalMapping``) can be derived from it."""

    class Equation(Thing):
        """OWL counterpart of a single causal-ordering equation, mirroring
        :class:`hyppo.coa._base.Equation` (formula + its free variables)."""

    class Variable(Thing):
        """OWL individual for a variable appearing in an ``Equation`` /
        ``Structure`` (sympy Symbol on the pure-Python side)."""

    class has_for_varname(Variable >> str, DataProperty, FunctionalProperty):
        """Functional data property: the symbolic name of a ``Variable``."""

        python_name = "name"

    class FullCausalMapping(Artefact):
        """The causal mapping (assignment equation -> output variable)
        derived from a ``FullStructure`` by the causal ordering algorithm."""

    class has_for_fcm(FullStructure >> FullCausalMapping):
        """Object property: a ``FullStructure`` -> its derived
        ``FullCausalMapping`` (only)."""

        class_property_type = ["only"]

    class has_for_structure(Hypothesis >> Structure):
        """Object property: a ``Hypothesis`` -> the ``Structure`` of
        equations it is built from (only)."""

        class_property_type = ["only"]

    class DependencySet(Artefact):
        """Set of variable/hypothesis dependencies derived from a
        ``FullStructure``'s causal mapping."""

    class has_for_dependecy_set(FullStructure >> DependencySet):
        """Object property: a ``FullStructure`` -> its ``DependencySet``
        (only)."""

        class_property_type = ["only"]

    class TransitiveClosure(DependencySet):
        """A ``DependencySet`` closed under transitivity (all indirect
        dependencies made explicit)."""

    class ResearchLattice(Artefact):
        """OWL counterpart of the hypothesis lattice built by Algorithm 1
        (:class:`hyppo.lattice_constructor._base.HypothesisLattice`): the set
        of hypotheses under study, linked via ``has_for_lattice_hypothesis``."""

    class has_for_lattice_hypothesis(ResearchLattice >> Hypothesis):
        """Object property: a ``ResearchLattice`` -> its member hypotheses
        (some Hypothesis)."""

        class_property_type = ["some"]

    # class has_for_vars(Equation >> Variable, DataProperty):
    #     class_property_type = ["some"]
    #     python_name = "vars"

    # class has_for_equation(Structure >> Equation, DataProperty):
    #     python_name = "equation"

    # class has_for_structure_variable(Structure >> Variable):
    #     class_property_type = ["some"]
    #     python_name = "vars"

    AllDisjoint([VirtualExperiment, Configuration, Workflow, Hypothesis, Model])

    # ── Theorem 1 axiomatic support (iip2026_planning.tex §3) ──────────────
    # Adds the three OWL axioms required for the C2 source-of-inconsistency
    # bijection in the consistency-check correctness proof:
    #   (i)  is_implemented_by_model is Functional → uniqueness of m
    #        (each hypothesis is implemented by at most one model);
    #   (ii) refers_to_hypothesis is Functional → uniqueness of h per model
    #        (paired-functional design across the inverse property);
    #   (iii) Hypothesis ⊑ ∃is_implemented_by_model.Model → existence of m
    #        (every Hypothesis has at least one implementing model).
    # Together these give the ∃! m ∈ Model. R(m) = h required by C2.
    # NOTE on UNA: OWL 2 DL does not assume Unique Name Assumption, so
    # FunctionalProperty alone only unifies (m1 ≡ m2) rather than yielding
    # owl:Nothing. Concrete VE instantiations must add AllDifferent on the
    # set of Model individuals to enforce C2-uniqueness inconsistency
    # detection by HermiT (see paper §2, Theorem 1 proof).
    is_implemented_by_model.is_a.append(FunctionalProperty)
    refers_to_hypothesis.is_a.append(FunctionalProperty)
    Hypothesis.is_a.append(is_implemented_by_model.some(Model))


if __name__ == "__main__":
    virtual_experiment_onto = get_ontology(
        "http://synthesis.ipi.ac.ru/virtual_experiment.owl"
    )
    print(list(virtual_experiment_onto.classes()))
    virtual_experiment_onto.save("ve.owl")
    art = Artefact("123")
    art.has_for_author = [123]
    print(has_for_authors.range)
    print(art.name)
