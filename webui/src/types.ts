export type Project = { id: string; name: string; description?: string };

export type Hypothesis = {
  id: string;
  label?: string;
  description?: string;
  params: Record<string, string[]>;
  epistemic_status?: string;
};

export type OntoClass = { name: string; label?: string; subclass_of?: string };
export type OntoProp = {
  name: string;
  domain?: string;
  range?: string;
  characteristics?: string[];
};
export type Ontology = {
  name: string;
  iri?: string;
  classes: OntoClass[];
  object_properties: OntoProp[];
  data_properties: OntoProp[];
};

export type Model = { id: string; label?: string; implements: string; kind?: string };
export type Mapping = { model: string; hypothesis: string };
export type ConfigAxis = { hypothesis: string; axis: string; levels: string[] };

export type VE = {
  domain?: string;
  ontology?: Ontology;
  hypotheses: Hypothesis[];
  models?: Model[];
  mapping?: Mapping[];
  workflow_edges: string[][];
  configuration?: ConfigAxis[];
  config_space_size: number;
};

export type RunResult = {
  status: string;
  metrics?: { r2?: number };
  epistemic_status?: string;
};
export type Iteration = {
  iteration: number;
  reused: number;
  best: { hypothesis: string | null; r2: number | null };
  results?: Record<string, RunResult>;
  note?: string;
};
export type CompareRow = { hypothesis: string; status: string; r2: number | null };

// ————— реальные данные (/api/real) —————
export type WellNode = { id: string; kind: "injector" | "producer" | "fusion"; label: string; task: string; models: string[] };
export type Deriv = { src: string; dst: string; via: string; reason: string };
export type WellGraph = {
  nodes: WellNode[]; edges: string[][]; derivation: Deriv[]; r_map: string;
  tasks: { id: string; label: string; hypotheses: string[] }[];
  task_edges: string[][];
};
export type Model = { id: string; label: string; class: string; python_ref?: string; config?: string; params?: string[]; desc?: string };
export type Task = { id: string; label: string; hypotheses: string[] };
export type Plan = { changed: string[]; p_ne: string[]; p_e: string[]; recompute_frac: number };
export type RealField = {
  producers: number; injectors: number; months: number; fit: string;
  r2: { CRM: number; Hybrid: number; WCT: number; OPR: number };
  bayes_factor: number; physics_verdict: string;
  graph: WellGraph;
  epistemic_status: Record<string, string>;
  concept_status: Record<string, string>;
  algorithm4: Record<string, Plan>;
};
export type OntoClass = { name: string; parent: string | null; group?: string };
export type ArchComponent = { id: string; name: string; layer: string; module: string; desc: string; deps: string[] };
export type OntoRel = { property: string; domain: string; range: string };
export type RealData = {
  domain: string;
  ve: {
    ontology: { name: string; classes: OntoClass[]; relations: OntoRel[]; total_classes: number };
    models: Model[];
    configuration: { name: string; section: string; levels: (string | number | boolean)[] }[];
    config_space_size: number;
  };
  graph_conceptual: {
    nodes: { id: string; label: string; branch: string; status: string; metric?: string; desc?: string; competes?: string[];
             equation: { formula: string; output: string; latex?: string; inputs?: string[] }; model: string; models: string[] }[];
    edges: string[][]; derivation: Deriv[]; note: string;
    tasks: Task[]; task_edges: string[][]; task_preds?: Record<string, string[]>;
    formal_text?: string; is_dag: boolean; depth: number;
  };
  demos?: {
    alg2: { added: string; new_edges: string[][]; note: string };
    alg3: { scenarios: { case: string; status: string; ok: boolean; detail: string }[]; owa_note: string };
    alg4_plan: Record<string, { changed: string; p_ne: string[]; recompute_frac: number }>;
    rule5: { acyclic: boolean; cyclic_witness: number[] };
    complexity: Record<string, { points: { n: number; count: number; law: number | string }[]; law: string; note: string }>;
  };
  architecture: { layers: string[]; components: ArchComponent[]; note: string };
  scale: { note: string; speedup_10k: string;
    points: { hypotheses: number; ELK_s: number; HermiT_s: number; wells: string }[] };
  algorithm2_example: { add: string; label: string; note: string };
  algorithm3_conditions: { n: number; text: string; ok: boolean }[];
  algorithm4: Record<string, Plan>;
  fields: Record<string, RealField>;
  theorems: Record<string, string>;
};
