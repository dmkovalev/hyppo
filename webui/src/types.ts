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
export type RealHyp = {
  id: string; label: string; branch: string; description: string;
  model: string | null; model_classes: string[]; hyperparam_axes: string[];
  equation?: { formula: string; output: string }; variables?: string[];
};
export type RealField = {
  producers: number; injectors: number; months: number; train: number[];
  r2: { CRM: number; Hybrid: number; WCT: number; OPR: number };
  bayes_factor: number; physics_verdict: string;
  epistemic_status: Record<string, string>;
};
export type RealData = {
  domain: string;
  ve: {
    ontology: { name: string; iri: string; classes: string[];
                object_properties: { name: string }[]; data_properties: { name: string }[] };
    hypotheses: RealHyp[];
    mapping: { hypothesis: string; model: string | null; model_classes: string[] }[];
    configuration: { name: string; section: string; levels: (string | number | boolean)[] }[];
    config_space_size: number;
  };
  graph: { nodes: string[]; edges: string[][]; derivation: { src: string; dst: string; via: string; reason: string }[] };
  algorithm2_example: { add: string; label: string; equation: string; output: string; new_edges: string[][]; note: string };
  algorithm3_conditions: { n: number; text: string; ok: boolean }[];
  algorithm4: Record<string, { changed: string[]; p_ne: string[]; p_e: string[]; recompute_frac: number }>;
  fields: Record<string, RealField>;
  theorems: Record<string, string>;
};
