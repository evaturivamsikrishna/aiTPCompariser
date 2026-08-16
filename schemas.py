"""
schemas.py

This module defines the Pydantic models that enforce data structures throughout the LLM
benchmarking pipeline. These schemas are crucial for data validation, serialization, and
ensuring type safety across different components of the application, from CSV ingestion to
final evaluation.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

# Weights must sum to 1.0. Inverse metrics (redundancy, uniqueness, scope drift): 5 = none of the problem.
METRIC_WEIGHTS = {
    "requirement_coverage": 0.12,
    "match_confidence": 0.07,
    "area_coverage": 0.07,
    "source_coverage": 0.11,
    "source_fidelity": 0.07,
    "peer_coverage": 0.10,
    "edge_case_quality": 0.07,
    "actionability": 0.08,
    "str_completeness": 0.09,
    "expected_result_quality": 0.09,
    "redundancy": 0.05,
    "procedure_uniqueness": 0.04,
    "scope_drift": 0.03,
    "traceability": 0.01,
    "llm_gdd_coverage": 0.14,
    "llm_gdd_fidelity": 0.08,
    "llm_tester_readiness": 0.08,
}

# --- Input and State Schemas ---

class CSVRequirement(BaseModel):
    """
    Represents a single, parsed requirement from the input CSV file.
    """
    id: str = Field(..., description="Unique identifier for the requirement.")
    description: str = Field(...,
                             description="The full text of the software requirement or user story.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary to hold any extra columns from the CSV as key-value pairs.")

class BenchmarkState(TypedDict):
    """
    Represents the state of the LangGraph. This TypedDict is passed between nodes, accumulating
    data as the pipeline progresses.
    """
    csv_file_path: str
    parsed_requirements: List[CSVRequirement]
    raw_outputs: List[Dict[str, Any]]  # e.g., [{"model_name": "gpt-4o", "output": "...", "error": None}]
    normalized_plans: List["GeneratedTestPlan"]
    evaluations: List["ModelEvaluation"]
    final_report_md: str
    final_report_csv: str

# --- AI Model Output Schemas ---

class TestCase(BaseModel):
    """
    Represents a single, atomic test case designed to verify a piece of functionality.
    """
    id: Optional[str] = Field(None, description="A unique identifier for the test case, e.g., 'TC-001'.")
    title: str = Field(..., description="A brief, descriptive title for the test case.")
    area: Optional[str] = Field(
        None, description="Area/feature grouping, e.g. 'Mission 1.1' or 'Squatter’s Den / Loot'.")
    preconditions: List[str] = Field(
        default_factory=list,
        description="A list of conditions that must be met before executing the test.")
    steps: List[str] = Field(default_factory=list,
                             description="A sequence of actions to be performed by the tester.")
    expected_result: Optional[str] = Field(
        None, description="The expected outcome after executing the test steps.")
    edge_cases_addressed: List[str] = Field(
        default_factory=list,
        description="Specific edge cases or boundary conditions this test covers.")

class GeneratedTestPlan(BaseModel):
    """
    The core structured output from a generator LLM or a parsed CSV, representing its
    comprehensive plan for testing.
    """
    model_name: str = Field(..., description="The name of the LLM that generated this test plan.")
    test_strategy: Optional[str] = Field(None, description="An overview of the approach and methodology for testing.")
    requirements_covered: List[str] = Field(
        default_factory=list, description="A list of requirement IDs that this test plan aims to cover.")
    test_cases: List[TestCase] = Field(..., description="A list of detailed, individual test cases.")

# --- Evaluation Schemas ---

class MetricScore(BaseModel):
    """
    A structure to hold the score and justification for a single evaluation metric.
    """
    score: int = Field(..., ge=1, le=5, description="The score from 1 to 5 for the metric.")
    reasoning: str = Field(..., description="A brief justification for the assigned score.")


class CoverageAlignment(BaseModel):
    """
    Semantic mapping from one stock test intent to candidate test case(s).
    Matching is by intent/title, never by requiring identical STR or expected results.
    """
    stock_id: str = Field(..., description="ID of the stock test case being matched.")
    stock_title: str = Field(..., description="Title/intent of the stock test case.")
    matched_candidate_ids: List[str] = Field(
        default_factory=list,
        description="Candidate test case IDs that cover this stock intent. Empty if unmatched.",
    )
    match_type: Literal["semantic", "partial", "unmatched"] = Field(
        ...,
        description="semantic = full intent covered; partial = related but incomplete; unmatched = missing.",
    )
    best_similarity: int = Field(
        0, ge=0, le=100, description="Best title similarity (0-100) against generated cases.")
    rationale: str = Field(..., description="Why this mapping was chosen.")


class ModelEvaluation(BaseModel):
    """
    Complete evaluation for a generated test plan. The stock plan is never scored.
    overall_score is calculated from METRIC_WEIGHTS.
    """
    model_name: str = Field(..., description="The name of the generated plan being evaluated.")
    alignments: List[CoverageAlignment] = Field(default_factory=list)
    extra_in_scope_case_ids: List[str] = Field(default_factory=list)
    extra_out_of_scope_case_ids: List[str] = Field(default_factory=list)
    source_alignments: List[CoverageAlignment] = Field(default_factory=list)
    requirement_coverage: Optional[MetricScore] = None
    match_confidence: Optional[MetricScore] = None
    area_coverage: Optional[MetricScore] = None
    source_coverage: Optional[MetricScore] = None
    source_fidelity: Optional[MetricScore] = None
    peer_coverage: Optional[MetricScore] = None
    edge_case_quality: Optional[MetricScore] = None
    actionability: Optional[MetricScore] = None
    str_completeness: Optional[MetricScore] = None
    expected_result_quality: Optional[MetricScore] = None
    redundancy: Optional[MetricScore] = None
    procedure_uniqueness: Optional[MetricScore] = None
    scope_drift: Optional[MetricScore] = None
    traceability: Optional[MetricScore] = None
    llm_gdd_coverage: Optional[MetricScore] = None
    llm_gdd_fidelity: Optional[MetricScore] = None
    llm_tester_readiness: Optional[MetricScore] = None
    llm_alignments: List[CoverageAlignment] = Field(default_factory=list)
    llm_summary: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    overall_score: Optional[float] = Field(
        None, description="The final weighted score for the generated plan.")
    error: Optional[str] = Field(None, description="Any error message if the evaluation failed.")

    @model_validator(mode='after')
    def calculate_overall_score(self) -> 'ModelEvaluation':
        self.recompute_overall_score()
        return self

    def recompute_overall_score(self) -> None:
        if self.error:
            self.overall_score = 0.0
            return
        weighted_score = 0.0
        present_weight = 0.0
        for field_name, weight in METRIC_WEIGHTS.items():
            metric = getattr(self, field_name, None)
            if metric is None:
                continue
            weighted_score += metric.score * weight
            present_weight += weight
        if present_weight == 0:
            return
        self.overall_score = round(weighted_score / present_weight, 2)

# --- Head-to-head comparison ---

class ClusterMember(BaseModel):
    model_name: str
    case_id: str
    title: str
    area: Optional[str] = None


class IntentCluster(BaseModel):
    label: str
    members: List[ClusterMember] = Field(default_factory=list)


class PairwiseOverlap(BaseModel):
    model_a: str
    model_b: str
    shared: int
    union: int
    jaccard: float


class GddBeatCoverage(BaseModel):
    beat_id: str
    beat_text: str
    origin: str = ""
    covered_by: List[str] = Field(default_factory=list)
    match_type: Literal["semantic", "partial", "unmatched"] = "unmatched"


class CrossPlanComparison(BaseModel):
    cluster_count: int = 0
    consensus: List[IntentCluster] = Field(default_factory=list)
    partial_overlap: List[IntentCluster] = Field(default_factory=list)
    unique_by_model: Dict[str, List[IntentCluster]] = Field(default_factory=dict)
    pairwise: List[PairwiseOverlap] = Field(default_factory=list)
    gdd_beats: List[GddBeatCoverage] = Field(default_factory=list)
    model_peer_ratios: Dict[str, float] = Field(default_factory=dict)
    model_peer_notes: Dict[str, str] = Field(default_factory=dict)


# --- CSV Parser Helper Schemas ---

class ComparisonRunMeta(BaseModel):
    """How a comparison was produced — filenames and filters, never API keys."""
    model_names: List[str] = Field(default_factory=list)
    gdd_filename: Optional[str] = None
    stock_filename: Optional[str] = None
    selected_areas: List[str] = Field(default_factory=list)
    selected_gdd_origins: List[str] = Field(default_factory=list)
    judge_backend: str = "off"
    judge_model: str = ""
    has_stock: bool = False
    has_gdd: bool = False
    overall_scores: Dict[str, float] = Field(default_factory=dict)


class ComparisonRunSummary(BaseModel):
    """Index row for the history sidebar."""
    id: str
    created_at: str
    title: str
    model_names: List[str] = Field(default_factory=list)
    overall_scores: Dict[str, float] = Field(default_factory=dict)
    judge_backend: str = "off"


class ComparisonRun(BaseModel):
    """A saved comparison entry (scores + mappings, not original CSV bytes)."""
    id: str
    created_at: str
    title: str
    meta: ComparisonRunMeta = Field(default_factory=ComparisonRunMeta)
    evaluations: List[ModelEvaluation] = Field(default_factory=list)
    cross: Optional[CrossPlanComparison] = None


class ColumnMapping(BaseModel):
    """
    A schema used by the LLM-assisted CSV parser to identify which columns
    map to the canonical 'id' and 'description' fields.
    """
    id_column: Optional[str] = Field(
        None,
        description="The name of the column header to be used as the requirement ID. Should be unique per row."
    )
    title_column: str = Field(...,
                              description="The name of the column header containing the test case title or description.")
    area_column: Optional[str] = Field(
        None, description="Area/feature column, e.g. AREA / FEATURE.")
    steps_column: Optional[str] = Field(None,
                                        description="The name of the column header containing the test steps.")
    expected_result_column: Optional[str] = Field(
        None, description="The name of the column header containing the expected result.")