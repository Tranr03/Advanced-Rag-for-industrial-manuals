"""GA Reranker configuration and parameter ranges."""
from dataclasses import dataclass, field


@dataclass
class ScoringConfig:
    """Ranges and settings for scoring functions."""
    retrieval_top_n: int = 50
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class GAConfig:
    """GA hyperparameters."""
    population_size: int = 50
    crossover_rate: int = 22
    mutation_rate: int = 12
    max_generations: int = 200
    adaptive_restart_threshold: int = 25
    local_solution_pct: float = 0.67
    max_step_size_pct: float = 0.20
    step_reduction_rate: float = 0.01
    step_size_min: float = 0.01
    elite_count: int = 2
    num_runs: int = 5
    fitness_alpha: float = 0.3
    fitness_beta: float = 0.2
    fitness_gamma: float = 0.3
    fitness_delta: float = 0.2


@dataclass
class ChromosomeConfig:
    """Defines what the chromosome encodes and the search ranges."""
    genes: list = field(default_factory=lambda: [
        ("w_cosine",    0.0, 1.0),
        ("w_bm25",      0.0, 1.0),
        ("w_diversity", 0.0, 1.0),
        ("w_position",  0.0, 1.0),
        ("w_coverage",  0.0, 1.0),
        ("top_k",       3.0, 15.0),
    ])
