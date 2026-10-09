"""Chromosome representation for the GA."""
import numpy as np
from .config import ChromosomeConfig


class Chromosome:
    """
    Real-valued chromosome encoding GA weights.
    Genes: [w_cosine, w_bm25, w_diversity, w_position, w_coverage, top_k]
    """

    def __init__(self, genes: np.ndarray, config: ChromosomeConfig):
        self.genes = genes.copy()
        self.config = config
        self.fitness = None

    @classmethod
    def random(cls, config: ChromosomeConfig) -> "Chromosome":
        genes = np.array([
            np.random.uniform(low, high)
            for _, low, high in config.genes
        ])
        return cls(genes, config)

    def to_weights_dict(self) -> dict:
        """Decode chromosome into named dict. Normalizes weights, rounds top_k."""
        raw = {name: self.genes[i] for i, (name, _, _) in enumerate(self.config.genes)}

        weight_names = ["w_cosine", "w_bm25", "w_diversity", "w_position", "w_coverage"]
        weight_sum = sum(raw[w] for w in weight_names)
        if weight_sum > 0:
            for w in weight_names:
                raw[w] = raw[w] / weight_sum
        else:
            for w in weight_names:
                raw[w] = 1.0 / len(weight_names)

        raw["top_k"] = max(3, min(15, int(round(raw["top_k"]))))
        return raw

    def clip(self):
        for i, (_, low, high) in enumerate(self.config.genes):
            self.genes[i] = np.clip(self.genes[i], low, high)

    def copy(self) -> "Chromosome":
        c = Chromosome(self.genes.copy(), self.config)
        c.fitness = self.fitness
        return c

    def __repr__(self):
        weights = self.to_weights_dict()
        fit = f", fitness={self.fitness:.4f}" if self.fitness is not None else ""
        return f"Chromosome({weights}{fit})"
