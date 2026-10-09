"""
Genetic Algorithm engine with adaptive restarting and local solution generation.
Based on: Dao, Abhary & Marian (2017) — "An innovative framework for designing
genetic algorithm structures", Expert Systems With Applications.
"""
import numpy as np
from typing import Callable
from .chromosome import Chromosome
from .config import GAConfig, ChromosomeConfig


class GeneticAlgorithm:
    """
    GA with three innovations from Dao et al. (2017):
    1. Adaptive restarting — restarts when stuck, preserving elite solutions
    2. Local solution generation — explores neighbourhood of best chromosome
    3. Systematic parameter control via GAConfig
    """

    def __init__(self, ga_config: GAConfig, chromo_config: ChromosomeConfig):
        self.ga_config    = ga_config
        self.chromo_config = chromo_config
        self.history: list = []

    def run(self, fitness_fn: Callable, verbose: bool = True) -> Chromosome:
        """
        Execute GA optimisation loop.

        Returns the best Chromosome found across all generations and restarts.
        """
        cfg = self.ga_config
        population  = [Chromosome.random(self.chromo_config)
                       for _ in range(cfg.population_size)]
        global_best   = None
        step_size     = cfg.max_step_size_pct
        no_improve    = 0
        generation    = 0
        restart_count = 0

        while generation < cfg.max_generations:
            # --- Crossover ---
            offspring_cx = []
            for _ in range(max(1, cfg.crossover_rate // 2)):
                p1, p2 = self._select_parents(population)
                c1, c2 = self._crossover(p1, p2)
                offspring_cx.extend([c1, c2])

            # --- Mutation ---
            offspring_mut = [
                self._mutate(population[np.random.randint(len(population))])
                for _ in range(cfg.mutation_rate)
            ]

            # --- Local solutions around current best ---
            current_best  = max(
                population,
                key=lambda c: c.fitness if c.fitness is not None else -np.inf
            )
            local_solutions = self._generate_local_solutions(current_best, step_size)

            # --- Evaluate unevaluated candidates ---
            all_candidates = population + offspring_cx + offspring_mut + local_solutions
            for chrom in all_candidates:
                if chrom.fitness is None:
                    chrom.fitness = fitness_fn(chrom)

            # --- Selection: keep top P ---
            all_candidates.sort(key=lambda c: c.fitness, reverse=True)
            population = [c.copy() for c in all_candidates[:cfg.population_size]]

            # --- Track global best ---
            gen_best = population[0]
            if global_best is None or gen_best.fitness > global_best.fitness:
                global_best = gen_best.copy()
                no_improve  = 0
            else:
                no_improve += 1

            avg_fitness = float(np.mean([c.fitness for c in population]))
            self.history.append({
                "generation":          generation,
                "best_fitness":        gen_best.fitness,
                "global_best_fitness": global_best.fitness,
                "avg_fitness":         avg_fitness,
                "restart":             False,
            })

            if verbose and generation % 10 == 0:
                print(f"Gen {generation:>4} | Best: {gen_best.fitness:.4f} | "
                      f"Global: {global_best.fitness:.4f} | Avg: {avg_fitness:.4f} | "
                      f"Step: {step_size:.4f} | Restarts: {restart_count}")

            # --- Decay step size ---
            step_size = max(cfg.step_size_min,
                            step_size - cfg.step_reduction_rate * step_size)

            # --- Adaptive restart ---
            if no_improve >= cfg.adaptive_restart_threshold:
                restart_count += 1
                if verbose:
                    print(f"  >>> RESTART #{restart_count} at gen {generation} "
                          f"(no improvement for {cfg.adaptive_restart_threshold} gens)")
                elites   = [c.copy() for c in population[:cfg.elite_count]]
                new_pop  = [Chromosome.random(self.chromo_config)
                            for _ in range(cfg.population_size - cfg.elite_count)]
                population  = elites + new_pop
                step_size   = cfg.max_step_size_pct
                no_improve  = 0
                self.history[-1]["restart"] = True

            generation += 1

        if verbose:
            print(f"\nGA done: {generation} generations, {restart_count} restarts")
            print(f"Best fitness: {global_best.fitness:.4f}")
            print(f"Best weights: {global_best.to_weights_dict()}")

        return global_best

    def get_convergence_data(self) -> list:
        return self.history

    # ── Private helpers ────────────────────────────────────────────────────

    def _select_parents(self, population: list) -> tuple:
        """Roulette-wheel selection proportional to fitness."""
        fitnesses = np.array([c.fitness if c.fitness is not None else 0.0
                              for c in population])
        shifted = fitnesses - fitnesses.min() + 1e-6
        probs   = shifted / shifted.sum()
        indices = np.random.choice(len(population), size=2, replace=False, p=probs)
        return population[indices[0]], population[indices[1]]

    def _crossover(self, p1: Chromosome, p2: Chromosome) -> tuple:
        """Single-point crossover on real-valued gene vectors."""
        point  = np.random.randint(1, len(p1.genes))
        genes1 = np.concatenate([p1.genes[:point], p2.genes[point:]])
        genes2 = np.concatenate([p2.genes[:point], p1.genes[point:]])
        c1, c2 = Chromosome(genes1, self.chromo_config), Chromosome(genes2, self.chromo_config)
        c1.clip(); c2.clip()
        return c1, c2

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """Random mutation: replace one random gene with a new random value."""
        mutant = chromosome.copy()
        mutant.fitness = None
        idx = np.random.randint(len(mutant.genes))
        _, low, high = self.chromo_config.genes[idx]
        mutant.genes[idx] = np.random.uniform(low, high)
        return mutant

    def _generate_local_solutions(self, best: Chromosome,
                                  step_size: float) -> list:
        """
        Generate 2n local solutions by nudging each gene of the best chromosome
        up and down. Randomly sample L% of them (Dao et al. 2017, Table 1).
        """
        all_local = []
        for i, (_, low, high) in enumerate(self.chromo_config.genes):
            gene_range = high - low
            max_delta  = step_size * gene_range

            for sign in (+1, -1):
                variant = best.copy()
                variant.fitness = None
                delta = np.random.uniform(0, max_delta)
                variant.genes[i] = best.genes[i] + sign * delta
                variant.clip()
                all_local.append(variant)

        n_select = max(1, int(self.ga_config.local_solution_pct * len(all_local)))
        if n_select < len(all_local):
            indices  = np.random.choice(len(all_local), size=n_select, replace=False)
            all_local = [all_local[i] for i in indices]

        return all_local
