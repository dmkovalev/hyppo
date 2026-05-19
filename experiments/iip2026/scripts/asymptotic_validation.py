"""
Асимптотическая валидация алгоритмов диссертации.

Эксперимент 1: Алгоритм 1 (build_lattice) — T(|H|) ∝ O(|H|²)
Эксперимент 2: Алгоритм 2 (add_hypothesis) — T(|H|) ∝ O(|H|), + сравнение с перестройкой
Эксперимент 3: Алгоритм 4 (planning) — |P_ne|/|H| vs cache_rate
"""

import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# --- Structures and Algorithms ---

class Structure:
    """Структура S(E, V) — набор уравнений и переменных."""
    def __init__(self, equations: list[set[str]]):
        self.equations = equations  # list of sets of variable names
        self.variables = set()
        for eq in equations:
            self.variables |= eq

    @property
    def size(self):
        return sum(len(eq) for eq in self.equations)

    def is_complete(self):
        return len(self.equations) == len(self.variables)


def union_structures(s1: Structure, s2: Structure) -> Structure:
    return Structure(s1.equations + s2.equations)


def transitive_closure(structure: Structure) -> set[tuple[str, str]]:
    """Построить транзитивное замыкание причинно-следственных зависимостей."""
    if not structure.is_complete():
        return set()

    # Простое полное причинно-следственное отображение: greedy matching
    vars_list = list(structure.variables)
    eqs = structure.equations
    mapping = {}  # eq_idx -> var
    used_vars = set()

    for i, eq in enumerate(eqs):
        for v in eq:
            if v not in used_vars:
                mapping[i] = v
                used_vars.add(v)
                break

    # Прямые зависимости
    direct = set()
    for i, eq in enumerate(eqs):
        if i in mapping:
            target = mapping[i]
            for v in eq:
                if v != target:
                    direct.add((v, target))

    # Транзитивное замыкание через BFS
    adj = defaultdict(set)
    for a, b in direct:
        adj[a].add(b)

    closure = set()
    for start in structure.variables:
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    closure.add((start, neighbor))
                    stack.append(neighbor)

    return closure


def build_lattice(workflow_edges, hypotheses):
    """Алгоритм 1: Построение решётки гипотез."""
    n = len(hypotheses)
    tc_calls = 0

    # Построить множество достижимости в DAG
    reachable = defaultdict(set)
    adj = defaultdict(set)
    for u, v in workflow_edges:
        adj[u].add(v)

    for start in range(n):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    reachable[start].add(nb)
                    stack.append(nb)

    # Основной цикл: для каждой пары (h_i, h_j) где h_j достижим из h_i
    lattice_edges = set()
    all_closures = set()

    for i in range(n):
        for j in reachable[i]:
            s_union = union_structures(hypotheses[i], hypotheses[j])
            if s_union.is_complete():
                tc_calls += 1
                closure = transitive_closure(s_union)
                all_closures |= closure

    return lattice_edges, tc_calls


def add_hypothesis(workflow_edges, hypotheses, h_add, existing_lattice_edges):
    """Алгоритм 2: Добавление новой гипотезы."""
    n = len(hypotheses)
    tc_calls = 0

    for i in range(n):
        s_union = union_structures(hypotheses[i], h_add)
        if s_union.is_complete():
            tc_calls += 1
            transitive_closure(s_union)

    return tc_calls


def plan_experiment(lattice_adj, n_hypotheses, cache_rate):
    """Алгоритм 4: Планирование с повторным использованием."""
    # Случайный кэш
    cached = set(random.sample(range(n_hypotheses),
                               int(cache_rate * n_hypotheses)))

    # Топологическая сортировка
    in_degree = defaultdict(int)
    adj = defaultdict(set)
    for u, v in lattice_adj:
        adj[u].add(v)
        in_degree[v] += 1

    queue = [i for i in range(n_hypotheses) if in_degree[i] == 0]
    topo_order = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for nb in adj[node]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    # Планирование
    p_ne = set()
    p_e = set()
    processed = set()

    for h in topo_order:
        if h in processed:
            continue
        if h not in cached:
            # Нужен пересчёт + все зависимые
            p_ne.add(h)
            # Найти все зависимые (транзитивно)
            stack = [h]
            while stack:
                node = stack.pop()
                for dep in adj[node]:
                    if dep not in p_ne:
                        p_ne.add(dep)
                        stack.append(dep)
            processed |= p_ne
        else:
            p_e.add(h)
            processed.add(h)

    return len(p_ne) / max(n_hypotheses, 1)


# --- Data Generation ---

def generate_random_structure(n_eq=5, var_pool_size=20):
    """Генерация случайной структуры S(E,V) с |E|=|V|=n_eq."""
    all_vars = [f"x_{k}" for k in range(var_pool_size)]
    # Выбираем n_eq уникальных переменных
    chosen_vars = random.sample(all_vars, n_eq)
    equations = []
    for i in range(n_eq):
        # Каждое уравнение содержит свою "целевую" переменную + 1-3 других
        eq_vars = {chosen_vars[i]}
        n_extra = random.randint(1, min(3, n_eq - 1))
        extras = random.sample([v for v in chosen_vars if v != chosen_vars[i]], n_extra)
        eq_vars.update(extras)
        equations.append(eq_vars)
    return Structure(equations)


def generate_random_dag(n_nodes, p=0.3):
    """Генерация случайного DAG (Erdos-Renyi + ацикликизация)."""
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < p:
                edges.append((i, j))
    return edges


def generate_ba_dag(n_nodes, m=2):
    """Генерация DAG по модели Barabási-Albert (scale-free).

    Каждый новый узел присоединяется к m существующим с вероятностью,
    пропорциональной их текущей степени. Рёбра направлены от нового к
    существующим (гарантирует ацикличность).
    """
    edges = []
    degrees = [0] * n_nodes
    # Начинаем с клики из m+1 узлов
    for i in range(min(m + 1, n_nodes)):
        for j in range(i + 1, min(m + 1, n_nodes)):
            edges.append((i, j))
            degrees[i] += 1
            degrees[j] += 1

    for new_node in range(m + 1, n_nodes):
        # Выбрать m целей с вероятностью пропорциональной степени
        total_deg = sum(degrees[:new_node])
        if total_deg == 0:
            targets = list(range(min(m, new_node)))
        else:
            probs = [degrees[i] / total_deg for i in range(new_node)]
            targets = list(np.random.choice(new_node, size=min(m, new_node),
                                            replace=False, p=probs))
        for t in targets:
            edges.append((t, new_node))  # t → new_node (DAG order)
            degrees[t] += 1
            degrees[new_node] += 1

    return edges


def generate_random_lattice_edges(n_nodes, p=0.2):
    """Генерация случайных рёбер решётки для Алгоритма 4."""
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < p:
                edges.append((i, j))
    return edges


# --- Experiments ---

def experiment_1(h_values, n_repeats=30):
    """Эксперимент 1: T(|H|) для Алгоритма 1."""
    print("=== Эксперимент 1: Построение решётки (Алгоритм 1) ===")
    results = {}

    for n_h in h_values:
        times = []
        tc_counts = []
        for rep in range(n_repeats):
            random.seed(42 + rep)
            np.random.seed(42 + rep)
            hypotheses = [generate_random_structure() for _ in range(n_h)]
            edges = generate_random_dag(n_h, p=0.3)

            t0 = time.perf_counter()
            _, tc = build_lattice(edges, hypotheses)
            t1 = time.perf_counter()

            times.append(t1 - t0)
            tc_counts.append(tc)

        results[n_h] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "mean_tc": np.mean(tc_counts),
        }
        print(f"  |H|={n_h:4d}: T={results[n_h]['mean_time']:.4f}±{results[n_h]['std_time']:.4f}s, "
              f"TC={results[n_h]['mean_tc']:.0f}")

    return results


def experiment_2(h_values, n_repeats=30):
    """Эксперимент 2: T_add(|H|) для Алгоритма 2 + сравнение с перестройкой."""
    print("\n=== Эксперимент 2: Добавление гипотезы (Алгоритм 2) ===")
    results = {}

    for n_h in h_values:
        t_add_list = []
        t_rebuild_list = []

        for rep in range(n_repeats):
            random.seed(42 + rep)
            np.random.seed(42 + rep)
            hypotheses = [generate_random_structure() for _ in range(n_h)]
            h_new = generate_random_structure()
            edges = generate_random_dag(n_h, p=0.3)

            # Добавление
            t0 = time.perf_counter()
            add_hypothesis(edges, hypotheses, h_new, set())
            t1 = time.perf_counter()
            t_add_list.append(t1 - t0)

            # Перестройка
            all_hyp = hypotheses + [h_new]
            edges_new = generate_random_dag(n_h + 1, p=0.3)
            t0 = time.perf_counter()
            build_lattice(edges_new, all_hyp)
            t1 = time.perf_counter()
            t_rebuild_list.append(t1 - t0)

        results[n_h] = {
            "mean_add": np.mean(t_add_list),
            "std_add": np.std(t_add_list),
            "mean_rebuild": np.mean(t_rebuild_list),
            "std_rebuild": np.std(t_rebuild_list),
            "speedup": np.mean(t_rebuild_list) / max(np.mean(t_add_list), 1e-9),
        }
        print(f"  |H|={n_h:4d}: T_add={results[n_h]['mean_add']:.4f}s, "
              f"T_rebuild={results[n_h]['mean_rebuild']:.4f}s, "
              f"k={results[n_h]['speedup']:.1f}x")

    return results


def experiment_3(h_values_short, cache_rates, n_repeats=30):
    """Эксперимент 3: |P_ne|/|H| vs cache_rate для Алгоритма 4."""
    print("\n=== Эксперимент 3: Планирование (Алгоритм 4) ===")
    results = {}

    for n_h in h_values_short:
        results[n_h] = {}
        for r in cache_rates:
            ratios = []
            for rep in range(n_repeats):
                random.seed(42 + rep)
                lattice_edges = generate_random_lattice_edges(n_h, p=0.2)
                ratio = plan_experiment(lattice_edges, n_h, r)
                ratios.append(ratio)
            results[n_h][r] = {
                "mean": np.mean(ratios),
                "std": np.std(ratios),
            }
        print(f"  |H|={n_h}: recomp@r=0.5 = {results[n_h][0.5]['mean']:.2f}")

    return results


# --- Plotting ---

def plot_experiment_1(results, output_path):
    """log-log график T(|H|) с O(|H|²) overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    hs = sorted(results.keys())
    means = [results[h]["mean_time"] for h in hs]
    stds = [results[h]["std_time"] for h in hs]

    ax.errorbar(hs, means, yerr=stds, fmt="o-", color="steelblue",
                capsize=3, label="Эксперимент")

    # Fit O(|H|^a)
    log_h = np.log(hs)
    log_t = np.log(means)
    a, b = np.polyfit(log_h, log_t, 1)
    fit_t = np.exp(b) * np.array(hs) ** a
    r2 = 1 - np.sum((np.array(log_t) - (a * log_h + b)) ** 2) / \
             np.sum((np.array(log_t) - np.mean(log_t)) ** 2)

    ax.plot(hs, fit_t, "--", color="tomato",
            label=f"$O(|H|^{{{a:.2f}}})$, $R^2={r2:.3f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число гипотез $|H|$", fontsize=12)
    ax.set_ylabel("Время, с", fontsize=12)
    ax.set_title("Алгоритм 1: построение решётки гипотез")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Сохранено: {output_path} (a={a:.2f}, R^2={r2:.3f})")
    plt.close(fig)


def plot_experiment_2(results, output_path):
    """T_add vs T_rebuild + speedup."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    hs = sorted(results.keys())
    t_add = [results[h]["mean_add"] for h in hs]
    t_rebuild = [results[h]["mean_rebuild"] for h in hs]
    speedups = [results[h]["speedup"] for h in hs]

    ax1.plot(hs, t_add, "o-", color="steelblue", label="Добавление (Алг. 2)")
    ax1.plot(hs, t_rebuild, "s--", color="tomato", label="Перестройка (Алг. 1)")
    ax1.set_xlabel("Число гипотез $|H|$", fontsize=12)
    ax1.set_ylabel("Время, с", fontsize=12)
    ax1.set_title("Сравнение: добавление vs перестройка")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(hs)), speedups, color="seagreen", alpha=0.8)
    ax2.set_xticks(range(len(hs)))
    ax2.set_xticklabels([str(h) for h in hs])
    ax2.set_xlabel("Число гипотез $|H|$", fontsize=12)
    ax2.set_ylabel("Ускорение $k$", fontsize=12)
    ax2.set_title("Коэффициент ускорения $k = T_{rebuild}/T_{add}$")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Сохранено: {output_path}")
    plt.close(fig)


def plot_experiment_3(results, cache_rates, output_path):
    """|P_ne|/|H| vs cache rate."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    for idx, n_h in enumerate(sorted(results.keys())):
        means = [results[n_h][r]["mean"] for r in cache_rates]
        stds = [results[n_h][r]["std"] for r in cache_rates]
        ax.errorbar(cache_rates, means, yerr=stds, fmt="o-",
                    color=colors[idx % len(colors)], capsize=2,
                    label=f"$|H|={n_h}$")

    # Идеальная прямая (без каскадного эффекта)
    ax.plot(cache_rates, [1 - r for r in cache_rates], "k--", alpha=0.4,
            label="Идеал: $1-r$")

    ax.set_xlabel("Доля кэшированных результатов $r$", fontsize=12)
    ax.set_ylabel("Доля пересчёта $|P_{ne}|/|H|$", fontsize=12)
    ax.set_title("Алгоритм 4: эффективность планирования")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Сохранено: {output_path}")
    plt.close(fig)


# --- Main ---

def bootstrap_power_law_ci(h_values, mean_times, n_boot=1000, alpha=0.05):
    """Bootstrap 95% CI для показателя степени a в T ~ c * |H|^a."""
    log_h = np.log(h_values)
    log_t = np.log(mean_times)
    n = len(log_h)
    a_samples = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a_b, _ = np.polyfit(log_h[idx], log_t[idx], 1)
        a_samples.append(a_b)
    a_samples = np.sort(a_samples)
    lo = a_samples[int(n_boot * alpha / 2)]
    hi = a_samples[int(n_boot * (1 - alpha / 2))]
    return lo, hi


def main():
    import sys
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")
    print(f"Seed: 42")

    random.seed(42)
    np.random.seed(42)

    images_dir = Path("F:/git-repos/wf/diss/thesis/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    h_values = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500]
    h_values_short = [10, 20, 50, 100]
    cache_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Эксперимент 1 (Erdos-Renyi)
    res1 = experiment_1(h_values)
    plot_experiment_1(res1, images_dir / "asymp_build_lattice.pdf")

    # Bootstrap CI для показателя степени
    hs = sorted(res1.keys())
    means = [res1[h]["mean_time"] for h in hs]
    ci_lo, ci_hi = bootstrap_power_law_ci(np.array(hs, dtype=float),
                                           np.array(means))
    print(f"\n  Bootstrap 95% CI для a: [{ci_lo:.2f}, {ci_hi:.2f}]")

    # Эксперимент 1b (Barabási-Albert) — анализ чувствительности к топологии
    print("\n=== Эксперимент 1b: Алгоритм 1 на BA-графах ===")
    res1b = {}
    for n_h in h_values:
        times = []
        for _ in range(30):
            hypotheses = [generate_random_structure() for _ in range(n_h)]
            edges = generate_ba_dag(n_h, m=2)
            t0 = time.perf_counter()
            _, tc = build_lattice(edges, hypotheses)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        res1b[n_h] = {"mean_time": np.mean(times), "std_time": np.std(times)}
        print(f"  |H|={n_h:4d}: T={res1b[n_h]['mean_time']:.4f}±{res1b[n_h]['std_time']:.4f}s")

    # Анализ чувствительности к p (Erdos-Renyi)
    print("\n=== Чувствительность к p (ER) для |H|=100 ===")
    for p_val in [0.1, 0.3, 0.5]:
        times = []
        for _ in range(30):
            hypotheses = [generate_random_structure() for _ in range(100)]
            edges = generate_random_dag(100, p=p_val)
            t0 = time.perf_counter()
            build_lattice(edges, hypotheses)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        print(f"  p={p_val}: T={np.mean(times):.4f}±{np.std(times):.4f}s")

    # Эксперимент 2
    res2 = experiment_2(h_values)
    plot_experiment_2(res2, images_dir / "asymp_add_hypothesis.pdf")

    # Эксперимент 3
    res3 = experiment_3(h_values_short, cache_rates)
    plot_experiment_3(res3, cache_rates, images_dir / "asymp_planning_cache.pdf")

    print("\n=== Готово ===")


if __name__ == "__main__":
    main()
