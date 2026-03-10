import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass
class MarketInputs:
    generators: list[str]
    max_capacity: np.ndarray
    variable_cost: np.ndarray
    demand: np.ndarray
    renewable_scenario: np.ndarray


def load_inputs(
    demand_file: str,
    generator_file: str,
    renewable_scenario_number: int,
) -> MarketInputs:
    generator_df = pd.read_excel(generator_file)
    demand_df = pd.read_excel(demand_file)

    scenario_col = f"Renewable Scenario {renewable_scenario_number}"
    required_generator_cols = {"Generators", "Max Capacity", "Variable Cost"}
    required_demand_cols = {"Demand", scenario_col}

    if not required_generator_cols.issubset(generator_df.columns):
        missing = sorted(required_generator_cols - set(generator_df.columns))
        raise ValueError(f"Missing generator columns: {missing}")

    if not required_demand_cols.issubset(demand_df.columns):
        missing = sorted(required_demand_cols - set(demand_df.columns))
        raise ValueError(f"Missing demand columns: {missing}")

    return MarketInputs(
        generators=generator_df["Generators"].astype(str).tolist(),
        max_capacity=generator_df["Max Capacity"].to_numpy(dtype=float),
        variable_cost=generator_df["Variable Cost"].to_numpy(dtype=float),
        demand=demand_df["Demand"].to_numpy(dtype=float),
        renewable_scenario=demand_df[scenario_col].to_numpy(dtype=float),
    )


def solve_exercise_1(
    inputs: MarketInputs,
    alpha: float,
    lambda_curtailment: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    if lambda_curtailment <= 0:
        raise ValueError("lambda_curtailment must be > 0.")

    demand = inputs.demand
    renewable_available = alpha * inputs.renewable_scenario

    t_count = demand.size
    g_count = len(inputs.generators)
    pg_size = t_count * g_count
    curt_size = t_count
    n_vars = pg_size + curt_size

    # Variables:
    # - p[g,t]: dispatch for each generator and time period
    # - curt[t]: renewable curtailment per time period
    c = np.zeros(n_vars)
    c[:pg_size] = np.tile(inputs.variable_cost, t_count)
    c[pg_size:] = lambda_curtailment

    # Power balance for every time period t:
    # sum_g p[g,t] + (renewable_available[t] - curt[t]) = demand[t]
    # => sum_g p[g,t] - curt[t] = demand[t] - renewable_available[t]
    a_eq = np.zeros((t_count, n_vars))
    b_eq = demand - renewable_available

    for t in range(t_count):
        start = t * g_count
        end = start + g_count
        a_eq[t, start:end] = 1.0
        a_eq[t, pg_size + t] = -1.0

    bounds = []
    for _ in range(t_count):
        for g in range(g_count):
            bounds.append((0.0, float(inputs.max_capacity[g])))
    for t in range(t_count):
        bounds.append((0.0, float(renewable_available[t])))

    result = linprog(
        c=c,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    p_opt = result.x[:pg_size].reshape(t_count, g_count)
    curt_opt = result.x[pg_size:]
    renewable_used = renewable_available - curt_opt

    dispatch_df = pd.DataFrame(p_opt, columns=inputs.generators)
    dispatch_df.insert(0, "Time Period", np.arange(1, t_count + 1))

    balance_df = pd.DataFrame(
        {
            "Time Period": np.arange(1, t_count + 1),
            "Demand": demand,
            "Renewable Available (alpha-scaled)": renewable_available,
            "Renewable Used": renewable_used,
            "Renewable Curtailed": curt_opt,
            "Conventional Generation": p_opt.sum(axis=1),
        }
    )

    summary = {
        "total_cost": float(result.fun),
        "total_demand": float(demand.sum()),
        "total_conventional_generation": float(p_opt.sum()),
        "total_renewable_available": float(renewable_available.sum()),
        "total_renewable_used": float(renewable_used.sum()),
        "total_renewable_curtailed": float(curt_opt.sum()),
    }
    return dispatch_df, balance_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise 1 market-clearing optimization."
    )
    parser.add_argument(
        "--demand-file",
        default="Realistic Data Demand.xlsx",
        help="Path to demand and renewable scenarios Excel file.",
    )
    parser.add_argument(
        "--generator-file",
        default="Realistic Data Generators (1).xlsx",
        help="Path to generator parameters Excel file.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Renewable scenario number to use (1-5).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Renewable scaling factor in [0, 1].",
    )
    parser.add_argument(
        "--lambda-curtailment",
        type=float,
        default=10.0,
        help="Curtailment penalty (lambda > 0).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save dispatch and balance tables as CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_inputs(
        demand_file=args.demand_file,
        generator_file=args.generator_file,
        renewable_scenario_number=args.scenario,
    )

    dispatch_df, balance_df, summary = solve_exercise_1(
        inputs=inputs,
        alpha=args.alpha,
        lambda_curtailment=args.lambda_curtailment,
    )

    print("Exercise 1 solved successfully.")
    print(f"Scenario: {args.scenario}")
    print(f"alpha: {args.alpha:.3f}")
    print(f"lambda (curtailment): {args.lambda_curtailment:.3f}")
    print("")
    print("System totals:")
    print(f"- Total demand: {summary['total_demand']:.2f}")
    print(
        f"- Total renewable available: "
        f"{summary['total_renewable_available']:.2f}"
    )
    print(f"- Total renewable used: {summary['total_renewable_used']:.2f}")
    print(
        f"- Total renewable curtailed: "
        f"{summary['total_renewable_curtailed']:.2f}"
    )
    print(
        f"- Total conventional generation: "
        f"{summary['total_conventional_generation']:.2f}"
    )
    print(f"- Total objective cost: {summary['total_cost']:.2f}")
    print("")

    print("Conventional generation by unit (sum over all periods):")
    generation_by_unit = dispatch_df.drop(columns=["Time Period"]).sum()
    for unit, total in generation_by_unit.items():
        print(f"- {unit}: {total:.2f}")

    print("")
    print("First 10 periods (balance check):")
    print(balance_df.head(10).to_string(index=False))

    if args.save_results:
        dispatch_df.to_csv("exercise1_dispatch.csv", index=False)
        balance_df.to_csv("exercise1_balance.csv", index=False)
        print("")
        print("Saved: exercise1_dispatch.csv")
        print("Saved: exercise1_balance.csv")


if __name__ == "__main__":
    main()
