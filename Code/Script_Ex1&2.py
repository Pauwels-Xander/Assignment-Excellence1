import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Paths relative to this script: script in Code/, data in Data/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "Data"
DEFAULT_DEMAND_FILE = str(_DATA_DIR / "Realistic Data Demand.xlsx")
DEFAULT_GENERATOR_FILE = str(_DATA_DIR / "Realistic Data Generators (1).xlsx")
DEFAULT_OUTPUT_DIR = str(_DATA_DIR)


@dataclass
class MarketInputs:
    generators: list[str]
    max_capacity: np.ndarray
    variable_cost: np.ndarray
    demand: np.ndarray
    renewable_scenario: np.ndarray


@dataclass
class MarketSolution:
    dispatch: pd.DataFrame
    balance: pd.DataFrame
    prices: pd.DataFrame
    summary: dict


def load_inputs(
    demand_file: str,
    generator_file: str,
    renewable_scenario_number: int,
) -> MarketInputs:
    demand_path = (
        Path(demand_file)
        if Path(demand_file).is_absolute()
        else _PROJECT_ROOT / demand_file
    )
    gen_path = (
        Path(generator_file)
        if Path(generator_file).is_absolute()
        else _PROJECT_ROOT / generator_file
    )
    generator_df = pd.read_excel(gen_path)
    demand_df = pd.read_excel(demand_path)

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


def solve_market_clearing(
    inputs: MarketInputs,
    alpha: float,
    lambda_curtailment: float,
) -> MarketSolution:
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
    prices = np.asarray(result.eqlin.marginals, dtype=float)

    dispatch_df = pd.DataFrame(p_opt, columns=inputs.generators)
    dispatch_df.insert(0, "Time Period", np.arange(1, t_count + 1))

    prices_df = pd.DataFrame(
        {
            "Time Period": np.arange(1, t_count + 1),
            "Market Price": prices,
        }
    )

    balance_df = pd.DataFrame(
        {
            "Time Period": np.arange(1, t_count + 1),
            "Demand": demand,
            "Renewable Available (alpha-scaled)": renewable_available,
            "Renewable Used": renewable_used,
            "Renewable Curtailed": curt_opt,
            "Conventional Generation": p_opt.sum(axis=1),
            "Market Price": prices,
        }
    )

    summary = {
        "total_cost": float(result.fun),
        "total_demand": float(demand.sum()),
        "total_conventional_generation": float(p_opt.sum()),
        "total_renewable_available": float(renewable_available.sum()),
        "total_renewable_used": float(renewable_used.sum()),
        "total_renewable_curtailed": float(curt_opt.sum()),
        "average_price": float(prices.mean()),
        "minimum_price": float(prices.min()),
        "maximum_price": float(prices.max()),
        "negative_price_periods": int((prices < 0).sum()),
    }
    return MarketSolution(
        dispatch=dispatch_df,
        balance=balance_df,
        prices=prices_df,
        summary=summary,
    )


def solve_exercise_1(
    inputs: MarketInputs,
    alpha: float,
    lambda_curtailment: float,
) -> MarketSolution:
    return solve_market_clearing(
        inputs=inputs,
        alpha=alpha,
        lambda_curtailment=lambda_curtailment,
    )


def summarize_price_sensitivity(
    inputs: MarketInputs,
    alpha_values: list[float],
    lambda_values: list[float],
) -> pd.DataFrame:
    rows = []
    for alpha in alpha_values:
        for lambda_curtailment in lambda_values:
            solution = solve_market_clearing(
                inputs=inputs,
                alpha=alpha,
                lambda_curtailment=lambda_curtailment,
            )
            prices = solution.prices["Market Price"].to_numpy(dtype=float)
            rows.append(
                {
                    "alpha": alpha,
                    "lambda_curtailment": lambda_curtailment,
                    "Average Price": float(prices.mean()),
                    "Minimum Price": float(prices.min()),
                    "Maximum Price": float(prices.max()),
                    "Negative Price Periods": int((prices < 0).sum()),
                    "Renewable Curtailed": float(
                        solution.summary["total_renewable_curtailed"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise 1 market-clearing optimization."
    )
    parser.add_argument(
        "--demand-file",
        default=DEFAULT_DEMAND_FILE,
        help="Path to demand and renewable scenarios Excel file.",
    )
    parser.add_argument(
        "--generator-file",
        default=DEFAULT_GENERATOR_FILE,
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
        help="Save dispatch, balance, and price tables as CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV files when using --save-results.",
    )
    parser.add_argument(
        "--compare-alphas",
        type=float,
        nargs="+",
        help="Optional alpha values for exercise 2 price comparison.",
    )
    parser.add_argument(
        "--compare-lambdas",
        type=float,
        nargs="+",
        help="Optional lambda values for exercise 2 price comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_inputs(
        demand_file=args.demand_file,
        generator_file=args.generator_file,
        renewable_scenario_number=args.scenario,
    )

    solution = solve_market_clearing(
        inputs=inputs,
        alpha=args.alpha,
        lambda_curtailment=args.lambda_curtailment,
    )
    dispatch_df = solution.dispatch
    balance_df = solution.balance
    prices_df = solution.prices
    summary = solution.summary

    sensitivity_df = None
    if args.compare_alphas or args.compare_lambdas:
        alpha_values = args.compare_alphas or [args.alpha]
        lambda_values = args.compare_lambdas or [args.lambda_curtailment]
        sensitivity_df = summarize_price_sensitivity(
            inputs=inputs,
            alpha_values=alpha_values,
            lambda_values=lambda_values,
        )

    print("Exercises 1 and 2 solved successfully.")
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
    print(f"- Average market price: {summary['average_price']:.2f}")
    print(f"- Minimum market price: {summary['minimum_price']:.2f}")
    print(f"- Maximum market price: {summary['maximum_price']:.2f}")
    print(
        f"- Periods with negative prices: "
        f"{summary['negative_price_periods']}"
    )
    print("")

    print("Conventional generation by unit (sum over all periods):")
    generation_by_unit = dispatch_df.drop(columns=["Time Period"]).sum()
    for unit, total in generation_by_unit.items():
        print(f"- {unit}: {total:.2f}")

    print("")
    print("First 10 periods (balance and price check):")
    print(balance_df.head(10).to_string(index=False))

    if sensitivity_df is not None:
        print("")
        print("Price sensitivity summary:")
        print(sensitivity_df.to_string(index=False))

    if args.save_results:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = _PROJECT_ROOT / args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dispatch_df.to_csv(out_dir / "exercise1_dispatch.csv", index=False)
        balance_df.to_csv(out_dir / "exercise1_balance.csv", index=False)
        prices_df.to_csv(out_dir / "exercise2_prices.csv", index=False)
        print("")
        print("Saved: exercise1_dispatch.csv")
        print("Saved: exercise1_balance.csv")
        print("Saved: exercise2_prices.csv")
        if sensitivity_df is not None:
            sensitivity_df.to_csv(out_dir / "exercise2_price_sensitivity.csv", index=False)
            print("Saved: exercise2_price_sensitivity.csv")


if __name__ == "__main__":
    main()
