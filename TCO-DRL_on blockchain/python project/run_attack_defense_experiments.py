"""
Run ME / OOA / OSA trust-attack defense experiments for the deployed HCRL-Oracle policy.

This script reuses the offline-trained weights used by run_policy_onchain.py and
runs attack-pattern simulations around the local oracle-selection environment.
It is designed to reproduce the *type* of attack-defense analysis reported in
TCO-DRL Section 7.3:
  - ME  : malicious-with-everyone, persistently malicious service;
  - OOA : on-off attack, alternating malicious and honest phases;
  - OSA : opportunistic service attack, malicious only while reputation is high.

Outputs:
  attack_output/<run_tag>/
    - attack_request_log.csv
    - attack_period_summary.csv
    - attack_reputation_trajectory.csv
    - attack_overall_summary.csv
    - figures/*.png

Important:
  The current enhanced environment normalizes reputation to [0, 1], so the
  default trust threshold is 0.45 rather than the -1.5 threshold used by the
  original paper's unbounded reputation score. Change --Trust_Threshold if you
  implement the original reputation scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
SIM_DIR = REPO_ROOT / "TCO-DRL_with baseline"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from env import SchedulingEnv  # noqa: E402
from utils import get_args  # noqa: E402
from policy_adapter import OnChainPolicyAdapter  # noqa: E402


ATTACKS = ["ME", "OOA", "OSA"]


def parse_int_list(value: str) -> List[int]:
    if value is None or str(value).strip() == "":
        return []
    out = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def parse_attack_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--Chain_Config", default="config_chain.json")
    p.add_argument("--Attacks", nargs="+", default=ATTACKS, choices=ATTACKS)
    p.add_argument("--Max_Requests", type=int, default=3000,
                   help="Attack evaluation length. Original paper uses 3000 requests for 50 periods when Time_Period_Size=60.")
    p.add_argument("--Attack_Start_Period", type=int, default=3)
    p.add_argument("--Attack_Oracles", type=str, default="4,19,29",
                   help="Comma-separated oracle IDs to attack. Defaults target the high-use primaries observed in the HCRL deployment.")
    p.add_argument("--Trusted_Reference_Oracles", type=str, default="3,14,24",
                   help="Comma-separated trusted/reference oracle IDs for comparison curves.")
    p.add_argument("--Trust_Threshold", type=float, default=0.45,
                   help="Normalized reputation threshold below which an oracle is masked out from selection.")
    p.add_argument("--Disable_Threshold_Gate", action="store_true",
                   help="Do not mask low-reputation oracles. Use this for ablations.")
    p.add_argument("--OSA_Margin", type=float, default=0.10,
                   help="OSA attacks while reputation > threshold + margin; otherwise it behaves honestly.")
    p.add_argument("--OOA_Off_Periods", type=int, default=1,
                   help="Number of malicious periods in each OOA cycle.")
    p.add_argument("--OOA_On_Periods", type=int, default=5,
                   help="Number of honest recovery periods in each OOA cycle.")
    p.add_argument("--Run_Tag", type=str, default="attack_defense_hcrl")
    p.add_argument("--Output_Dir", type=str, default="attack_output")
    p.add_argument("--Plot", action="store_true", default=True)
    p.add_argument("--No_Plot", action="store_true")
    attack_args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return attack_args


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def effective_reputation(env: SchedulingEnv, policy_name: str) -> np.ndarray:
    if hasattr(env, "_effective_reputation_vector"):
        return np.asarray(env._effective_reputation_vector(policy_name), dtype=float)
    return np.asarray(env.oracle_events[policy_name][2], dtype=float)


def audit_truth(env: SchedulingEnv, policy_name: str) -> np.ndarray:
    if hasattr(env, "audit_truth_score"):
        return np.asarray(env.audit_truth_score(policy_name), dtype=float)
    return np.full(env.oracleNum, np.nan, dtype=float)


def install_threshold_gate(env: SchedulingEnv, policy_name: str, threshold: float):
    """Mask oracles below a normalized trust threshold.

    The original TCO-DRL paper disallows oracles below a trust threshold during
    attack-defense experiments. The enhanced codebase does not hard-mask by
    reputation by default, so we wrap action-mask methods for this experiment.
    """
    original_primary = env.get_action_mask
    original_backup = env.get_backup_action_mask

    def gated_primary(request_attrs):
        base = np.asarray(original_primary(request_attrs), dtype=bool)
        rep = effective_reputation(env, policy_name)
        gated = base & (rep >= threshold)
        if not np.any(gated):
            # Keep the experiment running if all same-type oracles are below threshold.
            return base
        return gated

    def gated_backup(request_attrs, primary_action):
        base = np.asarray(original_backup(request_attrs, primary_action), dtype=bool)
        rep = effective_reputation(env, policy_name)
        gated = base & (rep >= threshold)
        if not np.any(gated):
            return base
        return gated

    env.get_action_mask = gated_primary
    env.get_backup_action_mask = gated_backup


def set_oracle_behavior(env: SchedulingEnv, oracle_ids: Sequence[int], mode: str,
                        base_validation: np.ndarray, base_behavior: np.ndarray):
    """Set attack-oracle behavior for the current period.

    mode='honest' restores strong/high-quality behavior;
    mode='malicious' forces low validation and harmful behavior;
    mode='base' restores the originally generated configuration.
    """
    for oid in oracle_ids:
        if oid < 0 or oid >= env.oracleNum:
            continue
        if mode == "malicious":
            env.oracleValidationProbs[oid] = 0.05
            env.oracleBehaviorProbs[oid] = np.array([0.08, 0.12, 0.35, 0.45], dtype=float)
        elif mode == "honest":
            env.oracleValidationProbs[oid] = 0.98
            env.oracleBehaviorProbs[oid] = np.array([0.98, 0.02, 0.0, 0.0], dtype=float)
        else:
            env.oracleValidationProbs[oid] = base_validation[oid]
            env.oracleBehaviorProbs[oid] = base_behavior[oid]


def attack_state_for_period(attack: str, period: int, attack_args, env: SchedulingEnv,
                            method: str, attack_oracles: Sequence[int]) -> str:
    if period < int(attack_args.Attack_Start_Period):
        return "base"

    if attack == "ME":
        return "malicious"

    if attack == "OOA":
        cycle = max(int(attack_args.OOA_Off_Periods), 1) + max(int(attack_args.OOA_On_Periods), 1)
        offset = (period - int(attack_args.Attack_Start_Period)) % cycle
        return "malicious" if offset < int(attack_args.OOA_Off_Periods) else "honest"

    if attack == "OSA":
        rep = effective_reputation(env, method)
        # If any attacked oracle still has high enough reputation, the attacker exploits it.
        high = [oid for oid in attack_oracles
                if 0 <= oid < env.oracleNum and rep[oid] > float(attack_args.Trust_Threshold + attack_args.OSA_Margin)]
        return "malicious" if high else "honest"

    raise ValueError(f"Unknown attack: {attack}")


def ensure_weights_from_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    weights = dict(cfg.get("weights", {}))
    if "weight_path" in cfg and "primary" not in weights:
        weights["primary"] = cfg["weight_path"]
    return weights


def build_env_and_adapter(args, cfg: Dict[str, Any], config_dir: Path, method: str):
    env = SchedulingEnv(args)
    weights = ensure_weights_from_config(cfg)
    adapter = OnChainPolicyAdapter(args, env, method=method, weights=weights, config_dir=config_dir)
    env.reset(args)
    env.reset_reputation_factors()
    env.initial_reputation()
    return env, adapter


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def summarize_period(attack: str, period: int, attack_state: str, env: SchedulingEnv, method: str,
                     attack_oracles: Sequence[int], trusted_oracles: Sequence[int],
                     period_rows: List[Dict[str, Any]], threshold: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rep = effective_reputation(env, method)
    truth = audit_truth(env, method)
    req_n = len(period_rows)
    selected_primary = [int(r["primary_oracle"]) for r in period_rows]
    selected_backup = [int(r["backup_oracle"]) for r in period_rows]

    attack_set = set(attack_oracles)
    trusted_set = set(trusted_oracles)
    primary_attack = sum(1 for x in selected_primary if x in attack_set)
    backup_attack = sum(1 for x in selected_backup if x in attack_set)
    any_attack = sum(1 for p, b in zip(selected_primary, selected_backup) if p in attack_set or b in attack_set)
    primary_trusted = sum(1 for x in selected_primary if x in trusted_set)
    any_trusted = sum(1 for p, b in zip(selected_primary, selected_backup) if p in trusted_set or b in trusted_set)

    mode_counts: Dict[str, int] = {}
    for r in period_rows:
        mode_counts[str(r["mode_name"])] = mode_counts.get(str(r["mode_name"]), 0) + 1

    period_summary = {
        "attack": attack,
        "period": period,
        "attack_state": attack_state,
        "requests": req_n,
        "attack_primary_count": primary_attack,
        "attack_backup_count": backup_attack,
        "attack_any_count": any_attack,
        "attack_primary_rate": primary_attack / max(req_n, 1),
        "attack_any_rate": any_attack / max(req_n, 1),
        "trusted_primary_count": primary_trusted,
        "trusted_any_count": any_trusted,
        "trusted_primary_rate": primary_trusted / max(req_n, 1),
        "trusted_any_rate": any_trusted / max(req_n, 1),
        "attack_mean_reputation": safe_mean(rep[oid] for oid in attack_oracles if 0 <= oid < env.oracleNum),
        "trusted_mean_reputation": safe_mean(rep[oid] for oid in trusted_oracles if 0 <= oid < env.oracleNum),
        "attack_below_threshold_count": sum(1 for oid in attack_oracles if 0 <= oid < env.oracleNum and rep[oid] < threshold),
        "trusted_below_threshold_count": sum(1 for oid in trusted_oracles if 0 <= oid < env.oracleNum and rep[oid] < threshold),
    }
    for k, v in mode_counts.items():
        period_summary[f"mode_count_{k}"] = v
        period_summary[f"mode_rate_{k}"] = v / max(req_n, 1)

    trajectory_rows = []
    for oid in list(attack_oracles) + list(trusted_oracles):
        if oid < 0 or oid >= env.oracleNum:
            continue
        trajectory_rows.append({
            "attack": attack,
            "period": period,
            "oracle": oid,
            "group": "attacked" if oid in attack_set else "trusted_reference",
            "attack_state": attack_state if oid in attack_set else "reference",
            "reputation": float(rep[oid]),
            "audit_truth": float(truth[oid]) if oid < len(truth) else float("nan"),
            "primary_selected_count": sum(1 for x in selected_primary if x == oid),
            "backup_selected_count": sum(1 for x in selected_backup if x == oid),
            "any_selected_count": sum(1 for p, b in zip(selected_primary, selected_backup) if p == oid or b == oid),
            "below_threshold": bool(rep[oid] < threshold),
        })
    return period_summary, trajectory_rows


def run_one_attack(attack: str, args, attack_args, cfg: Dict[str, Any], config_dir: Path, out_dir: Path):
    method = cfg.get("method") or (args.Baselines[0] if len(args.Baselines) == 1 else "HCRL-Oracle")
    env, adapter = build_env_and_adapter(args, cfg, config_dir, method)

    if not attack_args.Disable_Threshold_Gate:
        install_threshold_gate(env, method, float(attack_args.Trust_Threshold))

    base_validation = np.asarray(env.oracleValidationProbs, dtype=float).copy()
    base_behavior = np.asarray(env.oracleBehaviorProbs, dtype=float).copy()

    attack_oracles = parse_int_list(attack_args.Attack_Oracles)
    trusted_oracles = parse_int_list(attack_args.Trusted_Reference_Oracles)
    if not attack_oracles:
        attack_oracles = list(getattr(args, "Malicious_Oracle_Index", []))[:3]
    if not trusted_oracles:
        trusted_oracles = list(getattr(args, "Trusted_Oracle_Index", []))[:3]

    request_log: List[Dict[str, Any]] = []
    period_summaries: List[Dict[str, Any]] = []
    reputation_trajectory: List[Dict[str, Any]] = []

    time_period_size = int(args.Time_Period_Size)
    max_requests = int(attack_args.Max_Requests)
    current_period = 1
    current_state = attack_state_for_period(attack, current_period, attack_args, env, method, attack_oracles)
    set_oracle_behavior(env, attack_oracles, current_state, base_validation, base_behavior)
    period_rows: List[Dict[str, Any]] = []

    for request_c in range(1, max_requests + 1):
        period = (request_c - 1) // time_period_size + 1
        if period != current_period:
            # End previous period: update reputation and summarize.
            env.update_reputation(env.get_reputation_factors(method), current_period, method)
            period_summary, traj = summarize_period(
                attack, current_period, current_state, env, method,
                attack_oracles, trusted_oracles, period_rows, float(attack_args.Trust_Threshold)
            )
            period_summaries.append(period_summary)
            reputation_trajectory.extend(traj)
            env.reset_reputation_factors()
            period_rows = []

            current_period = period
            current_state = attack_state_for_period(attack, current_period, attack_args, env, method, attack_oracles)
            set_oracle_behavior(env, attack_oracles, current_state, base_validation, base_behavior)

        finish, request_attrs = env.workload(request_c)
        decision = adapter.infer(request_attrs)
        adapter.update_environment_after_submit(request_attrs, decision)

        row = {
            "attack": attack,
            "request_id": request_c,
            "period": current_period,
            "attack_state": current_state,
            "method": method,
            "mode_name": decision.mode_name,
            "mode_code": decision.mode_code,
            "primary_oracle": decision.primary_oracle,
            "backup_oracle": decision.backup_oracle,
            "selected_attacked_primary": int(decision.primary_oracle in set(attack_oracles)),
            "selected_attacked_backup": int(decision.backup_oracle in set(attack_oracles)),
            "selected_attacked_any": int(decision.primary_oracle in set(attack_oracles) or decision.backup_oracle in set(attack_oracles)),
            "selected_trusted_primary": int(decision.primary_oracle in set(trusted_oracles)),
            "selected_trusted_any": int(decision.primary_oracle in set(trusted_oracles) or decision.backup_oracle in set(trusted_oracles)),
        }
        request_log.append(row)
        period_rows.append(row)

        if finish:
            break

    # Final period summary.
    if period_rows:
        env.update_reputation(env.get_reputation_factors(method), current_period, method)
        period_summary, traj = summarize_period(
            attack, current_period, current_state, env, method,
            attack_oracles, trusted_oracles, period_rows, float(attack_args.Trust_Threshold)
        )
        period_summaries.append(period_summary)
        reputation_trajectory.extend(traj)
        env.reset_reputation_factors()

    total_requests = len(request_log)
    overall = {
        "attack": attack,
        "method": method,
        "total_requests": total_requests,
        "periods": len(period_summaries),
        "attack_oracles": ",".join(map(str, attack_oracles)),
        "trusted_reference_oracles": ",".join(map(str, trusted_oracles)),
        "trust_threshold": float(attack_args.Trust_Threshold),
        "threshold_gate_enabled": not bool(attack_args.Disable_Threshold_Gate),
        "attack_any_rate": safe_mean([r["selected_attacked_any"] for r in request_log]),
        "attack_primary_rate": safe_mean([r["selected_attacked_primary"] for r in request_log]),
        "trusted_any_rate": safe_mean([r["selected_trusted_any"] for r in request_log]),
        "trusted_primary_rate": safe_mean([r["selected_trusted_primary"] for r in request_log]),
        "final_attack_mean_reputation": period_summaries[-1]["attack_mean_reputation"] if period_summaries else float("nan"),
        "final_trusted_mean_reputation": period_summaries[-1]["trusted_mean_reputation"] if period_summaries else float("nan"),
        "min_attack_mean_reputation": safe_mean([min([p["attack_mean_reputation"] for p in period_summaries])]) if period_summaries else float("nan"),
        "max_attack_any_rate_period": max([p["attack_any_rate"] for p in period_summaries], default=float("nan")),
        "policy_hash": adapter.policy_hash,
    }

    return request_log, period_summaries, reputation_trajectory, overall


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out_dir: Path):
    if plt is None or pd is None:
        print("[WARN] matplotlib/pandas not available; skipping plots.")
        return

    traj_path = out_dir / "attack_reputation_trajectory.csv"
    period_path = out_dir / "attack_period_summary.csv"
    if not traj_path.exists() or not period_path.exists():
        return
    traj = pd.read_csv(traj_path)
    period = pd.read_csv(period_path)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for attack in sorted(traj["attack"].unique()):
        sub = traj[traj["attack"] == attack]
        plt.figure(figsize=(8, 5))
        for (group, oracle), g in sub.groupby(["group", "oracle"]):
            label = f"{group}-{oracle}"
            plt.plot(g["period"], g["reputation"], marker="o", linewidth=1.5, label=label)
        # threshold is repeated; use first.
        psub = period[period["attack"] == attack]
        if len(psub):
            threshold = float(psub["attack_below_threshold_count"].notna().iloc[0])  # placeholder not used
        # We cannot recover threshold from this file; add a neutral visual line if config later edits it.
        plt.xlabel("Time period")
        plt.ylabel("Normalized reputation / effective reputation")
        plt.title(f"{attack}: reputation trajectories")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{attack}_reputation.png")
        plt.close()

        p = period[period["attack"] == attack]
        plt.figure(figsize=(8, 5))
        plt.plot(p["period"], p["attack_any_rate"], marker="o", linewidth=1.5, label="attacked oracle selected (any)")
        plt.plot(p["period"], p["attack_primary_rate"], marker="o", linewidth=1.5, label="attacked oracle selected as primary")
        plt.plot(p["period"], p["trusted_any_rate"], marker="o", linewidth=1.5, label="trusted reference selected (any)")
        plt.xlabel("Time period")
        plt.ylabel("Empirical selection probability")
        plt.title(f"{attack}: selection probability by period")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{attack}_selection_probability.png")
        plt.close()


def main():
    attack_args = parse_attack_args()
    args = get_args()

    np.random.seed(int(args.Seed))

    config_path = resolve_path(attack_args.Chain_Config, Path.cwd())
    config_dir = config_path.parent
    cfg = load_json(config_path)

    method = cfg.get("method") or (args.Baselines[0] if len(args.Baselines) == 1 else "HCRL-Oracle")
    if method != "HCRL-Oracle":
        print(f"[WARN] This script was designed for HCRL-Oracle; current method={method}")

    timestamp = int(time.time())
    out_dir = resolve_path(attack_args.Output_Dir, Path.cwd()) / f"{attack_args.Run_Tag}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_request_rows: List[Dict[str, Any]] = []
    all_period_rows: List[Dict[str, Any]] = []
    all_traj_rows: List[Dict[str, Any]] = []
    all_overall_rows: List[Dict[str, Any]] = []

    print("[attack defense] method=", method)
    print("[attack defense] attacks=", attack_args.Attacks)
    print("[attack defense] attack_oracles=", attack_args.Attack_Oracles)
    print("[attack defense] trusted_reference_oracles=", attack_args.Trusted_Reference_Oracles)
    print("[attack defense] max_requests=", attack_args.Max_Requests)
    print("[attack defense] output=", out_dir)

    for attack in attack_args.Attacks:
        print(f"\n=== Running attack: {attack} ===")
        req, per, traj, overall = run_one_attack(attack, args, attack_args, cfg, config_dir, out_dir)
        all_request_rows.extend(req)
        all_period_rows.extend(per)
        all_traj_rows.extend(traj)
        all_overall_rows.append(overall)
        print(
            f"[{attack}] requests={overall['total_requests']} "
            f"attack_any_rate={overall['attack_any_rate']:.4f} "
            f"trusted_any_rate={overall['trusted_any_rate']:.4f} "
            f"final_attack_rep={overall['final_attack_mean_reputation']:.4f} "
            f"final_trusted_rep={overall['final_trusted_mean_reputation']:.4f}"
        )

    write_csv(out_dir / "attack_request_log.csv", all_request_rows)
    write_csv(out_dir / "attack_period_summary.csv", all_period_rows)
    write_csv(out_dir / "attack_reputation_trajectory.csv", all_traj_rows)
    write_csv(out_dir / "attack_overall_summary.csv", all_overall_rows)

    if not attack_args.No_Plot:
        make_plots(out_dir)

    print("\n[done] attack-defense results saved to:", out_dir)
    print("[done] key files:")
    print("  -", out_dir / "attack_overall_summary.csv")
    print("  -", out_dir / "attack_period_summary.csv")
    print("  -", out_dir / "attack_reputation_trajectory.csv")
    print("  -", out_dir / "figures")


if __name__ == "__main__":
    main()
