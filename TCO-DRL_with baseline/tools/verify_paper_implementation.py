"""Verify that the paper-version HCRL/GNN implementation is installed."""
import argparse
import importlib.util
import sys
from pathlib import Path


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    args, unknown = parser.parse_known_args()
    root = Path(args.repo_root).resolve()
    code = root
    if not code.exists():
        raise SystemExit(f"Cannot find {code}")

    sys.argv = ["verify", "--Scenario", "rl_harder", "--Use_RA_DDQN", "--Use_PB_SafeDQN", "--Use_COBRA", "--Use_HCRL", "--Request_Num", "80", "--Epoch", "1"]
    param = import_from(code / "param_parser.py", "param_parser_verify")
    env_mod = import_from(code / "env.py", "env_verify")
    parsed = param.parameter_parser()
    assert parsed.Use_HCRL is True
    assert parsed.Use_GNN_Encoder is True
    assert hasattr(parsed, "Disable_GNN_Encoder")
    assert hasattr(parsed, "Use_GNN_For_All_RL")

    env = env_mod.SchedulingEnv(parsed)
    env.initial_reputation()
    finish, request_attrs = env.workload(1)
    state = env.getState(request_attrs, "HCRL-Oracle")
    assert len(state) == env.s_features, (len(state), env.s_features)
    mode_state = env.get_hcrl_mode_state(request_attrs, "HCRL-Oracle")
    assert len(mode_state) == env.s_features + 6
    feedback = env.feedback_hcrl(request_attrs, 2, 0, 1, "HCRL-Oracle")
    for key in ["final_reward", "mode_reward", "primary_reward", "backup_reward"]:
        assert key in feedback, feedback
    print("OK: HCRL/GNN implementation is installed and callable.")


if __name__ == "__main__":
    main()
