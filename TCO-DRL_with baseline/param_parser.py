import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="TCO-DRL / HCRL-Oracle paper experiments")

    # ------------------------------------------------------------------
    # General experiment controls
    # ------------------------------------------------------------------
    parser.add_argument("--Baselines", nargs="+", default=["Random", "Round-Robin", "Earliest", "DQN", "BLOR", "SemiGreedy", "PPO"],
                        help="Methods to compare. Optional methods are appended by --Use_* flags.")
    parser.add_argument("--Baseline_num", type=int, default=0)
    parser.add_argument("--Epoch", type=int, default=10)
    parser.add_argument("--Seed", type=int, default=6)
    parser.add_argument("--Output_Dir", type=str, default="output")
    parser.add_argument("--Run_Tag", type=str, default="")

    # ------------------------------------------------------------------
    # Learning models
    # ------------------------------------------------------------------
    parser.add_argument("--Dqn_start_learn", type=int, default=300)
    parser.add_argument("--Dqn_learn_interval", type=int, default=1)
    parser.add_argument("--Dqn_hidden", type=int, default=96)
    parser.add_argument("--Dqn_batch_size", type=int, default=64)
    parser.add_argument("--Dqn_memory_size", type=int, default=3000)
    parser.add_argument("--Dqn_epsilon_increment", type=float, default=0.0015)
    parser.add_argument("--Dqn_lr", type=float, default=0.0025)

    parser.add_argument("--PPO_start_learn", type=int, default=500)
    parser.add_argument("--PPO_learn_interval", type=int, default=64)
    parser.add_argument("--PPO_batch_size", type=int, default=64)
    parser.add_argument("--PPO_update_epochs", type=int, default=5)
    parser.add_argument("--PPO_hidden", type=int, default=64)

    parser.add_argument("--Use_RA_DDQN", action="store_true")
    parser.add_argument("--RA_lr", type=float, default=0.0020)
    parser.add_argument("--RA_start_learn", type=int, default=300)
    parser.add_argument("--RA_learn_interval", type=int, default=1)

    parser.add_argument("--Use_PB_SafeDQN", action="store_true")
    parser.add_argument("--PB_lr", type=float, default=0.0022)
    parser.add_argument("--PB_start_learn", type=int, default=200)
    parser.add_argument("--PB_learn_interval", type=int, default=1)
    parser.add_argument("--PB_Backup_Mode", choices=["parallel", "serial"], default="parallel")
    parser.add_argument("--PB_Backup_Trigger", choices=["always", "cost_aware"], default="cost_aware")
    parser.add_argument("--PB_Min_Backup_Score", type=float, default=0.38)
    parser.add_argument("--PB_Backup_Recovery_Bonus", type=float, default=0.38)
    parser.add_argument("--PB_Backup_Used_Penalty", type=float, default=0.16)
    parser.add_argument("--PB_Primary_Success_Bonus", type=float, default=0.18)
    parser.add_argument("--PB_Backup_Skip_Penalty", type=float, default=0.04)
    parser.add_argument("--PB_Backup_Cost_Limit", type=float, default=1.05)
    parser.add_argument("--PB_W_RECENT_SUCCESS", type=float, default=0.42)
    parser.add_argument("--PB_W_REPUTATION", type=float, default=0.24)
    parser.add_argument("--PB_W_LOAD", type=float, default=0.18)
    parser.add_argument("--PB_W_COST", type=float, default=0.10)
    parser.add_argument("--PB_W_TOKEN", type=float, default=0.14)
    parser.add_argument("--PB_W_BEHAVIOR_RISK", type=float, default=0.20)
    parser.add_argument("--PB_W_DELAY", type=float, default=0.10)
    parser.add_argument("--PB_Prior_Strength", type=float, default=2.0)

    # COBRA-Oracle
    parser.add_argument("--Use_COBRA", action="store_true")
    parser.add_argument("--COBRA_lr", type=float, default=0.0016)
    parser.add_argument("--COBRA_start_learn", type=int, default=200)
    parser.add_argument("--COBRA_learn_interval", type=int, default=1)
    parser.add_argument("--COBRA_Teacher_Source", choices=["DQN", "RA-DDQN", "none"], default="DQN")
    parser.add_argument("--COBRA_WarmStart_Episode", type=int, default=3)
    parser.add_argument("--COBRA_Teacher_Guidance_Episodes", type=int, default=8)
    parser.add_argument("--COBRA_Teacher_Start_Prob", type=float, default=0.75)
    parser.add_argument("--COBRA_Min_Teacher_Prob", type=float, default=0.05)
    parser.add_argument("--COBRA_Gate_Mode", choices=["adaptive", "fixed", "always", "never"], default="adaptive")
    parser.add_argument("--COBRA_Min_Backup_Score", type=float, default=0.46)
    parser.add_argument("--COBRA_Gate_Alpha", type=float, default=0.15)
    parser.add_argument("--COBRA_Gate_Window", type=int, default=400)
    parser.add_argument("--COBRA_Primary_Success_Bonus", type=float, default=0.26)
    parser.add_argument("--COBRA_Backup_Recovery_Bonus", type=float, default=0.34)
    parser.add_argument("--COBRA_Backup_Used_Penalty", type=float, default=0.22)
    parser.add_argument("--COBRA_Backup_Skip_Penalty", type=float, default=0.03)
    parser.add_argument("--COBRA_Cost_Budget", type=float, default=1.00)
    parser.add_argument("--COBRA_Latency_Budget", type=float, default=6.0)
    parser.add_argument("--COBRA_Risk_Budget", type=float, default=0.08)
    parser.add_argument("--COBRA_Lambda_Cost", type=float, default=0.45)
    parser.add_argument("--COBRA_Lambda_Latency", type=float, default=0.35)
    parser.add_argument("--COBRA_Lambda_Risk", type=float, default=0.65)
    parser.add_argument("--COBRA_Primary_Malicious_Penalty", type=float, default=0.30)
    parser.add_argument("--COBRA_Random_Backup", action="store_true")
    parser.add_argument("--COBRA_No_Teacher", action="store_true")
    parser.add_argument("--COBRA_No_Decoupled_Reward", action="store_true")

    # HCRL-Oracle
    parser.add_argument("--Use_HCRL", action="store_true")
    parser.add_argument("--HCRL_lr", type=float, default=0.0013)
    parser.add_argument("--HCRL_Mode_lr", type=float, default=0.0010)
    parser.add_argument("--HCRL_Use_Actor_Critic", action="store_true", default=True)
    parser.add_argument("--HCRL_AC_Entropy", type=float, default=0.015)
    parser.add_argument("--HCRL_AC_Value_Coef", type=float, default=0.5)
    parser.add_argument("--HCRL_start_learn", type=int, default=200)
    parser.add_argument("--HCRL_learn_interval", type=int, default=1)
    parser.add_argument("--HCRL_Backup_learn_interval", type=int, default=1)
    parser.add_argument("--HCRL_Mode_learn_interval", type=int, default=1)
    parser.add_argument("--HCRL_Teacher_Source", choices=["DQN", "RA-DDQN", "COBRA-Oracle", "none"], default="DQN")
    parser.add_argument("--HCRL_WarmStart_Episode", type=int, default=3)
    parser.add_argument("--HCRL_Teacher_Guidance_Episodes", type=int, default=8)
    parser.add_argument("--HCRL_Teacher_Start_Prob", type=float, default=0.70)
    parser.add_argument("--HCRL_Min_Teacher_Prob", type=float, default=0.03)
    parser.add_argument("--HCRL_Mode_Start_Prob", type=float, default=0.10)
    parser.add_argument("--HCRL_Mode_Min_Prob", type=float, default=0.03)
    parser.add_argument("--HCRL_Primary_Success_Bonus", type=float, default=0.30)
    parser.add_argument("--HCRL_Backup_Recovery_Bonus", type=float, default=0.40)
    parser.add_argument("--HCRL_Backup_Used_Penalty", type=float, default=0.20)
    parser.add_argument("--HCRL_Unnecessary_Backup_Penalty", type=float, default=0.32)
    parser.add_argument("--HCRL_Skip_Recovery_Penalty", type=float, default=0.08)
    parser.add_argument("--HCRL_Primary_Malicious_Penalty", type=float, default=0.35)
    parser.add_argument("--HCRL_Backup_Malicious_Penalty", type=float, default=0.50)
    parser.add_argument("--HCRL_Backup_Guidance_Episodes", type=int, default=10)
    parser.add_argument("--HCRL_Backup_Start_Prob", type=float, default=0.85)
    parser.add_argument("--HCRL_Backup_Min_Prob", type=float, default=0.05)
    parser.add_argument("--HCRL_Cost_Budget", type=float, default=1.02)
    parser.add_argument("--HCRL_Latency_Budget", type=float, default=5.95)
    parser.add_argument("--HCRL_Risk_Budget", type=float, default=0.06)
    parser.add_argument("--HCRL_Lambda_Cost", type=float, default=0.55)
    parser.add_argument("--HCRL_Lambda_Latency", type=float, default=0.40)
    parser.add_argument("--HCRL_Lambda_Risk", type=float, default=0.80)
    parser.add_argument("--HCRL_Primal_Dual", action="store_true", default=True)
    parser.add_argument("--HCRL_Lambda_LR", type=float, default=0.01)
    parser.add_argument("--HCRL_Lambda_Min", type=float, default=0.0)
    parser.add_argument("--HCRL_Lambda_Max", type=float, default=3.0)
    parser.add_argument("--HCRL_Parallel_Cost_Discount", type=float, default=0.85)
    parser.add_argument("--HCRL_Mode_Names", nargs="+", default=["single", "serial", "parallel"])
    parser.add_argument("--HCRL_No_Teacher", action="store_true")
    parser.add_argument("--HCRL_No_Constrained", action="store_true")
    parser.add_argument("--HCRL_No_Decoupled_Reward", action="store_true")
    parser.add_argument("--HCRL_Random_Backup", action="store_true")
    parser.add_argument("--HCRL_Fixed_Single_Mode", action="store_true")
    parser.add_argument("--HCRL_Fixed_Parallel_Mode", action="store_true")

    # ------------------------------------------------------------------
    # Oracle and workload settings
    # ------------------------------------------------------------------
    parser.add_argument("--Oracle_Type", type=list, default=[])
    parser.add_argument("--Oracle_Cost", type=list, default=[])
    parser.add_argument("--Oracle_Acc", type=list, default=[])
    parser.add_argument("--Oracle_Tokens", type=list, default=[])
    parser.add_argument("--Oracle_Behavior_Probs", type=list, default=[])
    parser.add_argument("--Oracle_Validation_Probs", type=list, default=[])
    parser.add_argument("--Oracle_Num", type=int, default=15)
    parser.add_argument("--Oracles_Per_Type", type=int, default=5)
    parser.add_argument("--Service_Type_Num", type=int, default=3)
    parser.add_argument("--Oracle_Initial_Reputation", type=float, default=0.5)
    parser.add_argument("--Time_Window_Size", type=int, default=5)
    parser.add_argument("--Time_Period_Size", type=int, default=60)
    parser.add_argument("--Oracle_capacity", type=int, default=1000)

    parser.add_argument("--lamda", type=int, default=5)
    parser.add_argument("--Request_Num", type=int, default=6000)
    parser.add_argument("--Request_len_Mean", type=int, default=6000)
    parser.add_argument("--Request_len_Std", type=int, default=500)
    parser.add_argument("--Request_ddl", type=float, default=7.0)

    # ------------------------------------------------------------------
    # State, GNN encoder, reward, and stress scenarios
    # ------------------------------------------------------------------
    parser.add_argument("--Noise_Probability", type=float, default=0.0)
    parser.add_argument("--Noise_Delay", type=float, default=1.0)
    parser.add_argument("--State_Mode", choices=["original", "enhanced"], default="original")
    parser.add_argument("--Use_GNN_Encoder", action="store_true",
                        help="Enable graph message-passing oracle encoder.")
    parser.add_argument("--Disable_GNN_Encoder", action="store_true",
                        help="Ablation: disable the graph encoder even when HCRL is used.")
    parser.add_argument("--Use_GNN_For_All_RL", action="store_true",
                        help="Fairness control: use the same graph encoder for all learning-based RL methods.")
    parser.add_argument("--GNN_Message_Steps", type=int, default=2)
    parser.add_argument("--GNN_Self_Weight", type=float, default=0.55)
    parser.add_argument("--GNN_Neighbor_Weight", type=float, default=0.45)
    parser.add_argument("--GNN_Service_Weight", type=float, default=1.00)
    parser.add_argument("--GNN_Reliability_Weight", type=float, default=0.45)
    parser.add_argument("--GNN_Load_Weight", type=float, default=0.35)
    parser.add_argument("--GNN_Cost_Weight", type=float, default=0.25)

    parser.add_argument("--Reward_Mode", choices=["original", "risk_aware"], default="original")
    parser.add_argument("--Success_Mode", choices=["original", "validation_aware"], default="original")
    parser.add_argument("--Scenario", choices=["static", "validation_stress", "rl_hard", "rl_harder"], default="static")
    parser.add_argument("--SemiGreedy_View", choices=["myopic", "risk_aware"], default="myopic")
    parser.add_argument("--Action_Mask_Mode", choices=["none", "type"], default="none")
    parser.add_argument("--Fatigue_Strength", type=float, default=1.0)
    parser.add_argument("--Burstiness", type=float, default=0.80)
    parser.add_argument("--Expose_Validation_Prob", action="store_true",
                        help="Expose true validation probabilities in enhanced state. Disabled by default to avoid leakage.")
    parser.add_argument("--Harder_Request_DDL", type=float, default=6.6)

    parser.add_argument("--W_SUCCESS", type=float, default=2.2)
    parser.add_argument("--W_REPUTATION", type=float, default=0.35)
    parser.add_argument("--W_MATCH", type=float, default=0.8)
    parser.add_argument("--W_VALIDATION", type=float, default=1.2)
    parser.add_argument("--W_COST", type=float, default=1.15)
    parser.add_argument("--W_RESPONSE", type=float, default=1.35)
    parser.add_argument("--W_BEHAVIOR", type=float, default=0.55)
    parser.add_argument("--W_TIMEOUT", type=float, default=2.0)
    parser.add_argument("--Reward_Clip", type=float, default=3.0)
    parser.add_argument("--Reward_Scale", type=float, default=3.0)

    args = parser.parse_args()

    # Stress scenario defaults.
    if args.Scenario in ["validation_stress", "rl_hard", "rl_harder"]:
        args.Success_Mode = "validation_aware"
        if args.Reward_Mode == "original":
            args.Reward_Mode = "risk_aware"
        if args.State_Mode == "original":
            args.State_Mode = "enhanced"

    if args.Scenario in ["rl_hard", "rl_harder"]:
        if args.Action_Mask_Mode == "none":
            args.Action_Mask_Mode = "type"
        if args.W_SUCCESS == 2.2:
            args.W_SUCCESS = 2.6
        if args.W_RESPONSE == 1.35:
            args.W_RESPONSE = 1.5
        if args.W_TIMEOUT == 2.0:
            args.W_TIMEOUT = 2.2

    if args.Scenario == "rl_harder":
        if args.Burstiness == 0.80:
            args.Burstiness = 0.93
        if args.Fatigue_Strength == 1.0:
            args.Fatigue_Strength = 2.2
        if args.Request_ddl == 7.0:
            args.Request_ddl = args.Harder_Request_DDL
        if args.W_SUCCESS == 2.6:
            args.W_SUCCESS = 3.0
        if args.W_COST == 1.15:
            args.W_COST = 1.25
        if args.W_RESPONSE == 1.5:
            args.W_RESPONSE = 1.7
        if args.W_TIMEOUT == 2.2:
            args.W_TIMEOUT = 2.6
        if args.Dqn_hidden == 96:
            args.Dqn_hidden = 128
        if args.Dqn_start_learn == 300:
            args.Dqn_start_learn = 200
        if args.RA_start_learn == 300:
            args.RA_start_learn = 200

    # Append optional methods.
    if args.Use_RA_DDQN and "RA-DDQN" not in args.Baselines:
        args.Baselines.append("RA-DDQN")
    if args.Use_PB_SafeDQN and "PB-SafeDQN" not in args.Baselines:
        args.Baselines.append("PB-SafeDQN")
    if args.Use_COBRA and "COBRA-Oracle" not in args.Baselines:
        args.Baselines.append("COBRA-Oracle")
    if args.Use_HCRL and "HCRL-Oracle" not in args.Baselines:
        args.Baselines.append("HCRL-Oracle")

    # Full HCRL uses GNN by default, but the ablation can disable it.
    if args.Use_HCRL and not args.Disable_GNN_Encoder:
        args.Use_GNN_Encoder = True
    if args.Disable_GNN_Encoder:
        args.Use_GNN_Encoder = False
    if args.Use_GNN_For_All_RL:
        args.Use_GNN_Encoder = True

    if args.COBRA_No_Teacher:
        args.COBRA_Teacher_Source = "none"
        args.COBRA_Teacher_Start_Prob = 0.0
    if args.HCRL_No_Teacher:
        args.HCRL_Teacher_Source = "none"
        args.HCRL_Teacher_Start_Prob = 0.0

    _generate_oracle_community(args)
    args.Baseline_num = len(args.Baselines)
    return args


def _generate_oracle_community(args):
    role_pattern = ["malicious", "trusted_low", "normal", "trusted_mid", "trusted_high"]

    oracle_type = []
    oracle_cost = []
    oracle_acc = []
    oracle_tokens = []
    oracle_behavior_probs = []
    oracle_validation_probs = []
    oracle_fatigue_sensitivity = []
    malicious_index = []
    normal_index = []
    trusted_index = []

    idx = 0
    for service_type in range(args.Service_Type_Num):
        for k in range(args.Oracles_Per_Type):
            role = role_pattern[k % len(role_pattern)]
            oracle_type.append(service_type)

            if args.Scenario == "validation_stress":
                params = {
                    "malicious":   (0.25, 1.00, 100, [0.40, 0.25, 0.25, 0.10], 0.25, 0.06),
                    "trusted_low": (0.25, 1.00, 150, [0.75, 0.20, 0.05, 0.00], 0.55, 0.08),
                    "normal":      (0.45, 1.10, 250, [0.65, 0.20, 0.15, 0.00], 0.60, 0.05),
                    "trusted_mid": (0.65, 1.15, 400, [0.90, 0.10, 0.00, 0.00], 0.90, 0.02),
                    "trusted_high":(0.95, 1.25, 700, [0.98, 0.02, 0.00, 0.00], 0.98, 0.00),
                }[role]
            elif args.Scenario == "rl_hard":
                params = {
                    "malicious":   (0.18, 1.20,  80, [0.25, 0.25, 0.30, 0.20], 0.28, 0.12),
                    "trusted_low": (0.22, 1.20, 160, [0.70, 0.20, 0.10, 0.00], 0.72, 0.18),
                    "normal":      (0.42, 1.08, 260, [0.62, 0.22, 0.16, 0.00], 0.62, 0.10),
                    "trusted_mid": (0.72, 1.05, 430, [0.91, 0.09, 0.00, 0.00], 0.91, 0.03),
                    "trusted_high":(1.02, 1.00, 780, [0.98, 0.02, 0.00, 0.00], 0.98, 0.01),
                }[role]
            elif args.Scenario == "rl_harder":
                params = {
                    "malicious":   (0.16, 1.25,  70, [0.22, 0.22, 0.30, 0.26], 0.23, 0.16),
                    "trusted_low": (0.21, 1.22, 150, [0.66, 0.22, 0.12, 0.00], 0.68, 0.22),
                    "normal":      (0.40, 1.08, 250, [0.58, 0.24, 0.18, 0.00], 0.58, 0.13),
                    "trusted_mid": (0.74, 1.05, 440, [0.90, 0.10, 0.00, 0.00], 0.90, 0.04),
                    "trusted_high":(1.05, 1.00, 800, [0.985,0.015,0.00, 0.00], 0.985,0.01),
                }[role]
            else:
                params = {
                    "malicious":   (0.30, 1.00, 150, [0.50, 0.25, 0.15, 0.10], 0.50, 0.02),
                    "trusted_low": (0.30, 1.00, 150, [0.80, 0.18, 0.02, 0.00], 0.80, 0.02),
                    "normal":      (0.60, 1.10, 300, [0.70, 0.20, 0.10, 0.00], 0.70, 0.02),
                    "trusted_mid": (0.60, 1.10, 300, [0.90, 0.10, 0.00, 0.00], 0.85, 0.01),
                    "trusted_high":(0.90, 1.20, 500, [0.95, 0.05, 0.00, 0.00], 0.95, 0.00),
                }[role]

            cost, acc, tokens, behavior, validation, fatigue = params
            oracle_cost.append(cost); oracle_acc.append(acc); oracle_tokens.append(tokens)
            oracle_behavior_probs.append(behavior); oracle_validation_probs.append(validation)
            oracle_fatigue_sensitivity.append(fatigue)
            if role == "malicious":
                malicious_index.append(idx)
            elif role == "normal":
                normal_index.append(idx)
            else:
                trusted_index.append(idx)
            idx += 1

    args.Oracle_Type = oracle_type
    args.Oracle_Cost = oracle_cost
    args.Oracle_Acc = oracle_acc
    args.Oracle_Tokens = oracle_tokens
    args.Oracle_Behavior_Probs = oracle_behavior_probs
    args.Oracle_Validation_Probs = oracle_validation_probs
    args.Oracle_Fatigue_Sensitivity = oracle_fatigue_sensitivity
    args.Oracle_Num = len(oracle_type)
    args.Malicious_Oracle_Index = malicious_index
    args.Normal_Oracle_Index = normal_index
    args.Trusted_Oracle_Index = trusted_index
