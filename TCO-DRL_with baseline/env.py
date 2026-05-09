import numpy as np
from scipy import stats


class SchedulingEnv:
    """Scalable oracle-selection environment for TCO-DRL / HCRL-Oracle.

    This file is a complete replacement for the original simulation environment.
    It keeps the public methods used by main.py while completing the paper-version
    implementation:
      - scalable oracle communities;
      - validation-aware success and risk-aware reward;
      - type action masks;
      - primary-backup recovery for PB-SafeDQN / COBRA-Oracle;
      - hierarchical constrained feedback for HCRL-Oracle;
      - graph message-passing oracle encoder with HCRL-only and all-RL modes.
    """

    def __init__(self, args):
        self.args = args
        self.policy_names = list(args.Baselines)
        self.policy_num = len(self.policy_names)
        self.policy_name_to_id = {name: idx for idx, name in enumerate(self.policy_names)}
        self._load_static_settings(args)
        self._init_state_shape()
        self.arrival_Times = np.zeros(self.requestNum)
        self.requestsMI = np.zeros(self.requestNum)
        self.lengths = np.zeros(self.requestNum)
        self.request_type = np.zeros(self.requestNum, dtype=int)
        self._init_policy_records()
        self.gen_workload(self.lamda)

    # ------------------------------------------------------------------
    # Initialization and workload
    # ------------------------------------------------------------------
    def _load_static_settings(self, args):
        self.oracleTypes = np.asarray(args.Oracle_Type, dtype=int)
        self.oracleNum = int(args.Oracle_Num)
        if self.oracleNum != len(self.oracleTypes):
            raise ValueError("Oracle_Num must equal len(Oracle_Type)")
        self.oracleCapacity = float(args.Oracle_capacity)
        self.actionNum = self.oracleNum
        self.oracleInitialReputation = float(args.Oracle_Initial_Reputation)
        self.oracleAcc = np.asarray(args.Oracle_Acc, dtype=float)
        self.oracleCost = np.asarray(args.Oracle_Cost, dtype=float)
        self.oracleToken = np.asarray(args.Oracle_Tokens, dtype=float)
        self.oracleBehaviorProbs = np.asarray(args.Oracle_Behavior_Probs, dtype=float)
        self.oracleValidationProbs = np.asarray(args.Oracle_Validation_Probs, dtype=float)
        self.oracleFatigueSensitivity = np.asarray(getattr(args, "Oracle_Fatigue_Sensitivity", [0.0] * self.oracleNum), dtype=float)
        self.malicious_oracles = list(getattr(args, "Malicious_Oracle_Index", []))
        self.normal_oracles = list(getattr(args, "Normal_Oracle_Index", []))
        self.trusted_oracles = list(getattr(args, "Trusted_Oracle_Index", []))

        self.requestMI = float(args.Request_len_Mean)
        self.requestMI_std = float(args.Request_len_Std)
        self.requestNum = int(args.Request_Num)
        self.lamda = float(args.lamda)
        self.ddl = float(args.Request_ddl)
        self.noise_probability = float(args.Noise_Probability)
        self.noise_delay = float(args.Noise_Delay)
        self.timewindowSize = int(args.Time_Window_Size)
        self.timeperiodSize = int(args.Time_Period_Size)
        self.timeperiodNum = int(self.requestNum / max(self.timeperiodSize, 1)) + 2

    def _init_state_shape(self):
        if self.args.State_Mode == "original":
            self.s_features = 1 + 2 * self.oracleNum
        else:
            # request type, request length, deadline + eight features per oracle.
            # GNN keeps the same per-oracle dimensionality, so all existing models remain compatible.
            self.s_features = 3 + 8 * self.oracleNum

    def _init_policy_records(self):
        self.events = {}
        self.oracle_events = {}
        self.reputation_factors = {}
        self.oracle_reputation_history = {}
        self.reputation_timewindow = {}
        for name in self.policy_names:
            # rows: 0 oracle, 1 startT, 2 waitT, 3 duration, 4 leaveT, 5 reward,
            # 6 exeT, 7 final success, 8 cost, 9 type match
            self.events[name] = np.zeros((10, self.requestNum), dtype=float)
            # rows: 0 idleT, 1 assigned count, 2 reputation, 3 type matches, 4 validation successes
            self.oracle_events[name] = np.zeros((5, self.oracleNum), dtype=float)
            self.oracle_events[name][2] = self.oracleInitialReputation
            # rows: 0 count, 1 validation successes, 2 total duration, 3 behavior-risk sum
            self.reputation_factors[name] = np.zeros((4, self.oracleNum), dtype=float)
            self.oracle_reputation_history[name] = np.zeros((self.timeperiodNum, self.oracleNum), dtype=float)
            self.reputation_timewindow[name] = np.zeros((0, self.oracleNum), dtype=float)

        # rows: 0 primary_success, 1 backup_used, 2 backup_success, 3 backup_recovery,
        # 4 primary_action, 5 backup_action, 6 primary_malicious, 7 backup_malicious,
        # 8 primary_trusted, 9 backup_trusted, 10 backup_skipped,
        # 11 backup_score, 12 HCRL mode, 13 single, 14 serial, 15 parallel,
        # 16 any constraint violation, 17 cost violation, 18 latency violation,
        # 19 risk violation, 20 lambda cost, 21 lambda latency, 22 lambda risk.
        self.pb_records = {name: np.zeros((23, self.requestNum), dtype=float) for name in self.policy_names}
        for name in self.policy_names:
            self.pb_records[name][4, :] = -1
            self.pb_records[name][5, :] = -1
            self.pb_records[name][12, :] = -1

        self.backup_score_history = {name: [] for name in self.policy_names}
        self.hcrl_lambdas = {
            name: {
                "cost": float(getattr(self.args, "HCRL_Lambda_Cost", 0.55)),
                "latency": float(getattr(self.args, "HCRL_Lambda_Latency", 0.40)),
                "risk": float(getattr(self.args, "HCRL_Lambda_Risk", 0.80)),
            }
            for name in self.policy_names
        }

    def reset(self, args):
        self.args = args
        self.policy_names = list(args.Baselines)
        self.policy_num = len(self.policy_names)
        self.policy_name_to_id = {name: idx for idx, name in enumerate(self.policy_names)}
        self._load_static_settings(args)
        self._init_state_shape()
        self.arrival_Times = np.zeros(self.requestNum)
        self.requestsMI = np.zeros(self.requestNum)
        self.lengths = np.zeros(self.requestNum)
        self.request_type = np.zeros(self.requestNum, dtype=int)
        self._init_policy_records()
        self.gen_workload(args.lamda)

    def gen_workload(self, lamda):
        lamda = max(float(lamda), 1e-8)
        intervalT = stats.expon.rvs(scale=1.0 / lamda * 60.0, size=self.requestNum)
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        self.requestsMI = np.maximum(
            np.random.normal(self.requestMI, self.requestMI_std, self.requestNum).astype(int), 1
        )
        self.lengths = self.requestsMI / max(self.oracleCapacity, 1e-8)

        service_type_num = int(np.max(self.oracleTypes)) + 1
        if getattr(self.args, "Scenario", "static") in ["rl_hard", "rl_harder"]:
            burstiness = float(getattr(self.args, "Burstiness", 0.80))
            types = np.zeros(self.requestNum, dtype=int)
            types[0] = np.random.randint(0, service_type_num)
            for i in range(1, self.requestNum):
                types[i] = types[i - 1] if np.random.rand() < burstiness else np.random.randint(0, service_type_num)
            self.request_type = types
        else:
            self.request_type = np.random.choice(np.arange(service_type_num), size=self.requestNum)

        print("intervalT mean: ", round(np.mean(intervalT), 3), "  intervalT SD:", round(np.std(intervalT, ddof=1), 3))
        print("last request arrivalT:", round(float(self.arrival_Times[-1]), 3))
        print("MI mean: ", round(float(np.mean(self.requestsMI)), 3), "  MI SD:", round(float(np.std(self.requestsMI, ddof=1)), 3))
        print("length mean: ", round(float(np.mean(self.lengths)), 3), "  length SD:", round(float(np.std(self.lengths, ddof=1)), 3))

    def workload(self, request_count):
        request_id = int(request_count) - 1
        attrs = [
            request_id,
            float(self.arrival_Times[request_id]),
            float(self.lengths[request_id]),
            int(self.request_type[request_id]),
            float(self.ddl),
        ]
        return request_count == self.requestNum, attrs

    # ------------------------------------------------------------------
    # Reputation and state encoding
    # ------------------------------------------------------------------
    def initial_reputation(self):
        for name in self.policy_names:
            self.oracle_events[name][2] = self.oracleInitialReputation

    def reset_reputation_factors(self):
        for name in self.policy_names:
            if name != "BLOR":
                self.reputation_factors[name] = np.zeros((4, self.oracleNum), dtype=float)

    def reset_reputation_factors_BLOR(self):
        if "BLOR" in self.policy_names:
            self.reputation_factors["BLOR"] = np.zeros((4, self.oracleNum), dtype=float)

    def get_reputation_factors(self, policy_name):
        return self.reputation_factors[policy_name]

    def update_reputation(self, reputation_attributes, time_period, policy_name):
        counts = reputation_attributes[0]
        val = reputation_attributes[1]
        behavior = reputation_attributes[3]
        old = self.oracle_events[policy_name][2]
        recent_success = (val + self.oracleInitialReputation * 2.0) / np.maximum(counts + 2.0, 1e-8)
        behavior_penalty = np.log1p(np.maximum(behavior / np.maximum(counts, 1.0), 0.0)) / np.log1p(100.0)
        new_rep = np.clip(0.70 * old + 0.30 * (recent_success - 0.35 * behavior_penalty), 0.0, 1.0)
        self.oracle_events[policy_name][2] = new_rep
        tp = int(min(max(time_period, 0), self.timeperiodNum - 1))
        self.oracle_reputation_history[policy_name][tp] = new_rep
        self.reputation_timewindow[policy_name] = np.vstack((self.reputation_timewindow[policy_name], new_rep[None, :]))[-self.timewindowSize:]

    def _policy_uses_gnn(self, policy_name):
        if not getattr(self.args, "Use_GNN_Encoder", False):
            return False
        if getattr(self.args, "Disable_GNN_Encoder", False):
            return False
        if policy_name == "HCRL-Oracle":
            return True
        if getattr(self.args, "Use_GNN_For_All_RL", False) and policy_name in ["DQN", "PPO", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle"]:
            return True
        return False

    def getState(self, request_attrs, policy_name):
        request_id, arrival_time, length, request_type, ddl = request_attrs
        request_type = int(request_type)
        if self.args.State_Mode == "original":
            state = np.hstack(([request_type], self.oracle_events[policy_name][0] - arrival_time, self.oracle_events[policy_name][2]))
            return np.nan_to_num(state.astype(float), nan=0.0, posinf=10.0, neginf=-10.0)

        oracle_features = self._base_oracle_features(request_attrs, policy_name)
        if self._policy_uses_gnn(policy_name):
            oracle_features = self._graph_encode_oracles(oracle_features, request_type)
        prefix = np.array([
            request_type / max(float(np.max(self.oracleTypes)), 1.0),
            float(length) / max(float(getattr(self.args, "Request_len_Mean", 6000)) / max(float(getattr(self.args, "Oracle_capacity", 1000)), 1e-8), 1e-8),
            float(ddl) / max(float(getattr(self.args, "Harder_Request_DDL", 6.6)), 1e-8),
        ], dtype=float)
        state = np.hstack((prefix, oracle_features.reshape(-1)))
        return np.nan_to_num(state.astype(float), nan=0.0, posinf=10.0, neginf=-10.0)

    def _base_oracle_features(self, request_attrs, policy_name):
        _, arrival_time, length, request_type, ddl = request_attrs
        request_type = int(request_type)
        wait = np.maximum(self.oracle_events[policy_name][0] - float(arrival_time), 0.0)
        wait_norm = np.clip(wait / max(float(ddl), 1e-8), 0.0, 3.0) / 3.0
        rep = np.clip(self.oracle_events[policy_name][2], 0.0, 1.0)
        cost_norm = np.clip(self.oracleCost / max(float(np.max(self.oracleCost)), 1e-8), 0.0, 1.0)
        acc_norm = np.clip(self.oracleAcc / max(float(np.max(self.oracleAcc)), 1e-8), 0.0, 1.0)
        type_match = (self.oracleTypes == request_type).astype(float)

        counts = self.reputation_factors[policy_name][0]
        val = self.reputation_factors[policy_name][1]
        prior = 0.5 * rep + 0.5 * np.clip(self.oracleToken / max(float(np.max(self.oracleToken)), 1e-8), 0.0, 1.0)
        observed_success = (val + 2.0 * prior) / np.maximum(counts + 2.0, 1e-8)
        validation_feature = np.asarray(self.oracleValidationProbs if getattr(self.args, "Expose_Validation_Prob", False) else observed_success, dtype=float)
        recent_load = np.clip(counts / max(float(self.timeperiodSize), 1.0), 0.0, 1.0)
        behavior = self.reputation_factors[policy_name][3] / np.maximum(counts, 1.0)
        behavior_risk = np.clip(np.log1p(np.maximum(behavior, 0.0)) / np.log1p(100.0), 0.0, 1.0)
        delay_est = np.clip((wait + float(length) / np.maximum(self.oracleAcc, 1e-8)) / max(float(ddl), 1e-8), 0.0, 2.0) / 2.0

        return np.vstack((wait_norm, rep, cost_norm, acc_norm, type_match, validation_feature, recent_load, 0.5 * behavior_risk + 0.5 * delay_est)).T

    def _graph_encode_oracles(self, features, request_type):
        h = np.asarray(features, dtype=float).copy()
        n = h.shape[0]
        if n == 0:
            return h
        same_service = (self.oracleTypes[:, None] == self.oracleTypes[None, :]).astype(float)
        reliability = 1.0 - np.abs(h[:, 5][:, None] - h[:, 5][None, :])
        load_similarity = 1.0 - np.abs(h[:, 6][:, None] - h[:, 6][None, :])
        cost_similarity = 1.0 - np.abs(h[:, 2][:, None] - h[:, 2][None, :])
        adj = (
            float(getattr(self.args, "GNN_Service_Weight", 1.0)) * same_service
            + float(getattr(self.args, "GNN_Reliability_Weight", 0.45)) * reliability
            + float(getattr(self.args, "GNN_Load_Weight", 0.35)) * load_similarity
            + float(getattr(self.args, "GNN_Cost_Weight", 0.25)) * cost_similarity
        )
        np.fill_diagonal(adj, 0.0)
        row_sum = np.maximum(adj.sum(axis=1, keepdims=True), 1e-8)
        adj = adj / row_sum
        self_w = float(getattr(self.args, "GNN_Self_Weight", 0.55))
        neigh_w = float(getattr(self.args, "GNN_Neighbor_Weight", 0.45))
        steps = int(getattr(self.args, "GNN_Message_Steps", 2))
        request_gate = (self.oracleTypes == int(request_type)).astype(float)[:, None]
        for _ in range(max(steps, 0)):
            msg = adj.dot(h)
            h = np.tanh(self_w * h + neigh_w * msg + 0.05 * request_gate)
        return np.clip(0.5 * (h + 1.0), 0.0, 1.0)

    def get_action_mask(self, request_attrs):
        if getattr(self.args, "Action_Mask_Mode", "none") != "type":
            return np.ones(self.oracleNum, dtype=bool)
        mask = self.oracleTypes == int(request_attrs[3])
        if not np.any(mask):
            mask[:] = True
        return mask.astype(bool)

    def get_backup_action_mask(self, request_attrs, primary_action):
        mask = self.get_action_mask(request_attrs).astype(bool)
        if 0 <= int(primary_action) < self.oracleNum:
            mask[int(primary_action)] = False
        if not np.any(mask):
            mask[:] = True
            if 0 <= int(primary_action) < self.oracleNum:
                mask[int(primary_action)] = False
        if not np.any(mask):
            mask[:] = True
        return mask.astype(bool)

    # ------------------------------------------------------------------
    # Core simulation and feedback
    # ------------------------------------------------------------------
    def _effective_validation_prob(self, action, policy_name):
        base = float(self.oracleValidationProbs[int(action)])
        if getattr(self.args, "Scenario", "static") not in ["rl_hard", "rl_harder"]:
            return base
        recent_assigned = float(self.reputation_factors[policy_name][0, int(action)])
        avg_recent = max(self.timeperiodSize / max(self.oracleNum, 1), 1e-8)
        overload = max(0.0, recent_assigned / avg_recent - 1.0)
        if getattr(self.args, "Scenario", "static") == "rl_harder":
            fatigue_growth = np.sqrt(overload) + 0.35 * overload
            min_prob = 0.02
        else:
            fatigue_growth = np.log1p(overload)
            min_prob = 0.05
        fatigue = float(getattr(self.args, "Fatigue_Strength", 1.0)) * float(self.oracleFatigueSensitivity[int(action)]) * fatigue_growth
        return float(np.clip(base - fatigue, min_prob, 0.99))

    def _simulate_oracle_attempt(self, request_attrs, action, policy_name, arrival_override=None):
        request_id, arrival_time, length, request_type, ddl = request_attrs
        action = int(action)
        effective_arrival = float(arrival_time if arrival_override is None else arrival_override)
        acc = max(float(self.oracleAcc[action]), 1e-8)
        cost = float(self.oracleCost[action])
        oracle_type = int(self.oracleTypes[action])
        idleT = float(self.oracle_events[policy_name][0, action])
        reputation = float(self.oracle_events[policy_name][2, action])
        exeT = float(length) / acc
        waitT = max(idleT - effective_arrival, 0.0)
        startT = effective_arrival + waitT
        exe_time = exeT * (1.05 if action in self.malicious_oracles else 1.0)
        if np.random.rand() < self.noise_probability:
            exe_time += self.noise_delay
        durationT = waitT + exe_time
        leaveT = startT + exe_time
        match = 1 if int(request_type) == oracle_type else 0
        validation_raw = 1 if np.random.rand() < self._effective_validation_prob(action, policy_name) else 0
        probs = np.asarray(self.oracleBehaviorProbs[action], dtype=float)
        probs = probs / max(probs.sum(), 1e-8)
        behavior_record = float(np.random.choice([0, 1, 5, 100], p=probs))
        return {
            "action": action, "startT": startT, "waitT": waitT, "exeT": exeT,
            "durationT": durationT, "leaveT": leaveT, "cost": cost, "reputation": reputation,
            "match": match, "validation_raw": validation_raw, "behavior_record": behavior_record,
            "oracle_type": oracle_type,
            "is_malicious": 1 if action in self.malicious_oracles else 0,
            "is_trusted": 1 if action in self.trusted_oracles else 0,
        }

    def _is_success(self, attempt, ddl):
        if getattr(self.args, "Success_Mode", "original") == "validation_aware":
            return int(attempt["durationT"] <= ddl and attempt["match"] == 1 and attempt["validation_raw"] == 1)
        return int(attempt["durationT"] <= ddl and attempt["match"] == 1)

    def _original_reward(self, exeT, durationT, cost, reputation, request_type, oracle_type):
        penalty = 0 if int(request_type) == int(oracle_type) else 1
        return float((1 + 2.5 * np.exp(1.5 - float(cost))) * (float(exeT) / max(float(durationT), 1e-8)) + float(reputation) - 4 * penalty)

    def _risk_aware_reward(self, reputation, match, successful_validation, cost, durationT, ddl, behavior_record):
        ddl = max(float(ddl), 1e-8)
        timeout = 1.0 if float(durationT) > ddl else 0.0
        on_time = 1.0 - timeout
        rep_score = 0.5 * (np.tanh(float(reputation)) + 1.0)
        match_score = float(match)
        val_score = float(successful_validation)
        task_success = match_score * val_score * on_time
        cost_score = float(np.clip(float(cost), 0.0, 1.25) / 1.25)
        response_ratio = float(np.clip(float(durationT) / ddl, 0.0, 2.5))
        response_penalty = float(np.clip(min(response_ratio, 1.0) * 0.4 + max(response_ratio - 1.0, 0.0) * 0.9, 0.0, 1.0))
        behavior_risk = float(np.log1p(max(float(behavior_record), 0.0)) / np.log1p(100.0))
        a = self.args
        positive = a.W_SUCCESS * task_success + a.W_VALIDATION * val_score + a.W_MATCH * match_score + a.W_REPUTATION * rep_score
        negative = a.W_COST * cost_score + a.W_RESPONSE * response_penalty + a.W_BEHAVIOR * behavior_risk + a.W_TIMEOUT * timeout + 0.8 * (1.0 - task_success)
        normalizer = a.W_SUCCESS + a.W_VALIDATION + a.W_MATCH + a.W_REPUTATION + a.W_COST + a.W_RESPONSE + a.W_BEHAVIOR + a.W_TIMEOUT + 0.8
        return float(np.clip(a.Reward_Scale * (positive - negative) / max(normalizer, 1e-8), -a.Reward_Clip, a.Reward_Clip))

    def _reward_for_attempt(self, attempt, request_attrs, final_success=None, total_cost=None, final_duration=None, combined_behavior=None, combined_rep=None):
        _, _, _, request_type, ddl = request_attrs
        if getattr(self.args, "Reward_Mode", "original") == "risk_aware":
            return self._risk_aware_reward(
                combined_rep if combined_rep is not None else attempt["reputation"],
                attempt["match"],
                final_success if final_success is not None else attempt["validation_raw"],
                total_cost if total_cost is not None else attempt["cost"],
                final_duration if final_duration is not None else attempt["durationT"],
                ddl,
                combined_behavior if combined_behavior is not None else attempt["behavior_record"],
            )
        return self._original_reward(attempt["exeT"], final_duration or attempt["durationT"], total_cost or attempt["cost"], attempt["reputation"], request_type, attempt["oracle_type"])

    def _record_attempt_updates(self, policy_name, attempts):
        for attempt in attempts:
            if attempt is None:
                continue
            a = int(attempt["action"])
            self.oracle_events[policy_name][1, a] += 1
            self.oracle_events[policy_name][0, a] = max(self.oracle_events[policy_name][0, a], attempt["leaveT"])
            self.oracle_events[policy_name][3, a] += attempt["match"]
            self.oracle_events[policy_name][4, a] += attempt["validation_raw"]
            self.reputation_factors[policy_name][0, a] += 1
            self.reputation_factors[policy_name][1, a] += attempt["validation_raw"]
            self.reputation_factors[policy_name][2, a] += attempt["durationT"]
            self.reputation_factors[policy_name][3, a] += attempt["behavior_record"]

    def _record_request(self, policy_name, request_id, primary, reward, success, final_duration, final_leaveT, total_cost, match, backup=None):
        self.events[policy_name][0, request_id] = primary["action"]
        self.events[policy_name][1, request_id] = primary["startT"]
        self.events[policy_name][2, request_id] = primary["waitT"]
        self.events[policy_name][3, request_id] = final_duration
        self.events[policy_name][4, request_id] = final_leaveT
        self.events[policy_name][5, request_id] = reward
        self.events[policy_name][6, request_id] = primary["exeT"]
        self.events[policy_name][7, request_id] = success
        self.events[policy_name][8, request_id] = total_cost
        self.events[policy_name][9, request_id] = match

    def feedback(self, request_attrs, action, policy_name):
        request_id, _, _, _, ddl = request_attrs
        attempt = self._simulate_oracle_attempt(request_attrs, action, policy_name)
        success = self._is_success(attempt, float(ddl))
        reward = self._reward_for_attempt(attempt, request_attrs, final_success=success)
        self._record_attempt_updates(policy_name, [attempt])
        self._record_request(policy_name, int(request_id), attempt, reward, success, attempt["durationT"], attempt["leaveT"], attempt["cost"], attempt["match"])
        # Fill minimal primary diagnostics for single-oracle policies.
        self.pb_records[policy_name][0, int(request_id)] = success
        self.pb_records[policy_name][4, int(request_id)] = int(action)
        self.pb_records[policy_name][6, int(request_id)] = attempt["is_malicious"]
        self.pb_records[policy_name][8, int(request_id)] = attempt["is_trusted"]
        return reward

    def _backup_score_vector(self, request_attrs, primary_action, policy_name):
        _, arrival_time, length, _, ddl = request_attrs
        counts = self.reputation_factors[policy_name][0]
        val = self.reputation_factors[policy_name][1]
        behavior_sum = self.reputation_factors[policy_name][3]
        rep = np.clip(self.oracle_events[policy_name][2], 0.0, 1.0)
        token_norm = np.clip(self.oracleToken / max(float(np.max(self.oracleToken)), 1e-8), 0.0, 1.0)
        prior = 0.5 * rep + 0.5 * token_norm
        alpha = float(getattr(self.args, "PB_Prior_Strength", 2.0))
        recent_success = (val + alpha * prior) / np.maximum(counts + alpha, 1e-8)
        recent_load = counts / max(float(self.timeperiodSize), 1.0)
        cost_norm = np.clip(self.oracleCost / max(float(np.max(self.oracleCost)), 1e-8), 0.0, 1.0)
        avg_behavior = behavior_sum / np.maximum(counts, 1.0)
        behavior_risk = np.clip(np.log1p(np.maximum(avg_behavior, 0.0)) / np.log1p(100.0), 0.0, 1.0)
        estimated_wait = np.maximum(self.oracle_events[policy_name][0] - float(arrival_time), 0.0)
        estimated_exe = float(length) / np.maximum(self.oracleAcc, 1e-8)
        delay_penalty = np.clip((estimated_wait + estimated_exe) / max(float(ddl), 1e-8), 0.0, 2.0) / 2.0
        score = (
            self.args.PB_W_RECENT_SUCCESS * recent_success
            + self.args.PB_W_REPUTATION * rep
            + self.args.PB_W_TOKEN * token_norm
            - self.args.PB_W_LOAD * recent_load
            - self.args.PB_W_COST * cost_norm
            - self.args.PB_W_BEHAVIOR_RISK * behavior_risk
            - self.args.PB_W_DELAY * delay_penalty
        )
        score -= 0.08 * (self.oracleCost > float(getattr(self.args, "PB_Backup_Cost_Limit", 1.05)))
        score[int(primary_action)] = -1e9
        return np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)

    def choose_backup_oracle(self, request_attrs, primary_action, policy_name):
        candidates = np.where(self.get_backup_action_mask(request_attrs, primary_action))[0]
        if policy_name in ["COBRA-Oracle", "HCRL-Oracle"] and getattr(self.args, f"{policy_name.split('-')[0]}_Random_Backup", False):
            return int(np.random.choice(candidates))
        score = self._backup_score_vector(request_attrs, primary_action, policy_name)
        return int(candidates[np.argmax(score[candidates])])

    def _should_use_backup(self, request_attrs, primary, backup_action, backup_score, policy_name):
        if int(backup_action) == int(primary["action"]):
            return False
        if getattr(self.args, "PB_Backup_Mode", "parallel") == "serial":
            remaining = float(request_attrs[4]) - float(primary["durationT"])
            estimated_exe = float(request_attrs[2]) / max(float(self.oracleAcc[int(backup_action)]), 1e-8)
            if remaining <= 0 or estimated_exe > remaining:
                return False
        if policy_name == "COBRA-Oracle":
            mode = getattr(self.args, "COBRA_Gate_Mode", "adaptive")
            if mode == "always":
                return True
            if mode == "never":
                return False
            if mode == "fixed":
                return float(backup_score) >= float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46))
            hist = self.backup_score_history.get(policy_name, [])
            if len(hist) >= 20:
                recent = np.asarray(hist[-int(getattr(self.args, "COBRA_Gate_Window", 400)):], dtype=float)
                dyn_thr = float(np.mean(recent) + float(getattr(self.args, "COBRA_Gate_Alpha", 0.15)) * np.std(recent))
            else:
                dyn_thr = float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46))
            return float(backup_score) >= max(float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46)), dyn_thr)
        if getattr(self.args, "PB_Backup_Trigger", "cost_aware") == "always":
            return True
        return float(backup_score) >= float(getattr(self.args, "PB_Min_Backup_Score", 0.38))

    def feedback_primary_backup(self, request_attrs, primary_action, policy_name="PB-SafeDQN"):
        request_id, _, _, _, ddl = request_attrs
        primary = self._simulate_oracle_attempt(request_attrs, primary_action, policy_name)
        primary_success = self._is_success(primary, float(ddl))
        backup_action = self.choose_backup_oracle(request_attrs, primary_action, policy_name)
        score_vec = self._backup_score_vector(request_attrs, primary_action, policy_name)
        backup_score = float(score_vec[int(backup_action)])
        self.backup_score_history[policy_name].append(backup_score)
        use_backup = (primary_success == 0) and self._should_use_backup(request_attrs, primary, backup_action, backup_score, policy_name)
        backup = None
        backup_success = 0
        backup_recovery = 0
        final_duration = primary["durationT"]
        final_leaveT = primary["leaveT"]
        total_cost = primary["cost"]
        final_success = primary_success
        combined_behavior = primary["behavior_record"]
        combined_rep = primary["reputation"]

        if use_backup:
            arrival_override = primary["leaveT"] if getattr(self.args, "PB_Backup_Mode", "parallel") == "serial" else request_attrs[1]
            backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name, arrival_override=arrival_override)
            backup_success = self._is_success(backup, float(ddl))
            backup_recovery = int(primary_success == 0 and backup_success == 1)
            final_success = int(primary_success == 1 or backup_success == 1)
            final_duration = min(primary["durationT"] if primary_success else 1e18, backup["durationT"] if backup_success else max(primary["durationT"], backup["durationT"]))
            if not np.isfinite(final_duration) or final_duration > 1e17:
                final_duration = max(primary["durationT"], backup["durationT"])
            final_leaveT = max(primary["leaveT"], backup["leaveT"])
            total_cost += backup["cost"]
            combined_behavior = max(primary["behavior_record"], backup["behavior_record"])
            combined_rep = 0.5 * (primary["reputation"] + backup["reputation"])
        backup_skipped = int(primary_success == 0 and not use_backup)
        reward = self._reward_for_attempt(primary, request_attrs, final_success=final_success, total_cost=total_cost, final_duration=final_duration, combined_behavior=combined_behavior, combined_rep=combined_rep)
        if policy_name == "COBRA-Oracle":
            reward += getattr(self.args, "COBRA_Primary_Success_Bonus", 0.26) * primary_success
            reward += getattr(self.args, "COBRA_Backup_Recovery_Bonus", 0.34) * backup_recovery
            reward -= getattr(self.args, "COBRA_Backup_Used_Penalty", 0.22) * int(use_backup)
            reward -= getattr(self.args, "COBRA_Backup_Skip_Penalty", 0.03) * backup_skipped
            reward -= getattr(self.args, "COBRA_Primary_Malicious_Penalty", 0.30) * primary["is_malicious"]
        else:
            reward += getattr(self.args, "PB_Primary_Success_Bonus", 0.18) * primary_success
            reward += getattr(self.args, "PB_Backup_Recovery_Bonus", 0.38) * backup_recovery
            reward -= getattr(self.args, "PB_Backup_Used_Penalty", 0.16) * int(use_backup)
            reward -= getattr(self.args, "PB_Backup_Skip_Penalty", 0.04) * backup_skipped
        reward = float(np.clip(reward, -self.args.Reward_Clip, self.args.Reward_Clip))
        self._record_attempt_updates(policy_name, [primary, backup])
        self._record_request(policy_name, int(request_id), primary, reward, final_success, final_duration, final_leaveT, total_cost, primary["match"], backup)
        self._record_pb_diag(policy_name, int(request_id), primary, backup, primary_success, int(use_backup), backup_success, backup_recovery, backup_skipped, backup_score, mode=-1)
        return reward

    def _record_pb_diag(self, policy_name, request_id, primary, backup, primary_success, backup_used, backup_success, backup_recovery, backup_skipped, backup_score, mode):
        rec = self.pb_records[policy_name]
        rec[0, request_id] = primary_success
        rec[1, request_id] = backup_used
        rec[2, request_id] = backup_success
        rec[3, request_id] = backup_recovery
        rec[4, request_id] = primary["action"]
        rec[5, request_id] = -1 if backup is None else backup["action"]
        rec[6, request_id] = primary["is_malicious"]
        rec[7, request_id] = 0 if backup is None else backup["is_malicious"]
        rec[8, request_id] = primary["is_trusted"]
        rec[9, request_id] = 0 if backup is None else backup["is_trusted"]
        rec[10, request_id] = backup_skipped
        rec[11, request_id] = backup_score
        if mode >= 0:
            rec[12, request_id] = mode
            rec[13, request_id] = 1 if mode == 0 else 0
            rec[14, request_id] = 1 if mode == 1 else 0
            rec[15, request_id] = 1 if mode == 2 else 0

    # ------------------------------------------------------------------
    # HCRL hierarchical constrained feedback
    # ------------------------------------------------------------------
    def get_hcrl_mode_state(self, request_attrs, policy_name):
        base = self.getState(request_attrs, policy_name)
        rid = int(request_attrs[0])
        start, end = max(0, rid - self.timeperiodSize), max(1, rid)
        recent_cost = float(np.mean(self.events[policy_name][8, start:end])) if end > start else 0.0
        recent_success = float(np.mean(self.events[policy_name][7, start:end])) if end > start else 0.0
        recent_latency = float(np.mean(self.events[policy_name][3, start:end])) if end > start else 0.0
        recent_mal = float(np.mean(np.maximum(self.pb_records[policy_name][6, start:end], self.pb_records[policy_name][7, start:end]))) if end > start else 0.0
        budget_state = np.array([
            recent_success,
            recent_cost / max(float(getattr(self.args, "HCRL_Cost_Budget", 1.0)), 1e-8),
            recent_latency / max(float(getattr(self.args, "HCRL_Latency_Budget", 6.0)), 1e-8),
            recent_mal,
            float(getattr(self.args, "HCRL_Cost_Budget", 1.0)),
            float(getattr(self.args, "HCRL_Risk_Budget", 0.06)),
        ], dtype=float)
        return np.hstack((base, budget_state))

    def _get_hcrl_lambdas(self, policy_name):
        if policy_name not in self.hcrl_lambdas:
            self.hcrl_lambdas[policy_name] = {"cost": self.args.HCRL_Lambda_Cost, "latency": self.args.HCRL_Lambda_Latency, "risk": self.args.HCRL_Lambda_Risk}
        return self.hcrl_lambdas[policy_name]

    def _update_hcrl_lambdas(self, policy_name, cost_violation, latency_violation, risk_violation):
        lambdas = self._get_hcrl_lambdas(policy_name)
        if getattr(self.args, "HCRL_No_Constrained", False) or not getattr(self.args, "HCRL_Primal_Dual", True):
            return lambdas
        lr = float(getattr(self.args, "HCRL_Lambda_LR", 0.01))
        lo, hi = float(getattr(self.args, "HCRL_Lambda_Min", 0.0)), float(getattr(self.args, "HCRL_Lambda_Max", 3.0))
        lambdas["cost"] = float(np.clip(lambdas["cost"] + lr * cost_violation, lo, hi))
        lambdas["latency"] = float(np.clip(lambdas["latency"] + lr * latency_violation, lo, hi))
        lambdas["risk"] = float(np.clip(lambdas["risk"] + lr * risk_violation, lo, hi))
        return lambdas

    def feedback_hcrl(self, request_attrs, mode_action, primary_action, backup_action, policy_name="HCRL-Oracle"):
        request_id, arrival_time, length, _, ddl = request_attrs
        request_id = int(request_id)
        mode_action = int(np.clip(mode_action, 0, 2))
        primary = self._simulate_oracle_attempt(request_attrs, primary_action, policy_name)
        primary_success = self._is_success(primary, float(ddl))
        backup = None
        backup_used = 0
        backup_success = 0
        backup_recovery = 0
        backup_skipped = 0
        backup_score = 0.0
        final_success = primary_success
        final_duration = primary["durationT"]
        final_leaveT = primary["leaveT"]
        total_cost = primary["cost"]
        accounting_cost = primary["cost"]
        combined_behavior = primary["behavior_record"]
        combined_rep = primary["reputation"]

        if backup_action >= 0 and backup_action != primary_action:
            backup_score = float(self._backup_score_vector(request_attrs, primary_action, policy_name)[int(backup_action)])

        if mode_action == 1 and primary_success == 0 and backup_action >= 0 and backup_action != primary_action:
            # Serial backup starts after the primary result and must still be useful under the deadline.
            arrival_override = primary["leaveT"]
            backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name, arrival_override=arrival_override)
            backup_used = 1
        elif mode_action == 2 and backup_action >= 0 and backup_action != primary_action:
            # Parallel warm-standby committee is launched at the same arrival time.
            backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name, arrival_override=arrival_time)
            backup_used = 1

        if backup is not None:
            backup_success = self._is_success(backup, float(ddl))
            backup_recovery = int(primary_success == 0 and backup_success == 1)
            final_success = int(primary_success == 1 or backup_success == 1)
            if primary_success and backup_success:
                final_duration = min(primary["durationT"], backup["durationT"])
            elif backup_success:
                final_duration = backup["durationT"]
            else:
                final_duration = max(primary["durationT"], backup["durationT"])
            final_leaveT = max(primary["leaveT"], backup["leaveT"])
            accounting_cost = primary["cost"] + backup["cost"]
            if mode_action == 2:
                total_cost = primary["cost"] + float(getattr(self.args, "HCRL_Parallel_Cost_Discount", 0.85)) * backup["cost"]
            else:
                total_cost = accounting_cost
            combined_behavior = max(primary["behavior_record"], backup["behavior_record"])
            combined_rep = 0.5 * (primary["reputation"] + backup["reputation"])
        else:
            backup_skipped = int(primary_success == 0 and mode_action == 0)

        risk = float(max(primary["is_malicious"], 0 if backup is None else backup["is_malicious"]))
        cost_violation = max(0.0, total_cost - float(getattr(self.args, "HCRL_Cost_Budget", 1.02)))
        latency_violation = max(0.0, final_duration - float(getattr(self.args, "HCRL_Latency_Budget", 5.95)))
        risk_violation = max(0.0, risk - float(getattr(self.args, "HCRL_Risk_Budget", 0.06)))
        lambdas = self._update_hcrl_lambdas(policy_name, cost_violation, latency_violation, risk_violation)

        base_reward = self._reward_for_attempt(primary, request_attrs, final_success=final_success, total_cost=total_cost, final_duration=final_duration, combined_behavior=combined_behavior, combined_rep=combined_rep)
        constraint_penalty = 0.0 if getattr(self.args, "HCRL_No_Constrained", False) else (
            lambdas["cost"] * cost_violation + lambdas["latency"] * latency_violation + lambdas["risk"] * risk_violation
        )
        reward = base_reward - constraint_penalty
        reward += getattr(self.args, "HCRL_Primary_Success_Bonus", 0.30) * primary_success
        reward += getattr(self.args, "HCRL_Backup_Recovery_Bonus", 0.40) * backup_recovery
        reward -= getattr(self.args, "HCRL_Backup_Used_Penalty", 0.20) * backup_used
        reward -= getattr(self.args, "HCRL_Unnecessary_Backup_Penalty", 0.32) * int(backup_used and primary_success)
        reward -= getattr(self.args, "HCRL_Skip_Recovery_Penalty", 0.08) * backup_skipped
        reward -= getattr(self.args, "HCRL_Primary_Malicious_Penalty", 0.35) * primary["is_malicious"]
        if backup is not None:
            reward -= getattr(self.args, "HCRL_Backup_Malicious_Penalty", 0.50) * backup["is_malicious"]
        reward = float(np.clip(reward, -self.args.Reward_Clip, self.args.Reward_Clip))

        if getattr(self.args, "HCRL_No_Decoupled_Reward", False):
            mode_reward = primary_reward = backup_reward = reward
        else:
            mode_reward = float(np.clip(reward + 0.25 * final_success - 0.15 * (cost_violation + latency_violation + risk_violation), -self.args.Reward_Clip, self.args.Reward_Clip))
            primary_reward = float(np.clip(base_reward + getattr(self.args, "HCRL_Primary_Success_Bonus", 0.30) * primary_success - getattr(self.args, "HCRL_Primary_Malicious_Penalty", 0.35) * primary["is_malicious"], -self.args.Reward_Clip, self.args.Reward_Clip))
            backup_reward = float(np.clip(getattr(self.args, "HCRL_Backup_Recovery_Bonus", 0.40) * backup_recovery - getattr(self.args, "HCRL_Backup_Used_Penalty", 0.20) * backup_used - (0 if backup is None else getattr(self.args, "HCRL_Backup_Malicious_Penalty", 0.50) * backup["is_malicious"]), -self.args.Reward_Clip, self.args.Reward_Clip))

        self._record_attempt_updates(policy_name, [primary, backup])
        self._record_request(policy_name, request_id, primary, reward, final_success, final_duration, final_leaveT, total_cost, primary["match"], backup)
        self._record_pb_diag(policy_name, request_id, primary, backup, primary_success, backup_used, backup_success, backup_recovery, backup_skipped, backup_score, mode_action)
        rec = self.pb_records[policy_name]
        rec[16, request_id] = 1 if (cost_violation + latency_violation + risk_violation) > 0 else 0
        rec[17, request_id] = cost_violation
        rec[18, request_id] = latency_violation
        rec[19, request_id] = risk_violation
        rec[20, request_id] = lambdas["cost"]
        rec[21, request_id] = lambdas["latency"]
        rec[22, request_id] = lambdas["risk"]
        return {"final_reward": reward, "mode_reward": mode_reward, "primary_reward": primary_reward, "backup_reward": backup_reward}

    # ------------------------------------------------------------------
    # Heuristic support
    # ------------------------------------------------------------------
    def get_oracle_idleT(self, policy_name):
        return self.oracle_events[policy_name][0]

    def get_request_num(self, policy_name):
        return self.oracle_events[policy_name][1]

    def get_successful_validation(self, policy_name):
        return self.oracle_events[policy_name][4]

    def feedback_PSG_FWA(self, request_attrs, policy_name):
        rewards = np.zeros(self.oracleNum, dtype=float)
        costs = np.asarray(self.oracleCost, dtype=float).copy()
        for a in range(self.oracleNum):
            # Deterministic one-step estimate without mutating environment.
            wait = max(self.oracle_events[policy_name][0, a] - float(request_attrs[1]), 0.0)
            exe = float(request_attrs[2]) / max(self.oracleAcc[a], 1e-8)
            duration = wait + exe
            match = 1 if self.oracleTypes[a] == int(request_attrs[3]) else 0
            if getattr(self.args, "SemiGreedy_View", "myopic") == "risk_aware":
                rep = self.oracle_events[policy_name][2, a]
                val_est = self._effective_validation_prob(a, policy_name)
                rewards[a] = self._risk_aware_reward(rep, match, val_est, self.oracleCost[a], duration, request_attrs[4], 0.0)
            else:
                rewards[a] = self._original_reward(exe, duration, self.oracleCost[a], self.oracle_events[policy_name][2, a], request_attrs[3], self.oracleTypes[a])
        return rewards, costs

    # ------------------------------------------------------------------
    # Aggregate metrics used by main.py
    # ------------------------------------------------------------------
    def _slice(self, startP):
        start = int(max(0, min(startP, self.requestNum - 1)))
        return slice(start, self.requestNum)

    def _per_policy(self, fn):
        return np.asarray([fn(name) for name in self.policy_names], dtype=float)

    def get_totalRewards(self, baseline_num=None, startP=0):
        sl = self._slice(startP)
        return self._per_policy(lambda n: np.sum(self.events[n][5, sl]))

    def get_total_responseTs(self, baseline_num=None, startP=0):
        sl = self._slice(startP)
        return self._per_policy(lambda n: np.mean(self.events[n][3, sl]))

    def get_totalSuccess(self, baseline_num=None, startP=0):
        sl = self._slice(startP)
        return self._per_policy(lambda n: np.mean(self.events[n][7, sl]))

    def get_totalSuccessInTime(self, baseline_num=None, startP=0):
        sl = self._slice(startP)
        return self._per_policy(lambda n: np.mean((self.events[n][7, sl] > 0) & (self.events[n][3, sl] <= self.ddl)))

    def get_totalTimes(self, baseline_num=None, startP=0):
        sl = self._slice(startP)
        return self._per_policy(lambda n: np.max(self.events[n][4, sl]))

    def get_totalCost(self, baseline_num=None, startP=0):
        sl = self._slice(startP)
        return self._per_policy(lambda n: np.mean(self.events[n][8, sl]))

    def get_totalMatchRate(self, baseline_num=None, startP=0):
        sl = self._slice(startP)
        return self._per_policy(lambda n: np.mean(self.events[n][9, sl]))

    def _role_count(self, role_list, startP=0):
        role_set = set(int(x) for x in role_list)
        sl = self._slice(startP)
        return self._per_policy(lambda n: sum(int(a) in role_set for a in self.events[n][0, sl].astype(int)))

    def get_totalMaliciousNum(self, baseline_num=None, startP=0):
        return self._role_count(self.malicious_oracles, startP)

    def get_totalNormalNum(self, baseline_num=None, startP=0):
        return self._role_count(self.normal_oracles, startP)

    def get_totalTrustedNum(self, baseline_num=None, startP=0):
        return self._role_count(self.trusted_oracles, startP)

    def _pb_mean(self, row, startP=0):
        sl = self._slice(startP)
        return self._per_policy(lambda n: np.mean(self.pb_records[n][row, sl]))

    def _pb_sum(self, row, startP=0):
        sl = self._slice(startP)
        return self._per_policy(lambda n: np.sum(self.pb_records[n][row, sl]))

    def get_totalPrimarySuccessRate(self, baseline_num=None, startP=0): return self._pb_mean(0, startP)
    def get_totalBackupUsedRate(self, baseline_num=None, startP=0): return self._pb_mean(1, startP)
    def get_totalBackupRecoveryRate(self, baseline_num=None, startP=0): return self._pb_mean(3, startP)
    def get_totalBackupSkippedRate(self, baseline_num=None, startP=0): return self._pb_mean(10, startP)
    def get_totalBackupScoreMean(self, baseline_num=None, startP=0): return self._pb_mean(11, startP)
    def get_totalPrimaryMaliciousNum(self, baseline_num=None, startP=0): return self._pb_sum(6, startP)
    def get_totalBackupMaliciousNum(self, baseline_num=None, startP=0): return self._pb_sum(7, startP)
    def get_totalPrimaryTrustedNum(self, baseline_num=None, startP=0): return self._pb_sum(8, startP)
    def get_totalBackupTrustedNum(self, baseline_num=None, startP=0): return self._pb_sum(9, startP)
    def get_totalHCRLSingleModeRate(self, baseline_num=None, startP=0): return self._pb_mean(13, startP)
    def get_totalHCRLSerialModeRate(self, baseline_num=None, startP=0): return self._pb_mean(14, startP)
    def get_totalHCRLParallelModeRate(self, baseline_num=None, startP=0): return self._pb_mean(15, startP)
    def get_totalConstraintViolationRate(self, baseline_num=None, startP=0): return self._pb_mean(16, startP)
    def get_totalCostViolation(self, baseline_num=None, startP=0): return self._pb_mean(17, startP)
    def get_totalLatencyViolation(self, baseline_num=None, startP=0): return self._pb_mean(18, startP)
    def get_totalRiskViolation(self, baseline_num=None, startP=0): return self._pb_mean(19, startP)
    def get_totalHCRLLambdaCost(self, baseline_num=None, startP=0): return self._pb_mean(20, startP)
    def get_totalHCRLLambdaLatency(self, baseline_num=None, startP=0): return self._pb_mean(21, startP)
    def get_totalHCRLLambdaRisk(self, baseline_num=None, startP=0): return self._pb_mean(22, startP)

    def get_totalConditionalBackupRecoveryRate(self, baseline_num=None, startP=0):
        sl = self._slice(startP)
        out = []
        for name in self.policy_names:
            used = self.pb_records[name][1, sl]
            rec = self.pb_records[name][3, sl]
            out.append(float(np.sum(rec) / max(np.sum(used), 1.0)))
        return np.asarray(out, dtype=float)

    def get_totalCostPerSuccess(self, baseline_num=None, startP=0):
        cost = self.get_totalCost(baseline_num, startP)
        succ = self.get_totalSuccess(baseline_num, startP)
        return cost / np.maximum(succ, 1e-8)

    # Compatibility aliases for older main.py variants.
    def get_accumulateRewards(self, baseline_num, startP, request_c): return self.get_totalRewards(baseline_num, startP)
    def get_accumulateCost(self, baseline_num, startP, request_c): return self.get_totalCost(baseline_num, startP)
    def get_FinishTimes(self, baseline_num, startP, request_c): return self.get_totalTimes(baseline_num, startP)
    def get_executeTs(self, baseline_num, startP, request_c): return self.get_total_responseTs(baseline_num, startP)
    def get_waitTs(self, baseline_num, startP, request_c): return self._per_policy(lambda n: np.mean(self.events[n][2, self._slice(startP)]))
    def get_responseTs(self, baseline_num, startP, request_c): return self.get_total_responseTs(baseline_num, startP)
    def get_successTimes(self, baseline_num, startP, request_c): return self.get_totalSuccess(baseline_num, startP)
    def get_successInTime(self, baseline_num, startP, request_c): return self.get_totalSuccessInTime(baseline_num, startP)

    # Extra naming variants to tolerate small differences in main.py versions.
    def get_totalHCRLSingleMode(self, baseline_num=None, startP=0): return self.get_totalHCRLSingleModeRate(baseline_num, startP)
    def get_totalHCRLSerialMode(self, baseline_num=None, startP=0): return self.get_totalHCRLSerialModeRate(baseline_num, startP)
    def get_totalHCRLParallelMode(self, baseline_num=None, startP=0): return self.get_totalHCRLParallelModeRate(baseline_num, startP)
    def get_totalCostViolationRate(self, baseline_num=None, startP=0): return self.get_totalCostViolation(baseline_num, startP)
    def get_totalLatencyViolationRate(self, baseline_num=None, startP=0): return self.get_totalLatencyViolation(baseline_num, startP)
    def get_totalRiskViolationRate(self, baseline_num=None, startP=0): return self.get_totalRiskViolation(baseline_num, startP)
