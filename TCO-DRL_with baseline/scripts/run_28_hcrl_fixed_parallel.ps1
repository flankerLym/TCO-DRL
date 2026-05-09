# Ablation: force HCRL to parallel committee mode.
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
Set-Location -Path "TCO-DRL_with baseline"
python main.py --Scenario rl_harder --Use_RA_DDQN --Use_PB_SafeDQN --Use_COBRA --Use_HCRL --HCRL_Fixed_Parallel_Mode --Oracles_Per_Type 10 --Epoch 30 --Request_Num 6000 --Reward_Scale 2.0 --Reward_Clip 2.0 --Dqn_batch_size 128 --Dqn_memory_size 10000 --Run_Tag hcrl_fixed_parallel
