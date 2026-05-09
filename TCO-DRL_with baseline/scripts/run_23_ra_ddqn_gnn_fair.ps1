# Fairness control: RA-DDQN with the same GNN state encoder.
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
Set-Location -Path "TCO-DRL_with baseline"
python main.py `
  --Scenario rl_harder `
  --Use_RA_DDQN `
  --Use_GNN_For_All_RL `
  --Oracles_Per_Type 10 --Epoch 30 --Request_Num 6000 `
  --Reward_Scale 2.0 --Reward_Clip 2.0 `
  --Dqn_lr 0.0015 --RA_lr 0.0012 `
  --Dqn_batch_size 128 --Dqn_memory_size 10000 --Dqn_epsilon_increment 0.0008 `
  --Run_Tag ra_ddqn_gnn_fair
