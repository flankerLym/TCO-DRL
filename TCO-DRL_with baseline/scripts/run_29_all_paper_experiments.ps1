# Run the full paper experiment set.
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
$root = Get-Location
$jobs = @(
  "run_20_full_hcrl_gnn.ps1",
  "run_21_hcrl_no_gnn.ps1",
  "run_22_cobra_gnn_fair.ps1",
  "run_23_ra_ddqn_gnn_fair.ps1",
  "run_24_hcrl_no_teacher.ps1",
  "run_25_hcrl_no_constrained.ps1",
  "run_26_hcrl_random_backup.ps1",
  "run_27_hcrl_fixed_single.ps1",
  "run_28_hcrl_fixed_parallel.ps1"
)
foreach ($job in $jobs) {
  Write-Host "=== Running $job ==="
  powershell -ExecutionPolicy Bypass -File (Join-Path $root "scripts\$job")
}
python tools/collect_paper_results.py --output_dir "TCO-DRL_with baseline\output" --out_csv "paper_results_summary.csv"
