$ErrorActionPreference = 'Continue'
Set-Location d:\PROJECTS\latex-workflow\dt_pruning
$py = "D:\WorkSpace\Py\python\python.exe"
foreach ($a in 'cql', 'bcq', 'iql') {
    & $py train_offline.py --algo $a --max-trees 6 --steps 30 --log-every 10 --save-every 30 --init-from checkpoints_bc/bc_best.pt --out "smoke_$a" 2>&1
    & $py evaluate.py --policy $a --checkpoint "smoke_$a/${a}_last.pt" --split val --limit 1 --exp _smoke 2>&1 | Select-Object -Last 2
}
