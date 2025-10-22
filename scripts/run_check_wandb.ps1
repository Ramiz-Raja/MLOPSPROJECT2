Param()
Set-StrictMode -Version Latest
# run_check_wandb.ps1
# Sets WANDB env vars reliably and runs the check_wandb_fetch.py script using the project's venv

$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Path $MyInvocation.MyCommand.Definition -Parent }
$projectRoot = Resolve-Path -Path (Join-Path $scriptDir '..')
Push-Location $projectRoot

# <-- Replace these with the values you provided -->
$env:WANDB_API_KEY = '34d37fb4f02cafe74a2f9678ef11de119acae4cd'
$env:WANDB_ENTITY  = 'raja-ramiz-mukhtar6-szabist'
$env:WANDB_PROJECT = 'MLOPSPROJECT2'

Write-Host "Running check_wandb_fetch.py with WANDB_ENTITY=$($env:WANDB_ENTITY)"

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "python not found at $python"
    Pop-Location
    Exit 1
}

& $python (Join-Path $projectRoot 'scripts\check_wandb_fetch.py')

Pop-Location
