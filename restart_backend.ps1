Set-Location $PSScriptRoot
Write-Host "Restarting backend (port 8000)..."

# Find PIDs listening on port 8000
$lines = netstat -ano | Select-String ":8000" -SimpleMatch
$listenerPids = @()
foreach ($l in $lines) {
    $parts = ($l.Line -split '\s+') | Where-Object { $_ -ne '' }
    $last = $parts[-1]
    if ($last -match '^[0-9]+$') { $listenerPids += $last }
}
$listenerPids = $listenerPids | Select-Object -Unique
if ($listenerPids.Count -gt 0) {
    Write-Host "Found PIDs listening on :8000 ->" ($listenerPids -join ',')
    foreach ($p in $listenerPids) {
        try {
            Stop-Process -Id $p -Force -ErrorAction Stop
            Write-Host "Stopped pid $p"
        } catch {
            $msg = $_.Exception.Message
            Write-Host ("Failed to stop {0}: {1}" -f $p, $msg)
        }
    }
} else {
    Write-Host "No processes found listening on :8000"
}

# Start uvicorn bound to 127.0.0.1 (so Streamlit can connect to 127.0.0.1)
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host ".venv python not found at" $python; exit 1
}

Write-Host "Starting uvicorn with" $python
$out = Join-Path (Get-Location) 'backend.log'
$err = Join-Path (Get-Location) 'backend.err'
Start-Process -FilePath $python -ArgumentList '-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','8000' -RedirectStandardOutput $out -RedirectStandardError $err -PassThru | Out-Null
Start-Sleep -Seconds 3

Write-Host "--- backend.log (tail 200) ---"
if (Test-Path $out) { Get-Content $out -Tail 200 | ForEach-Object { Write-Host $_ } } else { Write-Host 'backend.log not found' }

Write-Host "--- backend.err (tail 200) ---"
if (Test-Path $err) { Get-Content $err -Tail 200 | ForEach-Object { Write-Host $_ } } else { Write-Host 'backend.err not found' }

Write-Host "Checking backend /health..."
try {
    $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Method GET -TimeoutSec 5
    Write-Host 'HEALTH:'
    $h | ConvertTo-Json -Depth 5 | ForEach-Object { Write-Host $_ }
} catch {
    Write-Host 'Health request failed:' $_.Exception.Message
}

Write-Host "Testing /predict..."
try {
    $body = @{ features = @(5.1,3.5,1.4,0.2) } | ConvertTo-Json
    $resp = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 5
    Write-Host 'PREDICT:'
    $resp | ConvertTo-Json -Depth 5 | ForEach-Object { Write-Host $_ }
} catch {
    Write-Host 'Predict request failed:' $_.Exception.Message
}
