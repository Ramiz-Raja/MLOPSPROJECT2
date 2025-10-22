Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Push-Location (Resolve-Path ..\)
Write-Host "Checking http://127.0.0.1:8000/health"
try {
    $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Method Get -TimeoutSec 5
    Write-Host "HEALTH:"; $h | ConvertTo-Json -Depth 5
} catch {
    Write-Host "HEALTH failed:"; Write-Host $_.Exception.Message
}

Write-Host "Testing /predict with sample features"
try {
    $body = @{ features = @(5.1,3.5,1.4,0.2) }
    $p = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -Body ($body | ConvertTo-Json) -ContentType 'application/json' -TimeoutSec 10
    Write-Host "PREDICT:"; $p | ConvertTo-Json -Depth 5
} catch {
    Write-Host "PREDICT failed:"; Write-Host $_.Exception.Message
}

Pop-Location
