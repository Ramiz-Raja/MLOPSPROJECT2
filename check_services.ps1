Set-Location $PSScriptRoot
Write-Host "Checking Python processes (uvicorn/streamlit)..."
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match 'uvicorn' -or $_.CommandLine -match 'streamlit') } | Select-Object ProcessId, CommandLine

Write-Host "\nNetstat (listening) for ports 8000 and 8501:" 
netstat -ano | Select-String ":8000" -SimpleMatch | ForEach-Object { $_.Line }
netstat -ano | Select-String ":8501" -SimpleMatch | ForEach-Object { $_.Line }

Write-Host "\nAttempting backend /health request..."
try {
    $r = Invoke-RestMethod -Uri 'http://localhost:8000/health' -Method GET -TimeoutSec 5
    Write-Host "HEALTH: "
    $r | ConvertTo-Json -Depth 5
} catch {
    Write-Host "Backend health request failed: " $_.Exception.Message
}
