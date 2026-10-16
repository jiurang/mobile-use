<#
.SYNOPSIS
    mobile-use local launcher for Windows.

.DESCRIPTION
    1. Fixes the GBK console encoding that crashes the CLI on emoji output
    2. Preflight checks: API key in .env, ADB device connection
    3. Runs minitap.mobile_use.main with the goal you pass in

.EXAMPLE
    .\run-local.ps1 "open settings and tell me the current battery level"

.EXAMPLE
    .\run-local.ps1 "open Gmail and list the first 3 unread emails" `
        -OutputDescription "JSON array with sender and subject keys"
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Goal,

    [Parameter(Mandatory = $false)]
    [string]$OutputDescription,

    [Parameter(Mandatory = $false)]
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# 1) Windows console defaults to GBK and cannot encode emoji -> force UTF-8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host "Virtualenv not found: $python" -ForegroundColor Red
    Write-Host "Run: uv venv; uv sync" -ForegroundColor Yellow
    exit 1
}

# 2) Preflight checks
$envFile = Join-Path $root '.env'
if (-not (Test-Path $envFile)) {
    Write-Host "Missing .env file" -ForegroundColor Red
    Write-Host "Run: Copy-Item .env.example .env" -ForegroundColor Yellow
    exit 1
}

$envContent = [System.IO.File]::ReadAllText($envFile, [System.Text.Encoding]::UTF8)
$keyMatch = [regex]::Match($envContent, '(?m)^MINIMAX_API_KEY\s*=\s*["'']?([^"''\r\n]*)')
$apiKey = if ($keyMatch.Success) { $keyMatch.Groups[1].Value.Trim() } else { '' }
if ([string]::IsNullOrWhiteSpace($apiKey) -or $apiKey -eq '...') {
    Write-Host "MINIMAX_API_KEY is not set" -ForegroundColor Red
    Write-Host "Fill your MiniMax API key into $envFile" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] API key configured" -ForegroundColor Green

if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
    Write-Host "adb not found. Install Android platform-tools and add it to PATH." -ForegroundColor Red
    exit 1
}

$devices = (& adb devices | Select-Object -Skip 1 | Where-Object { $_ -match '\sdevice$' })
if (-not $devices) {
    Write-Host "No Android device detected" -ForegroundColor Red
    Write-Host "Connect the phone by USB and enable Developer options -> USB debugging," -ForegroundColor Yellow
    Write-Host "then tap Allow on the 'Allow USB debugging' prompt on the phone." -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Device connected: $($devices -join ', ')" -ForegroundColor Green

if ($CheckOnly) {
    Write-Host "`nAll preflight checks passed." -ForegroundColor Green
    exit 0
}

# 3) Run
$arguments = @('-m', 'minitap.mobile_use.main', $Goal)
if ($OutputDescription) {
    $arguments += @('--output-description', $OutputDescription)
}

Write-Host "`nRunning: $Goal`n" -ForegroundColor Cyan
& $python @arguments
exit $LASTEXITCODE
