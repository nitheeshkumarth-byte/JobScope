[CmdletBinding()]
param([switch]$InstallBrowser)

$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
$venv = Join-Path $project '.venv'
$python = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        & $pythonLauncher.Source -3.14 -m venv $venv
    } else {
        & python -m venv $venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Install Python 3.14 and try again.' }
}

& $python -m pip install -r (Join-Path $project 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'JobScope dependency installation failed.' }

if ($InstallBrowser) {
    & $python -m playwright install chromium
    if ($LASTEXITCODE -ne 0) { throw 'Chromium installation failed.' }
}

$config = Join-Path $project '.env'
if (-not (Test-Path -LiteralPath $config)) {
    Copy-Item -LiteralPath (Join-Path $project '.env.example') -Destination $config
}

Write-Host 'JobScope dependencies are installed.'
Write-Host 'Start Ollama and pull the configured model if you want LLM features.'
Write-Host "Start JobScope from the project directory with: .\.venv\Scripts\python.exe dashboard.py"
