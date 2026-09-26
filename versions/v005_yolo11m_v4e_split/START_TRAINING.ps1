param(
    [switch]$Resume,
    [switch]$CheckOnly,
    [string]$PythonExe
)
$ErrorActionPreference = 'Stop'
if (-not $PythonExe) {
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
    $PythonExe = Join-Path $projectRoot 'steel-defect-yolo\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python environment not found: $PythonExe. Supply -PythonExe with the correct path."
}
$launchArgs = @('-u', (Join-Path $PSScriptRoot 'code\scripts\run_training.py'))
if ($Resume) { $launchArgs += '--resume' }
if ($CheckOnly) { $launchArgs += '--check' }
& $PythonExe @launchArgs
if ($LASTEXITCODE -ne 0) { throw "Training entry exited with code $LASTEXITCODE. See the error above." }
