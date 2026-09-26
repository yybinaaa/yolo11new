param(
    [switch]$CheckOnly,
    [switch]$KeepQilie,
    [switch]$Visualize,
    [string]$Device = '0',
    [string]$Source,
    [string[]]$Weights,
    [string]$PythonExe
)
$ErrorActionPreference = 'Stop'
if (-not $PythonExe) {
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
    $PythonExe = Join-Path $projectRoot 'steel-defect-yolo\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
$inferArgs = @('-u', (Join-Path $PSScriptRoot 'INFER_FUSAI.py'), '--device', $Device)
if ($CheckOnly) { $inferArgs += '--check' }
if ($KeepQilie) { $inferArgs += '--keep-qilie' }
if ($Visualize) { $inferArgs += '--visualize' }
if ($Source) { $inferArgs += @('--source', $Source) }
if ($Weights) { $inferArgs += '--weights'; $inferArgs += $Weights }
& $PythonExe @inferArgs
if ($LASTEXITCODE -ne 0) { throw "Inference exited with code $LASTEXITCODE. See the error above." }
