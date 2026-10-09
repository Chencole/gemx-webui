[CmdletBinding()]
param([string]$BackendRoot = 'E:\dev\GEM-X')
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $project 'runtime'
$uv = Get-Command uv -ErrorAction Stop
Write-Output "uv: $($uv.Source)"
& $uv.Source --version
& $uv.Source python list --only-installed
if (-not (Test-Path -LiteralPath (Join-Path $BackendRoot 'repo\gem\network\stats_compose.py'))) {
    throw "Missing official GEM-X source under $BackendRoot\repo. Restore the source/model manifest first."
}
# uv installs Python into its normal managed location, not inside the project.
# These operations preserve unrelated environments and global Python defaults.
& $uv.Source python install 3.12.14
if ($LASTEXITCODE -ne 0) { throw 'Managed Python installation failed.' }
& $uv.Source sync --project $project --locked
if ($LASTEXITCODE -ne 0) { throw 'Web UI dependency installation failed.' }
& $uv.Source sync --project $runtime --locked
if ($LASTEXITCODE -ne 0) { throw 'CUDA dependency installation failed.' }
$env:MOTION_BACKEND_ROOT = $BackendRoot
$env:MOTION_DEVICE = 'cuda'
& $uv.Source run --project $runtime --locked python (Join-Path $runtime 'runtime_env.py')
if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime check failed; no CPU fallback was selected.' }
& $uv.Source run --project $runtime --locked python (Join-Path $runtime 'sam3d_gpu.py') --validate-imports
if ($LASTEXITCODE -ne 0) { throw 'SAM3D dependency import check failed.' }
Write-Output 'GPU environment installed. Full inference still requires a real input video.'
Write-Output "Launch: $project\Start-GPU.cmd"
