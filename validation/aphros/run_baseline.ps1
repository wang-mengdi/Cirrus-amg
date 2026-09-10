param([string[]]$Cases = @('periodic_n8','periodic_n16','periodic_n32','periodic_3d_n8','perturb_3d_n8','duct_3d_n8','duct_3d_n16'))
$ErrorActionPreference = 'Stop'
& python (Join-Path $PSScriptRoot 'make_cases.py')
if ($LASTEXITCODE) { throw 'Case generation failed' }
$oldDump = $env:APHROS_SIMPLE_DUMP
try {
  $env:APHROS_SIMPLE_DUMP = 'simple'
  foreach ($case in $Cases) {
    Push-Location (Join-Path $PSScriptRoot $case)
    try {
      & ..\aphros\src\main.exe a.conf > run.log 2>&1
      if ($LASTEXITCODE) { throw "Aphros case $case failed; see run.log" }
    } finally { Pop-Location }
  }
  & python (Join-Path $PSScriptRoot 'analyze.py')
  if ($LASTEXITCODE) { throw 'Analysis failed' }
} finally { $env:APHROS_SIMPLE_DUMP = $oldDump }
