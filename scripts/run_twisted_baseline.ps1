param(
  [Parameter(Mandatory=$true)][string]$CaseDirectory,
  [Parameter(Mandatory=$true)][string]$Executable,
  [int]$Threads = 2,
  [switch]$UseAmg,
  [switch]$OriginalConjugate,
  [string]$GeometryState = '',
  [switch]$FixFluxHalo,
  [switch]$DumpAdvection,
  [switch]$FixExplicitResidualHalo,
  [switch]$DumpExplicitResidual,
  [switch]$DumpPressureSystem,
  [switch]$DumpPressureFaces,
  [switch]$VolumePressureCompatibility,
  [ValidateRange(0,1)][double]$PressureSolveTolerance = 0,
  [ValidateRange(1,8)][int]$FactorCacheEntries = 1,
  [switch]$TraceFactorCache
)
$ErrorActionPreference = 'Stop'
$casePath = (Resolve-Path -LiteralPath $CaseDirectory).Path
$exePath = (Resolve-Path -LiteralPath $Executable).Path
$logPath = Join-Path $casePath 'run.log'
if (Test-Path -LiteralPath $logPath) { throw 'Choose a fresh case directory; existing run.log is preserved.' }
if ($Threads -lt 1) { throw 'Threads must be positive.' }
if ($OriginalConjugate -and ($UseAmg -or $DumpPressureSystem -or $DumpPressureFaces -or $VolumePressureCompatibility -or $PressureSolveTolerance -gt 0)) {
  throw 'Original conjugate mode does not use the optional direct-backend controls or its pressure capture.'
}
$geometryPath = if ($GeometryState) { (Resolve-Path -LiteralPath $GeometryState).Path } else { $null }
$settings = @{
  OMP_NUM_THREADS = "$Threads"; OMP_WAIT_POLICY = 'PASSIVE'
  APHROS_TWISTED_DIRECT = $(if ($OriginalConjugate) {$null} else {'1'}); APHROS_SIMPLE_DUMP = 'simple'
  APHROS_TWISTED_GEOMETRY = 'tube'; APHROS_TWISTED_TIME_DUMP = 'tube'
  APHROS_TWISTED_AMG = $(if ($UseAmg) {'1'} else {$null})
  APHROS_TWISTED_FIX_FLUX_HALO = $(if ($FixFluxHalo) {'1'} else {$null})
  APHROS_TWISTED_ADV_DUMP = $(if ($DumpAdvection) {'adv'} else {$null})
  APHROS_TWISTED_GEOMETRY_ONLY = $null
  APHROS_TWISTED_GEOMETRY_STATE_ONLY = $null
  APHROS_TWISTED_GEOMETRY_STATE_OUT = $null
  APHROS_TWISTED_GEOMETRY_STATE_IN = $geometryPath
  APHROS_TWISTED_FIX_EXPLICIT_RESIDUAL_HALO = $(if ($FixExplicitResidualHalo) {'1'} else {$null})
  APHROS_TWISTED_EXPLICIT_RESIDUAL_DUMP = $(if ($DumpExplicitResidual) {'explicit_residual'} else {$null})
  APHROS_TWISTED_FACTOR_CACHE = "$FactorCacheEntries"
  APHROS_TWISTED_FACTOR_TRACE = $(if ($TraceFactorCache) {'1'} else {$null})
  APHROS_TWISTED_PRESSURE_TOLERANCE = $(if ($PressureSolveTolerance -gt 0) {$PressureSolveTolerance.ToString('R', [Globalization.CultureInfo]::InvariantCulture)} else {$null})
  APHROS_TWISTED_CAPTURE_PRESSURE = $(if ($DumpPressureSystem) {'1'} else {$null})
  APHROS_TWISTED_CAPTURE_PRESSURE_FACES = $(if ($DumpPressureFaces) {'1'} else {$null})
  APHROS_TWISTED_VOLUME_COMPATIBILITY = $(if ($VolumePressureCompatibility) {'1'} else {$null})
}
$saved = @{}
$manifest = [ordered]@{
  executable = $exePath
  executable_sha256 = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash.ToLower()
  config_sha256 = (Get-FileHash -LiteralPath (Join-Path $casePath 'a.conf') -Algorithm SHA256).Hash.ToLower()
  environment = $settings
  started_utc = [DateTime]::UtcNow.ToString('o')
}
if ($geometryPath) {
  $manifest.geometry_state = $geometryPath
  $manifest.geometry_state_sha256 = (Get-FileHash -LiteralPath $geometryPath -Algorithm SHA256).Hash.ToLower()
}
$manifestPath = Join-Path $casePath 'run_manifest.json'
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Push-Location -LiteralPath $casePath
try {
  foreach ($entry in $settings.GetEnumerator()) {
    $saved[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, 'Process')
    if ($null -eq $entry.Value) {
      Remove-Item -LiteralPath "Env:$($entry.Key)" -ErrorAction SilentlyContinue
    } else {
      Set-Item -LiteralPath "Env:$($entry.Key)" -Value $entry.Value
    }
  }
  if (Test-Path -LiteralPath Env:APHROS_TWISTED_GEOMETRY_ONLY) { throw 'Geometry-only mode must be absent for a flow run.' }
  & $exePath a.conf *> $logPath
  $solverExit = $LASTEXITCODE
  $completion = @{completed_utc=[DateTime]::UtcNow.ToString('o'); exit_code=$solverExit}
  if ($geometryPath) { $completion.geometry_state_unchanged = (Get-FileHash -LiteralPath $geometryPath -Algorithm SHA256).Hash.ToLower() -eq $manifest.geometry_state_sha256 }
  $completion |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $casePath 'run_completion.json') -Encoding utf8
  if ($solverExit) { throw "Aphros exited with $solverExit; see $logPath" }
} finally {
  foreach ($entry in $saved.GetEnumerator()) {
    if ($null -eq $entry.Value) {
      Remove-Item -LiteralPath "Env:$($entry.Key)" -ErrorAction SilentlyContinue
    } else {
      Set-Item -LiteralPath "Env:$($entry.Key)" -Value $entry.Value
    }
  }
  Pop-Location
}
