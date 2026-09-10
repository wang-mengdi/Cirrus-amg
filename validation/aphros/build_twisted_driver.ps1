param(
  [string]$AphrosRoot = 'D:/Dropbox/Agent-simulation/twisted-baseline/aphros',
  [string]$OutputDirectory = 'D:/Dropbox/Agent-simulation/twisted-baseline/driver_build_v1',
  [string]$Source = '',
  [string[]]$ExtraSources = @(),
  [string[]]$IncludeDirectories = @(),
  [string[]]$Defines = @(),
  [ValidateSet('14','17')][string]$CxxStandard = '14',
  [ValidatePattern('^[A-Za-z0-9_-]+\.exe$')][string]$ExecutableName = 'twisted_driver.exe',
  [string]$VcVars = 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat'
)
$ErrorActionPreference = 'Stop'
if (!$Source) { $Source = Join-Path $PSScriptRoot 'twisted_driver.cpp' }
$source = (Resolve-Path -LiteralPath $Source).Path
$root = (Resolve-Path -LiteralPath $AphrosRoot).Path
$library = Join-Path $root 'src/libaphros_static.lib'
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDirectory).Path
$executable = Join-Path $output $ExecutableName
if (Test-Path -LiteralPath $executable) { throw 'Use a fresh driver build directory; existing executables are retained.' }
# Match the library's NMakefile feature/ABI settings without rebuilding it.
$flags = '/std:c++14 /EHsc /O2 /nologo /openmp /DM_PI=3.141592653589793 /D_ALIGNBYTES_=16 /D_USE_AMGX_=0 /D_USE_AVX_=0 /D_USE_BACKEND_CUBISM_=0 /D_USE_BACKEND_LOCAL_=1 /D_USE_BACKEND_NATIVE_=1 /D_USE_DIM1_=0 /D_USE_DIM2_=1 /D_USE_DIM3_=1 /D_USE_DIM4_=0 /D_USE_FPZIP_=0 /D_USE_HDF_=0 /D_USE_HYPRE_=0 /D_USE_MPI_=0 /D_USE_OPENMP_=1'
$flags = $flags.Replace('/std:c++14', '/std:c++' + $CxxStandard)
$extra = @($ExtraSources | ForEach-Object { (Resolve-Path -LiteralPath $_).Path })
$extraArgs = ($extra | ForEach-Object { ' "' + $_ + '"' }) -join ''
$includes = @($IncludeDirectories | ForEach-Object { (Resolve-Path -LiteralPath $_).Path })
foreach ($define in $Defines) {
  if ($define -notmatch '^[A-Za-z_][A-Za-z0-9_]*(=[A-Za-z0-9_.+-]+)?$') { throw 'Invalid compiler definition' }
}
$includeArgs = ($includes | ForEach-Object { ' /I"' + $_ + '"' }) -join ''
$defineArgs = ($Defines | ForEach-Object { ' /D' + $_ }) -join ''
$command = '"' + $VcVars + '" && cl ' + $flags + $defineArgs + $includeArgs + ' /I"' + (Join-Path $root 'src') + '" /I"' + (Join-Path $root 'src/solver') + '" "' + $source + '"' + $extraArgs + ' "' + $library + '" psapi.lib /link /out:"' + $executable + '"'
$record = [ordered]@{
  source=$source; source_sha256=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower()
  library=$library; library_sha256=(Get-FileHash -LiteralPath $library -Algorithm SHA256).Hash.ToLower()
  started_utc=[DateTime]::UtcNow.ToString('o'); flags=$flags
  extra_sources=@($extra | ForEach-Object { @{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLower()} })
  include_directories=$includes; defines=$Defines
}
Push-Location -LiteralPath $output
try {
  & cmd /d /c $command
  $record.exit_code=$LASTEXITCODE
  if ($LASTEXITCODE) { throw 'Aphros utility build failed' }
  $record.executable_sha256=(Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLower()
} finally {
  $record.completed_utc=[DateTime]::UtcNow.ToString('o')
  $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'build_manifest.json') -Encoding utf8
  Pop-Location
}
