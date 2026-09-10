param(
  [string]$AphrosRoot = (Join-Path $PSScriptRoot 'aphros'),
  [string]$VcVars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
  [string]$Sh = 'C:\Program Files\Git\bin\sh.exe',
  [string]$EigenInclude = '',
  [string]$AmgclInclude = '',
  [string]$ExecutableName = 'main.exe'
)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $AphrosRoot 'src')
try {
  & $Sh ../make/bootstrap
  if ($LASTEXITCODE) { throw 'Aphros bootstrap failed' }
  # /Zi is intentionally omitted: PDB generation failed in this Dropbox tree.
  # Link explicitly; upstream NMakefile does not track main.exe -> library.
  $flags = '/O2 /nologo'
  if ($EigenInclude) { $flags += ' /I' + $EigenInclude }
  if ($AmgclInclude) { $flags += ' /openmp /DAPHROS_TWISTED_HAVE_AMGCL /I' + $AmgclInclude }
  $command = '"' + $VcVars + '" && nmake /nologo /f NMakefile CXXFLAGS="' + $flags + '" libaphros_static.lib && cl /c /O2 /nologo main.c && cl /nologo main.obj libaphros_static.lib /link /out:' + $ExecutableName
  & cmd /d /c $command
  if ($LASTEXITCODE) { throw 'Aphros build failed' }
} finally {
  Pop-Location
}
