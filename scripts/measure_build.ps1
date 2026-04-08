$sw = [System.Diagnostics.Stopwatch]::StartNew()
xmake -j8 2>&1 | Out-Default
$sw.Stop()
Write-Host ""
Write-Host "=== BUILD TIME ==="
Write-Host ("Total: {0:F1} seconds" -f $sw.Elapsed.TotalSeconds)
