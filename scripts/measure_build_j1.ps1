$sw = [System.Diagnostics.Stopwatch]::StartNew()
xmake -j1 2>&1 | ForEach-Object {
    $line = $_
    Write-Host $line
    if ($line -match 'compiling\.release\s+(\S+)') {
        $file = $Matches[1]
        $now = $sw.Elapsed.TotalSeconds
        Write-Host "  [TIMER] $file started at $now s"
    }
    elseif ($line -match 'linking\.release\s+(\S+)') {
        $file = $Matches[1]
        $now = $sw.Elapsed.TotalSeconds
        Write-Host "  [TIMER] LINK $file started at $now s"
    }
    elseif ($line -match 'devlinking\.release\s+(\S+)') {
        $file = $Matches[1]
        $now = $sw.Elapsed.TotalSeconds
        Write-Host "  [TIMER] DEVLINK $file started at $now s"
    }
}
$sw.Stop()
Write-Host ""
Write-Host "=== BUILD TIME (single-threaded) ==="
Write-Host ("Total: {0:F1} seconds" -f $sw.Elapsed.TotalSeconds)
