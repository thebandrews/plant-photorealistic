<#
  lab-runner.ps1 - action of the PhotorealLab-Runner scheduled task (runs as the VM user).
  Drains C:\Workstation\lab\queue with lab_runner.py (up to 3 experiments at once), then exits.
  Start it after queueing: Start-ScheduledTask PhotorealLab-Runner
#>
$ErrorActionPreference = 'Continue'
$lab = 'C:\Workstation\lab'
# Task processes don't always get the user's PATH (Claude Code, Codex, node)
$env:Path = "$lab\.venv\Scripts;" + [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User') + ";$env:USERPROFILE\.local\bin;$env:APPDATA\npm"
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HOME = "$lab\hf-cache"
New-Item -ItemType Directory -Force -Path "$lab\logs" | Out-Null
$log = "$lab\logs\runner.log"
Add-Content $log "=== $(Get-Date -Format s) lab runner starts"
& "$lab\.venv\Scripts\python.exe" "$lab\repo\bench\lab\lab_runner.py" run *>> $log
Add-Content $log "=== $(Get-Date -Format s) lab runner exit $LASTEXITCODE"
