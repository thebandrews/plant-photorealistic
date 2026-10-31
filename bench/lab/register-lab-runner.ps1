<#
  register-lab-runner.ps1 - run ONCE on the VM, signed in as yourself (ben).

  Registers the PhotorealLab-Runner scheduled task to run as you whether or not you're signed in,
  so the lab's Codex and Claude Code agents use your sign-ins. Windows needs your password for that;
  it's handed straight to Task Scheduler and is not written anywhere else.
#>
$ErrorActionPreference = 'Stop'
$me = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$me).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Re-opening as administrator (accept the prompt)...'
    Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-NoExit', '-File', $PSCommandPath
    return
}
$user = $me.Name
$sec = Read-Host "Windows password for $user" -AsSecureString
$plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument '-NoProfile -ExecutionPolicy Bypass -File C:\Workstation\lab\repo\bench\lab\lab-runner.ps1'
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
try {
    Register-ScheduledTask -TaskName 'PhotorealLab-Runner' -Action $action -Settings $settings -User $user `
        -Password $plain -RunLevel Limited -Description 'Photoreal lab: runs queued experiments (lab_runner.py)' -Force | Out-Null
} finally { $plain = $null }
Write-Host 'PhotorealLab-Runner registered.' -ForegroundColor Green
