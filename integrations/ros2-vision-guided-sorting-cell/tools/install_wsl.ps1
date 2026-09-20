$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]$identity
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Reopening this installer with administrator rights..."
    Start-Process powershell.exe -Verb RunAs -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath
    )
    exit
}

$scriptDir = Split-Path -Parent $PSCommandPath
$logPath = Join-Path $scriptDir "wsl_install.log"
$stdoutPath = Join-Path $scriptDir "wsl_install.stdout.log"
$stderrPath = Join-Path $scriptDir "wsl_install.stderr.log"
Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
"Started: $(Get-Date -Format o)" | Set-Content -LiteralPath $logPath -Encoding UTF8

try {
    $process = Start-Process -FilePath "$env:SystemRoot\System32\wsl.exe" `
        -ArgumentList @("--install", "-d", "Ubuntu-24.04", "--no-launch") `
        -Wait -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath
    "ExitCode: $($process.ExitCode)" | Add-Content -LiteralPath $logPath -Encoding UTF8
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath | Add-Content -LiteralPath $logPath -Encoding UTF8
    }
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath | Add-Content -LiteralPath $logPath -Encoding UTF8
    }
    if ($process.ExitCode -ne 0) {
        "Online installer failed; enabling inbox WSL features instead." | Add-Content -LiteralPath $logPath -Encoding UTF8
        $dism = Join-Path $env:SystemRoot "System32\dism.exe"
        $featureResults = @()
        foreach ($feature in @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")) {
            $featureLog = Join-Path $scriptDir ("dism_" + $feature + ".log")
            $featureProcess = Start-Process -FilePath $dism `
                -ArgumentList @("/Online", "/Enable-Feature", "/FeatureName:$feature", "/All", "/NoRestart") `
                -Wait -PassThru -RedirectStandardOutput $featureLog
            $featureResults += "$feature=$($featureProcess.ExitCode)"
        }
        & (Join-Path $env:SystemRoot "System32\bcdedit.exe") /set hypervisorlaunchtype auto | Out-Null
        "FeatureExitCodes: $($featureResults -join ', ')" | Add-Content -LiteralPath $logPath -Encoding UTF8
        "RestartRequired: true" | Add-Content -LiteralPath $logPath -Encoding UTF8
    }
} catch {
    "Installer error: $($_.Exception.Message)" | Add-Content -LiteralPath $logPath -Encoding UTF8
    throw
}
Write-Host ""
Write-Host "WSL installation command completed."
Write-Host "Restart Windows if requested, launch Ubuntu-24.04 once, then run tools/install_ros2_jazzy.sh."
Write-Host "Log: $logPath"
