[CmdletBinding()]
param(
    [ValidateRange(1, 1800)]
    [int]$TimeoutSeconds = 240,
    [string]$Distro = "Ubuntu-24.04",
    [string]$LinuxProjectDir = "",
    [string]$OutputDir = "",
    [switch]$SkipSyncBuild
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectName = Split-Path -Leaf $projectRoot

$installed = @(& wsl.exe --list --quiet) | ForEach-Object { $_.Trim([char]0).Trim() }
if ($installed -notcontains $Distro) {
    throw "WSL distribution '$Distro' is not installed."
}

$linuxUser = (& wsl.exe -d $Distro -- bash -lc "id -un").Trim()
if (-not $linuxUser) {
    throw "Could not determine the default Linux user for '$Distro'."
}
if (-not $LinuxProjectDir) {
    $LinuxProjectDir = "/home/$linuxUser/$projectName"
}
if (-not $LinuxProjectDir.StartsWith("/home/$linuxUser/")) {
    throw "LinuxProjectDir must stay inside /home/$linuxUser/."
}

if (-not $OutputDir) {
    $runStamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $OutputDir = Join-Path $projectRoot "results\ros2-runtime-$runStamp"
} elseif (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $projectRoot $OutputDir
}
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $SkipSyncBuild) {
    & (Join-Path $PSScriptRoot "setup_wsl_after_reboot.ps1") `
        -Distro $Distro `
        -LinuxUser $linuxUser `
        -LinuxProjectDir $LinuxProjectDir `
        -SkipRosInstall
}

if ($OutputDir -notmatch '^[A-Za-z]:\\') {
    throw "OutputDir must be a local Windows drive path: $OutputDir"
}
$outputDrive = $OutputDir.Substring(0, 1).ToLowerInvariant()
$outputRemainder = $OutputDir.Substring(2).Replace("\", "/")
$linuxOutputDir = "/mnt/$outputDrive$outputRemainder"

function ConvertTo-BashLiteral {
    param([Parameter(Mandatory = $true)][string]$Value)
    $singleQuote = "'"
    $doubleQuote = '"'
    $escapedQuote = $singleQuote + $doubleQuote + $singleQuote + $doubleQuote + $singleQuote
    return $singleQuote + $Value.Replace($singleQuote, $escapedQuote) + $singleQuote
}

$linuxProjectLiteral = ConvertTo-BashLiteral $LinuxProjectDir
$linuxOutputLiteral = ConvertTo-BashLiteral $linuxOutputDir
$linuxScript = @"
set -euo pipefail
cd $linuxProjectLiteral
exec ./tools/run_ros2_demo.sh --skip-build --timeout $TimeoutSeconds --output-dir $linuxOutputLiteral
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($linuxScript))

& wsl.exe -d $Distro -- bash -lc "echo '$encoded' | base64 -d | bash"
$runExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "Evidence directory: $OutputDir"
exit $runExitCode
