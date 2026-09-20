param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$LinuxUser = "",
    [string]$LinuxProjectDir = "",
    [switch]$SkipRosInstall
)

$ErrorActionPreference = "Stop"
$projectSource = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectName = Split-Path -Leaf $projectSource

function Invoke-WslScript {
    param([Parameter(Mandatory = $true)][string]$Script)

    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Script))
    & wsl.exe -d $Distro -u $LinuxUser -- bash -lc "echo '$encoded' | base64 -d | bash"
    if ($LASTEXITCODE -ne 0) {
        throw "WSL script failed with exit code $LASTEXITCODE"
    }
}

$installed = @(& wsl.exe --list --quiet) | ForEach-Object { $_.Trim([char]0).Trim() }
if ($installed -notcontains $Distro) {
    throw "WSL distribution '$Distro' is not installed. Run tools/install_wsl.ps1 first."
}

if (-not $LinuxUser) {
    $LinuxUser = (& wsl.exe -d $Distro -- bash -lc "id -un").Trim()
}
if (-not $LinuxUser) {
    throw "Could not determine the default Linux user for '$Distro'."
}
if (-not $LinuxProjectDir) {
    $LinuxProjectDir = "/home/$LinuxUser/$projectName"
}
if (-not $LinuxProjectDir.StartsWith("/home/$LinuxUser/")) {
    throw "LinuxProjectDir must stay inside /home/$LinuxUser/."
}

$driveLetter = $projectSource.Substring(0, 1).ToLowerInvariant()
$pathRemainder = $projectSource.Substring(2).Replace("\", "/")
$linuxSource = "/mnt/$driveLetter$pathRemainder"

$syncProject = @"
set -euo pipefail
source_dir='$linuxSource'
destination='$LinuxProjectDir'
mkdir -p "`$destination"
find "`$destination" -mindepth 1 -maxdepth 1 \
  ! -name build ! -name install ! -name log \
  -exec rm -rf -- {} +
tar -C "`$source_dir" \
  --exclude=.git --exclude=build --exclude=install --exclude=log \
  --exclude=.pytest_cache --exclude='*/__pycache__' --exclude='*.pyc' \
  -cf - . | tar -C "`$destination" -xf -
chmod +x "`$destination"/tools/*.sh
"@
Invoke-WslScript -Script $syncProject

$buildProject = @"
set -euo pipefail
cd '$LinuxProjectDir'
if [[ '${SkipRosInstall}' != 'True' ]]; then
  ./tools/install_ros2_jazzy.sh
fi
./tools/build_ros2.sh
"@
Invoke-WslScript -Script $buildProject

Write-Host ""
Write-Host "WSL workspace is ready."
Write-Host "Distribution: $Distro"
Write-Host "Linux project: $LinuxProjectDir"
Write-Host "Launch: wsl -d $Distro -u $LinuxUser -- bash -lc 'cd $LinuxProjectDir && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch sorting_cell_bringup sorting_cell.launch.py'"
