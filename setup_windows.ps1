# One-click Windows dependency bootstrap for the native CUDA solver.
[CmdletBinding()]
param()

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$cudaVersion = '13.3.1'
$hdf5Version = '2.1.1'
$cudaUri = 'https://developer.download.nvidia.com/compute/cuda/13.3.1/network_installers/cuda_13.3.1_windows_network.exe'
$hdf5Uri = 'https://github.com/HDFGroup/hdf5/releases/download/2.1.1/hdf5-2.1.1-win-vs2022_cl.msi'
$hdf5Sha256 = 'F3F62ADF1F82355FEE7627413BC7620998671AC0201F635CB2EC03BA50E31932'
$restartRequired = $false

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdministrator)) {
    Write-Host 'Requesting Administrator permission...'
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    try {
        $elevated = Start-Process -FilePath 'powershell.exe' -Verb RunAs `
            -ArgumentList $arguments -Wait -PassThru
        exit $elevated.ExitCode
    }
    catch {
        Write-Error 'Administrator permission was not granted. Setup cannot continue.'
        exit 1
    }
}

$logPath = Join-Path $PSScriptRoot 'SETUP_WINDOWS.log'
$setupExitCode = 0
Start-Transcript -Path $logPath -Append | Out-Null

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machinePath;$userPath"
}

function Add-MachinePathEntry {
    param([Parameter(Mandatory = $true)][string]$Entry)

    $current = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $parts = @($current -split ';' | Where-Object { $_ })
    $normalizedEntry = $Entry.TrimEnd('\')
    $alreadyPresent = $false
    foreach ($part in $parts) {
        if ($part.Trim().TrimEnd('\') -ieq $normalizedEntry) {
            $alreadyPresent = $true
            break
        }
    }
    if (-not $alreadyPresent) {
        $newPath = if ([string]::IsNullOrWhiteSpace($current)) {
            $Entry
        }
        else {
            $current.TrimEnd(';') + ';' + $Entry
        }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'Machine')
    }
}

function Add-UserPathEntry {
    param(
        [Parameter(Mandatory = $true)][string]$Entry,
        [switch]$Prepend
    )

    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    $normalizedEntry = $Entry.TrimEnd('\')
    $parts = @($current -split ';' | Where-Object { $_ })
    $filtered = @($parts | Where-Object {
        $_.Trim().TrimEnd('\') -ine $normalizedEntry
    })
    if ($Prepend) {
        $newParts = @($Entry) + $filtered
    }
    else {
        $newParts = $filtered + @($Entry)
    }
    [Environment]::SetEnvironmentVariable('Path', ($newParts -join ';'), 'User')
}

function Install-WingetPackage {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [string]$Override = ''
    )

    Write-Host "`n=== Installing or repairing $Id ===" -ForegroundColor Cyan
    $arguments = @(
        'install', '--exact', '--id', $Id, '--source', 'winget',
        '--accept-package-agreements', '--accept-source-agreements',
        '--disable-interactivity'
    )
    if ($Override) {
        $arguments += @('--override', $Override)
    }
    else {
        $arguments += '--silent'
    }
    & winget.exe @arguments
    if ($LASTEXITCODE -ne 0) {
        $installExitCode = $LASTEXITCODE
        & winget.exe list --exact --id $Id --source winget `
            --accept-source-agreements --disable-interactivity | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "winget failed for $Id with exit code $installExitCode"
        }
        Write-Host "$Id is already installed; continuing with explicit verification."
    }
}

function Download-File {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (Test-Path $Destination) {
        Write-Host "Using cached download: $Destination"
        return
    }
    Write-Host "Downloading $Uri"
    Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Destination
}

function Find-Python312 {
    Refresh-ProcessPath
    $launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if ($launcher) {
        $resolved = & $launcher.Source -3.12 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return [string]$resolved
        }
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles 'Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $null
}

function Find-CudaRoot {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
        try {
            $currentName = Split-Path -Leaf $env:CUDA_PATH
            $currentVersion = [version]$currentName.TrimStart('v')
            if ($currentVersion.Major -eq 13 -and $currentVersion.Minor -eq 3) {
                return $env:CUDA_PATH
            }
        }
        catch {}
    }

    $base = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (-not (Test-Path $base)) {
        return $null
    }
    $versions = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
        Sort-Object { try { [version]$_.Name.TrimStart('v') } catch { [version]'0.0' } } -Descending
    foreach ($version in $versions) {
        try { $parsedVersion = [version]$version.Name.TrimStart('v') } catch { continue }
        if ($parsedVersion.Major -eq 13 -and $parsedVersion.Minor -eq 3 -and
            (Test-Path (Join-Path $version.FullName 'bin\nvcc.exe'))) {
                return $version.FullName
            }
    }
    return $null
}

function Find-Hdf5Root {
    if ($env:HDF5_ROOT -and (Test-Path (Join-Path $env:HDF5_ROOT 'include\hdf5.h'))) {
        return $env:HDF5_ROOT
    }

    $base = Join-Path $env:ProgramFiles 'HDF_Group\HDF5'
    if (-not (Test-Path $base)) {
        return $null
    }
    $versions = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
        Sort-Object { try { [version]$_.Name } catch { [version]'0.0' } } -Descending
    foreach ($version in $versions) {
        if (Test-Path (Join-Path $version.FullName 'include\hdf5.h')) {
            return $version.FullName
        }
    }
    return $null
}

function Find-VisualStudio2022CppTools {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        & $vswhere -latest -version '[17.0,18.0)' -products '*' `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath | Select-Object -First 1
    }
}

function Install-VisualStudioBuildTools {
    if (Find-VisualStudio2022CppTools) {
        Write-Host 'Visual Studio 2022 C++ tools are already installed; continuing with verification.'
        return
    }

    try {
        Install-WingetPackage -Id 'Microsoft.VisualStudio.2022.BuildTools' `
            -Override '--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    }
    catch {
        Write-Host "WinGet could not install Visual Studio: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host 'Trying the official Microsoft Visual Studio 2022 installer directly...' -ForegroundColor Cyan
        $downloadDirectory = Join-Path $env:TEMP 'gpe_cuda_solver_setup'
        New-Item -ItemType Directory -Path $downloadDirectory -Force | Out-Null
        $vsBootstrapper = Join-Path $downloadDirectory 'vs_buildtools_2022.exe'
        try {
            Download-File -Uri 'https://aka.ms/vs/17/release/vs_buildtools.exe' -Destination $vsBootstrapper
            $signature = Get-AuthenticodeSignature -FilePath $vsBootstrapper
            if ($signature.Status -ne 'Valid' -or
                $signature.SignerCertificate.Subject -notmatch '(^|,\s*)O=Microsoft Corporation(,|$)') {
                throw 'The Visual Studio installer does not have a valid Microsoft signature.'
            }
        }
        catch {
            Remove-Item -Path $vsBootstrapper -Force -ErrorAction SilentlyContinue
            throw "Visual Studio download failed: $($_.Exception.Message) Download https://aka.ms/vs/17/release/vs_buildtools.exe in your browser, install Desktop development with C++, then rerun SETUP_WINDOWS.bat."
        }
        $vsProcess = Start-Process -FilePath $vsBootstrapper `
            -ArgumentList '--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended' `
            -Wait -PassThru
        if ($vsProcess.ExitCode -eq 3010) {
            $script:restartRequired = $true
        }
        elseif ($vsProcess.ExitCode -ne 0) {
            throw "Visual Studio Build Tools installer failed with exit code $($vsProcess.ExitCode)"
        }
    }
}

try {
    Write-Host 'Native CUDA solver Windows setup' -ForegroundColor Green
    Write-Host "Project: $PSScriptRoot"
    Write-Host "Log:     $logPath"

    if (-not (Get-Command 'winget.exe' -ErrorAction SilentlyContinue)) {
        throw 'winget is unavailable. Install Microsoft App Installer from the Microsoft Store, then rerun SETUP_WINDOWS.bat.'
    }

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    Install-WingetPackage -Id 'Python.Python.3.12'
    Install-WingetPackage -Id 'Git.Git'
    Install-WingetPackage -Id 'Kitware.CMake'
    Install-WingetPackage -Id 'Ninja-build.Ninja'
    Install-VisualStudioBuildTools

    Refresh-ProcessPath

    $ninjaCommand = Get-Command 'ninja.exe' -ErrorAction SilentlyContinue
    if ($ninjaCommand) {
        $ninjaPath = $ninjaCommand.Source
    }
    else {
        $ninjaCandidates = @(
            (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\ninja.exe'),
            (Join-Path $env:ProgramFiles 'Ninja\ninja.exe')
        )
        $ninjaPath = $ninjaCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    }
    if (-not $ninjaPath) {
        throw 'Ninja was installed, but ninja.exe could not be found.'
    }
    $ninjaBin = Split-Path -Parent $ninjaPath
    $env:Path = "$ninjaBin;$env:Path"

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw 'Visual Studio Installer did not provide vswhere.exe.'
    }
    $vsInstall = & $vswhere -latest -version '[17.0,18.0)' -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsInstall) {
        $buildToolsInstall = & $vswhere -latest -version '[17.0,18.0)' `
            -products Microsoft.VisualStudio.Product.BuildTools `
            -property installationPath
        $vsInstaller = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\setup.exe'
        if ($buildToolsInstall -and (Test-Path $vsInstaller)) {
            Write-Host 'Adding the missing Visual Studio C++ workload...' -ForegroundColor Cyan
            $vsArguments = "modify --installPath `"$buildToolsInstall`" --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
            $vsProcess = Start-Process -FilePath $vsInstaller -ArgumentList $vsArguments -Wait -PassThru
            if ($vsProcess.ExitCode -eq 3010) {
                $restartRequired = $true
            }
            elseif ($vsProcess.ExitCode -ne 0) {
                throw "Visual Studio C++ workload installation failed with exit code $($vsProcess.ExitCode)"
            }
            $vsInstall = & $vswhere -latest -version '[17.0,18.0)' -products '*' `
                -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                -property installationPath
        }
        if (-not $vsInstall) {
            throw 'Visual Studio 2022 C++ x64 tools were not installed. Rerun setup and allow the Visual Studio installer to finish.'
        }
    }

    $cudaRoot = Find-CudaRoot
    if (-not $cudaRoot) {
        Write-Host "`n=== Installing NVIDIA CUDA Toolkit $cudaVersion ===" -ForegroundColor Cyan
        $downloadDirectory = Join-Path $env:TEMP 'gpe_cuda_solver_setup'
        New-Item -ItemType Directory -Path $downloadDirectory -Force | Out-Null
        $cudaInstaller = Join-Path $downloadDirectory "cuda_${cudaVersion}_windows_network.exe"
        Download-File -Uri $cudaUri -Destination $cudaInstaller
        Write-Host 'The CUDA network installer is now downloading and installing its components.'
        $cudaProcess = Start-Process -FilePath $cudaInstaller -ArgumentList '-s' -Wait -PassThru
        if ($cudaProcess.ExitCode -eq 3010) {
            $restartRequired = $true
        }
        elseif ($cudaProcess.ExitCode -ne 0) {
            Remove-Item -Path $cudaInstaller -Force -ErrorAction SilentlyContinue
            throw "CUDA Toolkit installer failed with exit code $($cudaProcess.ExitCode)"
        }
        $cudaRoot = Find-CudaRoot
    }
    if (-not $cudaRoot) {
        throw 'CUDA installation finished, but bin\nvcc.exe was not found.'
    }
    Write-Host "CUDA Toolkit: $cudaRoot"

    $hdf5Root = Find-Hdf5Root
    if (-not $hdf5Root) {
        Write-Host "`n=== Installing HDF5 $hdf5Version for Visual Studio 2022 ===" -ForegroundColor Cyan
        $downloadDirectory = Join-Path $env:TEMP 'gpe_cuda_solver_setup'
        New-Item -ItemType Directory -Path $downloadDirectory -Force | Out-Null
        $hdf5Installer = Join-Path $downloadDirectory "hdf5-${hdf5Version}-win-vs2022_cl.msi"
        Download-File -Uri $hdf5Uri -Destination $hdf5Installer
        $actualHash = (Get-FileHash -Algorithm SHA256 -Path $hdf5Installer).Hash
        if ($actualHash -ne $hdf5Sha256) {
            Remove-Item -Path $hdf5Installer -Force
            throw 'The downloaded HDF5 installer checksum did not match the official release checksum.'
        }
        $hdf5Process = Start-Process -FilePath 'msiexec.exe' `
            -ArgumentList "/i `"$hdf5Installer`" /qn /norestart" -Wait -PassThru
        if ($hdf5Process.ExitCode -eq 3010) {
            $restartRequired = $true
        }
        elseif ($hdf5Process.ExitCode -ne 0) {
            throw "HDF5 installer failed with exit code $($hdf5Process.ExitCode)"
        }
        $hdf5Root = Find-Hdf5Root
    }
    if (-not $hdf5Root) {
        throw 'HDF5 installation finished, but include\hdf5.h was not found.'
    }
    Write-Host "HDF5:        $hdf5Root"

    $python = Find-Python312
    if (-not $python) {
        throw 'Python 3.12 was installed, but python.exe could not be found.'
    }
    Write-Host "Python:      $python"

    $cmakeCommand = Get-Command 'cmake.exe' -ErrorAction SilentlyContinue
    if ($cmakeCommand) {
        $cmakePath = $cmakeCommand.Source
    }
    else {
        $cmakeCandidate = Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe'
        if (-not (Test-Path $cmakeCandidate)) {
            throw 'CMake was installed, but cmake.exe could not be found.'
        }
        $cmakePath = $cmakeCandidate
    }
    $cmakeBin = Split-Path -Parent $cmakePath
    $cmakeVersionLine = (& $cmakePath --version | Select-Object -First 1)
    if ($cmakeVersionLine -notmatch '(\d+\.\d+\.\d+)') {
        throw 'CMake is installed, but its version could not be determined.'
    }
    if ([version]($Matches[1]) -lt [version]'3.27.0') {
        throw "CMake 3.27 or newer is required; found $($Matches[1])."
    }

    [Environment]::SetEnvironmentVariable('CUDA_PATH', $cudaRoot, 'Machine')
    [Environment]::SetEnvironmentVariable('HDF5_ROOT', $hdf5Root, 'Machine')
    Add-MachinePathEntry -Entry $cmakeBin
    Add-MachinePathEntry -Entry (Join-Path $cudaRoot 'bin')
    $cudaRuntimeBin = Join-Path $cudaRoot 'bin\x64'
    if (Test-Path $cudaRuntimeBin) {
        Add-MachinePathEntry -Entry $cudaRuntimeBin
    }
    Add-MachinePathEntry -Entry (Join-Path $hdf5Root 'bin')

    $env:CUDA_PATH = $cudaRoot
    $env:HDF5_ROOT = $hdf5Root
    Refresh-ProcessPath
    $env:Path = "$cudaRuntimeBin;$(Join-Path $cudaRoot 'bin');$(Join-Path $hdf5Root 'bin');$cmakeBin;$env:Path"

    Write-Host "`n=== Creating the project Python environment ===" -ForegroundColor Cyan
    $venv = Join-Path $PSScriptRoot '.venv'
    $venvPython = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $venvPython)) {
        & $python -m venv $venv
        if ($LASTEXITCODE -ne 0) {
            throw 'Python failed to create the .venv environment.'
        }
    }
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw 'pip upgrade failed.'
    }
    $requirements = Join-Path $PSScriptRoot 'phase_diagram\simulation_core\requirements_CUDA.txt'
    & $venvPython -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) {
        throw 'Python dependency installation failed.'
    }
    $mklRoot = Join-Path $venv 'Library'
    $mklDirectory = Join-Path $mklRoot 'lib\cmake\mkl'
    $mklConfig = Join-Path $mklDirectory 'MKLConfig.cmake'
    $mklBin = Join-Path $mklRoot 'bin'
    if (-not (Test-Path $mklConfig)) {
        throw 'The oneMKL development package was installed, but MKLConfig.cmake was not found.'
    }

    Write-Host "`n=== Saving dependency paths for future terminals ===" -ForegroundColor Cyan
    $pythonDirectory = Split-Path -Parent $python
    $pythonScripts = Join-Path $pythonDirectory 'Scripts'
    $pythonLauncher = Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher'
    $venvScripts = Join-Path $venv 'Scripts'
    Add-UserPathEntry -Entry $pythonDirectory
    if (Test-Path $pythonScripts) {
        Add-UserPathEntry -Entry $pythonScripts
    }
    if (Test-Path $pythonLauncher) {
        Add-UserPathEntry -Entry $pythonLauncher
    }
    Add-UserPathEntry -Entry $ninjaBin
    Add-UserPathEntry -Entry $venvScripts -Prepend
    Add-UserPathEntry -Entry $mklBin
    [Environment]::SetEnvironmentVariable('GPE_CUDA_PROJECT_ROOT', $PSScriptRoot, 'User')
    [Environment]::SetEnvironmentVariable('GPE_PYTHON', $venvPython, 'User')
    [Environment]::SetEnvironmentVariable('MKLROOT', $mklRoot, 'User')
    [Environment]::SetEnvironmentVariable('MKL_DIR', $mklDirectory, 'User')

    Refresh-ProcessPath
    $env:MKLROOT = $mklRoot
    $env:MKL_DIR = $mklDirectory
    $env:Path = "$venvScripts;$mklBin;$ninjaBin;$cudaRuntimeBin;$(Join-Path $cudaRoot 'bin');$(Join-Path $hdf5Root 'bin');$cmakeBin;$env:Path"
    & $venvPython -c 'import sys, numpy, scipy, tables, matplotlib; print(sys.executable)'
    if ($LASTEXITCODE -ne 0) {
        throw 'The project Python environment cannot import the required packages.'
    }
    Write-Host "Project Python: $venvPython"
    Write-Host "oneMKL:        $mklRoot"
    Write-Host 'Python, pip, CMake, Ninja, CUDA, HDF5, and oneMKL paths are now persistent.'

    Write-Host "`n=== Verifying the complete environment ===" -ForegroundColor Cyan
    $env:GPE_NO_PAUSE = '1'
    & (Join-Path $PSScriptRoot 'CHECK_CUDA.bat')
    if ($LASTEXITCODE -ne 0) {
        throw "CHECK_CUDA.bat failed with exit code $LASTEXITCODE"
    }

    Write-Host "`nSetup is complete." -ForegroundColor Green
    Write-Host 'Open a new terminal, or run BUILD_CUDA.bat next. The batch files load all paths automatically.'
    Write-Host 'Then run VALIDATE_COMPACT_CUDA.bat to compare compact metrics against saved snapshots.'
    Write-Host 'RUN_PHASE_DIAGRAM_CUDA.bat now saves one scalar per point by default; use --storage full for snapshots.'
    if ($restartRequired) {
        Write-Host 'One installer requested a Windows restart. Restart before building if CHECK_CUDA.bat reports a driver or compiler error.' -ForegroundColor Yellow
    }
}
catch {
    $setupExitCode = 1
    Write-Host "`nSETUP ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "See the complete log at: $logPath" -ForegroundColor Yellow
}
finally {
    Stop-Transcript | Out-Null
}

exit $setupExitCode
