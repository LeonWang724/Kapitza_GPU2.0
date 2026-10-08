# Run with powershell.exe -NoProfile -File tests\setup_windows_fallback_test.ps1.
# Parse the real setup source and load only the installation function. All
# downloads, signature checks and installer launches are replaced with stubs.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$setupPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'setup_windows.ps1'
$parseTokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $setupPath, [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw ($parseErrors | Out-String) }
$functionAst = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Install-VisualStudioBuildTools'
}, $true)
if (-not $functionAst) { throw 'The setup installation function is missing.' }
Invoke-Expression $functionAst.Extent.Text

$testDirectory = Join-Path ([IO.Path]::GetTempPath()) ('gpe setup test ' + [guid]::NewGuid())
$previousTemp = $env:TEMP
$env:TEMP = $testDirectory

function Assert-True($condition, $message) {
    if (-not $condition) { throw $message }
}

function Reset-Case {
    $script:alreadyInstalled = $false
    $script:wingetFails = $false
    $script:downloadFails = $false
    $script:signatureStatus = 'Valid'
    $script:signatureSubject = 'CN=Microsoft Corporation, O=Microsoft Corporation, C=US'
    $script:installerExitCode = 0
    $script:restartRequired = $false
    $script:wingetCalls = 0
    $script:downloadCalls = 0
    $script:launchCalls = 0
}

function Find-VisualStudio2022CppTools {
    if ($script:alreadyInstalled) { return 'C:\Program Files\VS2022' }
}

function Install-WingetPackage($Id, $Override) {
    $script:wingetCalls++
    Assert-True ($Id -eq 'Microsoft.VisualStudio.2022.BuildTools') 'Wrong WinGet package.'
    if ($script:wingetFails) { throw 'InternetOpenUrl() failed: 0x80072f78' }
}

function Download-File($Uri, $Destination) {
    $script:downloadCalls++
    Assert-True ($Uri -eq 'https://aka.ms/vs/17/release/vs_buildtools.exe') 'Wrong Microsoft installer URL.'
    Set-Content -LiteralPath $Destination -Value 'stand-in download'
    if ($script:downloadFails) { throw 'Download interrupted' }
}

function Get-AuthenticodeSignature($FilePath) {
    Assert-True (Test-Path -LiteralPath $FilePath) 'The downloaded file is missing.'
    return [pscustomobject]@{
        Status = $script:signatureStatus
        SignerCertificate = [pscustomobject]@{Subject = $script:signatureSubject}
    }
}

function Start-Process($FilePath, $ArgumentList, [switch]$Wait, [switch]$PassThru) {
    $script:launchCalls++
    Assert-True (Test-Path -LiteralPath $FilePath) 'The installer is missing.'
    Assert-True ($Wait -and $PassThru) 'The installer must be monitored until completion.'
    Assert-True ($ArgumentList -match '--wait' -and
        $ArgumentList -match '--add Microsoft.VisualStudio.Workload.VCTools' -and
        $ArgumentList -match '--includeRecommended') 'The C++ workload arguments are missing.'
    return [pscustomobject]@{ExitCode = $script:installerExitCode}
}

function Assert-Failure($expectedMessage) {
    $caught = $false
    try { Install-VisualStudioBuildTools }
    catch {
        $caught = $true
        Assert-True ($_.Exception.Message -like "*$expectedMessage*") 'Unexpected failure message.'
    }
    Assert-True $caught 'Expected setup to reject this case.'
}

try {
    Reset-Case
    $script:alreadyInstalled = $true
    Install-VisualStudioBuildTools
    Assert-True ($wingetCalls -eq 0 -and $launchCalls -eq 0) 'Existing tools should skip installation.'

    Reset-Case
    Install-VisualStudioBuildTools
    Assert-True ($wingetCalls -eq 1 -and $downloadCalls -eq 0) 'Successful WinGet should skip fallback.'

    Reset-Case
    $script:wingetFails = $true
    Install-VisualStudioBuildTools
    Assert-True ($downloadCalls -eq 1 -and $launchCalls -eq 1) 'Network failure should use the direct installer.'
    Assert-True (-not $restartRequired) 'Successful installation should not require a restart.'

    Reset-Case
    $script:wingetFails = $true
    $script:installerExitCode = 3010
    Install-VisualStudioBuildTools
    Assert-True $restartRequired 'Installer restart requests must be preserved.'

    Reset-Case
    $script:wingetFails = $true
    $script:installerExitCode = 7
    Assert-Failure 'exit code 7'

    Reset-Case
    $script:wingetFails = $true
    $script:signatureStatus = 'NotSigned'
    Assert-Failure 'valid Microsoft signature'
    Assert-True ($launchCalls -eq 0) 'An unsigned download must not be executed.'

    Reset-Case
    $script:wingetFails = $true
    $script:signatureSubject = 'CN=Other Publisher, O=Other Publisher, C=US'
    Assert-Failure 'valid Microsoft signature'
    Assert-True ($launchCalls -eq 0) 'An unexpected publisher must not be executed.'

    Reset-Case
    $script:wingetFails = $true
    $script:downloadFails = $true
    Assert-Failure 'Download interrupted'
    Assert-True ($launchCalls -eq 0) 'A partial download must not be executed.'
    Assert-True (-not (Test-Path (Join-Path $testDirectory 'gpe_cuda_solver_setup/vs_buildtools_2022.exe'))) 'Partial download was not removed.'

    Write-Host 'PASS: setup source parses and all 8 Visual Studio installation cases passed.'
}
finally {
    $env:TEMP = $previousTemp
    Remove-Item -LiteralPath $testDirectory -Recurse -Force -ErrorAction SilentlyContinue
}
