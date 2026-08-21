param(
    [string]$Destination = (Join-Path (Split-Path $PSScriptRoot -Parent) "python-3.11.9-amd64.exe")
)

$ErrorActionPreference = "Stop"
$Url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$ExpectedSha256 = "5ee42c4eee1e6b4464bb23722f90b45303f79442df63083f05322f1785f5fdde"

function Test-VerifiedRuntime([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    return $Actual -eq $ExpectedSha256
}

if (Test-VerifiedRuntime $Destination) {
    Write-Host "Verified Python runtime already exists: $Destination"
    exit 0
}

if (Test-Path -LiteralPath $Destination) {
    Remove-Item -LiteralPath $Destination -Force
}

Write-Host "Downloading Python 3.11.9 from python.org..."
Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing

if (-not (Test-VerifiedRuntime $Destination)) {
    Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
    throw "Python installer checksum verification failed."
}

Write-Host "Download complete and SHA-256 checksum verified."
