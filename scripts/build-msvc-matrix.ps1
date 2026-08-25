param(
    [string]$Out = "target/r2smt-bench-corpus/portable-matrix-msvc"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repo "bench/corpus/portable-matrix/source/main.c"
$outDir = Join-Path $repo $Out
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($vs)) {
    throw "Visual Studio C++ toolchain is required"
}
$vcvarsall = Join-Path $vs.Trim() "VC/Auxiliary/Build/vcvarsall.bat"

$variants = @(
    @{ artifact = "x86_64-msvc-O0.obj"; architecture = "x86_64"; toolchain = "x64"; optimization = "O0"; flags = "/Od" },
    @{ artifact = "x86_64-msvc-O2.obj"; architecture = "x86_64"; toolchain = "x64"; optimization = "O2"; flags = "/O2" },
    @{ artifact = "x86-msvc-O2.obj"; architecture = "x86"; toolchain = "x86"; optimization = "O2"; flags = "/O2" }
)

foreach ($variant in $variants) {
    $object = Join-Path $outDir $variant.artifact
    $command = "call `"$vcvarsall`" $($variant.toolchain) >nul && cl /nologo /c /TC $($variant.flags) /Fo`"$object`" `"$source`""
    & cmd.exe /d /c $command
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $object)) {
        throw "MSVC failed for $($variant.artifact)"
    }
}

@{
    schema_version = 1
    fixture = "portable-matrix"
    variants = $variants | ForEach-Object {
        @{
            artifact = $_.artifact
            architecture = $_.architecture
            compiler = "msvc"
            optimization = $_.optimization
            format = "coff-relocatable"
        }
    }
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $outDir "manifest.json")

Write-Output $outDir
