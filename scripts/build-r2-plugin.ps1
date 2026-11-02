[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Radare2Prefix,

    [Parameter(Mandatory = $true)]
    [string]$Output
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$prefix = (Resolve-Path $Radare2Prefix).Path
$source = (Resolve-Path (Join-Path $PSScriptRoot '..\r2pm\core_r2smt.c')).Path
$include = Join-Path $prefix 'include\libr'
$lib = Join-Path $prefix 'lib'
$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputDirectory = Split-Path -Parent $outputPath
$object = Join-Path $outputDirectory 'core_r2smt.obj'
$libraries = @(
    'r_core.lib',
    'r_config.lib',
    'r_bin.lib',
    'r_anal.lib',
    'r_io.lib',
    'r_cons.lib',
    'r_util.lib'
)

if (-not (Test-Path $include -PathType Container)) {
    throw "radare2 headers not found below $include"
}
foreach ($library in $libraries) {
    if (-not (Test-Path (Join-Path $lib $library) -PathType Leaf)) {
        throw "radare2 import library not found: $library"
    }
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$arguments = @(
    '/nologo',
    '/LD',
    '/O2',
    '/W4',
    "/I$include",
    "/Fo$object",
    $source,
    '/link',
    "/OUT:$outputPath",
    "/LIBPATH:$lib",
    '/EXPORT:radare_plugin'
) + $libraries

& cl.exe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "cl.exe failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $outputPath -PathType Leaf)) {
    throw "plugin output was not created: $outputPath"
}

Write-Output $outputPath
