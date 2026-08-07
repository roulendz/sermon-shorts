<#
.SYNOPSIS
    Migrate an event folder from the old flat layout to the new clean layout.

.DESCRIPTION
    Old layout (root-level clutter):
        <event>/transcriptions/        <event>/whisperx/
        <event>/clips/                 <event>/manus_response_*.json
        <event>/manus_tasks.json

    New layout:
        <event>/Audio TRANSCRIPT/transcriptions/
        <event>/Audio TRANSCRIPT/whisperx/
        <event>/Clips RAW/
        <event>/.data/manus_response_*.json
        <event>/.data/manus_tasks.json

    Idempotent: re-running after a partial/complete migration is a no-op.
    If a destination folder already exists, child entries are merged in
    (existing same-named files are left untouched and reported, never overwritten).

.PARAMETER EventRoot
    Path to the event folder. Defaults to the 2026-06-07 sermon.

.PARAMETER WhatIf
    Show planned moves without touching the disk.
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$EventRoot = 'U:\2026-06-07 Kā atnāk Dieva valstība'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $EventRoot)) {
    throw "Event root not found: $EventRoot"
}
Write-Host "Event root: $EventRoot" -ForegroundColor Cyan

# Move a single file into a destination directory (created on demand).
function Move-FileInto {
    param([string]$SourceFile, [string]$DestinationDirectory)

    if (-not (Test-Path -LiteralPath $SourceFile)) { return }   # already moved / never existed

    if (-not (Test-Path -LiteralPath $DestinationDirectory)) {
        if ($PSCmdlet.ShouldProcess($DestinationDirectory, 'Create directory')) {
            New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
        }
    }

    $fileName = Split-Path -Leaf $SourceFile
    $target = Join-Path $DestinationDirectory $fileName
    if (Test-Path -LiteralPath $target) {
        Write-Host "  SKIP (exists at dest): $fileName" -ForegroundColor Yellow
        return
    }
    if ($PSCmdlet.ShouldProcess($SourceFile, "Move -> $DestinationDirectory")) {
        Move-Item -LiteralPath $SourceFile -Destination $DestinationDirectory
        Write-Host "  MOVED: $fileName -> $DestinationDirectory" -ForegroundColor Green
    }
}

# Move a whole folder to a new path. If dest exists, merge children instead.
function Move-FolderTo {
    param([string]$SourceDirectory, [string]$DestinationDirectory)

    if (-not (Test-Path -LiteralPath $SourceDirectory)) {
        Write-Host "  SKIP (no source): $SourceDirectory" -ForegroundColor DarkGray
        return
    }

    if (-not (Test-Path -LiteralPath $DestinationDirectory)) {
        # Simple case: dest does not exist -> move/rename whole folder.
        $parent = Split-Path -Parent $DestinationDirectory
        if (-not (Test-Path -LiteralPath $parent)) {
            if ($PSCmdlet.ShouldProcess($parent, 'Create directory')) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
        }
        if ($PSCmdlet.ShouldProcess($SourceDirectory, "Move -> $DestinationDirectory")) {
            Move-Item -LiteralPath $SourceDirectory -Destination $DestinationDirectory
            Write-Host "  MOVED FOLDER: $SourceDirectory -> $DestinationDirectory" -ForegroundColor Green
        }
        return
    }

    # Merge case: dest exists -> move each child, then drop empty source.
    Write-Host "  MERGE into existing: $DestinationDirectory" -ForegroundColor Yellow
    foreach ($child in Get-ChildItem -LiteralPath $SourceDirectory -Force) {
        $target = Join-Path $DestinationDirectory $child.Name
        if (Test-Path -LiteralPath $target) {
            Write-Host "    SKIP (exists): $($child.Name)" -ForegroundColor Yellow
            continue
        }
        if ($PSCmdlet.ShouldProcess($child.FullName, "Move -> $DestinationDirectory")) {
            Move-Item -LiteralPath $child.FullName -Destination $DestinationDirectory
            Write-Host "    MOVED: $($child.Name)" -ForegroundColor Green
        }
    }
    # Remove source only if it ended up empty.
    if ((Get-ChildItem -LiteralPath $SourceDirectory -Force | Measure-Object).Count -eq 0) {
        if ($PSCmdlet.ShouldProcess($SourceDirectory, 'Remove empty source folder')) {
            Remove-Item -LiteralPath $SourceDirectory -Force
            Write-Host "  REMOVED empty: $SourceDirectory" -ForegroundColor DarkGreen
        }
    } else {
        Write-Host "  KEPT (not empty): $SourceDirectory" -ForegroundColor Yellow
    }
}

$audioTranscript = Join-Path $EventRoot 'Audio TRANSCRIPT'
$dataDirectory   = Join-Path $EventRoot '.data'

Write-Host "`n[1/4] transcriptions -> Audio TRANSCRIPT\transcriptions" -ForegroundColor Cyan
Move-FolderTo (Join-Path $EventRoot 'transcriptions') (Join-Path $audioTranscript 'transcriptions')

Write-Host "`n[2/4] whisperx -> Audio TRANSCRIPT\whisperx" -ForegroundColor Cyan
Move-FolderTo (Join-Path $EventRoot 'whisperx') (Join-Path $audioTranscript 'whisperx')

Write-Host "`n[3/4] clips -> Clips RAW" -ForegroundColor Cyan
Move-FolderTo (Join-Path $EventRoot 'clips') (Join-Path $EventRoot 'Clips RAW')

Write-Host "`n[4/4] manus_*.json -> .data" -ForegroundColor Cyan
foreach ($manusFile in Get-ChildItem -LiteralPath $EventRoot -Filter 'manus_response_*.json' -File -ErrorAction SilentlyContinue) {
    Move-FileInto $manusFile.FullName $dataDirectory
}
Move-FileInto (Join-Path $EventRoot 'manus_tasks.json') $dataDirectory

Write-Host "`nMigration complete." -ForegroundColor Cyan
