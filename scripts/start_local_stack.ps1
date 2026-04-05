# Phase 8: start API stack with Docker Compose (Windows PowerShell).
# Run from repository root:  .\scripts\start_local_stack.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
docker compose -f docker/docker-compose.yml up --build
