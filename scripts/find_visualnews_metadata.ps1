param(
  [string[]]$Roots = @("E:\OOC_Datasets\VisualNews", "D:\MY_PROJECT\OOC\datasets"),
  [int]$Top = 80
)

Write-Host "Searching VisualNews metadata pickle files..."
foreach ($root in $Roots) {
  if (-not (Test-Path $root)) {
    Write-Host "Skip missing root: $root"
    continue
  }
  Write-Host "Root: $root"
  Get-ChildItem $root -Recurse -File -Include *.p,*.pkl,*.pickle,*.gz -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "processed|article|metadata|visual|news" } |
    Sort-Object Length -Descending |
    Select-Object -First $Top FullName, Length, LastWriteTime |
    Format-Table -AutoSize
}

Write-Host "If you see processed_*.p files, use their parent directory as -VisualNewsMetadataDir."
