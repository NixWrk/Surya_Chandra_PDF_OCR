# Reusable TXT Artifacts (2026-03-30)

Purpose: avoid unnecessary OCR reruns and reuse latest existing text outputs.

## Primary Folder

1. `artifacts/ocr_obs_gost_oldbook_20260327_165121/_compare_txt`

## Files

1. `ГОСТ с плохим качеством скана__chandra.txt`
2. `ГОСТ с плохим качеством скана__olmocr.txt`
3. `ГОСТ с плохим качеством скана__surya.txt`
4. `Старая книга с частично рукописным текстом__chandra.txt`
5. `Старая книга с частично рукописным текстом__olmocr.txt`
6. `Старая книга с частично рукописным текстом__surya.txt`

## Validation Command

```powershell
Get-ChildItem "D:\Git_Code\img_2_pdf\artifacts\ocr_obs_gost_oldbook_20260327_165121\_compare_txt\*.txt" |
  Select-Object Name,@{n='chars';e={(Get-Content $_.FullName -Raw).Length}} |
  Sort-Object Name
```

Use these files as first-choice inputs for searchable-PDF layer building.
