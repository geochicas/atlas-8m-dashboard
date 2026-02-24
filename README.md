# Atlas 8M Dashboard (Geochicas)

Carpeta lista para GitHub Pages (repo `geochicas/atlas-8m-dashboard`).

## Archivos
- `index.html`: dashboard (Plotly) con filtros por Año / Tema / País.
- `Base_2019_2025.csv`: fuente (separador `;`).
- `build_dashboard.py`: genera `geochicas_8m_dashboard_data.json` desde el CSV.
- `geochicas_8m_dashboard_data.json`: salida consumida por el dashboard.
- `.nojekyll`: evita problemas de GitHub Pages con archivos que empiezan con `_`.

## Build (sin OCR)
Genera temas desde columnas existentes (si las hubiera) + `temas_ocr/themes` (si trae algo).

```bash
python build_dashboard.py
```

## Build (con OCR desde imagen_url)
Descarga imágenes y corre OCR para inferir temas. Es más lento, pero permite poblar `themes` si tu CSV no trae texto.

```bash
python build_dashboard.py --do-ocr
```

Tips:
- El script guarda cache en `ocr_cache.json` para no re-procesar imágenes.
- Si el repo corre en GitHub Actions, instalá tesseract y dependencias.

## Deploy
Subí estos archivos al branch/carpeta que GitHub Pages use en el repo. El dashboard carga el JSON con `fetch("./geochicas_8m_dashboard_data.json")`.
