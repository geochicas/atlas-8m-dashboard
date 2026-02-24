#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build del dataset para el Atlas 8M Dashboard (GitHub Pages).

Qué hace:
- Lee Base_2019_2025.csv (separado por ';')
- Estandariza columnas mínimas que el dashboard necesita
- Genera 'themes' como lista (multietiqueta) a partir de:
  a) columna existente de temas (si la hay)
  b) columna 'temas_ocr/themes' (si trae algo)
  c) OCR sobre 'imagen_url' (opcional; requiere internet)
- Exporta geochicas_8m_dashboard_data.json con schema compatible con index.html

Uso:
  python build_dashboard.py
  python build_dashboard.py --do-ocr  (más lento; descarga imágenes y hace OCR)

Requisitos OCR:
  pip install requests pillow pytesseract
  (tesseract ya viene en muchos entornos; en GitHub Actions conviene instalarlo)

Notas:
- Facebook/Instagram a veces bloquean scraping. Por eso el OCR se centra en imagen_url.
- Este builder crea cache de OCR para no re-procesar imágenes: ocr_cache.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pycountry
from babel import Locale

# OCR (opcional)
try:
    import requests
    from PIL import Image
    import pytesseract
except Exception:
    requests = None
    Image = None
    pytesseract = None


INPUT_CSV = "Base_2019_2025.csv"
OUTPUT_JSON = "geochicas_8m_dashboard_data.json"
OCR_CACHE = "ocr_cache.json"

# Paleta (de tu imagen)
LOCALE_ES = Locale.parse('es')

PALETTE = ["#381737", "#7B2355", "#2B525D", "#4A88B7", "#F2CC69"]

THEME_RULES = {
    # Claves = IDs usados en el dashboard
    "cuidados_y_trabajo": [
        r"\bcuidad", r"\bcare\b", r"\btrabaj", r"\blabor\b",
        r"trabajo domestico", r"trabajo doméstico",
        r"economia del cuidado", r"economía del cuidado",
        r"cuidados comunitarios", r"reproducci[oó]n social",
    ],
    "violencias": [
        r"\bviolenc", r"\bacoso", r"\babus", r"\bharass",
        r"violencia machista", r"violencia patriarcal",
        r"agresi[oó]n", r"violador", r"abusador",
    ],
    "antirracismo": [
        r"\bracis", r"\bantirrac", r"\bindigen", r"\bafro",
        r"abya yala", r"pueblos originarios",
        r"colonial", r"decolonial", r"decoloniz",
    ],
    "lgbtiq": [
        r"\blgbt", r"\blgbti", r"\blgbtiq", r"\blgbtq",
        r"\btrans\b", r"\bqueer\b", r"\blesbian", r"\bgay\b",
        r"disidenc", r"no binari", r"no-binari",
    ],
    "feminicidios": [
        r"\bfeminicid", r"\bfemicid", r"\bfemicide",
        r"ni una menos", r"vivas nos queremos",
    ],
    "derechos_digitales": [
        r"\bdigital", r"\btecnolog", r"\binternet",
        r"\bdatos\b", r"\bprivacy\b",
        r"soberania tecnologica", r"soberanía tecnológica",
        r"vigilancia", r"censura", r"algoritm",
    ],
}

# Algunas columnas típicas
COL_ORG = "Organización"
COL_CITY = "ciudad"
COL_YEAR = "anio"
COL_COUNTRY_DASH = "pais_dash"
COL_COUNTRY_ES = "pais_es"
COL_COUNTRY_RAW = "pais"
COL_COUNTRY_ISO2 = "pais_iso2"
COL_LAT = "lat"
COL_LON = "long"
COL_IMAGE = "imagen_url"
COL_CALL = "convocatoria"
COL_THEMES_OCR = "temas_ocr/themes"



def country_to_es(name_or_code: str):
    if not name_or_code:
        return None
    raw = str(name_or_code).strip()
    if raw == "":
        return None

    code = raw.upper()
    obj = None
    if len(code) == 2:
        obj = pycountry.countries.get(alpha_2=code)
    elif len(code) == 3:
        obj = pycountry.countries.get(alpha_3=code)

    if obj is None:
        try:
            obj = pycountry.countries.search_fuzzy(raw)[0]
        except Exception:
            obj = None

    if obj is None:
        return None

    try:
        return LOCALE_ES.territories.get(obj.alpha_2) or obj.name
    except Exception:
        return obj.name


def to_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", ".")
        if s == "":
            return None
        v = float(s)
        return v
    except Exception:
        return None


def normalize_year(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return int(m.group(1)) if m else None


def infer_themes(text: str) -> list[str]:
    if not text:
        return []
    t = text.lower()
    found = []
    for theme, pats in THEME_RULES.items():
        for pat in pats:
            if re.search(pat, t, flags=re.IGNORECASE):
                found.append(theme)
                break
    return sorted(set(found))


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: dict):
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def ocr_image_url(url: str, lang: str = "spa+eng") -> str:
    """
    Descarga una imagen y corre OCR.
    Devuelve texto (string). Si falla, devuelve "".
    """
    if not url or not url.startswith("http"):
        return ""
    if requests is None or Image is None or pytesseract is None:
        return ""

    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        # PIL puede abrir desde bytes
        from io import BytesIO
        img = Image.open(BytesIO(r.content)).convert("RGB")

        # Prepro simple: agrandar un poco ayuda
        w, h = img.size
        scale = 2 if max(w, h) < 1200 else 1
        if scale != 1:
            img = img.resize((w * scale, h * scale))

        txt = pytesseract.image_to_string(img, lang=lang)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt
    except Exception:
        return ""


def parse_theme_cell(cell: str) -> list[str]:
    """
    Acepta 'violencias;antirracismo' o '["violencias","antirracismo"]' o similar.
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if s == "":
        return []
    # JSON list
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # delimiters
    parts = re.split(r"[;,|/]+", s)
    parts = [p.strip().lower() for p in parts if p.strip()]
    # normalizaciones mínimas
    norm = []
    for p in parts:
        p = p.replace("lgbtq+", "lgbtiq").replace("lgbtq", "lgbtiq")
        p = p.replace("derechos digitales", "derechos_digitales")
        p = p.replace("cuidados y trabajo", "cuidados_y_trabajo")
        p = p.replace("femicidios", "feminicidios")
        p = p.replace("femicidio", "feminicidios")
        p = p.replace("feminicidio", "feminicidios")
        p = p.replace("antirracista", "antirracismo")
        p = p.replace("anti-racismo", "antirracismo")
        p = p.replace("violencia", "violencias")
        p = p.replace("violencias de género", "violencias")
        p = p.replace("violencias de genero", "violencias")
        norm.append(p)
    # filtrar a los temas permitidos
    allowed = set(THEME_RULES.keys())
    return sorted(set([p for p in norm if p in allowed]))


def build_points(df: pd.DataFrame, do_ocr: bool, ocr_lang: str) -> tuple[list[dict], dict]:
    cache_path = Path(OCR_CACHE)
    cache = load_cache(cache_path)

    points = []
    ocr_used = 0

    for _, row in df.iterrows():
        anio = normalize_year(row.get(COL_YEAR, ""))
        lat = to_float(row.get(COL_LAT, ""))
        lon = to_float(row.get(COL_LON, ""))
        if anio is None:
            continue

        pais_es = str(row.get(COL_COUNTRY_ES, "")).strip()
        pais_dash_raw = str(row.get(COL_COUNTRY_DASH, "")).strip()
        pais_raw = str(row.get(COL_COUNTRY_RAW, "")).strip()
        pais_iso2 = str(row.get(COL_COUNTRY_ISO2, "")).strip()
        pais_dash = pais_es or pais_dash_raw or country_to_es(pais_iso2) or country_to_es(pais_raw) or "Sin país"
        ciudad = str(row.get(COL_CITY, "")).strip()
        organizacion = str(row.get(COL_ORG, "")).strip()
        convocatoria_url = str(row.get(COL_CALL, "")).strip()
        imagen_url = str(row.get(COL_IMAGE, "")).strip()

        # 1) temas manuales (si existieran en el CSV en alguna columna)
        manual_themes = []
        for col in ["themes", "temas", "tema", "tags"]:
            if col in df.columns:
                manual_themes = parse_theme_cell(row.get(col, ""))
                if manual_themes:
                    break

        # 2) temas pre-existentes en 'temas_ocr/themes'
        ocr_cell_themes = parse_theme_cell(row.get(COL_THEMES_OCR, ""))

        themes = sorted(set(manual_themes + ocr_cell_themes))

        # 3) OCR (opcional) si sigue vacío
        ocr_text = ""
        if do_ocr and not themes and imagen_url.startswith("http"):
            if imagen_url in cache:
                ocr_text = cache.get(imagen_url, "")
            else:
                ocr_text = ocr_image_url(imagen_url, lang=ocr_lang)
                cache[imagen_url] = ocr_text
                ocr_used += 1
            # inferir desde OCR + org + ciudad (org/ciudad ayudan poco, pero suman)
            themes = infer_themes(" ".join([ocr_text, organizacion, ciudad]))

        point = {
            "anio": anio,
            "lat": lat,
            "lon": lon,
            "pais_dash": pais_dash,
            "ciudad": ciudad,
            "organizacion": organizacion,
            "convocatoria_url": convocatoria_url if convocatoria_url.startswith("http") else "",
            "imagen_url": imagen_url if imagen_url.startswith("http") else "",
            "themes": themes,
            # campo extra útil para depuración / mejoras
            "ocr_text": ocr_text[:800] if ocr_text else "",
        }
        points.append(point)

    # guardar cache si se usó OCR
    if do_ocr and ocr_used:
        save_cache(cache_path, cache)

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "palette": PALETTE,
        "ocr": {"enabled": do_ocr, "lang": ocr_lang, "new_images_processed": ocr_used},
    }
    return points, meta


def compute_counts_year(points: list[dict]) -> dict:
    c = Counter([p["anio"] for p in points if p.get("anio") is not None])
    return {str(k): int(v) for k, v in sorted(c.items(), key=lambda x: x[0])}


def compute_hallazgos(points: list[dict]) -> dict:
    # top themes global
    theme_counts = Counter()
    for p in points:
        ts = p.get("themes") or []
        if not ts:
            theme_counts["sin_tema_detectado"] += 1
        else:
            for t in ts:
                theme_counts[t] += 1
    top_themes = [{"theme": k, "acciones": int(v)} for k, v in theme_counts.most_common(10)]

    # top countries latest year
    years = sorted({p["anio"] for p in points if p.get("anio") is not None})
    base_year = years[0] if years else None
    latest_year = years[-1] if years else None

    def top_countries_for_year(y: int):
        cc = Counter([p.get("pais_dash") or "Sin país (pendiente)" for p in points if p.get("anio") == y])
        return [{"pais_dash": k, "acciones": int(v)} for k, v in cc.most_common(10)]

    top_countries_latest = top_countries_for_year(latest_year) if latest_year else []

    # growth since base
    if base_year and latest_year and base_year != latest_year:
        c0 = Counter([p.get("pais_dash") or "Sin país (pendiente)" for p in points if p.get("anio") == base_year])
        c1 = Counter([p.get("pais_dash") or "Sin país (pendiente)" for p in points if p.get("anio") == latest_year])
        deltas = []
        for k in set(c0.keys()) | set(c1.keys()):
            deltas.append((k, int(c1.get(k, 0)) - int(c0.get(k, 0))))
        deltas_sorted = sorted(deltas, key=lambda x: x[1], reverse=True)
        top_growth = [{"pais_dash": k, "delta": d} for k, d in deltas_sorted[:10]]
        top_drop = [{"pais_dash": k, "delta": d} for k, d in sorted(deltas, key=lambda x: x[1])[:10]]
    else:
        top_growth, top_drop = [], []

    return {
        "top_themes": top_themes,
        "top_countries_latest": top_countries_latest,
        "top_growth_since_base": top_growth,
        "top_drop_since_base": top_drop,
    }, base_year, latest_year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_CSV)
    ap.add_argument("--output", default=OUTPUT_JSON)
    ap.add_argument("--do-ocr", action="store_true", help="Descargar imagen_url y correr OCR para inferir temas")
    ap.add_argument("--ocr-lang", default="spa+eng", help="Idiomas tesseract, ej: 'spa+eng'")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"No encuentro {args.input}. Ponelo en el mismo directorio del script.")

    df = pd.read_csv(in_path, sep=";", dtype=str, keep_default_na=False)

    # columnas mínimas esperadas por el dashboard
    needed = [COL_YEAR, COL_COUNTRY_DASH, COL_LAT, COL_LON]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas mínimas en el CSV: {missing}")

    points, meta = build_points(df, do_ocr=args.do_ocr, ocr_lang=args.ocr_lang)
    # filtrar solo puntos con coords válidas
    points = [p for p in points if isinstance(p.get("lat"), (int, float)) and isinstance(p.get("lon"), (int, float))
              and p.get("lat") is not None and p.get("lon") is not None
              and -90 <= float(p["lat"]) <= 90 and -180 <= float(p["lon"]) <= 180]

    counts_year = compute_counts_year(points)
    hallazgos, base_year, latest_year = compute_hallazgos(points)

    # domain (para selects)
    years = sorted({str(p["anio"]) for p in points if p.get("anio") is not None})
    countries = sorted({p.get("pais_dash") or "Sin país (pendiente)" for p in points})
    themes = sorted({t for p in points for t in (p.get("themes") or [])})
    # mantener sin_tema_detectado como opción si existe
    if any((p.get("themes") or []) == [] for p in points):
        themes = themes + ["sin_tema_detectado"]

    out = {
        "generated_at": meta["generated_at"],
        "meta": {
            "latest_year": latest_year,
            "base_year": base_year,
            "palette": meta["palette"],
            "ocr": meta["ocr"],
            "nota_temas": (
                "Temas detectados por reglas (multietiqueta). "
                "Si una acción no matchea keywords, cae en 'sin_tema_detectado'."
            ),
        },
        "domain": {"years": years, "countries": countries, "themes": themes},
        "counts_year": counts_year,
        "hallazgos": hallazgos,
        "points": points,
    }

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: generado {args.output} con {len(points)} puntos.")
    if args.do_ocr:
        print(f"OCR: {meta['ocr']} (cache: {OCR_CACHE})")


if __name__ == "__main__":
    main()
