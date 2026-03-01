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
  (tesseract debe estar instalado en el sistema)

Temas soportados (9):
  cuidados_y_trabajo, violencias, antirracismo, lgbtiq,
  feminicidios, derechos_digitales,
  transfeminismos, indigenismo, defensa_territorial
"""

import argparse
import json
import re
from collections import Counter
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


INPUT_CSV  = "Base_2019_2025.csv"
OUTPUT_JSON = "geochicas_8m_dashboard_data.json"
OCR_CACHE  = "ocr_cache.json"

LOCALE_ES = Locale.parse("es")

PALETTE = ["#381737", "#7B2355", "#2B525D", "#4A88B7", "#F2CC69"]

# =============================================================================
# THEME_RULES — 9 temas (multietiqueta, orden estable para el dashboard)
# Cada tema es una lista de regex que se evalúan en OR sobre el texto completo
# en minúsculas. El primer match dentro de un tema ya lo activa.
#
# Criterios de diseño:
# - Específicos: evitar activar temas por palabras genéricas sueltas
# - Multilingüe: español, inglés, portugués, francés, italiano, alemán
# - Sin solapamiento innecesario entre temas (ver notas inline)
# =============================================================================
THEME_RULES = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "cuidados_y_trabajo": [
        r"\bcuidad",
        r"\bcare\b",
        r"\btrabaj",
        r"\blabor\b",
        r"trabajo dom[eé]stico",
        r"econom[íi]a del cuidado",
        r"cuidados comunitarios",
        r"reproducci[oó]n social",
    ],

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "violencias": [
        r"\bviolenc",
        r"\bacoso",
        r"\babus",
        r"\bharass",
        r"violencia machista",
        r"violencia patriarcal",
        r"agresi[oó]n",
        r"violador",
        r"abusador",
    ],

    # ── 3 ─────────────────────────────────────────────────────────────────────
    # Nota: 'indigen' y 'pueblos originarios' se movieron a 'indigenismo'
    # para evitar solapamiento. Aquí quedan racismo estructural, afrofeminismo,
    # y descolonización sin enfoque territorial/comunitario específico.
    "antirracismo": [
        r"\bracis",
        r"\bantirrac",
        r"\bafro",
        r"\bcolonial",
        r"\bdecolonial",
        r"\bdecoloniz",
    ],

    # ── 4 ─────────────────────────────────────────────────────────────────────
    # Nota: 'disidenc' se movió a transfeminismos para mayor precisión.
    # lgbtiq cubre identidades sexuales/genéricas sin el marco feminista trans.
    "lgbtiq": [
        r"\blgbt",
        r"\blgbti",
        r"\blgbtiq",
        r"\blgbtq",
        r"\bqueer\b",
        r"\blesbian",
        r"\bgay\b",
        r"no binari",
        r"no-binari",
    ],

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "feminicidios": [
        r"\bfeminicid",
        r"\bfemicid",
        r"\bfemicide",
        r"ni una menos",
        r"vivas nos queremos",
    ],

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "derechos_digitales": [
        r"\bdigital",
        r"\btecnolog",
        r"\binternet",
        r"\bdatos\b",
        r"\bprivacy\b",
        r"soberan[íi]a tecnol[oó]gica",
        r"\bvigilancia\b",
        r"\bcensura\b",
        r"\balgoritm",
    ],

    # ── 7 (NUEVO) ──────────────────────────────────────────────────────────────
    # Feminismos que centran las experiencias trans, travesti y no binarias.
    # Incluye 'disidencia sexual' que antes estaba sin hogar.
    "transfeminismos": [
        r"transfeminis",           # transfeminismo/a/s (es/pt/it/de)
        r"transf[eé]minis",        # transféminis (fr)
        r"feminismo trans\b",
        r"trans feminist",         # en
        r"trans incluy",           # trans incluyente (es)
        r"trans inclus",           # trans inclusive (en) / trans inclusif (fr)
        r"transincluy",
        r"disidencia sexual",
        r"disidencias sexuales",
        r"feminismo disidente",
        r"perspectiva trans\b",
        r"mujeres trans\b",
        r"\btravestis\b",
        r"gender nonconform",      # en
        r"femminismo trans",       # it
        r"trans-feminis",          # de
    ],

    # ── 8 (NUEVO) ──────────────────────────────────────────────────────────────
    # Derechos y perspectivas feministas de pueblos indígenas y originarios.
    # Usa patrones específicos para no activarse solo por 'indígena' genérico.
    "indigenismo": [
        r"mujeres ind[íi]genas",
        r"\bind[íi]genas\b",       # activado por contexto del texto completo
        r"feminismo ind[íi]gena",
        r"feminismo comunitario",
        r"comunidades originarias",
        r"pueblos originarios",
        r"abya yala",
        r"sumak kawsay",
        r"buen vivir",
        r"territorio ancestral",
        r"pueblo mapuche",
        r"pueblo quechua",
        r"pueblo aymara",
        r"indigenous women",       # en
        r"indigenous feminist",    # en
        r"native women",           # en
        r"first nations women",    # en (Canadá)
        r"aboriginal women",       # en (Australia)
        r"mulheres ind[íi]genas",  # pt
        r"povos origin[áa]rios",   # pt
    ],

    # ── 9 (NUEVO) ──────────────────────────────────────────────────────────────
    # Mujeres y comunidades que defienden tierra, agua, territorio y naturaleza.
    # Incluye ecofeminismo como marco teórico-político.
    "defensa_territorial": [
        r"defensa territorial",
        r"defensa del territorio",
        r"defensa de la tierra",
        r"\bdefensoras\b",
        r"defensora territorial",
        r"extractivis",
        r"megaproyecto",
        r"miner[íi]a y mujeres",
        r"agua y mujeres",
        r"soberan[íi]a alimentaria",
        r"ecofeminis",
        r"ecolog[íi]a feminista",
        r"land defender",          # en
        r"territory defender",     # en
        r"environmental feminist", # en
        r"women land rights",      # en
        r"water rights women",     # en
        r"defensoras territoriais",# pt
        r"defesa do territ[oó]rio",# pt
    ],
}

# Orden canónico para el dashboard (determina orden en selects y leyendas)
THEME_ORDER = [
    "violencias",
    "feminicidios",
    "cuidados_y_trabajo",
    "lgbtiq",
    "transfeminismos",
    "antirracismo",
    "indigenismo",
    "defensa_territorial",
    "derechos_digitales",
]

# Columnas del CSV de entrada
COL_ORG         = "Organización"
COL_CITY        = "ciudad"
COL_YEAR        = "anio"
COL_COUNTRY_DASH = "pais_dash"
COL_COUNTRY_ES  = "pais_es"
COL_COUNTRY_RAW = "pais"
COL_COUNTRY_ISO2 = "pais_iso2"
COL_LAT         = "lat"
COL_LON         = "long"
COL_IMAGE       = "imagen_url"
COL_CALL        = "convocatoria"
COL_THEMES_OCR  = "temas_ocr/themes"


# =============================================================================
# Helpers de normalización geográfica
# =============================================================================

def country_to_es(name_or_code: str):
    if not name_or_code:
        return None
    raw = str(name_or_code).strip()
    if not raw:
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


def pretty_country(name: str) -> str:
    if not name:
        return name
    s = str(name).strip()
    if not s:
        return s
    low = s.lower()
    if low.startswith("sin "):
        return s[0].upper() + s[1:]
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    small = {"de", "del", "la", "las", "los", "y", "e", "da", "do", "das", "dos"}
    parts = s.split(" ")
    out = []
    for i, w in enumerate(parts):
        wl = w.lower()
        if i > 0 and wl in small:
            out.append(wl)
        else:
            out.append(wl[:1].upper() + wl[1:])
    return " ".join(out)


# =============================================================================
# Helpers numéricos
# =============================================================================

def to_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", ".")
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def normalize_year(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return int(m.group(1)) if m else None


# =============================================================================
# Detección de temas
# =============================================================================

def infer_themes(text: str) -> list[str]:
    """
    Aplica THEME_RULES sobre el texto (en minúsculas) y devuelve
    lista de tema-IDs ordenada según THEME_ORDER.
    """
    if not text:
        return []
    t = text.lower()
    found = []
    for theme, pats in THEME_RULES.items():
        for pat in pats:
            if re.search(pat, t, flags=re.IGNORECASE):
                found.append(theme)
                break
    # ordenar según THEME_ORDER; los que no estén van al final
    order_map = {t: i for i, t in enumerate(THEME_ORDER)}
    return sorted(set(found), key=lambda x: order_map.get(x, 99))


def parse_theme_cell(cell: str) -> list[str]:
    """
    Acepta múltiples formatos de celda de temas:
      - JSON list:  '["violencias","antirracismo"]'
      - Separado:   'violencias;antirracismo' o 'Violencias, Antirracismo'
      - Nombre legible: 'Cuidados y trabajo' → 'cuidados_y_trabajo'
    Devuelve lista de tema-IDs válidos (presentes en THEME_RULES).
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []

    # JSON list
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                parts = [str(x).strip() for x in arr if str(x).strip()]
                s = ";".join(parts)
        except Exception:
            pass

    parts = re.split(r"[;,|/]+", s)
    allowed = set(THEME_RULES.keys())
    norm = []
    for p in parts:
        p = p.strip().lower()
        if not p:
            continue
        # Si ya viene en snake_case válido, no transformar (evita doble-reemplazo)
        if p not in allowed:
            p = p.replace("lgbtq+", "lgbtiq").replace("lgbtq", "lgbtiq")
            p = p.replace("derechos digitales",  "derechos_digitales")
            p = p.replace("cuidados y trabajo",  "cuidados_y_trabajo")
            p = p.replace("defensa territorial", "defensa_territorial")
            # femicidio* → feminicidios (regex para cubrir singular/plural)
            p = re.sub(r"^f[ae]minicidi[ao]s?$", "feminicidios", p)
            p = re.sub(r"^femicidi[ao]s?$",       "feminicidios", p)
            p = p.replace("antirracista",  "antirracismo")
            p = p.replace("anti-racismo",  "antirracismo")
            # violencia(s) → violencias (solo formas simples, no sobreescribir frases)
            p = re.sub(r"^violencias? de g[eé]nero$", "violencias", p)
            p = re.sub(r"^violencia$",                "violencias", p)
        if p in allowed:
            norm.append(p)
    order_map = {t: i for i, t in enumerate(THEME_ORDER)}
    return sorted(set(norm), key=lambda x: order_map.get(x, 99))


# =============================================================================
# OCR
# =============================================================================

def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: dict):
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def ocr_image_url(url: str, lang: str = "spa+eng") -> str:
    if not url or not url.startswith("http"):
        return ""
    if requests is None or Image is None or pytesseract is None:
        return ""
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        from io import BytesIO
        img = Image.open(BytesIO(r.content)).convert("RGB")
        w, h = img.size
        if max(w, h) < 1200:
            img = img.resize((w * 2, h * 2))
        txt = pytesseract.image_to_string(img, lang=lang)
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return ""


# =============================================================================
# Build de puntos
# =============================================================================

def build_points(df: pd.DataFrame, do_ocr: bool, ocr_lang: str):
    cache_path = Path(OCR_CACHE)
    cache = load_cache(cache_path)
    points = []
    ocr_used = 0

    for _, row in df.iterrows():
        anio = normalize_year(row.get(COL_YEAR, ""))
        lat  = to_float(row.get(COL_LAT, ""))
        lon  = to_float(row.get(COL_LON, ""))
        if anio is None:
            continue

        # País normalizado
        pais_es      = str(row.get(COL_COUNTRY_ES,   "")).strip()
        pais_dash_raw = str(row.get(COL_COUNTRY_DASH, "")).strip()
        pais_raw     = str(row.get(COL_COUNTRY_RAW,  "")).strip()
        pais_iso2    = str(row.get(COL_COUNTRY_ISO2,  "")).strip()
        cand = (
            pais_es
            or country_to_es(pais_dash_raw)
            or country_to_es(pais_iso2)
            or country_to_es(pais_raw)
            or pais_dash_raw
            or pais_raw
        )
        pais_dash = pretty_country(cand) if cand else "Sin país"

        ciudad       = str(row.get(COL_CITY,  "")).strip()
        organizacion = str(row.get(COL_ORG,   "")).strip()
        conv_url     = str(row.get(COL_CALL,  "")).strip()
        imagen_url   = str(row.get(COL_IMAGE, "")).strip()

        # ── Temas ────────────────────────────────────────────────────────────
        # 1) columnas manuales (themes / temas / tema / tags)
        manual_themes: list[str] = []
        for col in ["themes", "temas", "tema", "tags"]:
            if col in df.columns:
                manual_themes = parse_theme_cell(row.get(col, ""))
                if manual_themes:
                    break

        # 2) columna pre-calculada 'temas_ocr/themes'
        pre_themes = parse_theme_cell(row.get(COL_THEMES_OCR, ""))

        themes = sorted(set(manual_themes + pre_themes),
                        key=lambda x: {t: i for i, t in enumerate(THEME_ORDER)}.get(x, 99))

        # 3) inferencia automática desde texto disponible (siempre activa)
        #    Enriquece incluso si ya hay temas manuales
        text_for_inference = " ".join(filter(None, [
            str(row.get("convocatoria_titulo", "")),
            str(row.get("descripcion", "")),
            ciudad,
            organizacion,
        ]))
        inferred = infer_themes(text_for_inference)
        themes = sorted(
            set(themes + inferred),
            key=lambda x: {t: i for i, t in enumerate(THEME_ORDER)}.get(x, 99),
        )

        # 4) OCR (opcional) si sigue vacío
        ocr_text = ""
        if do_ocr and not themes and imagen_url.startswith("http"):
            if imagen_url in cache:
                ocr_text = cache[imagen_url]
            else:
                ocr_text = ocr_image_url(imagen_url, lang=ocr_lang)
                cache[imagen_url] = ocr_text
                ocr_used += 1
            ocr_themes = infer_themes(
                " ".join([ocr_text, organizacion, ciudad])
            )
            themes = sorted(
                set(themes + ocr_themes),
                key=lambda x: {t: i for i, t in enumerate(THEME_ORDER)}.get(x, 99),
            )

        points.append({
            "anio":            anio,
            "lat":             lat,
            "lon":             lon,
            "pais_dash":       pais_dash,
            "ciudad":          ciudad,
            "organizacion":    organizacion,
            "convocatoria_url": conv_url if conv_url.startswith("http") else "",
            "imagen_url":      imagen_url if imagen_url.startswith("http") else "",
            "themes":          themes,
            "ocr_text":        ocr_text[:800] if ocr_text else "",
        })

    if do_ocr and ocr_used:
        save_cache(cache_path, cache)

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "palette":  PALETTE,
        "ocr": {
            "enabled":               do_ocr,
            "lang":                  ocr_lang,
            "new_images_processed":  ocr_used,
        },
    }
    return points, meta


# =============================================================================
# Cómputo de estadísticas
# =============================================================================

def compute_counts_year(points: list[dict]) -> dict:
    c = Counter(p["anio"] for p in points if p.get("anio") is not None)
    return {str(k): int(v) for k, v in sorted(c.items())}


def compute_hallazgos(points: list[dict]):
    theme_counts: Counter = Counter()
    for p in points:
        ts = p.get("themes") or []
        if ts:
            for t in ts:
                theme_counts[t] += 1
        else:
            theme_counts["sin_tema_detectado"] += 1

    top_themes = [
        {"theme": k, "acciones": int(v)}
        for k, v in theme_counts.most_common(10)
    ]

    years = sorted({p["anio"] for p in points if p.get("anio") is not None})
    base_year   = years[0]  if years else None
    latest_year = years[-1] if years else None

    def top_countries_for_year(y):
        cc = Counter(
            p.get("pais_dash") or "Sin país"
            for p in points if p.get("anio") == y
        )
        return [{"pais_dash": k, "acciones": int(v)} for k, v in cc.most_common(10)]

    top_countries_latest = top_countries_for_year(latest_year) if latest_year else []

    top_growth, top_drop = [], []
    if base_year and latest_year and base_year != latest_year:
        c0 = Counter(
            p.get("pais_dash") or "Sin país"
            for p in points if p.get("anio") == base_year
        )
        c1 = Counter(
            p.get("pais_dash") or "Sin país"
            for p in points if p.get("anio") == latest_year
        )
        deltas = [
            (k, int(c1.get(k, 0)) - int(c0.get(k, 0)))
            for k in set(c0) | set(c1)
        ]
        top_growth = [
            {"pais_dash": k, "delta": d}
            for k, d in sorted(deltas, key=lambda x: x[1], reverse=True)[:10]
        ]
        top_drop = [
            {"pais_dash": k, "delta": d}
            for k, d in sorted(deltas, key=lambda x: x[1])[:10]
        ]

    hallazgos = {
        "top_themes":           top_themes,
        "top_countries_latest": top_countries_latest,
        "top_growth_since_base": top_growth,
        "top_drop_since_base":   top_drop,
    }
    return hallazgos, base_year, latest_year


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Genera geochicas_8m_dashboard_data.json desde Base_2019_2025.csv"
    )
    ap.add_argument("--input",    default=INPUT_CSV)
    ap.add_argument("--output",   default=OUTPUT_JSON)
    ap.add_argument("--do-ocr",   action="store_true",
                    help="Descarga imagen_url y corre OCR para inferir temas")
    ap.add_argument("--ocr-lang", default="spa+eng",
                    help="Idiomas tesseract, ej: 'spa+eng+fra'")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(
            f"No encuentro {args.input}. Ponelo en el mismo directorio del script."
        )

    df = pd.read_csv(in_path, sep=";", dtype=str, keep_default_na=False)

    needed = [COL_YEAR, COL_COUNTRY_DASH, COL_LAT, COL_LON]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas mínimas en el CSV: {missing}")

    points, meta = build_points(df, do_ocr=args.do_ocr, ocr_lang=args.ocr_lang)

    # Filtrar puntos con coordenadas válidas
    points = [
        p for p in points
        if isinstance(p.get("lat"), float) and isinstance(p.get("lon"), float)
        and -90 <= p["lat"] <= 90 and -180 <= p["lon"] <= 180
    ]

    counts_year = compute_counts_year(points)
    hallazgos, base_year, latest_year = compute_hallazgos(points)

    years     = sorted({str(p["anio"]) for p in points if p.get("anio") is not None})
    countries = sorted({p.get("pais_dash") or "Sin país" for p in points})

    # Temas en orden canónico + sin_tema_detectado al final si aplica
    themes_present = {t for p in points for t in (p.get("themes") or [])}
    themes_ordered = [t for t in THEME_ORDER if t in themes_present]
    if any(not p.get("themes") for p in points):
        themes_ordered.append("sin_tema_detectado")

    out = {
        "generated_at": meta["generated_at"],
        "meta": {
            "latest_year": latest_year,
            "base_year":   base_year,
            "palette":     meta["palette"],
            "ocr":         meta["ocr"],
            "temas_disponibles": THEME_ORDER,
            "nota_temas": (
                "Temas detectados por reglas multilingüe (9 temas, multietiqueta). "
                "Si una acción no matchea ningún keyword, cae en 'sin_tema_detectado'."
            ),
        },
        "domain": {
            "years":     years,
            "countries": countries,
            "themes":    themes_ordered,
        },
        "counts_year": counts_year,
        "hallazgos":   hallazgos,
        "points":      points,
    }

    Path(args.output).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"✅ Generado {args.output} con {len(points)} puntos.")
    print(f"   Temas disponibles: {', '.join(themes_ordered)}")
    if args.do_ocr:
        print(f"   OCR: {meta['ocr']}")


if __name__ == "__main__":
    main()
