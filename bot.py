#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Personal Housing Finder - Zapopan (Unidad Deportiva El Briseño) + Telegram

Cambios importantes:
- Telegram manda SIEMPRE un mensaje por corrida (aunque no haya nuevos)
- Playwright mejorado: espera networkidle + scroll para lazy load
- Inmuebles24: selectores/heurísticas más flexibles (links relativos incluidos)
- Filtro por radio: por defecto incluye los que no se pueden geocodificar (para no quedar en 0)
- Métricas de geocoding: geocoded vs unknown

Instalar:
  pip install requests beautifulsoup4
  # opcional (si quieres):
  pip install lxml
  # navegador fallback:
  pip install playwright
  playwright install

Variables de entorno:
  TELEGRAM_BOT_TOKEN="..."
  TELEGRAM_CHAT_ID="..."

Ejecutar:
  python bot.py
"""

import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ----------------------------
# CONFIG
# ----------------------------
PRICE_MIN = 1_500_000
PRICE_MAX = 2_000_000

# Centro: Unidad Deportiva El Briseño (aprox)
HOME_LAT = 20.626058
HOME_LON = -103.439071

RADIUS_KM = 7.0

MAX_GEOCODE_PER_RUN = 40     # empieza con 20–40
SKIP_GEO_IF_LOCATION_TOO_GENERIC = True

# CLAVE: para que no te quede todo en 0, dejamos pasar anuncios sin geocoding por ahora
# Cuando ya tengas extracción de ubicación más fina, puedes regresarlo a False.
INCLUDE_UNKNOWN_GEO = True

DB_PATH = "housing.db"
GEO_CACHE_PATH = Path("geocache.json")

NOMINATIM_BASE = "https://nominatim.openstreetmap.org/search"

SEARCH_URLS = {
    "inmuebles24": "https://www.inmuebles24.com/casas-en-venta-en-zapopan-de-1500000-a-2000000-pesos.html",
    "mercadolibre": "https://inmuebles.mercadolibre.com.mx/casas/venta/jalisco/zapopan/_PriceRange_1500000MXN-8000000MXN",
    "vivanuncios": "https://www.vivanuncios.com.mx/s-casas-en-venta/zapopan/v1c1293l14828p1",
}

# BS4 parser (sin lxml por defecto)
USE_LXML_IF_AVAILABLE = False
BS4_PARSER = "lxml" if USE_LXML_IF_AVAILABLE else "html.parser"

USE_BROWSER_FALLBACK = {
    "inmuebles24": True,
    "mercadolibre": False,
    "vivanuncios": True,
}
BROWSER_HEADLESS = False

# Telegram (env vars)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Siempre manda mensaje por corrida
ALWAYS_SEND_TELEGRAM_SUMMARY = True
TELEGRAM_MAX_ITEMS = 8

# ----------------------------
# HTTP SESSION
# ----------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
})

# ----------------------------
# MODEL
# ----------------------------
@dataclass
class Listing:
    portal: str
    url: str
    title: str
    price_mxn: Optional[int]
    location: str  # texto (best effort)
    distance_km: Optional[float] = None  # se llena si se puede


# ----------------------------
# HELPERS
# ----------------------------
def normalize_price_to_int(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.upper().replace("\xa0", " ").strip()

    # Formato "1.95M"
    m = re.search(r"(\d+(?:\.\d+)?)\s*M", t)
    if m:
        try:
            return int(float(m.group(1)) * 1_000_000)
        except ValueError:
            return None

    digits = re.sub(r"[^\d]", "", t)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def in_price_range(price: Optional[int]) -> bool:
    return price is not None and PRICE_MIN <= price <= PRICE_MAX


def dedup_by_url(items: List[Listing]) -> List[Listing]:
    seen = set()
    out = []
    for it in items:
        u = (it.url or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(it)
    return out


def is_http_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


# ----------------------------
# PLAYWRIGHT FETCH
# ----------------------------
def fetch_html_browser(url: str) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright no está instalado. Instala con:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install\n"
        ) from e

    ua = SESSION.headers.get("User-Agent")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=BROWSER_HEADLESS)
        context = browser.new_context(user_agent=ua, locale="es-MX")
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # Espera a que carguen requests extra
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # Scroll para lazy-load
        for _ in range(5):
            page.mouse.wheel(0, 1500)
            page.wait_for_timeout(800)

        html = page.content()
        browser.close()
        return html


def fetch_html(url: str, portal: str) -> str:
    r = SESSION.get(url, timeout=30)
    if r.status_code == 403 and USE_BROWSER_FALLBACK.get(portal, False):
        print(f"[INFO] {portal}: 403 con requests, intentando fallback navegador (Playwright)...")
        return fetch_html_browser(url)
    r.raise_for_status()
    return r.text


# ----------------------------
# GEO + DISTANCE
# ----------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_geo_cache() -> Dict[str, Optional[Dict[str, float]]]:
    if GEO_CACHE_PATH.exists():
        try:
            return json.loads(GEO_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_geo_cache(cache: Dict[str, Optional[Dict[str, float]]]) -> None:
    GEO_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def geocode_osm(query: str, cache: Dict[str, Optional[Dict[str, float]]]) -> Optional[Tuple[float, float]]:
    q = (query or "").strip()
    if not q:
        return None

    if q in cache:
        v = cache[q]
        if not v:
            return None
        return float(v["lat"]), float(v["lon"])

    url = f"{NOMINATIM_BASE}?q={quote(q)}&format=json&limit=1"
    r = SESSION.get(url, timeout=30)
    if r.status_code != 200:
        cache[q] = None
        return None

    try:
        data = r.json()
    except Exception:
        cache[q] = None
        return None

    time.sleep(1.1)

    if not data:
        cache[q] = None
        return None

    try:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
    except Exception:
        cache[q] = None
        return None

    cache[q] = {"lat": lat, "lon": lon}
    return lat, lon


def compute_distance_km_for_listing(it: Listing, cache: Dict[str, Optional[Dict[str, float]]]) -> Optional[float]:
    loc = (it.location or "").strip()

    # Si la ubicación es demasiado genérica, no gastes geocoding (sería basura y lento)
    if SKIP_GEO_IF_LOCATION_TOO_GENERIC:
        generic_patterns = [
            r"^zapopan$",
            r"^zapopan,\s*jalisco$",
            r"^jalisco$",
            r"^guadalajara$",
        ]
        if not loc or any(re.match(p, loc.strip().lower()) for p in generic_patterns):
            return None

    # Enriquecer query
    query = loc
    if "ZAPOPAN" not in query.upper():
        query = f"{query}, Zapopan, Jalisco, México"

    coords = geocode_osm(query, cache)
    if coords is None:
        return None

    lat, lon = coords
    return haversine_km(HOME_LAT, HOME_LON, lat, lon)


def passes_radius_filter(it: Listing, cache: Dict[str, Optional[Dict[str, float]]]) -> bool:
    dist = compute_distance_km_for_listing(it, cache)
    it.distance_km = dist

    if dist is None:
        return INCLUDE_UNKNOWN_GEO

    return dist <= RADIUS_KM


# ----------------------------
# TELEGRAM
# ----------------------------
def telegram_send_message(text: str) -> None:
    if not TELEGRAM_ENABLED:
        print("[INFO] Telegram no configurado (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID).")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        r = SESSION.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            print(f"[WARN] Telegram error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[WARN] Telegram exception: {e}")


def fmt_listing_line(it: Listing) -> str:
    p = f"${it.price_mxn:,}" if isinstance(it.price_mxn, int) else "s/p"
    loc = it.location or "sin ubicación"
    d = f"{it.distance_km:.1f} km" if isinstance(it.distance_km, float) else "dist ?"
    return f"• [{it.portal}] {p} ({d})\n{loc}\n{it.url}"
  
def send_cheapest_summary(filtered_items: List[Listing]) -> None:
    if not TELEGRAM_ENABLED:
        return

    # Quita los que no tienen precio
    priced = [x for x in filtered_items if isinstance(x.price_mxn, int)]

    # Ordena por precio
    priced.sort(key=lambda x: x.price_mxn)

    cheapest = priced[:15]

    header = (
        "💸 Top 15 más baratas\n"
        f"📍 ≤ {RADIUS_KM} km | {PRICE_MIN:,}-{PRICE_MAX:,} MXN\n\n"
    )

    if not cheapest:
        telegram_send_message(header + "No se encontraron propiedades con precio válido.")
        return

    lines = []
    for it in cheapest:
        p = f"${it.price_mxn:,}"
        d = f"{it.distance_km:.1f} km" if isinstance(it.distance_km, float) else "dist ?"
        lines.append(
            f"{p} ({d})\n{it.url}"
        )

    telegram_send_message(header + "\n\n".join(lines))


def send_run_summary(
    portal_counts: Dict[str, int],
    after_radius_count: int,
    new_items: List[Listing],
    sample_items: List[Listing],
    geocoded_ok: int,
    geocoded_unknown: int,
) -> None:
    # Siempre mandamos summary (si está activado)
    if not (TELEGRAM_ENABLED and ALWAYS_SEND_TELEGRAM_SUMMARY):
        if new_items:
            # fallback mínimo si alguien desactiva ALWAYS_SEND...
            telegram_send_message(f"🏠 Nuevos: {len(new_items)}")
        return

    header = (
        "🏠 Corrida completada (El Briseño)\n"
        f"💰 {PRICE_MIN:,}–{PRICE_MAX:,} MXN | 📍 ≤ {RADIUS_KM} km\n"
        f"📦 Portales: " + ", ".join([f"{k}:{v}" for k, v in portal_counts.items()]) + "\n"
        f"🧭 Tras radio: {after_radius_count}\n"
        f"🗺️ Geocode ok:{geocoded_ok} | unknown:{geocoded_unknown}\n"
        f"🆕 Nuevos: {len(new_items)}\n"
    )

    # Si hay nuevos, listarlos; si no, mandar muestra del top actual
    lines: List[str] = []
    if new_items:
        for it in new_items[:TELEGRAM_MAX_ITEMS]:
            lines.append(fmt_listing_line(it))
        if len(new_items) > TELEGRAM_MAX_ITEMS:
            lines.append(f"(+{len(new_items) - TELEGRAM_MAX_ITEMS} más en DB)")
        body = "\n\n".join(lines)
        telegram_send_message(header + "\n" + body)
    else:
        # Muestra algunos resultados actuales (para que “siempre mande algo”)
        for it in sample_items[:TELEGRAM_MAX_ITEMS]:
            lines.append(fmt_listing_line(it))
        body = "\n\n".join(lines) if lines else "Sin resultados tras filtros (revisa geocoding/portal)."
        telegram_send_message(header + "\n" + body)


# ----------------------------
# SCRAPERS
# ----------------------------
def scrape_inmuebles24(search_url: str) -> List[Listing]:
    html = fetch_html(search_url, "inmuebles24")
    soup = BeautifulSoup(html, BS4_PARSER)
    out: List[Listing] = []

    base = "https://www.inmuebles24.com"

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        # normaliza URL
        if href.startswith("/"):
            full = urljoin(base, href)
        else:
            full = href

        if "inmuebles24.com" not in full:
            continue

        # Heurística: evita navegación/anchors y queda con links “profundos”
        if any(x in full for x in ["javascript:", "mailto:", "#"]):
            continue
        if "/casas-en-venta" in full:
            continue

        # Intentar capturar detalle: suele ser url larga
        if len(full) < 35:
            continue

        title = a.get_text(" ", strip=True)
        if not title or len(title) < 8:
            continue

        card = a.find_parent()
        price_text = ""
        location_text = ""

        if card:
            cand_price = card.find(string=re.compile(r"(\$|MXN|MN)\s*[\d,.]+", re.I))
            if cand_price:
                price_text = str(cand_price)

            # busca texto de ubicación en el card
            cand_loc = card.find(string=re.compile(r"Zapopan|Jalisco|El Briseño|Briseño|Guadalajara", re.I))
            if cand_loc:
                location_text = str(cand_loc).strip()

        price = normalize_price_to_int(price_text)

        out.append(Listing("inmuebles24", full.split("#")[0], title[:200], price, location_text[:200]))

    return dedup_by_url(out)


def scrape_mercadolibre(search_url: str) -> List[Listing]:
    html = fetch_html(search_url, "mercadolibre")
    soup = BeautifulSoup(html, BS4_PARSER)
    out: List[Listing] = []

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href or "mercadolibre.com.mx" not in href:
            continue

        title = a.get_text(" ", strip=True)
        if not title or len(title) < 10:
            continue

        card = a.find_parent()
        price_text = ""
        location_text = ""

        if card:
            cand_price = card.find(string=re.compile(r"\$\s*[\d,.]+|MXN\s*[\d,.]+", re.I))
            if cand_price:
                price_text = str(cand_price)

            cand_loc = card.find(string=re.compile(r"Zapopan|Jalisco|El Briseño|Briseño", re.I))
            if cand_loc:
                location_text = str(cand_loc).strip()

        price = normalize_price_to_int(price_text)
        out.append(Listing("mercadolibre", href.split("#")[0], title[:200], price, location_text[:200]))

    return dedup_by_url(out)


def scrape_vivanuncios(search_url: str) -> List[Listing]:
    html = fetch_html(search_url, "vivanuncios")
    soup = BeautifulSoup(html, BS4_PARSER)
    out: List[Listing] = []
    base = "https://www.vivanuncios.com.mx"

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        if href.startswith("/"):
            full = urljoin(base, href)
        elif "vivanuncios.com.mx" in href:
            full = href
        else:
            continue

        title = a.get_text(" ", strip=True)
        if not title or len(title) < 10:
            continue

        card = a.find_parent()
        price_text = ""
        location_text = ""

        if card:
            cand_price = card.find(string=re.compile(r"(MN|MXN|\$)\s*[\d,.]+", re.I))
            if cand_price:
                price_text = str(cand_price)

            cand_loc = card.find(string=re.compile(r"Zapopan|Jalisco|El Briseño|Briseño", re.I))
            if cand_loc:
                location_text = str(cand_loc).strip()

        price = normalize_price_to_int(price_text)
        out.append(Listing("vivanuncios", full.split("#")[0], title[:200], price, location_text[:200]))

    return dedup_by_url(out)


# ----------------------------
# DB
# ----------------------------
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS listings (
            url TEXT PRIMARY KEY,
            portal TEXT NOT NULL,
            title TEXT,
            price_mxn INTEGER,
            location TEXT,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def upsert(conn: sqlite3.Connection, items: Iterable[Listing]) -> List[Listing]:
    new_items: List[Listing] = []
    for it in items:
        cur = conn.execute("SELECT url FROM listings WHERE url = ?", (it.url,))
        exists = cur.fetchone() is not None

        if not exists:
            conn.execute(
                "INSERT INTO listings(url, portal, title, price_mxn, location) VALUES(?,?,?,?,?)",
                (it.url, it.portal, it.title, it.price_mxn, it.location),
            )
            new_items.append(it)
        else:
            conn.execute(
                "UPDATE listings SET last_seen=CURRENT_TIMESTAMP, price_mxn=?, title=?, location=? WHERE url=?",
                (it.price_mxn, it.title, it.location, it.url),
            )
    conn.commit()
    return new_items


# ----------------------------
# RUN
# ----------------------------
def run_all() -> Dict[str, List[Listing]]:
    results: Dict[str, List[Listing]] = {}

    for portal, url in SEARCH_URLS.items():
        try:
            if portal == "inmuebles24":
                items = scrape_inmuebles24(url)
            elif portal == "mercadolibre":
                items = scrape_mercadolibre(url)
            elif portal == "vivanuncios":
                items = scrape_vivanuncios(url)
            else:
                items = []

            # precio: deja pasar None (best effort)
            items = [x for x in items if x.price_mxn is None or in_price_range(x.price_mxn)]
            results[portal] = items

        except Exception as e:
            print(f"[WARN] {portal} falló: {e}")
            results[portal] = []

    return results


def main() -> None:
    print("== Personal Housing Finder (El Briseño) + Telegram ==")
    print(f"Precio: {PRICE_MIN:,} - {PRICE_MAX:,} MXN")
    print(f"Centro: {HOME_LAT}, {HOME_LON} | Radio: {RADIUS_KM} km")
    print(f"Parser BS4: {BS4_PARSER}")
    print(f"Telegram: {'ON' if TELEGRAM_ENABLED else 'OFF'}")
    print(f"Browser fallback: {USE_BROWSER_FALLBACK} (headless={BROWSER_HEADLESS})\n")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    by_portal = run_all()

    portal_counts = {k: len(v) for k, v in by_portal.items()}

    all_items: List[Listing] = []
    for portal, items in by_portal.items():
        print(f"{portal}: {len(items)} items (best effort)")
        all_items.extend(items)

    # Filtro por radio
    cache = load_geo_cache()
    before = len(all_items)

    filtered: List[Listing] = []
    geocoded_ok = 0
    geocoded_unknown = 0
    geo_used = 0

    total = len(all_items)
    for idx, it in enumerate(all_items, start=1):
        # Progreso cada 25 items
        if idx % 25 == 0:
            print(f"[PROGRESS] {idx}/{total} | geo_used={geo_used}/{MAX_GEOCODE_PER_RUN} | kept={len(filtered)}")

        # filtro por precio si está parseado
        if it.price_mxn is not None and not in_price_range(it.price_mxn):
            continue

        # Si ya gastamos el presupuesto de geocoding, no geocodifiques más
        dist = None
        if geo_used < MAX_GEOCODE_PER_RUN:
            dist = compute_distance_km_for_listing(it, cache)
            if dist is not None:
                geo_used += 1

        it.distance_km = dist

        if dist is None:
            geocoded_unknown += 1
            if INCLUDE_UNKNOWN_GEO:
                filtered.append(it)
        else:
            geocoded_ok += 1
            if dist <= RADIUS_KM:
                filtered.append(it)

    save_geo_cache(cache)

    after = len(filtered)
    print(f"\nFiltro radio: {before} -> {after} (dentro de {RADIUS_KM} km o unknown={INCLUDE_UNKNOWN_GEO})\n")

    new_items = upsert(conn, filtered)

    print("=== NUEVOS (precio y radio) ===")
    for it in new_items[:50]:
        p = f"${it.price_mxn:,}" if isinstance(it.price_mxn, int) else "s/p"
        loc = it.location or "sin ubicación"
        d = f"{it.distance_km:.1f} km" if isinstance(it.distance_km, float) else "dist ?"
        print(f"- [{it.portal}] {p} | {d} | {loc} | {it.title}\n  {it.url}")

    print(f"\nTotal nuevos: {len(new_items)}")
    print(f"DB: {DB_PATH} | Geocache: {GEO_CACHE_PATH}\n")

    # Enviar SIEMPRE summary a Telegram:
    # - si hay nuevos: lista nuevos
    # - si no hay: manda una muestra del “top actual” para confirmar que corre
    sample_items = filtered[:TELEGRAM_MAX_ITEMS]
    send_run_summary(portal_counts, after, new_items, sample_items, geocoded_ok, geocoded_unknown)
    send_cheapest_summary(filtered)


if __name__ == "__main__":
    main()