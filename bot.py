#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Personal Housing Finder - Zapopan (Unidad Deportiva El Briseño) + Telegram (Render-ready)

Highlights:
- Playwright headless automático en Render
- Fallback robusto para 403 (requests -> Playwright)
- Si Playwright falla, no rompe la corrida (solo devuelve 0 para ese portal)
- Telegram siempre manda resumen + Top 15 baratas
- Geocoding limitado por corrida para no tardarse eternidad (Nominatim)
- Progreso impreso en consola

Requisitos:
  pip install requests beautifulsoup4 playwright
  python -m playwright install --with-deps chromium

Env vars:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

Render Build Command recomendado:
  pip install -r requirements.txt && python -m playwright install --with-deps chromium

Render Schedule (10 PM Guadalajara):
  Guadalajara UTC-6 => 22:00 local = 04:00 UTC
  0 4 * * *
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

HOME_LAT = 20.626058
HOME_LON = -103.439071
RADIUS_KM = 7.0

# Para no quedarte en 0 si la ubicación es mala
INCLUDE_UNKNOWN_GEO = True

# Limita geocoding por corrida (Nominatim es lento y rate-limited)
MAX_GEOCODE_PER_RUN = 40
SKIP_GEO_IF_LOCATION_TOO_GENERIC = True

DB_PATH = "housing.db"
GEO_CACHE_PATH = Path("geocache.json")

NOMINATIM_BASE = "https://nominatim.openstreetmap.org/search"

SEARCH_URLS = {
    "inmuebles24": "https://www.inmuebles24.com/casas-en-venta-en-zapopan-de-1500000-a-2000000-pesos.html",
    "mercadolibre": "https://inmuebles.mercadolibre.com.mx/casas/venta/jalisco/zapopan/_PriceRange_1500000MXN-8000000MXN",
    "vivanuncios": "https://www.vivanuncios.com.mx/s-casas-en-venta/zapopan/v1c1293l14828p1",
}

# BS4 parser
BS4_PARSER = "html.parser"

# Playwright fallback por portal (403 -> navegador real)
USE_BROWSER_FALLBACK = {
    "inmuebles24": True,
    "mercadolibre": False,
    "vivanuncios": True,
}

# Render env: define RENDER="true" en Render internamente
IS_RENDER = bool(os.getenv("RENDER"))

# En Render SIEMPRE headless
BROWSER_HEADLESS = True if IS_RENDER else False

# Telegram env vars (NO hardcode)
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Telegram: siempre manda resumen + Top 15
ALWAYS_SEND_TELEGRAM = True
TELEGRAM_MAX_ITEMS = 15

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
    location: str
    distance_km: Optional[float] = None


# ----------------------------
# HELPERS
# ----------------------------
def normalize_price_to_int(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.upper().replace("\xa0", " ").strip()

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
            "Playwright no está instalado. En Render usa build:\n"
            "pip install -r requirements.txt && python -m playwright install --with-deps chromium"
        ) from e

    ua = SESSION.headers.get("User-Agent")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=BROWSER_HEADLESS,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(user_agent=ua, locale="es-MX")
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except Exception:
            pass

        # Scroll para lazy-load
        for _ in range(6):
            page.mouse.wheel(0, 1800)
            page.wait_for_timeout(700)

        title = (page.title() or "").lower()
        if any(k in title for k in ["captcha", "attention", "verify", "robot", "cloudflare"]):
            print(f"[WARN] Playwright posible botwall | title='{title}' | url={url}")

        html = page.content()
        browser.close()
        return html


def fetch_html(url: str, portal: str) -> str:
    r = SESSION.get(url, timeout=30)

    if r.status_code == 403 and USE_BROWSER_FALLBACK.get(portal, False):
        print(f"[INFO] {portal}: 403 con requests, intentando fallback navegador (Playwright)...")
        try:
            return fetch_html_browser(url)
        except Exception as e:
            print(f"[WARN] {portal}: Playwright falló: {e}")
            return ""

    try:
        r.raise_for_status()
    except Exception as e:
        print(f"[WARN] {portal}: requests falló: {e}")
        return ""

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

    if SKIP_GEO_IF_LOCATION_TOO_GENERIC:
        low = loc.lower().strip()
        generic = {"zapopan", "zapopan, jalisco", "jalisco", "guadalajara", ""}
        if low in generic:
            return None

    if not loc:
        # usar título como pista, pero puede ser ruido
        t = (it.title or "").strip()
        loc = t if t else ""

    if not loc:
        return None

    query = loc
    if "ZAPOPAN" not in query.upper():
        query = f"{query}, Zapopan, Jalisco, México"

    coords = geocode_osm(query, cache)
    if coords is None:
        return None

    lat, lon = coords
    return haversine_km(HOME_LAT, HOME_LON, lat, lon)


# ----------------------------
# TELEGRAM
# ----------------------------
def telegram_send_message(text: str) -> None:
    if not TELEGRAM_ENABLED:
        print("[INFO] Telegram OFF (faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    r = SESSION.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        print(f"[WARN] Telegram error {r.status_code}: {r.text[:200]}")


def fmt_listing_line(it: Listing) -> str:
    p = f"${it.price_mxn:,}" if isinstance(it.price_mxn, int) else "s/p"
    d = f"{it.distance_km:.1f} km" if isinstance(it.distance_km, float) else "dist ?"
    loc = it.location or "sin ubicación"
    return f"• [{it.portal}] {p} ({d})\n{loc}\n{it.url}"


def send_summary_and_top15(
    portal_counts: Dict[str, int],
    filtered: List[Listing],
    new_items: List[Listing],
    geocoded_ok: int,
    geocoded_unknown: int,
) -> None:
    if not ALWAYS_SEND_TELEGRAM:
        if new_items:
            telegram_send_message(f"🆕 Nuevos: {len(new_items)}")
        return

    header = (
        "🏠 Corrida completada (El Briseño)\n"
        f"💰 {PRICE_MIN:,}–{PRICE_MAX:,} MXN | 📍 ≤ {RADIUS_KM} km\n"
        "📦 Portales: " + ", ".join([f"{k}:{v}" for k, v in portal_counts.items()]) + "\n"
        f"🧭 Tras filtros: {len(filtered)} | 🆕 Nuevos: {len(new_items)}\n"
        f"🗺️ Geocode ok:{geocoded_ok} | unknown:{geocoded_unknown}\n"
    )

    # Top 15 más baratas con precio válido
    priced = [x for x in filtered if isinstance(x.price_mxn, int)]
    priced.sort(key=lambda x: x.price_mxn)
    top = priced[:TELEGRAM_MAX_ITEMS]

    if not top:
        telegram_send_message(header + "\n(No hay resultados con precio parseado aún.)")
        return

    body = "\n\n".join(fmt_listing_line(x) for x in top)
    telegram_send_message(header + "\n💸 Top 15 más baratas:\n\n" + body)


# ----------------------------
# SCRAPERS
# ----------------------------
def scrape_inmuebles24(url: str) -> List[Listing]:
    html = fetch_html(url, "inmuebles24")
    if not html:
        return []

    soup = BeautifulSoup(html, BS4_PARSER)
    out: List[Listing] = []
    base = "https://www.inmuebles24.com"

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        full = urljoin(base, href) if href.startswith("/") else href
        if "inmuebles24.com" not in full:
            continue
        if any(x in full for x in ["javascript:", "mailto:", "#"]):
            continue
        if "/casas-en-venta" in full:
            continue
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

            cand_loc = card.find(string=re.compile(r"Zapopan|Jalisco|El Briseño|Briseño|Guadalajara", re.I))
            if cand_loc:
                location_text = str(cand_loc).strip()

        price = normalize_price_to_int(price_text)
        out.append(Listing("inmuebles24", full.split("#")[0], title[:200], price, location_text[:200]))

    return dedup_by_url(out)


def scrape_mercadolibre(url: str) -> List[Listing]:
    html = fetch_html(url, "mercadolibre")
    if not html:
        return []

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


def scrape_vivanuncios(url: str) -> List[Listing]:
    html = fetch_html(url, "vivanuncios")
    if not html:
        return []

    soup = BeautifulSoup(html, BS4_PARSER)
    out: List[Listing] = []
    base = "https://www.vivanuncios.com.mx"

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        full = urljoin(base, href) if href.startswith("/") else href
        if "vivanuncios.com.mx" not in full:
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
# MAIN
# ----------------------------
def main() -> None:
    print("== Personal Housing Finder (El Briseño) + Telegram ==")
    print(f"Precio: {PRICE_MIN:,} - {PRICE_MAX:,} MXN")
    print(f"Centro: {HOME_LAT}, {HOME_LON} | Radio: {RADIUS_KM} km")
    print(f"Parser BS4: {BS4_PARSER}")
    print(f"Render: {IS_RENDER} | Playwright headless: {BROWSER_HEADLESS}")
    print(f"Telegram: {'ON' if TELEGRAM_ENABLED else 'OFF'}")
    print(f"Browser fallback: {USE_BROWSER_FALLBACK}\n")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Run scrapers
    portal_items: Dict[str, List[Listing]] = {
        "inmuebles24": scrape_inmuebles24(SEARCH_URLS["inmuebles24"]),
        "mercadolibre": scrape_mercadolibre(SEARCH_URLS["mercadolibre"]),
        "vivanuncios": scrape_vivanuncios(SEARCH_URLS["vivanuncios"]),
    }
    portal_counts = {k: len(v) for k, v in portal_items.items()}

    for k, v in portal_counts.items():
        print(f"{k}: {v} items (best effort)")

    all_items: List[Listing] = []
    for items in portal_items.values():
        # filtra por precio si viene parseado
        all_items.extend([x for x in items if x.price_mxn is None or in_price_range(x.price_mxn)])

    # Filtro por radio con geocoding limitado
    cache = load_geo_cache()
    filtered: List[Listing] = []
    geocoded_ok = 0
    geocoded_unknown = 0
    geo_used = 0

    total = len(all_items)
    print(f"\nTotal candidatos pre-radio: {total}")
    for idx, it in enumerate(all_items, start=1):
        if idx % 25 == 0:
            print(f"[PROGRESS] {idx}/{total} geo_used={geo_used}/{MAX_GEOCODE_PER_RUN} kept={len(filtered)}")

        if it.price_mxn is not None and not in_price_range(it.price_mxn):
            continue

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

    print(f"\nTras filtros (radio/unknown): {len(filtered)}")
    print(f"Geocode ok={geocoded_ok} unknown={geocoded_unknown}")
    print(f"DB: {DB_PATH} | Geocache: {GEO_CACHE_PATH}\n")

    # Persist
    new_items = upsert(conn, filtered)
    print(f"Total nuevos: {len(new_items)}\n")

    # Telegram
    send_summary_and_top15(portal_counts, filtered, new_items, geocoded_ok, geocoded_unknown)


if __name__ == "__main__":
    main()