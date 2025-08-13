import io
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import requests
import numpy as np
import math

from django.db import transaction, DataError
from django.db.models import F, Q
from django.apps import apps


METRICS_SLUGS: Dict[str, List[str]] = {
    # GDP per capita, PPP preferred; fall back to World Bank GDP per capita if PPP unavailable
    "gdp_ppp_per_capita": [
        # PPP (if available)
        "gdp-per-capita-worldbank-ppp-constant-2017",
        "gdp-per-capita-ppp-constant-2017",
        # Fall back to non-PPP World Bank series if PPP not available
        "gdp-per-capita-worldbank",
        "gdp-per-capita",
        # Maddison fallback
        "gdp-per-capita-maddison-2020",
        "gdp-per-capita-maddison",
    ],
    # Life expectancy at birth (years)
    "life_expectancy": [
        "life-expectancy",
    ],
    # Education index computed from schooling components; keep empty to avoid direct fetch attempts
    "education_index": [],
    # Safety: prefer computing from homicide; leave empty to skip direct Grapher fetch attempts that 404
    "safety_index": [
        # If a direct safety index exists on OWID in future, put it here
    ],
    # Prosperity proxy: Human Development Index (simple and widely available)
    "prosperity_index": [
        "human-development-index",
    ],
    # Freedom of speech proxy: WGI Voice & Accountability (if available on OWID)
    "freedom_of_speech_index": [
        "wgi-voice-and-accountability-estimate",
        "voice-and-accountability",
        "voice-and-accountability-wgi",
    ],
    # Democracy: try Polity and V-Dem fallbacks if present
    "democracy_index": [
        "polity2",
        "polity2-score",
        "v-dem-electoral-democracy-index",
        "v-dem-liberal-democracy-index",
        "eiu-democracy-index",
    ],
    # Corruption: use WGI Control of Corruption proxies if available
    "corruption_index": [
        "wgi-control-of-corruption-estimate",
        "control-of-corruption",
        "control-of-corruption-wgi",
    ],
}

# World Bank indicator fallbacks for select metrics when OWID sources are unavailable.
# Only include indicators that are reliably available in the World Bank API without extra source params.
WB_FALLBACKS: Dict[str, str] = {
    # GDP, PPP (constant 2017 international $) per capita
    "gdp_ppp_per_capita": "NY.GDP.PCAP.PP.KD",
    # Life expectancy at birth, total (years)
    "life_expectancy": "SP.DYN.LE00.IN",
    # Education and safety have custom fallbacks handled elsewhere; leave others empty by default.
}

GRPAHER_BASE = "https://ourworldindata.org/grapher"
WB_API_BASE = "https://api.worldbank.org/v2"

# Use browser-like headers to avoid occasional 403/429 from CDN
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
    "Referer": "https://ourworldindata.org/",
    "Connection": "keep-alive",
}

# World Bank JSON headers
WB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive",
}

# Cache for WB country list to filter out aggregates
_WB_COUNTRIES_CACHE: Optional[Dict[str, str]] = None

logger = logging.getLogger(__name__)


# Historical data sources URLs
FREEDOM_HOUSE_ALL_DATA_2013_2024 = "https://freedomhouse.org/sites/default/files/2024-02/All_data_FIW_2013-2024.xlsx"
FREEDOM_HOUSE_PRESS_HISTORICAL = "https://freedomhouse.org/sites/default/files/2020-02/FOTP1980-FOTP2017_Public-Data.xlsx"

def _fetch_freedom_house_historical() -> Optional[pd.DataFrame]:
    """Download and parse Freedom House historical data (2013-2024) for freedom of speech and democracy."""
    try:
        # Download Excel file
        df = pd.read_excel(FREEDOM_HOUSE_ALL_DATA_2013_2024, sheet_name=0)
        
        # The Excel contains multiple metrics, focus on relevant ones for our use case
        # Columns might include: Country, Year, PR (Political Rights), CL (Civil Liberties), etc.
        # We'll map these to our freedom_of_speech_index and democracy_index
        
        # Rename columns to match our format
        df = df.rename(columns={
            "Country/Territory": "Entity",
            "Year": "Year",
            "CL": "civil_liberties",  # For freedom of speech proxy
            "PR": "political_rights",  # For democracy proxy
        })
        
        # Filter to 2016-2024 range to match our database
        df = df[(df["Year"] >= 2016) & (df["Year"] <= 2024)]
        
        # Create normalized indices (Freedom House uses 1-7 scale, lower is better)
        # Convert to 0-1 scale where higher is better
        if "civil_liberties" in df.columns:
            df["freedom_of_speech_index"] = (8 - df["civil_liberties"]) / 7.0
        if "political_rights" in df.columns:
            df["democracy_index"] = (8 - df["political_rights"]) / 7.0
            
        return df[["Entity", "Year", "freedom_of_speech_index", "democracy_index"]].dropna(subset=["Entity", "Year"])
        
    except Exception as e:
        logger.warning("Failed to fetch/parse Freedom House historical data: %s", e)
        return None


def _fetch_transparency_international_historical() -> Optional[pd.DataFrame]:
    """Get Transparency International CPI 2024 data (hardcoded from official website)."""
    try:
        logger.info("Loading TI CPI 2024 data (hardcoded from official source)")
        
        # TI CPI 2024 data extracted from https://www.transparency.org/en/cpi/2024
        # Score scale: 0-100 (higher is better, less corrupt)
        cpi_data_2024 = {
            'Denmark': 90, 'Finland': 88, 'Singapore': 84, 'New Zealand': 83,
            'Luxembourg': 81, 'Norway': 81, 'Switzerland': 81, 'Sweden': 80,
            'Netherlands': 78, 'Australia': 77, 'Iceland': 77, 'Ireland': 77,
            'Estonia': 76, 'Uruguay': 76, 'Canada': 75, 'Germany': 75,
            'Hong Kong': 74, 'Bhutan': 72, 'Seychelles': 72, 'Japan': 71,
            'United Kingdom': 71, 'Belgium': 69, 'Barbados': 68, 'United Arab Emirates': 68,
            'Austria': 67, 'France': 67, 'Taiwan': 67, 'Bahamas': 65,
            'United States of America': 65, 'Israel': 64, 'South Korea': 64,
            'Chile': 63, 'Lithuania': 63, 'Saint Vincent and the Grenadines': 63,
            'Cabo Verde': 62, 'Dominica': 60, 'Slovenia': 60, 'Latvia': 59,
            'Qatar': 59, 'Saint Lucia': 59, 'Saudi Arabia': 59, 'Costa Rica': 58,
            'Botswana': 57, 'Portugal': 57, 'Rwanda': 57, 'Cyprus': 56,
            'Czechia': 56, 'Grenada': 56, 'Spain': 56, 'Fiji': 55, 'Oman': 55,
            'Italy': 54, 'Bahrain': 53, 'Georgia': 53, 'Poland': 53,
            'Mauritius': 51, 'Malaysia': 50, 'Vanuatu': 50, 'Greece': 49,
            'Jordan': 49, 'Namibia': 49, 'Slovakia': 49, 'Armenia': 47,
            'Croatia': 47, 'Kuwait': 46, 'Malta': 46, 'Montenegro': 46,
            'Romania': 46, 'Benin': 45, 'Côte d\'Ivoire': 45, 'Sao Tome and Principe': 45,
            'Senegal': 45, 'Jamaica': 44, 'Kosovo': 44, 'Timor-Leste': 44,
            'Bulgaria': 43, 'China': 43, 'Moldova': 43, 'Solomon Islands': 43,
            'Albania': 42, 'Ghana': 42, 'Burkina Faso': 41, 'Cuba': 41,
            'Hungary': 41, 'South Africa': 41, 'Tanzania': 41, 'Trinidad and Tobago': 41,
            'Kazakhstan': 40, 'North Macedonia': 40, 'Suriname': 40, 'Vietnam': 40,
            'Colombia': 39, 'Guyana': 39, 'Tunisia': 39, 'Zambia': 39,
            'Gambia': 38, 'India': 38, 'Maldives': 38, 'Argentina': 37,
            'Ethiopia': 37, 'Indonesia': 37, 'Lesotho': 37, 'Morocco': 37,
            'Dominican Republic': 36, 'Serbia': 35, 'Ukraine': 35, 'Algeria': 34,
            'Brazil': 34, 'Malawi': 34, 'Nepal': 34, 'Niger': 34, 'Thailand': 34,
            'Turkey': 34, 'Belarus': 33, 'Bosnia and Herzegovina': 33, 'Laos': 33,
            'Mongolia': 33, 'Panama': 33, 'Philippines': 33, 'Sierra Leone': 33,
            'Angola': 32, 'Ecuador': 32, 'Kenya': 32, 'Sri Lanka': 32, 'Togo': 32,
            'Uzbekistan': 32, 'Djibouti': 31, 'Papua New Guinea': 31, 'Peru': 31,
            'Egypt': 30, 'El Salvador': 30, 'Mauritania': 30, 'Bolivia': 28,
            'Guinea': 28, 'Eswatini': 27, 'Gabon': 27, 'Liberia': 27, 'Mali': 27,
            'Pakistan': 27, 'Cameroon': 26, 'Iraq': 26, 'Madagascar': 26,
            'Mexico': 26, 'Nigeria': 26, 'Uganda': 26, 'Guatemala': 25,
            'Kyrgyzstan': 25, 'Mozambique': 25, 'Central African Republic': 24,
            'Paraguay': 24, 'Bangladesh': 23, 'Congo': 23, 'Iran': 23,
            'Azerbaijan': 22, 'Honduras': 22, 'Lebanon': 22, 'Russia': 22,
            'Cambodia': 21, 'Chad': 21, 'Comoros': 21, 'Guinea-Bissau': 21,
            'Zimbabwe': 21, 'Democratic Republic of the Congo': 20, 'Tajikistan': 19,
            'Afghanistan': 17, 'Burundi': 17, 'Turkmenistan': 17, 'Haiti': 16,
            'Myanmar': 16, 'North Korea': 15, 'Sudan': 15, 'Nicaragua': 14,
            'Equatorial Guinea': 13, 'Eritrea': 13, 'Libya': 13, 'Yemen': 13,
            'Syria': 12, 'Venezuela': 10, 'Somalia': 9, 'South Sudan': 8
        }
        
     
        data = []
        for country, score in cpi_data_2024.items():
            
            normalized_score = float(score) / 100.0
            data.append({
                'Entity': country,
                'Year': 2024,
                'corruption_index': normalized_score
            })
        
        result_df = pd.DataFrame(data)
        logger.info("Loaded TI CPI data for %d countries", len(result_df))
        return result_df
        
    except Exception as e:
        logger.warning("Failed to load TI CPI data: %s", e)
        return None
    return None


# RSF World Press Freedom Index 2025 CSV URL
RSF_PRESS_FREEDOM_2025_CSV = "https://rsf.org/sites/default/files/import_classement/2025.csv"

def _fetch_rsf_press_freedom_2025() -> Optional[pd.DataFrame]:
    """Download and parse the RSF 2025 Press Freedom Index CSV."""
    try:
        # Try different encodings for the CSV file
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                logger.info("Trying RSF CSV with encoding: %s", encoding)
                df = pd.read_csv(RSF_PRESS_FREEDOM_2025_CSV, sep=';', encoding=encoding)
                logger.info("RSF CSV loaded successfully with encoding: %s", encoding)
                logger.info("RSF CSV shape: %s, columns: %s", df.shape, list(df.columns))
                
                # Check for expected columns
                expected_cols = ['Country_EN', 'Score 2025']
                found_cols = [col for col in expected_cols if col in df.columns]
                logger.info("Found expected columns: %s", found_cols)
                
                if len(found_cols) >= 2:
                    # Normalize column names
                    df = df.rename(columns={
                        "Country_EN": "Entity",
                        "Score 2025": "press_freedom_score",
                    })
                    
                    # Clean the score column (remove commas and convert to float)
                    df["press_freedom_score"] = pd.to_numeric(
                        df["press_freedom_score"].astype(str).str.replace(',', '.'), 
                        errors='coerce'
                    )
                    df["Year"] = 2024  # Use 2024 to match main data timeframe
                    
                    result = df[["Entity", "Year", "press_freedom_score"]].dropna()
                    logger.info("RSF data processed: %d countries", len(result))
                    return result
                    
            except Exception as e:
                logger.warning("RSF encoding %s failed: %s", encoding, e)
                continue
                
        logger.warning("All RSF encodings failed")
        return None
        
    except Exception as e:
        logger.warning("Failed to fetch RSF press freedom data: %s", e)
        return None
    except Exception as e:
        logger.warning("Failed to fetch/parse RSF Press Freedom Index: %s", e)
        return None


def _get_models():
    """Dynamically retrieve Country and CountryMetric models to avoid import-time errors."""
    Country = apps.get_model("countries", "Country")
    CountryMetric = apps.get_model("countries", "CountryMetric")
    return Country, CountryMetric


# =========================
# Remote fetching functions
# =========================

def _fetch_grapher_csv(slug: str) -> Optional[pd.DataFrame]:
    """Download a Grapher CSV for the given variable slug.

    Returns a DataFrame with columns: Entity, Code, Year, value (renamed to 'value').
    Returns None on 404 or parsing issues.
    """
    url = f"{GRPAHER_BASE}/{slug}.csv"
    try:
        # Simple retry with backoff to mitigate transient 403/429/5xx
        last_status = None
        for attempt in range(3):
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            last_status = resp.status_code
            if resp.status_code == 200:
                break
            if resp.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(1.0 * (attempt + 1))
                continue
            # For other statuses (e.g., 404), don't retry
            break
        if last_status != 200:
            logger.warning("Grapher fetch failed for %s: %s", slug, last_status)
            return None
        # Load CSV into DataFrame
        df = pd.read_csv(io.StringIO(resp.text))
        # Ensure required columns exist with common variants
        # Some datasets include a 'Continent' column; exclude non-numeric columns when picking value.
        cols_lower = {c.lower(): c for c in df.columns}
        entity_col = cols_lower.get("entity") or cols_lower.get("country")
        code_col = cols_lower.get("code") or cols_lower.get("iso_code") or cols_lower.get("iso code")
        year_col = cols_lower.get("year")
        if not (entity_col and code_col and year_col):
            logger.warning("Unexpected CSV columns for %s: %s", slug, list(df.columns))
            return None

        # Determine the metric column by selecting the column with the most numeric values,
        # excluding known non-metric columns.
        known_non_metric = {entity_col, code_col, year_col, cols_lower.get("continent"), cols_lower.get("region")}
        candidate_cols = [c for c in df.columns if c not in known_non_metric and c is not None]
        if not candidate_cols:
            logger.warning("No candidate metric columns for %s", slug)
            return None

        best_col = None
        best_numeric = -1
        for c in candidate_cols:
            # Count numeric (coercible) values
            s = pd.to_numeric(df[c], errors="coerce")
            numeric_count = int(s.notna().sum())
            if numeric_count > best_numeric:
                best_numeric = numeric_count
                best_col = c
        if not best_col or best_numeric <= 0:
            logger.warning("No numeric metric column found for %s among %s", slug, candidate_cols)
            return None

        # Build standardized frame
        out = df[[entity_col, code_col, year_col, best_col]].rename(columns={
            entity_col: "Entity",
            code_col: "Code",
            year_col: "Year",
            best_col: "value",
        })
        # Ensure standard dtypes
        out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
        # Coerce value to numeric, keep NaN for now
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out
    except (requests.RequestException, pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        logger.exception("Error fetching slug %s: %s", slug, e)
        return None


def _fetch_metric_prefer_first_available(slugs: List[str]) -> Optional[pd.DataFrame]:
    """Try a list of slugs and return the first successful DataFrame.

    Adds a 'metric' column with the metric key later in the merge step, so here we just return the value series.
    """
    for slug in slugs:
        df = _fetch_grapher_csv(slug)
        if df is not None and not df.empty:
            df.attrs["_slug"] = slug  # annotate chosen slug for debugging
            return df
    return None


# ---------------- World Bank helpers ----------------

def _get_wb_countries() -> Dict[str, str]:
    """Return mapping of ISO3 code -> country name, excluding aggregates."""
    global _WB_COUNTRIES_CACHE
    if _WB_COUNTRIES_CACHE is not None:
        return _WB_COUNTRIES_CACHE
    url = f"{WB_API_BASE}/country?format=json&per_page=400"
    try:
        resp = requests.get(url, headers=WB_HEADERS, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2:
            _WB_COUNTRIES_CACHE = {}
            return _WB_COUNTRIES_CACHE
        countries = {}
        for item in payload[1]:
            # Exclude aggregates where region.id == 'NA'
            region = item.get("region", {})
            if isinstance(region, dict) and region.get("id") == "NA":
                continue
            iso3 = item.get("id")  # iso3 code
            name = item.get("name")
            if iso3 and name:
                countries[iso3] = name
        _WB_COUNTRIES_CACHE = countries
        return countries
    except Exception as e:
        logger.warning("Failed to load WB countries: %s", e)
        _WB_COUNTRIES_CACHE = {}
        return _WB_COUNTRIES_CACHE


def _fetch_worldbank_indicator(indicator: str) -> Optional[pd.DataFrame]:
    """Fetch a World Bank indicator as a DataFrame with OWID-like columns.

    Columns: Entity, Code, Year, value
    """
    url = f"{WB_API_BASE}/country/all/indicator/{indicator}?format=json&per_page=20000"
    try:
        resp = requests.get(url, headers=WB_HEADERS, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return None
        wb_countries = _get_wb_countries()
        rows = []
        for row in payload[1]:
            code = row.get("countryiso3code")
            if not code or code not in wb_countries:
                continue  # skip aggregates and unknowns
            entity = wb_countries.get(code) or (row.get("country") or {}).get("value")
            year = row.get("date")
            value = row.get("value")
            # Coerce types
            try:
                year_i = int(year)
            except (TypeError, ValueError):
                continue
            rows.append({"Entity": entity, "Code": code, "Year": year_i, "value": value})
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        return df
    except Exception as e:
        logger.warning("WB fetch failed for %s: %s", indicator, e)
        return None

# ---------------- End World Bank helpers ----------------


def _compute_education_index(last_n_years: int) -> Optional[pd.DataFrame]:
    """Compute UNDP-style education index from schooling components if available.

    Falls back to World Bank School life expectancy (SE.SCH.LIFE) if OWID MYS/EYS not available.
    If only EYS is available, returns EYSI as a lower-bound proxy for the education_index.
    """
    # Candidate slugs for schooling years
    mys_slugs = [
        "mean-years-of-schooling",
        "barro-lee-mean-years-of-schooling",
        "mean-years-of-schooling-united-nations",
    ]
    eys_slugs = [
        "expected-years-of-schooling",
        "expected-years-of-schooling-long-run",
    ]

    mys = _fetch_metric_prefer_first_available(mys_slugs)
    eys = _fetch_metric_prefer_first_available(eys_slugs)

    # WB fallback for expected years if needed
    if eys is None:
        eys_wb = _fetch_worldbank_indicator("SE.SCH.LIFE")
        if eys_wb is not None and not eys_wb.empty:
            eys = eys_wb

    if mys is None and eys is None:
        return None

    key_cols = ["Entity", "Code", "Year"]

    if mys is not None and eys is not None:
        mys = mys.rename(columns={"value": "mys"})
        eys = eys.rename(columns={"value": "eys"})
        edu = pd.merge(mys[key_cols + ["mys"]], eys[key_cols + ["eys"]], on=key_cols, how="inner")

        current_year = datetime.utcnow().year
        year_min = current_year - (last_n_years - 1)
        edu = edu[(edu["Year"] >= year_min) & (edu["Year"] <= current_year)].copy()

        edu["MYSI"] = (pd.to_numeric(edu["mys"], errors="coerce") / 15.0).clip(upper=1)
        edu["EYSI"] = (pd.to_numeric(edu["eys"], errors="coerce") / 18.0).clip(upper=1)
        edu["education_index"] = (edu["MYSI"] + edu["EYSI"]) / 2.0
        return edu[key_cols + ["education_index"]]

    # If only EYS available, use EYSI as proxy
    if eys is not None and mys is None:
        eys = eys.rename(columns={"value": "eys"})
        current_year = datetime.utcnow().year
        year_min = current_year - (last_n_years - 1)
        eys = eys[(eys["Year"] >= year_min) & (eys["Year"] <= current_year)].copy()
        eys["education_index"] = (pd.to_numeric(eys["eys"], errors="coerce") / 18.0).clip(upper=1)
        return eys[key_cols + ["education_index"]]

    return None


def _compute_safety_index_from_homicide(last_n_years: int) -> Optional[pd.DataFrame]:
    """Compute a safety proxy by inverting homicide rates (lower homicide => higher safety, 0..1).

    safety_index = (max_homicide_y - homicide) / (max_homicide_y - min_homicide_y)
    computed per-year to avoid period bias.
    """
    homicide_slugs = [
        "homicide-rate",
        "intentional-homicides-per-100000",
        "unodc-intentional-homicide-rate",
        "intentional-homicide-victims-per-100000",
    ]

    df = _fetch_metric_prefer_first_available(homicide_slugs)
    if (df is None or df.empty):
        # World Bank fallback: Intentional homicides (per 100,000 people)
        wb_homicide = _fetch_worldbank_indicator("VC.IHR.PSRC.P5")
        if wb_homicide is None or wb_homicide.empty:
            logger.warning("No homicide data found from World Bank fallback for safety_index.")
            return None
        logger.info("Using World Bank homicide data as fallback for safety_index.")
        df = wb_homicide
    else:
        logger.info("Using Grapher homicide data for safety_index.")

    current_year = datetime.utcnow().year
    year_min = current_year - (last_n_years - 1)

    key_cols = ["Entity", "Code", "Year"]
    df = df[(df["Year"] >= year_min) & (df["Year"] <= current_year)].copy()
    df = df[df["Code"].notna()]
    df = df[~df["Code"].astype(str).str.startswith("OWID_")]

    # Compute min/max per year and invert scale
    df = df.rename(columns={"value": "homicide_rate"})
    grouped = df.groupby("Year")["homicide_rate"].agg(["min", "max"]).reset_index()
    grouped.columns = ["Year", "min_rate", "max_rate"]

    df = pd.merge(df, grouped, on="Year", how="left")
    span = (df["max_rate"] - df["min_rate"]).replace(0, np.nan)
    df["safety_index"] = (df["max_rate"] - df["homicide_rate"]) / span
    df["safety_index"] = df["safety_index"].clip(lower=0, upper=1)

    return df[key_cols + ["safety_index"]]


def get_country_metrics(last_n_years: int = 10) -> pd.DataFrame:
    """Get merged metrics for all countries for the last N years.

    Returns a DataFrame with columns:
      - Entity (country name)
      - Code (ISO alpha-3 where available)
      - Year
      - gdp_ppp_per_capita, life_expectancy, education_index, safety_index,
        prosperity_index, freedom_of_speech_index, democracy_index, corruption_index

    Missing series will be left as NaN.
    Regional aggregates (OWID_*) are excluded.
    """
    current_year = datetime.utcnow().year
    year_min = current_year - (last_n_years - 1)

    merged: Optional[pd.DataFrame] = None

    for metric_key, slugs in METRICS_SLUGS.items():
        # education_index is computed separately; skip here
        if metric_key == "education_index":
            continue

        df = _fetch_metric_prefer_first_available(slugs) if slugs else None
        if df is None:
            # Try WB fallback if available for this metric
            wb_code = WB_FALLBACKS.get(metric_key)
            if wb_code:
                df = _fetch_worldbank_indicator(wb_code)
        if df is None:
            logger.warning("No data found for metric %s (tried: %s)", metric_key, slugs)
            continue

        # Filter years first
        df = df[(df["Year"] >= year_min) & (df["Year"] <= current_year)].copy()
        # Drop aggregates and rows without a proper country code
        df = df[df["Code"].notna()]
        df = df[~df["Code"].astype(str).str.startswith("OWID_")]

        # Rename value column to metric_key
        df = df.rename(columns={"value": metric_key})

        # Merge into the final DataFrame
        key_cols = ["Entity", "Code", "Year"]
        if merged is None:
            merged = df[key_cols + [metric_key]].copy()
        else:
            merged = pd.merge(merged, df[key_cols + [metric_key]], on=key_cols, how="outer")

    # Compute and merge education index if possible
    edu = _compute_education_index(last_n_years=last_n_years)
    if edu is not None and not edu.empty:
        key_cols = ["Entity", "Code", "Year"]
        if merged is None:
            merged = edu[key_cols + ["education_index"]].copy()
        else:
            merged = pd.merge(merged, edu[key_cols + ["education_index"]], on=key_cols, how="outer")


    # Fallback: compute safety_index from homicide if not fetched
    if merged is not None and "safety_index" not in merged.columns:
        safety_df = _compute_safety_index_from_homicide(last_n_years=last_n_years)
        if safety_df is not None and not safety_df.empty:
            key_cols = ["Entity", "Code", "Year"]
            merged = pd.merge(merged, safety_df[key_cols + ["safety_index"]], on=key_cols, how="outer")

    # Fallback: use historical Freedom House data for freedom_of_speech_index and democracy_index
    if merged is not None:
        fh_df = _fetch_freedom_house_historical()
        if fh_df is not None and not fh_df.empty:
            key_cols = ["Entity", "Year"]
            
            # Merge freedom_of_speech_index if missing
            if "freedom_of_speech_index" not in merged.columns or merged["freedom_of_speech_index"].isna().all():
                fh_freedom = fh_df[key_cols + ["freedom_of_speech_index"]].dropna()
                merged = pd.merge(merged, fh_freedom, on=key_cols, how="left", suffixes=('', '_fh'))
                if "freedom_of_speech_index_fh" in merged.columns:
                    merged["freedom_of_speech_index"] = merged["freedom_of_speech_index"].fillna(merged["freedom_of_speech_index_fh"])
                    merged = merged.drop(columns=["freedom_of_speech_index_fh"])
            
            # Merge democracy_index if missing  
            if "democracy_index" not in merged.columns or merged["democracy_index"].isna().all():
                fh_democracy = fh_df[key_cols + ["democracy_index"]].dropna()
                merged = pd.merge(merged, fh_democracy, on=key_cols, how="left", suffixes=('', '_fh'))
                if "democracy_index_fh" in merged.columns:
                    merged["democracy_index"] = merged["democracy_index"].fillna(merged["democracy_index_fh"])
                    merged = merged.drop(columns=["democracy_index_fh"])

    # Fallback: use Transparency International CPI for corruption_index
    if merged is not None and ("corruption_index" not in merged.columns or merged["corruption_index"].isna().all()):
        ti_df = _fetch_transparency_international_historical()
        if ti_df is not None and not ti_df.empty:
            key_cols = ["Entity", "Year"]
            merged = pd.merge(merged, ti_df[key_cols + ["corruption_index"]], on=key_cols, how="left", suffixes=('', '_ti'))
            if "corruption_index_ti" in merged.columns:
                merged["corruption_index"] = merged["corruption_index"].fillna(merged["corruption_index_ti"])
                merged = merged.drop(columns=["corruption_index_ti"])

    # Fallback: use RSF Press Freedom Index for freedom_of_speech_index if not fetched
    if merged is not None and ("freedom_of_speech_index" not in merged.columns or merged["freedom_of_speech_index"].isna().all()):
        rsf_df = _fetch_rsf_press_freedom_2025()
        if rsf_df is not None and not rsf_df.empty:
            # Map RSF score to freedom_of_speech_index (normalize 0-100 to 0-1, invert: higher score = more freedom)
            rsf_df = rsf_df.rename(columns={"press_freedom_score": "freedom_of_speech_index"})
            rsf_df["freedom_of_speech_index"] = rsf_df["freedom_of_speech_index"] / 100.0
            key_cols = ["Entity", "Year"]
            merged = pd.merge(merged, rsf_df[key_cols + ["freedom_of_speech_index"]], on=["Entity", "Year"], how="left")

    if merged is None:
        # Return empty DataFrame with expected columns
        cols = ["Entity", "Code", "Year"] + list(METRICS_SLUGS.keys())
        return pd.DataFrame(columns=cols)

    # Sort for determinism
    merged = merged.sort_values(["Code", "Year"]).reset_index(drop=True)

    # Drop rows where all available metric columns are missing
    metric_cols = [c for c in METRICS_SLUGS.keys() if c in merged.columns]
    if metric_cols:
        merged = merged.dropna(how="all", subset=metric_cols).reset_index(drop=True)

    return merged


def _json_sanitize_records(df: pd.DataFrame) -> List[Dict]:
    """Convert a DataFrame to JSON-serializable records by replacing NaN/NA/inf with None."""
    # Work on object dtype to avoid pandas restoring special NA scalars on export
    df = df.astype(object)
    # Replace infinities first
    df = df.replace({np.inf: None, -np.inf: None})
    # Replace any remaining NA-like values with None
    df = df.where(pd.notna(df), None)

    records: List[Dict] = df.to_dict(orient="records")
    # Final pass to catch any lingering float('nan') that may survive
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float):
                if math.isnan(v) or v == float("inf") or v == float("-inf"):
                    rec[k] = None
    return records


def get_country_metrics_json(last_n_years: int = 10) -> List[Dict]:
    """Convenience wrapper: return list of records suitable for JSON response."""
    df = get_country_metrics(last_n_years=last_n_years)
    if df.empty:
        return []
    # Standardize keys for frontend
    df = df.rename(columns={
        "Entity": "country",
        "Code": "code",
        "Year": "year",
    })
    # Ensure JSON-safe output
    return _json_sanitize_records(df)


# Backward compatibility with the initial skeleton
# You can remove this later if not needed by other modules.
def get_charts(last_n_years: int = 10):
    """Kept for compatibility: returns the same as get_country_metrics_json."""
    return get_country_metrics_json(last_n_years=last_n_years)


# =========================
# Persistence helpers
# =========================

def _split_records(records: List[Dict]) -> Tuple[Dict[str, str], List[Dict]]:
    """Collect countries {code: name} and metric rows from JSON-like records."""
    countries: Dict[str, str] = {}
    rows: List[Dict] = []
    for r in records:
        code = (r.get("code") or "").strip()
        name = (r.get("country") or "").strip()
        year = r.get("year")
        # Ensure code is a valid 3-char ISO code before processing
        if not code or len(code) != 3 or not isinstance(year, (int, float)):
            continue
        if name:
            countries[code] = name
        metric = {
            "code": code,
            "year": int(year),
            "gdp_ppp_per_capita": r.get("gdp_ppp_per_capita"),
            "life_expectancy": r.get("life_expectancy"),
            "education_index": r.get("education_index"),
            "safety_index": r.get("safety_index"),
            "prosperity_index": r.get("prosperity_index"),
            "freedom_of_speech_index": r.get("freedom_of_speech_index"),
            "democracy_index": r.get("democracy_index"),
            "corruption_index": r.get("corruption_index"),
        }
        rows.append(metric)
    return countries, rows


def upsert_metrics_from_records(records: List[Dict]) -> int:
    """Bulk upsert Country and CountryMetric from JSON-like records.

    Returns number of CountryMetric rows inserted or updated.
    """
    countries_map, rows = _split_records(records)
    if not rows:
        return 0

    Country, CountryMetric = _get_models()

    # Upsert countries (separate transaction scope)
    existing = {c.code: c for c in Country.objects.filter(code__in=list(countries_map.keys()))}
    to_create = [
        Country(code=code, name=name)
        for code, name in countries_map.items()
        if code not in existing
    ]
    if to_create:
        # Safe to bulk create without wrapping in outer atomic
        Country.objects.bulk_create(to_create, ignore_conflicts=True)

    # Refresh mapping to include newly created
    countries_qs = Country.objects.filter(code__in=list(countries_map.keys()))
    code_to_country = {c.code: c for c in countries_qs}

    # Build CountryMetric objects
    metric_objs: List[Any] = []
    for r in rows:
        country = code_to_country.get(r["code"])  # type: ignore[index]
        if not country:
            continue
        metric_objs.append(
            CountryMetric(
                country=country,
                year=r["year"],
                gdp_ppp_per_capita=r.get("gdp_ppp_per_capita"),
                life_expectancy=r.get("life_expectancy"),
                education_index=r.get("education_index"),
                safety_index=r.get("safety_index"),
                prosperity_index=r.get("prosperity_index"),
                freedom_of_speech_index=r.get("freedom_of_speech_index"),
                democracy_index=r.get("democracy_index"),
                corruption_index=r.get("corruption_index"),
            )
        )

    if not metric_objs:
        return 0

    # Use bulk upsert if supported (PostgreSQL). Scope in its own atomic block so we can
    # safely fallback without leaving the connection in a broken transaction state.
    try:
        with transaction.atomic():
            CountryMetric.objects.bulk_create(
                metric_objs,
                update_conflicts=True,
                update_fields=[
                    "gdp_ppp_per_capita",
                    "life_expectancy",
                    "education_index",
                    "safety_index",
                    "prosperity_index",
                    "freedom_of_speech_index",
                    "democracy_index",
                    "corruption_index",
                    "updated_at",
                ],
                unique_fields=["country", "year"],
            )
            return len(metric_objs)
    except Exception as e:
        logger.warning("bulk upsert not supported, falling back to per-row upsert: %s", e)

    # Fallback: update_or_create per row in a clean transaction context
    count = 0
    for obj in metric_objs:
        CountryMetric.objects.update_or_create(
            country=obj.country,
            year=obj.year,
            defaults={
                "gdp_ppp_per_capita": obj.gdp_ppp_per_capita,
                "life_expectancy": obj.life_expectancy,
                "education_index": obj.education_index,
                "safety_index": obj.safety_index,
                "prosperity_index": obj.prosperity_index,
                "freedom_of_speech_index": obj.freedom_of_speech_index,
                "democracy_index": obj.democracy_index,
                "corruption_index": obj.corruption_index,
            },
        )
        count += 1
    return count


def get_metrics_from_db(
    last_n_years: int = 10,
    code_filter: Optional[str] = None,
    strict_not_null: bool = False,
    any_null: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch records from the DB, optionally filtering.

    Args:
        last_n_years: How many recent years of data to fetch.
        code_filter: Optional ISO-3 country code to filter by.
        strict_not_null: If True, only return records where all key metrics are non-null.
        any_null: If True, only return records where at least one key metric is null.

    Returns:
        A list of dictionaries, each representing a country-year metric record.
    """
    current_year = datetime.utcnow().year
    year_min = current_year - (last_n_years - 1)

    _, CountryMetric = _get_models()
    
    qs = (
        CountryMetric.objects.select_related("country")
        .filter(year__gte=year_min, year__lte=current_year)
    )
    if code_filter:
        qs = qs.filter(country__code=code_filter)

    if strict_not_null:
        # All metric fields must be non-null
        for metric in METRICS_SLUGS:
            if metric:
                qs = qs.filter(**{f"{metric}__isnull": False})

    if any_null:
        # Any metric field is null
        null_filter = Q()
        for metric in METRICS_SLUGS:
            if metric:
                null_filter |= Q(**{f"{metric}__isnull": True})
        qs = qs.filter(null_filter)

    # Order by country name and year descending
    qs = qs.order_by("country__name", "-year")

    values_qs = qs.values(
        "year",
        "gdp_ppp_per_capita",
        "life_expectancy",
        "education_index",
        "safety_index",
        "prosperity_index",
        "freedom_of_speech_index",
        "democracy_index",
        "corruption_index",
        code=F("country__code"),
        country_name=F("country__name"),
    ).order_by("country__code", "year")

    try:
        data = list(values_qs)
    except DataError as e:  # pragma: no cover
        logger.error("DataError fetching metrics: %s", e)
        return []

    df = pd.DataFrame(data)
    if df.empty:
        return []
    df = df.rename(columns={"country_name": "country"})
    return _json_sanitize_records(df)


def refresh_remote_and_persist(last_n_years: int = 10) -> int:
    """Fetch latest metrics from remote sources and persist them. Returns upsert count."""
    records = get_country_metrics_json(last_n_years=last_n_years)
    return upsert_metrics_from_records(records)