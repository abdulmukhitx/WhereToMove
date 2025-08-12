from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import math

from django.core.cache import cache

from .services import (
    get_metrics_from_db,
    refresh_remote_and_persist,
)


CACHE_KEY_ALL = "country_metrics:{years}:{code}"
CACHE_TIMEOUT = 60 * 60  # 1 hour


def _sanitize_records_for_json(data):
    """Extra safety: replace any lingering NaN/±inf with None in list[dict]."""
    if not isinstance(data, list):
        return data
    for rec in data:
        if isinstance(rec, dict):
            for k, v in list(rec.items()):
                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        rec[k] = None
    return data


@api_view(["GET"])
def country_metrics(request):
    """Return metrics from DB/cache. If missing or refresh=1, fetch remote, persist, then serve.

    Query params:
      - years: integer, number of recent years to include (default: 10)
      - code: ISO-3 country code to filter results (optional)
      - refresh: set to 1/true to force remote re-fetch and cache refresh (optional)
    """
    years_param = request.query_params.get("years")
    code_filter = request.query_params.get("code")
    refresh_flag = str(request.query_params.get("refresh", "0")).lower() in ("1", "true", "yes")

    try:
        last_n_years = int(years_param) if years_param is not None else 10
        # simple guardrails
        if last_n_years <= 0 or last_n_years > 100:
            last_n_years = 10
    except (TypeError, ValueError):
        last_n_years = 10

    cache_key = CACHE_KEY_ALL.format(years=last_n_years, code=(code_filter or "*").upper())

    if refresh_flag:
        # Force refresh remote + DB and override cache
        refresh_remote_and_persist(last_n_years=last_n_years)
        data = get_metrics_from_db(last_n_years=last_n_years, code_filter=(code_filter or None))
        cache.set(cache_key, data, CACHE_TIMEOUT)
    else:
        data = cache.get(cache_key)
        if data is None:
            # Try DB first
            data = get_metrics_from_db(last_n_years=last_n_years, code_filter=(code_filter or None))
            if not data:
                # Fetch remote, persist, then load from DB
                refresh_remote_and_persist(last_n_years=last_n_years)
                data = get_metrics_from_db(last_n_years=last_n_years, code_filter=(code_filter or None))
            cache.set(cache_key, data, CACHE_TIMEOUT)

    if code_filter:
        code_filter = code_filter.strip().upper()
        data = [row for row in data if str(row.get("code", "")).upper() == code_filter]

    data = _sanitize_records_for_json(data)

    return Response(data, status=status.HTTP_200_OK)


@api_view(["GET"])  # Serve only from DB/cache, never fetch remote
def country_metrics_db_only(request):
    years_param = request.query_params.get("years")
    code_filter = request.query_params.get("code")

    try:
        last_n_years = int(years_param) if years_param is not None else 10
        if last_n_years <= 0 or last_n_years > 100:
            last_n_years = 10
    except (TypeError, ValueError):
        last_n_years = 10

    cache_key = CACHE_KEY_ALL.format(years=last_n_years, code=(code_filter or "*").upper())
    data = cache.get(cache_key)

    if data is None:
        data = get_metrics_from_db(last_n_years=last_n_years, code_filter=(code_filter or None))
        cache.set(cache_key, data, CACHE_TIMEOUT)

    if code_filter:
        code_filter = code_filter.strip().upper()
        data = [row for row in data if str(row.get("code", "")).upper() == code_filter]

    data = _sanitize_records_for_json(data)
    return Response(data, status=status.HTTP_200_OK)


@api_view(["POST"])  # Admin/ops: Refresh remote sources and update DB + cache
def refresh_metrics(request):
    years_param = request.query_params.get("years")
    try:
        last_n_years = int(years_param) if years_param is not None else 10
        if last_n_years <= 0 or last_n_years > 100:
            last_n_years = 10
    except (TypeError, ValueError):
        last_n_years = 10

    count = refresh_remote_and_persist(last_n_years=last_n_years)

    # Bust caches related to this years window
    try:
        cache.delete_pattern(CACHE_KEY_ALL.format(years=last_n_years, code="*"))
    except Exception:
        pass

    return Response({"updated": count}, status=status.HTTP_200_OK)


@api_view(["GET"])
def country_metrics_null(request):
    """Return metric records from DB/cache where at least one metric is null.

    Query params:
      - years: integer, number of recent years to include (default: 10)
      - code: ISO-3 country code to filter results (optional)
    """
    years_param = request.query_params.get("years")
    code_filter = request.query_params.get("code")

    try:
        last_n_years = int(years_param) if years_param is not None else 10
        if last_n_years <= 0 or last_n_years > 100:
            last_n_years = 10
    except (TypeError, ValueError):
        last_n_years = 10

    # Use a distinct cache key for null-filtered data
    cache_key = f"country_metrics_null:{last_n_years}:{(code_filter or '*').upper()}"
    data = cache.get(cache_key)

    if data is None:
        # Fetch from DB with the any_null=True filter
        data = get_metrics_from_db(
            last_n_years=last_n_years,
            code_filter=(code_filter or None),
            any_null=True,
        )
        cache.set(cache_key, data, CACHE_TIMEOUT)

    if code_filter:
        code_filter = code_filter.strip().upper()
        data = [row for row in data if str(row.get("code", "")).upper() == code_filter]

    data = _sanitize_records_for_json(data)

    return Response(data, status=status.HTTP_200_OK)


@api_view(["POST"])
def refresh_country_metrics(request):
    """Force a remote fetch and refresh the database and cache."""
    years_param = request.query_params.get("years")
    try:
        last_n_years = int(years_param) if years_param is not None else 10
        if last_n_years <= 0 or last_n_years > 100:
            last_n_years = 10
    except (TypeError, ValueError):
        last_n_years = 10

    count = refresh_remote_and_persist(last_n_years=last_n_years)

    # Bust caches related to this years window
    try:
        cache.delete_pattern(CACHE_KEY_ALL.format(years=last_n_years, code="*"))
        cache.delete_pattern(f"country_metrics_null:{last_n_years}:*")
    except Exception:
        pass

    return Response({"updated": count}, status=status.HTTP_200_OK)
