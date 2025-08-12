from django.urls import path

from .views import (
    country_metrics,
    country_metrics_db_only,
    refresh_country_metrics,
    country_metrics_null,
)

app_name = 'countries'

urlpatterns = [
    path('country-metrics/', country_metrics, name='country-metrics'),
    path('country-metrics/db-only/', country_metrics_db_only, name='country-metrics-db-only'),
    path('country-metrics/refresh/', refresh_country_metrics, name='refresh-country-metrics'),
    path('country-metrics/null/', country_metrics_null, name='country-metrics-null'),
]
