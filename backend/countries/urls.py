from django.urls import path
from . import views

app_name = 'countries'

urlpatterns = [
    path('country-metrics/', views.country_metrics, name='country-metrics'),
    path('country-metrics/refresh/', views.refresh_metrics, name='country-metrics-refresh'),
    path('country-metrics/db-only/', views.country_metrics_db_only, name='country-metrics-db-only'),
]
