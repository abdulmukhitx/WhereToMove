from django.contrib import admin
from .models import Country, CountryMetric


@admin.register(Country)
class CountryAdmin(admin.ModelAdmin):  # type: ignore[misc]
    list_display = ("code", "name")
    search_fields = ("code", "name")


@admin.register(CountryMetric)
class CountryMetricAdmin(admin.ModelAdmin):  # type: ignore[misc]
    list_display = ("country", "year", "gdp_ppp_per_capita", "life_expectancy")
    list_filter = ("year",)
    search_fields = ("country__code", "country__name")
