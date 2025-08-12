from django.db import models


class Country(models.Model):
    """A country identified by ISO-3 code."""
    code = models.CharField(max_length=3, unique=True, db_index=True)
    name = models.CharField(max_length=255)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.code} - {self.name}"


class CountryMetric(models.Model):
    """Yearly metrics per country for the indicators served by the API."""
    country = models.ForeignKey(Country, on_delete=models.CASCADE, related_name="metrics")
    year = models.IntegerField(db_index=True)

    # Metrics (all optional)
    gdp_ppp_per_capita = models.FloatField(null=True, blank=True)
    life_expectancy = models.FloatField(null=True, blank=True)
    education_index = models.FloatField(null=True, blank=True)
    safety_index = models.FloatField(null=True, blank=True)
    prosperity_index = models.FloatField(null=True, blank=True)
    freedom_of_speech_index = models.FloatField(null=True, blank=True)
    democracy_index = models.FloatField(null=True, blank=True)
    corruption_index = models.FloatField(null=True, blank=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["country", "year"], name="uniq_country_year"),
        ]
        indexes = [
            models.Index(fields=["country", "year"], name="idx_country_year"),
        ]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.country.code} {self.year}"
