from django.test import TestCase
from .services import upsert_metrics_from_records, get_metrics_from_db


class MetricsPersistenceTests(TestCase):
    def test_upsert_and_fetch(self):
        sample = [
            {
                "country": "Testland",
                "code": "TST",
                "year": 2020,
                "gdp_ppp_per_capita": 12345.0,
                "life_expectancy": 70.5,
                "education_index": 0.7,
                "safety_index": 0.8,
                "prosperity_index": 0.65,
                "freedom_of_speech_index": 0.1,
                "democracy_index": 0.2,
                "corruption_index": -0.3,
            }
        ]
        count = upsert_metrics_from_records(sample)
        self.assertEqual(count, 1)
        data = get_metrics_from_db(last_n_years=10, code_filter="TST")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["code"], "TST")
        self.assertEqual(data[0]["year"], 2020)
