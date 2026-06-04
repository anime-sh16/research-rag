from src.config.config import settings


def test_api_rate_limit_defaults():
    assert settings.api.rate_limit_per_minute == 5
    assert settings.api.rate_limit_per_day == 50
    assert settings.api.rate_limit_enabled is True


def test_api_query_length_defaults():
    assert settings.api.query_min_length == 3
    assert settings.api.query_max_length == 1000
