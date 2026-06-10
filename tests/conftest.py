import pytest


@pytest.fixture(params=["sentinel1", "sentinel2"])
def sensor(request) -> str:
    """Run STURM tests for both supported sensors."""
    return request.param


@pytest.fixture
def max_samples() -> int:
    """Keep integration-style exporter tests small and fast."""
    return 2


@pytest.fixture(params=["binary_water", "original"])
def mode(request) -> str:
    """Run mask exporter tests for both supported mask modes."""
    return request.param
