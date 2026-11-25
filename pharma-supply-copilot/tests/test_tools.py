import pytest

from copilot import tools


def test_mkt_example_from_spec_is_about_4_6_c():
    assert tools.mkt_celsius([2.1, 3.4, 9.8, 4.0, 2.5, 3.0]) == pytest.approx(4.6, abs=0.2)


def test_mkt_is_not_the_arithmetic_mean():
    temps = [2.1, 3.4, 9.8, 4.0, 2.5, 3.0]
    assert abs(tools.mkt_celsius(temps) - sum(temps) / len(temps)) > 0.3


def test_mkt_constant_temperature_returns_that_temperature():
    assert tools.mkt_celsius([5, 5, 5]) == pytest.approx(5.0, abs=0.01)


def test_mkt_empty_raises():
    with pytest.raises(ValueError):
        tools.mkt_celsius([])


def test_run_tool_dispatch_and_missing_args():
    assert tools.run_tool("demurrage", {"days_late": 3, "rate_per_day": 150}) == 450
    with pytest.raises(ValueError):
        tools.run_tool("demurrage", {"days_late": 3})
    with pytest.raises(ValueError):
        tools.run_tool("delete_everything", {})


def test_impurity_pass_and_fail():
    assert tools.impurity_passes(0.18, 0.20) is True
    assert tools.impurity_passes(0.42, 0.20) is False
    assert tools.impurity_passes(0.11, 0.10) is False


def test_cost_delta_and_markup():
    assert tools.cost_delta(12.40, 1.10, 1000)["delta"] == 11300.0
    assert tools.markup_percent(48, 1.80) == pytest.approx(2566.67, abs=0.01)


def test_quarantine_and_manifest():
    assert tools.quarantine_cost(40_000, 6.50) == 260_000
    assert tools.manifest_mismatch(1200, 1140) == 60
