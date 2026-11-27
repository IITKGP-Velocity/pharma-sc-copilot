"""Deterministic tools return the numbers the eval set expects."""
import pytest

from copilot.agent import tools


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


def test_lookup_finds_approved_products_with_te_codes():
    out = tools.run_tool("lookup_approved_products", {"drug": "Cisplatin", "dosage_form": "injectable"})
    assert 0 < len(out["rows"]) <= 5 <= out["rows_found"]
    assert {"application_number", "sponsor_name", "te_code", "reference_drug"} <= set(out["rows"][0])
    assert out["rows"][0]["marketing_status"] != "Discontinued"


def test_lookup_dosage_form_narrows_the_search():
    every_form = tools.run_tool("lookup_approved_products", {"drug": "morphine sulfate"})["rows_found"]
    tablets = tools.run_tool("lookup_approved_products", {"drug": "morphine sulfate", "dosage_form": "tablet"})
    assert 0 < tablets["rows_found"] < every_form


def test_lookup_returns_no_rows_for_a_drug_not_in_the_table():
    # B01 asks about amlodipine, which the table does not hold: the agent must abstain, not guess.
    assert tools.run_tool("lookup_approved_products", {"drug": "amlodipine"})["rows_found"] == 0


def test_lookup_treats_sql_in_the_argument_as_plain_text():
    out = tools.run_tool("lookup_approved_products", {"drug": "cisplatin' OR '1'='1"})
    assert out["rows_found"] == 0
