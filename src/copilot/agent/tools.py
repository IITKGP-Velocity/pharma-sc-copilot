"""Deterministic tools. The model chooses a tool and its arguments; these functions do the arithmetic."""
import math

R_J_PER_MOL_K = 8.314


def mkt_celsius(temps_c: list[float], delta_h_kj: float = 83.144) -> float:
    """Haynes Mean Kinetic Temperature in degrees Celsius: (dH/R) / -ln(mean(exp(-dH/(R*T_i))))."""
    if not temps_c:
        raise ValueError("no readings")
    ratio = (delta_h_kj * 1000.0) / R_J_PER_MOL_K
    kelvin = [t + 273.15 for t in temps_c]
    mean_exp = sum(math.exp(-ratio / t) for t in kelvin) / len(kelvin)
    return ratio / -math.log(mean_exp) - 273.15


def demurrage_total(days_late: float, rate_per_day: float) -> float:
    """Port fee for a late shipment: days late x daily rate."""
    return days_late * rate_per_day


def impurity_passes(value_pct: float, limit_pct: float) -> bool:
    """True when the measured impurity is at or below the limit (both in percent)."""
    return value_pct <= limit_pct


def cost_delta(brand_unit: float, generic_unit: float, units: int) -> dict:
    """Brand vs generic cost for a number of units: both totals, the difference and the price ratio."""
    brand_total = brand_unit * units
    generic_total = generic_unit * units
    return {"brand_total": round(brand_total, 2), "generic_total": round(generic_total, 2),
            "delta": round(brand_total - generic_total, 2), "ratio": round(brand_unit / generic_unit, 2)}


def markup_percent(price: float, reference: float) -> float:
    """How far a price is above a reference price, in percent (e.g. grey-market vs NADAC)."""
    return (price - reference) / reference * 100.0


def quarantine_cost(units: int, unit_cost: float) -> float:
    """Value of stock held in quarantine: units x unit cost."""
    return units * unit_cost


def manifest_mismatch(declared_units: int, invoice_units: int) -> int:
    """Units on the manifest minus units on the invoice (0 means they match)."""
    return declared_units - invoice_units


# Tool registry: name -> (function, required argument names). The agent may only call these.
TOOLS = {
    "mkt": (mkt_celsius, ["temps_c"]),
    "demurrage": (demurrage_total, ["days_late", "rate_per_day"]),
    "impurity_passes": (impurity_passes, ["value_pct", "limit_pct"]),
    "cost_delta": (cost_delta, ["brand_unit", "generic_unit", "units"]),
    "markup_percent": (markup_percent, ["price", "reference"]),
    "quarantine_cost": (quarantine_cost, ["units", "unit_cost"]),
    "manifest_mismatch": (manifest_mismatch, ["declared_units", "invoice_units"]),
}


def run_tool(name: str, args: dict):
    """Run a registered tool. Raises ValueError for unknown tools or missing arguments."""
    if name not in TOOLS:
        raise ValueError(f"unknown tool: {name}")
    func, required = TOOLS[name]
    missing = [a for a in required if a not in args]
    if missing:
        raise ValueError(f"tool {name} missing arguments: {missing}")
    return func(**{a: args[a] for a in required})
