"""Deterministic tools. The model chooses a tool and its arguments; these functions do the arithmetic
and the table lookups, so numbers and identifiers never come from the model's memory."""
import csv
import math
import sqlite3

from copilot.config import ROOT

R_J_PER_MOL_K = 8.314

# FDA approved-products table (Drugs@FDA, public domain): one row per approved product, with its TE code.
PRODUCTS_CSV = ROOT / "data" / "tables" / "fda_approved_products.csv"
_DB = None  # in-memory SQLite copy of the table, built on first use


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


def _products_db() -> sqlite3.Connection:
    """Load the CSV into an in-memory SQLite table once, then reuse it."""
    global _DB
    if _DB is None:
        with PRODUCTS_CSV.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        columns = list(rows[0].keys())  # column names come from our own CSV header, not from the model
        _DB = sqlite3.connect(":memory:", check_same_thread=False)
        _DB.row_factory = sqlite3.Row  # rows can then be turned into dicts
        _DB.execute(f"CREATE TABLE products ({', '.join(columns)})")
        _DB.executemany(f"INSERT INTO products VALUES ({', '.join('?' * len(columns))})",
                        [[r[c] for c in columns] for r in rows])
    return _DB


# Fixed SQL. The model's words only fill the two ? placeholders, so it can never change the query.
LOOKUP_SQL = """
    SELECT application_number, sponsor_name, active_ingredients, marketing_status, te_code, reference_drug
    FROM products
    WHERE lower(drug) = lower(?)
      AND lower(dosage_form) LIKE '%' || lower(?) || '%'
    ORDER BY marketing_status = 'Discontinued', application_number
"""  # the ORDER BY puts products still on the market first


def lookup_approved_products(drug: str, dosage_form: str = "") -> dict:
    """Find FDA-approved products for a drug: applicant, TE code, reference drug and marketing status.
    No rows means the drug is not in the table, so the agent must abstain rather than guess."""
    rows = [dict(r) for r in _products_db().execute(LOOKUP_SQL, (str(drug).strip(), str(dosage_form).strip()))]
    # Return the count plus the first 5 rows only, to keep the planner's next prompt short.
    return {"source": "FDA Drugs@FDA products table", "rows_found": len(rows), "rows": rows[:5]}


# Tool registry: name -> (function, required argument names). The agent may only call these.
TOOLS = {
    "mkt": (mkt_celsius, ["temps_c"]),
    "demurrage": (demurrage_total, ["days_late", "rate_per_day"]),
    "impurity_passes": (impurity_passes, ["value_pct", "limit_pct"]),
    "cost_delta": (cost_delta, ["brand_unit", "generic_unit", "units"]),
    "markup_percent": (markup_percent, ["price", "reference"]),
    "quarantine_cost": (quarantine_cost, ["units", "unit_cost"]),
    "manifest_mismatch": (manifest_mismatch, ["declared_units", "invoice_units"]),
    "lookup_approved_products": (lookup_approved_products, ["drug"]),
}

# Extra arguments a tool accepts when the model supplies them.
OPTIONAL = {"lookup_approved_products": ["dosage_form"]}


def run_tool(name: str, args: dict):
    """Run a registered tool. Raises ValueError for unknown tools or missing arguments."""
    if name not in TOOLS:
        raise ValueError(f"unknown tool: {name}")
    func, required = TOOLS[name]
    missing = [a for a in required if a not in args]
    if missing:
        raise ValueError(f"tool {name} missing arguments: {missing}")
    allowed = required + OPTIONAL.get(name, [])
    return func(**{a: args[a] for a in allowed if a in args})
