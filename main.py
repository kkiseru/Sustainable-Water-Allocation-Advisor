# file: main.py  (drop-in full script)
from __future__ import annotations
import argparse, csv, math, os, sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# --------------------------
# Utilities & Data Handling
# --------------------------

DATA_PATH = os.path.join("dataset", "cleaned_global_water_consumption.csv")

SCARCITY_MAP = {"Low": 0.0, "Moderate": 1.0, "High": 2.0, "Severe": 3.0}

def scarcity_to_idx(s: str) -> float:
    s_norm = (s or "").strip().title()
    return SCARCITY_MAP.get(s_norm, SCARCITY_MAP["Moderate"])

@dataclass
class Row:
    country: str
    year: int
    rainfall_mm: float
    gw_depl_pct: float
    scarcity_idx: float
    ag_pct: float
    ind_pct: float
    hh_pct: float

def read_dataset(path: str = DATA_PATH) -> List[Row]:
    rows: List[Row] = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(Row(
                    country=r["Country"].strip(),
                    year=int(float(r["Year"])),
                    rainfall_mm=float(r["Rainfall Impact (Annual Precipitation in mm)"]),
                    gw_depl_pct=float(r["Groundwater Depletion Rate (%)"]),
                    scarcity_idx=scarcity_to_idx(r.get("Water Scarcity Level", "")),
                    ag_pct=float(r["Agricultural Water Use (%)"]),
                    ind_pct=float(r["Industrial Water Use (%)"]),
                    hh_pct=float(r["Household Water Use (%)"]),
                ))
            except Exception:
                continue
    return rows

def find_row(rows: List[Row], country: str, year: int) -> Optional[Row]:
    for r in rows:
        if r.country.lower() == country.lower() and r.year == year:
            return r
    return None

# --------------------------
# Fuzzy Primitives
# --------------------------

MF = Callable[[float], float]

def trap(a: float, b: float, c: float, d: float) -> MF:
    def f(x: float) -> float:
        if x <= a or x >= d: return 0.0
        if b <= x <= c: return 1.0
        if a < x < b: return (x - a) / (b - a) if b != a else 0.0
        return (d - x) / (d - c) if d != c else 0.0
    return f

def tri(a: float, b: float, c: float) -> MF:
    def f(x: float) -> float:
        if x <= a or x >= c: return 0.0
        if x == b: return 1.0
        if a < x < b: return (x - a) / (b - a) if b != a else 0.0
        return (c - x) / (c - b) if c != b else 0.0
    return f

def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

# --------------------------
# Universes & Linguistic Terms
# --------------------------

class Universe:
    def __init__(self, start: float, end: float, step: float = 0.5):
        self.start = start; self.end = end; self.step = step
        self.points = self._build_points()
    def _build_points(self) -> List[float]:
        n = int((self.end - self.start)/self.step) + 1
        return [self.start + i*self.step for i in range(n)]

U_RAIN = Universe(0.0, 3000.0, 1.0)
U_GW   = Universe(0.0, 10.0, 0.05)
U_SCAR = Universe(0.0, 3.0, 0.01)
U_PCT  = Universe(0.0, 100.0, 0.5)

RAIN = {
    "very_low":  trap(0, 0, 200, 500),
    "low":       tri(300, 600, 900),
    "medium":    tri(800, 1200, 1600),
    "high":      tri(1400, 1800, 2200),
    "very_high": trap(2000, 2400, 3000, 3000),
}
GW = {
    "low":      trap(0, 0, 1.5, 2.5),
    "moderate": tri(2.0, 3.5, 5.0),
    "high":     tri(4.0, 6.0, 8.0),
    "critical": trap(7.0, 8.5, 10.0, 10.0),
}
SCAR = {
    "low":      trap(0.0, 0.0, 0.2, 0.7),
    "moderate": tri(0.6, 1.0, 1.6),
    "high":     tri(1.4, 2.0, 2.6),
    "severe":   trap(2.4, 2.8, 3.0, 3.0),
}
DEMAND = {
    "low":    trap(0, 0, 20, 40),
    "medium": tri(30, 50, 70),
    "high":   trap(60, 80, 100, 100),
}
ALLOC = {
    "very_low":  trap(0, 0, 8, 15),
    "low":       tri(10, 20, 30),
    "medium":    tri(25, 40, 55),
    "high":      tri(50, 65, 80),
    "very_high": trap(75, 85, 100, 100),
}

# --------------------------
# Rule Base
# --------------------------

@dataclass
class Antecedent:
    terms: List[Tuple[MF, float]]
    def degree(self) -> float:
        return min((mf(val) for mf, val in self.terms), default=0.0)

@dataclass
class Consequent:
    labels: Dict[str, Tuple[str, float]]

@dataclass
class Rule:
    antecedent: Antecedent
    consequent: Consequent
    weight: float = 1.0

def centroid(universe: Universe, mf_out: MF) -> float:
    num = 0.0; den = 0.0
    for x in universe.points:
        mu = mf_out(x); num += x * mu; den += mu
    return num / den if den > 1e-9 else 0.0

def aggregate(universe: Universe, clips: List[Tuple[MF, float]]) -> MF:
    def f(x: float) -> float:
        best = 0.0
        for mf, alpha in clips:
            best = max(best, min(alpha, mf(x)))
        return best
    return f

# --------------------------
# Inference (adds debug shapes)
# --------------------------

MF = Callable[[float], float]

@dataclass
class Inputs:
    rainfall_mm: float
    gw_depl_pct: float
    scarcity_idx: float
    ag_demand_pct: float
    ind_demand_pct: float
    hh_demand_pct: float

@dataclass
class Outputs:
    ag_alloc: float
    ind_alloc: float
    hh_alloc: float

def _aggregate_shapes_for_output(universe: Universe, clips: List[Tuple[MF, float]]) -> List[Tuple[float, float]]:
    shape: List[Tuple[float, float]] = []
    for x in universe.points:
        mu = 0.0
        for mf, alpha in clips:
            mu = max(mu, min(alpha, mf(x)))
        shape.append((x, mu))
    return shape

def _centroid_from_shape(shape: List[Tuple[float, float]]) -> float:
    num = 0.0
    den = 0.0
    for x, mu in shape:
        num += x * mu
        den += mu
    return (num / den) if den > 1e-9 else 0.0

def build_rules(inp: Inputs) -> List[Rule]:
    r = inp.rainfall_mm
    g = inp.gw_depl_pct
    s = inp.scarcity_idx
    agd = inp.ag_demand_pct
    ind = inp.ind_demand_pct
    hh  = inp.hh_demand_pct
    R, G, S, D = RAIN, GW, SCAR, DEMAND
    rules: List[Rule] = []
    rules.append(Rule(Antecedent([(R["very_low"], r), (G["high"], g), (S["high"], s)]),
                      Consequent({"ag": ("low", 1.0), "ind": ("medium", 1.0), "hh": ("high", 1.0)}), weight=1.2))
    rules.append(Rule(Antecedent([(R["low"], r), (G["critical"], g), (S["severe"], s)]),
                      Consequent({"ag": ("very_low", 1.0), "ind": ("low", 1.0), "hh": ("very_high", 1.0)}), weight=1.4))
    rules.append(Rule(Antecedent([(R["high"], r), (G["low"], g)]),
                      Consequent({"ag": ("high", 1.0), "ind": ("medium", 1.0), "hh": ("medium", 1.0)})))
    rules.append(Rule(Antecedent([(R["very_high"], r), (G["low"], g), (S["low"], s)]),
                      Consequent({"ag": ("very_high", 1.0), "ind": ("high", 1.0), "hh": ("medium", 1.0)})))
    rules.append(Rule(Antecedent([(S["severe"], s)]),
                      Consequent({"ag": ("low", 1.0), "ind": ("low", 1.0), "hh": ("very_high", 1.0)}), weight=1.3))
    not_severe = 1.0 - max(S["severe"](s), S["high"](s))
    if not_severe > 0.0:
        rules.append(Rule(Antecedent([(D["high"], agd)]), Consequent({"ag": ("high", not_severe)}), weight=0.9))
        rules.append(Rule(Antecedent([(D["high"], ind)]), Consequent({"ind": ("high", not_severe)}), weight=0.9))
        rules.append(Rule(Antecedent([(D["high"], hh)]), Consequent({"hh": ("high", not_severe)}), weight=0.9))
    rules.append(Rule(Antecedent([(G["high"], g)]), Consequent({"ind": ("medium", 1.0)}), weight=1.1))
    rules.append(Rule(Antecedent([(G["critical"], g)]), Consequent({"ind": ("low", 1.0)}), weight=1.2))
    rules.append(Rule(Antecedent([(R["very_low"], r)]), Consequent({"hh": ("medium", 1.0)}), weight=0.8))
    return rules

def infer_with_shapes(inp: Inputs) -> Tuple[Outputs, Dict[str, List[Tuple[float, float]]]]:
    rules = build_rules(inp)
    ag_clips: List[Tuple[MF, float]] = []; ind_clips: List[Tuple[MF, float]] = []; hh_clips: List[Tuple[MF, float]] = []
    for rule in rules:
        deg = rule.antecedent.degree() * rule.weight
        if deg <= 0.0: continue
        for out, (lbl, strength) in rule.consequent.labels.items():
            alpha = max(0.0, min(1.0, deg * strength))
            if out == "ag":  ag_clips.append((ALLOC[lbl], alpha))
            if out == "ind": ind_clips.append((ALLOC[lbl], alpha))
            if out == "hh":  hh_clips.append((ALLOC[lbl], alpha))
    if not ag_clips:  ag_clips.append((ALLOC["medium"], 0.5))
    if not ind_clips: ind_clips.append((ALLOC["medium"], 0.5))
    if not hh_clips:  hh_clips.append((ALLOC["medium"], 0.6))

    shapes: Dict[str, List[Tuple[float, float]]] = {
        "ag":  _aggregate_shapes_for_output(U_PCT, ag_clips),
        "ind": _aggregate_shapes_for_output(U_PCT, ind_clips),
        "hh":  _aggregate_shapes_for_output(U_PCT, hh_clips),
    }

    ag_out  = _centroid_from_shape(shapes["ag"])
    ind_out = _centroid_from_shape(shapes["ind"])
    hh_out  = _centroid_from_shape(shapes["hh"])

    floor_hh = 15.0
    ag_out  = max(0.0, min(100.0, ag_out))
    ind_out = max(0.0, min(100.0, ind_out))
    hh_out  = max(floor_hh, max(0.0, min(100.0, hh_out)))

    total = ag_out + ind_out + hh_out
    if total > 1e-6:
        scale = 100.0 / total
        ag_out, ind_out, hh_out = ag_out*scale, ind_out*scale, hh_out*scale
    else:
        ag_out, ind_out, hh_out = 33.34, 33.33, 33.33

    return Outputs(ag_out, ind_out, hh_out), shapes

def infer(inp: Inputs) -> Outputs:
    out, _ = infer_with_shapes(inp)
    return out

# --------------------------
# Public API
# --------------------------

def advise_allocation(
    country: Optional[str] = None,
    year: Optional[int] = None,
    *,
    rainfall_mm: Optional[float] = None,
    gw_depl_pct: Optional[float] = None,
    scarcity_level: Optional[str] = None,
    ag_demand_pct: Optional[float] = None,
    ind_demand_pct: Optional[float] = None,
    hh_demand_pct: Optional[float] = None,
    data_path: str = DATA_PATH
) -> Tuple[Outputs, Inputs]:
    base = None
    if country and year:
        rows = read_dataset(data_path)
        base = find_row(rows, country, year)
        if base is None:
            base = Row(country, year, 1000.0, 3.0, SCARCITY_MAP["Moderate"], 50.0, 25.0, 25.0)

    r_mm = rainfall_mm if rainfall_mm is not None else (base.rainfall_mm if base else 1000.0)
    gw   = gw_depl_pct if gw_depl_pct is not None else (base.gw_depl_pct if base else 3.0)
    scar = scarcity_to_idx(scarcity_level) if scarcity_level is not None else (base.scarcity_idx if base else SCARCITY_MAP["Moderate"])
    agd  = ag_demand_pct if ag_demand_pct is not None else (base.ag_pct if base else 50.0)
    ind  = ind_demand_pct if ind_demand_pct is not None else (base.ind_pct if base else 25.0)
    hh   = hh_demand_pct if hh_demand_pct is not None else (base.hh_pct if base else 25.0)

    inp = Inputs(
        rainfall_mm=clip(r_mm, U_RAIN.start, U_RAIN.end),
        gw_depl_pct=clip(gw, U_GW.start, U_GW.end),
        scarcity_idx=clip(scar, U_SCAR.start, U_SCAR.end),
        ag_demand_pct=clip(agd, 0.0, 100.0),
        ind_demand_pct=clip(ind, 0.0, 100.0),
        hh_demand_pct=clip(hh, 0.0, 100.0),
    )
    return infer(inp), inp

def advise_batch(data_path: str = DATA_PATH, country_filter: Optional[str] = None) -> List[Tuple[Row, Outputs]]:
    rows = read_dataset(data_path)
    results: List[Tuple[Row, Outputs]] = []
    for r in rows:
        if country_filter and r.country.lower() != country_filter.lower():
            continue
        inp = Inputs(r.rainfall_mm, r.gw_depl_pct, r.scarcity_idx, r.ag_pct, r.ind_pct, r.hh_pct)
        results.append((r, infer(inp)))
    return results

def export_allocations(rows_out: List[Tuple[Row, Outputs]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Country","Year","Rainfall(mm)","GW Depletion(%)","ScarcityIdx",
                    "AgDemand(%)","IndDemand(%)","HHDemand(%)","Alloc_Ag(%)","Alloc_Ind(%)","Alloc_HH(%)"])
        for r, o in rows_out:
            w.writerow([r.country, r.year, f"{r.rainfall_mm:.2f}", f"{r.gw_depl_pct:.2f}", f"{r.scarcity_idx:.2f}",
                        f"{r.ag_pct:.2f}", f"{r.ind_pct:.2f}", f"{r.hh_pct:.2f}",
                        f"{o.ag_alloc:.2f}", f"{o.ind_alloc:.2f}", f"{o.hh_alloc:.2f}"])

# --------------------------
# Plotting support
# --------------------------

def _force_gui_backend():
    # Helps Windows show figures instead of silently using Agg
    try:
        import matplotlib
        if "Agg" in matplotlib.get_backend():
            for cand in ("TkAgg", "Qt5Agg", "QtAgg", "WXAgg"):
                try:
                    matplotlib.use(cand, force=True)
                    break
                except Exception:
                    continue
    except Exception:
        pass

def get_plt():
    try:
        _force_gui_backend()
        import matplotlib.pyplot as plt
        return plt, True
    except Exception:
        print("Plotting disabled: matplotlib not available or GUI backend issue.")
        print("Tip: pip install matplotlib OR run with --save-dir to export PNGs.")
        return None, False

def _save_or_note(plt, save_dir: Optional[str], filename: str) -> None:
    if not plt: return
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")

def _maybe_save_or_show(save_dir: Optional[str], filename: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        plt.savefig(path, bbox_inches="tight"); plt.close()
        print(f"Saved: {path}")
    else:
        plt.show()

# --------------------------
# Visualizations
# --------------------------

def plot_fuzzy_output_shape_for(var_name: str,
                                shape: List[Tuple[float, float]],
                                title_suffix: str = "",
                                save_dir: Optional[str] = None) -> bool:
    plt, ok = get_plt()
    if not ok: return False
    xs = [x for x, _ in shape]
    ys = [mu for _, mu in shape]
    plt.figure()
    plt.plot(xs, ys)
    plt.title(f"Aggregated Fuzzy Output – {var_name.upper()} {title_suffix}".strip())
    plt.xlabel("Allocation (%)")
    plt.ylabel("Membership μ")
    plt.ylim(0, 1.05)
    _save_or_note(plt, save_dir, f"fuzzy_{var_name.lower()}.png")
    return True

def viz_fuzzy_outputs(country: Optional[str],
                      year: Optional[int],
                      *,
                      rainfall_mm: Optional[float],
                      gw_depl_pct: Optional[float],
                      scarcity_level: Optional[str],
                      ag_demand_pct: Optional[float],
                      ind_demand_pct: Optional[float],
                      hh_demand_pct: Optional[float],
                      data_path: str,
                      save_dir: Optional[str]) -> None:
    _, base_inp = advise_allocation(
        country=country, year=year,
        rainfall_mm=rainfall_mm, gw_depl_pct=gw_depl_pct, scarcity_level=scarcity_level,
        ag_demand_pct=ag_demand_pct, ind_demand_pct=ind_demand_pct, hh_demand_pct=hh_demand_pct,
        data_path=data_path,
    )
    out, shapes = infer_with_shapes(base_inp)
    suffix = f"– {country} {year}" if country and year else ""
    any_plots = False
    any_plots |= plot_fuzzy_output_shape_for("agriculture", shapes["ag"], suffix, save_dir)
    any_plots |= plot_fuzzy_output_shape_for("industry",    shapes["ind"], suffix, save_dir)
    any_plots |= plot_fuzzy_output_shape_for("households",  shapes["hh"], suffix, save_dir)
    if any_plots and not save_dir:
        plt, ok = get_plt()
        if ok and plt: plt.show()

def plot_memberships(save_dir: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure()
    for name, mf in RAIN.items():
        y = [mf(x) for x in U_RAIN.points]; plt.plot(U_RAIN.points, y, label=name)
    plt.title("Rainfall Membership Functions"); plt.xlabel("mm"); plt.ylabel("Membership"); plt.legend()
    _maybe_save_or_show(save_dir, "mf_rainfall.png")

    plt.figure()
    for name, mf in GW.items():
        y = [mf(x) for x in U_GW.points]; plt.plot(U_GW.points, y, label=name)
    plt.title("Groundwater Depletion Membership Functions"); plt.xlabel("%"); plt.ylabel("Membership"); plt.legend()
    _maybe_save_or_show(save_dir, "mf_groundwater.png")

    plt.figure()
    for name, mf in SCAR.items():
        y = [mf(x) for x in U_SCAR.points]; plt.plot(U_SCAR.points, y, label=name)
    plt.title("Water Scarcity Membership Functions"); plt.xlabel("Index (0..3)"); plt.ylabel("Membership"); plt.legend()
    _maybe_save_or_show(save_dir, "mf_scarcity.png")

    plt.figure()
    for name, mf in DEMAND.items():
        y = [mf(x) for x in U_PCT.points]; plt.plot(U_PCT.points, y, label=name)
    plt.title("Sector Demand Membership Functions"); plt.xlabel("%"); plt.ylabel("Membership"); plt.legend()
    _maybe_save_or_show(save_dir, "mf_demand.png")

    plt.figure()
    for name, mf in ALLOC.items():
        y = [mf(x) for x in U_PCT.points]; plt.plot(U_PCT.points, y, label=name)
    plt.title("Allocation Output Membership Functions"); plt.xlabel("%"); plt.ylabel("Membership"); plt.legend()
    _maybe_save_or_show(save_dir, "mf_allocation.png")

def plot_single_allocation_bar(country: str, year: int, out: Outputs, save_dir: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = ["Agriculture", "Industry", "Households"]; values = [out.ag_alloc, out.ind_alloc, out.hh_alloc]
    plt.figure(); plt.bar(labels, values)
    plt.title(f"Recommended Allocation – {country} {year}"); plt.ylabel("Percent"); plt.ylim(0, 100)
    _maybe_save_or_show(save_dir, f"alloc_{country}_{year}.png".replace(" ", "_"))

def plot_country_series(country: str, data_path: str = DATA_PATH, save_dir: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [r for r in read_dataset(data_path) if r.country.lower() == country.lower()]
    if not rows:
        print(f"No rows found for country: {country}"); return
    rows.sort(key=lambda r: r.year)
    years, ags, inds, hhs = [], [], [], []
    for r in rows:
        o = infer(Inputs(r.rainfall_mm, r.gw_depl_pct, r.scarcity_idx, r.ag_pct, r.ind_pct, r.hh_pct))
        years.append(r.year); ags.append(o.ag_alloc); inds.append(o.ind_alloc); hhs.append(o.hh_alloc)
    plt.figure()
    plt.plot(years, ags, label="Agriculture"); plt.plot(years, inds, label="Industry"); plt.plot(years, hhs, label="Households")
    plt.title(f"Allocation Trends – {country}"); plt.xlabel("Year"); plt.ylabel("Percent"); plt.ylim(0, 100); plt.legend()
    _maybe_save_or_show(save_dir, f"alloc_trends_{country}.png".replace(" ", "_"))

def plot_reasoning(shapes: Dict[str, List[Tuple[float, float]]],
                   centroids: Dict[str, float],
                   title_prefix: str,
                   save_dir: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for key, label in [("ag","Agriculture"),("ind","Industry"),("hh","Households")]:
        data = shapes[key]; xs = [x for x,_ in data]; ys = [m for _,m in data]
        plt.figure()
        plt.plot(xs, ys)
        plt.axvline(centroids[key])
        plt.title(f"{title_prefix} – {label} Aggregated μ(x)")
        plt.xlabel("%"); plt.ylabel("Membership")
        _maybe_save_or_show(save_dir, f"reason_{title_prefix}_{key}.png".replace(" ", "_"))

# --------------------------
# CLI & Interactive Menu
# --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sustainable Water Allocation Advisor (Fuzzy Logic + Viz)")
    p.add_argument("--country", type=str)
    p.add_argument("--year", type=int)
    p.add_argument("--rainfall", type=float)
    p.add_argument("--gw", type=float, help="Groundwater depletion rate (%)")
    p.add_argument("--scarcity", type=str, choices=list(SCARCITY_MAP.keys()))
    p.add_argument("--ag", type=float, help="Historical Agricultural Water Use (%)")
    p.add_argument("--ind", type=float, help="Historical Industrial Water Use (%)")
    p.add_argument("--hh", type=float, help="Historical Household Water Use (%)")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--country_filter", type=str)
    p.add_argument("--export", type=str)
    p.add_argument("--viz-membership", action="store_true")
    p.add_argument("--viz-country-year", action="store_true")
    p.add_argument("--viz-country-series", action="store_true")
    p.add_argument("--viz-like-example", action="store_true")
    p.add_argument("--viz-fuzzy-outputs", action="store_true",
                   help="Plot aggregated fuzzy output shapes (Ag/Ind/HH) pre-defuzzification")
    p.add_argument("--viz-reason", action="store_true",
                   help="Plot aggregated fuzzy output μ(x) with centroid markers (Ag/Ind/HH)")
    p.add_argument("--plot", action="store_true",
                   help="Show single-year allocation bar when used with --country/--year")
    p.add_argument("--save-dir", type=str)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--data", type=str, default=DATA_PATH)
    return p.parse_args()

# ---- Interactive helpers

def ask_text(prompt: str, default: Optional[str] = None) -> str:
    p = f"{prompt}" + (f" [{default}]" if default else "") + ": "
    s = input(p).strip()
    return s if s else (default or "")

def ask_int(prompt: str, default: Optional[int] = None) -> int:
    while True:
        s = ask_text(prompt, str(default) if default is not None else None)
        try:
            return int(s)
        except Exception:
            print("Please enter an integer.")

def ask_yesno(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        s = input(f"{prompt} [{d}]: ").strip().lower()
        if not s: return default
        if s in ("y","yes"): return True
        if s in ("n","no"): return False
        print("Please answer y/n.")

def ask_choice(title: str, options: List[str]) -> int:
    print("\n" + title)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        s = input("Select an option (number): ").strip()
        if s.isdigit():
            k = int(s)
            if 1 <= k <= len(options):
                return k
        print("Invalid choice, try again.")

def interactive_menu(args: argparse.Namespace) -> None:
    # Load dataset hints
    rows = read_dataset(args.data)
    countries = sorted({r.country for r in rows}) if rows else []
    default_country = countries[0] if countries else (args.country or "Argentina")
    years_for_default = sorted([r.year for r in rows if r.country == default_country]) if rows else []
    default_year = years_for_default[0] if years_for_default else (args.year if args.year else 2003)

    print("\nSustainable Water Allocation Advisor – Interactive Mode")
    if not rows:
        print("Note: Dataset not found or empty at:", args.data)
        print("You can still run visuals; defaults will be used where needed.\n")

    while True:
        choice = ask_choice("What would you like to visualize?", [
            "Fuzzy output graphs (Ag/Ind/HH) – explainability",
            "Single-year allocation bar (country & year)",
            "Membership functions (inputs/outputs)",
            "Country allocation trends over years",
            "Reason view: fuzzy graphs with centroid markers",
            "Exit"
        ])

        save = ask_yesno("Save PNGs instead of showing pop-ups?", default=False)
        save_dir = None
        if save:
            save_dir = ask_text("Save directory", default="out/plots")

        if choice == 1:
            country = ask_text("Country", default=default_country)
            year = ask_int("Year", default=default_year)
            viz_fuzzy_outputs(country, year,
                              rainfall_mm=None, gw_depl_pct=None, scarcity_level=None,
                              ag_demand_pct=None, ind_demand_pct=None, hh_demand_pct=None,
                              data_path=args.data, save_dir=save_dir)

        elif choice == 2:
            country = ask_text("Country", default=default_country)
            year = ask_int("Year", default=default_year)
            out, _ = advise_allocation(country=country, year=year, data_path=args.data)
            plot_single_allocation_bar(country, year, out, save_dir)

        elif choice == 3:
            plot_memberships(save_dir)

        elif choice == 4:
            country = ask_text("Country", default=default_country)
            plot_country_series(country, args.data, save_dir)

        elif choice == 5:
            country = ask_text("Country", default=default_country)
            year = ask_int("Year", default=default_year)
            _, inp = advise_allocation(country=country, year=year, data_path=args.data)
            _, shapes = infer_with_shapes(inp)
            cents = {
                "ag": _centroid_from_shape(shapes["ag"]),
                "ind": _centroid_from_shape(shapes["ind"]),
                "hh": _centroid_from_shape(shapes["hh"]),
            }
            title = f"{country}_{year}"
            plot_reasoning(shapes, cents, title, save_dir)

        elif choice == 6:
            print("Goodbye!")
            return

        # If showing (not saving), ensure any pending figures pop
        if not save:
            plt, ok = get_plt()
            if ok and plt:
                plt.show()

        # Another?
        if not ask_yesno("\nDo you want to visualize something else?", default=True):
            print("Goodbye!")
            return

# --------------------------
# main()
# --------------------------

def main() -> None:
    args = parse_args()

    # Batch path
    if args.batch:
        results = advise_batch(args.data, args.country_filter)
        if args.export:
            export_allocations(results, args.export)
            print(f"Exported: {args.export}")
        for (r, o) in results[:5]:
            print(f"{r.country} {r.year}: Ag {o.ag_alloc:.1f}%, Ind {o.ind_alloc:.1f}%, HH {o.hh_alloc:.1f}%")
        return

    # If any explicit viz flags are used, respect them (non-interactive)
    if any([args.viz_fuzzy_outputs, args.viz_membership, args.viz_country_year,
            args.viz_country_series, args.viz_like_example, args.viz_reasons if hasattr(args,'viz_reasons') else False,
            args.plot]):
        if args.viz_fuzzy_outputs and not args.no_plots:
            viz_fuzzy_outputs(
                args.country, args.year,
                rainfall_mm=args.rainfall, gw_depl_pct=args.gw, scarcity_level=args.scarcity,
                ag_demand_pct=args.ag, ind_demand_pct=args.ind, hh_demand_pct=args.hh,
                data_path=args.data, save_dir=args.save_dir
            ); return

        if args.viz_membership:
            plot_memberships(args.save_dir); return

        if args.viz_country_year:
            if not (args.country and args.year):
                print("--viz-country-year requires --country and --year"); return
            out, _ = advise_allocation(country=args.country, year=args.year,
                                       rainfall_mm=args.rainfall, gw_depl_pct=args.gw, scarcity_level=args.scarcity,
                                       ag_demand_pct=args.ag, ind_demand_pct=args.ind, hh_demand_pct=args.hh,
                                       data_path=args.data)
            plot_single_allocation_bar(args.country, args.year, out, args.save_dir); return

        if args.viz_country_series:
            if not args.country:
                print("--viz-country-series requires --country"); return
            plot_country_series(args.country, args.data, args.save_dir); return

        if args.viz_reasons if hasattr(args,'viz_reasons') else False:
            # (kept for compatibility if you rename flags)
            if not (args.country and args.year):
                print("--viz-reason(s) requires --country and --year"); return
            _, inp = advise_allocation(country=args.country, year=args.year,
                                       rainfall_mm=args.rainfall, gw_depl_pct=args.gw, scarcity_level=args.scarcity,
                                       ag_demand_pct=args.ag, ind_demand_pct=args.ind, hh_demand_pct=args.hh,
                                       data_path=args.data)
            _, shapes = infer_with_shapes(inp)
            cents = { "ag": _centroid_from_shape(shapes["ag"]),
                      "ind": _centroid_from_shape(shapes["ind"]),
                      "hh": _centroid_from_shape(shapes["hh"]) }
            title = f"{args.country}_{args.year}"
            plot_reasoning(shapes, cents, title, args.save_dir); return

        if args.plot and args.country and args.year:
            out, _ = advise_allocation(country=args.country, year=args.year, data_path=args.data)
            plot_single_allocation_bar(args.country, args.year, out, args.save_dir); return

    # No flags: launch interactive menu
    interactive_menu(args)

if __name__ == "__main__":
    main()
