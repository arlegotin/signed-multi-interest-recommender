"""Package the accepted G2 early stop from verified saved metadata only.

Run from the repository root. This is G6 analysis packaging, not a change to
the experimental renderer, admission policy, or scientific implementation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from signed_study.budget import Ledger
from signed_study.protocol import canonical_sha256
from signed_study.report import render_report, report_config
from signed_study.stage_evidence import verify_record, verify_reference


DIAGNOSTIC_SHA = "740422f895ad2696747ee525ed7a4ce441a5dcd796e4aca971d8ef6e94b0d874"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def table(root, name, rows):
    path = root / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def markdown(rows):
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[k]:.9f}" if isinstance(row[k], float) else str(row[k]) for k in keys) + " |")
    return "\n".join(lines)


def figures(root, actual, forecast):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    labels = {
        "cache load verify": "Cache verification", "catalog encode": "Catalog encoding",
        "cold model load": "Cold model load", "probe": "Raw probe", "repeat": "Raw repeat",
        "canonical score and metrics": "Canonical score + metrics", "fit": "Fit",
        "numerical validation": "Numerical comparison",
        "pipeline write resource forecast and finalization residual": "Unattributed pipeline residual",
        "repeated score and metrics": "Repeated score + metrics",
    }
    for ax, rows, title in zip(axes, (actual[:5], actual[5:]),
                               ("Shared encoding phases", "Calibration workload components")):
        ax.barh(range(len(rows)), [r["seconds"] for r in rows], color="#315b78")
        ax.set_yticks(range(len(rows)), [labels[r["component"]] for r in rows], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Measured wall time (seconds)")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Actual G2 work — separate measured phases; residual is unattributed")
    actual_path = root / "actual-calibration-costs.png"
    fig.savefig(actual_path, dpi=140, metadata={"Software": "signed-study G6 saved-evidence packaging"})

    fig = Figure(figsize=(9, 4), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    colors = ("#315b78", "#77a1a6", "#cf9d62")
    for i, row in enumerate(forecast):
        left = 0
        for key, color in zip(("fit_seconds", "score_and_metrics_seconds", "write_and_pipeline_overhead_seconds"), colors):
            ax.barh(i, row[key], left=left, color=color, label=key.replace("_seconds", "").replace("_", " ") if i == 0 else None)
            left += row[key]
        ax.plot(row["safety_seconds"], i, "o", color="#763743", label="1.5× safety forecast" if i == 0 else None)
        ax.plot(row["effective_limit"], i, "|", color="black", markersize=17, markeredgewidth=2,
                label="Effective stage allocation" if i == 0 else None)
    ax.set_yticks(range(3), [r["stage"] for r in forecast])
    ax.invert_yaxis()
    ax.set_xlabel("Forecast wall time (seconds); linear axis")
    ax.set_title("UNEXECUTED smallest schedule (256 development / 1000 confirmation)")
    ax.legend(fontsize=8, loc="lower right")
    forecast_path = root / "forecast-vs-allocation.png"
    fig.savefig(forecast_path, dpi=140, metadata={"Software": "signed-study G6 saved-evidence packaging"})
    return [actual_path, forecast_path]


def package(root, ledger):
    diagnostic_path = root / "operator_logs/task13-forecast-diagnostic.json"
    if digest(diagnostic_path) != DIAGNOSTIC_SHA:
        raise ValueError("Accepted saved diagnostic changed")
    d = json.loads(diagnostic_path.read_text())
    for ref in (d["calibration"], d["workload"]):
        verify_reference(ref)
    if (root / "protocol.lock.json").exists():
        raise ValueError("This packaging recipe is restricted to the accepted no-lock early stop")
    state = ledger.snapshot()
    assert state["attempted_evaluations"] == 1056
    assert len(state["reallocations"]) == 5
    assert state["stage_limits"]["G1"] == 3300 and state["stage_limits"]["G4"] == 2100
    assert all(state["stage_seconds"][g] == 0 for g in ("G3", "G4", "G5"))
    base = render_report(root)
    expected_limit = "Inspection metadata is self-checksummed but lacks matching external ledger publication; hardware is unverified historical metadata"
    if any(item != expected_limit for item in base["limitations"]):
        raise ValueError(f"Saved-evidence validation failed: {base['limitations']}")
    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text())
    verify_record(environment)
    historical_hardware_verified = any(
        a["stage"] == "G0" and a["status"] == "completed"
        and Path(a.get("outputs", {}).get("environment", {}).get("path", "")).resolve() == environment_path.resolve()
        and a.get("outputs", {}).get("environment", {}).get("sha256") == digest(environment_path)
        for a in state["attempts"].values())
    assert base["outcome"] == "practical infeasibility"
    actual = [dict(component=k.replace("_seconds", "").replace("_", " "), seconds=v)
              for k, v in d["timings"].items()]
    # The two panels have separate phase scopes, so they are not summed together.
    actual += [dict(component=k.replace("_seconds", "").replace("_", " "), seconds=v)
               for k, v in d["measured_costs"].items()]
    forecast = [dict(stage=g, **row, evaluations=d["stage_evaluations"][g])
                for g, row in d["projected_stage_components"].items()]
    pairs = [dict(development=r["pair"][0], confirmation=r["pair"][1], chunk=r["chunk"],
                  G3_raw_seconds=r["seconds"]["G3"], G4_raw_seconds=r["seconds"]["G4"],
                  G5_raw_seconds=r["seconds"]["G5"], admitted=r["fits"])
             for r in d["all_projection_replays"]]
    outputs = [table(root, "actual-calibration-costs.csv", actual),
               table(root, "forecast-stage-costs.csv", forecast),
               table(root, "forecast-scale-attempts.csv", pairs)]
    outputs += figures(root, actual, forecast)
    text = (root / "report.md").read_text()
    if historical_hardware_verified:
        text = text.replace("Hardware (saved inspection; not externally verified)",
                            "Hardware (saved inspection; exact SHA and same resolved path verified against completed G0 publication)")
        text = text.replace('Evidence qualifications: ["' + expected_limit + '"]',
                            "Evidence qualifications: generic renderer's hardware warning came from relative/absolute reference spelling. The G6 supplement independently verified the record checksum, exact file SHA and same resolved path against a completed G0 publication. This is historical evidence, not a fresh hardware certification.")
    text = text.replace("Scope: this metadata/encoder setting and frozen cohort only.",
                        "Scope: declared pipeline forecast gate failure in this metadata/encoder setting. No frozen cohort or primary result exists; method efficacy remains untested.")
    text = text.replace("Scaling qualification: measured saved calibration plus matching permanent G2 failure and failed-stage reason; smallest tested scale shown above",
                        "Scaling qualification: measured calibration feeds an unexecuted conservative forecast. All six permitted scales failed; no final scoring policy or scale was admitted. Precise saved projections appear below.")
    text = text.replace("Whole frozen denominator: 0.", "Frozen denominator: unavailable; no cohort was frozen.")
    text = text.replace("calibrate: missing or invalid; not executed/verified",
                        "calibrate: no successful stage receipt; verified completed measurement publication retained under the terminal failed G2 forecast attempt")
    lines = text.splitlines()
    lines = [line.split(". Dominant saved term:")[0] + ". Dominant G3 forecast term: conservative write/pipeline overhead, 51333.382569912 seconds; this is not an observed scaled runtime. The 44.414098640-second measured residual is not isolated I/O." if ". Dominant saved term:" in line else line for line in lines]
    text = "\n".join(lines)
    supplement = "\n\n".join([
        "### Verified G2 measurement and forecast supplement",
        "G0 acquisition/audit, current G1 (413 tests, 250.32 seconds), and G2 encoding/calibration measurements were reached. G2 failed prospective forecast admission. Smoke, G3 development/endpoint expansion, S*/B*, precision/history thresholds, freeze, G4 confirmation and G5 diagnostics were not admitted. Auditor eligibility access is separate from model calibration, which opened only 32 development users. Confirmation outcomes were not opened by selection or confirmation.",
        "The measured 3883×384 shared cache, raw repeat maximum difference 0 (tolerance 1e-5), and chunk checks 1/7/512 passed. There were 352 fits/canonical candidate-score evaluations and 704 scoring repeats: 1056 completed evaluations, no failed evaluation or requested solver/prototype repair. Canonical calibration used chunk512; the last failed forecast policy used chunk7 and is not an admitted primary scoring choice.",
        "Measured phase/component seconds (shared encoding phases and the calibration workload components have separate scopes):\n\n" + markdown(actual),
        "![Measured calibration costs](actual-calibration-costs.png)",
        "Unexecuted smallest schedule, 256 development / 1000 confirmation. Raw components, 1.5× safety bounds and effective allocations, in seconds:\n\n" + markdown(forecast),
        "![Unexecuted forecast versus allocation](forecast-vs-allocation.png)",
        "The forecast uses upper measured size-bin fit costs with nonnegative slopes, maximum measured score rates, and maximum per-entry family write/pipeline overhead. G3 smoke reuses 92-entry prices for 8 users while reserving 8×78 evaluations. The dominant G3 upper envelope is a pipeline estimate, not evidence that the method itself cannot scale. Actual wall-time caps were not exhausted.",
        f"Raw future forecast {d['raw_forecast_seconds']:.9f}s; safety forecast {d['safety_forecast_seconds']:.9f}s; original terminal-plus-future {d['terminal_plus_safety_projection']:.9f}s versus protected allowance {d['protected_work_cap']}s (14400−1800). Original active-time basis {d['experimental_seconds_before_final_attempt']:.9f}–{d['experimental_seconds_at_terminal']:.9f}s. Future {d['projected_evaluations']} + actual {d['original_attempted_evaluations']} = 41432/100000 evaluations; time forecasts bind, evaluation count does not.",
        "Unexecuted G5 full-catalog timing alone: 4112.387355s (fit 13.033975s, scoring 2059.276603s, overhead 2040.076777s). This is distinct from the measured 11.186897s catalog encoding. No full-catalog G5 timing campaign occurred.",
        "All saved scale/policy attempts (initial chunk512 followed by chunk7 reduction; six distinct size pairs):\n\n" + markdown(pairs),
        "The efficacy question, delta=.01 effect, paired confidence interval, power/precision and component attribution are unestimated. There is no accuracy table, practical superiority claim, informative negative result, or method-level runtime impossibility conclusion. Operational taxonomy 'practical infeasibility' applies only to the declared pipeline admission policy.",
        "Two material repair cycles were used; combined attributed validation 269.685277538257s is already included in experimental accounting, not added again. Exactly five G4→G1 transfers leave G1=3300s, G4=2100s and allocated stage sum=13800s; global 14400s, setup 2700s and reporting 1800s are unchanged. Native torch redirects/FutureWarning noise did not cause the forecast stop. Existing exact-pin compatibility checks do not prove a fresh clean-room environment rebuild.",
        f"Diagnostic SHA256: `{DIAGNOSTIC_SHA}`. Its calibration/workload hashes were verified before packaging. The diagnostic replays saved arithmetic at the original terminal G2 prefix, with no new benchmark or inference. This supplement corrects known generic-renderer frozen-cohort/forecast/dominant-term wording without changing study code. See README and docs/study-decisions.md for reproduction and the controller decision history.",
    ])
    text = text.replace("## 8. Correctness, failures and limitations", supplement + "\n\n## 8. Correctness, failures and limitations")
    text = text.replace("Next step: assess the frozen evidence limitations before deciding whether any further study is warranted.",
                        "Next step: audit the overhead measurement/scaling and evidence pipeline before another run of the same scientific question.")
    (root / "report.md").write_text(text + "\n")
    scientific = {str(p): digest(p) for p in [root / "actual-work.csv", *outputs]}
    old_path = root / "outcome-reproduction.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else None
    if old and old["script_sha256"] == digest(__file__):
        assert old["scientific_sha256"] == scientific, "Scientific artifacts changed on saved-evidence regeneration"
    inventory_path = root / "artifact-inventory.json"
    inventory = json.loads(inventory_path.read_text())
    # Replace the generic renderer's report hash after supplementing it.
    generated = [root / "report.md", root / "costs.csv", root / "actual-work.csv", *outputs]
    generated_names = {str(p.relative_to(root)) for p in generated}
    inventory["artifacts"] = [r for r in inventory["artifacts"] if r["name"] not in generated_names]
    inventory["artifacts"] += [dict(name=str(p.relative_to(root)), sha256=digest(p), status="generated aggregate export") for p in generated]
    ledger_path = root / "budget_ledger.jsonl"
    ledger_bytes = ledger_path.read_bytes()
    # Replay precisely this immutable byte prefix, avoiding live-attempt time drift.
    events = [json.loads(line) for line in ledger_bytes.splitlines()]
    reader = Ledger(ledger_path, report_config(root))
    try:
        snapshot = reader._state(events)
    finally:
        reader.close()
    as_of = dict(bytes=len(ledger_bytes), sha256=hashlib.sha256(ledger_bytes).hexdigest(),
                 event_count=len(snapshot["events"]), events_sha256=canonical_sha256({"events": snapshot["events"]}))
    inventory["packaging"] = dict(script=str(Path(__file__).relative_to(Path.cwd())) if Path(__file__).is_absolute() else __file__,
                                 script_sha256=digest(__file__), diagnostic_sha256=DIAGNOSTIC_SHA,
                                 ledger_prefix=as_of, recipe=".venv/bin/python scripts/render_study_outcome.py --results results")
    inventory["packaging"]["historical_hardware_verified"] = historical_hardware_verified
    inventory["artifacts"] += [dict(name="operator_logs/task13-forecast-diagnostic.json", sha256=DIAGNOSTIC_SHA,
                                     status="verified immutable saved forecast diagnostic")]
    # The ledger is still active; its full-file hash will change at owner finish.
    for item in inventory["artifacts"]:
        if item["name"] == "budget_ledger.jsonl":
            item["sha256"] = as_of["sha256"]
            item["status"] = "exact ledger byte prefix; see packaging.ledger_prefix"
    write_json(inventory_path, inventory)
    result = dict(schema_version=1, outcome="declared pipeline forecast gate failure; efficacy untested",
                  script_sha256=digest(__file__), diagnostic_sha256=DIAGNOSTIC_SHA,
                  scientific_sha256=scientific, ledger_prefix=as_of,
                  equal_previous_scientific_artifacts=bool(old and old["scientific_sha256"] == scientific),
                  snapshot_scope="Active G6 packaging prefix; its terminal cost is recorded later in permanent ledger",
                  experimental_seconds=snapshot["experimental_seconds"], stage_seconds=snapshot["stage_seconds"],
                  setup_seconds=snapshot["setup_seconds"])
    write_json(old_path, result)
    write_json(root / f"operator_logs/task14-regeneration-{as_of['event_count']}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    ledger = Ledger(args.results / "budget_ledger.jsonl", report_config(args.results))
    try:
        with ledger.work("G6", 60, 0):
            result = package(args.results, ledger)
    finally:
        ledger.close()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
