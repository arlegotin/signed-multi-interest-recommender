"""Command discovery only; stage implementations are added in later tasks."""

import argparse


_COMMANDS = {
    "inspect": "G0: inspect hardware/resources; no outcomes accessed.",
    "acquire": "G0: acquire fixed data/model manifests; no comparison outcomes.",
    "audit": "G0: validate data and eligibility; auditor-only outcome access.",
    "calibrate": "G2: requires G1; encode catalog and calibrate development users.",
    "smoke": "G3: requires calibrated cache; development outcomes only.",
    "develop": "G3: requires calibrated cache and forecast; development outcomes only.",
    "freeze": "G3: requires validated development and clean committed source; lock protocol.",
    "confirm": "G4: requires verified immutable lock before opening confirmation outcomes.",
    "diagnose": "G5: requires verified lock; bounded frozen confirmation diagnostics.",
    "report": "G6: summarize saved evidence, including early stops; no new experiments.",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded CPU signed recommendation study.")
    commands = parser.add_subparsers(dest="command")
    for name, description in _COMMANDS.items():
        command = commands.add_parser(name, help=description, description=description)
        if name in ("confirm", "diagnose"):
            command.add_argument("--lock", default="results/protocol.lock.json")
        elif name == "report":
            command.add_argument("--results", default="results")
        else:
            command.add_argument("--config", default="config/study.json")
        if name == "smoke":
            command.add_argument("--development-users", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    parser.error(f"{args.command}: stage execution is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
