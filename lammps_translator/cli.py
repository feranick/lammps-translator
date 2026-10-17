"""cli.py — command line: lmpt format | validate | run | inputs | bench | status
Version: 2026.10.16.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import api


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _summary(name: str, r: dict) -> str:
    if not r.get("ok"):
        return f"{name}: FAILED  {r.get('why') or r.get('errors')}"
    ref = (r.get("checks") or {}).get("reference", {})
    line = f"{name}: ok  reliable={r.get('reliable')}  reference={'pass' if ref.get('pass') else ref.get('why', 'FAIL')}"
    for e in r.get("expect") or []:
        line += f"\n   expect {e['quantity']}: {e.get('got')} vs {e['value']} {'pass' if e['pass'] else 'FAIL'}"
    for w in r.get("unreliable_because") or []:
        line += f"\n   unreliable: {w}"
    for w in r.get("warnings") or []:
        line += f"\n   warning: {w}"
    return line + f"\n   -> {r.get('files', {}).get('folder')}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lmpt", description="lammps-translator: studies -> LAMMPS -> checked results")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("format", help="print the study format (JSON)")
    sub.add_parser("status", help="versions, LAMMPS, potentials folders")
    p = sub.add_parser("validate", help="static checks of a studies file")
    p.add_argument("file")
    p = sub.add_parser("run", help="run a studies file ({name: study})")
    p.add_argument("file")
    p.add_argument("names", nargs="*", help="only these studies")
    p.add_argument("--json", action="store_true", help="print the full results")
    p = sub.add_parser("inputs", help="write the LAMMPS inputs of one study without running")
    p.add_argument("file")
    p.add_argument("name")
    p = sub.add_parser("bench", help="benchmarks with known answers")
    p.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "format":
        print(json.dumps(api.study_format(), indent=1))
    elif a.cmd == "status":
        print(json.dumps(api.tool_status(), indent=1))
    elif a.cmd == "validate":
        r = api.validate(_load(a.file))
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    elif a.cmd == "run":
        st = _load(a.file)
        if a.names:
            st = {k: v for k, v in st.items() if k in a.names}
        r = api.run(st)
        if a.json:
            print(json.dumps(r, indent=1))
        else:
            for n, v in r["studies"].items():
                print(_summary(n, v))
            print(f"results: {r['request']}/results.json")
        return 0 if all(v.get("ok") for v in r["studies"].values()) else 1
    elif a.cmd == "inputs":
        st = _load(a.file)[a.name]
        r = api.inputs(st, a.name)
        if "error" in r:
            print(json.dumps(r, indent=1))
            return 1
        for k, v in r["inputs"].items():
            print(f"===== {k}\n{v}\n")
        print(f"written to {r['folder']}")
    elif a.cmd == "bench":
        r = api.bench(a.names or None)
        for n, v in r["cases"].items():
            if "skipped" in v:
                print(f"{n:24s} skipped ({v['skipped']})")
                continue
            print(f"{n:24s} {'pass' if v.get('passed') else 'FAIL'}  {v.get('seconds', '')} s")
            for c in v.get("checks", []):
                print(f"    {c['quantity']:34s} got {c.get('got')!s:>22}  expected {c.get('expected')!s:>22}  "
                      f"{'rel ' + str(c.get('rel_error')) if 'rel_error' in c else ''}  {'ok' if c['pass'] else 'FAIL'}")
            if v.get("why"):
                print(f"    why: {v['why']}")
            for u in v.get("unreliable_because") or []:
                print(f"    unreliable: {u}")
        print(f"passed {len(r['passed'])}, failed {len(r['failed'])}, skipped {len(r['skipped'])}  ({r['request']})")
        return 0 if not r["failed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
