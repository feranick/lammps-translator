"""fakeslurm.py — stand-ins for sbatch, squeue, scancel, sacct and sinfo (tests only). Version: 2026.10.20.1

State in $FAKE_SLURM/state.json. sbatch starts job.sh in the background (or holds it PENDING when the file
$FAKE_SLURM/hold exists, and fails when $FAKE_SLURM/fail_submit exists); squeue reports RUNNING while the process
lives; scancel kills it (and writes SLURM's CANCELLED line to the job's output); sacct answers from
$FAKE_SLURM/sacct_<id> or the recorded state, and fails when $FAKE_SLURM/sacct_fail exists.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path(os.environ["FAKE_SLURM"])
STATE = D / "state.json"


def load():
    return json.loads(STATE.read_text()) if STATE.is_file() else {"next": 1001, "jobs": {}}


def save(s):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s))
    os.replace(tmp, STATE)


def alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        st = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        return st != "Z"
    except OSError:
        return True


def sbatch(args):
    if (D / "fail_submit").exists():
        print("sbatch: error: Batch job submission failed: Unable to contact slurm controller", file=sys.stderr)
        return 1
    script = Path(args[-1]).resolve()
    opts = {}
    for ln in script.read_text().splitlines():
        if ln.startswith("#SBATCH --"):
            k, _, v = ln[len("#SBATCH --"):].partition("=")
            opts[k] = v
    s = load()
    jid = str(s["next"])
    s["next"] += 1
    job = {"name": opts.get("job-name", script.name), "cpus": int(opts.get("ntasks", "1")), "dir": opts.get("chdir"),
           "out": opts.get("output"), "script": str(script), "submit": time.time(), "state": "PENDING", "pid": None,
           "args": args[:-1]}
    if not (D / "hold").exists():
        env = dict(os.environ, SLURM_JOB_ID=jid, SLURM_NTASKS=str(job["cpus"]))
        out = open(job["out"], "a")
        p = subprocess.Popen(["bash", str(script)], cwd=job["dir"], stdout=out, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        job.update({"state": "RUNNING", "pid": p.pid, "start": time.time()})
    s["jobs"][jid] = job
    save(s)
    print(jid)
    return 0


def current(job):
    if job["state"] == "RUNNING" and not alive(job["pid"]):
        return "COMPLETED"
    return job["state"]


def squeue(args):
    fmt = args[args.index("-o") + 1] if "-o" in args else "%i"
    only = args[args.index("-t") + 1] if "-t" in args else None
    s = load()
    for jid, job in sorted(s["jobs"].items(), reverse=True):     # like squeue: not in scheduling order
        st = current(job)
        if st not in ("PENDING", "RUNNING") or job.get("vanished"):
            continue
        if only and st != only:
            continue
        el = int(time.time() - job.get("start", time.time())) if st == "RUNNING" else 0
        val = {"i": jid, "T": st, "r": "Resources" if st == "PENDING" else "None", "S": "N/A",
               "M": f"{el // 60}:{el % 60:02d}", "j": job["name"], "C": str(job["cpus"]), "Q": "1000"}
        line = fmt
        for k, v in val.items():
            line = line.replace("%" + k, v)
        print(line)
    return 0


def scancel(args):
    s = load()
    for jid in args:
        job = s["jobs"].get(jid)
        if not job:
            print(f"scancel: error: Invalid job id {jid}", file=sys.stderr)
            continue
        if job["state"] == "RUNNING" and alive(job["pid"]):
            try:
                os.killpg(job["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            with open(job["out"], "a") as f:
                f.write(f"slurmstepd: error: *** JOB {jid} ON fake CANCELLED AT {time.strftime('%Y-%m-%dT%H:%M:%S')} ***\n")
        job["state"] = "CANCELLED"
    save(s)
    return 0


def sacct(args):
    if (D / "sacct_fail").exists():
        print("sacct: error: Slurm accounting storage is disabled", file=sys.stderr)
        return 1
    jid = args[args.index("-j") + 1]
    o = D / f"sacct_{jid}"
    if o.exists():
        print(o.read_text().strip())
        return 0
    job = load()["jobs"].get(jid)
    if job:
        print(current(job))
    return 0


def sinfo(args):
    s = load()
    used = sum(j["cpus"] for j in s["jobs"].values() if current(j) == "RUNNING")
    print(f"{used}/{8 - used}/0/8")
    return 0


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    sys.exit({"sbatch": sbatch, "squeue": squeue, "scancel": scancel, "sacct": sacct, "sinfo": sinfo}[cmd](rest))
