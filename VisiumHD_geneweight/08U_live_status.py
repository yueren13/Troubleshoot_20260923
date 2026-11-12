#!/usr/bin/env python3
"""Read-only 08U live status monitor (Linux, Python standard library only).

Run in a NEW JupyterLab terminal, while the analysis continues:
    python3 08U_live_status.py --project . --sample Ada-6
Optional: --pid PID (the busy notebook kernel or spatial worker), --seconds 60.
Writes only its own timestamped diagnostic folder and ZIP. Does not signal,
attach a debugger to, import analysis libraries into, or modify the target.
Only small known metadata/log files are read; expression matrices are not read.
Counter reference: https://docs.kernel.org/filesystems/proc.html
Pressure reference: https://docs.kernel.org/accounting/psi.html
GPU utilization is device-wide; other workloads can contribute. Linux read_bytes
counts storage I/O, so cached reads can advance rchar without read_bytes.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile

VERSION = "08U-live-status-1.0.0"
PROC = Path("/proc")
TICKS = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")
MAX_SMALL = 1_048_576
FILES = ("job.json", "console.log", "mean.inprogress.npy", "gradient.inprogress.npy",
         "mean.npy", "gradient.npy", "focal_global_rows.npy", "spatial_blocks.json",
         "spatial_spec.json")


def read_text(path, limit=MAX_SMALL):
    try:
        path = Path(path)
        if not stat.S_ISREG(path.stat().st_mode):
            return None
        with path.open("rb") as handle:
            return handle.read(limit).decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def small_json(path):
    try:
        path = Path(path)
        if path.stat().st_size > MAX_SMALL:
            return None
        value = json.loads(read_text(path) or "null")
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def process(pid):
    """A /proc snapshot; missing/restricted counters remain None."""
    try:
        folder = PROC / str(pid)
        if folder.stat().st_uid != os.getuid():
            return None
        raw = read_text(folder / "stat", 8192)
        fields = raw[raw.rfind(")") + 2:].split()
        status = {}
        for line in (read_text(folder / "status", 65536) or "").splitlines():
            key, _, val = line.partition(":")
            status[key] = val.strip()
        def kb(name):
            value = status.get(name)
            return int(value.split()[0]) * 1024 if value else None
        io = {}
        for line in (read_text(folder / "io", 8192) or "").splitlines():
            key, _, val = line.partition(":")
            try:
                io[key] = int(val.strip())
            except ValueError:
                pass
        return {"pid": int(pid), "ppid": int(fields[1]), "state": fields[0],
                "start_ticks": int(fields[19]),
                "cpu_seconds": (int(fields[11]) + int(fields[12])) / TICKS,
                "rss_bytes": kb("VmRSS"), "swap_bytes": kb("VmSwap"),
                "threads": int(fields[17]), "io": io,
                "wchan": (read_text(folder / "wchan", 256) or "").strip() or None}
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def ancestors():
    found, pid = set(), os.getpid()
    while pid and pid not in found:
        found.add(pid)
        item = process(pid)
        pid = item["ppid"] if item else 0
    return found


def option(argv, flag):
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def catalog(excluded):
    """Read same-UID process metadata; never retain complete command lines."""
    found = {}
    for entry in PROC.iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        item = process(int(entry.name))
        if item is None:
            continue
        argv = (read_text(entry / "cmdline", 65536) or "").split("\0")
        bases = [Path(a).name for a in argv if a]
        item["kind"] = "process"
        if any("ipykernel" in a for a in argv):
            item["kind"] = "notebook kernel"
        if "tcell_banksy_gpu.py" in bases:
            stage = option(argv, "--stage")
            item["kind"] = "08U " + str(stage or "worker")
            job = option(argv, "--job")
            if job and stage == "spatial":
                if not Path(job).is_absolute():
                    try:
                        job = str((entry / "cwd").resolve() / job)
                    except OSError:
                        job = None
                if job:
                    item["job_path"] = job
        found[item["pid"]] = item
    return found


def descendants(items, roots):
    selected = set(roots)
    while True:
        more = {pid for pid, p in items.items() if p["ppid"] in selected}
        new = selected | more
        if new == selected:
            return new
        selected = new


def delta(before, after, elapsed):
    if not before or not after or before["start_ticks"] != after["start_ticks"]:
        return {"identity_changed_or_first_seen": True}
    value = {"cpu_percent_one_core_100": max(0., (after["cpu_seconds"] - before["cpu_seconds"]) * 100 / elapsed)}
    for key in ("read_bytes", "write_bytes", "rchar", "wchar", "syscr", "syscw"):
        a, b = before["io"].get(key), after["io"].get(key)
        value[key + "_delta"] = max(0, b - a) if a is not None and b is not None else None
    return value


def system_snapshot():
    memory = {}
    for line in (read_text(PROC / "meminfo", 16384) or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Dirty", "Writeback"):
            memory[key] = int(value.split()[0]) * 1024
    vm = {}
    for line in (read_text(PROC / "vmstat", 65536) or "").splitlines():
        key, _, value = line.partition(" ")
        if key in ("pswpin", "pswpout", "pgmajfault", "oom_kill"):
            vm[key] = int(value)
    psi = {key: read_text(PROC / "pressure" / key, 2048) for key in ("cpu", "memory", "io")}
    cgroup = {}
    for line in (read_text(PROC / "self/cgroup", 8192) or "").splitlines():
        if line.startswith("0::"):
            rel = line.split("::", 1)[1].lstrip("/")
            if ".." not in Path(rel).parts:
                base = Path("/sys/fs/cgroup") / rel
                for name in ("memory.current", "memory.max", "memory.events", "memory.pressure"):
                    cgroup[name] = read_text(base / name, 8192)
    return {"memory_bytes": memory, "vm_counters": vm, "pressure": psi,
            "monitor_cgroup_v2": cgroup, "loadavg": read_text(PROC / "loadavg", 256)}


def gpu_snapshot():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "note": "nvidia-smi not found"}
    command = [exe, "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
               "--format=csv,noheader,nounits"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=4, check=False)
        return {"available": done.returncode == 0, "returncode": done.returncode,
                "columns": "index,name,gpu_percent,memory_controller_percent,used_MiB,total_MiB",
                "stdout": done.stdout[:8192], "stderr": done.stderr[:2048]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "note": type(exc).__name__ + ": " + str(exc)[:300]}


def bounded_dirs(path, limit=64):
    try:
        with os.scandir(path) as entries:
            result = []
            for i, entry in enumerate(entries):
                if i >= 512 or len(result) >= limit:
                    break
                if entry.is_dir(follow_symlinks=False):
                    result.append(Path(entry.path))
            return result
    except OSError:
        return []


def discover(project, sample):
    """Only known 08U layout, at most four ancestors; no recursive project walk."""
    roots = set()
    for parent in [project, *list(project.parents)[:4]]:
        if parent == Path("/"):
            continue
        if parent.name == "20_tcell_banksy_harmony":
            roots.add(parent)
        direct = parent / "20_tcell_banksy_harmony"
        if direct.is_dir():
            roots.add(direct)
        for run in bounded_dirs(parent / "derived" / "08_ct26_joint_scvi", 32):
            direct = run / "20_tcell_banksy_harmony"
            if direct.is_dir():
                roots.add(direct)
    found, inspected = [], 0
    for root in sorted(roots)[:8]:
        for source in bounded_dirs(root, 32):
            for spatial in bounded_dirs(source / "spatial", 32):
                for folder in bounded_dirs(spatial, 64):
                    inspected += 1
                    if inspected > 512:
                        return sorted(set(found))[-32:]
                    if not re.fullmatch(r"sample_\d+", folder.name):
                        continue
                    job = small_json(folder / "job.json")
                    if job and str(job.get("sample_id")) == sample:
                        found.append(folder)
    return sorted(set(found))[-32:]


def file_stat(path):
    try:
        s = path.stat()
        return {"exists": True, "bytes": s.st_size, "mtime_ns": s.st_mtime_ns,
                "allocated_bytes": s.st_blocks * 512 if hasattr(s, "st_blocks") else None}
    except OSError:
        return {"exists": False}


def tail(path, limit=4096):
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            return None
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - limit))
            return handle.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return None


def sample_snapshot(folder):
    job = small_json(folder / "job.json") or {}
    spec = small_json(folder / "spatial_spec.json") or {}
    files = {name: file_stat(folder / name) for name in FILES}
    match = bool(job.get("job_fingerprint") and spec.get("job_fingerprint") == job["job_fingerprint"])
    outputs_exist = all(Path(spec[k]).is_file() for k in ("mean_path", "gradient_path", "global_rows_path") if isinstance(spec.get(k), str)) if spec else False
    outputs_exist = outputs_exist and all(isinstance(spec.get(k), str) for k in ("mean_path", "gradient_path", "global_rows_path"))
    matrix = Path(job["matrix_path"]) if isinstance(job.get("matrix_path"), str) else None
    manifest = (small_json(matrix.parent / "sample_manifest.json") or {}) if matrix else {}
    details = {key: manifest[key] for key in ("n_tissue_cells", "n_focal_cells", "n_features") if key in manifest}
    if isinstance(job.get("gene_names"), list):
        details.setdefault("n_features", len(job["gene_names"]))
    if match and outputs_exist:
        phase = "Matching completion receipt and output files exist (could be a prior compatible run)."
    elif files["mean.npy"]["exists"] and files["gradient.npy"]["exists"]:
        phase = "Final arrays exist; final hashing/receipt may still be pending, or files may be from a prior run."
    elif files["mean.inprogress.npy"]["exists"] or files["gradient.inprogress.npy"]["exists"]:
        phase = "In-progress arrays exist: graph setup was reached in this or an earlier attempt."
    else:
        phase = "No spatial arrays yet: may be loading input, initializing CUDA, or building CPU neighbor graph."
    try:
        space = shutil.disk_usage(folder)
        filesystem = {"total": space.total, "used": space.used, "free": space.free}
    except OSError:
        filesystem = None
    return {"directory": str(folder), "sample_id": job.get("sample_id"),
            "filesystem_bytes": filesystem, "size_metadata": details,
            "job_fingerprint": job.get("job_fingerprint"), "receipt_fingerprint_matches": match,
            "receipt_and_outputs_present": bool(match and outputs_exist), "phase_hint": phase,
            "files": files, "console_tail": tail(folder / "console.log"),
            "source_matrix_stat": file_stat(matrix) if matrix else None,
            "statistics_complete": file_stat(folder.parent / "statistics_complete.json")}


def gib(value):
    return "?" if value is None else f"{value / 2**30:.2f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=".", help="Notebook/project folder, or 20_tcell_banksy_harmony folder")
    parser.add_argument("--sample", default="Ada-6")
    parser.add_argument("--pid", type=int, help="Busy notebook kernel or spatial-worker PID; optional")
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--interval", type=float, default=5.)
    parser.add_argument("--output", help="New diagnostic output directory; default is timestamped in current directory")
    parser.add_argument("--no-gpu", action="store_true", help="Skip bounded nvidia-smi queries")
    args = parser.parse_args(argv)
    if not (0 < args.seconds <= 600 and .2 <= args.interval <= 60):
        parser.error("Use 0 < --seconds <= 600 and 0.2 <= --interval <= 60.")
    if sys.platform != "linux":
        parser.error("Run this script inside the Linux workspace hosting the analysis.")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output or f"08U_live_status_{stamp}_{os.getpid()}").expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    lines = []
    def say(text=""):
        print(text, flush=True)
        lines.append(text)
    excluded = ancestors()
    if args.pid in excluded:
        parser.error("Choose the other busy kernel/worker; the monitor and its ancestors are excluded.")
    initial = catalog(excluded)
    jobs = {}
    eligible = descendants(initial, {args.pid}) if args.pid in initial else (set() if args.pid else set(initial))
    for pid, item in initial.items():
        if pid in eligible and item.get("job_path"):
            job = small_json(item["job_path"])
            if job and str(job.get("sample_id")) == args.sample:
                jobs[pid] = Path(item["job_path"]).parent
    roots = ({args.pid} if args.pid in initial else set()) if args.pid else set(jobs)
    if args.pid and args.pid not in initial:
        say(f"PID {args.pid} is absent, inaccessible, or not owned by your user.")
    kernels = [p for p in initial.values() if p["kind"] == "notebook kernel"]
    project = Path(args.project).expanduser().absolute()
    folders = sorted(set(jobs.values()))
    if not folders:
        folders = discover(project, args.sample)
    for pid in list(roots):
        if args.pid:
            continue
        item = initial.get(pid)
        for _ in range(8):
            if not item or item["ppid"] in excluded:
                break
            parent = initial.get(item["ppid"])
            if parent and parent["kind"] == "notebook kernel":
                roots.add(parent["pid"])
                break
            item = parent
    if not roots:
        say("No matching active spatial worker was found. Candidate notebook kernels are listed below;")
        say("none has been automatically selected. Rerun with --pid PID if needed.")
    if len(jobs) > 1:
        say("Multiple workers match this sample; reporting all of them, without guessing the intended run.")
    say(f"{VERSION} | sample={args.sample} | observation={args.seconds:g}s")
    say("Read-only observation; target processes are not paused or modified.")
    say("CPU 100% means one fully used core; >100% is valid for threaded work.")
    for p in kernels:
        say(f"Candidate kernel PID {p['pid']}: state={p['state']}, RSS={gib(p['rss_bytes'])} GiB")
    for pid, folder in jobs.items():
        say(f"Matching spatial worker PID {pid}: {folder}")
    for folder in folders:
        say(f"Watching sample output: {folder}")
    say("Section 3 mixes CPU neighbor-graph work, GPU gene chunks, file writes, and CPU hashing.")
    say("Preallocated .inprogress.npy size/mtime can stay unchanged during useful work.")
    say("GPU utilization is device-wide and may include other processes.")
    root_identities = {pid: initial[pid]["start_ticks"] for pid in roots if pid in initial}
    records, previous, previous_time = [], {}, None
    started = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            items = catalog(excluded)
            for pid in list(roots):
                if pid in items and pid in root_identities and items[pid]["start_ticks"] != root_identities[pid]:
                    say(f"PID {pid} was reused; removing the old process identity from selection.")
                    roots.remove(pid)
            # A supplied PID confines discovery to its original process tree.
            eligible = set(items)
            if args.pid:
                original = initial.get(args.pid)
                current_root = items.get(args.pid)
                eligible = descendants(items, {args.pid}) if (original and current_root and
                    original["start_ticks"] == current_root["start_ticks"]) else set()
            for pid, item in items.items():
                if pid in eligible and item.get("job_path"):
                    job = small_json(item["job_path"])
                    if job and str(job.get("sample_id")) == args.sample:
                        roots.add(pid)
                        root_identities[pid] = item["start_ticks"]
                        folder = Path(item["job_path"]).parent
                        if folder not in folders:
                            folders.append(folder)
            selected = descendants(items, roots)
            current = {pid: items[pid] for pid in selected if pid in items}
            elapsed = max(.0001, now - previous_time) if previous_time is not None else None
            interval = {pid: delta(previous.get(pid), p, elapsed) if elapsed else {}
                        for pid, p in current.items()}
            uptime = float((read_text(PROC / "uptime", 128) or "0").split()[0])
            for p in current.values():
                p["age_seconds"] = max(0., uptime - p["start_ticks"] / TICKS)
            system = system_snapshot()
            if records:
                old_vm = records[-1]["system"]["vm_counters"]
                system["vm_deltas"] = {k: max(0, v - old_vm[k]) for k, v in system["vm_counters"].items() if k in old_vm}
            gpu = {"skipped": True} if args.no_gpu else gpu_snapshot()
            sample_states = [sample_snapshot(folder) for folder in folders]
            record = {"observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "elapsed_seconds": now - started, "processes": list(current.values()),
                      "interval": interval, "system": system, "gpu": gpu, "samples": sample_states}
            records.append(record)
            say(f"\n[{now-started:5.1f}s] host MemAvailable={gib(system['memory_bytes'].get('MemAvailable'))} GiB")
            if system.get("vm_deltas"):
                say("Host VM counter deltas: " + json.dumps(system["vm_deltas"], sort_keys=True))
            for pid, p in sorted(current.items()):
                d = interval[pid]
                cpu = d.get("cpu_percent_one_core_100")
                cpu_text = "baseline" if cpu is None else f"{cpu:.1f}%"
                rb, wb = d.get("read_bytes_delta"), d.get("write_bytes_delta")
                io_text = "?" if rb is None or wb is None else f"{rb / 2**20:.2f}/{wb / 2**20:.2f}"
                say(f"PID {pid} {p['kind']}: age={p['age_seconds']/60:.1f}min state={p['state']} CPU={cpu_text} RSS={gib(p['rss_bytes'])}GiB swap={gib(p['swap_bytes'])}GiB disk-read/write={io_text}MiB wchan={p['wchan']}")
            if not current:
                say("No selected process is currently visible.")
            if gpu.get("stdout"):
                say("GPU [index,name,GPU%,memory-controller%,usedMiB,totalMiB]: " + gpu["stdout"].strip().replace("\n", " | "))
            elif not args.no_gpu:
                say("GPU query unavailable: " + str(gpu.get("note") or gpu.get("stderr") or "unknown"))
            for state in sample_states:
                say(f"{Path(state['directory']).name}: {state['phase_hint']}")
                if state["filesystem_bytes"]:
                    say(f"  Output filesystem free={gib(state['filesystem_bytes']['free'])} GiB")
            (output / "report.json").write_text(json.dumps({"version": VERSION, "sample": args.sample,
                "project": str(project), "candidate_kernel_pids": [p['pid'] for p in kernels],
                "observations": records}, indent=2), encoding="utf-8")
            (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            previous, previous_time = current, now
            remaining = args.seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            time.sleep(min(args.interval, remaining))
    except KeyboardInterrupt:
        say("Stopped only the monitor; the analysis was not signaled.")
    say("\nInterpretation: CPU/GPU/I/O activity shows recent work, not proof of healthy forward progress.")
    say("A quiet 30-second interval alone cannot establish a deadlock. Compare intervals and file phases.")
    say("Memory pressure and swap-in/out increases are included in report.json; GPU idle can occur in CPU/I/O phases.")
    say("Output files and completion receipts may predate this attempt; the report checks receipt/job fingerprints.")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive_path = Path(str(output) + ".zip")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("summary.txt", "report.json"):
            path = output / name
            if path.is_file():
                archive.write(path, name)
    print(f"\nReturn this small diagnostic ZIP: {archive_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
