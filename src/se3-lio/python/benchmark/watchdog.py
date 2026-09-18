"""Standalone resource watchdog for benchmark runs (or any command).

Wraps a command in its own process group, samples system RAM from /proc/meminfo
(stdlib only -- no psutil, no image rebuild), and kills the whole group if RAM
stays above a threshold. CPU/load are reported but never trigger a kill (a sweep
is meant to saturate CPU). Fully decoupled from run.py: the benchmark code is
untouched; combos already save their TUMs as they finish, so an aborted run is
still scorable with `benchmark.score`.

    python -m benchmark.watchdog [--max-ram-pct 90] [--interval 2] [--hits 3] \
        [--grace 10] -- python -m benchmark.run oxspires
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def _meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            info[k] = int(rest.split()[0])  # kB
    return info


def _ram_pct():
    """Used RAM as a percentage, from MemAvailable (accurate headroom)."""
    m = _meminfo()
    total = m["MemTotal"]
    avail = m.get("MemAvailable", m["MemFree"] + m.get("Buffers", 0) + m.get("Cached", 0))
    return 100.0 * (total - avail) / total


def _cpu_times():
    """(total, idle) jiffies from /proc/stat for a CPU%% delta between samples."""
    with open("/proc/stat") as f:
        vals = list(map(int, f.readline().split()[1:]))
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    return sum(vals), idle


def _split_argv(argv):
    """Split into (watchdog opts, command-after-'--')."""
    if "--" not in argv:
        return argv, None
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opt_argv, cmd = _split_argv(argv)
    ap = argparse.ArgumentParser(
        prog="benchmark.watchdog",
        description="Run a command under a RAM watchdog: '... -- <command>'.",
    )
    ap.add_argument("--max-ram-pct", type=float, default=90.0,
                    help="kill when used RAM%% stays >= this (default 90)")
    ap.add_argument("--interval", type=float, default=2.0, help="sample period s (default 2)")
    ap.add_argument("--hits", type=int, default=3,
                    help="consecutive over-threshold samples before killing (default 3)")
    ap.add_argument("--grace", type=float, default=10.0,
                    help="seconds between SIGTERM and SIGKILL (default 10)")
    args = ap.parse_args(opt_argv)
    if not cmd:
        ap.error("no command: put it after '--', e.g. '... -- python -m benchmark.run oxspires'")

    print(f"[watchdog] RAM kill >= {args.max_ram_pct:.0f}% x{args.hits} "
          f"(every {args.interval}s) -- {' '.join(cmd)}", flush=True)
    child = subprocess.Popen(cmd, start_new_session=True)  # own process group
    pgid = os.getpgid(child.pid)

    def _kill(reason):
        print(f"[watchdog] {reason} -> SIGTERM group {pgid}", flush=True)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.grace and child.poll() is None:
            time.sleep(0.5)
        if child.poll() is None:
            print(f"[watchdog] still alive after {args.grace:.0f}s -> SIGKILL", flush=True)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait()

    over = 0
    prev = _cpu_times()
    try:
        while True:
            rc = child.poll()
            if rc is not None:
                print(f"[watchdog] command exited ({rc})", flush=True)
                return rc
            time.sleep(args.interval)
            ram = _ram_pct()
            tot, idle = _cpu_times()
            dt, di = tot - prev[0], idle - prev[1]
            cpu = 100.0 * (1 - di / dt) if dt > 0 else 0.0
            prev = (tot, idle)
            over = over + 1 if ram >= args.max_ram_pct else 0
            print(f"[watchdog] RAM {ram:5.1f}%  CPU {cpu:5.1f}%"
                  + (f"  OVER {over}/{args.hits}" if over else ""), flush=True)
            if over >= args.hits:
                _kill(f"RAM {ram:.1f}% >= {args.max_ram_pct:.0f}% x{args.hits}")
                print("[watchdog] aborted on overload; finished combos kept their TUMs "
                      "-> rerun with fewer --jobs, or score what completed", flush=True)
                return 137
    except KeyboardInterrupt:
        _kill("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
