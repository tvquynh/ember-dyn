#!/usr/bin/env python3
# Redact best-effort PII + sandbox-infrastructure identifiers from EMBER-Dyn dynamic
# Small and Full tarballs, and exclude binary-payload + analyzer-diagnostic report sections.
#
# Three passes: (1) stream-decompress + extract members to a temp dir; (2) process every
# .json member in a parallel pool (fail-loud: abort if any member cannot be processed);
# (3) re-archive to .tar.zst (zstd-19).
#
# Per-report processing (VALUE level, on the parsed JSON, so escapes are never broken and
# both backslash and forward-slash paths are handled):
#   - remove sections: top-level dropped / procdump / procmemory / debug, and CAPE.payloads
#   - scrub the analyzer machine identity (info.machine name / label / ip / timestamps)
#   - redact, in every string value and key:
#       Windows user paths and the analysis account 'cape' in ALL forms -- drive-letter,
#       %SystemDrive%, device-path, the Avast-CTU reduced-schema separator rewrites
#       ( \  /  :  |  _ ) used in mutex/registry/file strings, and the bare account token in
#       command arguments (e.g. /RU "cape", USERNAME eq cape); the constant analyzer guest SID
#       (S-1-5-21-... machine/domain SID, any case); CAPE roots and any ..CAPE.. path,
#       /opt/CAPEv2 paths, capemon module, /home/runner build paths, ..analysis..<id>,
#       emails, RFC1918 private/sandbox IPs, /user:, password
#   - KEEPS standard Windows directories (Program Files, Windows, ...), public IPs, network
#     domains (research IOCs), CAPEv2 product name, and well-known SIDs (S-1-5-18/19/20/32-*).
#     Output re-serialised with json.dumps and re-parsed (validity).
#
# Usage: python redact_pii.py <in.tar.zst> <out.tar.zst> [--workers N] [--tmp DIR]
from __future__ import annotations
import argparse, json, multiprocessing as mp, os, re, shutil, tarfile, tempfile, time
from pathlib import Path
import zstandard as zstd

SEP   = r'[\x5c/]+'
NOSEP = r'[^\x5c/"<>\s]'
USERANY = re.compile(r'(?i)(' + SEP + r'Users' + SEP + r')(?!Public|Default|All Users)(' + NOSEP + r'{1,64})')
# reduced-schema separator-rewritten account home: users<any-sep(s)>cape  (covers : | _ and all-caps)
USERSEP = re.compile(r'(?i)(users)([\x5c/:|_]+)cape(?![a-z0-9])')
CAPEW = re.compile(r'(?i)[A-Za-z]:' + SEP + r'CAPE(?:' + SEP + r'[^\s"<>]{0,400})?')
CAPEANY = re.compile(r'[\x5c/]CAPE(?![A-Za-z0-9])[^\s"<>]{0,200}')
CAPEMON = re.compile(r'(?i)capemon[A-Za-z0-9._-]*')
# standalone analysis-account token under any rewritten separator (incl underscore); cape+alnum excluded
CAPESEP = re.compile(r'(?i)([\x5c/:|_])cape(?![a-z0-9])')
# value-start account token immediately followed by a hard path separator (e.g. "cape:AppData")
CAPEVAL = re.compile(r'(?i)\bcape(?=[\x5c/:|])')
CAPEUSER = re.compile(r'([\x5c/])cape(?![A-Za-z0-9])')
# standalone lowercase account token bounded by non-alphanumerics (command args: /RU "cape",
# USERNAME eq cape). Case-sensitive + excludes a trailing underscore, so it does NOT touch the
# CAPE provider key, cape_current_commit/cape_type/cape_yara, cape0.bat, CAPEv2, capemon, escape, or landscape.
CAPEWORD = re.compile(r'(?<![A-Za-z0-9])cape(?![A-Za-z0-9_])')
SBOX  = re.compile(r'/opt/CAPEv2[^\s"<>]*')
ANALYS= re.compile(r'(?i)([\x5c/])analysis([\x5c/])\d+')
BUILD = re.compile(r'/home/runner/[^\s"<>]{1,400}')
EMAIL = re.compile(r'[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,64}\.[A-Za-z]{2,16}')
NATIP = re.compile(r'(?<!\d)(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})')
UFLAG = re.compile(r'(?i)(/user:)([^\s"<>]{1,64})')
PWD   = re.compile(r'(?i)(password\s*[:=])\s*([^\s";,<>]{1,128})')
# machine/domain account SID (any case, any sub-authority count); well-known
# SIDs (S-1-5-18/19/20, S-1-5-32-*) are NOT matched (only the -21- family).
SID   = re.compile(r'(?i)S-1-5-21(?:-\d+)+')
DUP   = re.compile(r'(<user>){2,}')

def _red(s):
    s = USERANY.sub(r'\1<user>', s)
    s = USERSEP.sub(r'\1\2<user>', s)
    s = CAPEW.sub('<analyzer-path>', s)
    s = CAPEANY.sub('<analyzer-path>', s)
    s = CAPEMON.sub('<analyzer-module>', s)
    s = CAPESEP.sub(r'\1<user>', s)
    s = CAPEVAL.sub('<user>', s)
    s = CAPEUSER.sub(r'\1<user>', s)
    s = CAPEWORD.sub('<user>', s)
    s = SBOX.sub('<sandbox-storage>', s)
    s = ANALYS.sub(r'\1analysis\g<2><id>', s)
    s = BUILD.sub('<build-path>', s)
    s = EMAIL.sub('<email>', s)
    s = NATIP.sub('<sandbox-ip>', s)
    s = UFLAG.sub(r'\1<user>', s)
    s = PWD.sub(r'\1<redacted>', s)
    s = SID.sub('<sandbox-sid>', s)
    s = DUP.sub('<user>', s)
    return s

def _walk(o):
    if isinstance(o, dict):  return {(_red(k) if isinstance(k, str) else k): _walk(v) for k, v in o.items()}
    if isinstance(o, list):  return [_walk(v) for v in o]
    if isinstance(o, str):   return _red(o)
    return o

def _scrub_machine(r):
    info = r.get("info")
    if isinstance(info, dict) and isinstance(info.get("machine"), dict):
        mc = info["machine"]
        for k in ("name", "label"):
            if mc.get(k): mc[k] = "<sandbox-host>"
        for k in ("started_on", "shutdown_on"):
            if mc.get(k): mc[k] = "<ts>"
        if mc.get("ip"): mc["ip"] = "<sandbox-ip>"

def process_report(raw: bytes) -> bytes:
    r = json.loads(raw)
    for k in ("dropped", "procdump", "procmemory", "debug"): r.pop(k, None)
    cape = r.get("CAPE")
    if isinstance(cape, dict): cape.pop("payloads", None)
    _scrub_machine(r)
    r = _walk(r)
    out = json.dumps(r, ensure_ascii=False).encode("utf-8")
    json.loads(out)
    return out

def _worker(path_str):
    p = Path(path_str)
    try:
        p.write_bytes(process_report(p.read_bytes()))
        return (path_str, 1, "")
    except Exception as e:
        return (path_str, 0, repr(e)[:200])

def main(in_path, out_path, workers, tmp_root):
    t0 = time.time()
    in_p, out_p = Path(in_path), Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    wd = Path(tempfile.mkdtemp(prefix="_pii_", dir=tmp_root or out_p.parent))
    try:
        with open(in_p, "rb") as f, zstd.ZstdDecompressor().stream_reader(f) as rd:
            with tarfile.open(fileobj=rd, mode="r|") as tar:
                try: tar.extractall(wd, filter="data")
                except TypeError: tar.extractall(wd)
        files = [str(p) for p in wd.rglob("*.json")]
        res = {}
        try:
            with mp.Pool(workers) as pool:
                for (p, ok, err) in pool.imap_unordered(_worker, files, chunksize=4):
                    res[p] = (ok, err)
        except Exception as e:
            print(f"  POOL_BROKE {e!r} -> sequential", flush=True)
        still = []
        for p in [p for p in files if res.get(p, (0,))[0] != 1]:
            r = _worker(p)
            if r[1] != 1: still.append((p, r[2]))
        if still:
            raise SystemExit(f"ABORT: {len(still)} unrecoverable: "
                             + "; ".join(os.path.basename(p) + '->' + e for p, e in still[:5]))
        cctx = zstd.ZstdCompressor(level=19, threads=-1)
        with open(out_p, "wb") as g, cctx.stream_writer(g) as w:
            with tarfile.open(fileobj=w, mode="w|") as tw:
                for p in sorted(files):
                    tw.add(p, arcname=Path(p).relative_to(wd).as_posix(), recursive=False)
        print(f"[redact] {in_p.name} -> {out_p.name}: {len(files)} members, "
              f"{out_p.stat().st_size/1e6:.1f} MB ({time.time()-t0:.0f}s)", flush=True)
    finally:
        shutil.rmtree(wd, ignore_errors=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--tmp", default=None)
    a = ap.parse_args()
    main(a.input, a.output, a.workers, a.tmp)
