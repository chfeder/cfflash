#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2025

import os, sys, shutil, filecmp
from pathlib import Path
from cfpack import print, stop


# synchronise all str-matched grep in files from source to destination directory, excluding some subdirs
if __name__ == "__main__":

    # ========== USER INPUTS ==========
    SRC = Path("~/flash/source").expanduser() # source path
    DST = Path("./source").expanduser() # destination path
    NEEDLES = ["sink"]                 # terms to find (all must be present); case-insensitive
    EXCLUDE_DIR_SUBSTRS = [            # exclude any dir whose RELATIVE PATH contains any of these substrings
        ".git", ".hg", ".svn", ".venv", "venv", "__pycache__",
        "build", "dist", ".mypy_cache", ".pytest_cache",
        "flashUtilities",
        "IO/IOMain",
        "Particles/ParticlesMain/Sink/Outflow/Bturb",
        "Particles/ParticlesMain/Sink/StellarEvolution/standalone",
        "Particles/ParticlesMain/Sink/Supernova",
        "physics/Eos",
        "physics/Gravity/GravityMain/Poisson/BHTree",
        "physics/materialProperties",
        "physics/RadTrans",
        "physics/sourceTerms/Cool",
        "physics/sourceTerms/MolChem",
        "Simulation"
    ]
    manifest_path = Path("manifest.txt")
    # =================================

    if not SRC.is_dir():
        print(f"ERROR: SRC not found: {SRC}", file=sys.stderr, error=True)
    DST.mkdir(parents=True, exist_ok=True)

    # Normalize exclude substrings to POSIX style
    excl = [s.replace("\\", "/") for s in EXCLUDE_DIR_SUBSTRS]
    needles_bytes = [n.encode("utf-8").lower() for n in NEEDLES]

    matched_rel = []
    copied = 0
    unchanged = 0

    for dirpath, dirnames, filenames in os.walk(SRC):
        cur_rel = Path(dirpath).relative_to(SRC).as_posix()
        # Prune subdirs in-place if their REL path (from SRC) contains any exclude substring
        keep = []
        for d in dirnames:
            rel_dir = (Path(cur_rel) / d).as_posix() if cur_rel != "." else d
            if any(x in rel_dir for x in excl):
                continue
            keep.append(d)
        dirnames[:] = keep

        for fname in filenames:
            src = Path(dirpath) / fname
            rel = src.relative_to(SRC)
            rel_posix = rel.as_posix()
            # Skip if file path contains an excluded substring
            if any(x in rel_posix for x in excl):
                continue

            # Read bytes, case-insensitive search; no binary filtering
            try:
                data = src.read_bytes().lower()
            except Exception:
                continue
            if not all(nb in data for nb in needles_bytes):
                continue

            dst = DST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                    unchanged += 1
                else:
                    shutil.copy2(src, dst)
                    copied += 1
                matched_rel.append(rel_posix)
            except Exception as e:
                print(f"Warning copying {src} -> {dst}: {e}", file=sys.stderr)

    # De-dup + sort
    matched_rel = sorted(set(matched_rel))

    # Stale cleanup
    previous = []
    if manifest_path.exists():
        try:
            previous = [l.strip() for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception:
            previous = []
    to_delete = sorted(set(previous) - set(matched_rel))
    deleted = 0
    for rel in to_delete:
        victim = DST / rel
        try:
            if victim.exists():
                victim.unlink()
                deleted += 1
        except Exception as e:
            print(f"Warning removing stale {victim}: {e}", file=sys.stderr)

    # Prune empty dirs (best-effort)
    for dpath, dnames, fnames in os.walk(DST, topdown=False):
        p = Path(dpath)
        if p == DST:
            continue
        try:
            if not any(p.iterdir()):
                p.rmdir()
        except Exception:
            pass

    # Write manifest
    manifest_path.write_text("\n".join(matched_rel) + ("\n" if matched_rel else ""), encoding="utf-8")

    print(f"source dir  (SRC): {SRC}", color='lightblue_ex')
    print(f"destination (DST): {DST}", color='lightblue_ex')
    print(f"Matched:   {len(matched_rel)}", color='yellow')
    print(f"Copied:    {copied}", color='yellow')
    print(f"Unchanged: {unchanged}", color='yellow')
    print(f"Deleted:   {deleted}", color='yellow')
    print(f"Manifest:  {manifest_path}", color='green')
