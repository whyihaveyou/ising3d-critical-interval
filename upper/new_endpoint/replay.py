"""Standalone entry points for the ordered endpoint 3775/2386."""

import argparse
from fractions import Fraction as F
import json
from math import floor
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import run as raw

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent
RECEIPTS = HERE / "receipts"
POINT = F(3775,2386)
ARCHIVE_SHA = "711f6a8446ecb17eea03384ab964a9dd60f0586be1401e9edd46a321c873f314"
SOURCE_SHA = "b8c8cec1644bf8bd29978d514643a7044377eb12fcd0e157d950cd2980bd2cb6"
INPUT_HASHES = {
    "model.txt":"98229bd878ded03d8db70077ae4d945edc00c4469b981eea8eaca85a1e4f38ce",
    "owners.u16le":"f18d73822bf2857df9f760a5429c8b2937abf1394c8f80504041308709ffcce5",
}
sys.path.insert(0,str(FROZEN))


def inputs():
    label, record = raw.read_inputs()
    assert (label["archive_sha256"],label["source_sha256"]) == (ARCHIVE_SHA,SOURCE_SHA)
    assert label["files"] == INPUT_HASHES
    return label, record


def arithmetic():
    import verify_stage33_upper_margin as baseline
    import margin
    _, record = inputs()
    # The frozen helper keeps the physical-row path as a separate constant.
    # Redirect only its endpoint-record directory to this subpackage.
    assert baseline.SOURCE == FROZEN / "STAGE21_ORDERED_780_493.json"
    baseline.HERE = HERE
    delta = F(3,2*10**14)
    baseline.verify(POINT,delta)
    u,q = F(record["U"]),F(record["Q"])
    assert F(record["L"]) == -u/q
    assert floor(-10**15*u/q) == 215386342
    uniform = -(u+18000000*delta/(POINT-delta))/q
    assert floor(10**15*uniform) == 44732026 and uniform > F(44732026,10**15)
    margin.run()


def receipts():
    label, record = inputs()
    path = RECEIPTS/"raw_coverage.json"
    assert raw.digest(path) == "ef2ccc7a0a34de5773b835b4a1e46669a06f9c48f060dd8739bc4824eeb25bbe"
    recorded = json.loads(path.read_text())
    assert recorded["binding"]["inputs"] == label
    assert recorded["binding"]["kernel_source_sha256"] == raw.digest(HERE/"blocked_raw.cpp")
    assert recorded["binding"]["prefix_bits"] == raw.PREFIX_BITS
    checked = []
    for c in raw.EXPECTED_CLASSES:
        path = RECEIPTS / f"class_{c}.log"
        checked.append(raw.receipt(path, c, record["class_bounds"][str(c)]))
    assert checked == recorded["classes"]
    for r in checked:
        for block in r["bounds"].values():
            assert block["minimum_sign"] == 1 and block["negative_exempt"] == block["exempt"]
    upper = sum(r["bounds"]["upper"]["checked"] for r in checked)
    low = sum(r["bounds"]["low"]["checked"] for r in checked)
    assert upper == recorded["upper_checked"] == 3670016
    assert low == recorded["low_checked"] == 3661604 and upper-low == recorded["high_exempt"] == 8412
    log = RECEIPTS / "directed.log"
    assert raw.digest(log) == "cbf88b30c58aac61f81d0808e26208909e6ba11ba00bb099b87d44cdc5371aa0"
    assert "PASS directed 14 classes, all 3670016 fields" in log.read_text().splitlines()[-1]
    print("PASS archived directed completion, 14 raw class digests, 3670016 upper and 3661604 complement checks")


def python(script,*args):
    env = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
        env[key] = "1"
    subprocess.run([sys.executable,"-E","-B","-u",str(HERE/script),*map(str,args)],env=env,check=True)


def compile_kernel(scratch):
    cpp = scratch/"blocked_raw.cpp"
    shutil.copyfile(HERE/"blocked_raw.cpp",cpp)
    flags = []
    prefix = os.environ.get("GMP_PREFIX")
    if not prefix and shutil.which("brew"):
        probe = subprocess.run(["brew","--prefix","gmp"],capture_output=True,text=True)
        if probe.returncode == 0:
            prefix = probe.stdout.strip()
    if prefix:
        flags = ["-isystem",str(Path(prefix)/"include"),"-L",str(Path(prefix)/"lib")]
    command = [os.environ.get("CXX","c++"),"-std=c++17","-O3",*flags,str(cpp),
               "-lgmpxx","-lgmp","-o",str(scratch/"blocked_raw")]
    subprocess.run(command,check=True)


def full_raw():
    frozen, record = inputs()
    with tempfile.TemporaryDirectory(prefix="ising-raw-3775-") as directory:
        scratch = Path(directory)
        python("prepare.py","--output",scratch)
        rebuilt = json.loads((scratch/"inputs.json").read_text())
        assert rebuilt == frozen
        compile_kernel(scratch)
        python("crosscheck.py","small","--kernel",scratch/"blocked_raw")
        python("run.py","--workdir",scratch)
        for c in raw.EXPECTED_CLASSES:
            original = RECEIPTS / f"class_{c}.log"
            new = scratch / original.name
            raw.receipt(new,c,record["class_bounds"][str(c)])
            # The last line records wall time and resident memory on this host.
            assert new.read_text().splitlines()[:-1] == original.read_text().splitlines()[:-1]
    print("PASS freshly rebuilt raw fields, all prefix counts, class minima and 56 sampled integers match")


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("run with Python assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("arithmetic","receipts","directed","raw","samples"))
    mode = parser.parse_args().mode
    if mode == "arithmetic":
        arithmetic()
    elif mode == "receipts":
        receipts()
    elif mode == "directed":
        inputs()
        python("prepare.py","--directed")
    elif mode == "raw":
        full_raw()
    else:
        inputs()
        python("crosscheck.py","samples")
