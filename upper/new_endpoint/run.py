"""Replay every class in an isolated process, binding partial receipts to the inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "STAGE30_UPPER_REVIEW_3775_2386.json"
SOURCE = HERE.parent / "STAGE21_ORDERED_780_493.json"
PREFIX_BITS = 2
EXPECTED_CLASSES = [0, 1, 3, 6, 7, 15, 22, 23, 24, 25, 27, 30, 60, 105]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_inputs():
    manifest = json.loads((HERE / "inputs.json").read_text())
    archive = json.loads(ARCHIVE.read_text())
    assert manifest["point"] == archive["r"] == "3775/2386"
    assert manifest["archive_sha256"] == digest(ARCHIVE)
    assert manifest["source_sha256"] == digest(SOURCE) == archive["source_sha256"]
    assert manifest["classes"] == EXPECTED_CLASSES and manifest["fields_per_class"] == 2**18
    assert set(map(int, archive["class_bounds"])) == set(EXPECTED_CLASSES)
    assert set(manifest["files"]) == {"model.txt", "owners.u16le"}
    for name, hashed in manifest["files"].items():
        assert digest(HERE / name) == hashed
    return manifest, archive


def receipt(path, c, record):
    lines = path.read_text().splitlines()
    rows = [line.split() for line in lines]
    assert rows[-1][:3] == ["PASS", "class", str(c)]
    assert rows[-1][3] == "seconds" and rows[-1][5] == "peak_rss_bytes"
    assert sum(r[0] == "PASS" for r in rows) == 1
    assert all(r[0] in {"value","prefix","bound","PASS"} for r in rows)
    words = {f for f,i,j in record["words"]}
    assert len(words) == len(record["words"]) and words
    proofs = {}
    for label, skipped in (("upper",0), ("low",len(words))):
        prefix = [tuple(map(int,r[2:])) for r in rows if r[:2] == ["prefix",label]]
        assert prefix == [(p, 1 << (18-PREFIX_BITS)) for p in range(1 << PREFIX_BITS)]
        bounds = [list(map(int,r[2:])) for r in rows if r[:2] == ["bound",label]]
        assert len(bounds) == 1
        checked, ignored, negative_ignored, field, sign = bounds[0]
        assert checked == 2**18-skipped and ignored == skipped
        assert 0 <= negative_ignored <= ignored and 0 <= field < 2**18 and sign in (0,1)
        assert label == "upper" or field not in words
        samples = [int(r[2]) for r in rows if r[:2] == ["value",label]]
        assert len(samples) == len(set(samples)) == 2
        assert sum(f in words for f in samples) == 1
        proofs[label] = dict(checked=checked, exempt=ignored, negative_exempt=negative_ignored,
                             minimum_field=field, minimum_sign=sign)
    return dict(source_class=c, bounds=proofs, seconds=float(rows[-1][4]),
                peak_rss_bytes=int(rows[-1][6]), log_sha256=digest(path))


def save(path, item):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(item, indent=2)+"\n")
    tmp.replace(path)


def run(resume):
    assert __debug__
    manifest, archive = read_inputs()
    binding = dict(kernel_source_sha256=digest(HERE/"blocked_raw.cpp"),
                   binary_sha256=digest(HERE/"blocked_raw"), inputs=manifest,
                   prefix_bits=PREFIX_BITS)
    progress = HERE / "progress.json"
    coverage = HERE / "raw_coverage.json"
    completed = {}
    if resume and progress.exists():
        previous = json.loads(progress.read_text())
        assert previous["binding"] == binding
        completed.update(previous["log_hashes"])
    elif resume and coverage.exists():
        previous = json.loads(coverage.read_text())
        assert previous["binding"] == binding
        completed.update({str(r["source_class"]):r["log_sha256"] for r in previous["classes"]})
    save(HERE / "binding.json", binding)
    save(progress, {"binding":binding, "log_hashes":completed})
    before = time.monotonic()
    reports = []
    for c in EXPECTED_CLASSES:
        log = HERE / f"class_{c}.log"
        if not log.exists() or completed.get(str(c)) != digest(log):
            partial = log.with_suffix(".partial")
            with partial.open("w") as out:
                subprocess.run([str(HERE/"blocked_raw"), str(HERE/"model.txt"),
                                str(HERE/"owners.u16le"), str(c), str(PREFIX_BITS)],
                               check=True, stdout=out)
            partial.replace(log)
        report = receipt(log, c, archive["class_bounds"][str(c)])
        reports.append(report)
        completed[str(c)] = report["log_sha256"]
        save(progress, {"binding":binding, "log_hashes":completed})
        print("PASS class", c, "total", len(reports), "of", len(EXPECTED_CLASSES), flush=True)
    upper = sum(r["bounds"]["upper"]["checked"] for r in reports)
    low = sum(r["bounds"]["low"]["checked"] for r in reports)
    high = sum(r["bounds"]["low"]["exempt"] for r in reports)
    assert upper == low + high == 3670016
    result = dict(binding=binding, classes=reports, upper_checked=upper, low_checked=low,
                  high_exempt=high, individual_peak_rss_bytes=max(r["peak_rss_bytes"] for r in reports),
                  sum_class_seconds=sum(r["seconds"] for r in reports))
    save(coverage, result)
    print("PASS 14 classes", upper, "full upper and", low, "complement lower-cap comparisons;",
          "maximum individual RSS bytes", result["individual_peak_rss_bytes"],
          "driver wall seconds", round(time.monotonic()-before, 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="use completed logs from the same binary, source, inputs and partition")
    parser.add_argument("--workdir", type=Path, default=HERE)
    options = parser.parse_args()
    HERE = options.workdir.resolve()
    run(options.resume)
