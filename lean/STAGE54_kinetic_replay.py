"""Recompile and independently recheck the chord count on periodic cubic graphs."""

from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from lean_common import ROOT, TOOLCHAIN, run


def replay():
    accepted = {"propext", "Quot.sound", "Classical.choice"}
    with TemporaryDirectory(prefix="STAGE54_kinetic_", dir=ROOT) as tmp:
        work = Path(tmp)
        env = dict(os.environ, LEAN_PATH=str(work))
        elan = ["elan", "run", TOOLCHAIN]
        run([*elan, "lean", "--version"], work, env)
        for name in ("STAGE54Kinetic", "STAGE54Torus", "STAGE54PrismData", "STAGE54Prism"):
            source = ROOT / (name + ".lean")
            print(name, "sha256", sha256(source.read_bytes()).hexdigest(), flush=True)
            shutil.copyfile(source, work / source.name)
            output = run([*elan, "lean", "-DwarningAsError=true", "-o",
                          name + ".olean", source.name], work, env)
            requested = set(re.findall(r"^#print axioms (\w+)$", source.read_text(), re.M))
            reports = re.findall(r"'[^']+\.(\w+)' (?:depends on axioms: \[([^]]*)\]"
                                 r"|does not depend on any axioms)", output)
            if (not requested and name != "STAGE54PrismData") or {t for t,_ in reports} != requested:
                raise RuntimeError("incomplete declaration check")
            for theorem,axioms in reports:
                dependencies = {a.strip() for a in axioms.split(",") if a.strip()}
                if not dependencies <= accepted:
                    raise RuntimeError("unexpected dependency of " + theorem)
            run([*elan, "leanchecker", "-v", name], work, env)
            print(len(requested), "declarations rechecked")
    print("PASS all-length weighted torus paths, terminal contacts, and finite prism allocations")


if __name__=="__main__":
    replay()
