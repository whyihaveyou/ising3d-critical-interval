"""Independent replay for the Stage 57 finite-certificate interface."""

from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
TOOLCHAIN = (HERE / "lean-toolchain").read_text(encoding="utf-8").strip()
SOURCE = HERE / "STAGE57CertificateInterface.lean"
MODULE = "STAGE57CertificateInterface"
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}


def run(args, build, env, log):
    line = "$ " + " ".join(args)
    print(line, flush=True)
    print(line, file=log, flush=True)
    result = subprocess.run(
        args,
        cwd=build,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    print(result.stdout, end="", file=log)
    print(f"exit {result.returncode}")
    print(f"exit {result.returncode}", file=log, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout


def main():
    log_path = HERE / "STAGE57_CERTIFICATE_REPLAY.log"
    with log_path.open("w", encoding="utf-8") as log:
        with TemporaryDirectory(prefix="stage57_lean_", dir=HERE) as tmp:
            build = Path(tmp)
            env = dict(os.environ, LEAN_PATH=str(build))
            tool = ["elan", "run", TOOLCHAIN]
            print(f"source sha256 {sha256(SOURCE.read_bytes()).hexdigest()}")
            print(f"source sha256 {sha256(SOURCE.read_bytes()).hexdigest()}", file=log)
            run(tool + ["lean", "--version"], build, env, log)
            shutil.copyfile(SOURCE, build / SOURCE.name)
            out = run(
                tool + [
                    "lean",
                    "-DwarningAsError=true",
                    "-o",
                    MODULE + ".olean",
                    SOURCE.name,
                ],
                build,
                env,
                log,
            )
            if "sorryAx" in out:
                raise RuntimeError("unexpected sorryAx in Lean output")
            expected = set(re.findall(r"^#print axioms (\w+)$", SOURCE.read_text(), re.M))
            found = re.findall(
                r"'[^']+\.(\w+)' (?:depends on axioms: \[([^\]]*)\]|"
                r"does not depend on any axioms)",
                out,
            )
            found_names = {name for name, _ in found}
            if expected != found_names:
                raise RuntimeError(f"axiom report mismatch: expected {expected}, found {found_names}")
            for name, axioms in found:
                deps = {x.strip() for x in axioms.split(",") if x.strip()}
                if deps - ALLOWED_AXIOMS:
                    raise RuntimeError(f"unexpected axiom for {name}: {deps}")
            run(tool + ["leanchecker", "-v", MODULE], build, env, log)
    print("PASS Stage 57 finite-certificate interface")


if __name__ == "__main__":
    main()

