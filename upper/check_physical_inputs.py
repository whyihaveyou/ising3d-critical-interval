"""Low-memory entry into the same geometry and moment modules as the full-field checker."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import sys

import numpy as np
import STAGE30_UPPER_REVIEW_bridge as bridge
import verify_ordered_136_endpoint_extension_20261121 as lineage
import verify_deleted_walsh_order as walsh
from verify_boundary_reflection_blocks import layout
from verify_conditional_residual_spectrum import class_matrices
from verify_free_channel import improve_watson
from verify_pair_energy_test import return_terms


def main():
    point = Fraction(5441, 3439)
    assert walsh.fields.model is walsh.geometry
    assert len(walsh.geometry.BOUNDARY) == 18
    classes = walsh.fields.corner_classes(list(product((-1, 1), repeat=3)))
    assert len(classes) == 14
    assert [len(r) for r in lineage.audited_record().values()] == [91, 14, 136]
    source, _ = bridge.precise.base.stage.source()
    assert [len(source[k]) for k in ("translations", "ordinary", "walsh")] == [91, 14, 136]
    assert list(map(len, layout("axis"))) == [32, 32, 256]
    assert list(map(len, layout("diagonal"))) == [128, 128, 16]
    assert sum(1 for _ in return_terms(16)) == 17
    low, high = improve_watson()
    assert low < high and callable(class_matrices)
    data = np.asarray([2, 3, 5, 7], dtype=object)
    field = walsh.fields.transform(data, [Fraction(1,2), Fraction(2,3)])
    assert list(field) == [55, 62, 65, 73]
    sizes, _ = bridge.endpoint.corner.verify(point)
    assert set(classes) == set(sizes) and sum(sizes.values()) == 256
    here = Path(__file__).resolve().parent
    helpers = [Path(module.__file__).resolve() for name, module in sys.modules.items()
               if name.startswith(("STAGE", "verify_", "threshold_corner_"))
               and getattr(module, "__file__", None)]
    assert all(here in path.parents for path in helpers)
    print("PASS packaged upper dependencies, physical counts, parent multipliers and field transform")

if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("keep Python assertions enabled")
    main()
