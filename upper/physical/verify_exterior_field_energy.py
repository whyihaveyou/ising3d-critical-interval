"""Exact exterior-field geometry and transform used by the ordered certificate."""
from fractions import Fraction
from itertools import permutations, product
import numpy as np
import verify_conditional_energy_star as model


def orbit_owners(words):
    count = 1 << len(model.BOUNDARY)
    owners = np.full(count, -1, dtype=np.int32)
    for i, word in enumerate(words):
        images = {m >> len(model.INTERIOR) for m in model.orbit(word << len(model.INTERIOR))}
        images |= {count - 1 - m for m in images}
        columns = list(images)
        assert np.all(owners[columns] == -1)
        owners[columns] = i
    assert np.all(owners >= 0) and len(words) == 3236
    return owners


def neighbours(site):
    return {model.add(site, delta) for delta in model.DIRECTIONS}


def transform(values, ratios):
    """Unnormalized tensor of [[den, num], [num, den]], indexed by little-endian bits."""
    output = np.array(values, dtype=object, copy=True)
    assert output.ndim == 1 and len(output) == 1 << len(ratios)
    for j, ratio in enumerate(ratios):
        width = 1 << j
        pairs = output.reshape(-1, 2 * width)
        left = pairs[:, :width].copy()
        right = pairs[:, width:]
        a, b = ratio.denominator, ratio.numerator
        pairs[:, :width] = a * left + b * right
        pairs[:, width:] = b * left + a * right
    return output


def upper_over_fields(values, z, words, owners, normalization, ratios):
    numerator = transform((z * values)[owners], ratios)
    denominator = transform(z[owners], ratios)
    assert np.all(numerator == numerator[words][owners])
    assert np.all(denominator == denominator[words][owners])
    assert np.all(denominator[words] > 0)
    ratios_by_word = [Fraction(numerator[w], denominator[w] * normalization) for w in words]
    i = max(range(len(words)), key=ratios_by_word.__getitem__)
    return ratios_by_word[i], words[i]


def corner_classes(corners):
    """Corner source configurations modulo cubic symmetry and global spin reversal."""
    group = []
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            group.append([corners.index(tuple(s*v[j] for s, j in zip(signs, axes))) for v in corners])
    seen = set()
    classes = []
    for word in range(1 << len(corners)):
        if word in seen:
            continue
        classes.append(word)
        for action in group:
            image = sum(((word >> i) & 1) << target for i, target in enumerate(action))
            seen.update((image, 255 ^ image))
    assert len(seen) == 256
    return classes
