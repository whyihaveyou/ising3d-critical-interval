"""Exact recurrence for the three-dimensional even return counts."""
from math import comb


def return_terms(nmax):
    """Yield the 2n-step closed walk count and 36**n for each n."""
    yield 1, 1
    if nmax == 0:
        return
    previous, current = 1, 3
    central_binomial = 2
    denominator = 36
    yield 6, denominator
    for n in range(2, nmax + 1):
        value = (10*n*n - 10*n + 3)*current - 9*(n - 1)**2*previous
        next_count, remainder = divmod(value, n*n)
        assert remainder == 0
        if n <= 16:
            assert next_count == sum(comb(n, j)**2 * comb(2*j, j) for j in range(n + 1))
        central_binomial, remainder = divmod(central_binomial*(4*n - 2), n)
        assert remainder == 0
        denominator *= 36
        yield central_binomial*next_count, denominator
        previous, current = current, next_count
