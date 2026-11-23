# The ordered endpoint \(3775/2386\)

Let \(r=e^{2K}\), \(r_b=3775/2386\), and \(m_+\) be the spontaneous magnetization of the simple-cubic nearest-neighbour zero-field ferromagnet. The two exhaustive integer checks at \(r_b\) give

\[
Q=\frac{8943787728743670589}{8943788333028500000},\qquad
U=-2.15386328295063879046\ldots\,10^{-7},\qquad
m_+^2\ge L=-U/Q>\frac{215386342}{10^{15}} .
\]

The finite-volume zero mode \(q=\lim\mathbb E[M_N^2]\) is taken along a common subsequence of even periodic boxes. The nonzero-mode infrared bound gives \(0\le U+Qq\), and the pressure tilt gives \(q\le m_+^2\). The archive contains \(U,Q,L\) and the local rounding radius as exact fractions. [Endpoint record](STAGE30_UPPER_REVIEW_3775_2386.json)

## Arithmetic and archived evidence

Run these commands from this directory in the unpacked supplement, keeping the surrounding **upper** directory in place. Both commands use only the Python standard library.

~~~sh
python3 -B replay.py arithmetic
python3 -B replay.py receipts
~~~

**Arithmetic** reconstructs the bridge-adjusted class caps and applies the 13 nonnegative class multipliers. It checks the hash of the signed physical rows, the endpoint fractions, the 24 shared bridges, the 40 class odds constraints and the odds on each stated interval. The first interval below uses the common \(18\,000\,000\) log-coupling rate from the parent directory. The last two use the four row families separately and the supplied residuals on 3236 orbits. Each number in the right column is the floor of \(10^{15}\) times a certified lower bound on \(m_+^2\). [Arithmetic replay](replay.py) · [Refined interval arithmetic](margin.py) · [Clean-directory arithmetic receipt](receipts/portable_arithmetic.log)

| Closed interval | Lower-bound floor |
| --- | ---: |
| \(\{r_b\}\) | 215386342 |
| \([r_b-3/(2\cdot10^{14}),r_b]\) | 44732026 |
| \([r_b-407/10^{13},r_b]\) | 348897 |
| \([r_b-4075/10^{14},r_b]\) | 84723 |

In particular, positive magnetization at the left endpoint of the last row gives

\[
e^{2K_c}\le r_b-\frac{4075}{10^{14}}
<r_b-\frac{407}{10^{13}}
=\frac{18874999999514449}{11930000000000000}<r_b .
\]

**Receipts** authenticates the archived directed completion and re-reads every class log from the arbitrary-precision run. It checks the four output prefixes, the cap and complement counts, the sign of each class minimum, and the file digests. It consumes previously computed field results. [Directed completion](receipts/directed.log) · [Raw coverage](receipts/raw_coverage.json) · [Raw run](receipts/run.log) · [Clean-directory receipt check](receipts/portable_receipts.log)

## Full recomputation from the frozen physical inputs

The directed command below needs Python, NumPy and SciPy. On macOS or Linux, the raw command also needs a C++17 compiler and GMP with its C++ headers and library. On Homebrew it can find GMP through **brew**. On a system with a separate installation, set **GMP_PREFIX** to the installation prefix containing the **include** and **lib** directories. **CXX** can specify the compiler executable.

~~~sh
env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B -u prepare.py --directed
python3 -B -u replay.py raw
~~~

**Directed** rebuilds the local Ising moments, signed rows, shell weights, field ratios, radius and \(Q\) from the frozen parent sources. It then computes all field inequalities with directed integer rounding. **Raw** repeats the physical rebuild in a new temporary directory and checks the two regenerated data files against their frozen hashes. It compiles the supplied GMP source, checks small tensors against explicit signed sums and the original Python transform, then checks all 14 classes with exact integers. It compares the new prefix counts, minima, exception counts and sampled integer values against the archived raw output. Each mode works through all \(14\cdot2^{18}=3,670,016\) exterior fields. [Physical rebuild](prepare.py) · [Blocked integer kernel](blocked_raw.cpp) · [Input digests](inputs.json)

The two commands were run in a clean copy of the portable upper sources. The distributed data tables contain integer coefficients and a little-endian orbit-index array. The executable is compiled in the temporary run directory. The arithmetic and receipt commands also passed after a ZIP compression and extraction round trip. [Clean-directory directed run](receipts/portable_directed.log) · [Clean-directory raw run](receipts/portable_raw.log) · [ZIP round-trip receipt](receipts/zip_roundtrip.log)

For an upper cap \(C_c\), the raw run checks all \(2^{18}\) fields in class \(c\). It checks the lower cap \(D_c\) throughout the complement of the recorded high set \(H_c\). Across all classes, this gives 3,670,016 full-cap comparisons and 3,661,604 complement comparisons. Every required numerator is strictly positive. The other 8,412 lower-cap numerators are strictly negative and have precisely the recorded high-field indices. The original full raw replay took 360.6 seconds with a largest single-class peak resident set of 203.70 MiB. Its four output blocks each held 65,536 arbitrary-precision integers. Timings depend on the host.

The two field kernels share the 104 translation rows, five infrared rows, four reflection rows, seven deleted-spin rows, five rational Green upper bounds, orbit assignment, local rounding radius, physical field ratios and the archived choices of bridges and class multipliers. Their field summations and sign checks use different arithmetic. The separate Python sample check recomputes two actual fields per class, one in \(H_c\) and one outside it. It evaluates both cap numerators using the original corner trace and direct Hamming field kernel, for a total of 56 exact comparisons. [Cross-check source](crosscheck.py) · [Small-tensor checks](receipts/small.log) · [Direct physical-field checks](receipts/samples.log)

~~~sh
python3 -B -u replay.py samples
~~~

## Bound on the explicit left interval

Use \(y=\log r\). The fixed infrared rows use the same rational \(J^+\) values throughout. The individual log-coupling Lipschitz constants for translation, infrared, reflection and deleted-spin rows are 144, 228, 57792 and 14208. Summing against their absolute physical multipliers gives an orbitwise residual rate below 8326. At the center, every true residual is within the archived radius of its integer residual divided by its scale. For \(\Delta=4075/10^{14}\), the largest absolute center residual plus this radius and \(8326\Delta/(r_b-\Delta)\) is below \(71/1000\). The conditional weights admit a normalization with \(\lvert\partial_y\log W\rvert\le468\). Cauchy–Schwarz on the covariance in the derivative of the conditional expectation gives the field rate \(8326+468(71/1000)=2089807/250\). [Physical row and weight bounds](../verify_stage33_upper_margin.py) · [Orbitwise calculation](margin.py)

On 16 closed subintervals of \(1/r\), outward-rounded rational arithmetic bounds all six local odds strictly above their frozen values. Adding the same field variation to the full and complement caps adds that variation to the bridge-averaged cap. Hence the uniform lower bound for \(0<d\le\Delta\) is

\[
 -\bigl[U+(2089807/250)d/(r_b-d)\bigr]/Q .
\]

The finite-volume and pressure limits apply at each fixed \(r\) in the interval. [Refined interval receipt](receipts/margin.log) · [Original \(3/(2\cdot10^{14})\) interval receipt](receipts/interval.log)
