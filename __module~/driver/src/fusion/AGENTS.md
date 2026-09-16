# AGENTS.md - Fusion Module Guide

## Conventions

### Reference Frames


- **RUB (Right-Up-Back):** Android sensor coordinates used by raw device events.
- **FRD (Forward-Right-Down):** Aerospace coordinates used by fusion state and outputs.
- **Custom frames:** Configurable AHRS output frames.

Treat shared sensor-event documentation and the fusion module as the source of truth for frames and units. Keep frame
transformations explicit and centralized, document device-specific deviations, and use the existing linear-algebra
types.

## Magnetometer calibration

`MagCalibrator<N>` retains finite, nonzero FRD magnetometer samples. Each row may also carry an optional normalized,
co-timestamped body-frame FRD gravity direction, and every row stores a device timestamp. Invalid gravity is ignored
without rejecting the magnetometer sample.

Old samples expire according to `max_sample_lifespan_us` before the incoming magnetometer is validated. An invalid
magnetometer can therefore change retained support, normalization, and live quality through expiry, but it is not
retained and does not run an optimizer update. The first `N` valid samples fill the cache unconditionally. Once the
cache is full, a $k$-nearest-neighbor diversity heuristic decides whether a new sample replaces a retained row. Every
valid current sample still participates in one online optimizer update even when diversity rejects it. While no
calibration has been published yet, each valid sample also triggers a ramped number of cache-only replay updates that
accelerate cold-start convergence.

The calibrator maintains the raw first moment and second outer-product moment when rows are appended, replaced, or
expired; cache normalization is derived from these fixed-size statistics without a row scan. Directional coverage and
both live fitness statistics are recomputed from the current cache on each quality update: coverage is the smallest
eigenvalue of the $9 \times 9$ Gram matrix of the mean-centered unit directions. Before nine retained samples,
calibration is explicitly pending with confidence zero; beyond that model minimum, publication follows the
sustained-quality rule under "Candidate conversion and live quality" below.

### Production cache size

The cache must outlast one motion pattern, not merely contain enough rows for nine coefficients. With `N = 255`, the
SimMotion integration test spans only about one motion segment: the retained readings cover a near-planar circle, the
ellipsoid fit can drift along the unobserved axis, and the worst-case heading error exceeds 20 degrees. `FusionState`
therefore uses `N = 1023`, spanning several motion segments.

### Physical model

Let $b$ be the hard-iron offset, $D$ the symmetric positive-definite soft-iron distortion, $A = D^{-1}$ its correction,
and $m_i$ an ideal unit magnetic vector:

$$
x_i = b + D\, m_i, \qquad \|m_i\| = 1, \qquad m_i = A\, (x_i - b).
$$

For the current cache mean $\mu$ and RMS radius $r$, samples are normalized as

$$
u_i = (x_i - \mu) / r,
$$

and the ellipsoid equation is

$$
u_i^T Q\, u_i + q^T u_i = 1.
$$

The nine online coefficients (code field `parameters`) are

$$
\theta = [Q_{00}, Q_{11}, Q_{22}, Q_{01}, Q_{02}, Q_{12}, q_0, q_1, q_2],
$$

and their sample feature vector is

$$
\phi(u) = [u_x^2, u_y^2, u_z^2, 2 u_x u_y, 2 u_x u_z, 2 u_y u_z, u_x, u_y, u_z].
$$

### Convex online objective

The radial algebraic residual and regularizer are

$$
e_{r,i} = \phi_i^T \theta - 1, \qquad
J_r = \frac{1}{2 n} \sum_i e_{r,i}^2 + \frac{\lambda}{2}\, \|Q - c I\|_F^2,
\qquad \lambda = 10^{-3}, \quad c = 2.
$$

The scaled-identity prior counters algebraic ellipsoid inflation under noise and biases unpublished working state away
from indefinite shapes. In coefficient coordinates its weights are

$$
R = \operatorname{diag}(1, 1, 1, 2, 2, 2, 0, 0, 0).
$$

#### Gravity surrogate

The optional gravity term uses the ellipsoid normal

$$
n_i = Q\, u_i + q / 2.
$$

For a normalized gravity direction $g_i$, the projection $s_i = g_i^T n_i$ is linear in $\theta$ through the feature
vector

$$
\begin{aligned}
\psi(u, g) = [\; & g_x u_x,\; g_y u_y,\; g_z u_z,\\
    & g_x u_y + g_y u_x,\; g_x u_z + g_z u_x,\; g_y u_z + g_z u_y,\\
    & g_x / 2,\; g_y / 2,\; g_z / 2\; ].
\end{aligned}
$$

The optimizer learns a scalar projection $\kappa$ and, with the configured weight $w_g$ (`gravity_weight`), minimizes

$$
e_{g,i} = \psi_i^T \theta - \kappa, \qquad
J_g = \frac{w_g}{2 n_g} \sum_i (\psi_i^T \theta - \kappa)^2.
$$

$J_r + J_g$ is convex and quadratic in $(\theta, \kappa)$; matrix square roots occur only during physical candidate
conversion, not in the optimizer. Gravity is optional and disabled by default (weight `0`); `gravity_weight(w)` with a
positive `w` opts back in.

This term is a physical surrogate rather than the exact magnetic dip. The model gives

$$
Q\, (u_i - d) = \gamma\, r\, A\, m_i,
$$

so the surrogate keeps $g_i^T A m_i$ approximately constant instead of the exact $g_i^T m_i$. It is exact for isotropic
correction and biased by anisotropic soft iron. Validation sweeps beyond the fixed simulator distortion (condition
numbers up to 8, rotated eigenvectors, inconsistent acceleration, dip angles from 12 to 83 degrees) found a repeatable
accuracy regression under rotated-eigenvector soft iron at every tested nonzero weight, while the fixed-seed SimMotion
benchmark at weight `0.01` differed from the disabled baseline by under 0.04 degrees; the surrogate therefore ships
disabled. Any change to an opted-in weight must be validated the same way: compare fixed-seed with-gravity integration
results against magnetometer-only results from the SimMotion regression test named under "Calibration validation".

Gravity changes the shared $\theta$. Physical candidate conversion still uses only $\theta$; there is no second
gravity-refined candidate and no relaxed radial-error allowance for gravity-assisted fits.

### Minibatch update

`minibatch_size` defaults to 32 and is clamped to `1..=N.max(1)`. Each update contains:

1. the current valid sample, whether retained or rejected by diversity;
2. random retained rows for the remaining slots, sampled uniformly with replacement.

When the current sample was retained, its row is excluded from random draws so it occurs exactly once. Magnetometer and
gravity data are always sampled together. Sampling uses a private deterministic SplitMix64 generator.

#### Cold-start cache replay

While no calibration has been published yet, the sample-anchored update is followed by up to `replay_updates`
additional updates (default 4) whose minibatches contain `replay_minibatch_size` observations (default 8) drawn
uniformly with replacement from the retained rows only. The arriving sample is never a required replay member; once
retained, it is an ordinary cache row that replay may draw like any other. The replay count ramps with the retained
fraction, `replay_updates * sample_row_count / N`, because repeatedly fitting a small, low-coverage cache overfits it
and can strand the working shape outside the publishable region. Replay steps share the current learning rate but do
not advance the step counter, so annealing stays tied to the rate of arriving data rather than to compute.
`replay_updates(0)` disables replay.

For minibatch $B$ with gravity-carrying subset $G$, the analytic gradients are

$$
\theta_{\mathrm{prior}} = [c, c, c, 0, 0, 0, 0, 0, 0],
$$

$$
\nabla_\theta = \frac{1}{|B|} \sum_{i \in B} e_{r,i}\, \phi_i
    + \frac{w_g}{|G|} \sum_{i \in G} e_{g,i}\, \psi_i
    + \lambda R\, (\theta - \theta_{\mathrm{prior}}),
\qquad
\nabla_\kappa = -\frac{w_g}{|G|} \sum_{i \in G} e_{g,i}.
$$

Omit the gravity terms when $G$ is empty, and add the regularization once per update rather than once per observation.
The diagonal feature-energy scales are

$$
s_{\theta,j} = \frac{1}{|B|} \sum_{i \in B} \phi_{i,j}^2
    + \frac{w_g}{|G|} \sum_{i \in G} \psi_{i,j}^2
    + \lambda R_{jj} + \epsilon,
\qquad
s_\kappa = w_g + \epsilon.
$$

The optimizer divides each gradient component by its scale. Its learning rate decays from a private initial value to a
nonzero floor, and its step norm is bounded. A bounded half-step search accepts only finite updates that lower the same
minibatch objective. Working $Q$ may temporarily be indefinite; publication still requires SPD. Preventing all
intermediate indefinite states can stall descent at the SPD boundary even when the convex optimum is valid.

### Changing normalization

Append, replacement, and expiry change $\mu$ and $r$. The working coefficients are not rebased into the new
normalization and no working state is ever reset: the drift per cache mutation is `O(1 / sample_row_count)`, and the
online optimizer already tracks a moving convex optimum as cache replacements improve coverage, so it absorbs the
normalization drift through its ordinary gradient updates. A single centered sample has zero radius, so the first
informative gradient requires two distinct samples even though state exists immediately.

### Candidate conversion and live quality

For a working candidate,

$$
d = -\tfrac{1}{2}\, Q^{-1} q, \qquad
\gamma = 1 + d^T Q\, d, \qquad
M = Q / \gamma, \qquad
b = \mu + r d, \qquad
A = M^{1/2} / r.
$$

The principal symmetric square root uses a $3 \times 3$ eigendecomposition. A working candidate is valid only when the
coefficients, normalization, offset, and correction are finite, $Q$ is positive-definite, $\gamma$ is positive, and the
correction condition is at most `10`. Invalid candidates have quality zero; raw samples are never substituted for
corrected samples.

Directional coverage is the E-optimality score of the retained mean-centered unit directions $\hat{u}_i$: the smallest
eigenvalue of the mean Gram matrix

$$
H = \frac{1}{n} \sum_i \varphi(\hat{u}_i)\, \varphi(\hat{u}_i)^T,
$$

relative to its uniform-sphere reference $2/15$ and clamped to $[0, 1]$. The coverage-feature vector $\varphi$ holds
the nine ellipsoid-fit features with $\sqrt{2}$ cross-term weights, which makes the induced rotation on feature space
orthogonal, so the score is exactly rotation-invariant. The Gram sum is recomputed from the current cache on every
quality update: directions stored at insertion go stale as the centering mean drifts (the earliest rows of a
still-forming cache keep chord-like directions, which collapses the smallest eigenvalue), and no incremental Gram state
survives that drift. The cache mean is the center, not the fitted hard-iron offset: the offset's component along the
thinnest data direction is itself unconstrained for near-planar support, which destabilizes the score exactly where it
must be decisive. Rank deficiency detects lower-dimensional support by construction: near-planar motion leaves the Gram
matrix rank-deficient and scores near zero, so partial-arc caches cannot inflate coverage. Physical radial fitness is
the mean square of $\|A (x_i - b)\| - 1$ recomputed over every retained row with the current working candidate on
each quality update, sharing the same $O(N)$ cache rescan as coverage; like coverage, the statistic is strictly
bounded by the retained cache and never outlives the rows that produced it. Radial fitness is a linear ramp from `1`
at radial RMS `0` to `0` at radial RMS `0.1`.

Gravity fitness likewise recomputes the mean square of the gravity-projection residual
$\psi(u_i, g_i)^T \theta - \kappa$ over the retained rows that carry a valid gravity direction, and ramps linearly
from `1` at the `0.1` RMS floor to `0` at the `0.35` ceiling (the cache-wide mean square rides slightly above the
trailing-window estimate the original `0.3` ceiling was tuned against). The floor absorbs the surrogate's known
anisotropic soft-iron bias: even a perfect fit keeps an irreducible residual, and it must not drag down a good
calibration. A
missing statistic — gravity disabled (`gravity_weight(0)`), the projection $\kappa$ not yet seeded from a gravity
observation, or carried by no retained row — maps to a neutral `1` rather than `0`: gravity is optional, so an absent
or disabled gravity term never penalizes a magnetometer-only calibration, unlike the mandatory radial statistic whose
absence scores `0`.

Live fitness is the product of the radial and gravity factors, and live confidence is coverage times fitness, clamped
to $[0, 1]$. `MagCalibrationResult` reports every factor: `confidence`, `coverage`, `fitness`, `radial_fitness`, and
`gravity_fitness`.

Working coefficients and published correction parameters are separate. The hard-iron offset and soft-iron correction
change only after 55 valid updates at confidence at least `0.0125`, including while the cache is partial. Confidence in
the hysteresis band $[0.01, 0.0125)$ pauses the streak instead of resetting it; invalid observations, unusable
candidates, and confidence below the `0.01` floor always reset it. Before first publication, `evaluate_correct` returns
`Ok` with `direction` set to `None`. After publication, a candidate without the required streak reports its current
confidence while leaving the last published correction in use, so fusion callers consume only `Some` directions for
attitude updates. Correcting a reading remains one matrix-vector multiplication followed by normalization:

$$
m = A\, (x - b).
$$

### Complexity

For a minibatch of size $|B|$:

- online fitting is $O(10\, |B|)$;
- cold-start replay adds $O(10\, p\, B_r)$ for $p$ ramped replay updates of size $B_r$, only until first publication;
- candidate conversion uses fixed $3 \times 3$ operations;
- normalization uses fixed-size raw moments and is $O(1)$ in $N$;
- coverage and fitness are one $O(N)$ pass over the retained rows (Gram-matrix accumulation plus residual mean
  squares) and one $9 \times 9$ symmetric eigendecomposition per quality update;
- diversity maintenance is expected $O(N)$ for a full cache;
- persistent online-optimizer, moment, and quality state is $O(1)$ in $N$.

The call remains $O(N)$ overall because sample diversity is linear; the coverage and fitness scans share that budget
and store no per-row state of their own.

### Diversity neighbor cache

Each retained row stores its nearest other rows as a sorted trusted prefix with a small overshoot pad. Append and
replacement update prefixes in amortized $O(k)$ per row. Expiry remaps cached indices, and a row is rescanned only when
its trusted prefix falls below $k$. Configurations with $k$ above the fixed cache capacity scan rows directly. Distance
selection operates on squared values and takes square roots only for selected neighbors.

### Known adaptation limitation

Fitness and coverage are recomputed from the retained rows on every quality update, so `max_sample_lifespan_us`
strictly bounds their history. Only the online-optimizer parameters, including the learned gravity projection
$\kappa$, retain historical gradient influence after a row is replaced or expires, diluting through the floored
learning rate; that residual history is non-strict by design. The backlog tracks explicit replay or forgetting work
needed before sample lifespan can be interpreted as a strict optimizer-history bound.

### Calibration validation

When changing the calibrator, preserve deterministic coverage for cache expiry and readiness, neighbor-cache
invariants, invalid magnetometer and gravity inputs, minibatch clamping, repeatability, full-SPD and asymmetric
distortion, degenerate samples, stable repeated correction, and last-known-good fallback. Run the focused checks first
from `__module~/driver`:

```bash
cargo test --package ar-drivers --no-default-features --lib fusion::mag_calibrator::mag_calibrator_test
cargo test --package ar-drivers --no-default-features --lib fusion::naive_cf::naive_cf_test
cargo test --package ar-drivers --no-default-features --test mag_calibrator_sim_motion regression -- --nocapture
```

Then run the applicable broad Rust checks from the parent guide.
