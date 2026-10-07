use nalgebra::{Matrix3, Vector3};

/// Incrementally maintained sample statistics of the retained magnetometer
/// cache rows, grouped behind `MagSamples::stats` for accelerated,
/// continuous update: the raw first and second moments are maintained on
/// append, replacement, and expiry, and the sample normalization
/// $(\mu, r)$ is derived from them in $O(1)$ at point of use, without a row
/// scan and without any refresh step that could go stale.
pub(super) struct SampleStats {
    /// Number of retained cache rows `0..sample_row_count`: the zeroth raw
    /// moment the first and second moments below are averaged over,
    /// maintained on append, replacement, and expiry alongside them.
    pub(super) sample_row_count: usize,
    /// Raw first moment of the retained magnetometer samples, maintained
    /// incrementally on append, replacement, and expiry. Backs
    /// `raw_mean_and_covariance` and thus `normalization`.
    pub(super) raw_sample_sum: Vector3<f64>,
    /// Raw second outer-product moment of the retained magnetometer
    /// samples; see `raw_sample_sum`.
    pub(super) raw_outer_product_sum: Matrix3<f64>,
}

impl Default for SampleStats {
    fn default() -> Self {
        Self {
            sample_row_count: Default::default(),
            raw_sample_sum: Vector3::zeros(),
            raw_outer_product_sum: Matrix3::zeros(),
        }
    }
}

impl SampleStats {
    pub(super) fn normalized_sample(&self, sample: Vector3<f32>) -> Vector3<f32> {
        let (mean, rms_radius) = self.normalization();
        (sample - mean) / rms_radius
    }

    /// Whether the sample normalization $(\mu, r)$ of the retained
    /// magnetometer samples is usable: both finite and the radius above
    /// `f32::EPSILON`, which requires two distinct samples. Derived from the
    /// raw moments at point of use, so usability can never disagree with the
    /// state it describes.
    pub(super) fn sample_normalization_usable(&self) -> bool {
        let (mean, rms_radius) = self.normalization();
        mean.iter().all(|value| value.is_finite())
            && rms_radius.is_finite()
            && rms_radius > f32::EPSILON
    }

    /// Current sample normalization $(\mu, r)$ derived from the raw moments:
    /// zeros on an empty or non-finite cache, otherwise the sample mean
    /// $\mu$ and RMS radius $r$ of $u_i = (x_i - \mu) / r$. Every append,
    /// replacement, and expiry drifts the mean and radius; the working
    /// coefficients keep their meaning in the new normalization directly,
    /// because the drift per cache mutation is `O(1 / sample_row_count)` and
    /// the online optimizer is already designed to track the moving convex
    /// optimum as cache replacements improve coverage. Working state is
    /// therefore never rebased or reset: only a zero-radius (empty or
    /// single-point) cache makes the normalization unusable, which keeps the
    /// optimizer idle until two distinct samples exist and reports quality
    /// zero through the usual unusable-candidate path.
    pub(super) fn normalization(&self) -> (Vector3<f32>, f32) {
        let Some((mean, covariance)) = self.raw_mean_and_covariance() else {
            return (Vector3::zeros(), 0.0);
        };
        (mean, covariance.trace().sqrt())
    }

    pub(super) fn add_raw_moment(&mut self, sample: Vector3<f32>) {
        let sample = sample.cast::<f64>();
        self.raw_sample_sum += sample;
        self.raw_outer_product_sum += sample * sample.transpose();
    }

    pub(super) fn remove_raw_moment(&mut self, sample: Vector3<f32>) {
        let sample = sample.cast::<f64>();
        self.raw_sample_sum -= sample;
        self.raw_outer_product_sum -= sample * sample.transpose();
    }

    pub(super) fn clear_raw_moments(&mut self) {
        self.raw_sample_sum = Vector3::zeros();
        self.raw_outer_product_sum = Matrix3::zeros();
    }

    pub(super) fn raw_mean_and_covariance(&self) -> Option<(Vector3<f32>, Matrix3<f32>)> {
        if self.sample_row_count == 0 {
            return None;
        }
        let count = self.sample_row_count as f64;
        let mean = self.raw_sample_sum / count;
        let covariance = self.raw_outer_product_sum / count - mean * mean.transpose();
        let covariance = 0.5 * (covariance + covariance.transpose());
        let mean = mean.cast::<f32>();
        let covariance = covariance.cast::<f32>();
        if mean.iter().all(|value| value.is_finite())
            && covariance.iter().all(|value| value.is_finite())
        {
            Some((mean, covariance))
        } else {
            None
        }
    }
}
