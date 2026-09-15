use nalgebra::{SMatrix, Vector3};

/// One retained cache row: the raw FRD magnetometer sample stored as a row
/// of the sample matrix, paired with the optional normalized, co-timestamped
/// FRD gravity direction carried by that row. The two columns always move
/// together through append, replacement, and expiry compaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct MagSampleRow {
    /// Raw FRD magnetometer sample of this row.
    pub(super) sample: Vector3<f32>,
    /// Optional normalized co-timestamped body-frame FRD gravity direction.
    pub(super) gravity: Option<Vector3<f32>>,
}

/// Retained magnetometer sample cache of [`super::mag_model::MagModel`]: the
/// `N x 3` matrix of raw FRD samples and the per-row optional gravity
/// directions. Only rows `0..sample_row_count` (tracked by the owning model)
/// are live; the remaining rows hold stale data that is never read.
pub(super) struct MagSamples<const N: usize> {
    sample_matrix: SMatrix<f32, N, 3>,
    gravity_directions: [Option<Vector3<f32>>; N],
}

impl<const N: usize> Default for MagSamples<N> {
    fn default() -> Self {
        Self {
            sample_matrix: SMatrix::zeros(),
            gravity_directions: std::array::from_fn(|_| None),
        }
    }
}

impl<const N: usize> MagSamples<N> {
    /// Returns the magnetometer sample stored at `index`.
    pub(super) fn sample(&self, index: usize) -> Vector3<f32> {
        self.sample_matrix.row(index).transpose().into_owned()
    }

    /// Slices row `index` into its structured observation.
    pub(super) fn row(&self, index: usize) -> MagSampleRow {
        MagSampleRow {
            sample: self.sample(index),
            gravity: self.gravity_directions[index],
        }
    }

    /// Writes `row` into the sample matrix and gravity array at `index`.
    pub(super) fn set_row(&mut self, index: usize, row: MagSampleRow) {
        self.sample_matrix.set_row(index, &row.sample.transpose());
        self.gravity_directions[index] = row.gravity;
    }
}
