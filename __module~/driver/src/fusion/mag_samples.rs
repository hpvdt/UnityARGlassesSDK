use nalgebra::{SMatrix, Vector3};

/// Read access to the magnetometer sample and the optional gravity direction
/// of one retained cache row, shared by the owned [`ConcreteRow`] and the
/// borrowed [`Slice`].
pub(super) trait Row {
    /// The raw FRD magnetometer sample of the row.
    fn sample(&self) -> Vector3<f32>;
    /// The optional normalized, co-timestamped body-frame FRD gravity
    /// direction carried by the row.
    fn gravity(&self) -> Option<Vector3<f32>>;
}

/// One retained cache row: the raw FRD magnetometer sample stored as a row
/// of the sample matrix, paired with the optional normalized, co-timestamped
/// FRD gravity direction carried by that row. The two columns always move
/// together through append, replacement, and expiry compaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ConcreteRow {
    sample: Vector3<f32>,
    gravity: Option<Vector3<f32>>,
}

impl ConcreteRow {
    /// Bundles one raw FRD magnetometer sample with its optional normalized,
    /// co-timestamped FRD gravity direction.
    pub(super) fn new(sample: Vector3<f32>, gravity: Option<Vector3<f32>>) -> Self {
        Self { sample, gravity }
    }
}

impl Row for ConcreteRow {
    fn sample(&self) -> Vector3<f32> {
        self.sample
    }

    fn gravity(&self) -> Option<Vector3<f32>> {
        self.gravity
    }
}

/// Borrowed, zero-copy view of one retained row of a [`MagSamples`] cache:
/// each column is materialized from the backing arrays only when its
/// [`Row`] accessor runs, so reading just the sample or just the
/// gravity of a row never touches the other column.
pub(super) struct Slice<'a, const N: usize> {
    samples: &'a MagSamples<N>,
    index: usize,
}

impl<const N: usize> Row for Slice<'_, N> {
    fn sample(&self) -> Vector3<f32> {
        self.samples
            .sample_matrix
            .row(self.index)
            .transpose()
            .into_owned()
    }

    fn gravity(&self) -> Option<Vector3<f32>> {
        self.samples.gravity_directions[self.index]
    }
}

impl<const N: usize> Slice<'_, N> {
    /// Copies the viewed row into an owned [`ConcreteRow`], releasing the
    /// borrow on the cache so the row can be written back to another index.
    pub(super) fn copied(&self) -> ConcreteRow {
        ConcreteRow::new(self.sample(), self.gravity())
    }
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
    /// Borrows row `index` as a zero-copy [`Slice`].
    pub(super) fn view(&self, index: usize) -> Slice<'_, N> {
        Slice {
            samples: self,
            index,
        }
    }

    /// Writes `row` into the sample matrix and gravity array at `index`.
    pub(super) fn set_row(&mut self, index: usize, row: ConcreteRow) {
        self.sample_matrix.set_row(index, &row.sample.transpose());
        self.gravity_directions[index] = row.gravity;
    }
}
