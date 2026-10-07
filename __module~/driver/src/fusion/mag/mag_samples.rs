use nalgebra::{SMatrix, Vector3};

use super::sample_stats::SampleStats;

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
/// FRD gravity direction carried by that row. The two columns and the row's
/// device timestamp always move together through append, replacement, and
/// expiry compaction.
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
/// `N x 3` matrix of raw FRD samples, the per-row optional gravity
/// directions, the per-row device timestamps, and the incrementally
/// maintained statistics of the retained rows ([`SampleStats`]). Only rows
/// `0..stats.sample_row_count` are live; the remaining rows hold stale data
/// that is never read. Every mutation writes the row columns and updates the
/// statistics together, so the two can never desynchronize.
pub(super) struct MagSamples<const N: usize> {
    sample_matrix: SMatrix<f32, N, 3>,
    gravity_directions: [Option<Vector3<f32>>; N],
    /// Device timestamp of each row, in microseconds, moved together with
    /// the sample and gravity columns by append, replacement, and expiry.
    pub(super) timestamps_us: [u64; N],
    /// Incrementally maintained statistics of the retained rows: the row
    /// count and the raw moments backing the sample normalization, updated
    /// together with the row columns by every mutation.
    pub(super) stats: SampleStats,
}

impl<const N: usize> Default for MagSamples<N> {
    fn default() -> Self {
        Self {
            sample_matrix: SMatrix::zeros(),
            gravity_directions: std::array::from_fn(|_| None),
            timestamps_us: [0; N],
            stats: SampleStats::default(),
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

    /// Writes the sample and gravity columns of `row` at `index`, leaving the
    /// timestamp column and the maintained statistics to the mutation method
    /// driving the write: a compaction move, unlike a replacement, must not
    /// change the raw moments.
    fn write_row(&mut self, index: usize, row: ConcreteRow) {
        self.sample_matrix.set_row(index, &row.sample().transpose());
        self.gravity_directions[index] = row.gravity();
    }

    /// Appends `row` with its device `timestamp_us` at the live tail of the
    /// cache: writes the sample, gravity, and timestamp columns together,
    /// adds the row's raw moment, and advances the retained row count.
    /// Call only while the cache has free capacity.
    pub(super) fn append(&mut self, row: ConcreteRow, timestamp_us: u64) {
        let index = self.stats.sample_row_count;
        self.stats.add_raw_moment(row.sample());
        self.write_row(index, row);
        self.timestamps_us[index] = timestamp_us;
        self.stats.sample_row_count += 1;
    }

    /// Replaces the retained row at `index` with `row` and its device
    /// `timestamp_us`: rewrites the sample, gravity, and timestamp columns
    /// together and swaps the row's raw moment, leaving the retained row
    /// count unchanged.
    pub(super) fn replace(&mut self, index: usize, row: ConcreteRow, timestamp_us: u64) {
        let new_sample = row.sample();
        let replaced_sample = self.view(index).sample();
        self.stats.remove_raw_moment(replaced_sample);
        self.write_row(index, row);
        self.timestamps_us[index] = timestamp_us;
        self.stats.add_raw_moment(new_sample);
    }

    /// Expires the retained rows whose timestamp is older than
    /// `max_sample_lifespan_us` relative to `now` and compacts the survivors
    /// down: the sample, gravity, and timestamp columns move together and
    /// each expired row's raw moment is removed, so the statistics track
    /// exactly the retained rows; a cache emptied by expiry resets its
    /// moments to exact zeros. `index_map` must be pre-filled by the caller
    /// with `u32::MAX` and receives each surviving row's new index; every
    /// other slot keeps the `u32::MAX` expiry marker for the caller's
    /// neighbor-cache remap. The map is a caller-owned buffer rather than a
    /// return value so the update path holds one `[u32; N]` instance at a
    /// time, keeping its peak stack flat: the bounded-stack fusion test runs
    /// the whole update path on a 384 KiB stack.
    pub(super) fn expire(
        &mut self,
        now: u64,
        max_sample_lifespan_us: u64,
        index_map: &mut [u32; N],
    ) {
        let mut retained_count = 0;
        for (index, map_slot) in index_map
            .iter_mut()
            .enumerate()
            .take(self.stats.sample_row_count)
        {
            if now.saturating_sub(self.timestamps_us[index]) <= max_sample_lifespan_us {
                *map_slot = retained_count as u32;
                // Rows are only materialized when compaction actually moves
                // them; the common no-expiry scan keeps every row in place.
                if retained_count != index {
                    let row = self.view(index).copied();
                    let timestamp_us = self.timestamps_us[index];
                    self.write_row(retained_count, row);
                    self.timestamps_us[retained_count] = timestamp_us;
                }
                retained_count += 1;
            } else {
                let expired_sample = self.view(index).sample();
                self.stats.remove_raw_moment(expired_sample);
            }
        }
        if retained_count != self.stats.sample_row_count {
            self.stats.sample_row_count = retained_count;
            if retained_count == 0 {
                // Incremental subtraction can leave round-off residue after
                // the last retained row expires. An empty cache has exact
                // zero moments by definition.
                self.stats.clear_raw_moments();
            }
        }
    }

    /// Writes `sample - row_sample` of each of the first `count` retained
    /// rows into `differences`. Rows at and beyond `count` are left
    /// untouched. Used by the diversity heuristic's squared-distance scans,
    /// so per-row differences avoid materializing row vectors one at a time
    /// through the generic row-view machinery.
    ///
    /// Per-row arithmetic is identical to
    /// `sample - view(index).sample()`: each component is a single f32
    /// subtraction evaluated in the same x, y, z order.
    pub(super) fn row_differences(
        &self,
        sample: Vector3<f32>,
        count: usize,
        differences: &mut [Vector3<f32>],
    ) {
        for (index, difference) in differences.iter_mut().enumerate().take(count) {
            *difference = Vector3::new(
                sample.x - self.sample_matrix[(index, 0)],
                sample.y - self.sample_matrix[(index, 1)],
                sample.z - self.sample_matrix[(index, 2)],
            );
        }
    }
}
