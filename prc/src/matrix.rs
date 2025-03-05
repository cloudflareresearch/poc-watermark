// Copyright (c) 2025 Cloudflare, Inc.

//! A sparse matrix implementation over GF(2). Field elements are represented by `bool`.

use std::{
    cmp::Ordering,
    io::{Cursor, Read},
};

const INIT_ROW_CAPACITY: usize = 100;
const INIT_COL_CAPACITY: usize = 100;

/// An entry of a sparse matrix.
#[derive(Copy, Clone)]
struct Entry {
    i: usize,
    j: usize,

    // Position of `i, j` in the sparse representation.
    row: usize,
    col: usize,

    // The value of the entry.
    val: bool,
}

/// A row of a matrix.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Row {
    pub(crate) i: usize,
    pub(crate) js: Vec<usize>,
}

impl Row {
    pub(crate) fn new(i: usize) -> Self {
        Self {
            i,
            js: Vec::with_capacity(INIT_COL_CAPACITY),
        }
    }

    /// The order of the row, defined to be the distance of the leading coefficient from the start
    /// of the row. If the row is empty, then its order is `usize::MAX`. This is used for
    /// converting a matrix to row echelon form.
    fn ord(&self) -> usize {
        self.js.first().cloned().unwrap_or(usize::MAX)
    }

    /// Find the sparse entry corresponding to row `j`.
    pub(crate) fn col_entry(&self, j: usize) -> usize {
        binsearch(&self.js, j, |col| *col)
    }

    fn is_empty(&self) -> bool {
        self.js.is_empty()
    }
}

/// A matrix over GF(2).
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub(crate) num_rows: usize,
    pub(crate) num_cols: usize,
    pub(crate) rows: Vec<Row>,
}

type N = u32;

impl Matrix {
    /// Construct a matrix `M` with the give dimensions. All entries are initialized to `false`.
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            rows: Vec::with_capacity(INIT_ROW_CAPACITY),
        }
    }

    pub(crate) fn encode_dense(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Encode the number of rows.
        let rows_len_bytes = u32::try_from(self.rows.len()).unwrap().to_be_bytes();
        bytes.extend_from_slice(&rows_len_bytes);

        // Encode each row.
        let mut js_bytes = vec![0; (self.num_cols + 7) / 8];
        for Row { i, js } in self.rows.iter() {
            // Encode `i`.
            let i_bytes = N::try_from(*i).unwrap().to_be_bytes();
            bytes.extend_from_slice(&i_bytes);

            // Encode `js`.
            for j in js.iter() {
                let byte_index = j / 8;
                let bit_index = j % 8;
                js_bytes[byte_index] |= 1 << bit_index;
            }
            bytes.extend_from_slice(&js_bytes);

            // Clear the buffer.
            for byte in js_bytes.iter_mut() {
                *byte = 0;
            }
        }
        bytes
    }

    pub(crate) fn decode_dense(
        bytes: &mut Cursor<&[u8]>,
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Self, String> {
        let mut m = Self::new(num_rows, num_cols);

        // Decode the number of rows.
        let mut rows_len_bytes = [0; { size_of::<N>() }];
        bytes
            .read_exact(&mut rows_len_bytes)
            .map_err(|e| format!("num rows: {e}"))?;
        let rows_len = usize::try_from(N::from_be_bytes(rows_len_bytes)).unwrap();

        // Decode the rows.
        let mut i_bytes = [0; { size_of::<N>() }];
        let mut js_bytes = vec![0; (m.num_cols + 7) / 8];
        for row_entry in 0..rows_len {
            // Decode `i`.
            bytes
                .read_exact(&mut i_bytes)
                .map_err(|e| format!("row {row_entry}: i: {e}"))?;
            let i = usize::try_from(N::from_be_bytes(i_bytes)).unwrap();

            // Decode `js`.
            bytes
                .read_exact(&mut js_bytes)
                .map_err(|e| format!("row {row_entry}: js: {e}"))?;
            let mut row = Row::new(i);
            for j in 0..m.num_cols {
                let byte_index = j / 8;
                let bit_index = j % 8;
                if js_bytes[byte_index] & (1 << bit_index) > 0 {
                    row.js.push(j);
                }
            }

            m.rows.push(row);
        }
        Ok(m)
    }

    pub(crate) fn encode_sparse(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Encode the number of rows.
        let rows_len_bytes = u32::try_from(self.rows.len()).unwrap().to_be_bytes();
        bytes.extend_from_slice(&rows_len_bytes);

        // Encode the rows.
        for Row { i, js } in self.rows.iter() {
            // Encode `i`.
            let i_bytes = N::try_from(*i).unwrap().to_be_bytes();
            bytes.extend_from_slice(&i_bytes);

            // Encode `js.len()`.
            let js_len_bytes = N::try_from(js.len()).unwrap().to_be_bytes();
            bytes.extend_from_slice(&js_len_bytes);

            // Encode `js`.
            for j in js.iter() {
                let j_bytes = N::try_from(*j).unwrap().to_be_bytes();
                bytes.extend_from_slice(&j_bytes);
            }
        }
        bytes
    }

    pub(crate) fn decode_sparse(
        bytes: &mut Cursor<&[u8]>,
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Self, String> {
        let mut m = Matrix::new(num_rows, num_cols);

        // Decode the number of rows.
        let mut rows_len_bytes = [0; { size_of::<N>() }];
        bytes
            .read_exact(&mut rows_len_bytes)
            .map_err(|e| format!("num rows: {e}"))?;
        let rows_len = usize::try_from(N::from_be_bytes(rows_len_bytes)).unwrap();

        // Decode the rows.
        let mut i_bytes = [0; { size_of::<N>() }];
        let mut j_bytes = [0; { size_of::<N>() }];
        let mut js_len_bytes = [0; { size_of::<N>() }];
        for row_entry in 0..rows_len {
            // Decode `i`.
            bytes
                .read_exact(&mut i_bytes)
                .map_err(|e| format!("parity check {row_entry}: i: {e}"))?;
            let i = usize::try_from(N::from_be_bytes(i_bytes)).unwrap();

            // Decode `js.len()`.
            bytes
                .read_exact(&mut js_len_bytes)
                .map_err(|e| format!("parity check {row_entry}: js.len(): {e}"))?;
            let js_len = usize::try_from(N::from_be_bytes(js_len_bytes)).unwrap();

            // Decode `js`.
            let mut row = Row::new(i);
            for col_entry in 0..js_len {
                bytes
                    .read_exact(&mut j_bytes)
                    .map_err(|e| format!("parity check {row_entry}: js {col_entry}: {e}"))?;
                let j = usize::try_from(N::from_be_bytes(j_bytes)).unwrap();
                row.js.push(j);
            }

            m.rows.push(row);
        }
        Ok(m)
    }

    /// Get the value of `M[i, ]`.
    pub fn get(&self, i: usize, j: usize) -> bool {
        self.entry(i, j).val
    }

    /// Set `M[i, j]` to `val`.
    pub fn set(&mut self, i: usize, j: usize, val: bool) {
        self.set_entry(self.entry(i, j), val);
    }

    /// Convert `M` to row-echelon form.
    pub fn row_echelon(mut self) -> Self {
        // Convert non-empty rows.
        for q in 0..self.rows.len() {
            let p = self.next_pivot(q);
            self.swap_rows(p, q);
            for r in q + 1..self.rows.len() {
                if self.rows[r].ord() == self.rows[q].ord() {
                    self.add_row(r, q);
                }
            }
        }

        // Delete empty rows.
        for q in (0..self.rows.len()).rev() {
            if self.rows[q].is_empty() {
                self.rows.remove(q);
            }
        }

        // Shift empty rows to the end of the matrix.
        for q in 0..self.rows.len() {
            self.rows[q].i = q;
        }

        self
    }

    /// Return `M^T`.
    pub fn transpose(self) -> Self {
        let mut m = Matrix::new(self.num_cols, self.num_rows);
        for Row { i, js } in self.rows.into_iter() {
            for j in js.into_iter() {
                m.set(j, i, true);
            }
        }
        m
    }

    /// Return `M*x`.
    pub fn mul_by_vec(&self, x: &[bool]) -> Vec<bool> {
        debug_assert_eq!(x.len(), self.num_cols);
        let mut y = vec![false; self.num_rows];
        for Row { i, js } in self.rows.iter() {
            for j in js {
                y[*i] ^= x[*j];
            }
        }
        y
    }

    /// Multiply `M` by a matrix.
    //
    // TODO(cjpatton) Optimize for sparse representation.
    #[cfg(test)]
    pub(crate) fn mul(&self, other: &Matrix) -> Matrix {
        debug_assert_eq!(
            self.num_cols, other.num_rows,
            "Matrices cannot be multiplied"
        );

        let mut result = Matrix::new(self.num_rows, other.num_cols);

        for i in 0..self.num_rows {
            for j in 0..other.num_cols {
                for k in 0..self.num_cols {
                    result.set(i, j, result.get(i, j) ^ self.get(i, k) & other.get(k, j));
                }
            }
        }

        result
    }

    /// Indicate whether the `M[i,j]==false` for all `i,j`.
    pub(crate) fn is_zero(&self) -> bool {
        !self.rows.iter().any(|row| !row.js.is_empty())
    }

    #[cfg(test)]
    pub(crate) fn pretty_print(&self) {
        let mut pretty = String::new();
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                pretty += if self.get(i, j) { "1" } else { "0" };
            }
            pretty += "\n";
        }
        println!("{}\n", pretty);
    }

    /// Set `M[i,j] = M[i,j] + val`.
    pub(crate) fn add_to(&mut self, i: usize, j: usize, val: bool) {
        let entry = self.entry(i, j);
        self.set_entry(entry, val ^ entry.val);
    }

    /// Return the sparse entry corresponding to row `i`.
    pub(crate) fn row_entry(&self, i: usize) -> usize {
        binsearch(&self.rows, i, |row| row.i)
    }

    /// Return the sparse entry corresponding to row `i` and column `j`.
    fn entry(&self, i: usize, j: usize) -> Entry {
        let row = self.row_entry(i);

        let Some(r) = self.rows.get(row) else {
            return Entry {
                i,
                j,
                row,
                col: 0, // This is the first col of the new row
                val: false,
            };
        };

        let col = r.col_entry(j);
        let val = r.js.get(col).cloned().unwrap_or(0) == j;
        Entry {
            i,
            j,
            row,
            col,
            val,
        }
    }

    fn set_entry(&mut self, entry: Entry, new_val: bool) {
        if !new_val && entry.val {
            // Unset the entry. If the row is now empty, then delete the row.
            self.rows[entry.row].js.remove(entry.col);
            if self.rows[entry.row].js.is_empty() {
                self.rows.remove(entry.row);
            }
        } else if new_val && !entry.val {
            // Insert a new row if needed. This resets the column entry to 0.
            let col = if !self
                .rows
                .get(entry.row)
                .map(|r| r.i == entry.i)
                .unwrap_or(false)
            {
                self.rows.insert(entry.row, Row::new(entry.i));
                0
            } else {
                entry.col
            };

            let r = self.rows.get_mut(entry.row).unwrap();
            r.js.insert(col, entry.j);
        }
    }

    fn add_row(&mut self, dst_entry: usize, src_entry: usize) {
        let (dst_row, src_row) = match dst_entry.cmp(&src_entry) {
            Ordering::Less => {
                let split = self.rows.split_at_mut(src_entry);
                (&mut split.0[dst_entry].js, &mut split.1[0].js)
            }
            Ordering::Greater => {
                let split = self.rows.split_at_mut(dst_entry);
                (&mut split.1[0].js, &mut split.0[src_entry].js)
            }
            Ordering::Equal => {
                // We're adding a row into itself, so just clear the row.
                self.rows[src_entry].js.clear();
                return;
            }
        };

        // Merge the source row into the destination row.
        //
        // TODO(cjpatton) Figure out if we can optimize this further.
        dst_row.extend(src_row.iter());
        dst_row.sort();
        if dst_row.len() > 1 {
            let mut i = dst_row.len() - 1;
            while i > 0 {
                if dst_row[i] == dst_row[i - 1] {
                    // If both rows have the same column set, then the column should be unset after
                    // merge since 1^1 == 0.
                    dst_row.remove(i);
                    dst_row.remove(i - 1);
                    i = i.saturating_sub(1);
                }
                i = i.saturating_sub(1);
            }
        }
    }

    fn swap_rows(&mut self, entry1: usize, entry2: usize) {
        if entry1 == entry2 {
            return;
        }
        self.rows[entry1].i ^= self.rows[entry2].i;
        self.rows[entry2].i ^= self.rows[entry1].i;
        self.rows[entry1].i ^= self.rows[entry2].i;
        self.rows.swap(entry1, entry2);
    }

    fn next_pivot(&self, start_entry: usize) -> usize {
        self.rows[start_entry..]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.ord().cmp(&b.ord()))
            .map(|(i, _)| i + start_entry)
            .unwrap()
    }
}

fn binsearch<T, F: Fn(&T) -> usize>(v: &[T], idx: usize, f: F) -> usize {
    let mut l = 0;
    let mut r = v.len();
    let mut m = (l + r) / 2;
    while l < r {
        match f(&v[m]).cmp(&idx) {
            Ordering::Less => {
                l = m + 1;
            }
            Ordering::Greater => {
                r = m;
            }
            Ordering::Equal => {
                return m;
            }
        };
        m = (l + r) / 2;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::prelude::*;

    #[test]
    fn basic_operations() {
        let mut m = Matrix::new(16, 10);

        m.set(0, 7, true);
        assert!(m.get(0, 7));

        m.set(0, 0, true);
        assert!(m.get(0, 0));
        assert!(m.get(0, 7));

        m.set(1, 4, true);
        assert!(m.get(0, 0));
        assert!(m.get(0, 7));
        assert!(m.get(1, 4));

        m.set(1, 3, true);
        assert!(m.get(0, 0));
        assert!(m.get(0, 7));
        assert!(m.get(1, 4));
        assert!(m.get(1, 3));

        m.set(1, 3, false);
        assert!(m.get(0, 0));
        assert!(m.get(0, 7));
        assert!(m.get(1, 4));
        assert!(!m.get(1, 3));

        m.set(10, 3, false);
        assert!(m.get(0, 0));
        assert!(m.get(0, 7));
        assert!(m.get(1, 4));
        assert!(!m.get(1, 3));
        assert!(!m.get(10, 3));

        m.set(10, 3, true);
        assert!(m.get(10, 3));

        m.set(1, 4, false);
        assert!(!m.get(1, 4));
        assert!(m.get(0, 7));
        assert!(m.get(10, 3));
        assert!(!m.get(10, 9));
        assert!(!m.get(1, 4));
        assert!(!m.get(13, 13));

        m.add_to(0, 7, false);
        assert!(m.get(0, 7));
        m.add_to(0, 7, true);
        assert!(!m.get(0, 7));
        m.add_to(0, 7, true);
        assert!(m.get(0, 7));
    }

    #[test]
    fn add_row() {
        // src < dst
        let mut m = Matrix::new(10, 10);
        m.set(1, 0, true);
        m.set(1, 3, true);
        m.set(2, 0, true);
        m.set(2, 1, true);
        m.set(2, 9, true);
        m.add_row(m.row_entry(1), m.row_entry(2));
        assert!(!m.get(1, 0));
        assert!(m.get(1, 1));
        assert!(m.get(1, 3));
        assert!(m.get(1, 9));

        // dst > src
        let mut m = Matrix::new(10, 10);
        m.set(1, 0, true);
        m.set(1, 3, true);
        m.set(2, 0, true);
        m.set(2, 1, true);
        m.set(2, 9, true);
        m.add_row(m.row_entry(2), m.row_entry(1));
        assert!(!m.get(2, 0));
        assert!(m.get(2, 3));
        assert!(m.get(2, 1));
        assert!(m.get(2, 9));

        // dst == src
        let mut m = Matrix::new(10, 10);
        m.set(1, 0, true);
        m.set(1, 3, true);
        m.add_row(m.row_entry(1), m.row_entry(1));
        assert!(m.rows[0].js.is_empty());
    }

    #[test]
    fn swap_rows() {
        let mut m = Matrix::new(10, 10);
        m.set(1, 0, true);
        m.set(1, 3, true);
        m.set(2, 0, true);
        m.set(2, 1, true);
        m.set(2, 9, true);
        m.swap_rows(m.row_entry(1), m.row_entry(2));
        assert!(m.get(2, 0));
        assert!(m.get(2, 3));
        assert!(m.get(1, 0));
        assert!(m.get(1, 1));
        assert!(m.get(1, 9));
    }

    #[test]
    fn mul_by_vec() {
        let mut rng = thread_rng();

        // Construct an identity matrix.
        let mut m = Matrix::new(10, 10);
        for i in 0..10 {
            m.set(i, i, true);
        }

        // For any vector, multiplying my by the vector should result in the same vector.
        let mut x = vec![false; 10];
        rng.fill(&mut x[..]);
        assert_eq!(m.mul_by_vec(&x), x);
    }

    #[test]
    fn transpose() {
        let mut rng = thread_rng();
        let mut m = Matrix::new(5, 5);
        for i in 0..m.num_rows {
            for j in 0..m.num_cols {
                if rng.gen::<bool>() & rng.gen::<bool>() {
                    println!("{i}, {j}");
                    m.set(i, j, true);
                }
            }
        }
        m.pretty_print();

        let mt = m.clone().transpose();
        mt.pretty_print();

        assert_eq!(m, mt.transpose());
    }

    #[test]
    fn row_echelon() {
        let mut rng = thread_rng();

        // Generate a random matrix for testing.
        let mut m1 = Matrix::new(5, 10);
        for i in 0..m1.num_rows {
            for j in 0..m1.num_cols {
                m1.set(i, j, rng.gen::<bool>());
            }
        }
        m1.pretty_print();

        let m2 = m1.row_echelon();
        m2.pretty_print();

        // Make sure m2 is in row echelon form.
        let mut prev_ord = None;
        for row in m2.rows.iter() {
            // Empty rows don't move.
            if row.ord() == usize::MAX {
                continue;
            }

            // Make sure the order of the rows strictly increase.
            if let Some(prev_ord) = prev_ord {
                assert!(prev_ord < row.ord());
            }
            prev_ord = Some(row.ord());
        }
    }
}
