// Copyright (c) 2025 Cloudflare, Inc.

//! Implementations of zero-bit, pseudorandom codes (ia.cr/2024/235).
//!
//! **WARNING:** This code is not yet production ready and is intended for experimental evaluation
//! only. No effort has been made to resist side channels or make it fast. Users should also expect
//! bugs that impact security.

#![allow(clippy::needless_range_loop)]

pub(crate) mod matrix;

use std::io::{Cursor, Read};

use crate::matrix::{Matrix, Row};

use aes::cipher::{generic_array::GenericArray, BlockEncrypt, KeyInit};
use aes::Aes128;
use pyo3::prelude::*;
use rand::{distributions::Bernoulli, prelude::*, seq::SliceRandom};

// Python bindings, built with `maturin develop --release`.
mod py {
    use super::*;
    use lazy_static::lazy_static;
    use pyo3::exceptions::PyValueError;

    lazy_static! {
        /// Code to use for the Python API.
        static ref CODE: LdpcCode = LdpcCode::new(16_384, 1, 0.05, 0.75);
    }

    #[pyfunction]
    fn key_gen() -> LdpcKey {
        CODE.key_gen()
    }

    #[pyfunction]
    pub fn encode(key: &LdpcKey) -> Vec<bool> {
        CODE.encode(key)
    }

    #[pyfunction]
    pub fn detect(key: &LdpcKey, word: Vec<bool>) -> bool {
        CODE.detect(key, &word)
    }

    #[pyfunction]
    pub fn key_from_pem(pem: &str) -> PyResult<LdpcKey> {
        CODE.pem_to_key(pem).map_err(PyValueError::new_err)
    }

    #[pyfunction]
    pub fn key_to_pem(key: &LdpcKey) -> String {
        CODE.key_to_pem(key)
    }

    #[pymodule]
    fn prc(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(key_gen, m)?)?;
        m.add_function(wrap_pyfunction!(encode, m)?)?;
        m.add_function(wrap_pyfunction!(detect, m)?)?;
        m.add_function(wrap_pyfunction!(key_from_pem, m)?)?;
        m.add_function(wrap_pyfunction!(key_to_pem, m)?)?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn large_key() {
            // Generated with `python key_gen.py`.
            let pem_str = include_str!("../../test_key.pem");
            let key = py::CODE.pem_to_key(pem_str).unwrap();
            assert_eq!(CODE.key_to_pem(&key), pem_str);
        }
    }
}

pub trait ZeroBitCode {
    type Key;
    fn key_gen(&self) -> Self::Key;
    fn encode(&self, key: &Self::Key) -> Vec<bool>;
    fn detect(&self, key: &Self::Key, word: &[bool]) -> bool;
    fn codeword_len(&self) -> usize;
}

/// The LDPC ("Low-Density Parity-Check") code from Christ and Gunn (ia.cr/2024/235).
#[pyclass]
#[derive(Debug)]
pub struct LdpcCode {
    // See ia.cr/2024/235, Construction 2 for an explanation.
    pub n: usize,
    pub t: usize,
    pub g: usize,
    pub r: usize,
    pub eta: f64,
    pub zeta: f64,
}

/// The key for LDPC codes.
#[derive(Debug, PartialEq)]
#[pyclass]
pub struct LdpcKey {
    generator: Matrix,
    parity_check: Matrix,
    otp: Vec<bool>,
}

#[pymethods]
impl LdpcCode {
    #[new]
    /// Create an `n`-bit code.
    ///
    /// Note that the smaller the `t`, the higher the robustness, but the weaker the
    /// pseudorandomness.
    pub fn new(n: usize, t: usize, eta: f64, epsilon: f64) -> Self {
        // Theorem 1
        let ne = (n as f64).powf(epsilon);
        let g = ne as usize;
        let r = ne as usize;
        let zeta = ne.powf(-0.25);
        LdpcCode {
            n,
            g,
            t,
            r,
            eta,
            zeta,
        }
    }
}

impl LdpcCode {
    const PEM_TAG_KEY: &'static str = "LDPC KEY";

    /// Encode a key as a PEM block.
    //
    // TODO(cjpatton) Change the key format so that the parity-check matrix and one-time pad are
    // expanded from a PRG seed. We could also store the PRG seed from which the generator matrix
    // is derived as well, but this computation is currently rather expensive, so we wouldn't want
    // to do it when decoding a key. However, note that the generator matrix makes up the bulk of
    // the key.
    pub fn key_to_pem(&self, key: &LdpcKey) -> String {
        let mut bytes = Vec::new();

        bytes.append(&mut key.generator.encode_dense());

        bytes.append(&mut key.parity_check.encode_sparse());

        let mut otp_bytes = vec![0; (key.otp.len() + 7) / 8];
        for (j, x) in key.otp.iter().enumerate() {
            let byte_index = j / 8;
            let bit_index = j % 8;
            otp_bytes[byte_index] |= (*x as u8) << bit_index;
        }
        bytes.extend_from_slice(&otp_bytes);

        pem::encode(&pem::Pem::new(Self::PEM_TAG_KEY, bytes))
    }

    /// Decode a key from a PEM block.
    pub fn pem_to_key(&self, pem_str: &str) -> Result<LdpcKey, String> {
        let pem = pem::parse(pem_str).map_err(|e| e.to_string())?;
        if pem.tag() != Self::PEM_TAG_KEY {
            return Err("unexpected PEM tag".into());
        }

        let mut bytes = Cursor::new(pem.contents());

        let generator = Matrix::decode_dense(&mut bytes, self.n, self.g)
            .map_err(|e| format!("generator: {e}"))?;

        let parity_check = Matrix::decode_sparse(&mut bytes, self.r, self.n)
            .map_err(|e| format!("parity_check: {e}"))?;

        // Parse the one-time pad.
        let mut otp = Vec::with_capacity(self.n);
        let mut otp_bytes = vec![0; (self.n + 7) / 8];
        bytes
            .read_exact(&mut otp_bytes)
            .map_err(|e| format!("otp: {e}"))?;
        for j in 0..self.n {
            let byte_index = j / 8;
            let bit_index = j % 8;
            otp.push(otp_bytes[byte_index] & (1 << bit_index) > 0);
        }

        if usize::try_from(bytes.position()).unwrap() != pem.contents().len() {
            return Err("left over bytes".into());
        }

        Ok(LdpcKey {
            generator,
            parity_check,
            otp,
        })
    }
}

impl ZeroBitCode for LdpcCode {
    type Key = LdpcKey;

    fn key_gen(&self) -> LdpcKey {
        let mut rng = thread_rng();

        let (generator, parity_check) =
            generator_matrix_with_trapdoor(&mut rng, self.n, self.t, self.g, self.r);

        let otp = rand_bit_vec(&mut rng, self.n);

        LdpcKey {
            generator,
            parity_check,
            otp,
        }
    }

    fn encode(&self, key: &LdpcKey) -> Vec<bool> {
        let mut rng = thread_rng();
        let LdpcKey {
            generator,
            parity_check: _,
            otp,
        } = key;

        let mut u = vec![false; self.g];
        rng.fill(&mut u[..]);

        let e = bern_bit_vec(&mut rng, self.n, self.eta);

        let mut x = generator.mul_by_vec(&u);
        debug_assert_eq!(x.len(), self.n);

        for i in 0..self.n {
            x[i] ^= otp[i] ^ e[i];
        }

        x
    }

    fn detect(&self, key: &LdpcKey, x: &[bool]) -> bool {
        let LdpcKey {
            parity_check,
            generator: _,
            otp,
        } = key;

        // Count the number of successful parity checks.
        let count = parity_check
            .mul_by_vec(x)
            .into_iter()
            .zip(parity_check.mul_by_vec(otp))
            .filter(|(left, right)| left ^ right)
            .count() as f64;
        let r = self.r as f64;
        count / r < (0.5 - self.eta)
    }

    fn codeword_len(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod ldpc_tests {
    use super::*;

    impl LdpcCode {
        fn test() -> Self {
            Self::new(500, 2, 0.1, 0.75)
        }
    }

    #[test]
    fn pseudorandomness_smoke_test() {
        let code = LdpcCode::new(20, 3, 0.1, 0.75);
        let key = code.key_gen();
        key.generator.pretty_print();
        let w1 = code.encode(&key);
        let w2 = code.encode(&key);
        println!("{w1:?}");
        println!("{w2:?}");
        let mut count = 0;
        for (b1, b2) in w1.into_iter().zip(w2.into_iter()) {
            if b1 == b2 {
                count += 1;
            }
        }
        if count > 3 * code.n / 4 {
            panic!("codewords are highly correlated");
        }
    }

    #[test]
    fn roundtrip_key_pair() {
        let code = LdpcCode::test();
        let key = code.key_gen();
        let pem = code.key_to_pem(&key);
        assert_eq!(code.pem_to_key(&pem).unwrap(), key);
    }

    #[test]
    fn decode() {
        let code = LdpcCode::test();
        let key = code.key_gen();

        let word = code.encode(&key);
        assert!(code.detect(&key, &word))
    }

    #[test]
    fn decode_non_codeword() {
        let mut rng = thread_rng();
        let code = LdpcCode::test();
        let key = code.key_gen();

        let word = rand_bit_vec(&mut rng, code.n);
        assert!(!code.detect(&key, &word))
    }

    #[test]
    fn decode_edit_one() {
        let code = LdpcCode::test();
        let key = code.key_gen();

        let mut word = code.encode(&key);
        word[0] ^= true;
        assert!(code.detect(&key, &word))
    }

    #[test]
    fn decode_edit_many() {
        let code = LdpcCode::test();
        let key = code.key_gen();

        let mut word = code.encode(&key);
        for i in 0..((code.n as f64) * 0.1) as usize {
            word[i] ^= true;
        }
        assert!(code.detect(&key, &word))
    }
}

/// A second LDPC-based code from aarxiv.org/abs/2410.07369.
pub struct Ldpc2Code {
    /// The codeword length.
    pub n: usize,

    /// The density of each parity check.
    pub t: usize,

    /// sage: (log(binomial(n, t))/log(2)).n()
    pub lambda: usize,

    /// sage: (1 - 2^(-1/lambda)).n()
    pub eta: f64,

    /// sage: ceil(log(1/F)/log(2))
    ///
    /// F is the false positive rate, i.e., the probability of a bit string being a codeword.
    pub num_test_bits: usize,
}

impl Ldpc2Code {
    const MESSAGE: &'static [u8] = b"cool";
    const MESSAGE_LENGTH: usize = { Self::MESSAGE.len() * 8 };

    fn gen_len(&self) -> usize {
        Self::MESSAGE_LENGTH + self.lambda + self.num_test_bits
    }

    fn parity_len(&self) -> usize {
        self.n - self.gen_len()
    }
}

/// Key for the LDPC2 code.
pub struct Ldpc2Key {
    generator: Matrix,
    parity_check: Matrix,
    testbits: Vec<bool>,
    otp: Vec<bool>,
}

impl ZeroBitCode for Ldpc2Code {
    type Key = Ldpc2Key;

    fn key_gen(&self) -> Ldpc2Key {
        let mut rng = thread_rng();
        let k = self.gen_len();
        let r = self.parity_len();

        // Algorithm 1, Line 7: Sample otp and testbits
        let otp = rand_bit_vec(&mut rng, self.n);
        let testbits = rand_bit_vec(&mut rng, self.num_test_bits);

        // Algorithm 1, Lines 8-13: Sample the generator and parity check matrices.
        //
        // NOTE(cjpatton) This appears to be a different sampling procedure than Algorithm 1. I'm
        // not sure how theirs compares to this one. It may end up being faster and/or more secure.
        let (generator, parity_check) =
            generator_matrix_with_trapdoor(&mut rng, self.n, self.t, k, r);

        // TODO(cjpatton) Algorithm 1, Line 14: Permute the generator and parity-check matrix.

        Ldpc2Key {
            generator,
            parity_check,
            testbits,
            otp,
        }
    }

    fn encode(&self, key: &Ldpc2Key) -> Vec<bool> {
        let mut rng = thread_rng();
        let k = self.gen_len();

        // Algorithm 2: Line 2
        let r = rand_bit_vec(&mut rng, self.lambda);

        // Algorithm 2: Line 3
        let mut y = Vec::with_capacity(k);
        y.extend_from_slice(&key.testbits);
        y.extend_from_slice(&r);
        y.extend_from_slice(&bytes_to_bools(Self::MESSAGE));

        // Algorithm 2: Line 4
        let e = bern_bit_vec(&mut rng, self.n, self.eta);

        // Algorithm 2: Line 5
        let mut c = key.generator.mul_by_vec(&y);
        for i in 0..self.n {
            c[i] ^= key.otp[i] ^ e[i];
        }

        c
    }

    fn detect(&self, key: &Ldpc2Key, word: &[bool]) -> bool {
        let r = self.parity_len();

        // Algorithm 3: Line 2
        let mut s = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut s_i: f64 = -1.0;
            s_i = s_i.powi(i32::from(key.otp[i]));
            s_i *= 1.0 - 2.0 * self.eta;
            s_i *= if word[i] { 1.0 } else { -1.0 };
            s.push(s_i);
        }

        // Algorithm 3: Line 3
        let mut s_hat = Vec::with_capacity(r);
        for row in key.parity_check.rows.iter() {
            let mut s_hat_w = 1.0;
            for i in row.js.iter() {
                s_hat_w *= s[*i];
            }
            s_hat.push(s_hat_w);
        }

        // Algorithm 3: Line 4
        let mut a = 0.0;
        let mut b = 0.0;
        let mut c = 0.0;
        for s_hat_w in s_hat {
            // NOTE(cjpatton) The paper doesn't specify the base for the log here. They may mean
            // base 10, but base 2 is more convenient since log_2(1/F) == num_test_bits.
            a += ((1.0 + s_hat_w) / 2.0).log2();
            b += ((1.0 - s_hat_w.powi(2)) / 4.0).log2();
            c += ((1.0 + s_hat_w).log2() - (1.0 - s_hat_w).log2()).powi(2);
        }
        c *= 0.5;
        b *= 0.5;

        // Algorithm 3: Line 5
        let thresh = (c * (self.num_test_bits as f64)).sqrt() + b;
        a >= thresh
    }

    fn codeword_len(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod ldpc2_tests {
    use super::*;

    impl Ldpc2Code {
        // NOTE(cjpatton) It seems like these test parameters aren't suitable for decoding codewords. I
        // find I can get these tests to pass if I modify `detect()` by not multiplying `b` and `c`
        // by `0.5` in `thresh` and flipping the inequality to `a < thresh` instead of `a >=
        // thresh`.
        //
        // The parameters in `examples/test.rs` seem to work, which makes me think the parameters
        // here are too small.
        fn test() -> Ldpc2Code {
            Ldpc2Code {
                n: 500,
                t: 1,
                lambda: 9,
                eta: 0.074,
                num_test_bits: 20,
            }
        }
    }

    #[test]
    fn pseudorandomness_smoke_test() {
        let code = Ldpc2Code::test();
        let key = code.key_gen();
        let w1 = code.encode(&key);
        let w2 = code.encode(&key);
        let mut count = 0;
        for (b1, b2) in w1.into_iter().zip(w2.into_iter()) {
            if b1 == b2 {
                count += 1;
            }
        }
        if count > 3 * code.n / 4 {
            panic!("codewords are highly correlated");
        }
    }

    #[ignore = "test parameters are too small"]
    #[test]
    fn decode() {
        let code = Ldpc2Code::test();
        let key = code.key_gen();

        let word = code.encode(&key);
        assert!(code.detect(&key, &word))
    }

    #[ignore = "test parameters are too small"]
    #[test]
    fn decode_edited_codeword() {
        let code = Ldpc2Code::test();
        let key = code.key_gen();

        let mut word = code.encode(&key);
        for i in 0..((code.n as f64) * 0.1) as usize {
            word[i] ^= true;
        }
        assert!(code.detect(&key, &word))
    }

    #[test]
    fn decode_non_codeword() {
        let mut rng = thread_rng();
        let code = Ldpc2Code::test();
        let key = code.key_gen();

        let word = rand_bit_vec(&mut rng, code.n);
        assert!(!code.detect(&key, &word))
    }
}

/// The PRF-based code from Golowich and Moitra (ia.cr/2024/898).
///
/// Motivation for this scheme, from Section 3:
///
///   In this section, we discuss a new construction of binary PRCs for substitution channels.
///   Though such PRCs were also obtained by [CG24], the codes in [CG24] relied on relatively
///   strong average-case hardness assumptions, in the sense that they imply the existence of
///   public-key cryptography (i.e., in the context of Impagliazzo’s Five Worlds [Imp95], they
///   imply primitives in “Cryptomania”). In contrast, our construction relies only on the hardness
///   of the existence of a family of pseudorandom functions that enjoys a certain locality
///   property; such an assumption is generally believed to be weaker than the ones in [CG24], in
///   the sense that it is only known to yield cryptographic primitives in “Minicrypt”.
pub struct PrfCode {
    pub m: usize,
    pub n: usize,
    pub q: f64,
}

/// The key for the PRF code.
pub struct PrfKey {
    pub s: [u8; 16],
    pub z: Vec<bool>,
    pub pi: Vec<usize>,
}

impl ZeroBitCode for PrfCode {
    type Key = PrfKey;

    fn key_gen(&self) -> PrfKey {
        let mut rng = thread_rng();
        let s = rng.gen();
        let z = std::iter::repeat_with(|| rng.gen()).take(self.n).collect();
        let mut pi = (0..self.n).collect::<Vec<_>>();
        pi.shuffle(&mut rng);
        PrfKey { s, z, pi }
    }

    fn encode(&self, k: &PrfKey) -> Vec<bool> {
        let mut rng = thread_rng();
        let error_dist = Bernoulli::new(self.q).unwrap();
        let mut w1 = Vec::with_capacity(self.n);

        debug_assert!(self.n % (self.m + 1) == 0);
        debug_assert!(self.m <= 128);
        let aes = Aes128::new(&GenericArray::from(k.s));
        for _ in (0..self.n).step_by(self.m + 1) {
            let mut x = GenericArray::from(rng.gen::<[u8; 16]>());
            for j in 0..128 {
                let byte_index = j / 8;
                let bit_index = j % 8;
                if j < self.m {
                    // Include the bit in the codeword.
                    w1.push(x[byte_index] & (1 << bit_index) > 0);
                } else {
                    // Clear the bit.
                    x[byte_index] &= !(1 << bit_index);
                }
            }

            // NOTE We may be making an invalid assumption about AES here. What we need is a "weak
            // PRF with noise level q" (Definition 3.1). AES is a weak PRF, but maybe not with
            // "noise level q".
            aes.encrypt_block(&mut x);
            let y = x[0] & 1 > 0;
            let e = error_dist.sample(&mut rng);
            w1.push(y ^ e);
        }

        for i in 0..self.n {
            w1[i] ^= k.z[i];
        }

        let mut w2 = vec![false; self.n];
        for (i, bit) in k.pi.iter().copied().zip(w1.into_iter()) {
            w2[i] = bit;
        }

        w2
    }

    fn detect(&self, k: &PrfKey, w2: &[bool]) -> bool {
        let mut w1 = vec![false; self.n];
        for (i, j) in k.pi.iter().copied().enumerate() {
            w1[i] = w2[j] ^ k.z[i];
        }

        let aes = Aes128::new(&GenericArray::from(k.s));

        let mut weight = 0;
        for chunk in w1.chunks_exact(self.m + 1) {
            let mut x = GenericArray::from([0; 16]);
            for j in 0..self.m {
                let byte_index = j / 8;
                let bit_index = j % 8;
                x[byte_index] |= u8::from(chunk[j]) << bit_index;
            }

            aes.encrypt_block(&mut x);
            let w = x[0] & 1 > 0;
            weight += usize::from(w == chunk[self.m])
        }

        let threshold = {
            debug_assert!(self.n % (self.m + 1) == 0);
            let chunks = (self.n / (self.m + 1)) as f64;
            chunks / 2.0 + chunks.ln() * chunks.sqrt()
        };
        (weight as f64) > threshold
    }

    fn codeword_len(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod prf_tests {
    use super::*;

    impl PrfCode {
        fn test() -> PrfCode {
            PrfCode {
                m: 31,
                n: 16_384,
                q: 0.01,
            }
        }
    }

    #[test]
    fn pseudorandomness_smoke_test() {
        let code = PrfCode::test();
        let key = code.key_gen();
        let w1 = code.encode(&key);
        let w2 = code.encode(&key);
        let mut count = 0;
        for (b1, b2) in w1.into_iter().zip(w2.into_iter()) {
            if b1 == b2 {
                count += 1;
            }
        }
        if count > 3 * code.n / 4 {
            panic!("codewords are highly correlated");
        }
    }

    #[test]
    fn decode() {
        let code = PrfCode::test();
        let key = code.key_gen();
        let word = code.encode(&key);
        assert!(code.detect(&key, &word));
    }

    #[test]
    fn decode_edit_many() {
        let code = PrfCode::test();
        let key = code.key_gen();
        let mut word = code.encode(&key);
        for i in 0..(0.01 * code.n as f64) as usize {
            word[i] ^= true;
        }
        assert!(code.detect(&key, &word));
    }

    #[test]
    fn decode_non_codeword() {
        let mut rng = thread_rng();
        let code = PrfCode::test();
        let key = code.key_gen();

        let word = rand_bit_vec(&mut rng, code.n);
        assert!(!code.detect(&key, &word));
    }
}

fn rand_bit_vec<R: Rng + ?Sized>(rng: &mut R, len: usize) -> Vec<bool> {
    std::iter::repeat_with(|| rng.gen()).take(len).collect()
}

fn bern_bit_vec<R: Rng + ?Sized>(rng: &mut R, len: usize, eta: f64) -> Vec<bool> {
    let dist = Bernoulli::new(eta).unwrap();
    std::iter::repeat_with(|| dist.sample(rng))
        .take(len)
        .collect()
}

fn bytes_to_bools(bytes: &[u8]) -> Vec<bool> {
    bytes
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |i| (byte & (1 << i)) != 0))
        .collect()
}

/// Sample a generator matrix with a corresponding trapdoor parity-check matrix as defined in
/// ia.cr/2024/235, Definition 3.
///
/// The generator matrix `n`-by-`g`; the parity check matrix is `r`-by-`n` and has `t`-sparse rows,
/// meaning exactly `t` rows are set.
fn generator_matrix_with_trapdoor<R: Rng + ?Sized>(
    rng: &mut R,
    n: usize,
    t: usize,
    g: usize,
    r: usize,
) -> (Matrix, Matrix) {
    // We start off by choosing a random parity-check matrix. Next, we compute the basis of the
    // kernel for that matrix. The generator matrix is comprised of vectors sampled randomly from
    // the kernel. To sample a vector, we take a random linear combination of the basis vectors.
    //
    // TODO(cjpatton) There are some unnecessary matrix transpose operations in this code that
    // should be easy to remove. However, for some parameter choices, this algorithm can take quite
    // a long time (a matter of seconds), and the transposes are not a significant part of the
    // runtime.

    // Step 1: Generate the parity-check matrix used for decoding. This is a random matrix where
    // each row has exactly `t` columns set.
    let mut parity_check = Matrix::new(r, n);
    for i in 0..r {
        // t of the entries are set to 1.
        let mut row = vec![false; n];
        for j in 0..t {
            row[j] = true;
        }
        row.shuffle(rng);

        for j in 0..n {
            if row[j] {
                parity_check.set(i, j, true);
            }
        }
    }

    let mut m = Matrix {
        num_rows: r + n,
        num_cols: n,
        rows: parity_check.rows.clone(),
    };

    // Step 2: Augment the parity-check matrix with the `n`-by-`n` identity matrix.
    for i in 0..n {
        m.set(i + r, i, true);
    }

    #[cfg(test)]
    {
        println!("augmented parity check");
        m.pretty_print();
    }

    // Step 3: Convert the augmented matrix to column echelon form. This is the computationally
    // heavy step.
    let m = m.transpose();
    let m = m.row_echelon();
    let m = m.transpose();
    #[cfg(test)]
    {
        println!("col echelon");
        m.pretty_print();
    }

    // Step 4: Compute the dimension of the basis. The is given by the number of columns for which
    // the first `r` columns are zero. See
    // https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination.
    let mut leading_col = 0;
    for Row { i: _, js } in m.rows[0..m.row_entry(r)].iter() {
        leading_col = std::cmp::max(leading_col, js.last().copied().unwrap());
    }
    let basis_dim = n - leading_col - 1;
    assert_ne!(basis_dim, n);
    assert_ne!(basis_dim, 0);

    // Step 5: Extract the basis vectors. We're carving out the bottom-right corner of the matrix,
    // so we'll have to shift the row and column indices accordingly.
    let basis = {
        let mut basis = Matrix::new(n, basis_dim);
        for row_entry in m.row_entry(r)..m.rows.len() {
            let col_entry = m.rows[row_entry].col_entry(n - basis_dim);
            basis.rows.push(Row {
                i: m.rows[row_entry].i - r,
                js: m.rows[row_entry].js[col_entry..]
                    .iter()
                    .map(|j| *j - (n - basis_dim))
                    .collect(),
            });
        }
        basis
    };
    #[cfg(test)]
    {
        println!("basis");
        basis.pretty_print();
    }

    // Step 6: Sample vectors from the kernel by taking random linear combinations of the kernel's
    // basis vectors.
    let mut generator = Matrix::new(g, n);
    for i in 0..g {
        let c = rand_bit_vec(rng, basis_dim);
        for Row { i: k, js } in basis.rows.iter() {
            for j in js {
                generator.add_to(i, *k, c[*j] & basis.get(*k, *j));
            }
        }
    }
    let generator = generator.transpose();
    debug_assert!(!generator.is_zero());
    #[cfg(test)]
    {
        println!("generator");
        generator.pretty_print();
    }

    (generator, parity_check)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_matrix_with_trapdoor() {
        let (generator, parity_check) =
            generator_matrix_with_trapdoor(&mut thread_rng(), 100, 10, 100, 75);

        let m = parity_check.mul(&generator);
        m.pretty_print();

        // Check that the columns of the generator matrix G are in the kernel of the parity-check
        // matrix P, i.e., PG = 0. See ia.cr/2024/235, Definition 3.
        //
        // NOTE We use `n=100` for this test so that matrix multiplication takes less time.
        assert!(m.is_zero());
    }
}
