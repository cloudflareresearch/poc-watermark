// Copyright (c) 2024 Cloudflare, Inc.

use rand::prelude::*;
use std::time::Instant;

use prc::{LdpcCode, PrfCode, ZeroBitCode};

fn run_test<C: ZeroBitCode>(code: &C, p: f64) {
    let mut rng = thread_rng();
    let n = code.codeword_len();

    println!("Generating key (this may take a few seconds) ...");
    let start = Instant::now();
    let key = code.key_gen();
    let duration = start.elapsed();
    println!("Generated key in {duration:?}");

    let start = Instant::now();
    let word = code.encode(&key);
    let duration = start.elapsed();
    println!("Generated codeword in {duration:?}");

    let start = Instant::now();
    assert!(code.detect(&key, &word), "failed to decode codeword");
    let duration = start.elapsed();
    println!("Decoded codeword in {duration:?}");

    println!("Let's try some non-codewords ...");
    for i in 0..1000 {
        let non_word = std::iter::repeat_with(|| rng.gen())
            .take(n)
            .collect::<Vec<_>>();
        assert!(
            !code.detect(&key, &non_word),
            "decoding a non-codeword succeeded after {} tries",
            i + 1
        );
    }

    let num_modified_bits = (n as f64 * p) as usize;
    println!("Let's try tweaking {num_modified_bits} bits of a bunch of codewords ...");
    for i in 0..100 {
        let mut modified_word = code.encode(&key);
        for i in 0..num_modified_bits {
            modified_word[i] ^= true;
        }
        assert!(
            code.detect(&key, &modified_word),
            "failed to decode modified codeword after {} tries",
            i + 1
        );
    }
}

fn main() {
    println!("LDPC w/ plausibly pseudorandom parameters");
    run_test(&LdpcCode::new(16_384, 10, 0.05, 0.75), 0.02);
    println!("");

    println!("LDPC cranked up as much as possible");
    run_test(&LdpcCode::new(16_384, 1, 0.05, 0.75), 0.33);
    println!("");

    //println!("LDPC2 w/ plausibly pseudorandom parameters");
    //run_test(
    //    &Ldpc2Code {
    //        n: 16_384,
    //        t: 10,
    //        lambda: 118,
    //        eta: 0.0054,
    //        num_test_bits: 20,
    //    },
    //    0.10,
    //);
    //println!("");

    println!("PRF w/ somewhat pseudorandom parameters");
    run_test(
        &PrfCode {
            // The birthday bound is dangerously close here.
            m: 31,
            n: 16_384,
            q: 0.01,
        },
        0.01,
    );
    println!("");

    println!("PRF cranked up as much as possible");
    run_test(
        &PrfCode {
            m: 7,
            n: 16_384,
            q: 0.01,
        },
        0.05,
    );
}
