import ndarray from 'ndarray';
import pool from 'ndarray-scratch';
import ops from 'ndarray-ops';
import gemm from 'ndarray-gemm';

import inverse from './ndarray-inverse.js';

test('inverts matrices', () => {
    const I2 = pool.eye([2,2]);

    const A = ndarray([1, 2, 3, 4], [2,2]);
    const B = pool.zeros([2,2]);
    const C = pool.zeros([2,2]);

    expect(inverse(B, A)).toBe(true);
    gemm(C,A,B);

    expect(ops.equals(C,I2)).toBe(true);
});

test('returns false if input is non-invertible', () => {
    const A = pool.zeros([2,2]);
    const B = pool.zeros([2,2]);
    expect(inverse(B,A)).toBe(false);
});

test('throws error unless input is 2-dimensional', () => {
    let A = pool.zeros([3]);
    let B = pool.zeros([3]);
    expect(() => inverse(B,A)).toThrow();

    A = pool.zeros([3,3,3]);
    B = pool.zeros([3,3,3]);
    expect(() => inverse(B,A)).toThrow();
});

test('throws error if matrix dimensions to not match', () => {
    let A = pool.eye([3,3]);
    let B = pool.zeros([2,3]);
    expect(() => inverse(B,A)).toThrow();

    B = pool.zeros([3,2]);
    expect(() => inverse(B,A)).toThrow();
});

