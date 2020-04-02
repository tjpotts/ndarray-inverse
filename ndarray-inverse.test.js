import ndarray from 'ndarray';
import pool from 'ndarray-scratch';
import ops from 'ndarray-ops';
import gemm from 'ndarray-gemm';

import inverse from './ndarray-inverse.js';

test('inverts matrices', () => {
    const I2 = pool.eye([2,2]);

    const A = ndarray([1, 2, 3, 4], [2,2]);
    const B = ndarray([0, 0, 0, 0], [2,2]);
    const C = ndarray([0, 0, 0, 0], [2,2]);

    inverse(B, A);
    gemm(C,A,B);

    console.log(A.data);
    console.log(B.data);
    console.log(C.data);

    expect(ops.equals(C,I2)).toBe(true);
});

