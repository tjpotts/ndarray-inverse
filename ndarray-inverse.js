import ndarray from 'ndarray';
import pool from 'ndarray-scratch';
import crout from 'ndarray-crout-decomposition';
import gemm from 'ndarray-gemm';

// Invert a lower triangular matrix via forward-substitution
// Source:
// https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
function inverse_lower_triangular(X, L) {
    for (let col = 0; col < L.shape[1]; col++) {
        for (let m = 0; m < L.shape[0]; m++) {
            let b = (col == m) ? 1 : 0;
            let acc = 0;
            for (let i = 0; i < m; i++) {
                acc += L.get(m,i) * X.get(i,col);
            }
            X.set(m, col, (b - acc) / L.get(m,m));
        }
    }
}

// Inverts a matrix by performing a crout decomposition, inverting the resulting triangular matrices
// and then multiplying the component inverses
export default function inverse(b, a) {
    // TODO: Check det(a) != 0
    if (a.dimension !== 2 || a.shape[0] !== a.shape[1]) {
        throw new Error("Non-square matrix a");
    }
    if (a.shape[0] !== b.shape[0] || a.shape[1] !== b.shape[1]) {
        throw new Error("Array dimension mismatch");
    }

    let L = pool.zeros(a.shape);
    let U = pool.zeros(a.shape);
    // NOTE: crout implementation used assumes column-first indexing, hence the transposes
    if (!crout(a.transpose(1,0), L.transpose(1,0), U.transpose(1,0))) {
        return false;
    }

    let Linv = pool.zeros(a.shape);
    let Uinv = pool.zeros(a.shape);
    inverse_lower_triangular(Linv, L);
    inverse_lower_triangular(Uinv.transpose(1,0), U.transpose(1, 0));

    gemm(b, Uinv, Linv);

    pool.free(L);
    pool.free(U);
    pool.free(Linv);
    pool.free(Uinv);

    return true;
}

