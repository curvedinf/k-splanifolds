// FP32: coefficient folding, 16-patch SIMD, bounded rational tanh, refined reciprocals.
#include <immintrin.h>

using V = __m512;
inline V splat(float x) { return _mm512_set1_ps(x); }
inline V fma(V a, V b, V c) { return _mm512_fmadd_ps(a, b, c); }
inline V floor_at(V x, float lo) { return _mm512_max_ps(x, splat(lo)); }
inline V clamp(V x, float lo, float hi) {
    return floor_at(_mm512_min_ps(x, splat(hi)), lo);
}
inline V reciprocal(V x) {
    V r = _mm512_rcp14_ps(x);
    return r * (splat(2) - x * r);
}
inline V tanh_bounded(V x) {
    x = clamp(x, -8, 8);
    V y = x * x;
    V n = fma(y, y + splat(378), splat(17325));
    n = fma(y, n, splat(135135));
    V d = fma(y, splat(28), splat(3150));
    d = fma(y, d, splat(62370));
    d = fma(y, d, splat(135135));
    return clamp(x * n * reciprocal(d), -1, 1);
}
inline V polynomial(const float* a, V t, V z2, V z3) {
    V p = fma(splat(a[216]), t, splat(a[144]));
    p = fma(p, t, splat(a[72]));
    p = fma(p, t, splat(a[0]));
    return fma(z3, splat(a[360]), fma(z2, splat(a[288]), p));
}

extern "C" void ks2_horner(
    const float* input, const float* a, const float* q,
    const float* dd, const float* dw, const float* warp, float* output,
    int batch, int spaces, int threads
) {
    int blocks = (batch + 15) / 16;
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (int n = 0; n < blocks * spaces; ++n) {
        int s = n % spaces, b = (n / spaces) * 16;
        __mmask16 mask = batch - b >= 16 ? 65535 : (1u << (batch - b)) - 1;
        __m512i offsets = _mm512_mullo_epi32(
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm512_set1_epi32(spaces * 8)
        );
        V c[8];
        for (int i = 0; i < 8; ++i)
            c[i] = clamp(_mm512_mask_i32gather_ps(
                splat(0), mask, offsets, input + (b * spaces + s) * 8 + i, 4
            ), -65504, 65504);

        V total = ((c[0] + c[1]) + (c[2] + c[3])) + ((c[4] + c[5]) + (c[6] + c[7]));
        V energy = splat(0);
        for (int i = 0; i < 8; ++i) energy = fma(c[i], c[i], energy);
        V inv_energy = reciprocal(floor_at(energy, 1e-30f)), mean = total * splat(.125f);
        const float* w = warp + s * 4;
        V t = splat(4) * tanh_bounded(total * splat(1.f / 32));
        V z = tanh_bounded(splat(w[0]) * (t + splat(w[1])));
        V z2 = splat(2) * (z * z) - splat(1);
        V z3 = splat(4) * ((z * z) * z) - splat(3) * z;
        V acc[8];
        for (int j = 0; j < 8; ++j) acc[j] = polynomial(a + s * 432 + j, t, z2, z3);

        for (int i = 0; i < 8; ++i) {
            V e = tanh_bounded(splat(w[2]) * (c[i] + splat(w[3])));
            V delta = c[i] - mean, ds = delta * delta;
            V distance = ds * reciprocal(floor_at(ds + energy * splat(.125f), 1e-30f));
            V importance = (c[i] * c[i]) * inv_energy;
            #pragma GCC unroll 8
            for (int j = 0; j < 8; ++j) {
                const float* qi = q + s * 256 + i * 8 + j;
                V ex = fma(splat(qi[192]), e, splat(qi[128]));
                ex = fma(ex, e, splat(qi[64]));
                ex = fma(ex, e, splat(qi[0]));
                V den = fma(splat(dd[s * 64 + i * 8 + j]), distance, splat(1));
                den = fma(splat(dw[s * 64 + i * 8 + j]), importance, den);
                V edge = polynomial(a + s * 432 + 8 + i * 8 + j, t, z2, z3) * ex;
                acc[j] = fma(edge, reciprocal(den), acc[j]);
            }
        }
        for (int j = 0; j < 8; ++j)
            _mm512_mask_i32scatter_ps(
                output + (b * spaces + s) * 8 + j, mask, offsets, c[j] + acc[j], 4
            );
    }
}
