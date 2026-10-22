// INT8: GS128 prepacking, VNNI dots, INT32 edges, empty-block pruning, FP32 nonlinearities.
#include <immintrin.h>
#include <cmath>
#include <cstdint>

using V = __m256;
inline V splat(float x) { return _mm256_set1_ps(x); }
inline V load(const float* x) { return _mm256_loadu_ps(x); }
inline V fma(V a, V b, V c) { return _mm256_fmadd_ps(a, b, c); }
inline V clamp(V x, float lo, float hi) {
    return _mm256_max_ps(splat(lo), _mm256_min_ps(splat(hi), x));
}
inline V reciprocal(V x) {
    V r = _mm256_rcp14_ps(x);
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
inline float tanh_bounded(float x) {
    return _mm_cvtss_f32(_mm256_castps256_ps128(tanh_bounded(splat(x))));
}
inline int quantize(float x, float scale) {
    return int(fmaxf(-127, fminf(127, nearbyintf(x * (1.f / scale)))));
}
inline __m256i quantize(V x, float scale) {
    return _mm256_cvtps_epi32(clamp(x * splat(1.f / scale), -127, 127));
}

extern "C" void ks2_vnni(
    const float* input, const int8_t* wt, const float* ts,
    const int8_t* wx, const float* xs, const float* aa, const float* bb,
    const float* warp, const float* ranges, float* output,
    int batch, int spaces, int threads
) {
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (int n = 0; n < batch * spaces; ++n) {
        int s = n % spaces;
        V cv = clamp(load(input + n * 8), -65504, 65504);
        float c[8];
        _mm256_storeu_ps(c, cv);
        float total = ((c[0] + c[1]) + (c[2] + c[3])) + ((c[4] + c[5]) + (c[6] + c[7]));
        const float* w = warp + s * 4;
        const float* rg = ranges + s * 3;
        float t = 4 * tanh_bounded(total / 32), z = tanh_bounded(w[0] * (t + w[1]));
        float t2 = t * t, t3 = t2 * t, z2 = z * z, z3 = z2 * z;
        float basis[6] = {2*t3 - 3*t2 + 1, -2*t3 + 3*t2, t3 - 2*t2 + t,
                          t3 - t2, 2*z2 - 1, 4*z3 - 3*z};
        uint32_t qu[2] = {};
        for (int k = 0; k < 8; ++k)
            qu[k / 4] |= uint32_t(k < 6 ? quantize(basis[k], rg[0]) + 128 : 128) << (8 * (k % 4));
        const int8_t* tw = wt + s * 2560;
        const int32_t* tc = reinterpret_cast<const int32_t*>(wt + spaces * 2560) + s * 320;
        float p[80];
        #pragma GCC unroll 5
        for (int j = 0; j < 5; ++j) {
            __m512 sum = _mm512_setzero_ps();
            #pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
                __m512i acc = _mm512_loadu_si512(tc + (j * 4 + g) * 16);
                #pragma GCC unroll 2
                for (int k = 0; k < 2; ++k) {
                    bool active = false;
                    for (int term = k * 4; term < k * 4 + 4 && term < 6; ++term)
                        if ((term * 72 + j * 16) / 128 == g ||
                            (term * 72 + (j * 16 + 15 < 72 ? j * 16 + 15 : 71)) / 128 == g)
                            active = true;
                    if (active) acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(qu[k]),
                        _mm512_loadu_si512(tw + ((j * 4 + g) * 2 + k) * 64));
                }
                sum = _mm512_add_ps(sum, _mm512_mul_ps(
                    _mm512_cvtepi32_ps(acc), _mm512_set1_ps(rg[0] * ts[s * 4 + g])));
            }
            _mm512_storeu_ps(p + j * 16, sum);
        }

        float e[8], energy[8], es = 0, mean = 0;
        _mm256_storeu_ps(e, tanh_bounded(splat(w[2]) * (cv + splat(w[3]))));
        for (int i = 0; i < 8; ++i) { energy[i] = c[i] * c[i]; es += energy[i]; mean += c[i]; }
        mean /= 8;
        __m256i sum = _mm256_setzero_si256();
        for (int i = 0; i < 8; ++i) {
            float e2 = e[i] * e[i], e3 = e2 * e[i];
            float eb[4] = {1, e[i], 2*e2 - 1, 4*e3 - 3*e[i]};
            uint32_t eq = 0;
            for (int k = 0; k < 4; ++k) eq |= uint32_t(quantize(eb[k], 1.f / 127) + 128) << (8 * k);
            const int32_t* xc = reinterpret_cast<const int32_t*>(wx + spaces * 256) + s * 64 + i * 8;
            __m256i dot = _mm256_dpbusd_epi32(_mm256_loadu_si256((const __m256i*)xc),
                _mm256_set1_epi32(eq), _mm256_loadu_si256((const __m256i*)(wx + s * 256 + i * 32)));
            V ex = _mm256_cvtepi32_ps(dot) * splat(xs[s * 2 + i / 4] / 127);
            float ds = (c[i] - mean) * (c[i] - mean), div = ds + es / 8;
            float importance = energy[i] / (es > 0 ? es : 1), distance = ds / (div > 0 ? div : 1);
            V den = (splat(1) + load(aa + s * 64 + i * 8) * splat(distance))
                  + load(bb + s * 64 + i * 8) * splat(importance);
            V atten = load(p + 8 + i * 8) * reciprocal(den);
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(quantize(atten, rg[1]), quantize(ex, rg[2])));
        }
        V acc = _mm256_cvtepi32_ps(sum) * splat(rg[1] * rg[2]);
        _mm256_storeu_ps(output + n * 8, cv + (load(p) + acc));
    }
}
