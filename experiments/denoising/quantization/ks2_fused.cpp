// Rank-8 CPU inference: all chart geometry and residual in one register-local pass.
#include <immintrin.h>
#include <cmath>
extern "C" __m256 _ZGVdN8v_tanhf(__m256);
static inline __m256 poly(const float* a,float t,float z2,float z3){
 const __m256 tv=_mm256_set1_ps(t);
 __m256 v=_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(a+216),tv),_mm256_loadu_ps(a+144));
 v=_mm256_add_ps(_mm256_mul_ps(v,tv),_mm256_loadu_ps(a+72));
 v=_mm256_add_ps(_mm256_mul_ps(v,tv),_mm256_loadu_ps(a));
 v=_mm256_add_ps(v,_mm256_mul_ps(_mm256_set1_ps(z2),_mm256_loadu_ps(a+288)));
 return _mm256_add_ps(v,_mm256_mul_ps(_mm256_set1_ps(z3),_mm256_loadu_ps(a+360)));
}
extern "C" void ks2_geo(const float* c,const float* a,const float* q,const float* dd,const float* dw,const float* warp,float* out,int batch,int spaces,int threads){
 #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
 for(int n=0;n<batch*spaces;n++){
  const int s=n%spaces;const float* x=c+n*8;const float* ap=a+s*432;const float* qp=q+s*256;const float* wp=warp+s*4;
  // Balanced reduction agrees more closely with the tensor reduction than a serial sum.
  float total=((x[0]+x[1])+(x[2]+x[3]))+((x[4]+x[5])+(x[6]+x[7]));
  float t=total*.125f;float z=tanhf(wp[0]*(t+wp[1]));float z2=2*z*z-1,z3=4*z*z*z-3*z;
  const __m256 xv=_mm256_loadu_ps(x);
  __m256 ev=_ZGVdN8v_tanhf(_mm256_mul_ps(_mm256_set1_ps(wp[2]),_mm256_add_ps(xv,_mm256_set1_ps(wp[3]))));
  alignas(32) float es[8];_mm256_store_ps(es,ev);
  __m256 acc=_mm256_setzero_ps();
  for(int i=0;i<8;i++){
   __m256 e=_mm256_set1_ps(es[i]);const float* qi=qp+i*8;
   __m256 ex=_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(qi+192),e),_mm256_loadu_ps(qi+128));
   ex=_mm256_add_ps(_mm256_mul_ps(ex,e),_mm256_loadu_ps(qi+64));
   ex=_mm256_add_ps(_mm256_mul_ps(ex,e),_mm256_loadu_ps(qi));
   float w=fabsf(total)<1e-6f?.125f:x[i]/total;
   __m256 den=_mm256_add_ps(_mm256_set1_ps(1),_mm256_mul_ps(_mm256_loadu_ps(dd+s*64+i*8),_mm256_set1_ps(fabsf(x[i]-t))));
   den=_mm256_add_ps(den,_mm256_mul_ps(_mm256_loadu_ps(dw+s*64+i*8),_mm256_set1_ps(w*w)));
   acc=_mm256_add_ps(acc,_mm256_div_ps(_mm256_mul_ps(poly(ap+8+i*8,t,z2,z3),ex),den));
  }
  _mm256_storeu_ps(out+n*8,_mm256_add_ps(_mm256_add_ps(xv,poly(ap,t,z2,z3)),acc));
 }
}
