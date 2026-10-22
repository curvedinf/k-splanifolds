// Analytic VJP: 16-token AVX-512, chart-local gradient reductions, shared-statistic
// adjoints, fused edge derivatives, bounded predicates, libmvec tanh, FP32 sums.
#include <immintrin.h>
#include <cmath>
#include <algorithm>

using V = __m512;
extern "C" V _ZGVeN16v_tanhf(V);
constexpr int P = 820;
inline V splat(float x) {return _mm512_set1_ps(x);}
inline V zero() {return _mm512_setzero_ps();}
inline V vmax(V x,float lo) {return _mm512_max_ps(x,splat(lo));}
inline V select(__mmask16 mask,V x) {return _mm512_maskz_mov_ps(mask,x);}
inline float sigmoid(float x) {return 1.f/(1.f+std::exp(-x));}
inline float softplus(float x) {return x>20 ? x : std::log1p(std::exp(x));}

template<bool BACK>
void geometry(const float* input,const float* packed,const float* grad,
              float* output,float* dp,int batch,int spaces,int threads,bool residual) {
    constexpr float offset = -2.086258912947793f;
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
    for(int s=0;s<spaces;++s) {
        const float* p=packed+s*P;
        V g[P],aa[64],bb[64];
        if constexpr(BACK) for(int k=0;k<P;++k)g[k]=zero();
        for(int e=0;e<64;++e) {aa[e]=splat(8*sigmoid(p[688+e]+offset));bb[e]=splat(8*sigmoid(p[752+e]+offset));}
        V scale=splat(softplus(p[816])+1e-4f),shift=splat(.5f*std::tanh(p[817]));
        V escale=splat(softplus(p[818])+1e-4f),eshift=splat(.5f*std::tanh(p[819]));
        __m512i offsets=_mm512_mullo_epi32(_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),_mm512_set1_epi32(spaces*8));
        for(int n=0;n<batch;n+=16) {
            int base=(n*spaces+s)*8;
            __mmask16 mask=batch-n>=16 ? 65535 : (1u<<(batch-n))-1;
            V raw[8],x[8],dx[8],u[8],result[8],mean=zero(),energy=zero();
            for(int i=0;i<8;++i) {
                raw[i]=_mm512_mask_i32gather_ps(zero(),mask,offsets,input+base+i,4);
                x[i]=_mm512_min_ps(vmax(raw[i],-65504.f),splat(65504.f));
                mean+=x[i]*splat(.125f);energy+=x[i]*x[i];dx[i]=zero();
                if constexpr(BACK)u[i]=_mm512_mask_i32gather_ps(zero(),mask,offsets,grad+base+i,4);
            }
            V t0=_ZGVeN16v_tanhf(mean*splat(.25f)),t=splat(4)*t0,z=_ZGVeN16v_tanhf(scale*(t+shift));
            V t2=t*t,t3=t2*t,z2=z*z;
            V basis[6]={splat(2)*t3-splat(3)*t2+splat(1),-splat(2)*t3+splat(3)*t2,t3-splat(2)*t2+t,t3-t2,splat(2)*z2-splat(1),splat(4)*z2*z-splat(3)*z};
            V dt[4]={splat(6)*t2-splat(6)*t,-splat(6)*t2+splat(6)*t,splat(3)*t2-splat(4)*t+splat(1),splat(3)*t2-splat(2)*t};
            V db[6],de=zero(),dm=zero(),gshift=zero(),gscale=zero();
            for(int k=0;k<6;++k)db[k]=zero();
            for(int j=0;j<8;++j) {
                if constexpr(!BACK)result[j]=residual ? x[j] : zero();
                for(int k=0;k<6;++k) {
                    if constexpr(BACK) {g[k*8+j]+=u[j]*basis[k];db[k]+=u[j]*splat(p[k*8+j]);}
                    else result[j]+=splat(p[k*8+j])*basis[k];
                }
            }
            for(int i=0;i<8;++i) {
                V delta=x[i]-mean,ds=delta*delta,rawden=ds+energy*splat(.125f);
                V ed=vmax(energy,1e-30f),dd=vmax(rawden,1e-30f);
                V importance=x[i]*x[i]/ed,distance=ds/dd;
                V w=_ZGVeN16v_tanhf(escale*(x[i]+eshift)),w2=w*w;
                V xb[4]={splat(1),w,splat(2)*w2-splat(1),splat(4)*w2*w-splat(3)*w};
                V dxb[4]={zero(),splat(1),splat(4)*w,splat(12)*w2-splat(3)};
                V gw=zero(),gd=zero(),gi=zero();
                for(int j=0;j<8;++j) {
                    int e=i*8+j;
                    V a=zero(),b=zero();
                    for(int k=0;k<6;++k)a+=splat(p[48+k*64+e])*basis[k];
                    for(int k=0;k<4;++k)b+=splat(p[432+k*64+e])*xb[k];
                    V inv=splat(1)/(splat(1)+aa[e]*distance+bb[e]*importance);
                    if constexpr(!BACK)result[j]+=a*b*inv;
                    else {
                        V ga=u[j]*b*inv,gb=u[j]*a*inv,den_grad=-(u[j]*a*inv)*(b*inv);
                        gd+=den_grad*aa[e];gi+=den_grad*bb[e];
                        g[688+e]+=den_grad*distance;g[752+e]+=den_grad*importance;
                        for(int k=0;k<6;++k) {g[48+k*64+e]+=ga*basis[k];db[k]+=ga*splat(p[48+k*64+e]);}
                        for(int k=0;k<4;++k) {g[432+k*64+e]+=gb*xb[k];gw+=gb*splat(p[432+k*64+e])*dxb[k];}
                    }
                }
                if constexpr(BACK) {
                    V gu=gw*(splat(1)-w2);
                    gscale+=gu*(x[i]+eshift);gshift+=gu*escale;dx[i]+=gu*escale;
                    V denomgrad=select(_mm512_cmp_ps_mask(rawden,splat(1e-30f),_CMP_GE_OQ),gd*distance/dd);
                    V ddelta=(gd/dd-denomgrad)*(splat(2)*delta);
                    dx[i]+=ddelta;dm-=ddelta;de-=denomgrad*splat(.125f);
                    dx[i]+=gi*(splat(2)*x[i]/ed);
                    de-=select(_mm512_cmp_ps_mask(energy,splat(1e-30f),_CMP_GE_OQ),gi*importance/ed);
                }
            }
            if constexpr(BACK) {
                V gz=db[4]*splat(4)*z+db[5]*(splat(12)*z2-splat(3));
                V gu=gz*(splat(1)-z2),gt=gu*scale;
                for(int k=0;k<4;++k)gt+=db[k]*dt[k];
                g[816]+=gu*(t+shift);g[817]+=gu*scale;g[818]+=gscale;g[819]+=gshift;
                dm+=gt*(splat(1)-t0*t0);
                for(int i=0;i<8;++i) {
                    __mmask16 active=mask & _mm512_cmp_ps_mask(raw[i],splat(-65504.f),_CMP_GE_OQ) & _mm512_cmp_ps_mask(raw[i],splat(65504.f),_CMP_LE_OQ);
                    V v=select(active,dx[i]+dm*splat(.125f)+de*(splat(2)*x[i])+(residual ? u[i] : zero()));
                    _mm512_mask_i32scatter_ps(output+base+i,mask,offsets,v,4);
                }
            } else for(int j=0;j<8;++j)_mm512_mask_i32scatter_ps(output+base+j,mask,offsets,result[j],4);
        }
        if constexpr(BACK) {
            for(int e=0;e<64;++e) {g[688+e]*=aa[e]*(splat(1)-aa[e]*splat(.125f));g[752+e]*=bb[e]*(splat(1)-bb[e]*splat(.125f));}
            g[816]*=splat(sigmoid(p[816]));g[818]*=splat(sigmoid(p[818]));
            float sh=std::tanh(p[817]),esh=std::tanh(p[819]);
            g[817]*=splat(.5f*(1-sh*sh));g[819]*=splat(.5f*(1-esh*esh));
            for(int k=0;k<P;++k)dp[s*P+k]=_mm512_reduce_add_ps(g[k]);
        }
    }
}
extern "C" void ks2_train_forward(const float* x,const float* p,float* y,int n,int s,int threads,int residual) {geometry<false>(x,p,nullptr,y,nullptr,n,s,threads,residual);}
extern "C" void ks2_train_backward(const float* x,const float* p,const float* dy,float* dx,float* dp,int n,int s,int threads,int residual) {geometry<true>(x,p,dy,dx,dp,n,s,threads,residual);}
