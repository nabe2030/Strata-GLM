// mb.c: per-thread throughput of the IQ3_S / IQ4_XS x Q8_K dot-product kernels, small buffer (in L2) vs large (DRAM, 4 KB pages). Needs the ggml quant sources; see the header includes.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <sched.h>
#include <stdint.h>
typedef void (*dotfn)(int n,float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc);
extern void ggml_vec_dot_iq3_s_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_iq4_xs_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_cpu_init(void);
static int K=4096; static size_t RB; static size_t BUF; static dotfn fn; static int cpu_map[16]; static double secs=2.0;
static char q8[16*(4+256+32)+64];
typedef struct{int id;double gbs;}T;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void*run(void*a){T*t=a;cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu_map[t->id],&s);pthread_setaffinity_np(pthread_self(),sizeof s,&s);
 uint8_t*b=aligned_alloc(4096,BUF);for(size_t i=0;i<BUF;i++)b[i]=(uint8_t)(i*2654435761u>>13);
 size_t nrows=BUF/RB;float o;double t0=now();size_t bytes=0;
 while(now()-t0<secs){for(int rep=0;rep<64;rep++){for(size_t r=0;r<nrows;r++){fn(K,&o,0,b+r*RB,0,q8,0,1);}bytes+=nrows*RB; if(now()-t0>=secs)break;}}
 t->gbs=bytes/(now()-t0)/1e9;return 0;}
int main(int c,char**v){ggml_cpu_init();int th=atoi(v[1]);BUF=(size_t)atof(v[2])*1048576;const char*k=v[3];
 if(!strcmp(k,"iq3s")){fn=ggml_vec_dot_iq3_s_q8_K;RB=(K/256)*110;}else{fn=ggml_vec_dot_iq4_xs_q8_K;RB=(K/256)*136;}
 BUF=BUF/RB*RB; memset(q8,0,sizeof q8);for(int i=0;i<16;i++){*(float*)(q8+i*292)=0.01f;memset(q8+i*292+4,1,256);}
 for(int i=0;i<12;i++)cpu_map[i]=i;
 pthread_t p[12];T t[12];for(int i=0;i<th;i++){t[i].id=i;pthread_create(&p[i],0,run,&t[i]);}
 double tot=0;for(int i=0;i<th;i++){pthread_join(p[i],0);tot+=t[i].gbs;}
 printf("%s th=%d buf=%.0fMiB total=%.2f GB/s per-thread=%.2f\n",k,th,BUF/1048576.0,tot,tot/th);return 0;}
