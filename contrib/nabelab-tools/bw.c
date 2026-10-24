// bw.c: each thread reads its own 512 MiB buffer (AVX2); prints total GB/s per thread count. Run with the server stopped. Build: gcc -O2 -mavx2 -o bw bw.c -lpthread
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#define SZ (512ull<<20)
static char* buf[16]; static int NT; static volatile double sink[16];
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static pthread_barrier_t bar;
static void* run(void* a){ long id=(long)a; const __m256i* p=(const __m256i*)buf[id]; size_t n=SZ/32;
  __m256i s0=_mm256_setzero_si256(),s1=s0,s2=s0,s3=s0;
  pthread_barrier_wait(&bar);
  for(int rep=0;rep<4;rep++) for(size_t i=0;i<n;i+=8){ s0=_mm256_add_epi64(s0,_mm256_load_si256(p+i));s1=_mm256_add_epi64(s1,_mm256_load_si256(p+i+1));s2=_mm256_add_epi64(s2,_mm256_load_si256(p+i+2));s3=_mm256_add_epi64(s3,_mm256_load_si256(p+i+3));
    s0=_mm256_add_epi64(s0,_mm256_load_si256(p+i+4));s1=_mm256_add_epi64(s1,_mm256_load_si256(p+i+5));s2=_mm256_add_epi64(s2,_mm256_load_si256(p+i+6));s3=_mm256_add_epi64(s3,_mm256_load_si256(p+i+7)); }
  long long o[4];_mm256_storeu_si256((__m256i*)o,_mm256_add_epi64(_mm256_add_epi64(s0,s1),_mm256_add_epi64(s2,s3)));sink[id]=o[0]; pthread_barrier_wait(&bar); return 0; }
int main(int c,char**v){ NT=atoi(v[1]); for(int i=0;i<NT;i++){buf[i]=aligned_alloc(64,SZ);memset(buf[i],i+1,SZ);}
  pthread_barrier_init(&bar,0,NT+1); pthread_t t[16]; for(long i=0;i<NT;i++)pthread_create(&t[i],0,run,(void*)i);
  pthread_barrier_wait(&bar); double t0=now(); pthread_barrier_wait(&bar); double dt=now()-t0;
  printf("threads %d: %.1f GB/s total (%.1f per thread)\n",NT,4.0*NT*SZ/dt/1e9,4.0*SZ/dt/1e9); return 0; }
