// PCIe/host-transfer micro-benchmark for two GPUs: pinned host <-> GPU copies, mapped (zero-copy) access, GPU<->GPU (P2P / via host), concurrent transfers, hand-off latency. Output labels are in Japanese. Build: nvcc -O2 -o pcie_bench pcie_bench.cu -lpthread. Run it while the GPUs are awake (links drop to Gen1 at idle).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__);exit(1);}}while(0)
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static const char* fmt(size_t b){static char s[8][32];static int i=0;char*p=s[i++&7];if(b>=(1<<20))snprintf(p,32,"%zuMiB",b>>20);else if(b>=1024)snprintf(p,32,"%zuKiB",b>>10);else snprintf(p,32,"%zuB",b);return p;}

__global__ void k_read(const float4* src, float4* dst, size_t n){ // mapped host -> device 読み
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x; float4 a={0,0,0,0};
  for(;i<n;i+=st){float4 v=src[i];a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w;}
  if(a.x==12345.f) dst[0]=a;}
__global__ void k_write(float4* dst, size_t n){ // device -> mapped host 書き
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x; float4 v={1,2,3,4};
  for(;i<n;i+=st) dst[i]=v;}
__global__ void k_copy(const float4* s, float4* d, size_t n){ // P2P 直接ロード/ストア
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  for(;i<n;i+=st) d[i]=s[i];}
__global__ void k_wait_flag(volatile int* f,int want){ while(*f!=want){} }
__global__ void k_set_flag(volatile int* f,int v){ *f=v; __threadfence_system(); }

// 反復して GB/s を返す (cudaEvent ではなく壁時計: 同時転送のため)
template<class F> double bw(size_t bytes,int iters,F f){ f(); double t=now(); for(int i=0;i<iters;i++) f(); double d=now()-t; return bytes*(double)iters/d/1e9; }

__global__ void k_pong(volatile int*f,int N){for(int i=0;i<N;i++){while(*f!=2*i+1){} *f=2*i+2; __threadfence_system();}}
__global__ void k_ping(volatile int*f,int N){for(int i=0;i<N;i++){*f=2*i+1; __threadfence_system(); while(*f!=2*i+2){}}}
int main(int argc,char**argv){
  int n; CK(cudaGetDeviceCount(&n)); printf("devices=%d\n",n);
  int can01=0,can10=0; cudaDeviceCanAccessPeer(&can01,0,1); cudaDeviceCanAccessPeer(&can10,1,0);
  printf("canAccessPeer 0->1=%d 1->0=%d\n",can01,can10);
  for(int d=0;d<n;d++){cudaDeviceProp p;cudaGetDeviceProperties(&p,d);printf("GPU%d %s asyncEngines=%d canMapHost=%d\n",d,p.name,p.asyncEngineCount,p.canMapHostMemory);}
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  std::vector<size_t> sizes={4096,65536,1<<20,4<<20,16<<20,64<<20,256<<20};
  const size_t MAXB=256<<20;
  // ===== 1. ホスト<->GPU (pinned, cudaMemcpyAsync) =====
  printf("\n## 1. ピン留めホスト <-> GPU  cudaMemcpyAsync (GB/s)\nsize     | GPU0 H2D | GPU0 D2H | GPU1 H2D | GPU1 D2H\n");
  void* h[2]; void* dv[2]; cudaStream_t st[2];
  for(int d=0;d<n;d++){CK(cudaSetDevice(d));CK(cudaHostAlloc(&h[d],MAXB,cudaHostAllocPortable|cudaHostAllocMapped));memset(h[d],1,MAXB);CK(cudaMalloc(&dv[d],MAXB));CK(cudaStreamCreate(&st[d]));}
  for(size_t s:sizes){ double r[4];
    for(int d=0;d<2;d++){CK(cudaSetDevice(d)); int it=s>=(16<<20)?10:(s>=(1<<20)?50:300);
      r[d*2]=bw(s,it,[&]{cudaMemcpyAsync(dv[d],h[d],s,cudaMemcpyHostToDevice,st[d]);cudaStreamSynchronize(st[d]);});
      r[d*2+1]=bw(s,it,[&]{cudaMemcpyAsync(h[d],dv[d],s,cudaMemcpyDeviceToHost,st[d]);cudaStreamSynchronize(st[d]);});}
    printf("%-8s | %8.2f | %8.2f | %8.2f | %8.2f\n",fmt(s),r[0],r[1],r[2],r[3]);}
  // 小サイズ遅延 (往復1回の同期込み μs)
  printf("\n## 1b. 小サイズ転送の遅延 (cudaMemcpyAsync+同期, μs/回)\nsize | GPU0 H2D | GPU0 D2H | GPU1 H2D | GPU1 D2H\n");
  for(size_t s:{(size_t)64,(size_t)4096,(size_t)65536}){ double r[4];
    for(int d=0;d<2;d++){CK(cudaSetDevice(d));int it=2000;
      for(int k=0;k<2;k++){ double t0; cudaMemcpyAsync(dv[d],h[d],s,k?cudaMemcpyDeviceToHost:cudaMemcpyHostToDevice,st[d]);cudaStreamSynchronize(st[d]);
        t0=now(); for(int i=0;i<it;i++){ if(k)cudaMemcpyAsync(h[d],dv[d],s,cudaMemcpyDeviceToHost,st[d]); else cudaMemcpyAsync(dv[d],h[d],s,cudaMemcpyHostToDevice,st[d]); cudaStreamSynchronize(st[d]);}
        r[d*2+k]=(now()-t0)/it*1e6; }}
    printf("%-5s| %8.1f | %8.1f | %8.1f | %8.1f\n",fmt(s),r[0],r[1],r[2],r[3]);}
  // ===== 2. マップしたホストメモリ (ゼロコピー) をカーネルから =====
  printf("\n## 2. マップしたホストメモリ(ゼロコピー)をカーネルから読み書き (GB/s)\nsize     | GPU0 read | GPU0 write | GPU1 read | GPU1 write\n");
  for(size_t s:{(size_t)(1<<20),(size_t)(16<<20),(size_t)(256<<20)}){ double r[4];
    for(int d=0;d<2;d++){CK(cudaSetDevice(d));void*hd;CK(cudaHostGetDevicePointer(&hd,h[d],0));size_t nv=s/16;int it=s>=(16<<20)?10:100;
      r[d*2]=bw(s,it,[&]{k_read<<<256,256,0,st[d]>>>((const float4*)hd,(float4*)dv[d],nv);cudaStreamSynchronize(st[d]);});
      r[d*2+1]=bw(s,it,[&]{k_write<<<256,256,0,st[d]>>>((float4*)hd,nv);cudaStreamSynchronize(st[d]);});}
    printf("%-8s | %9.2f | %10.2f | %9.2f | %10.2f\n",fmt(s),r[0],r[1],r[2],r[3]);}
  // ===== 3. GPU間 =====
  printf("\n## 3. GPU 間転送 (GB/s)\n");
  if(can01&&can10){ CK(cudaSetDevice(0));CK(cudaDeviceEnablePeerAccess(1,0));CK(cudaSetDevice(1));CK(cudaDeviceEnablePeerAccess(0,0));}
  printf("P2P有効=%d\nsize     | Peer 0->1 | Peer 1->0 | 同時(0->1と1->0)合計 | ホスト経由 0->1(1段) | ホスト経由 0->1(チャンク4MiBパイプライン)\n",(can01&&can10));
  cudaStream_t s01,s10; CK(cudaSetDevice(0));CK(cudaStreamCreate(&s01));CK(cudaSetDevice(1));CK(cudaStreamCreate(&s10));
  for(size_t s:sizes){ int it=s>=(16<<20)?10:(s>=(1<<20)?50:200);
    double a=bw(s,it,[&]{cudaMemcpyPeerAsync(dv[1],1,dv[0],0,s,s01);cudaStreamSynchronize(s01);});
    double b=bw(s,it,[&]{cudaMemcpyPeerAsync(dv[0],0,dv[1],1,s,s10);cudaStreamSynchronize(s10);});
    double c=bw(2*s,it,[&]{cudaMemcpyPeerAsync(dv[1],1,dv[0],0,s,s01);cudaMemcpyPeerAsync(dv[0],0,dv[1],1,s,s10);cudaStreamSynchronize(s01);cudaStreamSynchronize(s10);});
    // ホスト経由 1段: D2H(GPU0) -> 同期 -> H2D(GPU1)
    double e=bw(s,it,[&]{CK(cudaSetDevice(0));cudaMemcpyAsync(h[0],dv[0],s,cudaMemcpyDeviceToHost,st[0]);cudaStreamSynchronize(st[0]);CK(cudaSetDevice(1));cudaMemcpyAsync(dv[1],h[0],s,cudaMemcpyHostToDevice,st[1]);cudaStreamSynchronize(st[1]);});
    // パイプライン: チャンク毎に D2H完了イベントを待って H2D
    double f=0; if(s>=(8<<20)){ size_t ch=4<<20; int nc=s/ch; std::vector<cudaEvent_t> ev(nc); CK(cudaSetDevice(0)); for(auto&x:ev)CK(cudaEventCreateWithFlags(&x,cudaEventDisableTiming));
      f=bw(s,it,[&]{ CK(cudaSetDevice(0)); for(int i=0;i<nc;i++){cudaMemcpyAsync((char*)h[0]+i*ch,(char*)dv[0]+i*ch,ch,cudaMemcpyDeviceToHost,st[0]);cudaEventRecord(ev[i],st[0]);}
        CK(cudaSetDevice(1)); for(int i=0;i<nc;i++){cudaStreamWaitEvent(st[1],ev[i],0);cudaMemcpyAsync((char*)dv[1]+i*ch,(char*)h[0]+i*ch,ch,cudaMemcpyHostToDevice,st[1]);} cudaStreamSynchronize(st[1]);});
      for(auto&x:ev)cudaEventDestroy(x);}
    printf("%-8s | %9.2f | %9.2f | %20.2f | %20.2f | %12.2f\n",fmt(s),a,b,c,e,f);}
  printf("\n## 3b. GPU間 小サイズ遅延 (μs/回, 同期込み)\nsize | Peer 0->1 | Peer 1->0 | ホスト経由 0->1\n");
  for(size_t s:{(size_t)64,(size_t)4096,(size_t)65536,(size_t)(1<<20)}){ int it=2000; double t0;
    t0=now();for(int i=0;i<it;i++){cudaMemcpyPeerAsync(dv[1],1,dv[0],0,s,s01);cudaStreamSynchronize(s01);} double a=(now()-t0)/it*1e6;
    t0=now();for(int i=0;i<it;i++){cudaMemcpyPeerAsync(dv[0],0,dv[1],1,s,s10);cudaStreamSynchronize(s10);} double b=(now()-t0)/it*1e6;
    t0=now();for(int i=0;i<it;i++){CK(cudaSetDevice(0));cudaMemcpyAsync(h[0],dv[0],s,cudaMemcpyDeviceToHost,st[0]);cudaStreamSynchronize(st[0]);CK(cudaSetDevice(1));cudaMemcpyAsync(dv[1],h[0],s,cudaMemcpyHostToDevice,st[1]);cudaStreamSynchronize(st[1]);} double c=(now()-t0)/it*1e6;
    printf("%-5s| %9.1f | %9.1f | %14.1f\n",fmt(s),a,b,c);}
  // カーネルによる P2P ストア/ロード
  if(can01&&can10){ printf("\n## 3c. カーネルからの P2P アクセス (GB/s)\nsize | GPU0が1へ書く | GPU0が1から読む\n");
    for(size_t s:{(size_t)(1<<20),(size_t)(16<<20),(size_t)(256<<20)}){ int it=s>=(16<<20)?10:100; size_t nv=s/16; CK(cudaSetDevice(0));
      double a=bw(s,it,[&]{k_copy<<<256,256,0,st[0]>>>((const float4*)dv[0],(float4*)dv[1],nv);cudaStreamSynchronize(st[0]);});
      double b=bw(s,it,[&]{k_copy<<<256,256,0,st[0]>>>((const float4*)dv[1],(float4*)dv[0],nv);cudaStreamSynchronize(st[0]);});
      printf("%-8s | %12.2f | %14.2f\n",fmt(s),a,b);}}
  // ===== 4. 同時転送 =====
  printf("\n## 4. 同時転送 (ホスト<->GPU, 各スレッドで駆動, GB/s)\n");
  { size_t s=64<<20; int it=20; CK(cudaSetDevice(0));
    struct Res{double v;}; 
    auto run=[&](bool g0h2d,bool g0d2h,bool g1h2d,bool g1d2h,double*out){
      std::atomic<int> go{0}; double res[4]={0,0,0,0}; std::vector<std::thread> th;
      auto job=[&](int d,bool h2d,int idx){ CK(cudaSetDevice(d)); cudaStream_t x; CK(cudaStreamCreate(&x)); void*hh=h[d];
        while(!go.load()){}
        double t=now(); for(int i=0;i<it;i++){ if(h2d)cudaMemcpyAsync(dv[d],hh,s,cudaMemcpyHostToDevice,x); else cudaMemcpyAsync(hh,dv[d],s,cudaMemcpyDeviceToHost,x); } cudaStreamSynchronize(x);
        res[idx]=s*(double)it/(now()-t)/1e9; };
      if(g0h2d)th.emplace_back(job,0,true,0); if(g0d2h)th.emplace_back(job,0,false,1); if(g1h2d)th.emplace_back(job,1,true,2); if(g1d2h)th.emplace_back(job,1,false,3);
      std::this_thread::sleep_for(std::chrono::milliseconds(200)); go=1; for(auto&t:th)t.join();
      for(int i=0;i<4;i++)out[i]=res[i]; };
    double r[4];
    printf("条件 | GPU0 H2D | GPU0 D2H | GPU1 H2D | GPU1 D2H\n");
    run(1,0,0,0,r);printf("GPU0 H2D のみ | %.2f |\n",r[0]);
    run(0,0,1,0,r);printf("GPU1 H2D のみ | | | %.2f |\n",r[2]);
    run(1,0,1,0,r);printf("GPU0 H2D + GPU1 H2D 同時 | %.2f | | %.2f |\n",r[0],r[2]);
    run(0,1,0,1,r);printf("GPU0 D2H + GPU1 D2H 同時 | | %.2f | | %.2f\n",r[1],r[3]);
    run(1,1,0,0,r);printf("GPU0 H2D+D2H 双方向 | %.2f | %.2f | |\n",r[0],r[1]);
    run(0,0,1,1,r);printf("GPU1 H2D+D2H 双方向 | | | %.2f | %.2f\n",r[2],r[3]);
    run(1,1,1,1,r);printf("全4方向 同時 | %.2f | %.2f | %.2f | %.2f\n",r[0],r[1],r[2],r[3]);
    // P2P と GPU0 H2D を同時
    { std::atomic<int> go{0}; double a=0,b=0; std::thread t1([&]{CK(cudaSetDevice(0));cudaStream_t x;CK(cudaStreamCreate(&x));while(!go.load()){}double t=now();for(int i=0;i<it;i++)cudaMemcpyAsync(dv[0],h[0],s,cudaMemcpyHostToDevice,x);cudaStreamSynchronize(x);a=s*(double)it/(now()-t)/1e9;});
      std::thread t2([&]{CK(cudaSetDevice(1));cudaStream_t x;CK(cudaStreamCreate(&x));while(!go.load()){}double t=now();for(int i=0;i<it;i++)cudaMemcpyPeerAsync(dv[0],0,dv[1],1,s,x);cudaStreamSynchronize(x);b=s*(double)it/(now()-t)/1e9;});
      std::this_thread::sleep_for(std::chrono::milliseconds(200));go=1;t1.join();t2.join();printf("GPU0 H2D + Peer 1->0 同時 | %.2f | (peer %.2f)\n",a,b);} }
  // ===== 5. フラグ往復遅延 (マップしたホストメモリ) =====
  printf("\n## 5. 層間受け渡しの遅延: GPU0 -> GPU1 -> GPU0 の往復 (μs/往復)\n");
  { int *fl; CK(cudaHostAlloc((void**)&fl,64,cudaHostAllocMapped|cudaHostAllocPortable)); int*fd; CK(cudaHostGetDevicePointer((void**)&fd,fl,0));
    // 方式A: ホストスレッドが各GPUのストリームを仲介 (cudaMemcpyAsync 4KiB + イベント同期)
    size_t s=4096; int it=3000; double t0=now();
    for(int i=0;i<it;i++){ CK(cudaSetDevice(0));cudaMemcpyPeerAsync(dv[1],1,dv[0],0,s,s01);cudaStreamSynchronize(s01);cudaMemcpyPeerAsync(dv[0],0,dv[1],1,s,s10);cudaStreamSynchronize(s10);}
    printf("A: cudaMemcpyPeer 4KiB 往復(ホスト同期) = %.1f\n",(now()-t0)/it*1e6);
    t0=now(); for(int i=0;i<it;i++){ CK(cudaSetDevice(0));cudaMemcpyAsync(h[0],dv[0],s,cudaMemcpyDeviceToHost,st[0]);cudaStreamSynchronize(st[0]);CK(cudaSetDevice(1));cudaMemcpyAsync(dv[1],h[0],s,cudaMemcpyHostToDevice,st[1]);cudaStreamSynchronize(st[1]);
      cudaMemcpyAsync(h[1],dv[1],s,cudaMemcpyDeviceToHost,st[1]);cudaStreamSynchronize(st[1]);CK(cudaSetDevice(0));cudaMemcpyAsync(dv[0],h[1],s,cudaMemcpyHostToDevice,st[0]);cudaStreamSynchronize(st[0]);}
    printf("B: ホスト経由(D2H→H2D) 4KiB 往復(ホスト同期) = %.1f\n",(now()-t0)/it*1e6);
    // 方式C: GPU カーネルがマップしたホストメモリのフラグを待つ (ホストCPUは介在せずポーリング)。GPU0が書き、GPU1が待つ。
    volatile int* vf=(volatile int*)fl; 
    CK(cudaSetDevice(0)); cudaStream_t c0; CK(cudaStreamCreate(&c0)); CK(cudaSetDevice(1)); cudaStream_t c1; CK(cudaStreamCreate(&c1));
    int it2=2000; *vf=0; t0=now();
    for(int i=0;i<it2;i++){
      CK(cudaSetDevice(1)); k_wait_flag<<<1,1,0,c1>>>(fd,2*i+1); k_set_flag<<<1,1,0,c1>>>(fd,2*i+2);
      CK(cudaSetDevice(0)); k_set_flag<<<1,1,0,c0>>>(fd,2*i+1); k_wait_flag<<<1,1,0,c0>>>(fd,2*i+2);
      cudaStreamSynchronize(c0); cudaStreamSynchronize(c1);}
    printf("C: フラグ(マップホストメモリ)往復、毎回カーネル起動+同期 = %.1f (起動込み)\n",(now()-t0)/it2*1e6);
    // 方式D: 常駐カーネル2本が往復 (起動なし)
    int N=20000; *vf=0; CK(cudaSetDevice(1));
    // GPU1: for i: wait flag==2i+1 ; set 2i+2.  GPU0: set 2i+1; wait 2i+2

    cudaStream_t p1,p0; CK(cudaSetDevice(1));CK(cudaStreamCreate(&p1));CK(cudaSetDevice(0));CK(cudaStreamCreate(&p0));
    CK(cudaSetDevice(1)); k_pong<<<1,1,0,p1>>>(fd,N); CK(cudaSetDevice(0)); t0=now(); k_ping<<<1,1,0,p0>>>(fd,N); cudaStreamSynchronize(p0); double dt=now()-t0; cudaStreamSynchronize(p1);
    printf("D: 常駐カーネル2本がマップホストメモリのフラグで往復 = %.2f μs/往復 (GPU0->host->GPU1->host->GPU0)\n",dt/N*1e6);
    // 方式E: P2P 直接フラグ
    if(can01&&can10){ int*pf; CK(cudaSetDevice(1)); CK(cudaMalloc(&pf,64)); CK(cudaMemset(pf,0,64)); CK(cudaSetDevice(0));
      CK(cudaSetDevice(1)); k_pong<<<1,1,0,p1>>>(pf,N); CK(cudaSetDevice(0)); t0=now(); k_ping<<<1,1,0,p0>>>(pf,N); cudaStreamSynchronize(p0); dt=now()-t0; cudaStreamSynchronize(p1);
      printf("E: 常駐カーネル2本が GPU1 メモリ上のフラグ(P2P経由)で往復 = %.2f μs/往復\n",dt/N*1e6);}
  }
  return 0; }
