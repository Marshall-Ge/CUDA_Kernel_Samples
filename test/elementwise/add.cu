#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// &(a) 取地址符操作
// (float4*) - 类型强制转换
// * 解引用运算符
// 可以把连续的个float（从a开始的内存）当成一个float4来访问
#define FLOAT4(a) *(float4*)(&(a))
// 向上取整
#define CEIL(a,b) ((a+b-1)/(b))
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line) {
    // CUDA API 都返回一个 cudaError_t, 以此判断是否成功
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d): \n%s\n", file, line, cudaGetErrorString(error));
        // EXIT_FAILURE 表示程序以失败状态退出，通常对应非零返回值。
        exit(EXIT_FAILURE);
    }
    return;
}

// __global__ 修饰符：代表这是GPU核函数，CPU调用、GPU执行
__global__ void elementwise_add_float4(float* a, float* b, float* c, int N) {
    // blockIdx.x: 当前block的编号
    // blockDim.x: 当前block有多少线程
    // threadIdx.x: 当前线程在block内的编号
    // 全局线程 ID = blockIdx.x * blockDim.x + threadIdx.x
    // 如果不 *4， 线程0处理：0,1,2,3；线程1处理：1,2,3,4；线程2处理：2,3,4,5；
    // 线程之间会重叠访问，数据会乱掉，结果完全错误。
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    // N是数组的总元素个数（Float的数量）
    if (idx >= N) return;

    // *(float4*)(&(a[idx]))  →  一个 float4
    // 包含 a[idx] a[idx+1] a[idx+2] a[idx+3]
    float4 tmp_a = FLOAT4(a[idx]);
    // float4: cuda 内置向量类型
    float4 tmp_b = FLOAT4(b[idx]);
    float4 tmp_c;
    //  float4 最早用于图形学
    // x：第 1 个分量
    // y：第 2 个分量
    // z：第 3 个分量
    // w：第 4 个分量（齐次坐标）
    tmp_c.x = tmp_a.x + tmp_b.x;  // 加第 1 个 float
    tmp_c.y = tmp_a.y + tmp_b.y;  // 加第 2 个 float
    tmp_c.z = tmp_a.z + tmp_b.z;  // 加第 3 个 float
    tmp_c.w = tmp_a.w + tmp_b.w;  // 加第 4 个 float
    // 写回结果
    FLOAT4(c[idx]) = tmp_c;
}

int main() {
    // constexpr 编译期常量关键字（比const更强）
    constexpr int N = 7;
    // CPU端分配变量和初始化
    float* a_h = (float*)malloc(N * sizeof(float));
    float* b_h = (float*)malloc(N * sizeof(float));
    float* c_h = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++){
        a_h[i] = i;
        b_h[i] = N-1-i;
    }

    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;
    // 调用 cudaMalloc 在 GPU 上分配内存
    // 把返回的错误码交给 cudaCheck 检查
    // 把分配到的 GPU 地址写入 a_d
    cudaCheck(cudaMalloc((void**)&a_d, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&b_d, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&c_d, N * sizeof(float)));
    // Host: CPU; Device: GPU;
    // 把 CPU 内存里的数组 b_h 拷贝到 GPU 内存 b_d，并检查是否拷贝成功。
    cudaCheck(cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b_d, b_h, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 每个block启动1024个线程，1024是CUDA允许的最大线程数。
    int block_size = 1024; 
    // 计算需要多少个 block 才能让所有线程覆盖 N 个 float（每线程处理 4 个 float）
    int grid_size = CEIL(CEIL(N,4), 1024);
    // <<< >>> 是 CUDA 的 kernel 启动语法
    // grid_size = block 数量
    // block_size = 每个 block 的线程数
    // 括号里的参数传给 GPU kernel
    elementwise_add_float4<<<grid_size, block_size>>>(a_d, b_d, c_d, N);
    cudaCheck(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    // 打印CPU端的结果进行check
    printf("a_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) printf("%f\n", a_h[i]);
        else printf("%f ", a_h[i]);
    }
    printf("b_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) printf("%f\n", b_h[i]);
        else printf("%f ", b_h[i]);
    }
    printf("c_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) printf("%f\n", c_h[i]);
        else printf("%f ", c_h[i]);
    }
    return 0;
}