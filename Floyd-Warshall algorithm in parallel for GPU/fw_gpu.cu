#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "workshop.h"

#define GRAPH_SIZE 2000
#define THREADS_PER_BLOCK 1024
#define BLOCKS MIN(32, (GRAPH_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
#define TILE_WIDTH 32
#define TILE_HEIGHT 32


#define EDGE_COST(graph, graph_size, a, b) graph[a * graph_size + b]
#define D(a, b) EDGE_COST(output, graph_size, a, b)

#define INF 0x1fffffff

__device__
int Min(int a, int b) { return a < b ? a : b; }

void generate_random_graph(int *output, int graph_size) {
  int i, j;

  srand(0xdadadada);

  for (i = 0; i < graph_size; i++) {
    for (j = 0; j < graph_size; j++) {
      if (i == j) {
        D(i, j) = 0;
      } else {
        int r;
        r = rand() % 40;
        if (r > 20) {
          r = INF;
        }

        D(i, j) = r;
      }
    }
  }
}

__global__ void floyd_warshall_gpu(int graph_size, int *output, int k) {
  int i, j;
  i = blockIdx.x * blockDim.x + threadIdx.x; 
  j = blockIdx.y * blockDim.y + threadIdx.y;
  while(i < graph_size && j < graph_size){
      if (D(i, k) + D(k, j) < D(i, j)) {
        D(i , j) = D(i, k) + D(k, j);
        __syncthreads();
    }
  }
}

void floyd_warshall_cpu(const int *graph, int graph_size, int *output) {
  int i, j, k;

  memcpy(output, graph, sizeof(int) * graph_size * graph_size);

  for (k = 0; k < graph_size; k++) {
    for (i = 0; i < graph_size; i++) {
      for (j = 0; j < graph_size; j++) {
        if (D(i, k) + D(k, j) < D(i, j)) {
          D(i , j) = D(i, k) + D(k, j);
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  #define TIMER_START() gettimeofday(&tv1, NULL)
  #define TIMER_STOP()                                                           \
    gettimeofday(&tv2, NULL);                                                    \
    timersub(&tv2, &tv1, &tv);                                                   \
    time_delta = (float)tv.tv_sec + tv.tv_usec / 1000000.0
  
    struct timeval tv1, tv2, tv;
    float time_delta;
  
    int *graph, *output_cpu, *host_output_gpu, *output_gpu;
    int size;
  
    size = sizeof(int) * GRAPH_SIZE * GRAPH_SIZE;
  
    graph = (int *)malloc(size);
    assert(graph);
  
    host_output_gpu = (int *)malloc(size);
    assert(host_output_gpu);
    memset(host_output_gpu, 0, size);

    output_cpu = (int *)malloc(size);
    assert(output_cpu);
    memset(output_cpu, 0, size);
  
    output_gpu = (int *)malloc(size);
    assert(output_gpu);

    generate_random_graph(graph, GRAPH_SIZE);
  
    fprintf(stderr, "running on cpu...\n");
    TIMER_START();
    floyd_warshall_cpu(graph, GRAPH_SIZE, output_cpu);
    TIMER_STOP();
    fprintf(stderr, "%f secs\n", time_delta);
    
    HANDLE_ERROR(cudaMalloc(&output_gpu, size));
    cudaMemcpy(output_gpu, graph, size, cudaMemcpyHostToDevice);

    fprintf(stderr, "running on gpu...\n");
    TIMER_START();
    for (int k = 0; k < GRAPH_SIZE; k++) {
      floyd_warshall_gpu<<<BLOCKS, THREADS_PER_BLOCK>>>(GRAPH_SIZE, output_gpu, k);
    }
    TIMER_STOP();
    
    cudaMemcpy(graph, output_gpu, size, cudaMemcpyDeviceToHost);

    fprintf(stderr, "%f secs\n", time_delta);

    if (memcmp(output_cpu, host_output_gpu, size) != 0) {
      fprintf(stderr, "FAIL!\n");
    }

    cudaFree(output_gpu);


    free(graph);
    free(output_cpu);
    free(host_output_gpu);
    
  
    return 0;
  }