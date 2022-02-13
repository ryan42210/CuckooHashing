#include "include/hash_table.cuh"
#include <iostream>
constexpr uint thread_per_block = 1024;
const int max_rebuild_num = 16;


/* --------------------------- helper function --------------------------- */
__device__ __forceinline__ uint id(uint idx, uint func_id, uint table_size) {
  return idx + func_id * table_size;
}

__device__ inline uint doHash(int key, uint func_id, uint table_size, uint seed) {
  auto res = static_cast<uint>(key);
  res = res << seed | res >> (32-seed);
  switch (func_id) {
    case 0: {
      res ^= res >> 17;
      res *= 0xed5ad4bb;
      res ^= res >> 11;
      res *= 0xac4c1b51;
      res ^= res >> 15;
      res *= 0x31848bab;
      res ^= res >> 14;
      return res % table_size;
    }
    case 1: {
      res ^= res >> 16;
      res *= 0xaeccedab;
      res ^= res >> 14;
      res *= 0xac613e37;
      res ^= res >> 16;
      res *= 0x19c89935;
      res ^= res >> 17;
      return res % table_size;
    }
    case 2: {
      res ^= res >> 16;
      res *= 0x236f7153;
      res ^= res >> 12;
      res *= 0x33cd8663;
      res ^= res >> 15;
      res *= 0x3e06b66b;
      res ^= res >> 16;
      return res % table_size;
    }
    default: {
      return res * (func_id * 2654435761) % table_size;
    }
  }
}

/* --------------------------- insertion --------------------------- */
__device__ inline bool insert_one_key(int *d_table, uint table_size, uint func_num, uint seed, int key, uint max_evict_num) {
  int victim_key;
  uint func_id = 0;
  for (uint evict_cnt = 0; evict_cnt < max_evict_num; evict_cnt++) {
    uint idx = doHash(key, func_id, table_size, seed);
    victim_key = atomicExch(&d_table[id(idx, func_id, table_size)], key);
    if (victim_key == 0) {
      return true;
    }
    key = victim_key;
    func_id = (func_id + 1) % func_num;
    evict_cnt++;
  }
  return false;
}

__global__ void insertKernel(int *d_table,
                             uint table_size,
                             uint func_num,
                             uint seed,
                             const int *d_insert_key,
                             uint insert_key_num,
                             uint max_evict_num,
                             int* d_result) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= insert_key_num) return;
  int key = d_insert_key[tid];
  if (!insert_one_key(d_table, table_size, func_num, seed, key, max_evict_num)) {
    // *d_result = 1;
    atomicCAS(d_result, 0, 1);
  }
}

bool insertKeys(int *d_table,
                HashTableConfig &config,
                const int *d_insert_key,
                uint insert_key_num,
                int *d_result,
                int *h_result) {
  int cnt = 0;
  uint block_num = (insert_key_num-1) / thread_per_block + 1;
  while (cnt < max_rebuild_num) {
    insertKernel<<<block_num, thread_per_block>>>(d_table,
                                                  config._table_size,
                                                  config._func_num,
                                                  config._seed,
                                                  d_insert_key,
                                                  insert_key_num,
                                                  config._max_evict_num,
                                                  d_result);
    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (*h_result == 0) {
      return true;
    }
    cudaMemset(d_table, 0, sizeof(int) * config._table_size * config._func_num);
    config._seed = ++cnt;
  }
  return false;
}

/* -------------------------------- lookup ------------------------------- */
__global__ void lookupKernel(const int *d_table,
                             uint table_size,
                             uint func_num,
                             uint seed,
                             const int *d_lookup_key,
                             uint lookup_key_num,
                             bool *d_result) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= lookup_key_num) return;
  int key = d_lookup_key[tid];
  for (uint func_id = 0; func_id < func_num; func_id++) {
    uint idx = doHash(key, func_id, table_size, seed);
    if (d_table[id(idx, func_id, table_size)] == key) {
      d_result[tid] = true;
      return;
    }
  }
  d_result[tid] = false;
}

void lookUpKeys(const int *d_table,
                HashTableConfig &config,
                const int *d_lookup_key,
                uint lookup_key_num,
                bool *d_result) {
  uint block_num = (lookup_key_num - 1) / thread_per_block + 1;
  lookupKernel<<<block_num, thread_per_block>>>(d_table,
                                                config._table_size,
                                                config._func_num,
                                                config._seed,
                                                d_lookup_key,
                                                lookup_key_num,
                                                d_result);
}


void printHashTable(int *d_table, uint table_size, uint func_num) {
  printf("\nPrinting table content...\n");
  printf("table size is %d\n", table_size);
  auto *h_table = new int[table_size * func_num];
  cudaMemcpy(h_table, d_table, sizeof(int) * table_size * func_num, cudaMemcpyDeviceToHost);
  for (uint i = 0; i < table_size; i++) {
    for (uint j = 0; j < func_num; j++) {
      printf("%d, ", h_table[i + j * table_size]);
    }
    printf("\n");
  }
  delete[] h_table;
}

