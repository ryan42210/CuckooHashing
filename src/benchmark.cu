#include "include/benchmark.cuh"
#include "include/key_gen.h"

#include <unordered_set>
#include <iostream>

constexpr int TEST_NUM = 5;

void testAvgTime(uint hash_table_size,
                 int insert_key_num,
                 int lookup_key_num,
                 float new_key_frac,
                 uint max_evict_num,
                 uint func_num,
                 OpType type,
                 bool need_verify,
                 std::ofstream &ofs) {
  float total_insert_time = 0;
  float total_lookup_time = 0;
  for (int i = 0; i < TEST_NUM; i++) {
    KeyGenerator input;
    input.genInsertKeys(insert_key_num);
    input.genLookUpKeys(lookup_key_num, new_key_frac);
    HashTableConfig hash_table_config(hash_table_size, func_num, max_evict_num);

    // Allocate mem for hash table
    int *d_table;
    cudaMalloc(&d_table, sizeof(int) * func_num * hash_table_size);
    cudaMemset(d_table, 0, sizeof(int) * func_num * hash_table_size);

    bool insert_res = true;
    total_insert_time += insertTest(input, hash_table_config, d_table, insert_res);
    if (insert_res) {
      total_lookup_time += lookupTest(input, hash_table_config, d_table, need_verify);
    } else {
      ofs << "insert failed" << std::endl;
      cudaFree(d_table);
      return;
    }

    cudaFree(d_table);
  }
  total_insert_time /= TEST_NUM;
  total_lookup_time /= TEST_NUM;
  printf(" | Insertion time: %6.3f ms | Mops= %6.3f | Look up time %6.3f ms | Mops= %6.3f \n",
         total_insert_time,
         (float)insert_key_num / total_insert_time / 1e3,
         total_lookup_time,
         (float)lookup_key_num / total_lookup_time / 1e3);
  if (type == OpType::insertion) {
    ofs << total_insert_time << " " << (float)insert_key_num / total_insert_time / 1e3 << std::endl;
  } else {
    ofs << total_lookup_time << " " << (float)lookup_key_num / total_lookup_time / 1e3 << std::endl;
  }
}

float insertTest(KeyGenerator &input, HashTableConfig &hash_table_config, int *d_table, bool &succeed) {
  const int *h_insert_keys = input.getInsertKeyPtr();
  int key_num = input.getInsertKeyNum();
  // Generate random insertion keys on host.
  int *d_insert_keys;
  cudaMalloc(&d_insert_keys, sizeof(int) * key_num);
  cudaMemcpy(d_insert_keys,
             h_insert_keys,
             sizeof(int) * key_num,
             cudaMemcpyHostToDevice);

  int *d_result;
  int h_result = 0;
  cudaMalloc(&d_result, sizeof(int));
  cudaMemset(d_result, 0, sizeof(int));

  float dt_ms;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventRecord(start);

  succeed = insertKeys(d_table, hash_table_config, d_insert_keys, key_num, d_result, &h_result);

  cudaEventCreate(&stop);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&dt_ms, start, stop);

  cudaFree(d_result);
  cudaFree(d_insert_keys);
  if (!succeed) {
    std::cout << "Inserting " << input.getInsertKeyNum() << " keys into size " << hash_table_config._table_size
              << " table. Failed after " << dt_ms << " ms." << std::endl;
  }
  return dt_ms;
}

float lookupTest(KeyGenerator &input, HashTableConfig &hash_table_config, int *d_table, bool need_verify) {
  const int *h_lookup_keys = input.getLookUpKeyPtr();
  int key_num = input.getLookUpKeyNum();
  int *d_lookup_keys;
  cudaMalloc(&d_lookup_keys, sizeof(int) * key_num);
  cudaMemcpy(d_lookup_keys,
             h_lookup_keys,
             sizeof(int) * key_num,
             cudaMemcpyHostToDevice);

  bool *d_result;
  cudaMalloc(&d_result, sizeof(bool) * key_num);
  cudaMemset(d_result, 0, sizeof(bool) * key_num);
  // create hash table and insert keys into it.
  float dt_ms;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventRecord(start);

  lookUpKeys(d_table, hash_table_config, d_lookup_keys, key_num, d_result);

  cudaEventCreate(&stop);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&dt_ms, start, stop);

  // if (!verifyResult(d_result, input)) {
  //   printHashTable(d_table, hash_table_config._table_size, hash_table_config._func_num);
  // }
  if (need_verify && !verifyResult(d_result, input)) {
    exit(-1);
  }

  cudaFree(d_lookup_keys);
  cudaFree(d_result);
  return dt_ms;
}

bool verifyResult(bool *d_result, KeyGenerator &input) {
  // Copy result back to host
  int lookup_key_num = input.getLookUpKeyNum();
  bool *h_result = new bool[lookup_key_num];
  cudaMemcpy(h_result, d_result, sizeof(bool) * lookup_key_num, cudaMemcpyDeviceToHost);
  std::unordered_set<int> hash_ref;
  // insert keys into reference
  for (int i = 0; i < input.getInsertKeyNum(); i++) {
    hash_ref.insert(input.getInsertKeyPtr()[i]);
  }

  // check lookup keys in reference and result
  int false_in_ref = 0;
  int true_in_ref = 0;
  for (int j = 0; j < lookup_key_num; j++) {
    if (hash_ref.find(input.getLookUpKeyPtr()[j]) == hash_ref.end()) {
      if (h_result[j])
        false_in_ref++;
    } else {
      if (!h_result[j])
        true_in_ref++;
    }
  }
  delete[] h_result;
  if (true_in_ref == 0 && false_in_ref == 0) {
    return true;
  } else {
    std::cout << "Verification not pass..." << std::endl;
    std::cout << false_in_ref << " keys in gpu table but not in reference." << std::endl;
    std::cout << true_in_ref << " keys not in gpu table but in reference." << std::endl;
    return false;
  }
}
