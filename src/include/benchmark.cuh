#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "include/hash_table.cuh"
#include "include/key_gen.h"
#include "key_gen.h"
#include "hash_table.cuh"

#include <fstream>

enum OpType { insertion, lookup };

void testAvgTime(uint hash_table_size,
                 int insert_key_num,
                 int lookup_key_num,
                 float new_key_frac,
                 uint max_evict_num,
                 uint func_num,
                 OpType type,
                 bool need_verify,
                 std::ofstream &ofs);

float insertTest(KeyGenerator &input, HashTableConfig &hash_table_config, int *d_table, bool &succeed);
float lookupTest(KeyGenerator &input, HashTableConfig &hash_table_config, int *d_table, bool need_verify);


bool verifyResult(bool *d_result, KeyGenerator &input);

#endif