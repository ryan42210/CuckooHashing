#ifndef HASH_TABLE_H
#define HASH_TABLE_H

using uint = unsigned int;

struct HashTableConfig {
  HashTableConfig(uint table_size, uint func_num, uint max_evict_num)
  : _table_size(table_size),
  _func_num(func_num),
  _max_evict_num(max_evict_num),
  _seed(0) {};
  const uint _table_size;
  const uint _func_num;
  const uint _max_evict_num;
  uint _seed;
};

bool insertKeys(int *d_table, HashTableConfig &config, const int *d_insert_key, uint insert_key_num, int *d_result, int *h_result);
void lookUpKeys(const int *d_table, HashTableConfig &config, const int *d_lookup_key, uint lookup_key_num, bool *d_result);

void printHashTable(int *d_table, uint table_size, uint func_num);
#endif