#include <iostream>
#include <fstream>

#include "include/benchmark.cuh"

void test1() {
  std::ofstream ofs;
  ofs.open("../result/test1.txt", std::ios::out);
  uint table_size = 1 << 25;
  auto max_evict_num = static_cast<uint>(4 * logf(static_cast<float>(table_size)));
  ofs << "Test.1 increased key num, fixed table size of 2^25" << std::endl;
  std::cout << "Test.1 increased key num, fixed table size of 2^25" << std::endl;

  printf("2 hash function\n");
  ofs << "--- 2 func ---" << std::endl;
  ofs << "key num (2^i) | time | MOPS" << std::endl;
  for (int i = 10; i <= 24; i++) {
    ofs << i << " ";
    printf("insert 2^%d keys", i);
    int insert_key_num = 1 << i;
    int lookup_key_num = 1 << 9;
    testAvgTime(1 << 25, insert_key_num, lookup_key_num, 0,
                max_evict_num, 2, insertion, false, ofs);
  }

  printf("\n----------------------------\n3 hash function\n");
  ofs << "--- 3 func ---" << std::endl;
  ofs << "key num (2^i) | time | MOPS" << std::endl;
  for (int i = 10; i <= 24; i++) {
    ofs << i << " ";
    printf("insert 2^%d keys", i);
    int insert_key_num = 1 << i;
    int lookup_key_num = 1 << 9;
    testAvgTime(1 << 25, insert_key_num, lookup_key_num, 0,
                max_evict_num, 3, insertion, false, ofs);
  }
  ofs.close();
}

void test2() {
  std::ofstream ofs;
  ofs.open("../result/test2.txt", std::ios::out);
  uint table_size = 1 << 25;
  auto max_evict_num = static_cast<uint>(4 * logf(static_cast<float>(table_size)));
  ofs << "Test.2 fixed table size of 2^25, look up key num 2^24" << std::endl;
  std::cout << "Test.2 fixed table size of 2^25, look up key num 2^24" << std::endl;
  printf("2 hash function\n");

  ofs << "--- 2 func ---" << std::endl;
  ofs << "(100 - i)% in table | time | MOPS" << std::endl;
  for (int i = 0; i <= 10; i++) {
    ofs << i << " ";
    printf("%d percent old keys", (100 - i * 10));
    testAvgTime(table_size, 1 << 24, 1 << 24, (float)i/10,
                max_evict_num, 2, lookup, false, ofs);
  }

  printf("\n------------------------------\n3 hash function\n");
  ofs << "--- 3 func ---" << std::endl;
  ofs << "(100 - i)% in table | time | MOPS" << std::endl;
  for (int i = 0; i <= 10; i++) {
    ofs << i << " ";
    printf("%d percent old keys", (100 - i * 10));
    testAvgTime(table_size, 1 << 24, 1 << 24, (float)i/10,
                max_evict_num, 3, lookup, false, ofs);
  }
  ofs.close();
}

void test3() {
  float table_size_factor[] = {1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0};
  std::ofstream ofs;
  ofs.open("../result/test3.txt", std::ios::out);
  uint table_size = 1 << 25;
  auto max_evict_num = static_cast<uint>(4 * logf(static_cast<float>(table_size)));
  ofs << "Test.3 fixed key num of 2^24, change table size" << std::endl;
  std::cout << "Test.3 fixed key num of 2^24, change table size" << std::endl;
  printf("2 hash function\n");

  ofs << "--- 2 func ---" << std::endl;
  ofs << "table size (n) | time | MOPS" << std::endl;
  for (int i = 0; i < 13; i++) {
    ofs << table_size_factor[i] << " ";
    printf("table size %.2f ", table_size_factor[i]);
    testAvgTime((uint)(table_size_factor[i]*(1 << 24)), 1 << 24, 1 << 10, 0,
                max_evict_num, 2, insertion, false, ofs);
  }

  printf("\n------------------------------\n3 hash function\n");
  ofs << "--- 3 func ---" << std::endl;
  ofs << "table size (n) | time | MOPS" << std::endl;
  for (int i = 0; i < 13; i++) {
    ofs << table_size_factor[i] << " ";
    printf("table size %.2f ", table_size_factor[i]);
    testAvgTime((uint)(table_size_factor[i]*(1 << 24)), 1 << 24, 1 << 10, 0,
                max_evict_num, 3, insertion, false, ofs);
  }
  ofs.close();
}

void test4() {
  const int bound_size = 16;
  uint evict_bound[bound_size] = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80};
  std::ofstream ofs;
  ofs.open("../result/test4.txt", std::ios::out);
  uint table_size = 1.4 * (1 << 24);
  ofs << "Test.4 fixed key num of 2^24, fixed table size, change evict bound." << std::endl;
  std::cout << "Test.4 fixed key num of 2^24, fixed table size, change evict bound." << std::endl;
  printf("2 hash function\n");

  ofs << "--- 2 func ---" << std::endl;
  ofs << "evict bound | time | MOPS" << std::endl;
  for (int i = 0; i < bound_size; i++) {
    auto max_evict_num = evict_bound[i];
    ofs << max_evict_num << " ";
    printf("evict bound %u ", max_evict_num);
    testAvgTime(table_size, 1 << 24, 1 << 10, 0,
                max_evict_num, 2, insertion, false, ofs);
  }

  printf("\n------------------------------\n3 hash function\n");
  ofs << "--- 3 func ---" << std::endl;
  ofs << "evict bound | time | MOPS" << std::endl;
  for (int i = 0; i < bound_size; i++) {
    auto max_evict_num = evict_bound[i];
    ofs << max_evict_num << " ";
    printf("evict bound %u ", max_evict_num);
    testAvgTime(table_size, 1 << 24, 1 << 10, 0,
                max_evict_num, 3, insertion, false, ofs);
  }
  ofs.close();
}

int main() {
  test1();
  test2();
  test3();
  test4();
  printf("all tests done.\n");
  return 0;
}
