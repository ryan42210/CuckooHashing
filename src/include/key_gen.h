#ifndef KEY_GEN_H
#define KEY_GEN_H

#include <algorithm>
#include <random>
#include <iostream>
#include <vector>

class KeyGenerator {
 public:
  KeyGenerator() {
    initUsedKey();
  }
  void initUsedKey() {
    if (key_used.empty()) {
      key_used.resize(UINT32_MAX, false);
    }
  }
  void genInsertKeys(int n) {
    insert_keys.resize(n);
    std::mt19937 rng(std::random_device{}());
    std::generate(insert_keys.begin(), insert_keys.end(), [&](){
      uint32_t res = rng();
      while (key_used[res] || res == 0) {
        res = rng();
      }
      key_used[res] = true;
      return static_cast<int >(res);
    });
  }
  // Generate N keys from S, and generate M more random keys
  void genLookUpKeys(int num, float f) {
    int n = num * (1-f);
    int m = num - n;
    if (n > insert_keys.size()) {
      std::cerr << "Failed! Not enough inserted keys to select." << std::endl;
      return;
    }

    if (!lookup_keys.empty()) lookup_keys.clear();
    lookup_keys.resize(m);
    std::generate_n(lookup_keys.begin(), m, std::mt19937(std::random_device{}()));
    lookup_keys.insert(lookup_keys.end(), insert_keys.begin(), insert_keys.begin() + n);
    std::shuffle(lookup_keys.begin(), lookup_keys.end(), std::mt19937(std::random_device{}()));
  }

  [[nodiscard]] const int *getInsertKeyPtr() const { return insert_keys.data(); }
  [[nodiscard]] const int *getLookUpKeyPtr() const {  return lookup_keys.data(); }
  [[nodiscard]] int getInsertKeyNum() const{ return static_cast<int>(insert_keys.size()); }
  [[nodiscard]] int getLookUpKeyNum() const { return static_cast<int>(lookup_keys.size()); }
  void printInsertKeys() {
    std::cout << "insert keys are" << std::endl;
    for (auto i : insert_keys) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;
  }
  void printLookUpKeys() {
    std::cout << "look up keys are" << std::endl;
    for (auto i : lookup_keys) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;
  }
 private:
  std::vector<bool> key_used;
  std::vector<int > insert_keys;
  std::vector<int > lookup_keys;
};

#endif
