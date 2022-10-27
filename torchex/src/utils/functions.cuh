#ifndef TORCHEX
#define TORCHEX
__device__ __forceinline__ int up_2n_d(int n);

__device__ __forceinline__ int up_2n_d(int n){
    if (n == 1) return 1;
    int temp = n - 1;
    temp |= temp >> 1;
    temp |= temp >> 2;
    temp |= temp >> 4;
    temp |= temp >> 8;
    temp |= temp >> 16;
    return temp + 1;
}

int up_2n(int n);
int up_2n(int n){
    if (n == 1) return 1;
    int temp = n - 1;
    temp |= temp >> 1;
    temp |= temp >> 2;
    temp |= temp >> 4;
    temp |= temp >> 8;
    temp |= temp >> 16;
    return temp + 1;
}


// A simple hash table
__device__ __forceinline__ int double_hash(int key, int probe_i, int table_size);
__device__ __forceinline__ int double_hash(int key, int probe_i, int table_size){
  // equivalent to (key +  probe_i * (key * 2 + 1)) % table_size, keep the one more mod op for better understanding.
  return (key % table_size +  probe_i * (key * 2 + 1)) % table_size;
}

__device__ void setvalue(const int key, const int value, int *table, const int table_size);
__device__ void setvalue(const int key, const int value, int *table, const int table_size){
  for (int i = 0; i < table_size; i++){
    int slot_idx = double_hash(key, i, table_size);
    int old_key = atomicCAS(&table[2 * slot_idx], -1, key); // even pos: key, odd pos: value
    if (old_key == -1){
      table[2 * slot_idx + 1] = value;
      return;
    }
  }
  printf("\n ********* This should not happen ********* \n");
  assert(false);
}

__device__ int getvalue(const int key, const int *table, const int table_size);
__device__ int getvalue(const int key, const int *table, const int table_size){
  for (int i = 0; i < table_size; i++){
    int slot_idx = double_hash(key, i, table_size);
    int slot_key = table[2 * slot_idx]; // even pos: key, odd pos: value
    if (slot_key == key){
      return table[2 * slot_idx + 1];
    }
  }
  return -1;
}

#endif