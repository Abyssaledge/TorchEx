#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <vector>
#include <tuple>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void build_hash_table_launcher(
    const int *coors_ptr,
    const int *values_ptr,
    int *table_ptr,
    int table_size,
    int N
);

void probe_hash_table_launcher(
    const int *coors_ptr,
    const int *table_ptr,
    int *out_values_ptr,
    int table_size,
    int N
);

__inline__ int up_2n(int n){
    if (n == 1) return 1;
    int temp = n - 1;
    temp |= temp >> 1;
    temp |= temp >> 2;
    temp |= temp >> 4;
    temp |= temp >> 8;
    temp |= temp >> 16;
    return temp + 1;
}


torch::Tensor build_hash_table(
  torch::Tensor coors,
  torch::Tensor values
);

torch::Tensor build_hash_table(
  torch::Tensor coors,
  torch::Tensor values
) {

  CHECK_INPUT(coors);
  int N = coors.size(0);
  assert (coors.ndimension() == 1);
  assert (N == values.size(0));

  auto int_opts = coors.options().dtype(torch::kInt32);
  int table_size = up_2n(N);

  torch::Tensor table = torch::full({table_size, 2}, -1, int_opts); // the first channel is flattened coors, the second channel is the mapped 1D index


  const int *coors_ptr = coors.data_ptr<int>();
  const int *values_ptr = values.data_ptr<int>();
  int *table_ptr = table.data_ptr<int>();

  build_hash_table_launcher(
      coors_ptr,
      values_ptr,
      table_ptr,
      table_size,
      N
  );
  return table;

}

torch::Tensor probe_hash_table(
  torch::Tensor coors,
  torch::Tensor table
);

torch::Tensor probe_hash_table(
  torch::Tensor coors,
  torch::Tensor table
) {

  CHECK_INPUT(coors);
  CHECK_INPUT(table);
  int N = coors.size(0);
  assert (coors.ndimension() == 1);
  torch::Tensor out_values = torch::full_like(coors, -1); // the first channel is flattened coors, the second channel is the mapped 1D index

  int table_size = table.size(0);

  const int *coors_ptr = coors.data_ptr<int>();
  const int *table_ptr = table.data_ptr<int>();
  int *out_values_ptr = out_values.data_ptr<int>();

  probe_hash_table_launcher(
      coors_ptr,
      table_ptr,
      out_values_ptr,
      table_size,
      N
  );

  return out_values;

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_table", &build_hash_table, "build hash table (coors -> 1D compact index) given coors ");
  m.def("probe_table", &probe_hash_table, "query the hash table acoording to the given coors");
}
