#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// -----------------------------------
// Kernel
// -----------------------------------
template <typename T>
__global__ void scale_kernel(const T* __restrict__ x, T* __restrict__ y, float a, long n) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = static_cast<T>(a * static_cast<float>(x[i]));
  }
}

static void launch_scale_kernel(const at::Tensor& x, const at::Tensor& y, double a) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(y.is_cuda(), "y must be CUDA");
  TORCH_CHECK(x.scalar_type() == y.scalar_type(), "dtype mismatch");
  TORCH_CHECK(x.numel() == y.numel(), "size mismatch");
  auto n = x.numel();
  constexpr int BS = 256;
  int grid = (n + BS - 1) / BS;
  auto stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == at::kFloat) {
    scale_kernel<float><<<grid, BS, 0, stream>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), static_cast<float>(a), n);
  } else if (x.scalar_type() == at::kHalf) {
    scale_kernel<at::Half><<<grid, BS, 0, stream>>>(
      x.data_ptr<at::Half>(), y.data_ptr<at::Half>(), static_cast<float>(a), n);
  } else if (x.scalar_type() == at::kBFloat16) {
    scale_kernel<at::BFloat16><<<grid, BS, 0, stream>>>(
      x.data_ptr<at::BFloat16>(), y.data_ptr<at::BFloat16>(), static_cast<float>(a), n);
  } else {
    TORCH_CHECK(false, "unsupported dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -----------------------------------
// Dispatcher-backed eager impl (allocates output)
//   my_ns::scale(Tensor x, float a) -> Tensor
// -----------------------------------
at::Tensor scale_cuda_impl(const at::Tensor& x, double a) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  auto y = at::empty_like(x);
  launch_scale_kernel(x, y, a);
  return y;
}

// Meta (FakeTensor) impl for compile-time shape/dtype propagation
at::Tensor scale_meta_impl(const at::Tensor& x, double a) {
  (void)a;
  return at::empty_like(x, x.options().device(at::kMeta));
}

// -----------------------------------
// RAW launcher (not a dispatcher op!)
//   def scale_cuda_raw_(x: Tensor, y: Tensor, a: float) -> None
// Writes into y, no allocation, uses current stream. This is what Inductor will call.
// -----------------------------------
void scale_cuda_raw_(const at::Tensor& x, const at::Tensor& y, double a) {
  launch_scale_kernel(x, y, a);
}

// -----------------------------------
// Registration
// -----------------------------------
TORCH_LIBRARY(my_ns, m) {
  m.def("scale(Tensor x, float a) -> Tensor");
}
TORCH_LIBRARY_IMPL(my_ns, CUDA, m) {
  m.impl("scale", scale_cuda_impl);
}
TORCH_LIBRARY_IMPL(my_ns, Meta, m) {
  m.impl("scale", scale_meta_impl);
}

// pybind: expose the raw launcher WITHOUT going through dispatcher
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scale_cuda_raw_", &scale_cuda_raw_, "raw scale launcher (x,y,a)");
}