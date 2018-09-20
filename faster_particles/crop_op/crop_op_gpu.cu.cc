#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "crop_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h" // for CUDA_1D_KERNEL_LOOP
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor" // FIXME necessary?

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
// Input image is size NxNxNxC
template <typename T>
__global__ void CropCudaKernel(
  const T* image_ptr,
  const int* crop_centers_ptr,
  int& image_size,
  int& channels,
  int& crop_size,
  int& num_crops,
  T* crops_ptr
) {
  const int crop_id = blockIdx.x;
  const int center_x = crop_centers_ptr[crop_id];
  const int center_y = crop_centers_ptr[num_crops + crop_id];
  const int center_z = crop_centers_ptr[num_crops * 2 + crop_id];


  for (int idx = threadIdx.x; idx < crop_size*crop_size*crop_size*channels; idx += blockDim.x) {
    // Coordinates inside the crop (0 <= coords < crop_size)
    const int c = idx % channels;
    idx /= channels;
    const int x = idx % crop_size;
    idx /= crop_size;
    const int y = idx % crop_size;
    const int z = idx / crop_size;

    // Corresponding coordinates in original image
    int new_x = x + (center_x - crop_size / 2);
    int new_y = y + (center_y - crop_size / 2);
    int new_z = z + (center_z - crop_size / 2);
    int img_idx = c + channels * (new_x + crop_size * (new_y + crop_size * new_z ));

    if ((img_idx >= image_size * image_size * image_size * channels) || (img_idx < 0)) continue;

    crops_ptr[c + channels * (x + crop_size * (y + crop_size * (z + num_crops * crop_id)))] = image_ptr[c + channels * (new_x + crop_size * (new_y + crop_size * new_z ))];
  }
  // threadId.x = c + channels * (x + crop_size * (y + crop_size * z))

  /*CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = c + channels * (x + crop_size * (y + crop_size * (z + crop_size * crop_id)))
    int idx = out_idx;
    const int c = idx % channels;
    idx /= channels;
    const int x = idx % crop_size;
    idx /= crop_size;
    const int y = idx % crop_size;
    idx /= crop_size;
    const int z = idx % crop_size;
    const int crop_id = idx / crop_size;

    const float center_x = crop_centers_ptr[crop_idx * 3];
    const float center_y = crop_centers_ptr[crop_idx * 3 + 1];
    const float center_z = crop_centers_ptr[crop_idx * 3 + 2];

    crops_ptr[out_idx] = static_cast<float>(
      image_ptr[c + channels * (center_x + image_size * (center_y + image_size * (center_z * image_size + ??)))]
    );

  }*/
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void CropFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d,
    const T* image_ptr,
    const int* crop_centers_ptr,
    int image_size,
    int channels,
    int crop_size,
    int num_crops,
    T* crops_ptr
  ) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = num_crops;
  int thread_per_block = 128;
  CropCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(
        image_ptr,
        crop_centers_ptr,
        image_size,
        channels,
        crop_size,
        num_crops,
        crops_ptr
      );
  //return d.ok();
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct CropFunctor<GPUDevice, float>;
template struct CropFunctor<GPUDevice, int32>;

#endif // GOOGLE_CUDA
