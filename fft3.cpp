#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#include <rocfft/rocfft.h>
#include "stdlib.h"
#include <fstream>
#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/complex.h>

rocfft_plan_description desc3d3d_ip_f, desc3d3d_ip_b, desc3d2d_ip_f, desc3d2d_ip_b, desc3d1d_ip_f, desc3d1d_ip_b;
rocfft_plan plan3d3d_ip_f, plan3d3d_ip_b, plan3d2d_ip_f, plan3d2d_ip_b, plan3d1d_ip_f, plan3d1d_ip_b;
    
size_t rocfft_buffer_size;
void *rocfft_buffer_d;
rocfft_execution_info rocfft_info;

void export_data(thrust::complex<double> *z, std::string fn, int N) {
    std::ofstream fl;
    fl.open(fn, std::ios::out | std::ios::binary);
    for (int i=0; i < N; ++i) {
        double rp = z[i].real();
        double ip = z[i].imag();
        fl.write(reinterpret_cast<char*>(&rp), sizeof(double));
        fl.write(reinterpret_cast<char*>(&ip), sizeof(double));
    }
    fl.close();
}

int main() {
    // data init
    int Nt = 1<<8;//4;
    int Ny = 1<<8;
    int Nx = 1<<12;//12;
    int N = Nt * Nx * Ny;
    
    std::cout << Nt * Nx * Ny << " values" << std::endl;
    std::cout << sizeof(double2) * Nt * Nx * Ny / pow(1024, 3) << " GB" << std::endl;
    
    // data host
    thrust::complex<double> *z21d = new thrust::complex<double>[N];
    thrust::complex<double> *z3d = new thrust::complex<double>[N];
    
    for (int tt=0; tt<Nt; ++tt) {
        for (int yy=0; yy<Ny; ++yy) {
            for (int xx=0; xx<Nx; ++xx) {
                z21d[Ny*Nx*tt + Nx*yy + xx] = 0.;
                z3d[Ny*Nx*tt + Nx*yy + xx] = 0.;
            }
        }
    }
    
    int fac = 8;
    for (int tt=0; tt<Nt/fac; ++tt) {
        for (int yy=0; yy<Ny/fac; ++yy) {
            for (int xx=0; xx<Nx/fac; ++xx) {
                z21d[Ny*Nx*tt + Nx*yy + xx] = 1.;
                z3d[Ny*Nx*tt + Nx*yy + xx] = 1.;
            }
        }
    }
    
    // export initial data to file
    export_data(z3d, "/tmp/Etyx.bin", N);

    // ROCFFT init
    rocfft_setup();
    
    desc3d3d_ip_f = nullptr;
    desc3d3d_ip_b = nullptr;
    desc3d2d_ip_f = nullptr;
    desc3d2d_ip_b = nullptr;
    desc3d1d_ip_f = nullptr;
    desc3d1d_ip_b = nullptr;
    
    const size_t stride_t = Nx*Ny;
    const size_t stride_y = Nx;
    
    rocfft_plan_description_create(&desc3d3d_ip_f);
    rocfft_plan_description_create(&desc3d3d_ip_b);
    rocfft_plan_description_create(&desc3d2d_ip_f);
    rocfft_plan_description_create(&desc3d2d_ip_b);
    rocfft_plan_description_create(&desc3d1d_ip_f);
    rocfft_plan_description_create(&desc3d1d_ip_b);
    
    rocfft_plan_description_set_scale_factor(desc3d3d_ip_b, 1./static_cast<double>(N));
    rocfft_plan_description_set_scale_factor(desc3d2d_ip_b, 1./static_cast<double>(Nx*Ny));
    rocfft_plan_description_set_scale_factor(desc3d1d_ip_b, 1./static_cast<double>(Nt));
    
    rocfft_plan_description_set_data_layout(desc3d1d_ip_f,
                                            rocfft_array_type_complex_interleaved,
                                            rocfft_array_type_complex_interleaved,
                                            0, 0,
                                            1, &stride_t, 1,
                                            1, &stride_t, 1);
    rocfft_plan_description_set_data_layout(desc3d1d_ip_b,
                                            rocfft_array_type_complex_interleaved,
                                            rocfft_array_type_complex_interleaved,
                                            0, 0,
                                            1, &stride_t, 1,
                                            1, &stride_t, 1);

    plan3d3d_ip_f = nullptr;
    plan3d3d_ip_b = nullptr;
    plan3d2d_ip_f = nullptr;
    plan3d2d_ip_b = nullptr;
    plan3d1d_ip_f = nullptr;
    plan3d1d_ip_b = nullptr;

    const size_t FFTlen_txy[3] = {(unsigned int)Nx, (unsigned int)Ny, (unsigned int)Nt};
    const size_t FFTlen_xy[2] = {(unsigned int)Nx, (unsigned int)Ny};
    const size_t FFTlen_t = Nt;

    rocfft_plan_create(&plan3d3d_ip_f,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_double,
                       3, FFTlen_txy, 1, desc3d3d_ip_f);
    rocfft_plan_create(&plan3d3d_ip_b,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_inverse,
                       rocfft_precision_double,
                       3, FFTlen_txy, 1, desc3d3d_ip_b);
    rocfft_plan_create(&plan3d2d_ip_f,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_double,
                       2, FFTlen_xy, Nt, desc3d2d_ip_f);
    rocfft_plan_create(&plan3d2d_ip_b,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_inverse,
                       rocfft_precision_double,
                       2, FFTlen_xy, Nt, desc3d2d_ip_b);
    rocfft_plan_create(&plan3d1d_ip_f,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_double,
                       1, &FFTlen_t, stride_t, desc3d1d_ip_f);
    rocfft_plan_create(&plan3d1d_ip_b,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_inverse,
                       rocfft_precision_double,
                       1, &FFTlen_t, stride_t, desc3d1d_ip_b);

    rocfft_buffer_size = 0;
    size_t work_buffer_size = 0;
    
    rocfft_plan_get_work_buffer_size(plan3d3d_ip_f, &work_buffer_size);
    rocfft_buffer_size = std::max((int)rocfft_buffer_size, (int)work_buffer_size);
    rocfft_plan_get_work_buffer_size(plan3d3d_ip_b, &work_buffer_size);
    rocfft_buffer_size = std::max((int)rocfft_buffer_size, (int)work_buffer_size);
    rocfft_plan_get_work_buffer_size(plan3d2d_ip_f, &work_buffer_size);
    rocfft_buffer_size = std::max((int)rocfft_buffer_size, (int)work_buffer_size);
    rocfft_plan_get_work_buffer_size(plan3d2d_ip_b, &work_buffer_size);
    rocfft_buffer_size = std::max((int)rocfft_buffer_size, (int)work_buffer_size);
    rocfft_plan_get_work_buffer_size(plan3d1d_ip_f, &work_buffer_size);
    rocfft_buffer_size = std::max((int)rocfft_buffer_size, (int)work_buffer_size);
    rocfft_plan_get_work_buffer_size(plan3d1d_ip_b, &work_buffer_size);
    rocfft_buffer_size = std::max((int)rocfft_buffer_size, (int)work_buffer_size);
    
    rocfft_info = nullptr;
    if(rocfft_buffer_size) {
        rocfft_execution_info_create(&rocfft_info);
        HIP_ASSERT(hipMalloc(&rocfft_buffer_d, rocfft_buffer_size));
        rocfft_execution_info_set_work_buffer(rocfft_info, rocfft_buffer_d, rocfft_buffer_size);
    }
    
    
    // data device
    double2 *z_d;
    HIP_ASSERT(hipMalloc(reinterpret_cast<void **>(&z_d), sizeof(double2)*N));
    HIP_ASSERT(hipMemcpy(z_d, z3d, N*sizeof(double2), hipMemcpyHostToDevice));
    HIP_ASSERT(hipDeviceSynchronize());
    
    // make 3d fft of 3d data
    rocfft_execute(plan3d3d_ip_f, (void**) &z_d, nullptr, rocfft_info);
    HIP_ASSERT(hipDeviceSynchronize());
    
    // copy 3d-FFTed data and save
    HIP_ASSERT(hipMemcpy(z3d, z_d, N*sizeof(double2), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipDeviceSynchronize());
    export_data(z3d, "/tmp/Efkykx3d.bin", N);
    
    // copy fresh init data again from host for 2d-1d fft
    HIP_ASSERT(hipMemcpy(z_d, z21d, N*sizeof(double2), hipMemcpyHostToDevice));
    HIP_ASSERT(hipDeviceSynchronize());
    
    // make 2d fft in x/y and 1d fft in t
    rocfft_execute(plan3d2d_ip_f, (void**) &z_d, nullptr, rocfft_info);
    HIP_ASSERT(hipDeviceSynchronize());
    rocfft_execute(plan3d1d_ip_f, (void**) &z_d, nullptr, rocfft_info);
    HIP_ASSERT(hipDeviceSynchronize());
    
    // copy data back to host and save
    HIP_ASSERT(hipMemcpy(z21d, z_d, N*sizeof(double2), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipDeviceSynchronize());
    export_data(z21d, "/tmp/Efkykx21d.bin", N);
    
    free(z3d);
    free(z21d);
    HIP_ASSERT(hipFree(z_d));
    return 0;
}
