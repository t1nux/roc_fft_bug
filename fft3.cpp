#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#include <rocfft/rocfft.h>
#include "stdlib.h"
#include <fstream>
#include <iostream>
#include <hip/hip_runtime.h>
#include <complex>
#include <fftw3.h>

// fftw
fftw_plan fftw_p;

// rocfft
rocfft_plan_description desc3d3d_ip_f, desc3d3d_ip_b, desc3d2d_ip_f, desc3d2d_ip_b, desc3d1d_ip_f, desc3d1d_ip_b;
rocfft_plan plan3d3d_ip_f, plan3d3d_ip_b, plan3d2d_ip_f, plan3d2d_ip_b, plan3d1d_ip_f, plan3d1d_ip_b;

size_t rocfft_buffer_size;
void *rocfft_buffer_d;
rocfft_execution_info rocfft_info;

// export function
void export_data(std::complex<double> *z, std::string fn, int N) {
    std::ofstream fl;
    fl.open(fn, std::ios::out | std::ios::binary);
    for (int i=0; i < N; ++i) {
        double rp = z[i].real();
        double ip = z[i].imag();
        fl.write(reinterpret_cast<char*>(&rp), sizeof(double));
        fl.write(reinterpret_cast<char*>(&ip), sizeof(double));
    }
    fl.close();
};

// norms added by evetsso: https://github.com/evetsso/roc_fft_bug
struct VectorNorms
{
    double l_2 = 0.0, l_inf = 0.0;
};

VectorNorms norm(const std::complex<double>* buf, int N)
{
  double linf = 0.0;
  double l2   = 0.0;
  for(int i = 0 ; i < N; ++i)
    {
      const double rval = std::abs(buf[i].real());
      linf = std::max(rval, linf);
      l2 += rval * rval;

      const double ival = std::abs(buf[i].imag());
      linf = std::max(ival, linf);
      l2 += ival * ival;
    }
  return {.l_2 = sqrt(l2), .l_inf = linf};
}

VectorNorms distance(const std::complex<double>* buf1, const std::complex<double>* buf2, int N)
{
  double linf = 0.0;
  double l2   = 0.0;
  for(int i = 0 ; i < N; ++i)
    {
      const double rdiff = std::abs(buf1[i].real() - buf2[i].real());
      linf = std::max(rdiff, linf);
      l2 += rdiff * rdiff;

      const double idiff = std::abs(buf1[i].imag() - buf2[i].imag());
      linf = std::max(idiff, linf);
      l2 += idiff * idiff;
    }
  return {.l_2 = sqrt(l2), .l_inf = linf};
}

int main() {
    // data init
    int Nt = 1<<8;//4;
    int Ny = 1<<12;
    int Nx = 1<<8;//12;
    int N = Nt * Nx * Ny;

    std::cout << Nt * Nx * Ny << " values" << std::endl;
    std::cout << "Nt = " << Nt << ", Ny = " << Ny << ", Nx = " << Nx << std::endl;
    std::cout << sizeof(double2) * Nt * Nx * Ny / pow(1024, 3) << " GB" << std::endl;

    // data host
    std::complex<double> *z21d = new std::complex<double>[N];
    std::complex<double> *z3d = new std::complex<double>[N];
    std::complex<double> *z3d_fftw = new std::complex<double>[N];

    for (int tt=0; tt<Nt; ++tt) {
        for (int yy=0; yy<Ny; ++yy) {
            for (int xx=0; xx<Nx; ++xx) {
                z21d[Ny*Nx*tt + Nx*yy + xx] = 0.;
                z3d[Ny*Nx*tt + Nx*yy + xx] = 0.;
                z3d_fftw[Ny*Nx*tt + Nx*yy + xx] = 0.;
            }
        }
    }

    int fac = 8;
    for (int tt=0; tt<Nt/fac; ++tt) {
        for (int yy=0; yy<Ny/fac; ++yy) {
            for (int xx=0; xx<Nx/fac; ++xx) {
                z21d[Ny*Nx*tt + Nx*yy + xx] = 1.;
                z3d[Ny*Nx*tt + Nx*yy + xx] = 1.;
                z3d_fftw[Ny*Nx*tt + Nx*yy + xx] = 1.;
            }
        }
    }

    auto input_norm = norm(z3d, N);

    // FFTW
    int rank[3] = {Nt, Ny, Nx};
    // this is in-place
    fftw_p = fftw_plan_dft(3, rank,
                           reinterpret_cast<fftw_complex*>(z3d_fftw), reinterpret_cast<fftw_complex*>(z3d_fftw),
                           FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fftw_p);


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

    // copy 3d-FFTed data
    HIP_ASSERT(hipMemcpy(z3d, z_d, N*sizeof(double2), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipDeviceSynchronize());

    // copy fresh init data again from host for 2d-1d fft
    HIP_ASSERT(hipMemcpy(z_d, z21d, N*sizeof(double2), hipMemcpyHostToDevice));
    HIP_ASSERT(hipDeviceSynchronize());

    // make 2d fft in x/y and 1d fft in t
    rocfft_execute(plan3d2d_ip_f, (void**) &z_d, nullptr, rocfft_info);
    HIP_ASSERT(hipDeviceSynchronize());
    rocfft_execute(plan3d1d_ip_f, (void**) &z_d, nullptr, rocfft_info);
    HIP_ASSERT(hipDeviceSynchronize());

    // copy data back to host
    HIP_ASSERT(hipMemcpy(z21d, z_d, N*sizeof(double2), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipDeviceSynchronize());

    // compare results
    auto diff = distance(z3d_fftw, z3d, N);

    printf("\n--- 3D ---\n");
    printf("l2 difference: %e\nl-inf difference: %e\n\n",
           diff.l_2 / input_norm.l_2 * sqrt(log2(N)), diff.l_inf / input_norm.l_inf / log(N));

    diff = distance(z3d_fftw, z21d, N);
    printf("--- 2D+1D ---\n");
    printf("l2 difference: %e\nl-inf difference: %e\n",
           diff.l_2 / input_norm.l_2 * sqrt(log2(N)), diff.l_inf / input_norm.l_inf / log(N));

    export_data(z3d, "/tmp/z3d.bin", N);
    export_data(z21d, "/tmp/z21d.bin", N);
    export_data(z3d_fftw, "/tmp/z3d_fftw.bin", N);

    free(z3d);
    free(z21d);
    free(z3d_fftw);

    HIP_ASSERT(hipFree(z_d));
    return 0;
}
