/opt/rocm/bin/hipcc -c fft3.cpp -o fft3.o -Ofast -march=native -mtune=native -fPIC -std=c++17 -I/opt/rocm/rocfft/include 
/opt/rocm/bin/hipcc fft3.o -o fft3 -Ofast -march=native -mtune=native -fPIC -std=c++17 -lrocfft
