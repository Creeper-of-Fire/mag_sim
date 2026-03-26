#!/bin/bash

cd /mnt/warpx/mag_sim

# 测试不同 OMP 线程数
for threads in 1 2 4 8 16 32; do
    echo "========================================="
    echo "Testing OMP_NUM_THREADS=$threads"
    echo "========================================="

    export OMP_NUM_THREADS=$threads
    export OMP_PROC_BIND=spread
    export CUDA_VISIBLE_DEVICES=0

    source /root/spack/share/spack/setup-env.sh && spack env activate warpx-4090

    # 运行小测试
    time python main.py -o test_omp_${threads} -c '{
      "n_plasma": 7.3e33,
      "T_plasma_eV": 84000,
      "target_sigma": 0.05,
      "LX": 10,
      "LT": 10,
      "NX": 32,
      "NY": 32,
      "NZ": 32,
      "NPPC": 10
    }' 2>&1 | grep -E "OMP initialized|TinyProfiler total time"

    echo ""
done
