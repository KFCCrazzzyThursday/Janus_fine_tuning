import time
import torch
import argparse
'''
python tflops_test.py --size 8192 --dtype float16 --sparsity 0.9

python tflops_test.py --comms--dtype float16 --comms_mb 256

'''
def make_sparse(matrix: torch.Tensor, sparsity: float):
    """
    将给定张量 matrix 的指定百分比元素置为 0，用于模拟稀疏场景。
    sparsity=0.9 表示 90% 的元素置 0。
    """
    if sparsity <= 0 or sparsity >= 1:
        return matrix  # 不做处理，或全为零也没意义
    num_elements = matrix.numel()
    k = int(num_elements * sparsity)
    # 随机选 k 个位置置 0
    idx = torch.randperm(num_elements, device=matrix.device)[:k]
    matrix.view(-1)[idx] = 0
    return matrix

def test_tflops(matrix_size=8192,
                dtype=torch.float16,
                sparsity=0.0):
    """
    测试在给定的矩阵尺寸、数据类型，以及稀疏度下的可观测 TFLOPS。
    仍是“单卡”版本（device=0），不会用多卡。
    """
    device = 'cuda:0'   # 固定用 GPU0
    gpu_name = torch.cuda.get_device_name(0)

    # 1) 准备数据
    A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)

    # 稀疏化
    if sparsity > 0:
        A = make_sparse(A, sparsity)
        B = make_sparse(B, sparsity)

    # 2) warm-up
    for _ in range(5):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize(device)

    # 3) 正式计时
    start = time.perf_counter()
    C = torch.matmul(A, B)
    torch.cuda.synchronize(device)
    end = time.perf_counter()

    elapsed = end - start

    # FLOPs
    flops = 2 * (matrix_size ** 3)  # 大约 2 * M*N*K
    gflops = flops / 1e9
    tflops = (flops / elapsed) / 1e12

    # 输出表格
    header = (
        f"{'GPU Name':<16} | {'Float Type':<10} | {'Matrix Size':<11} "
        f"| {'Sparsity':<8} | {'GFlops':<10} | {'Time(s)':<9} | {'TFLOPS/s':<9}"
    )
    divider = "-" * len(header)
    row = (
        f"{gpu_name:<16} | {str(dtype):<10} | {matrix_size}x{matrix_size:<5} "
        f"| {sparsity:<8.2f} | {gflops:<10.2f} | {elapsed:<9.5f} | {tflops:<9.2f}"
    )

    print(header)
    print(divider)
    print(row)

def test_2gpu_comms(dtype=torch.float16, array_mb=256, loops=5):
    """
    测试两张 GPU (cuda:0 与 cuda:1) 间的大块拷贝带宽，以及小块往返延迟。
    array_mb: 拷贝的大块张量大小（MB）。
    loops: 重复测量次数，取平均。
    """
    if torch.cuda.device_count() < 2:
        print("!!! 只检测到1张卡，无法测2卡通信!!!")
        return

    device0 = 'cuda:0'
    device1 = 'cuda:1'
    gpu_name0 = torch.cuda.get_device_name(0)
    gpu_name1 = torch.cuda.get_device_name(1)

    print(f"检测到2张卡: \n  [0]: {gpu_name0}\n  [1]: {gpu_name1}")

    # 1) 计算在给定 dtype 下，每个元素占多少字节
    # float16 一般2字节, float32 一般4字节, bfloat16也2字节 ...
    dtype_bytes = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
    }.get(dtype, 2)  # fallback 2

    # 2) 准备要拷贝的大张量
    num_bytes = array_mb * 1024 * 1024
    num_elems = num_bytes // dtype_bytes
    # 只要 1D 向量即可
    A = torch.randn(num_elems, dtype=dtype, device=device0)

    # warm-up
    B = A.to(device1)
    del B
    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)

    # 3) 大块拷贝测试 (one-way)
    times = []
    for _ in range(loops):
        torch.cuda.synchronize(device0)
        torch.cuda.synchronize(device1)
        start = time.perf_counter()
        B = A.to(device1)  # device0->device1
        torch.cuda.synchronize(device0)
        torch.cuda.synchronize(device1)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    bandwidth = (array_mb / avg_time)  # MB/s
    print(f"--- 大块拷贝 {array_mb}MB from GPU0->GPU1, loops={loops} ---")
    print(f"平均耗时: {avg_time*1e3:.3f} ms, 带宽: {bandwidth:.2f} MB/s")

    del B
    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)

    # 4) 小块往返延迟 (ping-pong)
    #   用很小的tensor, repeated,  测往返
    small_elems = 1  # 1 element
    smallA = torch.randn(small_elems, dtype=dtype, device=device0)
    # warm-up
    smallB = smallA.to(device1)
    smallB = smallB.to(device0)

    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)

    loops2 = 50
    times2 = []
    for _ in range(loops2):
        # ping device0->device1->device0
        torch.cuda.synchronize(device0)
        torch.cuda.synchronize(device1)
        start = time.perf_counter()

        tmp = smallA.to(device1)
        _ = tmp.to(device0)

        torch.cuda.synchronize(device0)
        torch.cuda.synchronize(device1)
        end = time.perf_counter()
        times2.append(end - start)

    avg_time2 = sum(times2)/len(times2)
    # 这是 1次往返的时间
    print(f"--- 小块 1elem ping-pong (device0<->device1), loops={loops2} ---")
    print(f"平均耗时: {avg_time2*1e6:.3f} us (往返)")

if __name__ == "__main__":
    # python tflops_test.py --size 8192 --dtype float16 --sparsity 0.9
    # 或 python tflops_test.py --comms True
    parser = argparse.ArgumentParser()
    parser.add_argument("--comms", action="store_true", help="Test 2-GPU communications instead of single-GPU matmul.")
    parser.add_argument("--size", type=int, default=8192, help="Matrix dimension (size x size) for matmul")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16","float32","bfloat16"], help="Data type")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity ratio, e.g. 0.9 means 90% zeros")
    parser.add_argument("--comms_mb", type=int, default=256, help="Data MB for big copy test in 2-GPU comm test")
    args = parser.parse_args()

    # 解析 dtype
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    if args.comms:
        # 测试两张卡的通信带宽和往返延迟
        test_2gpu_comms(dtype=torch_dtype, array_mb=args.comms_mb, loops=5)
    else:
        # 测试单卡 matmul 性能
        test_tflops(matrix_size=args.size,
                    dtype=torch_dtype,
                    sparsity=args.sparsity)
