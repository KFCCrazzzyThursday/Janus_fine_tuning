import time
import torch
import argparse

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
    """
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)

    # 1) 准备数据
    # 生成 [matrix_size, matrix_size] 的随机矩阵
    A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)

    # 如果有指定稀疏度，则手动将矩阵变为部分零
    if sparsity > 0:
        A = make_sparse(A, sparsity)
        B = make_sparse(B, sparsity)

    # 2) Warm-up (让 CUDA 先稳定)
    for _ in range(5):
        _ = torch.matmul(A, B)

    torch.cuda.synchronize()

    # 3) 正式计时
    start = time.perf_counter()
    C = torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start

    # 4) 理论 FLOPs 计算
    # 矩阵乘法 (M×N) × (N×K) 大约是 2×M×N×K FLOPs
    # 这里 M=N=K=matrix_size
    flops = 2 * (matrix_size ** 3)
    gflops = flops / 1e9
    tflops = (flops / elapsed) / 1e12

    # 5) 格式化输出到表格
    # 让各列有固定宽度，保证对齐
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


if __name__ == "__main__":
    # python tflops_test.py --size 8192 --dtype float16 --sparsity 0.9
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192, help="Matrix dimension (size x size)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16","float32","bfloat16"], help="Data type")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity ratio, e.g. 0.9 means 90% zeros")
    args = parser.parse_args()

    # 解析 dtype
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    test_tflops(matrix_size=args.size,
                dtype=torch_dtype,
                sparsity=args.sparsity)
