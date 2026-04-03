import unittest
import torch
from parameterized import parameterized
from sonicmoe.functional.topk_softmax_hopper import topk_softmax_triton
from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.reduction_over_k_gather_hopper import token_gather_and_sum_varlen_K_triton as gather_sum_hopper
from .test_commons import TestCommons


def _run_topk(logits, K):
    T, E = logits.shape
    scores = torch.zeros(T, K, device="cuda", dtype=torch.float32)
    indices = torch.zeros(T, K, device="cuda", dtype=torch.int32)
    topk_softmax_triton(logits, K, scores, indices)
    torch.cuda.synchronize()
    return scores.cpu(), indices.cpu()


class TestTopKCorrectness(TestCommons):
    CONFIGS = [
        (8192, 128, 8),
        (8192, 64, 4),
        (8192, 32, 2),
        (4096, 256, 16),
        (16384, 128, 8),
    ]

    @parameterized.expand(CONFIGS)
    def test_topk_indices_match(self, T, E, K):
        torch.manual_seed(42)
        logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)
        scores, idx = _run_topk(logits, K)
        ref_vals, ref_idx = torch.topk(logits.float(), K, dim=-1)
        ref_sorted, _ = ref_idx.cpu().sort(dim=-1)
        fused_sorted, _ = idx.sort(dim=-1)
        self.assertTrue(torch.equal(ref_sorted, fused_sorted),
            f"[T={T},E={E},K={K}] TopK indices differ")

    @parameterized.expand(CONFIGS)
    def test_softmax_scores_close(self, T, E, K):
        torch.manual_seed(42)
        logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)
        scores, idx = _run_topk(logits, K)
        ref_vals, ref_idx = torch.topk(logits.float(), K, dim=-1)
        ref_sm = torch.softmax(ref_vals, dim=-1).cpu()
        ref_order = ref_idx.cpu().argsort(dim=-1)
        fused_order = idx.argsort(dim=-1)
        ref_sorted = ref_sm.gather(1, ref_order)
        fused_sorted = scores.gather(1, fused_order)
        torch.testing.assert_close(ref_sorted, fused_sorted, atol=1e-5, rtol=1e-4,
            msg=f"[T={T},E={E},K={K}] Softmax scores differ")

    @parameterized.expand(CONFIGS)
    def test_softmax_sums_to_one(self, T, E, K):
        torch.manual_seed(0)
        logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)
        scores, _ = _run_topk(logits, K)
        row_sums = scores.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones(T), atol=1e-4, rtol=1e-4,
            msg=f"[T={T},E={E},K={K}] Softmax rows do not sum to 1")


class TestGatherSumCorrectness(TestCommons):
    CONFIGS = [
        (4096, 2048, 8),
        (4096, 1536, 4),
        (8192, 4096, 16),
    ]

    @parameterized.expand(CONFIGS)
    def test_gather_sum_matches_original(self, T, H, K):
        torch.manual_seed(42)
        Mtotal = T * K
        x = torch.randn(Mtotal, H, device="cuda", dtype=torch.bfloat16)
        w = torch.rand(Mtotal, device="cuda", dtype=torch.float32)
        w_2d = w.view(T, K)
        w_2d = w_2d / w_2d.sum(dim=-1, keepdim=True)
        w = w_2d.reshape(-1)
        M_perm = torch.randperm(Mtotal, device="cuda", dtype=torch.int32)
        M_offset = torch.arange(0, T + 1, device="cuda", dtype=torch.int32) * K
        out1 = torch.zeros(T, H, device="cuda", dtype=torch.float32)
        out2 = torch.zeros(T, H, device="cuda", dtype=torch.float32)
        gather_sum_hopper(x, w, out1, M_perm, M_offset, T, K, H, is_varlen_K=False)
        gather_sum_hopper(x, w, out2, M_perm, M_offset, T, K, H, is_varlen_K=False)
        torch.cuda.synchronize()
        torch.testing.assert_close(out1.cpu(), out2.cpu(), atol=1e-4, rtol=1e-4,
            msg=f"[T={T},H={H},K={K}] Gather-sum output differs")


class TestMoEHopperEndToEnd(TestCommons):
    _SEED = 42
    CONFIGS = [
        (8192, 768, 256, 128, 8),
        (8192, 1536, 512, 64, 4),
        (8192, 4096, 512, 128, 8),
    ]

    @parameterized.expand(CONFIGS)
    def test_forward_matches_torch(self, T, H, I, E, K):
        self.set_seed(self._SEED)
        device = torch.device("cuda")
        with torch.device(device):
            moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
                       intermediate_size=I, activation_function=ActivationType.SWIGLU,
                       add_bias=False, std=0.02).to(dtype=torch.bfloat16)
        x = 0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)
        with torch.autocast(device.type, torch.float32):
            y_sonic = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)[0]
            y_torch = moe(x, kernel_backend_moe=KernelBackendMoE.torch)[0]
        self.assert_equal_tensors(y_sonic.float(), y_torch.float(), exact_match=False,
            atol_bfloat16=1.4e-2, rtol_bfloat16=2e-2, dtype=torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
