import torch
import torch.nn.functional as F
import numpy as np

def analyze_sdpa_diff(manual_out, iree_out, tol=1e-3):
    # Ensure they are torch tensors
    if not isinstance(iree_out, torch.Tensor):
        iree_out = torch.from_numpy(np.array(iree_out))
    if not isinstance(manual_out, torch.Tensor):
        manual_out = torch.from_numpy(np.array(manual_out))

    # 1. Normalize both (in case one is pre-softmax and one is post)
    # If they are already post-softmax, this won't hurt.
    manual_sm = F.softmax(manual_out.float(), dim=-1)
    iree_sm = F.softmax(iree_out.float(), dim=-1)

    diff = torch.abs(manual_sm - iree_sm)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Post-Softmax Max Diff: {max_diff:.6f}")
    print(f"Post-Softmax Mean Diff: {mean_diff:.6f}")

    # 2. Check for Index Shifting (Common in Sliding Window)
    # If the max diff is high, check if it's just shifted by 1 index
    if manual_sm.shape[-1] > 1:
        shifted_diff = torch.abs(manual_sm[:, :, 1:] - iree_sm[:, :, :-1])
        if shifted_diff.mean() < mean_diff:
            print(
                "ALERT: Detected an index shift. Your sliding window offsets may be mismatched."
            )

    # 3. Sum check (Integrity test)
    manual_sums = manual_sm.sum(dim=-1)
    iree_sums = iree_sm.sum(dim=-1)
    if not torch.allclose(manual_sums, torch.ones_like(manual_sums), atol=1e-2):
        print("ERROR: Manual SDPA rows do not sum to 1.0. Check your masking logic.")
    if not torch.allclose(iree_sums, torch.ones_like(iree_sums), atol=1e-2):
        print("ERROR: IREE SDPA rows do not sum to 1.0. The export likely broke the softmax.")
