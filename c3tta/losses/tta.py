from __future__ import annotations

import torch
import torch.nn.functional as F


def _finite_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    """Return one finite-value flag per batch item."""
    if tensor.ndim == 0:
        return tensor.isfinite().reshape(1)
    return tensor.isfinite().reshape(tensor.shape[0], -1).all(dim=1)


def _safe_float(tensor: torch.Tensor) -> torch.Tensor:
    """Promote low-precision logits and neutralize invalid values."""
    tensor = tensor.float()
    return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))


def entropy_from_logits(logits: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Compute Bernoulli entropy without low-precision ``log(0)`` failures.

    In fp16/bf16, ``1 - 1e-5`` rounds to exactly one, so the common
    clamp-then-log implementation can evaluate ``0 * log(0)``.  The identity
    ``H(sigmoid(z)) = softplus(z) - z * sigmoid(z)`` is stable in fp32 and
    retains the entropy-minimization gradient.
    """
    logits_fp32 = _safe_float(logits)
    entropy = F.softplus(logits_fp32) - logits_fp32 * torch.sigmoid(logits_fp32)
    entropy = entropy.clamp_min(0.0)
    if reduction == "none":
        if entropy.ndim == 0:
            return entropy.reshape(1)
        return entropy.reshape(entropy.shape[0], -1).mean(dim=1)
    if reduction == "mean":
        return entropy.mean()
    raise ValueError(f"Unsupported entropy reduction: {reduction}")


def confidence_gate(
    student: object,
    teacher: object,
    threshold: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = (
        _finite_per_sample(student.seg_logits)
        & _finite_per_sample(student.cls_logits)
        & _finite_per_sample(student.soft_vcdr)
        & _finite_per_sample(teacher.seg_logits)
        & _finite_per_sample(teacher.cls_logits)
    )
    student_seg = torch.sigmoid(_safe_float(student.seg_logits))
    teacher_seg = torch.sigmoid(_safe_float(teacher.seg_logits))
    seg_disagreement = (student_seg - teacher_seg).abs().mean(dim=(1, 2, 3))
    student_cls = torch.sigmoid(_safe_float(student.cls_logits)).reshape(student_seg.shape[0], -1).mean(dim=1)
    teacher_cls = torch.sigmoid(_safe_float(teacher.cls_logits)).reshape(student_seg.shape[0], -1).mean(dim=1)
    cls_disagreement = (student_cls - teacher_cls).abs()
    # Probability ordering can flip by an arbitrarily small amount throughout
    # the low-confidence background. Penalize the violation magnitude instead
    # of counting every tiny flip as a full invalid pixel.
    invalid = torch.relu(teacher_seg[:, 1] - teacher_seg[:, 0]).mean(dim=(1, 2))
    score = seg_disagreement + cls_disagreement + invalid
    valid = valid & torch.isfinite(score)
    gate = (valid & (score < threshold)).float()
    invalid_score = torch.full_like(score, max(float(threshold), 0.0))
    score_for_stats = torch.where(valid, score, invalid_score)
    return gate, {
        "accept_rate": float(gate.mean().detach()),
        "gate_score": float(score_for_stats.mean().detach()),
        "nonfinite_rate": float((~valid).float().mean().detach()),
    }


def pairwise_monotonic_rank_loss(
    soft_vcdr: torch.Tensor,
    glaucoma_probability: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize only inversions between anatomy and diagnosis rankings.

    For every valid, accepted unordered pair ``(i, j)``, the loss is
    ``relu(-(r_i - r_j) * (p_i - p_j))``.  The product formulation is
    intentional: an inverted pair sends gradients through both the soft-vCDR
    branch and the glaucoma-probability branch.  Aligned pairs and ties have
    exactly zero loss.
    """
    batch_size = soft_vcdr.shape[0]
    if glaucoma_probability.shape[0] != batch_size or gate.numel() != batch_size:
        raise ValueError("soft_vcdr, glaucoma_probability, and gate must share a batch size")

    r_finite = _finite_per_sample(soft_vcdr)
    p_finite = _finite_per_sample(glaucoma_probability)
    r = _safe_float(soft_vcdr).reshape(batch_size, -1).mean(dim=1)
    p = _safe_float(glaucoma_probability).reshape(batch_size, -1).mean(dim=1)
    accepted = gate.detach().reshape(-1).bool() & r_finite & p_finite
    accepted_indices = torch.nonzero(accepted, as_tuple=False).flatten()

    # Keep the zero connected to both branches so callers may always call
    # backward(), including for singleton and all-rejected batches.
    graph_zero = (r.sum() + p.sum()) * 0.0
    if accepted_indices.numel() < 2:
        return graph_zero, {
            "pair_count": 0.0,
            "active_pair_rate": 0.0,
            "nonfinite_pair_rate": 0.0,
        }

    pairs = torch.combinations(accepted_indices, r=2)
    delta_r = r[pairs[:, 0]] - r[pairs[:, 1]]
    delta_p = p[pairs[:, 0]] - p[pairs[:, 1]]
    ordering_product = delta_r * delta_p
    finite_pairs = torch.isfinite(ordering_product)
    pair_count = finite_pairs.sum()
    safe_penalty = torch.where(
        finite_pairs,
        torch.relu(-ordering_product),
        torch.zeros_like(ordering_product),
    )
    loss = safe_penalty.sum() / pair_count.clamp_min(1)
    active_pairs = finite_pairs & (ordering_product < 0)
    candidate_pair_count = ordering_product.numel()
    return loss + graph_zero, {
        "pair_count": float(pair_count.detach()),
        "active_pair_rate": float((active_pairs.sum() / pair_count.clamp_min(1)).detach()),
        "nonfinite_pair_rate": float(
            ((~finite_pairs).sum() / max(candidate_pair_count, 1)).detach()
        ),
    }


def c3tta_loss(
    student: object,
    teacher: object,
    gate_threshold: float = 0.25,
    lambda_aug: float = 1.0,
    lambda_nest: float = 0.5,
    lambda_clin: float = 0.5,
    lambda_ent: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    gate, gate_stats = confidence_gate(student, teacher, gate_threshold)
    student_seg = torch.sigmoid(_safe_float(student.seg_logits))
    teacher_seg = torch.sigmoid(_safe_float(teacher.seg_logits)).detach()
    student_cls = torch.sigmoid(_safe_float(student.cls_logits))
    teacher_cls = torch.sigmoid(_safe_float(teacher.cls_logits)).detach()
    consistency = F.mse_loss(student_seg, teacher_seg, reduction="none").mean(dim=(1, 2, 3))
    cls_consistency = F.mse_loss(student_cls, teacher_cls, reduction="none").reshape(student_seg.shape[0], -1).mean(dim=1)
    aug = consistency + cls_consistency
    nest = torch.relu(student_seg[:, 1] - student_seg[:, 0]).mean(dim=(1, 2))
    ent = entropy_from_logits(student.cls_logits, reduction="none") + entropy_from_logits(
        student.seg_logits, reduction="none"
    )
    per_sample = lambda_aug * aug + lambda_nest * nest + lambda_ent * ent
    component_finite = torch.isfinite(per_sample)
    effective_gate = gate * component_finite.float()
    safe_per_sample = torch.where(component_finite, per_sample, torch.zeros_like(per_sample))
    denominator = effective_gate.sum().clamp_min(1.0)
    sample_loss = (safe_per_sample * effective_gate).sum() / denominator
    clin, pair_stats = pairwise_monotonic_rank_loss(
        student.soft_vcdr,
        student_cls,
        effective_gate,
    )
    loss = sample_loss + lambda_clin * clin

    def masked_mean(values: torch.Tensor) -> float:
        safe_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        return float(((safe_values * effective_gate).sum() / denominator).detach())

    component_nonfinite_rate = float((~component_finite).float().mean().detach())
    stats = {
        "aug": masked_mean(aug),
        "nest": masked_mean(nest),
        "clin": float(clin.detach()),
        "entropy": masked_mean(ent),
        "total": float(loss.detach()),
        "accepted_samples": float(effective_gate.sum().detach()),
    }
    stats.update(gate_stats)
    stats.update(pair_stats)
    # Component-level validation can only reduce the gate acceptance.
    stats["accept_rate"] = float(effective_gate.mean().detach())
    stats["nonfinite_rate"] = max(gate_stats["nonfinite_rate"], component_nonfinite_rate)
    return loss, stats
