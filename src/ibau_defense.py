# =========================
# PATCH for ibau_defense.py
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_project_(delta: torch.Tensor, c_delta: float):
    """
    Project delta onto l2-ball with radius c_delta (per-sample universal trigger in batch form).
    Here delta is shaped like input (N,C,H,W) but we maintain a single universal delta by using
    a batch-wise parameter; simplest is to keep delta same for all samples in batch.
    We project each sample delta, then you can average if you want a single delta.
    """
    if c_delta is None or c_delta <= 0:
        return delta
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factors = torch.clamp(c_delta / norms, max=1.0)
    flat = flat * factors
    return flat.view_as(delta)


def _hvp_wrt_delta(loss: torch.Tensor, delta: torch.Tensor, vec: torch.Tensor):
    """
    Hessian-vector product: (d^2 loss / d delta^2) @ vec
    """
    grad = torch.autograd.grad(loss, delta, create_graph=True, retain_graph=True)[0]
    hvp = torch.autograd.grad(grad, delta, grad_outputs=vec, retain_graph=True)[0]
    return hvp


def _solve_hinv_g_fixed_point(
    loss: torch.Tensor,
    delta: torch.Tensor,
    g: torch.Tensor,
    solver_iters: int = 5,
    damping: float = 20.0,
):
    """
    Approximate v = H^{-1} g with a fixed-point / Richardson iteration:
        v <- v - (H v - g)/damping
    Paper suggests using iterative solver (CG / fixed-point) with a small number of iterations.
    (Algorithm 1 line 6; Appendix mentions fixed-point algorithm and uses small iters in practice.)
    """
    v = torch.zeros_like(g)
    for _ in range(max(1, int(solver_iters))):
        Hv = _hvp_wrt_delta(loss, delta, v)
        v = v - (Hv - g) / float(damping)
    return v


def ibau_unlearn(
    model: nn.Module,
    clean_loader,
    device,
    outer_rounds: int = 1,
    inner_steps: int = 10,
    alpha_delta: float = 0.1,
    beta_theta: float = 5e-4,
    c_delta: float = 20.0,
    solver_iters: int = 5,
    max_batches_per_round: int = 200,
    # ---- compatibility alias (some callers used max_batches) ----
    max_batches: int = None,
    criterion=None,
    amp: bool = False,
):
    """
    I-BAU (ICLR 2022) style implicit backdoor adversarial unlearning, aligned with Algorithm 1:

    Inner (universal trigger):
        delta <- delta + alpha * ∇_1 H(delta, theta)
        project ||delta||_2 <= c_delta

    Outer (implicit hypergradient):
        \tilde{∇}ψ(θ) = ∇_2 H(delta, θ) + (∇_δ(θ))^T ∇_1 H(delta, θ)
    with response Jacobian approx:
        ∇_δ(θ) = - (∇^2_{11} H)^{-1} ∇^2_{1,2} H
    Therefore:
        \tilde{∇}ψ(θ) = ∇_2 H(delta, θ) - (∇^2_{1,2} H)^T (∇^2_{11} H)^{-1} ∇_1 H

    We approximate v = (∇^2_{11} H)^{-1} ∇_1 H by an iterative solver.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # honor alias
    if max_batches is not None:
        max_batches_per_round = max_batches

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp))

    for r in range(int(outer_rounds)):
        print(f"[I-BAU] Outer round {r+1}/{int(outer_rounds)}")

        # initialize universal delta (we keep a "batch-shaped" delta; it works as a practical approximation)
        delta = None

        used = 0
        for images, labels in clean_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if delta is None or delta.shape != images.shape:
                delta = torch.zeros_like(images, device=device, requires_grad=True)

            # ---------- Inner maximization: update delta ----------
            for _ in range(int(inner_steps)):
                with torch.cuda.amp.autocast(enabled=bool(amp)):
                    logits = model(images + delta)
                    loss_inner = criterion(logits, labels)

                grad_delta = torch.autograd.grad(loss_inner, delta, create_graph=False, retain_graph=True)[0]
                delta = (delta + float(alpha_delta) * grad_delta).detach()
                delta = _l2_project_(delta, float(c_delta)).detach()
                delta.requires_grad_(True)

            # ---------- Outer minimization: implicit hypergradient ----------
            with torch.cuda.amp.autocast(enabled=bool(amp)):
                logits = model(images + delta)
                loss = criterion(logits, labels)

            # g = ∇_1 H (w.r.t delta)
            g = torch.autograd.grad(loss, delta, create_graph=True, retain_graph=True)[0]

            # v ≈ (∇^2_{11} H)^{-1} g
            v = _solve_hinv_g_fixed_point(
                loss=loss,
                delta=delta,
                g=g,
                solver_iters=int(solver_iters),
                damping=float(c_delta if (c_delta is not None and c_delta > 0) else 20.0),
            )

            # direct grad: ∇_2 H (w.r.t theta)
            params = [p for p in model.parameters() if p.requires_grad]
            grad_theta = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

            # cross term: (∇^2_{1,2} H)^T v  == d/dθ <∇_1 H, v>
            # NOTE: IMPORTANT SIGN (paper Eq.(3)): response Jacobian has a minus sign,
            # so hypergradient = grad_theta - cross_term
            inner_prod = (g * v).sum()
            cross_term = torch.autograd.grad(inner_prod, params, retain_graph=True)

            # apply update: θ <- θ - beta * (grad_theta - cross_term)
            with torch.no_grad():
                for p, gt, ct in zip(params, grad_theta, cross_term):
                    if gt is None:
                        continue
                    upd = gt - ct
                    p.add_( -float(beta_theta) * upd )

            used += 1
            if max_batches_per_round and int(max_batches_per_round) > 0 and used >= int(max_batches_per_round):
                break

    return model
