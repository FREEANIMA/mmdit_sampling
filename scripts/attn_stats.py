import torch, math

@torch.no_grad()
def attn_summary(w, h, w_tok, MAX, n_txt):
    """w: (B, heads, N, N) softmax 결과. h, w_tok: 유효 토큰 그리드. n_txt: 유효 txt 토큰 수"""
    w = w[0]                                   # (nh, N, N)
    T = MAX * MAX
    dev = w.device

    ys, xs = torch.meshgrid(torch.arange(MAX, device=dev), torch.arange(MAX, device=dev), indexing="ij")
    valid = ((ys < h) & (xs < w_tok)).reshape(-1)          # (T,)
    vi = valid.nonzero().squeeze(-1)
    Nv = vi.numel()

    q = w[:, vi]                                           # (nh, Nv, N) img query rows
    k_img = q[:, :, :T][:, :, vi]                          # (nh, Nv, Nv) img→img
    k_txt = q[:, :, T:T + n_txt]                           # (nh, Nv, n_txt) img→txt

    img_mass = k_img.sum(-1)                               # (nh, Nv)
    txt_mass = k_txt.sum(-1)

    # 정규화 엔트로피 (1=완전 균일, 0=한 토큰에 몰림)
    p = torch.cat([k_img, k_txt], dim=-1).clamp_min(1e-12)
    ent = -(p * p.log()).sum(-1) / math.log(Nv + n_txt)

    # 평균 어텐션 거리 (토큰 grid 단위) → 국소성
    coords = torch.stack([ys.reshape(-1)[vi], xs.reshape(-1)[vi]], -1).float()
    dist = torch.cdist(coords, coords)                     # (Nv, Nv)
    mean_dist = (k_img * dist).sum(-1) / img_mass.clamp_min(1e-8)

    # 가장 센 단일 key에 몰린 정도 (sink 탐지)
    top1 = torch.cat([k_img, k_txt], -1).amax(-1)

    # txt query → img 로 가는 양 (텍스트가 이미지를 얼마나 읽는지)
    qt = w[:, T:T + n_txt][:, :, :T][:, :, vi]
    txt2img = qt.sum(-1).mean(-1)                          # (nh,)

    return {
        "entropy":   ent.mean(-1).cpu(),       # (nh,)
        "txt_mass":  txt_mass.mean(-1).cpu(),
        "mean_dist": mean_dist.mean(-1).cpu(),
        "top1":      top1.mean(-1).cpu(),
        "txt2img":   txt2img.cpu(),
    }

@torch.no_grad()
def tag_attn_map(w, h, w_tok, MAX, tok_idx):
    """특정 text token(예: hand, face)에 img query들이 주는 어텐션 → (h, w) 맵"""
    T = MAX * MAX
    m = w[0, :, :T, T + tok_idx].mean(0)       # head 평균, (T,)
    return m.reshape(MAX, MAX)[:h, :w_tok].cpu()


def stack_log(log):
    """log: [{"step", "sigma", "blocks": {b: summary}}] -> 지표별 (steps, blocks, heads) 텐서"""
    keys = log[0]["blocks"][0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([
            torch.stack([row["blocks"][b][k] for b in sorted(row["blocks"])])
            for row in log
        ])                                   # (S, B, nh)
    out["sigma"] = torch.tensor([r["sigma"] for r in log])
    return out

def print_stats(d, metrics=("entropy", "txt_mass", "mean_dist", "top1"), stride=1):
    """d: stack_log 결과. 지표별로 행=step, 열=block (head 평균) 표를 출력"""
    sigma = d["sigma"]
    for m in metrics:
        t = d[m].mean(-1)                     # (S, B)
        print(f"\n[{m}]  rows=step(sigma)  cols=block")
        print("step  sigma  " + " ".join(f"{'b'+str(b):>5}" for b in range(t.shape[1])))
        for s in range(0, t.shape[0], stride):
            vals = " ".join(f"{v:5.2f}" for v in t[s].tolist())
            print(f"{s:4d}  {sigma[s]:.3f}  {vals}")