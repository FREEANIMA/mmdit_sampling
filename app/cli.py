from .handler import generate

def ask(prompt, default, cast=str):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  잘못된 값 -> 기본값 {default} 사용")
        return default

def main(base_root):
    print("종료: 태그 입력에서 q")
    while True:
        tags = input("\ntags (쉼표 구분)> ").strip()
        if tags.lower() in ("q", "quit", "exit"):
            break
        if not tags:
            continue

        nsfw = ask("nsfw (y/n)", "n").lower() == "y"
        sampler = ask("sampler (euler/heun/ddim/ddpm_plus)", "euler")
        if sampler not in ("euler", "heun", "ddim", "ddpm_plus"):
            print("  지원하지 않는 sampler -> euler 사용")
            sampler = "euler"
        cfg = ask("cfg", 4.0, float)
        steps = ask("steps", 30, int)
        H = ask("H", 1024, int)
        W = ask("W", 1024, int)
        seed = ask("seed (-1=랜덤)", -1, int)

        try:
            img, path = generate(
                base_root=base_root,
                tags_text=tags,
                nsfw=nsfw,
                calc_type=sampler,
                cfg_scale=cfg,
                H=H,
                W=W,
                steps=steps,
                seed=seed,
            )
            print(f"saved: {path}")
        except Exception as e:
            print(f"실패: {type(e).__name__}: {e}")