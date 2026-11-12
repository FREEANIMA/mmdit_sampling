def v_to_x0_eps(x, v, sigma):
    x0  = x - sigma * v
    eps = x + (1.0 - sigma) * v
    return x0, eps

def euler(x, v, sigma, sigma_next):
    return x + (sigma_next - sigma) * v

def heun(x, v1, v2, sigma, sigma_next):
    return x + (sigma_next - sigma) * 0.5 * (v1 + v2)

def ddim(x, v, sigma, sigma_next):
    x0, eps = v_to_x0_eps(x, v, sigma)
    return (1.0 - sigma_next) * x0 + sigma_next * eps

def ddpm_plus(x, v, z, sigma, sigma_next, eta=1.0):
    x0, eps = v_to_x0_eps(x, v, sigma)

    sigma_up = eta * sigma_next * (1.0 - (sigma_next / sigma) ** 2).clamp(min=0).sqrt()
    sigma_down = (sigma_next ** 2 - sigma_up ** 2).clamp(min=0).sqrt()

    # 신호 계수를 (1 - sigma_next)에 맞추도록 보정
    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    ratio = alpha_next / alpha_down.clamp(min=1e-8)

    x_down = (1.0 - sigma_down) * x0 + sigma_down * eps
    return ratio * x_down + sigma_up * z