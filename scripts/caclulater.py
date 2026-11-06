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
    sigma_down = sigma_next * (1.0 - eta * (1.0 - sigma_next / sigma.clamp(min=1e-8)) ** 0.5 * 0 ) if False else None
    sigma_up = eta * sigma_next * (1.0 - (sigma_next / sigma) ** 2).clamp(min=0).sqrt()
    sigma_down = (sigma_next ** 2 - sigma_up ** 2).clamp(min=0).sqrt()
    x_down = (1.0 - sigma_down) * x0 + sigma_down * eps

    return x_down + sigma_up * z