import torch
from tqdm.auto import tqdm
import gc
import torch
from pprint import pprint

def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def default_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32

def strip_orig_mod(state_dict):
    prefix = "_orig_mod."
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }

def load_model(model_class, path):
    with tqdm(total=4, desc="Loading model") as bar:
        bar.set_postfix_str("체크포인트 읽기")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        config = checkpoint.get("config")
        if config is None:
            raise KeyError(f"모델 파라미터 키를 찾지 못했습니다. keys={list(checkpoint.keys())}")

        state_dict = next(
            (checkpoint[k] for k in ("ema_state_dict", "model_state_dict", "model") if k in checkpoint),
            None,
        )
        if state_dict is None:
            raise KeyError(f"state_dict 키를 찾지 못했습니다. keys={list(checkpoint.keys())}")
        state_dict = strip_orig_mod(state_dict)
        bar.update(1)

        bar.set_postfix_str("모델 생성")
        with torch.device("meta"):
            model = model_class(**config)
        bar.update(1)

        bar.set_postfix_str("가중치 로드")
        model.load_state_dict(state_dict, assign=True)
        bar.update(1)

        model.eval()
        del checkpoint, state_dict
        bar.update(1)

    print(f"[config] {model_class.__name__}")
    pprint(config, sort_dicts=False)   # 저장된 키 순서 그대로 출력

    return model, config

def free_cuda(verbose: bool = False) -> None:
    gc.collect()                      # 파이썬 쪽 참조부터 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # 캐시된 블록 반환
        torch.cuda.ipc_collect()
        if verbose:
            print(f"[cuda] allocated {torch.cuda.memory_allocated()/1024**3:.2f} GB, "
                  f"reserved {torch.cuda.memory_reserved()/1024**3:.2f} GB")