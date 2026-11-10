import cv2
import numpy as np

class ImageDecodeError(Exception):
    pass

def img_resize(img_bytes, allowable_size, patch_size, compression_ratio):
    unit = patch_size * compression_ratio
    try:
        arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    except (TypeError, ValueError, cv2.error) as e:
        raise ImageDecodeError(f"{type(e).__name__}: {e}") from e

    if arr is None:
        raise ImageDecodeError(f"size={len(img_bytes)}, head={img_bytes[:16]!r}")

    H, W = arr.shape[:2]
    long_side = max(H, W)

    def round_to_unit(x):
        r = round(x / unit) * unit
        return max(int(r), unit)

    if long_side <= allowable_size:
        new_H, new_W = round_to_unit(H), round_to_unit(W)
    else:
        scale = allowable_size / long_side
        if H >= W:
            new_H = round_to_unit(allowable_size)
            new_W = round_to_unit(W * scale)
        else:
            new_H = round_to_unit(H * scale)
            new_W = round_to_unit(allowable_size)

    if (new_W, new_H) != (W, H):
        shrinking = (new_W * new_H) < (W * H)
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
        arr = cv2.resize(arr, (new_W, new_H), interpolation=interp)

    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    arr = np.ascontiguousarray(arr)

    return arr, new_H, new_W