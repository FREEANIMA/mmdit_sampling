# app/ui_colab.py
import base64, io, time, os
from datetime import datetime
from IPython.display import HTML, display, clear_output
from google.colab.output import eval_js
from PIL import Image

from .handler import generate_txt2img, generate_inpainting

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _img_path_to_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def run_ui(base_root="."):
    """
    코랩에서 run_ui(base_root) 실행하면
    - 모드 선택 (txt2img / inpainting)
    - 공통 파라미터 (tags, nsfw, sampler, cfg, steps, seed)
    - txt2img: H, W 입력
    - inpainting: 이미지 경로 입력 -> 생성 (마스크는 InPaint 내부에서 그림)
    - 결과 인라인 표시 + 저장 경로 출력
    """
    html = """
    <style>
    .ui-box { font-family: sans-serif; width: 640px; padding: 12px; border:1px solid #ccc; border-radius:8px; }
    .ui-row { margin: 6px 0; }
    .ui-row label { display:inline-block; width: 110px; font-weight:600; }
    .ui-row input, .ui-row select { width: 380px; padding: 4px; }
    .ui-btns { margin-top: 10px; }
    .ui-btns button { padding: 6px 14px; margin-right: 6px; cursor:pointer; }
    #out { margin-top: 12px; font-size: 13px; color: #333; white-space: pre-wrap; }
    </style>
    <div class="ui-box">
      <div class="ui-row"><label>Mode</label>
        <select id="mode">
          <option value="txt2img">txt2img</option>
          <option value="inpainting">inpainting</option>
        </select>
      </div>
      <div class="ui-row"><label>Tags</label><input id="tags" placeholder="예: 1girl, cinematic, ..."></div>
      <div class="ui-row"><label>NSFW</label>
        <select id="nsfw"><option value="false">n</option><option value="true">y</option></select>
      </div>
      <div class="ui-row"><label>Sampler</label>
        <select id="sampler">
          <option>euler</option><option>heun</option><option>ddim</option><option>ddpm_plus</option>
        </select>
      </div>
      <div class="ui-row"><label>CFG</label><input id="cfg" value="4.0"></div>
      <div class="ui-row"><label>Steps</label><input id="steps" value="30"></div>
      <div class="ui-row"><label>Seed</label><input id="seed" value="-1"></div>

      <div id="txt2img-box">
        <div class="ui-row"><label>H</label><input id="H" value="1024"></div>
        <div class="ui-row"><label>W</label><input id="W" value="1024"></div>
      </div>

      <div id="inpaint-box" style="display:none;">
        <div class="ui-row"><label>Image Path</label><input id="img_path" placeholder="/content/xxx.png"></div>
        <div class="ui-row"><label>Brush</label><input id="brush" value="30"></div>
        <div class="ui-row"><label>Start T</label><input id="start_t" value="0.9"></div>
      </div>

      <div class="ui-btns">
        <button id="run">Generate</button>
        <button id="clear_out">Clear</button>
      </div>
      <div id="out"></div>
    </div>
    <script>
    const modeEl = document.getElementById('mode');
    modeEl.onchange = () => {
      const isT2I = modeEl.value === 'txt2img';
      document.getElementById('txt2img-box').style.display = isT2I ? 'block' : 'none';
      document.getElementById('inpaint-box').style.display = isT2I ? 'none' : 'block';
    };
    document.getElementById('clear_out').onclick = () => { document.getElementById('out').textContent = ''; };
    document.getElementById('run').onclick = () => {
      window.ui_params = {
        mode: modeEl.value,
        tags: document.getElementById('tags').value,
        nsfw: document.getElementById('nsfw').value === 'true',
        sampler: document.getElementById('sampler').value,
        cfg: document.getElementById('cfg').value,
        steps: document.getElementById('steps').value,
        seed: document.getElementById('seed').value,
        H: document.getElementById('H').value,
        W: document.getElementById('W').value,
        img_path: document.getElementById('img_path').value,
        brush: document.getElementById('brush').value,
        start_t: document.getElementById('start_t').value, // ✅ 추가
      };
      document.getElementById('out').textContent = 'running...';
    };
    </script>
    """
    display(HTML(html))
    eval_js("window.ui_params = null")
    print("파라미터를 입력하고 Generate 버튼을 누르세요...")
    while True:
        p = eval_js("window.ui_params")
        if p: break
        time.sleep(0.2)

    mode = p["mode"]
    tags = p["tags"]
    nsfw = bool(p["nsfw"])
    sampler = p["sampler"]
    cfg = float(p["cfg"])
    steps = int(p["steps"])
    seed_raw = int(p["seed"]) if p["seed"] not in ("", None) else -1
    seed = None if seed_raw == -1 else seed_raw

    try:
        if mode == "txt2img":
            H = int(p["H"]); W = int(p["W"])
            img, path = generate_txt2img(
                base_root=base_root, tags_text=tags, nsfw=nsfw,
                calc_type=sampler, cfg_scale=cfg,
                H=H, W=W, steps=steps, seed=seed,
            )
        else:
            img_path = p["img_path"].strip()
            if not img_path or not os.path.exists(img_path):
                raise FileNotFoundError(f"이미지 경로 없음: {img_path}")
            start_t = float(p.get("start_t", 0.9))
          
            img, path = generate_inpainting(
                base_root=base_root, tags_text=tags, nsfw=nsfw,
                calc_type=sampler, cfg_scale=cfg,
                steps=steps, seed=seed, img_path=img_path,
                start_t=start_t,
            )

        # 결과 표시
        b64 = _pil_to_b64(img)
        clear_output(wait=True)
        display(HTML(f"""
        <div style="font-family:sans-serif;">
          <div><b>saved:</b> {path}</div>
          <img src="data:image/png;base64,{b64}" style="max-width:512px; border:1px solid #ccc; margin-top:8px;">
        </div>
        """))
        print(f"saved: {path}")
    except Exception as e:
        clear_output(wait=True)
        print(f"실패: {type(e).__name__}: {e}")