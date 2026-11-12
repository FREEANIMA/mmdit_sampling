import base64
import io
import time
from pathlib import Path

from google.colab.output import eval_js
from IPython.display import HTML, clear_output, display
from PIL import Image

from .handler import Handler


class ColabUI:
    ASSET_DIR = Path(__file__).with_name("ui_assets")
    POLL_INTERVAL = 0.2  # seconds

    def __init__(self, handler: Handler):
        self.handler = handler

    def _read_asset(self, name: str) -> str:
        return (self.ASSET_DIR / name).read_text(encoding="utf-8")

    def _render_form(self) -> None:
        css = self._read_asset("style.css")
        layout = self._read_asset("layout.html")
        js = self._read_asset("script.js")
        display(HTML(f"<style>{css}</style>\n{layout}\n<script>{js}</script>"))

    def _wait_for_params(self) -> dict:
        """Generate 버튼이 눌려 window.ui_params 가 채워질 때까지 대기."""
        eval_js("window.ui_params = null")
        print("파라미터를 입력하고 Generate 버튼을 누르세요...")
        while True:
            params = eval_js("window.ui_params")
            if params:
                return params
            time.sleep(self.POLL_INTERVAL)

    # ---------- 생성 ----------
    def _generate(self, p: dict):
        seed_text = str(p["seed"]).strip()
        common = dict(
            tags_text=p["tags"],
            nsfw=bool(p["nsfw"]),
            calc_type=p["sampler"],
            cfg_scale=float(p["cfg"]),
            steps=int(p["steps"]),
            seed=int(seed_text) if seed_text else -1,  # -1 이면 랜덤 시드
        )

        if p["mode"] == "txt2img":
            return self.handler.generate_txt2img(H=int(p["H"]), W=int(p["W"]), **common)

        img_path = p["img_path"].strip()
        if not img_path or not Path(img_path).exists():
            raise FileNotFoundError(f"이미지 경로 없음: {img_path}")
        return self.handler.generate_inpainting(
            img_path=img_path, start_t=float(p.get("start_t", 0.9)), **common
        )

    # ---------- 결과 ----------
    @staticmethod
    def _pil_to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _show_result(self, img: Image.Image, path: str) -> None:
        clear_output(wait=True)
        display(HTML(f"""
        <div style="font-family:sans-serif;">
          <div><b>saved:</b> {path}</div>
          <img src="data:image/png;base64,{self._pil_to_b64(img)}"
               style="max-width:512px; border:1px solid #ccc; margin-top:8px;">
        </div>
        """))

    # ---------- 실행 ----------
    def run(self) -> None:
        self._render_form()
        params = self._wait_for_params()
        try:
            img, path = self._generate(params)
            self._show_result(img, path)
        except Exception as e:
            clear_output(wait=True)
            print(f"실패: {type(e).__name__}: {e}")