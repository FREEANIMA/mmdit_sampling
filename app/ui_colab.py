# app/ui_colab.py
from __future__ import annotations

import base64
import io
import json
import time
import traceback
from pathlib import Path

from google.colab.output import eval_js
from IPython.display import HTML, display
from PIL import Image

from common.utils import free_cuda

from .handler import Handler


class ColabUI:
    """Colab 입력 폼 + 연속 생성 루프.

    - 폼은 한 번만 그리고, Generate 를 누를 때마다 결과 영역만 갱신
    - 생성 중에는 버튼 비활성화, 끝나면 다시 활성화
    - NAS(속성) 옵션은 handler 에서 읽어 script.js 보다 먼저 주입
    - 종료: 셀 정지(■) 버튼 (KeyboardInterrupt)
    """

    ASSET_DIR = Path(__file__).with_name("ui_assets")
    POLL_INTERVAL = 0.2  # seconds

    def __init__(self, handler: Handler):
        self.handler = handler
        self._result_area = None

    # ---------- 폼 ----------
    def _read_asset(self, name: str) -> str:
        return (self.ASSET_DIR / name).read_text(encoding="utf-8")

    def _nas_options_script(self) -> str:
        """window.NAS_OPTIONS 주입용 <script>. script.js 보다 먼저 실행돼야 함."""
        options = self.handler.get_nas_options()
        payload = json.dumps(options, ensure_ascii=False)
        payload = payload.replace("</", "<\\/")  # 태그 안에 </script> 가 들어가도 깨지지 않게
        return f"<script>window.NAS_OPTIONS = {payload};</script>"

    def _render_form(self) -> None:
        css = self._read_asset("style.css")
        layout = self._read_asset("layout.html")
        js = self._read_asset("script.js")
        display(HTML(
            f"<style>{css}</style>\n"
            f"{layout}\n"
            f"{self._nas_options_script()}\n"
            f"<script>{js}</script>"
        ))
        # 결과가 표시될 자리 (폼 아래, 이후 update 로 계속 교체)
        self._result_area = display(HTML("<div></div>"), display_id=True)
        eval_js("window.ui_params = null")

    def _set_status(self, text: str, busy: bool) -> None:
        eval_js(
            f"document.getElementById('out').textContent = {json.dumps(text)};"
            f"document.getElementById('run').disabled = {str(busy).lower()};"
        )

    def _wait_for_params(self) -> dict:
        """Generate 버튼이 눌릴 때까지 대기. 읽자마자 비워서 다음 클릭에 대비."""
        while True:
            params = eval_js("window.ui_params")
            if params:
                eval_js("window.ui_params = null")
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
            nas=p.get("nas") or {},                    # {type: "__auto__" | "__none__" | tag}
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
        self._result_area.update(HTML(f"""
        <div style="font-family:sans-serif;">
          <div><b>saved:</b> {path}</div>
          <img src="data:image/png;base64,{self._pil_to_b64(img)}"
               style="max-width:512px; border:1px solid #ccc; margin-top:8px;">
        </div>
        """))

    def _show_error(self, e: Exception) -> None:
        tb = traceback.format_exc()
        self._result_area.update(HTML(
            f"<pre style='color:#b00; white-space:pre-wrap;'>"
            f"실패: {type(e).__name__}: {e}\n\n{tb}</pre>"
        ))

    # ---------- 실행 ----------
    def run(self) -> None:
        self._render_form()
        self._set_status("파라미터를 입력하고 Generate 를 누르세요", busy=False)

        try:
            while True:
                params = self._wait_for_params()
                self._set_status("running...", busy=True)
                try:
                    img, path = self._generate(params)
                    self._show_result(img, path)
                    status = f"done: {path}"
                except Exception as e:
                    self._show_error(e)
                    status = "failed"
                # except 블록을 벗어나야 예외(traceback)가 잡고 있던 텐서가 해제됨
                free_cuda(verbose=True)
                self._set_status(status, busy=False)
        except KeyboardInterrupt:
            self._set_status("중지됨 (run() 을 다시 실행하세요)", busy=True)