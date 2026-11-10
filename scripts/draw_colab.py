from IPython.display import HTML, display
from google.colab.output import eval_js
from base64 import b64encode, b64decode
import numpy as np
import cv2
from PIL import Image
import io, time

def draw_mask_colab(img_rgb, brush=20):
    H, W = img_rgb.shape[:2]
    # 원본을 PNG base64로 인코딩
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    img_b64 = b64encode(buf.getvalue()).decode()

    canvas_html = f"""
    <div style="position:relative; width:{W}px; height:{H}px; border:1px solid #888;">
      <canvas id="bg"   width="{W}" height="{H}" style="position:absolute; left:0; top:0;"></canvas>
      <canvas id="ov"   width="{W}" height="{H}" style="position:absolute; left:0; top:0; cursor:crosshair;"></canvas>
    </div>
    <br>
    <button id="finish">Finish</button>
    <button id="clear">Clear</button>
    <script>
    const bg = document.getElementById('bg').getContext('2d');
    const ov = document.getElementById('ov').getContext('2d');
    const top = document.getElementById('ov');
    const img = new Image();
    img.onload = () => bg.drawImage(img, 0, 0, {W}, {H});
    img.src = "data:image/png;base64,{img_b64}";

    ov.strokeStyle = 'rgba(255,0,0,0.6)';
    ov.lineWidth = {brush};
    ov.lineCap = 'round';
    let drawing = false;
    top.addEventListener('mousedown', e => {{ drawing = true; ov.beginPath(); ov.moveTo(e.offsetX, e.offsetY); }});
    top.addEventListener('mousemove', e => {{ if (drawing) {{ ov.lineTo(e.offsetX, e.offsetY); ov.stroke(); }} }});
    top.addEventListener('mouseup', () => drawing = false);
    top.addEventListener('mouseleave', () => drawing = false);
    document.getElementById('clear').onclick = () => ov.clearRect(0, 0, {W}, {H});
    document.getElementById('finish').onclick = () => {{
        // 오버레이만 뽑아서 마스크로 (투명=0, 빨강=칠함)
        const tmp = document.createElement('canvas');
        tmp.width = {W}; tmp.height = {H};
        const tctx = tmp.getContext('2d');
        tctx.drawImage(document.getElementById('ov'), 0, 0);
        const d = tctx.getImageData(0, 0, {W}, {H}).data;
        const out = document.createElement('canvas');
        out.width = {W}; out.height = {H};
        const octx = out.getContext('2d');
        const od = octx.createImageData({W}, {H});
        for (let i = 0; i < d.length; i += 4) {{
            const a = d[i + 3];                 // 알파
            od.data[i] = od.data[i+1] = od.data[i+2] = a > 30 ? 255 : 0;
            od.data[i+3] = 255;
        }}
        octx.putImageData(od, 0, 0);
        window.data = out.toDataURL('image/png');
    }};
    </script>
    """
    display(HTML(canvas_html))
    eval_js("window.data = null")
    print("마우스로 칠한 뒤 Finish 버튼을 누르세요...")
    while True:
        data = eval_js("window.data")
        if data: break
        time.sleep(0.2)

    binary = b64decode(data.split(',')[1])
    mask = np.array(Image.open(io.BytesIO(binary)).convert('L'))
    return (mask > 127).astype(np.uint8) * 255