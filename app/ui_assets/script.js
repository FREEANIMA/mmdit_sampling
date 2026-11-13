(() => {
  const $ = (id) => document.getElementById(id);
  const modeEl = $('mode');

  const NAS = [
    "style", "person_count", "gender", "hair_color", "hair_length", "hair_style",
    "eye_color", "body_type", "clothing", "accessory", "background", "pose",
    "expression", "composition", "gaze", "sex_act"
  ];

  // Python에서 주입: window.NAS_OPTIONS = { type: [tag, ...], ... }
  const opts = window.NAS_OPTIONS || {};

  const escapeHtml = (s) =>
    String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
             .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  NAS.forEach((k) => {
    const row = document.createElement('div');
    row.className = 'ui-row';
    const tagOptions = (opts[k] || [])
      .map((t) => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`)
      .join('');
    row.innerHTML = `
      <label>${k}</label>
      <select id="nas_${k}">
        <option value="__auto__">auto</option>
        <option value="__none__">none</option>
        ${tagOptions}
      </select>`;
    $('nas').appendChild(row);
  });

  const readNas = () =>
    Object.fromEntries(NAS.map((k) => [k, $('nas_' + k).value]));

  modeEl.onchange = () => {
    const isT2I = modeEl.value === 'txt2img';
    $('txt2img-box').style.display = isT2I ? 'block' : 'none';
    $('inpaint-box').style.display = isT2I ? 'none' : 'block';
  };

  $('clear_out').onclick = () => { $('out').textContent = ''; };

  $('run').onclick = () => {
    window.ui_params = {
      mode: modeEl.value,
      tags: $('tags').value,
      nsfw: $('nsfw').value === 'true',
      sampler: $('sampler').value,
      cfg: $('cfg').value,
      steps: $('steps').value,
      seed: $('seed').value,
      H: $('H').value,
      W: $('W').value,
      img_path: $('img_path').value,
      start_t: $('start_t').value,
      nas: readNas(),
    };
    $('out').textContent = 'running...';
  };
})();