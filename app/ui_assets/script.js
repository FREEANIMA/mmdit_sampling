(() => {
  const $ = (id) => document.getElementById(id);
  const modeEl = $('mode');

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
    };
    $('out').textContent = 'running...';
  };
})();