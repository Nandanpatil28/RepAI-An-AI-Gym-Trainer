(() => {
  const id = window.videoAnalysisId;
  const video = document.querySelector('#analysis-video');
  const canvas = document.querySelector('#analysis-canvas');
  const ctx = canvas.getContext('2d');
  const poseOverlay = document.querySelector('#analysis-pose-overlay');
  const message = document.querySelector('#analysis-message');
  const start = document.querySelector('#start-analysis');
  const restart = document.querySelector('#restart-analysis');
  const save = document.querySelector('#save-workout');
  let analysing = false;
  let busy = false;
  let finished = false;
  let lastSample = 0;
  let sampler = null;
  const AI_INTERVAL = 100;

  const post = async (url, data) => {
    const response = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: data ? JSON.stringify(data) : undefined
    });
    const json = await response.json();
    if (!response.ok) throw Error(json.error || 'Analysis request failed.');
    return json;
  };

  const hidePoseOverlay = () => {
    poseOverlay.removeAttribute('src');
    poseOverlay.hidden = true;
  };

  const updatePoseOverlay = data => {
    if (data.image && data.pose_quality === 'tracking') {
      poseOverlay.src = data.image;
      poseOverlay.hidden = false;
      return;
    }
    hidePoseOverlay();
  };

  const render = data => {
    document.querySelector('#mode').textContent = `${data.mode} (${data.mode_type})`;
    document.querySelector('#reps').textContent = data.reps;
    document.querySelector('#status').textContent = data.status;
    document.querySelector('#confidence').textContent = `${data.confidence}%`;
    document.querySelector('#detecting').hidden = !data.transitioning;
    Object.entries(data.counters).forEach(([key, value]) => {
      document.querySelector(`#counter-${key}`).textContent = value;
    });
    document.querySelector('#total-reps').textContent = Object.values(data.counters).reduce((a, b) => a + b, 0);
    document.querySelector('#progress').value = data.progress;
    document.querySelector('#progress-text').textContent = `${data.progress}%`;
    document.querySelector('#frame-progress').textContent = `Original: ${data.source_fps} FPS · ${data.frame_count} frames · ${data.duration_seconds}s | AI: up to ${data.ai_fps} FPS (${data.sampled_frames} samples)`;
    document.querySelector('#video-time').textContent = new Date(video.currentTime * 1000).toISOString().slice(14, 19);
    document.querySelector('#frame-label').textContent = `Frame ${data.frame} / ${data.frame_count}`;
    document.querySelector('#detected-exercises').textContent = `Detected exercises: ${data.detected_exercises.length ? data.detected_exercises.join(', ') : '—'}`;
    updatePoseOverlay(data);
  };

  async function sample() {
    if (!analysing || busy || video.paused || video.ended || video.readyState < 2) return;
    const now = performance.now();
    if (now - lastSample < AI_INTERVAL) return;
    lastSample = now;
    busy = true;
    canvas.width = 320;
    canvas.height = Math.round(320 * video.videoHeight / video.videoWidth) || 240;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    try {
      render(await post(`/api/video/${id}/frame`, {
        image: canvas.toDataURL('image/jpeg', .7),
        timestamp: video.currentTime
      }));
    } catch (error) {
      message.textContent = error.message;
      hidePoseOverlay();
    } finally {
      busy = false;
    }
  }

  function startSampler() {
    if (sampler) clearInterval(sampler);
    sampler = setInterval(sample, 40);
  }

  function stopSampler() {
    if (sampler) clearInterval(sampler);
    sampler = null;
  }

  async function finish() {
    if (finished) return;
    finished = true;
    analysing = false;
    stopSampler();
    try {
      render(await post(`/api/video/${id}/complete`));
      document.querySelector('#progress-title').textContent = 'Analysis Complete ✓';
      message.textContent = 'Analysis complete at the source video duration.';
      save.disabled = false;
      document.querySelector('#no-person').hidden = document.querySelector('#status').textContent !== 'NO POSE DETECTED';
    } catch (error) {
      message.textContent = error.message;
    }
  }

  start.onclick = async () => {
    analysing = true;
    finished = false;
    start.disabled = true;
    message.textContent = 'Playing at source speed; analysing sampled frames independently.';
    try {
      await video.play();
    } catch (error) {
      message.textContent = error.message;
      start.disabled = false;
    }
  };

  restart.onclick = async () => {
    analysing = false;
    finished = false;
    stopSampler();
    hidePoseOverlay();
    save.disabled = true;
    video.pause();
    video.currentTime = 0;
    try {
      render(await post(`/api/video/${id}/restart`));
      start.disabled = false;
      message.textContent = 'Analysis reset. Press Start Analysis to replay at original speed.';
    } catch (error) {
      message.textContent = error.message;
    }
  };

  video.addEventListener('play', () => {
    if (!finished) {
      analysing = true;
      startSampler();
    }
  });
  video.addEventListener('pause', () => {
    if (!video.ended) {
      stopSampler();
      message.textContent = 'Video paused. AI sampling is paused too.';
    }
  });
  video.addEventListener('seeking', hidePoseOverlay);
  video.addEventListener('ended', finish);
  video.addEventListener('loadedmetadata', () => {
    document.querySelector('#frame-progress').textContent = `Original: ${video.videoWidth}×${video.videoHeight} · ${video.duration.toFixed(2)} seconds`;
  });
  save.onclick = async () => {
    try {
      const data = await post(`/api/video/${id}/save`);
      message.textContent = data.message;
      save.disabled = true;
      setTimeout(() => { location.href = '/history'; }, 700);
    } catch (error) {
      message.textContent = error.message;
    }
  };
  window.addEventListener('beforeunload', stopSampler);
})();
