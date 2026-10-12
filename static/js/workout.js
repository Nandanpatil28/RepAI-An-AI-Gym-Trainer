(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const $$ = selector => Array.from(document.querySelectorAll(selector));

  const video = $('#camera');
  const canvas = $('#frame-canvas');
  const overlayCanvas = $('#overlay-canvas');
  const ctx = canvas ? canvas.getContext('2d') : null;
  const overlayCtx = overlayCanvas ? overlayCanvas.getContext('2d') : null;
  const message = $('#camera-message');
  const card = $('.camera-card');
  const placeholder = $('#camera-placeholder');
  const cameraState = $('#camera-state');
  const chooseWorkoutBtn = $('#choose-workout');
  const selectedWorkoutLabel = $('#selected-workout-label');
  const startCameraBtn = $('#start-camera');
  const stopCameraBtns = $$('#stop-camera');
  const endWorkoutBtns = $$('#end-workout');
  const confirmWorkoutBtn = $('#confirm-exercise');
  const pauseWorkoutBtn = $('#pause-workout');
  const exerciseSelection = $('#exercise-selection');
  const workoutMain = $('#workout-main');
  const workoutControlsBar = $('#workout-controls-bar');
  const modeToggle = $('#mode-toggle');
  const resetReps = $('#reset-reps');
  const feedbackPanel = $('#feedback-panel');
  const feedbackContent = $('#feedback-content');
  const summary = $('#workout-summary');

  if (!video || !canvas || !overlayCanvas || !ctx || !startCameraBtn || !confirmWorkoutBtn) {
    console.error('Workout UI elements missing; controls cannot be initialized.');
    if (message) message.textContent = 'Page failed to load camera controls. Please refresh.';
    return;
  }

  const DISPLAY_NAMES = {
    'push-up': 'Push-Up',
    'pull-up': 'Pull-Up',
    'sit-up': 'Sit-Up',
    squat: 'Squat',
    walk: 'Walking',
  };

  let stream = null;
  let frameTimer = null;
  let sending = false;
  let workoutActive = false;
  let cameraStarting = false;
  let paused = false;
  let selectedExercise = null;   // null = nothing chosen, 'auto' = automatic, or an exercise key
  let currentModeType = 'AUTO';

  const post = async (url, body = {}) => {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw Error(data.error || `Request failed (${response.status})`);
    return data;
  };

  function setMessage(text) {
    if (message) message.textContent = text;
  }

  function show(element, display = 'block') {
    if (!element) return;
    element.hidden = false;
    element.style.display = display;
  }

  function hide(element) {
    if (!element) return;
    element.style.display = 'none';
    element.hidden = true;
  }

  function setBusy(button, busy) {
    if (!button) return;
    button.disabled = busy;
    button.dataset.busy = busy ? '1' : '';
  }

  function formatTime(seconds) {
    const safeSeconds = Number.isFinite(seconds) ? seconds : 0;
    const mins = Math.floor(safeSeconds / 60);
    const secs = safeSeconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function updateCameraButtons() {
    if (cameraState) cameraState.textContent = stream ? 'CAMERA RUNNING' : (workoutActive ? 'CAMERA STOPPED' : 'CAMERA NOT STARTED');
    startCameraBtn.disabled = Boolean(stream) || cameraStarting || !workoutActive;
    if (pauseWorkoutBtn) pauseWorkoutBtn.disabled = !stream || cameraStarting || !workoutActive;
    stopCameraBtns.forEach(button => {
      button.disabled = !stream && !cameraStarting;
    });
  }

  function updateSelectionUI() {
    const selectedButton = selectedExercise
      ? $(`.exercise-btn[data-exercise="${selectedExercise}"]`)
      : null;
    const label = selectedButton ? selectedButton.dataset.label : '';

    $$('.exercise-btn').forEach(button => {
      const isSelected = button.dataset.exercise === selectedExercise;
      button.classList.toggle('selected', isSelected);
      button.setAttribute('aria-pressed', isSelected ? 'true' : 'false');
    });
    if (selectedWorkoutLabel) {
      selectedWorkoutLabel.textContent = label ? `Selected: ${label}` : 'No workout selected';
    }
    if (selectedExercise === 'auto') {
      confirmWorkoutBtn.textContent = 'START AUTO WORKOUT';
      confirmWorkoutBtn.disabled = false;
    } else if (selectedExercise) {
      confirmWorkoutBtn.textContent = `START ${DISPLAY_NAMES[selectedExercise].toUpperCase()} WORKOUT`;
      confirmWorkoutBtn.disabled = false;
    } else {
      confirmWorkoutBtn.textContent = 'SELECT AN EXERCISE TO START';
      confirmWorkoutBtn.disabled = true;
    }
    if (modeToggle) {
      modeToggle.textContent = currentModeType === 'AUTO' ? 'TARGETED MODE' : 'AUTO MODE';
    }
  }

  function setChooserOpen(open) {
    if (open) {
      show(exerciseSelection);
    } else {
      hide(exerciseSelection);
    }
    if (chooseWorkoutBtn) chooseWorkoutBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  function render(data) {
    if (!data) return;
    currentModeType = data.mode_type || currentModeType;

    const mode = $('#mode');
    const modeType = $('#mode-type');
    const reps = $('#reps');
    const countLabel = $('#count-label');
    const status = $('#status');
    const detecting = $('#detecting');
    const timerDisplay = $('#timer');
    const confidenceFill = $('#confidence-fill');
    const confidenceText = $('#confidence-text');

    if (mode) mode.textContent = data.mode || '-';
    if (modeType) modeType.textContent = data.mode_type || '';
    if (reps) reps.textContent = data.reps ?? 0;
    const isWalking = data.mode_key === 'walk';
    if (countLabel) countLabel.textContent = isWalking ? 'STEP COUNT' : 'REPETITION COUNT';
    if (resetReps) resetReps.textContent = isWalking ? 'RESET STEPS' : 'RESET REPS';
    if (status) status.textContent = paused ? 'PAUSED' : (data.status || 'WAITING');
    if (data.pose_message && workoutActive && !paused) setMessage(data.pose_message);
    if (detecting) detecting.hidden = !data.transitioning || paused;
    if (timerDisplay) timerDisplay.textContent = formatTime(data.timer || 0);
    if (confidenceFill) confidenceFill.style.width = `${data.confidence || 0}%`;
    if (confidenceText) confidenceText.textContent = `${data.confidence || 0}%`;

    if (data.counters) {
      Object.entries(data.counters).forEach(([key, value]) => {
        const counter = $(`#counter-${key}`);
        if (counter) counter.textContent = value;
      });
    }

    // Live feedback from the ML pipeline
    if (feedbackContent) {
      feedbackContent.innerHTML = '';
      if (workoutActive) {
        show(feedbackPanel);
        const feedback = data.feedback && data.feedback.length
          ? data.feedback
          : [data.pose_message || 'Waiting for person.'];
        feedback.forEach(text => {
          const item = document.createElement('div');
          item.className = 'feedback-item';
          item.textContent = text;
          feedbackContent.appendChild(item);
        });
      } else {
        feedbackContent.innerHTML = '<p class="feedback-placeholder">Waiting for person.</p>';
      }
    }

    if (data.overlay && overlayCtx) {
      const image = new Image();
      image.onload = () => {
        overlayCanvas.width = image.width;
        overlayCanvas.height = image.height;
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        overlayCtx.drawImage(image, 0, 0);
      };
      image.onerror = () => console.warn('Overlay image failed to decode');
      image.src = data.overlay;
    }

    updateSelectionUI();
  }

  function clearOverlay() {
    if (!overlayCtx || !overlayCanvas) return;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  function stopFrameLoop() {
    if (frameTimer) {
      clearInterval(frameTimer);
      frameTimer = null;
    }
  }

  function stopCamera({ keepWorkout = true } = {}) {
    stopFrameLoop();
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    video.pause();
    video.srcObject = null;
    video.hidden = true;
    overlayCanvas.hidden = true;
    clearOverlay();
    if (placeholder) placeholder.hidden = false;
    if (card) card.classList.remove('streaming');
    if (!keepWorkout) workoutActive = false;
    updateCameraButtons();
  }

  function cameraError(error) {
    const name = error && error.name;
    if (name === 'NotAllowedError' || name === 'PermissionDeniedError') return 'Unable to start camera: Permission denied.';
    if (name === 'NotFoundError' || name === 'DevicesNotFoundError') return 'Unable to start camera: No camera found.';
    if (name === 'NotReadableError' || name === 'TrackStartError') return 'Unable to start camera: It is already being used by another application.';
    if (name === 'OverconstrainedError') return 'Unable to start camera: The requested camera settings are unavailable.';
    if (name === 'SecurityError') return "Unable to start camera: Camera access is blocked by this page's security settings.";
    if (name === 'TypeError') return 'Unable to start camera: This browser does not support secure camera access.';
    return `Unable to start camera: ${(error && error.message) || 'Unknown error'}`;
  }

  function waitForVideoReady() {
    return new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        video.removeEventListener('loadedmetadata', ready);
        video.removeEventListener('canplay', ready);
        video.removeEventListener('error', failed);
      };
      const timeout = window.setTimeout(() => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(new Error('The camera did not provide video frames.'));
      }, 10000);
      const ready = () => {
        if (settled || video.videoWidth <= 0 || video.videoHeight <= 0) return;
        settled = true;
        window.clearTimeout(timeout);
        cleanup();
        resolve();
      };
      const failed = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeout);
        cleanup();
        reject(video.error || new Error('The camera video could not be played.'));
      };
      video.addEventListener('loadedmetadata', ready);
      video.addEventListener('canplay', ready);
      video.addEventListener('error', failed);
      ready();
    });
  }

  async function sendFrame() {
    if (!stream || paused || sending || video.readyState < 2 || !workoutActive) return;
    sending = true;
    try {
      canvas.width = 320;
      canvas.height = Math.round((320 * video.videoHeight) / video.videoWidth) || 240;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      render(await post('/api/workout/frame', { image: canvas.toDataURL('image/jpeg', 0.68) }));
    } catch (error) {
      setMessage(error.message);
      stopCamera();
    } finally {
      sending = false;
    }
  }

  function startFrameLoop() {
    stopFrameLoop();
    frameTimer = window.setInterval(sendFrame, 100);
  }

  async function startCamera() {
    if (cameraStarting || stream || !workoutActive) return;
    cameraStarting = true;
    updateCameraButtons();
    setMessage('Requesting camera permission...');
    try {
      if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
        const unsupported = new TypeError('navigator.mediaDevices.getUserMedia is unavailable');
        unsupported.name = 'TypeError';
        throw unsupported;
      }

      const newStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (!newStream.getVideoTracks().length) {
        newStream.getTracks().forEach(track => track.stop());
        throw new Error('The camera returned no video track.');
      }

      stream = newStream;
      paused = false;
      video.srcObject = stream;
      await video.play();
      await waitForVideoReady();
      video.hidden = false;
      overlayCanvas.hidden = false;
      if (placeholder) placeholder.hidden = true;
      if (card) card.classList.add('streaming');
      startFrameLoop();
      updateCameraButtons();
      setMessage('Computer camera is live. RepAI is analysing your movement.');
      if (pauseWorkoutBtn) pauseWorkoutBtn.textContent = 'PAUSE';
    } catch (error) {
      console.error('Unable to start computer camera:', error);
      stopCamera();
      setMessage(cameraError(error));
    } finally {
      cameraStarting = false;
      updateCameraButtons();
    }
  }

  async function startWorkout() {
    if (!selectedExercise) {
      setMessage('Select an exercise or Automatic Detection before starting.');
      return;
    }

    const isAuto = selectedExercise === 'auto';

    setBusy(confirmWorkoutBtn, true);
    try {
      const data = isAuto
        ? await post('/api/workout/start')
        : await post('/api/workout/select', { exercise: selectedExercise });
      workoutActive = true;
      paused = false;
      render(data);
      setChooserOpen(false);
      show(workoutMain, 'grid');
      show(workoutControlsBar);
      show(feedbackPanel);
      show(summary);
      updateCameraButtons();
      setMessage(isAuto
        ? 'Automatic detection workout ready. Press Start Camera.'
        : `${DISPLAY_NAMES[selectedExercise]} workout ready. Press Start Camera.`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(confirmWorkoutBtn, false);
      updateSelectionUI();
    }
  }

  async function pauseOrResumeWorkout() {
    if (!workoutActive || !stream) {
      setMessage('Start the camera before pausing the workout.');
      return;
    }
    setBusy(pauseWorkoutBtn, true);
    try {
      const nextPaused = !paused;
      const data = await post('/api/workout/control', { action: nextPaused ? 'pause' : 'resume' });
      paused = nextPaused;
      render(data);
      if (pauseWorkoutBtn) pauseWorkoutBtn.textContent = paused ? 'RESUME' : 'PAUSE';
      const statusEl = $('#status');
      if (statusEl) statusEl.textContent = paused ? 'PAUSED' : 'READY';
      setMessage(paused ? 'Workout paused. Camera stays on, but reps are not counted.' : 'Workout resumed.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(pauseWorkoutBtn, false);
    }
  }

  function stopCameraOnly() {
    if (!stream && !cameraStarting) {
      setMessage('Camera is already stopped.');
      return;
    }
    paused = false;
    stopCamera();
    if (pauseWorkoutBtn) pauseWorkoutBtn.textContent = 'PAUSE';
    setMessage('Camera stopped. Press Start Camera to resume this workout.');
  }

  async function endWorkout() {
    if (!workoutActive) {
      setMessage('Start a workout before ending it.');
      return;
    }
    endWorkoutBtns.forEach(button => setBusy(button, true));
    try {
      const data = await post('/api/workout/end');
      render(data);
      setMessage('Workout saved. Redirecting to your history...');
      window.setTimeout(() => {
        window.location.href = '/history';
      }, 700);
    } catch (error) {
      setMessage(error.message);
    } finally {
      stopCamera({ keepWorkout: false });
      paused = false;
      hide(workoutMain);
      hide(workoutControlsBar);
      selectedExercise = null;
      setChooserOpen(false);
      endWorkoutBtns.forEach(button => setBusy(button, false));
      updateSelectionUI();
    }
  }

  async function toggleMode() {
    if (!workoutActive) return;
    setBusy(modeToggle, true);
    try {
      const nextAuto = currentModeType !== 'AUTO';
      const payload = nextAuto
        ? { action: 'auto' }
        : { action: 'manual', exercise: selectedExercise && selectedExercise !== 'auto' ? selectedExercise : 'squat' };
      const data = await post('/api/workout/control', payload);
      render(data);
      setMessage(nextAuto ? 'Switched to automatic detection.' : `Switched to targeted ${DISPLAY_NAMES[payload.exercise]} mode.`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(modeToggle, false);
    }
  }

  async function control(action, successMessage) {
    if (!workoutActive) {
      setMessage('Start a workout first.');
      return;
    }
    const button = action === 'reset' ? resetReps : null;
    if (button) setBusy(button, true);
    try {
      render(await post('/api/workout/control', { action }));
      setMessage(successMessage);
    } catch (error) {
      setMessage(error.message);
    } finally {
      if (button) setBusy(button, false);
    }
  }

  // --- Initial page state: show exercise selection, hide workout ---
  function showInitialState() {
    workoutActive = false;
    paused = false;
    selectedExercise = window.preselectedExercise || null;
    currentModeType = 'AUTO';
    setChooserOpen(Boolean(selectedExercise));
    hide(workoutMain);
    hide(workoutControlsBar);
    hide(summary);
    updateCameraButtons();
    setMessage(selectedExercise ? `${DISPLAY_NAMES[selectedExercise]} selected for targeted mode.` : 'Choose automatic detection or select a targeted exercise, then start the workout.');
    updateSelectionUI();
  }

  // --- Exercise selection buttons (including "auto") ---
  $$('.exercise-btn').forEach(button => {
    button.addEventListener('click', () => {
      selectedExercise = button.dataset.exercise;
      if (selectedExercise === 'auto') {
        setMessage('Automatic detection selected. Press the button below to start.');
      } else {
        setMessage(`${DISPLAY_NAMES[selectedExercise]} selected for targeted mode.`);
      }
      updateSelectionUI();
    });
  });

  if (chooseWorkoutBtn) {
    chooseWorkoutBtn.addEventListener('click', () => {
      const isOpen = exerciseSelection && !exerciseSelection.hidden && exerciseSelection.style.display !== 'none';
      setChooserOpen(!isOpen);
      if (!isOpen) {
        setMessage('Choose Automatic Detection or a targeted exercise before starting.');
      }
    });
  }
  confirmWorkoutBtn.addEventListener('click', startWorkout);
  startCameraBtn.addEventListener('click', startCamera);
  stopCameraBtns.forEach(button => button.addEventListener('click', stopCameraOnly));
  endWorkoutBtns.forEach(button => button.addEventListener('click', endWorkout));
  if (pauseWorkoutBtn) pauseWorkoutBtn.addEventListener('click', pauseOrResumeWorkout);
  if (modeToggle) modeToggle.addEventListener('click', toggleMode);
  if (resetReps) resetReps.addEventListener('click', () => {
    const isWalking = $('#mode') && $('#mode').textContent === 'Walking';
    control('reset', `${isWalking ? 'Steps' : 'Reps'} reset for the current exercise.`);
  });

  window.addEventListener('beforeunload', () => stopCamera({ keepWorkout: false }));

  showInitialState();
})();
