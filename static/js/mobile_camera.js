(() => {
  const token = window.mobileCameraToken;
  const video = document.querySelector('#phone-camera');
  const canvas = document.querySelector('#phone-canvas');
  const ctx = canvas.getContext('2d');
  const message = document.querySelector('#mobile-message');
  const statusEl = document.querySelector('#mobile-status');
  const errorBox = document.querySelector('#camera-error');
  const startBtn = document.querySelector('#phone-start');
  const switchBtn = document.querySelector('#phone-switch');
  const stopBtn = document.querySelector('#phone-stop');

  let stream = null;
  let timer = null;
  let sending = false;
  let facing = 'environment';
  let framesSent = 0;
  let consecutiveErrors = 0;
  const MAX_ERRORS = 8;

  function setStatus(text, type) {
    statusEl.textContent = text;
    statusEl.className = 'connection-status ' + type;
  }

  function showError(text) {
    errorBox.textContent = text;
    errorBox.hidden = false;
  }

  function hideError() {
    errorBox.hidden = true;
  }

  function cameraErrorMessage(err) {
    const name = err.name || '';
    if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
      return 'Camera permission was denied. Please allow camera access in your browser settings and try again.';
    }
    if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
      return 'No camera was detected on this device.';
    }
    if (name === 'NotReadableError' || name === 'TrackStartError') {
      return 'Camera is in use by another application. Close other camera apps and try again.';
    }
    if (name === 'OverconstrainedError' || name === 'ConstraintNotSatisfiedError') {
      return `The selected camera (${facing}) is not available. Try switching cameras.`;
    }
    if (name === 'SecurityError') {
      return 'Camera access requires a secure connection (HTTPS). Ask the PC operator to enable HTTPS.';
    }
    if (name === 'TypeError') {
      return 'Camera API is not supported by this browser. Use a modern browser (Chrome, Safari, Firefox).';
    }
    return `Camera unavailable: ${err.message || name || 'Unknown error'}`;
  }

  async function sendFrame() {
    if (!stream || sending || video.readyState < 2) return;
    sending = true;
    canvas.width = 320;
    canvas.height = Math.round(320 * video.videoHeight / video.videoWidth) || 240;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    try {
      const response = await fetch(`/api/mobile-camera/${token}/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.68) })
      });
      const data = await response.json();
      if (!response.ok) {
        if (response.status === 409) {
          showError('Camera session already in use by another device. Disconnect from the PC first.');
          stopCamera();
          setStatus('Session in use by another device', 'error');
          return;
        }
        if (response.status === 404) {
          showError('Camera session has expired. Return to the PC and generate a new QR code.');
          stopCamera();
          setStatus('Session expired', 'error');
          return;
        }
        throw Error(data.error || 'Frame upload failed');
      }
      consecutiveErrors = 0;
      framesSent++;
      hideError();
      setStatus('Connected to RepAI ✓', 'connected');
      message.textContent = `Streaming ${facing === 'environment' ? 'rear' : 'front'} camera · ${framesSent} frames sent`;
    } catch (error) {
      consecutiveErrors++;
      if (consecutiveErrors >= MAX_ERRORS) {
        showError('Lost connection to RepAI. Check your Wi-Fi and try again.');
        stopCamera();
        setStatus('Connection lost', 'error');
      }
    } finally {
      sending = false;
    }
  }

  async function startCamera() {
    hideError();
    setStatus('Starting camera…', 'waiting');
    try {
      if (stream) stream.getTracks().forEach(track => track.stop());
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });
      video.srcObject = stream;
      await video.play();
      if (timer) clearInterval(timer);
      timer = setInterval(sendFrame, 120);
      startBtn.disabled = true;
      stopBtn.disabled = false;
      consecutiveErrors = 0;
      setStatus('Camera active — streaming to RepAI', 'connected');
      message.textContent = `Using ${facing === 'environment' ? 'rear' : 'front'} camera.`;
    } catch (error) {
      const msg = cameraErrorMessage(error);
      showError(msg);
      setStatus('Camera error', 'error');
      message.textContent = 'Could not start camera.';
    }
  }

  function stopCamera() {
    if (timer) clearInterval(timer);
    timer = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    if (statusEl.className.indexOf('error') === -1) {
      setStatus('Camera stopped', 'waiting');
    }
    message.textContent = 'Camera stopped. Press Start Camera to reconnect.';
  }

  startBtn.onclick = startCamera;
  switchBtn.onclick = async () => {
    facing = facing === 'environment' ? 'user' : 'environment';
    if (stream) {
      await startCamera();
    } else {
      message.textContent = `${facing === 'environment' ? 'Rear' : 'Front'} camera selected. Press Start Camera.`;
    }
  };
  stopBtn.onclick = stopCamera;
  window.addEventListener('beforeunload', stopCamera);

  // Initial status
  setStatus('Ready to connect', 'waiting');
})();
