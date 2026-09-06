(() => {
  const $ = selector => document.querySelector(selector);
  const message = $('#connection-message'), feed = $('#mobile-feed'), placeholder = $('#mobile-placeholder');
  const state = $('#pair-state'), disconnect = $('#disconnect-phone'), end = $('#end-mobile-workout');
  let token = null, pollTimer = null;
  const post = async (url, body = {}) => { const response = await fetch(url, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const data = await response.json(); if(!response.ok) throw Error(data.error || 'Request failed'); return data; };
  async function createSession() { try { const data=await post('/api/mobile-camera/create'); token=data.token; $('#qr-code').src=data.qr_code; $('#pair-url').textContent=data.mobile_url; state.textContent='Waiting for phone connection…'; message.textContent='Scan the QR code with your phone to connect its camera.'; pollTimer=setInterval(poll,500); } catch(error) { message.textContent=`Unable to create mobile connection: ${error.message}`; state.textContent='CONNECTION ERROR'; } }
  async function poll() { if(!token) return; try { const response=await fetch(`/api/mobile-camera/${token}/status`), data=await response.json(); if(!response.ok) throw Error(data.error || 'Connection unavailable'); if(data.image) { feed.src=data.image; feed.hidden=false; placeholder.hidden=true; $('.camera-card').classList.add('streaming'); state.textContent='MOBILE CAMERA CONNECTED'; disconnect.hidden=false; end.disabled=false; } else if(data.connected) { state.textContent='Phone connected — start the camera on your phone.'; disconnect.hidden=false; } } catch(error) { stopPolling(); state.textContent='PHONE DISCONNECTED'; message.textContent=error.message; } }
  function stopPolling() { clearInterval(pollTimer); pollTimer=null; }
  disconnect.addEventListener('click',async()=>{ if(!token) return; try { await post(`/api/mobile-camera/${token}/disconnect`); feed.src=''; feed.hidden=true; placeholder.hidden=false; $('.camera-card').classList.remove('streaming'); state.textContent='PHONE DISCONNECTED — SCAN THE SAME QR TO RECONNECT'; disconnect.hidden=true; end.disabled=false; } catch(error) { message.textContent=error.message; } });
  end.addEventListener('click',async()=>{ if(!token) return; try { await post(`/api/mobile-camera/${token}/end`); stopPolling(); token=null; message.textContent='Mobile workout saved. Redirecting to your history…'; setTimeout(()=>location.href='/history',700); } catch(error) { message.textContent=error.message; } });
  window.addEventListener('beforeunload',stopPolling);
  createSession();
})();
