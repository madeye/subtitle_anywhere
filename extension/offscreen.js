/**
 * Offscreen document for capturing tab audio, converting to PCM,
 * and streaming to a WebSocket-based ASR server.
 */

let playbackContext = null;  // native sample rate — keeps original audio quality
let asrContext = null;       // 16kHz mono — for speech recognition
let mediaStream = null;
let playbackSource = null;
let asrSource = null;
let workletNode = null;
let webSocket = null;

// Dedicated port to background for reliable messaging
let bgPort = null;

function ensurePort() {
  if (!bgPort) {
    bgPort = chrome.runtime.connect({ name: 'offscreen' });
    bgPort.onDisconnect.addListener(() => { bgPort = null; });
  }
  return bgPort;
}

function sendToBg(msg) {
  try {
    ensurePort().postMessage(msg);
  } catch (err) {
    console.error('[Subtitle Anywhere] Port send failed:', err);
  }
}

/**
 * Start capturing audio from the given stream and streaming PCM to the server.
 */
async function startCapture(streamId, wsUrl, targetLang) {
  try {
    console.log('[Subtitle Anywhere] Starting capture, streamId:', streamId, 'wsUrl:', wsUrl);

    // Get the tab's audio stream
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId
        }
      }
    });
    console.log('[Subtitle Anywhere] Got media stream');

    // Playback context at native sample rate — preserves original stereo audio
    playbackContext = new AudioContext();
    playbackSource = playbackContext.createMediaStreamSource(mediaStream);
    playbackSource.connect(playbackContext.destination);

    // ASR context at 16kHz mono — for speech recognition only
    asrContext = new AudioContext({ sampleRate: 16000 });
    asrSource = asrContext.createMediaStreamSource(mediaStream);

    // Load the PCM worklet processor
    await asrContext.audioWorklet.addModule(
      chrome.runtime.getURL('lib/pcm-processor.js')
    );

    workletNode = new AudioWorkletNode(asrContext, 'pcm-processor');

    // When we receive PCM chunks from the worklet, send over WebSocket
    workletNode.port.onmessage = (event) => {
      if (webSocket && webSocket.readyState === WebSocket.OPEN) {
        webSocket.send(event.data);
      }
    };

    asrSource.connect(workletNode);

    // Open WebSocket connection to ASR server
    await connectWebSocket(wsUrl, targetLang);

    sendToBg({ type: 'capture-status', status: 'started' });
    console.log('[Subtitle Anywhere] Capture fully started');
  } catch (err) {
    console.error('[Subtitle Anywhere] startCapture error:', err);
    sendToBg({ type: 'capture-status', status: 'error', error: err.message });
    cleanup();
  }
}

/**
 * Connect to the WebSocket ASR server.
 */
function connectWebSocket(wsUrl, targetLang) {
  return new Promise((resolve, reject) => {
    console.log('[Subtitle Anywhere] Connecting WebSocket to', wsUrl);
    webSocket = new WebSocket(wsUrl);

    webSocket.onopen = () => {
      console.log('[Subtitle Anywhere] WebSocket connected');

      const config = {
        mode: '2pass',
        chunk_size: [5, 10, 5],
        audio_fs: 16000,
        wav_format: 'pcm',
        is_speaking: true,
        target_lang: targetLang || ''
      };
      webSocket.send(JSON.stringify(config));
      resolve();
    };

    webSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Forward transcription results to background via port
        sendToBg({ type: 'transcription', data });
      } catch (err) {
        console.error('[Subtitle Anywhere] Failed to parse WS message:', err);
      }
    };

    webSocket.onerror = (event) => {
      console.error('[Subtitle Anywhere] WebSocket error:', event);
      sendToBg({ type: 'capture-status', status: 'ws-error' });
    };

    webSocket.onclose = (event) => {
      console.log('[Subtitle Anywhere] WebSocket closed:', event.code, event.reason);
      sendToBg({ type: 'capture-status', status: 'ws-closed' });
    };

    setTimeout(() => {
      if (webSocket.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket connection timeout'));
      }
    }, 10000);
  });
}

/**
 * Stop capturing and clean up all resources.
 */
async function stopCapture() {
  try {
    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
      webSocket.send(JSON.stringify({ is_speaking: false }));
      await new Promise((resolve) => setTimeout(resolve, 500));
      webSocket.close();
    }
    webSocket = null;
    cleanup();
    sendToBg({ type: 'capture-status', status: 'stopped' });
  } catch (err) {
    console.error('[Subtitle Anywhere] stopCapture error:', err);
    cleanup();
  }
}

/**
 * Clean up audio resources.
 */
function cleanup() {
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (asrSource) { asrSource.disconnect(); asrSource = null; }
  if (asrContext) { asrContext.close().catch(() => {}); asrContext = null; }
  if (playbackSource) { playbackSource.disconnect(); playbackSource = null; }
  if (playbackContext) { playbackContext.close().catch(() => {}); playbackContext = null; }
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
}

// Listen for messages from the background service worker
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'start-capture') {
    startCapture(message.streamId, message.wsUrl, message.targetLang).then(() => {
      sendResponse({ success: true });
    }).catch((err) => {
      sendResponse({ success: false, error: err.message });
    });
    return true;
  }

  if (message.type === 'stop-capture') {
    stopCapture().then(() => {
      sendResponse({ success: true });
    }).catch((err) => {
      sendResponse({ success: false, error: err.message });
    });
    return true;
  }
});
