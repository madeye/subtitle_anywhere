/**
 * Background service worker for Subtitle Anywhere.
 * Manages tab capture lifecycle and transcription forwarding.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let isCapturing = false;
let currentTabId = null;
let wsConnected = false;

// Migrate stored wsUrl from localhost to 127.0.0.1
chrome.storage.sync.get({ wsUrl: '' }, (r) => {
  if (r.wsUrl === 'ws://localhost:10095') {
    chrome.storage.sync.set({ wsUrl: 'ws://127.0.0.1:10095' });
  }
});

// ---------------------------------------------------------------------------
// Settings helpers
// ---------------------------------------------------------------------------
const DEFAULT_SETTINGS = {
  wsUrl: 'ws://127.0.0.1:10095',
  targetLang: 'none',
  fontSize: 20,
  fontFamily: '',
  textColor: '#ffffff',
  bgOpacity: 0.6,
  verticalPosition: 10,
  maxLines: 2,
  fadeTimeout: 5,
  sourceLang: 'auto'
};

async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(DEFAULT_SETTINGS, (result) => {
      resolve(result);
    });
  });
}

// ---------------------------------------------------------------------------
// Offscreen document management
// ---------------------------------------------------------------------------
async function hasOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT']
  });
  return contexts.length > 0;
}

async function createOffscreenDocIfNeeded() {
  if (await hasOffscreenDocument()) return;
  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['USER_MEDIA'],
    justification: 'Capture tab audio for real-time speech-to-text subtitles'
  });
}

async function closeOffscreenDoc() {
  if (await hasOffscreenDocument()) {
    await chrome.offscreen.closeDocument();
  }
}

// ---------------------------------------------------------------------------
// Capture toggle
// ---------------------------------------------------------------------------
async function ensureContentScript(tabId) {
  try {
    const response = await chrome.tabs.sendMessage(tabId, { type: 'ping' }).catch(() => null);
    if (response?.ok) return;
  } catch {}
  await chrome.scripting.executeScript({
    target: { tabId },
    files: ['content.js']
  });
  await chrome.scripting.insertCSS({
    target: { tabId },
    files: ['content.css']
  });
  console.log('[Subtitle Anywhere] Content script injected into tab', tabId);
}

async function startCapture(tabId) {
  if (isCapturing) return;

  const settings = await getSettings();

  try {
    await ensureContentScript(tabId);

    const streamId = await new Promise((resolve, reject) => {
      chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (id) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(id);
        }
      });
    });

    await createOffscreenDocIfNeeded();

    await chrome.runtime.sendMessage({
      type: 'start-capture',
      streamId,
      wsUrl: settings.wsUrl,
      targetLang: settings.targetLang
    });

    isCapturing = true;
    currentTabId = tabId;
    wsConnected = true;

    broadcastStatus();
  } catch (err) {
    console.error('[Subtitle Anywhere] startCapture error:', err);
    isCapturing = false;
    currentTabId = null;
    wsConnected = false;
    broadcastStatus();
  }
}

async function stopCapture() {
  if (!isCapturing) return;

  try {
    await chrome.runtime.sendMessage({ type: 'stop-capture' });
  } catch (err) {
    console.error('[Subtitle Anywhere] stopCapture sendMessage error:', err);
  }

  try {
    await closeOffscreenDoc();
  } catch (err) {
    console.error('[Subtitle Anywhere] closeOffscreen error:', err);
  }

  isCapturing = false;
  wsConnected = false;

  if (currentTabId) {
    chrome.tabs.sendMessage(currentTabId, { type: 'subtitle-stop' }).catch(() => {});
  }
  currentTabId = null;

  broadcastStatus();
}

// ---------------------------------------------------------------------------
// Status broadcast
// ---------------------------------------------------------------------------
function broadcastStatus() {
  const status = { type: 'status', isCapturing, connected: wsConnected };
  chrome.runtime.sendMessage(status).catch(() => {});
  updateBadge();
}

function updateBadge() {
  if (isCapturing) {
    chrome.action.setBadgeText({ text: 'ON' });
    chrome.action.setBadgeBackgroundColor({ color: '#22c55e' });
  } else {
    chrome.action.setBadgeText({ text: '' });
  }
}

// ---------------------------------------------------------------------------
// Port-based messaging from offscreen doc and content scripts
// ---------------------------------------------------------------------------
chrome.runtime.onConnect.addListener((port) => {
  if (port.name === 'offscreen') {
    console.log('[Subtitle Anywhere] Offscreen port connected');
    port.onMessage.addListener((message) => {
      if (message.type === 'transcription' && message.data) {
        handleTranscription(message.data);
      } else if (message.type === 'capture-status') {
        handleCaptureStatus(message);
      }
    });
    port.onDisconnect.addListener(() => {
      console.log('[Subtitle Anywhere] Offscreen port disconnected');
    });
  } else if (port.name === 'content-script') {
    port.onDisconnect.addListener(() => {});
  }
});

// ---------------------------------------------------------------------------
// Message handling
// ---------------------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'toggle-capture') {
    const tabId = message.tabId;
    if (isCapturing) {
      stopCapture().then(() => sendResponse({ success: true }));
    } else {
      startCapture(tabId).then(() => sendResponse({ success: true }));
    }
    return true;
  }

  if (message.type === 'get-status') {
    sendResponse({ isCapturing, connected: wsConnected });
    return false;
  }

  if (message.type === 'transcription' && message.data) {
    handleTranscription(message.data);
    return false;
  }

  if (message.type === 'capture-status') {
    handleCaptureStatus(message);
    return false;
  }
});

// ---------------------------------------------------------------------------
// Capture status handling
// ---------------------------------------------------------------------------
function handleCaptureStatus(message) {
  if (message.status === 'ws-error' || message.status === 'ws-closed') {
    wsConnected = false;
    broadcastStatus();
  } else if (message.status === 'started') {
    wsConnected = true;
    broadcastStatus();
  } else if (message.status === 'stopped') {
    wsConnected = false;
    broadcastStatus();
  } else if (message.status === 'error') {
    console.error('[Subtitle Anywhere] Capture error:', message.error);
    isCapturing = false;
    wsConnected = false;
    currentTabId = null;
    broadcastStatus();
  }
}

// ---------------------------------------------------------------------------
// Transcription handling
// ---------------------------------------------------------------------------
function handleTranscription(data) {
  if (!currentTabId) return;

  const text = data.text || '';
  const translatedText = data.translated_text || null;
  const isFinal = data.is_final !== undefined ? data.is_final : (data.mode === 'offline');
  const isPartial = !isFinal;

  if (!text.trim()) return;

  chrome.tabs.sendMessage(currentTabId, {
    type: 'subtitle',
    text,
    translatedText,
    isPartial,
    isFinal
  }).catch(() => {});
}
