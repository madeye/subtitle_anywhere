/**
 * Popup script for Subtitle Anywhere.
 */

const toggleBtn = document.getElementById('toggleBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const sourceLang = document.getElementById('sourceLang');
const targetLang = document.getElementById('targetLang');
const fontSize = document.getElementById('fontSize');
const fontSizeValue = document.getElementById('fontSizeValue');
const optionsLink = document.getElementById('optionsLink');

let isCapturing = false;

// ---------------------------------------------------------------------------
// Initialise
// ---------------------------------------------------------------------------
async function init() {
  // Load saved settings
  const stored = await chrome.storage.sync.get({
    sourceLang: 'auto',
    targetLang: 'none',
    fontSize: 20,
    translationEnabled: false
  });

  sourceLang.value = stored.sourceLang;
  targetLang.value = stored.targetLang;
  fontSize.value = stored.fontSize;
  fontSizeValue.textContent = stored.fontSize;

  // Get current status from background
  chrome.runtime.sendMessage({ type: 'get-status' }, (response) => {
    if (chrome.runtime.lastError) return;
    if (response) {
      updateStatus(response.isCapturing, response.connected);
    }
  });
}

// ---------------------------------------------------------------------------
// Status updates
// ---------------------------------------------------------------------------
function updateStatus(capturing, connected) {
  isCapturing = capturing;

  if (capturing) {
    toggleBtn.textContent = 'Stop Capture';
    toggleBtn.classList.add('active');
    statusDot.classList.toggle('connected', connected);
    statusText.textContent = connected ? 'Connected' : 'Disconnected';
  } else {
    toggleBtn.textContent = 'Start Capture';
    toggleBtn.classList.remove('active');
    statusDot.classList.remove('connected');
    statusText.textContent = 'Not capturing';
  }
}

// Listen for status updates from background
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'status') {
    updateStatus(message.isCapturing, message.connected);
  }
});

// ---------------------------------------------------------------------------
// Toggle capture
// ---------------------------------------------------------------------------
toggleBtn.addEventListener('click', async () => {
  toggleBtn.disabled = true;

  try {
    // Get the currently active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) {
      console.error('No active tab found');
      return;
    }

    chrome.runtime.sendMessage(
      { type: 'toggle-capture', tabId: tab.id },
      (response) => {
        toggleBtn.disabled = false;
        if (chrome.runtime.lastError) {
          console.error('Toggle error:', chrome.runtime.lastError.message);
        }
      }
    );
  } catch (err) {
    console.error('Toggle error:', err);
    toggleBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Settings controls
// ---------------------------------------------------------------------------
sourceLang.addEventListener('change', () => {
  chrome.storage.sync.set({ sourceLang: sourceLang.value });
});

targetLang.addEventListener('change', () => {
  const val = targetLang.value;
  const enabled = val !== 'none';
  chrome.storage.sync.set({
    targetLang: val,
    translationEnabled: enabled
  });
  chrome.runtime.sendMessage({ type: 'settings-changed' });
});

fontSize.addEventListener('input', () => {
  fontSizeValue.textContent = fontSize.value;
  chrome.storage.sync.set({ fontSize: parseInt(fontSize.value, 10) });
});

// ---------------------------------------------------------------------------
// Options link
// ---------------------------------------------------------------------------
optionsLink.addEventListener('click', () => {
  chrome.runtime.openOptionsPage();
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
init();
