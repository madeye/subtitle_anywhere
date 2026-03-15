/**
 * Options page script for Subtitle Anywhere.
 */

// Element references
const els = {
  wsUrl: document.getElementById('wsUrl'),
  targetLang: document.getElementById('targetLang'),
  fontFamily: document.getElementById('fontFamily'),
  fontSizeOpt: document.getElementById('fontSizeOpt'),
  fontSizeOptValue: document.getElementById('fontSizeOptValue'),
  fontPreview: document.getElementById('fontPreview'),
  textColor: document.getElementById('textColor'),
  bgOpacity: document.getElementById('bgOpacity'),
  bgOpacityValue: document.getElementById('bgOpacityValue'),
  verticalPosition: document.getElementById('verticalPosition'),
  verticalPositionValue: document.getElementById('verticalPositionValue'),
  maxLines: document.getElementById('maxLines'),
  maxLinesValue: document.getElementById('maxLinesValue'),
  fadeTimeout: document.getElementById('fadeTimeout'),
  fadeTimeoutValue: document.getElementById('fadeTimeoutValue'),
  saveBtn: document.getElementById('saveBtn'),
  savedNotice: document.getElementById('savedNotice')
};

// Default settings
const DEFAULTS = {
  wsUrl: 'ws://127.0.0.1:10095',
  targetLang: 'none',
  fontFamily: '',
  fontSize: 20,
  textColor: '#ffffff',
  bgOpacity: 0.6,
  verticalPosition: 10,
  maxLines: 2,
  fadeTimeout: 5
};

// ---------------------------------------------------------------------------
// Load settings
// ---------------------------------------------------------------------------
async function loadSettings() {
  const stored = await chrome.storage.sync.get(DEFAULTS);

  els.wsUrl.value = stored.wsUrl;
  els.targetLang.value = stored.targetLang;
  els.fontFamily.value = stored.fontFamily;
  els.fontSizeOpt.value = stored.fontSize;
  els.fontSizeOptValue.textContent = stored.fontSize;
  els.fontPreview.style.fontSize = stored.fontSize + 'px';
  els.textColor.value = stored.textColor;
  els.bgOpacity.value = stored.bgOpacity;
  els.bgOpacityValue.textContent = stored.bgOpacity;
  els.verticalPosition.value = stored.verticalPosition;
  els.verticalPositionValue.textContent = stored.verticalPosition;
  els.maxLines.value = stored.maxLines;
  els.maxLinesValue.textContent = stored.maxLines;
  els.fadeTimeout.value = stored.fadeTimeout;
  els.fadeTimeoutValue.textContent = stored.fadeTimeout;
}

// ---------------------------------------------------------------------------
// Save settings
// ---------------------------------------------------------------------------
async function saveSettings() {
  const settings = {
    wsUrl: els.wsUrl.value.trim() || DEFAULTS.wsUrl,
    targetLang: els.targetLang.value,
    fontFamily: els.fontFamily.value.trim(),
    fontSize: parseInt(els.fontSizeOpt.value, 10),
    textColor: els.textColor.value,
    bgOpacity: parseFloat(els.bgOpacity.value),
    verticalPosition: parseInt(els.verticalPosition.value, 10),
    maxLines: parseInt(els.maxLines.value, 10),
    fadeTimeout: parseInt(els.fadeTimeout.value, 10)
  };

  await chrome.storage.sync.set(settings);

  // Notify background
  chrome.runtime.sendMessage({ type: 'settings-changed' }).catch(() => {});

  // Show saved notice
  els.savedNotice.classList.add('visible');
  setTimeout(() => {
    els.savedNotice.classList.remove('visible');
  }, 2000);
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

// Range sliders: live update display values
els.fontSizeOpt.addEventListener('input', () => {
  els.fontSizeOptValue.textContent = els.fontSizeOpt.value;
  els.fontPreview.style.fontSize = els.fontSizeOpt.value + 'px';
});

els.bgOpacity.addEventListener('input', () => {
  els.bgOpacityValue.textContent = els.bgOpacity.value;
});

els.verticalPosition.addEventListener('input', () => {
  els.verticalPositionValue.textContent = els.verticalPosition.value;
});

els.maxLines.addEventListener('input', () => {
  els.maxLinesValue.textContent = els.maxLines.value;
});

els.fadeTimeout.addEventListener('input', () => {
  els.fadeTimeoutValue.textContent = els.fadeTimeout.value;
});

// Save button
els.saveBtn.addEventListener('click', saveSettings);

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
loadSettings();
