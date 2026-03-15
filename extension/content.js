/**
 * Content script for Subtitle Anywhere.
 * Finds video elements, creates a fixed overlay, and displays subtitles.
 */

(() => {
  'use strict';

  // Avoid double-injection
  if (window.__subtitleAnywhereInjected) return;
  window.__subtitleAnywhereInjected = true;

  // -------------------------------------------------------------------------
  // Settings (defaults, updated from storage)
  // -------------------------------------------------------------------------
  let settings = {
    fontSize: 20,
    fontFamily: '',
    textColor: '#ffffff',
    bgOpacity: 0.6,
    verticalPosition: 10,
    maxLines: 2,
    fadeTimeout: 5
  };

  chrome.storage.sync.get(settings, (stored) => {
    Object.assign(settings, stored);
    applyStyles();
  });

  chrome.storage.onChanged.addListener((changes) => {
    for (const [key, { newValue }] of Object.entries(changes)) {
      if (key in settings) {
        settings[key] = newValue;
      }
    }
    applyStyles();
  });

  // -------------------------------------------------------------------------
  // Overlay DOM — fixed position, appended to document.body
  // -------------------------------------------------------------------------
  let overlay = null;
  let translatedLine = null;
  let originalLine = null;
  let targetVideo = null;
  let resizeObserver = null;
  let fadeTimer = null;
  let positionRAF = null;

  function createOverlay() {
    if (overlay) return;

    overlay = document.createElement('div');
    overlay.id = 'subtitle-anywhere-overlay';
    overlay.setAttribute('aria-live', 'polite');

    translatedLine = document.createElement('div');
    translatedLine.className = 'sa-translated';

    originalLine = document.createElement('div');
    originalLine.className = 'sa-original';

    overlay.appendChild(translatedLine);
    overlay.appendChild(originalLine);

    document.documentElement.appendChild(overlay);
    applyStyles();
  }

  function applyStyles() {
    if (!overlay) return;

    overlay.style.cssText = `
      position: fixed;
      z-index: 2147483647;
      pointer-events: none;
      text-align: center;
      transition: opacity 0.4s ease;
      padding: 6px 14px;
      border-radius: 6px;
      background: rgba(0, 0, 0, ${settings.bgOpacity});
      font-family: ${settings.fontFamily || 'system-ui, -apple-system, sans-serif'};
      color: ${settings.textColor};
      line-height: 1.4;
      max-width: 80vw;
      display: none;
    `;

    if (translatedLine) {
      translatedLine.style.cssText = `
        font-size: ${settings.fontSize}px;
        font-weight: 500;
        margin-bottom: 2px;
        overflow-wrap: break-word;
        display: -webkit-box;
        -webkit-line-clamp: 1;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-shadow: 0 0 4px rgba(0,0,0,0.9), 0 0 8px rgba(0,0,0,0.7), 1px 1px 2px rgba(0,0,0,0.8);
      `;
    }

    if (originalLine) {
      originalLine.style.cssText = `
        font-size: ${Math.round(settings.fontSize * 0.8)}px;
        opacity: 0.7;
        overflow-wrap: break-word;
        display: -webkit-box;
        -webkit-line-clamp: 1;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-shadow: 0 0 4px rgba(0,0,0,0.9), 0 0 8px rgba(0,0,0,0.7), 1px 1px 2px rgba(0,0,0,0.8);
      `;
    }

    positionOverlay();
  }

  // -------------------------------------------------------------------------
  // Video detection
  // -------------------------------------------------------------------------
  function findBestVideo() {
    const videos = Array.from(document.querySelectorAll('video'));
    if (videos.length === 0) return null;

    const playing = videos.filter(
      (v) => !v.paused && !v.ended && v.readyState > 2
    );
    const candidates = playing.length > 0 ? playing : videos;

    let best = null;
    let bestArea = 0;
    for (const v of candidates) {
      const area = v.clientWidth * v.clientHeight;
      if (area > bestArea) {
        bestArea = area;
        best = v;
      }
    }
    return best;
  }

  function attachToVideo(video) {
    if (!video || video === targetVideo) return;

    detachFromVideo();
    targetVideo = video;
    createOverlay();

    console.log('[Subtitle Anywhere] Attached to video:', video.clientWidth, 'x', video.clientHeight);

    resizeObserver = new ResizeObserver(() => positionOverlay());
    resizeObserver.observe(video);

    positionOverlay();
  }

  function detachFromVideo() {
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    targetVideo = null;
  }

  function positionOverlay() {
    if (!overlay || !targetVideo) return;
    if (overlay.style.display === 'none') return;

    const rect = targetVideo.getBoundingClientRect();
    const bottomOffset = rect.height * (settings.verticalPosition / 100);

    overlay.style.left = `${rect.left + rect.width / 2}px`;
    overlay.style.top = `${rect.bottom - bottomOffset}px`;
    overlay.style.transform = 'translate(-50%, -100%)';
    overlay.style.maxWidth = `${rect.width * 0.85}px`;
  }

  // Continuously reposition while visible (handles scroll, resize, etc.)
  function startPositionLoop() {
    if (positionRAF) return;
    function loop() {
      positionOverlay();
      positionRAF = requestAnimationFrame(loop);
    }
    positionRAF = requestAnimationFrame(loop);
  }

  function stopPositionLoop() {
    if (positionRAF) {
      cancelAnimationFrame(positionRAF);
      positionRAF = null;
    }
  }

  // -------------------------------------------------------------------------
  // Fullscreen handling
  // -------------------------------------------------------------------------
  document.addEventListener('fullscreenchange', () => {
    // Fixed position works in fullscreen too, just reposition
    positionOverlay();
  });

  // -------------------------------------------------------------------------
  // MutationObserver for dynamically added videos
  // -------------------------------------------------------------------------
  const mutationObserver = new MutationObserver(() => {
    if (!targetVideo || !document.contains(targetVideo)) {
      const video = findBestVideo();
      if (video) attachToVideo(video);
    }
  });
  mutationObserver.observe(document.body || document.documentElement, {
    childList: true,
    subtree: true
  });

  // Initial video scan
  const initialVideo = findBestVideo();
  if (initialVideo) {
    attachToVideo(initialVideo);
  }

  // -------------------------------------------------------------------------
  // Subtitle display
  // -------------------------------------------------------------------------
  const recentLines = [];

  function showSubtitle(text, translatedText, isPartial) {
    if (!overlay) {
      const video = findBestVideo();
      if (video) attachToVideo(video);
      if (!overlay) {
        console.warn('[Subtitle Anywhere] No video found, cannot show subtitle');
        return;
      }
    }

    if (fadeTimer) {
      clearTimeout(fadeTimer);
      fadeTimer = null;
    }

    if (translatedText) {
      // With translation: each result takes 2 lines (translated + original),
      // so only show the latest result.
      recentLines.length = 0;
      recentLines.push(text);

      translatedLine.textContent = translatedText;
      translatedLine.style.display = 'block';
      originalLine.textContent = text;
      originalLine.style.display = 'block';
    } else {
      // Without translation: keep last maxLines results
      if (!isPartial) {
        recentLines.push(text);
        while (recentLines.length > settings.maxLines) {
          recentLines.shift();
        }
      }

      const displayText = isPartial
        ? [...recentLines, text].slice(-settings.maxLines).join('\n')
        : recentLines.slice(-settings.maxLines).join('\n');

      translatedLine.style.display = 'none';
      originalLine.textContent = displayText;
      originalLine.style.display = 'block';
      originalLine.style.opacity = isPartial ? '0.7' : '1';
      originalLine.style.fontSize = `${settings.fontSize}px`;
    }

    overlay.style.display = 'block';
    overlay.style.opacity = '1';
    positionOverlay();
    startPositionLoop();

    fadeTimer = setTimeout(() => {
      overlay.style.opacity = '0';
      setTimeout(() => {
        if (overlay) overlay.style.display = 'none';
        stopPositionLoop();
      }, 400);
    }, settings.fadeTimeout * 1000);
  }

  function clearSubtitles() {
    recentLines.length = 0;
    if (overlay) {
      overlay.style.display = 'none';
      translatedLine.textContent = '';
      originalLine.textContent = '';
    }
    stopPositionLoop();
    if (fadeTimer) {
      clearTimeout(fadeTimer);
      fadeTimer = null;
    }
  }

  // -------------------------------------------------------------------------
  // Message listener
  // -------------------------------------------------------------------------
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === 'ping') {
      sendResponse({ ok: true });
      return false;
    }
    if (message.type === 'subtitle') {
      console.log('[Subtitle Anywhere] Received subtitle:', message.text?.substring(0, 60));
      showSubtitle(message.text, message.translatedText, message.isPartial);
      sendResponse({ ok: true });
    } else if (message.type === 'subtitle-stop') {
      clearSubtitles();
      sendResponse({ ok: true });
    }
    return false;
  });

  // -------------------------------------------------------------------------
  // Keep-alive port to background (reconnects on SW restart)
  // -------------------------------------------------------------------------
  function connectToBackground() {
    try {
      const port = chrome.runtime.connect({ name: 'content-script' });
      port.onDisconnect.addListener(() => {
        setTimeout(connectToBackground, 1000);
      });
    } catch {
      clearSubtitles();
      detachFromVideo();
      mutationObserver.disconnect();
      window.__subtitleAnywhereInjected = false;
    }
  }
  connectToBackground();
})();
