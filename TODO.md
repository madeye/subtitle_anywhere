# TODO

- [ ] Cache extracted audio instead of deleting it. Store WAV files in
  `~/.cache/subtitle_anywhere/audio/` keyed by video file path + size + mtime
  so the cache auto-invalidates when the source changes. `--keep-audio` still
  copies the file beside the output as before.
