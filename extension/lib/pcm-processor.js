/**
 * AudioWorkletProcessor that converts Float32 audio to Int16 PCM
 * and posts buffered chunks (~600ms at 16kHz) to the main thread.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    // 9600 samples = 600ms at 16kHz
    this._chunkSize = 9600;
  }

  /**
   * Convert a Float32 sample to Int16.
   */
  static floatToInt16(sample) {
    const clamped = Math.max(-1, Math.min(1, sample));
    return clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF;
  }

  process(inputs, _outputs, _parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    const channelData = input[0];
    if (!channelData || channelData.length === 0) {
      return true;
    }

    // Convert Float32 to Int16 and accumulate
    for (let i = 0; i < channelData.length; i++) {
      this._buffer.push(PCMProcessor.floatToInt16(channelData[i]));
    }

    // When we have enough samples, post a chunk
    while (this._buffer.length >= this._chunkSize) {
      const chunk = new Int16Array(this._buffer.splice(0, this._chunkSize));
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
