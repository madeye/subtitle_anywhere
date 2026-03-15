from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 10095
    device: str = "cuda"  # or "cpu", "mps"
    model_id: str = "iic/SenseVoiceSmall"
    vad_model: str = "fsmn-vad"
    chunk_interval_ms: int = 600  # how often client sends audio
    inference_interval_s: float = 1.0  # how often to run partial inference
    max_buffer_s: float = 3.0  # max audio buffer before forced inference + translation
