"""
Sliding window duplex model inference implementation.

This module provides stateless sliding window inference for duplex audio generation.
The system processes pre-recorded user audio codes with a warmup period, then
autoregressively generates assistant responses frame-by-frame without caching.
"""

from typing import Optional, Tuple
import torch
import numpy as np
from transformers import MimiModel, AutoFeatureExtractor

from overfit_trial.model import MachOverfitModel


class SlidingDuplexModelInference:
    """
    Stateless sliding window inference engine for duplex audio generation.

    This class manages the frame-by-frame generation process, extracting user
    frames from a sliding window and generating assistant responses autoregressively.
    """

    def __init__(
        self,
        model: MachOverfitModel,
        num_quantizers: int = 32,
        window_size: int = 2048,
        device: str = "cuda",
    ):
        """
        Initialize the inference engine.

        Args:
            model: Trained MachOverfitModel instance
            num_quantizers: Number of RVQ quantizers (default: 32)
            window_size: Maximum context window size (default: 2048)
            device: Device to run inference on (default: "cuda")
        """
        self.model = model
        self.num_quantizers = num_quantizers
        self.window_size = window_size
        self.device = torch.device(device)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Initialize Mimi model for silence code generation and audio decoding
        self._init_mimi_model()

        # Store generated codes for potential decoding
        self.last_generated_codes = None

    def _init_mimi_model(self):
        """Initialize Mimi model for encoding/decoding."""
        self.mimi_model = MimiModel.from_pretrained("kyutai/mimi", num_quantizers=self.num_quantizers)
        self.mimi_model.to(self.device)
        self.mimi_model.eval()

        # Also initialize the feature extractor for potential audio preprocessing
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    def _generate_silence_codes(self, num_frames: int) -> torch.Tensor:
        """
        Generate silence codes for the specified number of frames.

        Args:
            num_frames: Number of frames of silence to generate

        Returns:
            (num_quantizers, num_frames) tensor of silence codes
        """
        # Calculate duration based on frame rate (12.5 Hz)
        frame_rate = 12.5
        duration_seconds = num_frames / frame_rate
        sample_rate = 24000
        num_samples = int(sample_rate * duration_seconds)

        # Generate silence audio
        silence_audio = torch.zeros(1, 1, num_samples, device=self.device)

        with torch.no_grad():
            silence_encoded = self.mimi_model.encode(silence_audio)
            silence_codes = silence_encoded.audio_codes[0]  # [num_quantizers, frames]

        # Ensure we have exactly the requested number of frames
        if silence_codes.shape[1] > num_frames:
            silence_codes = silence_codes[:, :num_frames]
        elif silence_codes.shape[1] < num_frames:
            # Repeat the last frame if needed
            pad_frames = num_frames - silence_codes.shape[1]
            last_frame = silence_codes[:, -1:].repeat(1, pad_frames)
            silence_codes = torch.cat([silence_codes, last_frame], dim=1)

        return silence_codes

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from a checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"Loaded checkpoint from {checkpoint_path}")

    def decode_to_audio(self, codes: Optional[torch.Tensor] = None, sample_rate: int = 24000) -> np.ndarray:
        """
        Decode audio codes to waveform using Mimi model.

        Args:
            codes: (num_quantizers, frames) tensor of audio codes.
                   If None, uses last generated codes.
            sample_rate: Target sample rate for output audio (default: 24000)

        Returns:
            Audio waveform as numpy array
        """
        if codes is None:
            if self.last_generated_codes is None:
                raise ValueError("No codes to decode. Run generate() first or provide codes.")
            codes = self.last_generated_codes

        # Ensure codes are on the right device and have batch dimension
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)  # Add batch dimension

        codes = codes.to(self.device)

        with torch.no_grad():
            decoded = self.mimi_model.decode(codes)
            audio_values = decoded.audio_values.detach().cpu().squeeze().numpy()

        return audio_values

    def generate(
        self,
        user_codes: torch.Tensor,
        start_frame: int = 0,
        warmup_frames: int = 50,
        num_steps: int = 100,
        temperature: float = 1.0,
        topk: int = 50,
    ) -> torch.Tensor:
        """
        Generate assistant audio codes using sliding window inference.

        Args:
            user_codes: (num_quantizers, total_frames) - complete user audio codes
            start_frame: Frame index to start processing from (default: 0)
            warmup_frames: Number of frames for warmup with silence (default: 50)
            num_steps: Number of frames to generate (default: 100)
            temperature: Sampling temperature (default: 1.0)
            topk: Top-k value for sampling (default: 50)

        Returns:
            (num_quantizers, num_steps) - generated assistant audio codes
        """
        # TODO: Implement the sliding window generation logic
        # This function should:
        # 1. Generate silence codes dynamically based on input shape
        # 2. Process user codes with a sliding window
        # 3. Generate assistant codes frame-by-frame during warmup (silence)
        # 4. After warmup, use model.generate_frame() for autoregressive generation
        # 5. Store generated codes in self.last_generated_codes for potential decoding
        raise NotImplementedError("The generate function is to be implemented")

    def generate_with_audio(self, user_codes: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate assistant codes and immediately decode to audio.

        Args:
            user_codes: User audio codes
            **kwargs: Additional arguments for generate()

        Returns:
            Tuple of (generated codes, audio waveform)
        """
        codes = self.generate(user_codes, **kwargs)
        audio = self.decode_to_audio(codes)
        return codes, audio
