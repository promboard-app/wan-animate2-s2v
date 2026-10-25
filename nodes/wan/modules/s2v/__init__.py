# Vendored from Wan-Video/Wan2.2 (Apache-2.0) wan/modules/s2v/__init__.py @ main (2026-10-25). PromBoard changes are marked 'PromBoard:'.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from .audio_encoder import AudioEncoder
from .model_s2v import WanModel_S2V

__all__ = ['WanModel_S2V', 'AudioEncoder']
