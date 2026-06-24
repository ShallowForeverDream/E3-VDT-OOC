from __future__ import annotations
from pathlib import Path
from typing import Optional
class VDTAdapterNotReady(RuntimeError): pass
class VDTAdapter:
    def __init__(self, checkpoint_path: Optional[str]=None):
        self.checkpoint_path=Path(checkpoint_path) if checkpoint_path else None
        if self.checkpoint_path and not self.checkpoint_path.exists(): raise FileNotFoundError(self.checkpoint_path)
    def predict(self, *args, **kwargs):
        raise VDTAdapterNotReady("VDT checkpoint adapter is not implemented yet. Use heuristic mode for demo or implement after strict reproduction.")
