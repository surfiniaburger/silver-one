import os
import tempfile
import pytest
from agentbeats.checkpoint import save_checkpoint, load_checkpoint, CheckpointError

def test_checkpoint_path_validation():
    """Verify that save_checkpoint and load_checkpoint canonicalize paths and prevent directory traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sub_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(sub_dir, exist_ok=True)
        ckpt_path = os.path.join(sub_dir, "test.json")

        # Valid save and load with base_dir scoping
        payload = {"status": "ok"}
        save_checkpoint(ckpt_path, payload, base_dir=tmpdir)
        loaded = load_checkpoint(ckpt_path, base_dir=tmpdir)
        assert loaded is not None
        assert loaded["status"] == "ok"

        # Traversal attempt escaping base_dir
        traversal_path = os.path.join(sub_dir, "../../../etc/passwd")
        with pytest.raises(CheckpointError, match="Path traversal detected"):
            save_checkpoint(traversal_path, payload, base_dir=sub_dir)

        with pytest.raises(CheckpointError, match="Path traversal detected"):
            load_checkpoint(traversal_path, base_dir=sub_dir)
