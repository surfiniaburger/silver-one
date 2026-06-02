import tempfile

from agentbeats.checkpoint import (
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)


def test_checkpoint_round_trip_and_validation():
    with tempfile.TemporaryDirectory() as td:
        path = td + "/run/42.json"
        payload = {
            "run_id": "run-a",
            "seed": 42,
            "mode": "record",
            "phase": "judge_complete",
            "models": {"judge": "m1"},
            "generation_config": {"default": {"temperature": 0.2}},
        }
        save_checkpoint(path, payload, clock_now="2026-05-31T00:00:00Z")
        loaded = load_checkpoint(path)
        assert loaded is not None
        assert loaded["schema_version"] == 1
        assert loaded["phase"] == "judge_complete"
        assert loaded["updated_at"] == "2026-05-31T00:00:00Z"

        validate_checkpoint(
            loaded,
            {
                "run_id": "run-a",
                "seed": 42,
                "mode": "record",
                "models": {"judge": "m1"},
                "generation_config": {"default": {"temperature": 0.2}},
            },
        )


def test_checkpoint_validation_rejects_drift():
    checkpoint = {
        "run_id": "run-a",
        "seed": 42,
        "mode": "record",
        "models": {"judge": "m1"},
    }
    try:
        validate_checkpoint(checkpoint, {"run_id": "run-a", "seed": 43})
    except CheckpointError as e:
        assert "seed" in str(e)
    else:
        raise AssertionError("Expected checkpoint validation to reject seed drift")


if __name__ == "__main__":
    test_checkpoint_round_trip_and_validation()
    test_checkpoint_validation_rejects_drift()
    print("ok")
