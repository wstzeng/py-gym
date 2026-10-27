import torch

from agent.components.buffer import TrajectoryBuffer


def make_step(state_value: float = 0.0) -> dict:
    return {
        "state": torch.full((1, 4), state_value),
        "action": torch.tensor([1]),
        "log_prob": torch.tensor([0.0]),
        "value": torch.tensor([0.0]),
    }


def test_single_step_keeps_batch_dimensions():
    buffer = TrajectoryBuffer()
    buffer.store(make_step(), reward=1.0, done=True)

    data = buffer.get_data()

    assert data["states"].shape == (1, 4)
    assert data["actions"].shape == (1,)
    assert data["rewards"].shape == (1,)


def test_multiple_steps_stack_along_batch_dimension():
    buffer = TrajectoryBuffer()
    buffer.store(make_step(0.0), reward=1.0, done=False)
    buffer.store(make_step(1.0), reward=2.0, done=True)

    data = buffer.get_data()

    assert data["states"].shape == (2, 4)
    assert data["states"][1, 0].item() == 1.0
    assert data["actions"].shape == (2,)


def test_empty_buffer_returns_empty_dict():
    assert TrajectoryBuffer().get_data() == {}
