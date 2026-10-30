"""The dashboard writes the record's CSV; plots are derived elsewhere."""

from utils.logger import Dashboard


def test_dashboard_writes_only_the_csv(tmp_path):
    dashboard = Dashboard(
        env_name="CartPole-v1",
        agent_name="PPOAgent",
        total_iterations=2,
        save_dir=str(tmp_path),
        modes=[],
    )

    dashboard.update(0, 1.0, loss=0.5)
    dashboard.close()

    assert (tmp_path / "training_log.csv").exists()
    assert not (tmp_path / "training_log.png").exists()
