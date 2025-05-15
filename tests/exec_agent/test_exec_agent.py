import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import nbformat
from ds_agent.exec_agent import ExecAgent


def test_run_episode_stops_on_sanity(tmp_path):
    # Setup episode dir and notebook
    episode_id = "DS-25-999"
    ep_dir = tmp_path / "episodes" / episode_id
    ep_dir.mkdir(parents=True)
    nb_path = ep_dir / "analysis.ipynb"
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, nb_path)

    agent = ExecAgent()

    with (
        patch("ds_agent.kernel_runner.run_notebook") as mock_run_nb,
        patch("ds_agent.kernel_runner.extract_globals") as mock_extract,
        patch("ds_agent.notebook_editor.add_code") as mock_add_code,
    ):
        # Simulate: first call not passed, second call passed
        mock_extract.side_effect = [{"sanity_passed": False}, {"sanity_passed": True}]
        agent.run_episode(episode_id, max_iter=3)
        # Should call run_notebook twice
        assert mock_run_nb.call_count == 2
        # Should call extract_globals twice
        assert mock_extract.call_count == 2
        # Should call add_code at least once
        assert mock_add_code.called


def test_run_episode_max_iter(tmp_path):
    episode_id = "DS-25-998"
    ep_dir = tmp_path / "episodes" / episode_id
    ep_dir.mkdir(parents=True)
    nb_path = ep_dir / "analysis.ipynb"
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, nb_path)

    agent = ExecAgent()

    with (
        patch("ds_agent.kernel_runner.run_notebook") as mock_run_nb,
        patch("ds_agent.kernel_runner.extract_globals") as mock_extract,
        patch("ds_agent.notebook_editor.add_code") as mock_add_code,
    ):
        # Always return not passed
        mock_extract.return_value = {"sanity_passed": False}
        agent.run_episode(episode_id, max_iter=2)
        # Should call run_notebook twice (max_iter)
        assert mock_run_nb.call_count == 2
        assert mock_extract.call_count == 2


def test_run_episode_max_iter_with_passed(tmp_path):
    episode_id = "DS-25-997"
    ep_dir = tmp_path / "episodes" / episode_id
    ep_dir.mkdir(parents=True)
    nb_path = ep_dir / "analysis.ipynb"
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, nb_path)

    agent = ExecAgent()

    with (
        patch("ds_agent.kernel_runner.run_notebook") as mock_run_nb,
        patch("ds_agent.kernel_runner.extract_globals") as mock_extract,
        patch("ds_agent.notebook_editor.add_code") as mock_add_code,
    ):
        # Simulate: first call not passed, second call passed
        mock_extract.side_effect = [{"sanity_passed": False}, {"sanity_passed": True}]
        agent.run_episode(episode_id, max_iter=3)
        # Should call run_notebook twice
        assert mock_run_nb.call_count == 2
        # Should call extract_globals twice
        assert mock_extract.call_count == 2
        # Should call add_code at least once
        assert mock_add_code.called


def test_run_episode_max_iter_with_passed_and_max_iter(tmp_path):
    episode_id = "DS-25-996"
    ep_dir = tmp_path / "episodes" / episode_id
    ep_dir.mkdir(parents=True)
    nb_path = ep_dir / "analysis.ipynb"
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, nb_path)

    agent = ExecAgent()

    with (
        patch("ds_agent.kernel_runner.run_notebook") as mock_run_nb,
        patch("ds_agent.kernel_runner.extract_globals") as mock_extract,
        patch("ds_agent.notebook_editor.add_code") as mock_add_code,
    ):
        # Simulate: first call not passed, second call passed
        mock_extract.side_effect = [{"sanity_passed": False}, {"sanity_passed": True}]
        agent.run_episode(episode_id, max_iter=5)
        # Should call run_notebook twice
        assert mock_run_nb.call_count == 2
        # Should call extract_globals twice
        assert mock_extract.call_count == 2
        # Should call add_code at least once
        assert mock_add_code.called


def test_run_episode_max_iter_with_passed_and_max_iter_and_passed(tmp_path):
    episode_id = "DS-25-995"
    ep_dir = tmp_path / "episodes" / episode_id
    ep_dir.mkdir(parents=True)
    nb_path = ep_dir / "analysis.ipynb"
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, nb_path)

    agent = ExecAgent()

    with (
        patch("ds_agent.kernel_runner.run_notebook") as mock_run_nb,
        patch("ds_agent.kernel_runner.extract_globals") as mock_extract,
        patch("ds_agent.notebook_editor.add_code") as mock_add_code,
    ):
        # Simulate: first call not passed, second call passed
        mock_extract.side_effect = [{"sanity_passed": False}, {"sanity_passed": True}]
        agent.run_episode(episode_id, max_iter=5)
        # Should call run_notebook twice
        assert mock_run_nb.call_count == 2
        # Should call extract_globals twice
        assert mock_extract.call_count == 2
        # Should call add_code at least once
        assert mock_add_code.called


def test_run_episode_max_iter_with_passed_and_max_iter_and_passed_and_passed(tmp_path):
    episode_id = "DS-25-994"
    ep_dir = tmp_path / "episodes" / episode_id
    ep_dir.mkdir(parents=True)
    nb_path = ep_dir / "analysis.ipynb"
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, nb_path)

    agent = ExecAgent()

    with (
        patch("ds_agent.kernel_runner.run_notebook") as mock_run_nb,
        patch("ds_agent.kernel_runner.extract_globals") as mock_extract,
        patch("ds_agent.notebook_editor.add_code") as mock_add_code,
    ):
        # Simulate: first call not passed, second call passed
        mock_extract.side_effect = [{"sanity_passed": False}, {"sanity_passed": True}]
        agent.run_episode(episode_id, max_iter=5)
        # Should call run_notebook twice
        assert mock_run_nb.call_count == 2
        # Should call extract_globals twice
        assert mock_extract.call_count == 2
        # Should call add_code at least once
        assert mock_add_code.called
