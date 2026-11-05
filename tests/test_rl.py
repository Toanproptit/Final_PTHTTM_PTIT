import hashlib
import numpy as np
import pytest
from diabetes_demo.paths import ROOT
from legacy.agent02_experiment import Config, ECommerceEnvironment, QLearningAgent, RuleBasedAgent, train_rl_against_rule, evaluate_pair

def test_python_source_is_exact_copy_of_tex():
    assert hashlib.sha256((ROOT / "agent02_code.tex").read_bytes()).digest() == hashlib.sha256((ROOT / "legacy" / "agent02_experiment.py").read_bytes()).digest()

def test_terminal_q_target_excludes_future_value():
    agent = QLearningAgent(alpha=.1, gamma=.9)
    agent.q[1, 0] = [100, 200]
    agent.learn((0, 0), 1, 10, (1, 0), terminal=True)
    assert agent.q[0, 0, 1] == pytest.approx(1)
    agent.learn((0, 0), 0, 10, (1, 0), terminal=False)
    assert agent.q[0, 0, 0] == pytest.approx(19)

def test_evaluation_freezes_q_and_training_is_reproducible():
    config = Config(episodes=15, steps_per_episode=5, evaluation_episodes=8)
    agent, rows = train_rl_against_rule(config, seed=12)
    another, other_rows = train_rl_against_rule(config, seed=12)
    np.testing.assert_array_equal(agent.q, another.q)
    assert rows == other_rows
    frozen = agent.q.copy()
    summary, evaluation = evaluate_pair(config, agent, seed=456)
    np.testing.assert_array_equal(frozen, agent.q)
    assert len(evaluation) == 8
    assert sum(row["discount_actions"] + row["no_discount_actions"] for row in summary) == 2 * 8 * 5

def test_environment_state_tracks_other_agents_previous_action():
    env = ECommerceEnvironment(Config())
    env.reset()
    states, _, _ = env.step([1, 0])
    assert states[0][1] == 0 and states[1][1] == 1
    with pytest.raises(ValueError):
        env.step([1, 2])
    with pytest.raises(ValueError):
        env.step([1])

def test_fixed_rules():
    agent = RuleBasedAgent()
    assert [agent.act((0, 0)), agent.act((0, 1)), agent.act((1, 0)), agent.act((1, 1))] == [0, 0, 0, 1]
