# HomieRL Architecture Overview (files / classes / configs / call graph)

## 1) Entry & Registration Chain
- `HomieRL/legged_gym/legged_gym/scripts/train.py`
  - Entry: `get_args()` → `task_registry.make_env()` → `task_registry.make_alg_runner()` → `runner.learn()`

- `HomieRL/legged_gym/legged_gym/envs/__init__.py`
  - Task registration:
    - `task_registry.register("g1", LeggedRobot, G1RoughCfg(), G1RoughCfgPPO())`

- `HomieRL/legged_gym/legged_gym/utils/task_registry.py`
  - Factory/registry hub:
    - `make_env()` creates env
    - `make_alg_runner()` creates PPO runner

---

## 2) Environment Layer (legged_gym)
### Core class
- `LeggedRobot`
  - File: `HomieRL/legged_gym/legged_gym/envs/base/legged_robot.py`
  - Inherits: `BaseTask`
  - Key functions:
    - `__init__` → init env, buffers, rewards
    - `step` → action merge (lower+upper) + sim + post_physics_step
    - `post_physics_step` → refresh tensors + reward/obs/termination
    - `compute_reward` / `_reward_xxx` → reward computation
    - `compute_observations` → obs assembly (history stacking)
    - `reset_idx` → partial reset
    - `_init_buffers` / `_parse_cfg` → buffer & config init

### Config class
- `G1RoughCfg`
  - File: `HomieRL/legged_gym/legged_gym/envs/g1/g1_29dof_config.py`
  - Inherits: `LeggedRobotCfg`
  - Overrides:
    - `init_state`, `control`, `commands`, `asset`
    - `domain_rand`, `rewards`, `env`, `noise`, `terrain`

- `LeggedRobotCfg`
  - File: `HomieRL/legged_gym/legged_gym/envs/base/legged_robot_config.py`
  - Default structure/fields

---

## 3) Training Layer (rsl_rl)
### Runner
- `HIMOnPolicyRunner`
  - File: `HomieRL/rsl_rl/rsl_rl/runners/him_on_policy_runner.py`
  - Role: rollout + update loop
  - Flow:
    - `__init__` builds Actor/Critic + Algorithm + Storage
    - `learn()`:
      - `alg.act()` → `env.step()` → `alg.process_env_step()`
      - `alg.update()` → parameter update

### Algorithm
- `HIMPPO`
  - File: `HomieRL/rsl_rl/rsl_rl/algorithms/him_ppo.py`
  - Role: PPO core + symmetry augmentation
  - Key functions:
    - `act()` → sample actions + store log_prob/value
    - `process_env_step()` → push transitions to rollout storage
    - `compute_returns()` → GAE/returns
    - `update()` → PPO clip loss + value loss + entropy + symmetry loss
    - `flip_g1_actor_obs/critic_obs/actions()` → symmetry mapping

### Policy / Value Network
- `HIMActorCritic`
  - File: `HomieRL/rsl_rl/rsl_rl/modules/him_actor_critic.py`
  - Role: actor + critic nets
  - Key functions:
    - `act()` → sample actions from Gaussian policy
    - `evaluate()` → V(s)
    - `get_actions_log_prob()` / `entropy`
  - Components:
    - `HIMEstimator` (latent dynamics)
    - `Terrain Encoder` (optional)

### Storage
- `HIMRolloutStorage`
  - File: `HomieRL/rsl_rl/rsl_rl/storage/him_rollout_storage.py`
  - Role: store trajectories + mini-batch generation

---

## 4) Config Inheritance
- `G1RoughCfg` → `LeggedRobotCfg`
- `G1RoughCfgPPO` → `LeggedRobotCfgPPO`
  - File: `HomieRL/legged_gym/legged_gym/envs/g1/g1_29dof_config.py`

---

## 5) Main Call Chain (CLI → PPO)
1. `train.py` entry
2. `task_registry.make_env()` → `LeggedRobot(cfg=G1RoughCfg)`
3. `task_registry.make_alg_runner()` → `HIMOnPolicyRunner(cfg=G1RoughCfgPPO)`
4. `HIMOnPolicyRunner.learn()`:
   - `HIMPPO.act()` → `LeggedRobot.step()` → `HIMPPO.process_env_step()`
   - `HIMPPO.update()` → backprop + optimizer step
5. logging + checkpoint

---

## 6) Key Mapping Between Config and Code
- `G1RoughCfg.rewards.scales` → `LeggedRobot._prepare_reward_function()` → `_reward_xxx()`
- `G1RoughCfg.env` → `_init_buffers()` / `compute_observations()`
- `G1RoughCfg.domain_rand` → `LeggedRobot.step()` / reset randomization
- `HIMActorCritic.act()` → `HIMPPO.act()` → runner rollout
- `HIMActorCritic.evaluate()` → `HIMPPO.update()` value loss
