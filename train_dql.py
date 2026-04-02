import gymnasium as gym
import logging
import random

import torch


def _setup_logger(
        log_file_path: str | None = None,
        reset_log_file: bool = True,
        logger_name: str = "train_dql"
        ) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    log_format = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    if log_file_path is not None:
        file_mode = "w" if reset_log_file else "a"
        file_handler = logging.FileHandler(log_file_path, mode=file_mode, encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def _select_action(
        env: gym.Env, 
        state: torch.Tensor, 
        policy_nn: torch.nn.Module, 
        device: torch.device,
        epsilon: float = 0.1
        ):
    rand_num = random.random()
    if rand_num < epsilon:
        # Explore: select a random action
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            actions = policy_nn(state.to(device))
        action = torch.argmax(actions).item()

    return action

def _select_batch(
        replay_buffer: list, 
        batch_size=64
        ):
    batch = random.sample(replay_buffer, batch_size)
    return batch

def _update_policy_nn(
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        parameters,
        max_grad_norm: float | None = None,
        ):
    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
    optimizer.step()

def _sync_target_nn(policy_nn: torch.nn.Module, target_nn: torch.nn.Module):
    weights = policy_nn.state_dict()
    target_nn.load_state_dict(weights)

def _add_to_replay_buffer(
        replay_buffer: list, 
        state: torch.Tensor, 
        action: int, 
        reward: float, 
        next_state: torch.Tensor, 
        done: bool, 
        max_buffer_size: int = 10000
        ):
    if len(replay_buffer) >= max_buffer_size:
        replay_buffer.pop(0)
    replay_buffer.append(
        # Add as a dict
        {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }
    )

def _parse_batch(batch: list, device: torch.device):
    states = torch.stack([item["state"] for item in batch]).to(device)
    actions = torch.tensor([item["action"] for item in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([item["reward"] for item in batch], dtype=torch.float32, device=device)
    next_states = torch.stack([item["next_state"] for item in batch]).to(device)
    dones = torch.tensor([item["done"] for item in batch], dtype=torch.float32, device=device)

    return states, actions, rewards, next_states, dones

def train_dql(
        n_episodes: int, 
        target_reward: float, 
        env: gym.Env, 
        policy_nn: torch.nn.Module, 
        target_nn: torch.nn.Module, 
        min_buffer_size: int, 
        batch_size: int, 
        loss_fn: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        train_times_per_step: int, 
        sync_freq: int, 
        avg_reward_window_size: int = 100,
        replay_buffer_capacity: int = 10000,
        gamma: float = 0.99,
        log_every: int = 1,
        device: torch.device | None = None,
        log_file_path: str | None = None,
        reset_log_file: bool = True,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        max_grad_norm: float | None = 10.0,
        ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_nn.to(device)
    target_nn.to(device)
    logger = _setup_logger(log_file_path, reset_log_file)

    # Start training
    replay_buffer = []
    espisode_rewards = []
    # env_frames = []

    # Loop over episodes
    for episode in range(n_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / n_episodes)
        n_train_steps = 0
        acc_reward = 0
        last_loss = None

        while not done:
        # Loop over steps in an episode

            # render the environment and save the frame
            # env_frames.append(env.render())
            # Select an action
            action = _select_action(env, state, policy_nn, device, epsilon)
            # Get observation and reward from the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            done = terminated or truncated
            # Add into replay buffer
            _add_to_replay_buffer(
                replay_buffer, 
                state, 
                action, 
                reward, 
                next_state, 
                done, 
                max_buffer_size=replay_buffer_capacity
            )
            # Update state and accumulated reward
            state = next_state
            acc_reward += reward

            # Train the policy nn
            if len(replay_buffer) > min_buffer_size:
                for _ in range(train_times_per_step):
                    # Sample a mini-batch from the replay buffer
                    batch = _select_batch(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = _parse_batch(batch, device)
                    # Get the current Q-values from the policy nn
                    q_values = policy_nn(states)
                    # Get the next Q-values from the target nn
                    next_q_values = target_nn(next_states)
                    # Compute the loss and update the policy nn
                    loss = loss_fn(
                        q_values, 
                        actions,
                        rewards, 
                        next_q_values, 
                        dones,
                        gamma)
                    #  Update the policy nn parameters using the loss
                    _update_policy_nn(loss, optimizer, policy_nn.parameters(), max_grad_norm)
                    last_loss = loss.item()
                    n_train_steps += 1

                # Sync the target nn with fixed frequency
                if n_train_steps % sync_freq == 0:
                    _sync_target_nn(policy_nn, target_nn)

        # Append the accumulated reward of the episode
        espisode_rewards.append(acc_reward)
        recent_rewards = espisode_rewards[-avg_reward_window_size:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        if log_every > 0 and (episode + 1) % log_every == 0:
            last_loss_text = f"{last_loss:.5f}" if last_loss is not None else "n/a"
            logger.info(
                f"Episode {episode + 1}/{n_episodes} | "
                f"Reward: {acc_reward:.2f} | "
                f"Avg({len(recent_rewards)}): {avg_reward:.2f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Replay: {len(replay_buffer)} | "
                f"Train steps: {n_train_steps} | "
                f"Loss: {last_loss_text}",
            )
        # Check whether avg reward is above target reward over the last 100 episodes, terminate training if so
        if len(espisode_rewards) >= avg_reward_window_size:
            if avg_reward >= target_reward:
                logger.info(
                    f"Episode {episode}: Average reward {avg_reward} over the last {avg_reward_window_size} episodes is above the target reward {target_reward}. Terminate training."
                )
                break