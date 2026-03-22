import gymnasium as gym
import random

import torch

def _select_action(
        env: gym.Env, 
        state, 
        policy_nn: torch.nn.Module, 
        epsilon: float = 0.1
        ):
    rand_num = random.random()
    if rand_num < epsilon:
        # Explore: select a random action
        action = env.action_space.sample()
    else:
        actions = policy_nn(state)
        action = torch.argmax(actions).item()

    return action

def _select_batch(
        replay_buffer: list, 
        batch_size=64
        ):
    batch = random.sample(replay_buffer, batch_size)
    return batch

def _update_policy_nn(loss: torch.Tensor, optimizer: torch.optim.Optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def _sync_target_nn(policy_nn: torch.nn.Module, target_nn: torch.nn.Module):
    weights = policy_nn.state_dict()
    target_nn.load_state_dict(weights)

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
        avg_reward_window_size: int = 100
        ):
    # Start training
    replay_buffer = []
    espisode_rewards = []
    env_frames = []

    # Loop over episodes
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        epsilon = max(0.1, 1 - episode / n_episodes)
        n_train_steps = 0
        acc_reward = 0

        while not done:
        # Loop over steps in an episode

            # render the environment and save the frame
            env_frames.append(env.render())
            # Select an action
            action = _select_action(env, state, policy_nn, epsilon)
            # Get observation and reward from the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Add into replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            # Update state and accumulated reward
            state = next_state
            acc_reward += reward

            # Train the policy nn
            if len(replay_buffer) > min_buffer_size:
                for _ in range(train_times_per_step):
                    # Sample a mini-batch from the replay buffer
                    batch = _select_batch(replay_buffer, batch_size)
                    # Compute the loss and update the policy nn
                    loss = loss_fn(batch, policy_nn, target_nn)
                    #  Update the policy nn parameters using the loss
                    _update_policy_nn(loss, optimizer)
                    n_train_steps += 1

                # Sync the target nn with fixed frequency
                if n_train_steps % sync_freq == 0:
                    _sync_target_nn(policy_nn, target_nn)

        # Append the accumulated reward of the episode
        espisode_rewards.append(acc_reward)
        # Check whether avg reward is above target reward over the last 100 episodes, terminate training if so
        if len(espisode_rewards) >= avg_reward_window_size:
            avg_reward = sum(espisode_rewards[-avg_reward_window_size:]) / avg_reward_window_size
            if avg_reward >= target_reward:
                print(f"Episode {episode}: Average reward {avg_reward} over the last {avg_reward_window_size} episodes is above the target reward {target_reward}. Terminate training.")
                break