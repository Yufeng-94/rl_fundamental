import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from q_function_nn import QFunctionNN
import torch
from train_dql import train_dql
from cartpole_loss import cartpole_loss_fn

def main(trained_model_path: str = "cartpole_policy_nn.pth"):
    # Init a built-in cartpole environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Config
    num_episodes = 500
    # max_steps_per_episode = 200
    input_dim = 4
    hidden_dim = 128
    output_dim = 2
    train_times_per_step = 10
    gamma = 0.99
    replay_buffer_capacity = 1000
    min_buffer_size = 300
    batch_size = 64
    target_reward = 400
    sync_freq = 50
    avg_reward_window_size = 100
    log_file_path = "cartpole_training.log"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Logging training to: {log_file_path}")

    # Init policy NN and target NN
    policy_nn = QFunctionNN(state_dim=input_dim, action_dim=output_dim, hidden_dim=hidden_dim)
    # copy weights from policy NN to target NN
    target_nn = QFunctionNN(state_dim=input_dim, action_dim=output_dim, hidden_dim=hidden_dim)
    target_nn.load_state_dict(policy_nn.state_dict())

    # if pre-trained model exists, load it, if not, start training from scratch
    try:
        policy_nn.load_state_dict(torch.load(trained_model_path))
        target_nn.load_state_dict(policy_nn.state_dict())
        print(f"Loaded pre-trained model from {trained_model_path}")
    except FileNotFoundError:
        print(f"No pre-trained model found at {trained_model_path}, starting training from scratch")

    # Create an optimizer
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=1e-3)

    train_dql(n_episodes=num_episodes, 
          target_reward=target_reward,
          env=env,
          policy_nn=policy_nn,
          target_nn=target_nn,
          min_buffer_size=min_buffer_size,
          batch_size=batch_size,
          loss_fn=cartpole_loss_fn,
          optimizer=optimizer,
          train_times_per_step=train_times_per_step,
          sync_freq=sync_freq,
          avg_reward_window_size=avg_reward_window_size,
          replay_buffer_capacity=replay_buffer_capacity,
          device=device,
          log_file_path=log_file_path,
          epsilon_start=0.25,
          max_grad_norm=5.0,
        )
    
    return policy_nn

if __name__ == "__main__":
    trained_model_path = "cartpole_policy_nn_v2.pth"
    train_policy_nn = main(trained_model_path=trained_model_path)

    # save the trained policy NN
    torch.save(train_policy_nn.state_dict(), trained_model_path)
    print(f"Saved trained model to {trained_model_path}")