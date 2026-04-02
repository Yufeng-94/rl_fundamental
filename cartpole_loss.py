"""CartPole Loss Functions"""
import torch
def cartpole_loss_fn(
        q_values: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        next_q_values: torch.Tensor, 
        dones: torch.Tensor, 
        gamma: float = 0.99
        ) -> torch.Tensor:
    # Get the Q-values for the taken actions
    action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    # Compute the target Q-values
    target_q_values = rewards + gamma * next_q_values.max(1)[0] * (1 - dones)
    # Compute TD error and normalize by number of samples in the batch.
    td_error = action_q_values - target_q_values.detach()
    n_samples = td_error.shape[0]
    loss = td_error.pow(2).sum() / n_samples
    return loss