import argparse
import gym
import numpy as np
import torch

from agent import DRQNAgent
from train import TrainDRQN

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    env = gym.make("Seaquest-v0")

    num_actions = env.action_space.n
    state_shape = env.observation_space.shape
    state_dtype = env.observation_space.dtype
    lstm_hidden_shape = (1,1,16)

    agent = DRQNAgent(capacity=args.buffer_capacity,
                        n_action=num_actions,
                        state_shape=state_shape,
                        state_dtype=state_dtype,
						hidden_shape=lstm_hidden_shape,
                        gamma=args.gamma,
                        batch_size=args.batch_size,
                        device=device
                    )


    model = TrainDRQN(env=env, agent=agent, args=args)
    model.Trainer(args)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='DQN')
	parser.add_argument("--envname", type=str,
						default="Pygame-v0",
						help="Name of the environment")
	parser.add_argument("--max-iteration", type=int, default=300,
						help="Number of training iterations")
	parser.add_argument("--start-update", type=int, default=100,
						help="Number of iterations until starting to update")
	parser.add_argument("--num-episode", type=int, default=30000,
						help="Maximum length of an episode before termination")
	parser.add_argument("--batch-size", type=int, default=16,
						help="Batch size of each update in training")
	parser.add_argument("--gamma", type=float, default=0.99,
						help="Discount Factor")
	parser.add_argument("--lr", type=float, default=1e-3,
						help="Discount Factor")
	parser.add_argument("--target-update-period", type=int, default=2000,
						help="Target network updating period")
	parser.add_argument("--buffer-capacity", type=int, default=1500,
						help="Replay buffer capacity")
	parser.add_argument("--epsilon-init", type=float, default=1,
						help="Initial value of the epsilon")
	parser.add_argument("--epsilon-min", type=float, default=0.1,
						help="Minimum value of the epsilon")
	parser.add_argument("--epsilon-decay", type=float, default=0.99996,
						help="Epsilon decay rate for exponential decaying")
	parser.add_argument("--epsilon-rang", type=float, default=None,
						help="Epsilon decaying range for linear decay")
	parser.add_argument("--clip-grad", action="store_true",
						help="Gradient Clip between -1 and 1. Default: No")
	parser.add_argument("--eval-period", type=int, default=500,
						help="Evaluation period in terms of iteration")
	parser.add_argument("--eval-episode", type=int, default=5,
						help="Number of episodes to evaluate")
	parser.add_argument("--save-model", action="store_true",
						help="If given most successful models so far will be saved")
	parser.add_argument("--model-dir", type=str, default="models/",
						help="Directory to save models")
	parser.add_argument("--write-period", type=int, default=1000,
						help="Writer period")
	parser.add_argument("--test", type=int, default=0,
						help="Test or Train Model")
	parser.add_argument("--load-weight", type=int, default=0,
						help="Transfer learning for Train")
	
	
	args = parser.parse_args()
	main(args)
