import numpy as np
import torch

from collections import namedtuple, deque
from agent import DRQNAgent
from network import DRQN

class TrainDRQN:
    def __init__(self, 
                env,
                agent, 
                num_episode, 
                num_frames, 
                epsilon, 
                epsilon_decay,
                epsilon_min, 
                lr, 
                batch_size, 
                target_update_period,
                write_period
                ):

        self.env = env
        self.agent = agent
        self.num_episode = num_episode
        self.frames = num_frames
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_period = target_update_period
        self.manuel_control = False
        self.write_period = write_period
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.Adam(self.agent.valuenet.parameters(), lr = self.lr)

    def Trainer(self, args):
        
        ix = 0
        loss = 0
        i_episode = 1

        state = self.env.reset()
        episode_reward = 0

        agent_loss = []
        scores_window = deque(maxlen=20)
        reward_list = []

        if args.load_weight:
            self.agent.valuenet.load_state_dict(torch.load("weights/single-valuenet1.ckpt"))
            self.agent.targetnet.load_state_dict(torch.load("weights/single-targetnet1.ckpt"))

        Transition = namedtuple("Transition", "state action reward next_state terminal")
        
        for frame in range(self.frames):
            ix += 1

            if args.obs_state == "vector":
                torch_state = DRQNAgent.state_to_torch(state, self.device).unsqueeze(0)
                torch_state = torch_state.view(-1, 10, 1, 1)
            elif args.obs_state == "image":
                torch_state = DRQNAgent.state_to_torch(state, self.device).unsqueeze(0)

            action = self.agent.e_greedy_policy(torch_state, self.epsilon, self.device)

            next_state, reward, done, _ = self.env.step(action)   
            
            trans1 = Transition(state, action, reward, next_state, done)
            self.agent.buffer.push(trans1)
            state = next_state
            episode_reward += reward

            if (frame % 100000 == 0 and frame != 0) or frame == self.frames-1:
                torch.save(self.agent.valuenet.state_dict(), f"weights/grid-valuenet1.ckpt")
                torch.save(self.agent.targetnet.state_dict(), f"weights/grid-targetnet1.ckpt")
                self.Evaluate(args)
            
            self.agent.valuenet.train()
            self.agent.targetnet.train()
            if (self.agent.buffer.size > self.batch_size):

                if ix % self.target_update_period == 0:
                    self.agent.update(self.device)

                self.optimizer.zero_grad()
                loss = self.agent.loss(self.batch_size, self.device)
                loss.backward()
                self.optimizer.step()

                agent_loss.append(loss.item())
                

            if done:
                scores_window.append(episode_reward)
                reward_list.append(episode_reward)
                state = self.env.reset()
                
                episode_reward = 0

                if self.epsilon >= self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
            
                print("Episode [%d] Frame [%i], Average20 Score = %f, loss = %f, epsilon = %f" % (i_episode, frame, np.mean(scores_window), loss, self.epsilon))
                i_episode += 1
            
            self.env.render()
            
    def Evaluate(self, args):
        print("### Test ###")

        self.agent.valuenet.load_state_dict(torch.load("weights/grid-valuenet1.ckpt"))
        self.agent.targetnet.load_state_dict(torch.load("weights/grid-targetnet1.ckpt"))
        i_episode = 1

        state = self.env.reset()
        episode_reward = 0
        
        self.agent.targetnet.eval()
        self.agent.valuenet.eval()

        for frame in range(1500):
            
            torch_state = DRQNAgent.state_to_torch(state, self.device).unsqueeze(0)

            if args.obs_state == "vector":
                torch_state = torch_state.view(-1, 10, 1, 1)

            action = self.agent.greedy_policy(torch_state, self.device)
            next_state, reward, done, _ = self.env.step(action)   

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                print("Eval Episode [%d] Frame [%i], Eval Score = %f" % (i_episode, frame, episode_reward))
                episode_reward = 0
                i_episode += 1

            self.env.render()