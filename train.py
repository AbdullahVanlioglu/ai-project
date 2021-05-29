import numpy as np
import torch

from collections import namedtuple, deque
from agent import DRQNAgent
from torch.autograd import Variable

class TrainDRQN:
    def __init__(self, env, agent, args):

        self.env = env
        self.agent = agent
        self.num_episode = args.num_episode
        self.max_iteration = args.max_iteration
        self.epsilon = args.epsilon_init
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.target_update_period = args.target_update_period
        self.manuel_control = False
        self.write_period = args.write_period
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.agent.drqn_net.parameters(), lr = self.lr)

    def Trainer(self, args):
        
        ix = 0
        i_episode = 1 
        scores_window = deque(maxlen=20)
        
        state = self.env.reset()

        if args.load_weight:
            self.agent.drqn_net.load_state_dict(torch.load("weights/single-drqn_net.ckpt"))
            self.agent.targetnet.load_state_dict(torch.load("weights/single-targetnet.ckpt"))

        Transition = namedtuple("Transition", "state action reward next_state terminal")
        
        for eps in range(self.num_episode):
            loss = 0
            episode_reward = 0
            agent_loss = []
            reward_list = []

            hidden_state = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
            state = self.env.reset()
            for _ in range(self.max_iteration):
                ix += 1

                torch_state = DRQNAgent.state_to_torch(state, self.device)
                action, hidden_state = self.agent.e_greedy_policy(torch_state, hidden_state, self.epsilon)

                next_state, reward, done, _ = self.env.step(action)   
                
                trans = Transition(state, action, reward, next_state, done)
                self.agent.buffer.push(trans)
                state = next_state
                episode_reward += reward

                if (ix % 100000 == 0 and ix != 0) or ix == self.frames-1:
                    torch.save(self.agent.valuenet.state_dict(), f"weights/valuenet.ckpt")
                    torch.save(self.agent.targetnet.state_dict(), f"weights/targetnet.ckpt")
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
                    break
                
                self.env.render()
            
    def Evaluate(self, args):
        print("### Test ###")

        self.agent.valuenet.load_state_dict(torch.load("weights/valuenet.ckpt"))
        self.agent.targetnet.load_state_dict(torch.load("weights/targetnet.ckpt"))
        i_episode = 1

        state = self.env.reset()
        episode_reward = 0
        
        self.agent.targetnet.eval()
        self.agent.valuenet.eval()

        for frame in range(1500):
            
            torch_state = DRQNAgent.state_to_torch(state, self.device).unsqueeze(0)

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