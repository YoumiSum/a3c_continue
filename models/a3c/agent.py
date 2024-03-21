from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from env.robot import Robot


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env: Robot = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.hidden_size = args.hidden_size
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def action_train(self):
        value, mean, p, self.hx, self.cx = self.model(
            self.state, self.hx, self.cx
        )
        # 智能体的四个轮子的输出范围都是从 [-100, 100]，这里用-1到1，是为了减少输出的y值变化太大，action会在最后乘上100
        mean = torch.clamp(mean, -1.0, 1.0)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                mean = mean.cuda()
                p = p.cuda()

        # 对p取对数变换
        var = F.softplus(p)
        conv = torch.diag_embed(var)
        # 以mean、var的正态分布，抽取随机action
        dist = MultivariateNormal(mean, conv)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        action_logprob = dist.log_prob(action)

        self.log_probs.append(action_logprob)

        # 计算熵
        entropy = dist.entropy()
        self.entropies.append(entropy)

        # 获取新的state，reward，done，info
        if self.gpu_id >= 0:
            action = action.cpu()

        act = action.numpy()[0]
        act *= 100
        state, reward, self.done, self.info = self.env.step(act)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = torch.from_numpy(state).float().cuda()
        else:
            self.state = torch.from_numpy(state).float()

        self.eps_len += 1
        self.values.append(value)
        self.rewards.append(reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = torch.zeros(1, self.hidden_size).cuda()
                        self.hx = torch.zeros(1, self.hidden_size).cuda()
                else:
                    self.cx = torch.zeros(1, self.hidden_size)
                    self.hx = torch.zeros(1, self.hidden_size)

            value, mean, p, self.hx, self.cx = self.model(
                self.state, self.hx, self.cx
            )
            # 智能体的四个轮子的输出范围都是从 [-100, 100]，这里用-1到1，是为了减少输出的y值变化太大，action会在最后乘上100
            mean = torch.clamp(mean, -1.0, 1.0)
            # 对p取对数变换
            var = F.softplus(p)
            conv = torch.diag_embed(var)
            # 以mean、var的正态分布，抽取随机action
            dist = MultivariateNormal(mean, conv)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            if self.gpu_id >= 0:
                action = action.cpu()

            act = action.numpy()[0]
            act *= 100
        state, self.reward, self.done, self.info = self.env.step(act)

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = torch.from_numpy(state).float().cuda()
        else:
            self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self

