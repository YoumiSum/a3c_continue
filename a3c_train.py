import os
import argparse
import datetime
import pickle

import numpy as np
from torch import nn

import shutil
import time

from setproctitle import setproctitle as ptitle

import torch
import torch.multiprocessing as mp

from a3c_test import test
from env.robot import create_env
from models.a3c.agent import Agent
from models.a3c.model import CNN, STRG
from optim import SharedAdam, SharedRMSprop

parser = argparse.ArgumentParser(description="A3C")
parser.add_argument("--render", type=bool, default=True, help="show visualized window")
parser.add_argument("--log_path", type=str, default="a3c_result/tensorboard")
parser.add_argument("--saved_model", type=str, default="a3c_result/trained_models/a3c.pkl")
parser.add_argument("--saved_path", type=str, default="a3c_result/temp")
parser.add_argument("--load", type=bool, default=True, help="Load weight from previous trained stage")
parser.add_argument("--skip_frame", type=int, default=4, help="how many frame will merge to one.")
parser.add_argument("--jobs", type=int, default=8, help="how many cpu will use (-1 means all.)")
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument("--gpu_ids", type=int, default=[0], nargs="+", help="GPUs to use [-1 CPU only] (default: -1)")
parser.add_argument("--save_max", type=bool, default=True, help="Save model on every test run high score matched or bested",)
parser.add_argument("--tensorboard_logger", type=bool, default=True, help="Creates tensorboard logger to see graph of model, view model weights and biases, and monitor test agent reward progress")

parser.add_argument("--num_steps", type=int, default=600, help="number of forward steps in A3C (default: 3600)")
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)")
parser.add_argument("--amsgrad", default=True, help="Adam optimizer amsgrad parameter")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards (default: 0.99)")
parser.add_argument("--tau", type=float, default=1.00, help="parameter for GAE (default: 1.00)")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy loss coefficient (default: 0.01)")

args = parser.parse_args()

def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()

def train(job_id, shared_model, optimizer, args):
    ptitle(f"Train Agent: {job_id}")
    gpu_id = args.gpu_ids[job_id % len(args.gpu_ids)]
    torch.manual_seed(args.seed + job_id)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + job_id)
    env, action_space, num_inputs = create_env(args)
    env.seed(args.seed + job_id)

    local_model = CNN(num_inputs=num_inputs, hidden_size=args.hidden_size, action_shape=action_space)

    agent = Agent(None, env, args, None)
    agent.model = local_model
    agent.gpu_id = gpu_id
    agent.state, *_ = agent.env.reset()

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            agent.state = torch.from_numpy(agent.state).float().cuda()
            agent.model = agent.model.cuda()
    else:
        agent.state = torch.from_numpy(agent.state).float()

    if job_id == 1 and args.tensorboard_logger:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f"{args.log_path}")

    ##################################################################################
    num_step_path = os.path.join(args.saved_path, "num_step.pickle")
    if args.load and os.path.exists(num_step_path):
        with open(num_step_path, "rb") as f:
            num_step = pickle.load(f)
    else:
        num_step = 0
    #################################################################################

    try:
        while True:
            agent.clear_actions()

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    agent.model.load_state_dict(shared_model.state_dict())
            else:
                agent.model.load_state_dict(shared_model.state_dict())

            if agent.done:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        agent.cx = torch.zeros(1, args.hidden_size).cuda()
                        agent.hx = torch.zeros(1, args.hidden_size).cuda()
                else:
                    agent.cx = torch.zeros(1, args.hidden_size)
                    agent.hx = torch.zeros(1, args.hidden_size)
            else:
                agent.cx = agent.cx.data
                agent.hx = agent.hx.data

            # 收集训练数据
            for step in range(args.num_steps):
                agent.action_train()

                if agent.done:
                    break

            if agent.done:
                agent.eps_len = 0
                state, *_ = agent.env.reset()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        agent.state = torch.from_numpy(state).float().cuda()
                else:
                    agent.state = torch.from_numpy(state).float()

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    R = torch.zeros(1, 1).cuda()
                    gae = torch.zeros(1, 1).cuda()
            else:
                R = torch.zeros(1, 1)
                gae = torch.zeros(1, 1)
            if not agent.done:
                state = agent.state
                value, *_ = agent.model(state, agent.hx, agent.cx)
                R = value.detach()
            agent.values.append(R)
            actor_loss = 0
            value_loss = 0
            entropy_loss = 0
            for i in reversed(range(len(agent.rewards))):
                R = args.gamma * R + agent.rewards[i]

                advantage = R - agent.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = agent.rewards[i] + args.gamma * agent.values[i + 1].data - agent.values[i].data
                gae = gae * args.gamma * args.tau + delta_t
                actor_loss = actor_loss + agent.log_probs[i] * gae

                entropy_loss = entropy_loss + agent.entropies[i]

            # print(sum(agent.rewards))

            agent.model.zero_grad()
            total_loss = -actor_loss + 0.5*value_loss - args.entropy_coef*entropy_loss
            total_loss.backward()
            # nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1)
            ensure_shared_grads(agent.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()

            ###############################################################################
            if job_id == 1 and args.tensorboard_logger and agent.done:
                writer.add_scalar("loss", total_loss, num_step)
                writer.add_scalar("actor_loss", actor_loss, num_step)
                writer.add_scalar("value_loss", value_loss, num_step)
                writer.add_scalar("entropy_loss", entropy_loss, num_step)
                writer.add_scalar("rewards", np.sum(agent.rewards), num_step)

                for name, weight in agent.model.named_parameters():
                    writer.add_histogram(name, weight, num_step)

                with open(num_step_path, "wb") as f:
                    pickle.dump(num_step, f)

            num_step += 1
            ###############################################################################


    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")
    finally:
        print(f"train agent {job_id} process finished")
        if job_id == 1 and args.tensorboard_logger:
            writer.close()


if __name__ == '__main__':
    # 文件夹创建
    if not args.load:
        if os.path.isdir(args.log_path):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H.%M.%S")
            args.log_path += f"-{formatted_time}"
            if os.path.isdir(args.log_path):
                shutil.rmtree(args.log_path)
    os.makedirs(args.log_path, exist_ok=True)
    os.chmod(args.log_path, 0o666)

    save_dir = os.path.dirname(args.saved_model)
    if not args.load:
        if os.path.isdir(save_dir):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H.%M.%S")
            save_dir += f"-{formatted_time}"
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.chmod(save_dir, 0o666)

    if not args.load and os.path.exists(args.saved_path):
        shutil.rmtree(args.saved_path)
    os.makedirs(args.saved_path, exist_ok=True)
    os.chmod(save_dir, 0o666)

    if args.gpu_ids != [-1]:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")

    env, action_space, num_inputs = create_env(args)
    env.env.close()
    shared_model = CNN(num_inputs=num_inputs, hidden_size=args.hidden_size, action_shape=action_space)
    strg_model = STRG(num_inputs=num_inputs, hidden_size=args.hidden_size, action_shape=action_space)
    if args.load and os.path.exists(args.saved_model):
        if args.gpu_ids[0] >= 0:
            checkpoint = torch.load(args.saved_model)
        else:
            checkpoint = torch.load(args.saved_model, map_location='cpu')
        shared_model.load_state_dict(checkpoint)
    strg_model_dict = strg_model.state_dict()
    checkpoint = torch.load('a3c_result/trained_models/a3c_strg.pkl')
    strg_model.load_state_dict(checkpoint)
    strg_model_dict = strg_model.state_dict()
    filtered_state_dict = {k: v for k, v in strg_model_dict.items() if k.startswith("actor_mean") or k.startswith("actor_var")}

    shared_model_dict = shared_model.state_dict()
    shared_model_dict.update(filtered_state_dict)
    shared_model.load_state_dict(shared_model_dict)

    for name, param in shared_model.named_parameters():
        if name.startswith("actor_mean") or name.startswith("actor_var"):
            param.requires_grad = False

    shared_model.share_memory()

    optimizer = SharedAdam(
        shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad
    )
    optimizer.share_memory()

    if args.jobs <= 0:
        args.jobs = os.cpu_count()

    # train(1, shared_model, optimizer, args)
    test(args, shared_model)

    # processes = []
    #
    # p = mp.Process(target=test, args=(args, shared_model))
    # p.start()
    # time.sleep(0.001)
    # processes.append(p)
    # for job_id in range(1, args.jobs):
    #     p = mp.Process(target=train, args=(job_id, shared_model, optimizer, args))
    #     p.start()
    #     time.sleep(0.001)
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()
    #     time.sleep(0.001)
