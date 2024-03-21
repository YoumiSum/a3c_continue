import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import time

import torch
from setproctitle import setproctitle as ptitle

from env.robot import create_env
from models.a3c.agent import Agent
from models.a3c.model import CNN


def test(args, shared_model):
    ptitle("Test Agent")
    gpu_id = args.gpu_ids[-1]
    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env, action_space, num_inputs = create_env(args, train=False)

    reward_sum = 0
    local_model = CNN(num_inputs=num_inputs, hidden_size=args.hidden_size, action_shape=action_space)

    agent = Agent(None, env, args, None)
    agent.gpu_id = gpu_id
    agent.model = local_model
    
    agent.state, *_ = agent.env.reset()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            agent.model = agent.model.cuda()
            agent.state = torch.from_numpy(agent.state).float().cuda()
    else:
        agent.state = torch.from_numpy(agent.state).float()

    agent.model.eval()
    max_score = 0

    try:
        while True:
            if agent.done:
                agent.clear_actions()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        agent.model.load_state_dict(shared_model.state_dict())
                else:
                    agent.model.load_state_dict(shared_model.state_dict())

            agent.action_test()
            reward_sum += agent.reward

            if agent.done:
                print("reward_sum: ", reward_sum)
                if (args.save_max and reward_sum >= max_score) or not args.save_max:
                    if reward_sum >= max_score:
                        max_score = reward_sum
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            state_to_save = agent.model.state_dict()
                            torch.save(
                                state_to_save, f"{args.saved_model}"
                            )
                    else:
                        state_to_save = agent.model.state_dict()
                        torch.save(
                            state_to_save, f"{args.saved_model}"
                        )

                reward_sum = 0
                reward_len = 0
                agent.eps_len = 0
                state, *_ = agent.env.reset()
                time.sleep(10)
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        agent.state = torch.from_numpy(state).float().cuda()
                else:
                    agent.state = torch.from_numpy(state).float()

    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")
    finally:
        print("test agent process finished")


if __name__ == '__main__':
    import argparse

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
    parser.add_argument("--save_max", type=bool, default=True,
                        help="Save model on every test run high score matched or bested", )
    parser.add_argument("--tensorboard_logger", type=bool, default=True,
                        help="Creates tensorboard logger to see graph of model, view model weights and biases, and monitor test agent reward progress")

    parser.add_argument("--max_episode_length", type=int, default=360, help="max step of Agent.")
    parser.add_argument("--num_steps", type=int, default=360, help="number of forward steps in A3C (default: 3600)")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-4)")
    parser.add_argument("--amsgrad", default=True, help="Adam optimizer amsgrad parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards (default: 0.99)")
    parser.add_argument("--tau", type=float, default=1.00, help="parameter for GAE (default: 1.00)")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy loss coefficient (default: 0.01)")

    args = parser.parse_args()

    env, action_space, num_inputs = create_env(args)
    env.env.close()
    shared_model = CNN(num_inputs=num_inputs, hidden_size=args.hidden_size, action_shape=action_space)
    checkpoint = torch.load(args.saved_path, map_location='cpu')
    shared_model.load_state_dict(checkpoint)

    test(args, shared_model)
