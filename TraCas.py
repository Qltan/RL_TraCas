import os
import time
import torch
import argparse
import matplotlib
from torch import optim
from Graph import Graph
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import TrafficDataset
from models import RLActor, RLCritic

matplotlib.use('Agg')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', device)


def epoch_process(actor, train_data, init_edge, K):
    """empty pattern graph"""
    G = Graph()
    group_logP = []

    state = train_data.init_state
    used_segments = set()
    actions_ = train_data.gen_actions(init_edge, used_segments)

    init_choice = True

    while G.edgeNum < K:

        actions = torch.tensor(actions_).to(device)

        if actions.size()[0] > 0:

            mask = train_data.action_mask(actions)
            action_prob = actor(state)
            probs = F.softmax(action_prob + mask * 100, dim=1)

            if actor.training:
                # if init_choice:
                """sampling by probability distribution"""
                m = torch.distributions.Categorical(probs)
                cur_action = m.sample()
                logp = m.log_prob(cur_action)
            else:
                prob, cur_action = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            if init_choice:
                start_edge = train_data.action_edge_dict[cur_action.item()]
                used_segments.add(start_edge)
                actions_ = train_data.gen_actions(start_edge, used_segments)
            else:
                end_edge = train_data.action_edge_dict[cur_action.item()]
                if G.addEdge(start_edge, end_edge, 0.0):
                    group_logP.append(logp.unsqueeze(1))
                    state = state.clone()
                    state[0, 0, G.edgeNum - 1] = start_edge / train_data.max_edge
                    state[0, 1, G.edgeNum - 1] = end_edge / train_data.max_edge
                    start_edge = end_edge
                    used_segments.add(start_edge)
                    actions_ = train_data.gen_actions(start_edge, used_segments)
                else:
                    actions_ = actions.detach().tolist()
                    if cur_action.item() in actions_:
                        actions_.remove(cur_action.item())
            init_choice = False
        else:
            used_segments.clear()
            actions_ = train_data.gen_actions(init_edge, used_segments)
            init_choice = True

    group_logP = torch.cat(group_logP, dim=1)  # (batch_size, seq_len)
    return G.edges, group_logP


def training(actor, critic, result_p, train_data, epoch_max, actor_lr, critic_lr, max_grad_norm, train_size, init_edge,
             K, checkpoint, time_window, **kwargs):
    # path for saving model
    if checkpoint:
        now = checkpoint
    else:
        now = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime()) + '-' + str(K) + '-' + str(time_window)

    ###############
    # path for saving results
    save_dir = os.path.join(result_p, now)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ###############

    # training information
    output_path = os.path.join(save_dir, 'output.txt')
    output_file = open(output_path, 'a+')

    # optimizer
    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    average_reward_list, actor_loss_list, critic_loss_list = [], [], []
    best_reward = 0

    state = train_data.init_state

    # training
    for epoch in range(epoch_max):
        actor.train()
        critic.train()

        start = time.time()

        total_reward = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for example_id in range(train_size):  # this loop accumulates a batch
            # generate one cascading pattern graph and calculate the generating probability
            pattern_graph, pattern_logP = epoch_process(actor, train_data, init_edge, K)

            reward = torch.tensor(train_data.evaluate(pattern_graph, 0, 21, time_window)).to(device)

            critic_est = critic(state).view(-1)

            ac_error = (reward - critic_est)

            # per_actor_loss = -ac_error.detach() * pattern_logP.mean()
            per_actor_loss = -ac_error.detach() * pattern_logP.sum(dim=1)
            per_critic_loss = ac_error ** 2

            total_actor_loss = total_actor_loss + per_actor_loss
            total_critic_loss = total_critic_loss + per_critic_loss
            total_reward = total_reward + reward

        # calculate the average reward and loss
        actor_loss = total_actor_loss / train_size
        critic_loss = total_critic_loss / train_size
        average_reward = total_reward / train_size

        average_reward_list.append(average_reward.half().item())
        actor_loss_list.append(actor_loss.half().item())
        critic_loss_list.append(critic_loss.half().item())

        # update
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_optim.step()

        # output the information of the current episode
        end = time.time()
        cost_time = end - start
        print('epoch %d, average_reward: %2.3f, actor_loss: %2.4f,  critic_loss: %2.4f, cost_time: %2.4fs'
              % (epoch, average_reward.item(), actor_loss.item(), critic_loss.item(), cost_time))
        output_file.write(
            'epoch %d, average_reward: %2.3f, actor_loss: %2.4f,  critic_loss: %2.4f, cost_time: %2.4fs' % (
                epoch, average_reward.item(), actor_loss.item(), critic_loss.item(), cost_time) + '\n')
        output_file.flush()
        torch.cuda.empty_cache()  # reduce memory

        ########
        # finish an update with a batch

        # Save best model parameters
        average_reward_value = average_reward.item()
        if average_reward_value > best_reward:
            best_reward = average_reward_value

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

    output_file.close()

    # loss
    records_path = os.path.join(save_dir, 'Reward_Aloss_Closs.txt')
    write_file = open(records_path, 'w')
    for i in range(epoch_max):
        per_average_reward_record = average_reward_list[i]
        per__actor_loss_record = actor_loss_list[i]
        per_critic_loss_record = critic_loss_list[i]

        to_write = str(per_average_reward_record) + '\t' + str(per__actor_loss_record) + '\t' + str(
            per_critic_loss_record) + '\n'

        write_file.write(to_write)
    write_file.close()

    # plot
    picture_path = os.path.join(save_dir, 'reward.png')
    # plt.subplot(2, 1, 1)
    plt.figure('Figure1')
    plt.cla()
    plt.plot(average_reward_list, '-', label="reward", color='orangered')
    # plt.title('Reward vs. Epochs')
    plt.ylabel('Reward')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig(picture_path, dpi=800)

    # plt.subplot(2, 1, 2)
    picture_path1 = os.path.join(save_dir, 'loss.png')
    plt.cla()
    plt.figure('Figure2')
    plt.plot(critic_loss_list, 'o-', label="critic_loss")
    # plt.xlabel('Critic_loss vs. Epochs')
    plt.ylabel('Critic loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig(picture_path1, dpi=800)


def pattern_infer(input_args):
    MATE_SIZE = 2

    if input_args.init_edge:
        init_edge = input_args.init_edge
    else:
        init_edge = None

    train_data = TrafficDataset(MATE_SIZE,
                                input_args.data_path,
                                input_args.temp_data_path,
                                input_args.region,
                                input_args.K,
                                start_timestamp=42,
                                end_timestamp=60)
    # Actor
    actor = RLActor(MATE_SIZE, input_args.hidden_size, train_data.relation_Num, input_args.K).to(device)

    # Critic
    critic = RLCritic(MATE_SIZE, args.hidden_size, input_args.K).to(device)

    kwargs = vars(args)  # dict

    kwargs['train_data'] = train_data
    result_p = input_args.result_path + input_args.region + '/'

    if not args.test:  # train
        training(actor, critic, result_p, **kwargs)
    else:
        if args.checkpoint:  # test: give model_solution
            actor_path = os.path.join(result_p + args.checkpoint, 'actor.pt')
            actor.load_state_dict(torch.load(actor_path, device))

            critic_path = os.path.join(result_p + args.checkpoint, 'critic.pt')
            critic.load_state_dict(torch.load(critic_path, device))

            actor.eval()
            critic.eval()

            pattern, logP = epoch_process(actor, train_data, init_edge, input_args.K)
            p_score = train_data.evaluate(pattern, 21, 28, input_args.time_window)

            model_solution_path = os.path.join(result_p + args.checkpoint)
            if not os.path.exists(model_solution_path):
                os.makedirs(model_solution_path)

            f = open(model_solution_path + '/cascade_pattern_' + str(input_args.K) + '_' + str(
                input_args.time_window) + '.txt', 'w')

            to_write = str(p_score) + ' '
            for i in pattern:
                to_write = to_write + str(i) + ','

            to_write1 = to_write.rstrip(',')
            print(to_write1)
            f.write(to_write1)
            f.close()
        else:
            print("### Please provide model parameters! ###")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')

    parser.add_argument('--K', default=50, type=int)

    parser.add_argument('--init_edge', default=None, type=int)

    parser.add_argument('--time_window', default=8, type=int)

    parser.add_argument('--checkpoint', default=None)

    parser.add_argument('--test', action='store_true', default=False)

    parser.add_argument('--region', default='lianhu', type=str)

    parser.add_argument('--hidden', dest='hidden_size', default=32, type=int)

    parser.add_argument('--max_grad_norm', default=2., type=float)
    # learning rate of the actor model
    parser.add_argument('--actor_lr', default=5e-5, type=float)
    # learning rate of the critic model
    parser.add_argument('--critic_lr', default=5e-5, type=float)
    # similar to batch size
    parser.add_argument('--train_size', default=64, type=int)
    # the number of total epoch
    parser.add_argument('--epoch_max', default=2000, type=int)

    parser.add_argument('--data_path', default='./data/', type=str)

    parser.add_argument('--result_path', default='./result/', type=str)

    parser.add_argument('--temp_data_path', default='./temp_data/', type=str)

    args = parser.parse_args()

    regions = ['yanta', 'beilin', 'lianhu']
    for region in regions:
        args.region = region
        pattern_infer(args)
#     pattern_infer(args)
