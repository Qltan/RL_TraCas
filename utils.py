import os
import torch
import numpy as np
from abc import ABC
from torch.utils.data.dataset import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', device)


class TrafficDataset(Dataset, ABC):
    def __init__(self, state_dimension, cur_path, temp_file_path, cur_region, K, start_timestamp, end_timestamp):

        super(TrafficDataset, self).__init__()
        # the maximum edge number of the cascade pattern graph
        self.K = K
        # the data path
        self.path = cur_path
        # the path used to store the mid data result
        self.temp_path = temp_file_path
        # the region where the current data from
        self.region = cur_region
        # the start timestamp of the data samples in each day
        self.s_ts = start_timestamp
        # the end timestamp of the data samples in each day
        self.e_ts = end_timestamp
        """topology information"""
        # edges_set: the set contains the id of all edges in the current region
        # edge_adj_dict: the adjoining edge set of each edge in the edges_set
        # edge_action_dict: map the old edge id to the new one, which starts from 0
        # action_edge_dict: a dictionary with inverse mapping relation to edge_action_dict
        self.edges_set, self.edge_adj_dict, self.edge_action_dict, self.action_edge_dict = self.gen_edgeInfo()
        """congestion information"""
        self.Cascades = self.gen_casInfo()
        # the total number of edges in the current region
        self.relation_Num = len(self.edges_set)
        # the largest id in the edge set
        self.max_edge = max(self.edges_set)

        self.init_state = torch.zeros((1, state_dimension, K)).float().to(device)

    def gen_edgeInfo(self):
        print('Begin extract road information ...')
        """edges information"""
        # edge_nodes: all edges' number in the RN
        # edge_relations: all possible propagation relation between edges and corresponding numbers
        # edge_adj: all adjacent edges of each edge
        if not os.path.exists(self.temp_path + self.region):
            os.makedirs(self.temp_path + self.region)
        try:
            edges_set = np.load(self.temp_path + self.region + '/edges.npy', allow_pickle=True)
            edge_adj_dict = np.load(self.temp_path + self.region + '/edge_adj.npy', allow_pickle=True).item()
            edge_action_dict = np.load(self.temp_path + self.region + '/edge_action.npy', allow_pickle=True).item()
            action_edge_dict = np.load(self.temp_path + self.region + '/action_edge.npy', allow_pickle=True).item()
        except FileNotFoundError:
            edges_set = set()
            edge_adj_dict = {}
            edge_action_dict = {}
            action_edge_dict = {}
            cur_dir = self.path + self.region + '/Road/'
            road_data = np.genfromtxt(cur_dir + 'network_selected.csv', delimiter=' ', encoding='UTF-8', dtype=int)
            for i in range(road_data.shape[0]):
                s_edge = road_data[i][0]
                e_edge = road_data[i][1]
                edges_set.add(s_edge)
                edges_set.add(e_edge)
                if s_edge in edge_adj_dict.keys():
                    edge_adj_dict[s_edge].append(e_edge)
                else:
                    edge_adj_dict[s_edge] = [e_edge]
            edges_set = list(edges_set)

            for i in range(len(edges_set)):
                edge_action_dict[edges_set[i]] = i
                action_edge_dict[i] = edges_set[i]
            np.save(self.temp_path + self.region + '/edges.npy', edges_set)
            np.save(self.temp_path + self.region + '/edge_adj.npy', edge_adj_dict)
            np.save(self.temp_path + self.region + '/edge_action.npy', edge_action_dict)
            np.save(self.temp_path + self.region + '/action_edge.npy', action_edge_dict)
        print('Road information extract done!')
        return edges_set, edge_adj_dict, edge_action_dict, action_edge_dict

    def gen_casInfo(self):
        print('Begin extract cascades ...')
        """congestions information"""
        if not os.path.exists(self.temp_path + self.region):
            os.makedirs(self.temp_path + self.region)
        try:
            Cascades = np.load(self.temp_path + self.region + '/cascades_' + str(self.s_ts) + '_' +
                               str(self.e_ts) + '.npy', allow_pickle=True)
        except FileNotFoundError:
            congestion_dir = self.path + self.region + '/Con_selected/'
            cascade_list = os.listdir(congestion_dir)
            Cascades = []
            for day in cascade_list:
                # the cascade of each day
                cascade_data = np.genfromtxt(congestion_dir + day, delimiter=' ', encoding='UTF-8', dtype=int)
                cascade = {}
                for i in range(len(cascade_data)):
                    road_id = cascade_data[i][0]  # road segment id
                    road_time = cascade_data[i][1]  # start time of congestion
                    # make sure congestion occurs within the time span [ts, te]
                    if self.s_ts <= road_time <= self.e_ts:
                        # keep the first timestamp of congestion on the road segment
                        if road_id not in cascade.keys():
                            cascade[road_id] = road_time
                        # keep the last timestamp of congestion on the road segment
                        # cascade[road_id] = [time_start, time_end]
                Cascades.append(cascade)
            np.save(self.temp_path + self.region + '/cascades_' + str(self.s_ts) + '_' + str(self.e_ts) +
                    '.npy', Cascades)
        print('Cascade information extract done!')
        return Cascades

    def gen_actions(self, cur_edge, used_set):
        # generate optional segments at the current state
        action_list = []
        action_adj1 = []
        action_adj2 = []
        action_adj3 = []
        # one-hop neighbors
        if cur_edge is not None and cur_edge in self.edge_adj_dict.keys():
            action_adj1 += self.edge_adj_dict[cur_edge]
            for edge in self.edge_adj_dict[cur_edge]:
                # two-hop neighbors
                if edge in self.edge_adj_dict.keys():
                    action_adj2 += self.edge_adj_dict[edge]
                    for edge1 in self.edge_adj_dict[edge]:
                        # three-hop neighbors
                        if edge1 in self.edge_adj_dict.keys():
                            action_adj3 += self.edge_adj_dict[edge1]
            action_list = action_adj1 + action_adj2 + action_adj3
        else:
            for action in self.edges_set:
                action_list.append(action)
        action_list = list(set(action_list))
        for a in action_list:
            if a in used_set:
                action_list.remove(a)
        action_list = [self.edge_action_dict[e] for e in action_list]
        return action_list

    def action_mask(self, actions):
        mask_initial = torch.zeros(1, len(self.edges_set), device=device).long()  # 1 : bacth_size
        mask = mask_initial.index_fill_(1, actions, 1).float()  # the first 1: dim , the second 1: value
        return mask

    def evaluate(self, g_edges, start_day, end_day, gama):
        congestion = self.Cascades[start_day: end_day]
        score_k = 0
        for (sour, tar) in g_edges:
            hit = 0
            m = 0
            for con in congestion:
                if sour in con.keys():
                    m = m + 1
                    if tar in con.keys() and con[sour] <= con[tar] <= con[sour] + gama:
                        hit = hit + 1
            if m != 0:
                prob_se = hit / m
            else:
                prob_se = 0.0
            score_k = score_k + prob_se
        # return score_k
        return score_k / len(g_edges)

    def evaluate1(self, e, start_day, end_day, gama):
        congestion = self.Cascades[start_day: end_day]
        hit = 0
        m = 0
        for con in congestion:
            if e[0] in con.keys():
                m = m + 1
                if e[1] in con.keys() and con[e[0]] <= con[e[1]] <= con[e[0]] + gama:
                    hit = hit + 1
        if m != 0:
            prob_se = hit / m
        else:
            prob_se = 0.0
        return prob_se
