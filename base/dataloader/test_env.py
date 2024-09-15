


# https://github.com/chaiwonkim/CLab

# from google.colab import drive
# drive.mount('/content/drive')

# %cd '/content/drive/MyDrive/SNU/CLabProject'

# pip install networkx matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from IPython.display import clear_output
import time
import warnings

from tqdm.auto import tqdm

# os.getcwd()
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------

random_state = 1
rng = np.random.RandomState(random_state)      # (RandomState) ★ Set_Params

n_nodes = 50                        # (Node수) ★ Set_Params

total_period = 7*24*60              # (전체 Horizon) ★ Set_Params
# T = 24*60                           # (주기) ★ Set_Params

# example = True
example = False

# visualize = True
visualize = False

# simulation =True
simulation = False

####################################################################################
def apply_function_to_matrix(matrix, func, *args, **kwargs):
    vectorized_func = np.vectorize(lambda x: func(x, *args, **kwargs))
    return vectorized_func(matrix)

def matrix_rank(matrix, ascending=True, axis=0):
    """
    ascending: True (low 1 -> high ...), False (high 1 -> low ...)
    axis: 0 (row-wise), 1 (column-wise)
    """
    if ascending:
        return (np.argsort(np.argsort(matrix, axis=axis), axis=axis) + 1)
    else:
        return (np.argsort(np.argsort(-matrix, axis=axis), axis=axis) + 1)

# (Create Basic GraphSetting) #####################################################
class GenerateNodeMap():
    def __init__(self, n_nodes, distance_scale=1, random_state=None):
        self.n_nodes = n_nodes
        self.centers = np.zeros((n_nodes,2))
        self.covs = np.ones(n_nodes)
        self.adj_base_matrix = np.zeros((n_nodes, n_nodes))
        self.adj_matrix = np.zeros((n_nodes, n_nodes))
        self.adj_dist_mat = np.zeros((n_nodes, n_nodes))

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.distance_scale = distance_scale
    
    # (Version 3.3)
    def generate_nodes(self, n_nodes):
        centers_x = self.rng.uniform(0,1, size=n_nodes) 
        centers_y = self.rng.uniform(0,1, size=n_nodes) 
        centers = np.stack([centers_x, centers_y]).T
        return centers

    # (Version 3.3)
    def make_adjacent_matrix(self, centers, scale=50):
        adj_matrix = np.zeros((len(centers), len(centers)))
        adj_dist_mat = np.zeros((len(centers), len(centers)))

        # node기반 adjacent matrix 구성
        # scale = 50
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                distance = np.sqrt((centers[i][0] - centers[j][0])**2 +
                                (centers[i][1] - centers[j][1])**2)
                # adj_matrix[i][j] = distance * self.rng.uniform(1, 3) * scale
                # adj_matrix[j][i] = distance * self.rng.uniform(1, 3) * scale
                adj_matrix[i][j] = distance * self.rng.normal(scale, 10/10)
                adj_matrix[j][i] = distance * self.rng.normal(scale, 10/10)
                
                adj_dist_mat[i][j] = distance
                adj_dist_mat[j][i] = distance
        return (adj_matrix, adj_dist_mat)

    # (Version 3.3)
    def verify_closeness(self, adj_dist_mat, n_nodes, criteria_scale=0.25):
        adj_dist_mat_copy = adj_dist_mat.copy()
        np.fill_diagonal(adj_dist_mat_copy, np.inf)
        d = np.sqrt(2) / np.sqrt(n_nodes)

        near_points = (adj_dist_mat_copy < d*criteria_scale).sum(1).astype(bool)
        # print(near_points.sum())
        return near_points

    # (Version 3.3)
    def create_node(self, node_scale=50, cov_knn=3):
        # (Version 3.3) -------------------------------------------------------------
        n_nodes = self.n_nodes
        centers = self.generate_nodes(n_nodes)
        adj_base_matrix, adj_dist_mat = self.make_adjacent_matrix(centers, scale=node_scale)
        v_closeness = self.verify_closeness(adj_dist_mat, n_nodes)

        centers = centers[~v_closeness,:]

        it = 0
        while(len(centers) < n_nodes):
            # print(it, end=" ")
            new_centers = self.generate_nodes(n_nodes - len(centers))
            centers = np.append(centers, new_centers, axis=0)
            adj_base_matrix, adj_dist_mat = self.make_adjacent_matrix(centers, scale=node_scale)
            v_closeness = self.verify_closeness(adj_dist_mat, n_nodes)
            centers = centers[~v_closeness,:]
            it +=1
            if it >= 100:
                raise Exception("need to be lower 'criteria_scale' in 'verify_closeness' function.")
        # print(it)

        self.centers = centers[np.argsort((centers**2).sum(1))]        # sorted
        self.adj_base_matrix, self.adj_dist_mat = self.make_adjacent_matrix(self.centers, scale=node_scale)

        # Covariance of Nodes
        adj_dist_mat_copy = self.adj_dist_mat.copy()
        np.fill_diagonal(adj_dist_mat_copy, np.inf)
        adj_dist_mat_rank = matrix_rank(adj_dist_mat_copy, axis=1)

        adj_nearnode = (adj_dist_mat_copy * (adj_dist_mat_rank <= cov_knn))
        np.fill_diagonal(adj_nearnode, 0)

        self.covs = adj_nearnode.mean(1) * (n_nodes/10)*2 * (3/cov_knn)**(1.3)
        # -----------------------------------------------------------------------------

    # (Version 3.3)
    def create_connect(self, connect_scale=0.13):
        if self.centers.sum() == 0:
            print("nodes are not created.")
        else:
            # (Version 3.3) -------------------------------------------------------------
            # assign probability for connection
            def connect_function(x, connect_scale=0):
                if x < 5:
                    x_con = (1/(x - connect_scale))
                elif x < 10:
                    x_con = (1/x)**(1.5 - connect_scale) 
                elif x < 20:
                    x_con = (1/x)**(2 - connect_scale) 
                else:
                    x_con = (1/x)**(5 - connect_scale) 
                return x_con
            
            n_nodes = self.n_nodes
            connect_scale = connect_scale

            # 모든 노드가 연결될때까지 재구성
            while(True):
                shape_mat = self.adj_base_matrix.shape
                adj_sym_mat = (self.adj_base_matrix + self.adj_base_matrix.T)/2
                np.fill_diagonal(adj_sym_mat, np.inf)
                adj_sym_rank_mat = (np.argsort(np.argsort(adj_sym_mat, axis=1), axis=1) + 1)

                adj_sym_rank_rev_mat = apply_function_to_matrix(adj_sym_rank_mat, connect_function, connect_scale=connect_scale)

                adj_sym_noise_mat = adj_sym_rank_rev_mat + self.rng.normal(0,0.01, size=shape_mat)
                np.fill_diagonal(adj_sym_noise_mat,0)

                adj_conn_prob_mat = adj_sym_noise_mat.copy()
                adj_sym_conn_prob_mat = (adj_conn_prob_mat.T + adj_conn_prob_mat)/2
                adj_sym_conn_prob_mat[adj_sym_conn_prob_mat<0] = 0
                adj_sym_conn_prob_mat[np.arange(shape_mat[0]),np.argmax(adj_sym_conn_prob_mat,axis=1)] = 1      # 최소 1개의 node끼리는 connect
                adj_sym_conn_prob_mat[adj_sym_conn_prob_mat>1] = 1

                connect_filter = (self.rng.binomial(1, adj_sym_conn_prob_mat)).astype(bool)
                connect_filter_sym = connect_filter.T + connect_filter

                if self.is_connected(connect_filter_sym):   # 모든 노드가 연결될 경우
                    print("connect ratio : ", connect_filter_sym.sum() / len(connect_filter_sym.ravel()) )
                    adj_matrix_con = self.adj_base_matrix * connect_filter_sym    # connect filtering
                    self.adj_matrix = adj_matrix_con
                    break
                else:
                    if connect_scale < 0.99:
                        connect_scale += 0.01
            # -----------------------------------------------------------------------------

    def is_connected(self, matrix, start=0):
        n = len(matrix)
        visited = np.zeros(n, dtype=bool)

        def dfs(node):
            stack = [node]
            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    neighbors = np.where(matrix[current] == 1)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            stack.append(neighbor)
        dfs(start)

        # 모든 노드가 방문되었는지 확인
        return np.all(visited)

    def predict(self, x):
        # # multivariate guassian base
        # class_prob = np.array([gaussian_instance.pdf(point) for gaussian_instance in self.gaussian_objects])
        # maxprob_class = np.argmax(class_prob)       # class

        # distance base
        class_prob = 1/ (np.sqrt((self.centers - x)**2).sum(1) / self.covs)
        class_argmax = np.argmax(class_prob)       # class
        return class_prob, class_argmax
    
    def nearest_distance(self, x, distance_scale=None):
        class_prob, class_argmax = self.predict(x)
        distance_scale = self.distance_scale if distance_scale is None else distance_scale
        return np.sqrt(((self.centers[class_argmax] - np.array(x))**2).sum()) * distance_scale

    def __call__(self, x):
        return self.nearest_distance(x)




# visualize_graph -------------------------------------------------------------------------
def visualize_graph(centers, adjacent_matrix=None, path=[], path_distance=[], distance=None,
            covs=None,  point=None, class_prob=None, class_argmax=None, point_from=None, weight_direction='both', weight_vis_base_mat=None,
            title=None, vmax=3, return_plot=False):
    """ 
        nodes : nodes coordinate
        weight_direction : 'both', 'forward', 'backward'
    """
    # plot
    fig, ax = plt.subplots(figsize=(20,15))

    if title is not None:
        ax.set_title(title, fontsize=25)
    else:
        if len(path) > 0:
            if distance is None:
                ax.set_title(f"The Shortest-Path from {path[0]} to {path[-1]} \n {path}", fontsize=25)
            else:
                if len(path_distance)>0:
                    sum_path_distance = np.sum(path_distance[:len(path)])
                    ax.set_title(f"The Shortest-Path from {path[0]} to {path[-1]} \n (Dist: {sum_path_distance:.1f}) {path}", fontsize=25)
                else:
                    ax.set_title(f"The Shortest-Path from {path[0]} to {path[-1]} \n (Dist: {distance:.1f}) {path}", fontsize=25)
        else:
            ax.set_title("Graph Visualization from Adjacency Matrix", fontsize=25)


    if weight_vis_base_mat is not None:
        line_cmap = plt.get_cmap("coolwarm")
        # line_norm = plt.Normalize(1, 3)
        line_norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=line_cmap, norm=line_norm)
        plt.colorbar(sm, ax=ax)

    # 엣지 그리기
    for i in range(len(centers)):
        # draw edge ---------------------------------------------------------------------------
        if adjacent_matrix is not None:
            for j in range(i+1, len(adjacent_matrix)):
                if adjacent_matrix[i][j] > 0:
                    u = [centers[i][0], centers[j][0]]
                    v = [centers[i][1], centers[j][1]]

                    if weight_vis_base_mat is None:
                        ax.plot(u, v, color='gray', alpha=0.2)
                    else:
                        if weight_direction == 'both':
                            line_weight = (adjacent_matrix[i][j] + adjacent_matrix[j][i])/(weight_vis_base_mat[i][j] + weight_vis_base_mat[j][i])
                        elif weight_direction == 'forward':
                            line_weight = adjacent_matrix[i][j] / weight_vis_base_mat[i][j]
                        elif weight_direction == 'backward':
                            line_weight = adjacent_matrix[j][i] / weight_vis_base_mat[j][i]
                        
                        ax.plot(np.array(u)-0.005, np.array(v)-0.005, 
                            color=line_cmap(line_norm( line_weight )), alpha=0.5)
                        # ax.plot(np.array(u)-0.005, np.array(v)-0.005, 
                        #         color=line_cmap(line_norm(adjacent_matrix[i][j])), alpha=0.5)
                        # ax.plot(np.array(u)+0.005, np.array(v)+0.005,
                        #         color=line_cmap(line_norm(adjacent_matrix[j][i])), alpha=0.5)

        # draw node ---------------------------------------------------------------------------
        node = i

        if (point is not None) and (class_prob is not None) and (class_argmax is not None):
            # node_color = 'blue' if i == class_argmax else 'steelblue'
            node_linewidth = 1 if i == class_argmax else 0.3
            
            ax.scatter(centers[i][0], centers[i][1], label=f'Node {node}', color='skyblue', s=500, edgecolor='blue',
                        alpha=node_linewidth)
            if covs is not None:
                circle = plt.Circle(centers[i], covs[i]*1.5, color='steelblue', fill=False, 
                                    alpha=max(class_prob[i]/class_prob.max(), 0.15), linewidth=node_linewidth)
                ax.add_patch(circle)
        else:
            ax.scatter(centers[i][0], centers[i][1], label=f'Node {node}', color='skyblue', s=500, edgecolor='steelblue')
            if covs is not None:
                circle = plt.Circle(centers[i], covs[i]*1.5, color='steelblue', fill=False, alpha=0.15)
                ax.add_patch(circle)
        
        node = i
        ax.text(centers[i][0], centers[i][1], f' {node}', fontsize=13, 
            verticalalignment='center', horizontalalignment='center'
            , fontweight='bold'
            )
        # ---------------------------------------------------------------------------------------


    # shortest path
    for p_i in range(len(path)):
        u_p = path[p_i]
        ax.scatter(centers[u_p][0], centers[u_p][1], s=500, facecolor='none', edgecolor='red', linewidths=3)

        if p_i < len(path)-1:
            v_p = path[p_i+1]
            # plt.plot([centers[u_p][0], centers[v_p][0]], [centers[u_p][1], centers[v_p][1]], color='red')
            # 화살표 추가
            ax.annotate('', xy=[centers[v_p][0], centers[v_p][1]], xytext=[centers[u_p][0], centers[u_p][1]],
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->,head_width=1,head_length=1.5', 
                            lw=2
                            # , headwidth=10, headlength=15
                            )
                )
            mid_x = (centers[u_p][0] + centers[v_p][0]) / 2
            mid_y = (centers[u_p][1] + centers[v_p][1]) / 2

            annot = path_distance[p_i] if len(path_distance) > 0 else adjacent_matrix[u_p][v_p]
            ax.text(mid_x, mid_y, f'{annot:.1f}', fontsize=13,
                    color='darkred', backgroundcolor='none',
                    horizontalalignment='center', verticalalignment='center')

    if point is not None:
        ax.scatter(point[0], point[1], facecolor='red', s=200, marker='*')
        if point_from is not None:
            ax.plot([point_from[0], point[0]], [point_from[1], point[1]], color='red', ls='--')
    ax.set_xlabel('lat')
    ax.set_ylabel('lng')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    if return_plot:
        plt.close()
        return fig
    else:
        plt.show()




# visualize node_map (centers, covs)
if visualize:
    node_map = GenerateNodeMap(n_nodes, random_state=random_state)
    # create node ------------------------------------------------------------------
    node_map.create_node(node_scale=50, cov_knn=3)

    # visualize_graph(centers=node_map.centers)
    visualize_graph(centers=node_map.centers, covs=node_map.covs)


# visualize node_map with some point
if visualize:
    point = rng.uniform(size=2)
    class_prob, class_argmax = node_map.predict(point)
    visualize_graph(centers=node_map.centers, covs=node_map.covs, 
            point=point, class_prob=class_prob, class_argmax=class_argmax)



# visualize node_map with connection
if visualize:
    # create connection ------------------------------------------------------------------
    node_map.create_connect(connect_scale=0)

    visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)
    # visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix, weight_vis_base_mat=node_map.adj_matrix)


# visualize node_map with connection and some point
if example:
    point = rng.uniform(size=2)
    class_prob, class_argmax = node_map.predict(point)
    visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix, covs=node_map.covs, 
        point=point, class_prob=class_prob, class_argmax=class_argmax)



#################################################################################
# (Create Periodic Setting) #####################################################

# Periodic 주기함수 --------------------------------------------------------------
class Periodic():
    def __init__(self, periodic_f=None, **kwargs):
        self.periodic_f = periodic_f
        self.kwargs = kwargs
       
    def generate(self, x, **kwargs):
        if len(kwargs) == 0:
            return self.periodic_f(x, **self.kwargs)
        else:
            return self.periodic_f(x, **kwargs)
    
    def __call__(self, x):
        return self.generate(x)


def periodic_cos(x, **kwargs):                      # (주기함수 function) ★ Set_Params
    periods = np.cos(kwargs['a'] * (x + kwargs['b']))    # -1 ~ 1
    periods = (periods +1)/2    # 0~1
    return periods
    # return kwargs['c'] * np.cos(kwargs['a'] * (x + kwargs['b'])) + kwargs['d']

# (Version 2.2 Update)
def periodic_improve(x, **kwargs):                      # (주기함수 function) ★ Set_Params
    time_list = []
    # 월~금
    for aw in [0,1,2,3,4, 7,8,9,10,11]:
        periods1_b1 = np.cos((np.pi*2)/(16*60) * (x - (8*60) - (aw*24*60) ))    # -1 ~ 1
        periods1_b2 = np.cos((np.pi*2)/(2*60) * (x - (9*60) - (aw*24*60) ))    # -1 ~ 1
        periods1_b4 = np.sin((np.pi*2)/(4*60) * (x - (11*60) - (aw*24*60) ))    # -1 ~ 1
        periods1_b6 = np.cos((np.pi*2)/(2*60) * (x - (13*60) - (aw*24*60) ))    # -1 ~ 1
        periods1_b8 = np.sin((np.pi*2)/(8*60) * (x - (16*60) - (aw*24*60) ))    # -1 ~ 1
        periods1_b9 = np.cos((np.pi*2)/(12*60) * (x - (6*60) - (aw*24*60) ))    # -1 ~ 1

        periods1_1 = (periods1_b1 + 1)/2 * (x >= aw*24*60) * (x < (aw+1)*24*60) * ((x+15*60) % (24*60) >= 15*60)
        day_alpha = 0.2
        periods1_2 = ((1-day_alpha-0.05)*(periods1_b2 + 1)/2 + day_alpha) * (x >= aw*24*60) * (x < (aw+1)*24*60) * ((x+14*60) % (24*60) >= 23*60)
        periods1_3 = day_alpha * (x >= aw*24*60) * (x < (aw+1)*24*60) * ((x+13*60) % (24*60) >= 23*60)
        lunch_alpha = 0.4
        periods1_4 = ((1-day_alpha-lunch_alpha)*(periods1_b4) + day_alpha) * (x >= aw*24*60) * (x < (aw+1)*24*60) * ((x+12*60) % (24*60) >= 23*60)
        periods1_5 = (day_alpha+lunch_alpha) * (x >= aw*24*60) * (x < (aw+1)*24*60) * ((x+11*60) % (24*60) >= 23*60)
        periods1_6 = ((1-day_alpha-lunch_alpha)*(periods1_b6+1)/2 + day_alpha) * (x >= aw*24*60) * (x < (aw+1)*24*60) * ((x+10*60) % (24*60) >= 23*60)
        day_alpha = 0.2
        periods1_7 = (day_alpha) * (x >= aw*24*60) * (x < (aw+1)*24*60) * ((x+8*60) % (24*60) >= 22*60)
        periods1_8 = ((1-day_alpha)*(periods1_b8) + day_alpha) * (x >= aw*24*60) * (x < (aw+1)*24*60) * ((x+6*60) % (24*60) >= 22*60)
        periods1_9 = (periods1_b9 + 1)/2 * (x >= aw*24*60) * (x < (aw+1)*24*60) * (x % (24*60) >= 18*60)

        time_list.append(periods1_1)    # 0~9
        time_list.append(periods1_2)    # 9~10
        time_list.append(periods1_3)    # 10~11
        time_list.append(periods1_4)    # 11~12
        time_list.append(periods1_5)    # 12~13
        time_list.append(periods1_6)    # 13~14
        time_list.append(periods1_7)    # 13~16
        time_list.append(periods1_8)    # 16~18
        time_list.append(periods1_9)    # 18~24
        

    # 토요일
    # 0시 ~ 10시
    periods2_1 = np.sin( (np.pi*2)/(20*60)* (x -(5*24*60 + 5*60) ))
    periods2_1 = (periods2_1 +1)/2    # 0~1
    periods2_1 = periods2_1 * (x >= 5*24*60) * (x < 5*24*60 + 10*60)

    # 10시 ~ 14시
    periods2_2 = 1 * (x >= 5*24*60 + 10*60) * (x < 5*24*60 + 14*60)
    
    # 14시 ~ 일요일 2시
    periods2_3 = np.cos( (np.pi*2)/(24*60)* (x -(5*24*60 + 14*60) ))
    periods2_3 = (periods2_3 +1)/2    # 0~1
    periods2_3 = periods2_3 * (x >= 5*24*60 + 14*60) * (x < 6*24*60 + 2*60)

    time_list.append(periods2_1)
    time_list.append(periods2_2)
    time_list.append(periods2_3)

    # 일요일
    # 2시 ~ 14시
    periods3_1 = np.sin( (np.pi*2)/(24*60)* (x -(6*24*60 + 8*60) ))
    periods3_1 = (periods3_1 +1)/2 *0.7    # 0~0.7
    periods3_1 = periods3_1 * (x >= 6*24*60 + 2*60) * (x < 6*24*60 + 14*60)

    # 14시 ~ 18시
    periods3_2 = 0.7 * (x >= 6*24*60 + 14*60) * (x < 6*24*60 + 18*60)

    # 18시 ~ 24시
    periods3_3 = np.cos( (np.pi*2)/(12*60)* (x -(6*24*60 + 18*60) ))
    periods3_3 = (periods3_3 +1)/2 *0.7   # 0~0.7
    periods3_3 = periods3_3 * (x >= 6*24*60 + 18*60) * (x < 7*24*60)
    
    time_list.append(periods3_1)
    time_list.append(periods3_2)
    time_list.append(periods3_3)

    return np.stack(time_list).sum(0)


# Example Visualize ----------------------------------------------------------------------
def visualize_periodic(obs, oracle=None, title=None,  return_plot=True):
    x = np.arange(len(obs))

    fig = plt.figure(figsize=(20,3))
    if title is not None:
        plt.title(title, fontsize=20)
    plt.plot(x, obs, alpha=0.5, color='steelblue')
    if oracle is not None:
        plt.plot(x, oracle, alpha=0.1, color='orange')
        oracle_copy = oracle
    else:
        oracle_copy = np.zeros_like(obs)
    for (x_p, y_p, y_t) in zip(x, obs, oracle_copy):
        if  x_p %(24*60) == 0:
            plt.axvline(x_p, color='black')
        if x_p % 60 == 0:
            plt.scatter(x_p, y_p, color='steelblue')
            if oracle is None:
                plt.text(x_p, y_p, x_p // (60) - x_p//(24*60)*24)
            
            if oracle is not None:
                plt.scatter(x_p, y_t, color='orange', alpha=0.3)
    # plt.axhline(0)
    if return_plot is True:
        plt.close()
        return fig
    else:
        plt.show()



# visualize temporal variance
if visualize:
    # Period
    x = np.arange(0,7*24*60)
    # x = np.arange(0,3*24*60)
    p = Periodic(periodic_improve)

    # plot
    plt.figure(figsize=(20,3))
    plt.plot(x, p(x), alpha=0.5)
    for (x_p,y_p) in zip(x, p(x)):
        if  x_p %(24*60) == 0:
            plt.axvline(x_p, color='black')
        if x_p % 60 == 0:
            plt.scatter(x_p, y_p, color='steelblue')
            plt.text(x_p, y_p, x_p // (60) - x_p//(24*60)*24)
    # plt.axhline(0)
    plt.show()



# multivariate periodic weights ------------------------------------------------------------
from scipy.stats import multivariate_normal

# def multivariate_gaussian(n_nodes, mean, cov):
#     x_pos = np.linspace(-3, 3, n_nodes)
#     y_pos = np.linspace(-3, 3, n_nodes)
#     X_mesh, Y_mesh = np.meshgrid(x_pos, y_pos)
    
#     mv_gen = multivariate_normal(mean, cov)     # Multivariate Gaussian

#     mv_pos = np.dstack((X_mesh, Y_mesh))
#     mv_rv = mv_gen.pdf(mv_pos)
#     mv_rv = ((mv_rv - mv_rv.min()) / (mv_rv.max() - mv_rv.min()))   # 0~1 normalize
#     return mv_rv


# # Example Visualize --------------------------------
# if visualize:
#     # (multivaraite gaussian visualization)
#     mv_weight_mat = multivariate_gaussian(n_nodes, np.zeros(2), np.identity(2)*5)
#     fig = plt.figure()
#     plt.imshow(mv_weight_mat, cmap='coolwarm')
#     plt.colorbar()
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.plot_surface(X_mesh, Y_mesh, mv_rv, cmap='coolwarm')
#     plt.show()


# --------------------------------------------------------------------------------------------
# multivariate periodic weights ------------------------------------------------------------
from scipy.stats import multivariate_normal

# (Version 4.1 Update)
class SyntheticGaussian():
    def __init__(self, means=None, covs=None):
        self.gaussians = []
        self.means = means
        self.covs = covs

    def synthesize(self, means=None, covs=None):
        means = self.means if means is None else means
        covs = self.covs if covs is None else covs

        for m,c in zip(means, covs):
            self.gaussians.append( multivariate_normal(m, c) )
    
    def predict(self, x, normalize=False):
        pred = np.zeros( len(x) )
        for g in self.gaussians:
            pred += g.pdf(x)
        if normalize:
            pred = (pred-pred.min())/(pred.max()-pred.min())  * normalize
        return pred

    def visualize(self, x_range=[0,1], y_range=[0,1], n_mesh=100, normalize=False, title=None):
        filter_map = np.zeros([n_mesh*n_mesh, 3])
        x_r = np.linspace(x_range[0], x_range[1], n_mesh)
        y_r = np.linspace(y_range[0], y_range[1], n_mesh)
        filter_map[:,:2] = np.dstack(np.meshgrid(x_r, y_r)).reshape(-1,2)
        filter_map[:,2] = self.predict( filter_map[:,:2] , normalize=normalize)

        plt.figure()
        if title is not None:
            plt.title(title)
        plt.scatter(filter_map[:,0], filter_map[:,1], c=filter_map[:,2], cmap='coolwarm')
        plt.colorbar()
        plt.show()
    
    def predict_vis(self, points, normalize=False, title=None):
        pred = self.predict(points, normalize=normalize)

        plt.figure()
        if title is not None:
            plt.title(title)
        plt.scatter(points[:,0], points[:,1], c=pred, cmap='coolwarm')
        plt.colorbar()
        plt.show()

#--------------------------------------------------------------------
# (Version 4.1 Update)
def circle_xy(center=[0,0], radius=1, n_points=100):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x_points = radius * np.cos(angles) + center[0]
    y_points = radius * np.sin(angles) + center[1]
    return np.stack((x_points, y_points)).T

#--------------------------------------------------------------------

if visualize:
    node_map = GenerateNodeMap(n_nodes, random_state)
    node_map.create_node(node_scale=50, cov_knn=3)           # create node
    node_map.create_connect(connect_scale=0)             # create connection

if visualize:
    sg1 = SyntheticGaussian()
    sg1.synthesize(means=[[0.5,0.5]], covs=[np.identity(2)])

    sg1.predict(node_map.centers)
    sg1.visualize(title="12 PM")

if visualize:
    n_points=50
    means = circle_xy(center = [0.5,0.5], radius = 0.4, n_points = n_points)
    covs = np.repeat((np.identity(2)/20)[np.newaxis,...], n_points, axis=0)

    sg2 = SyntheticGaussian()
    sg2.synthesize(means=means, covs=covs)
    sg2.predict(node_map.centers)
    sg2.visualize(title="6 PM")


#--------------------------------------------------------------------
# (Version 4.1 Update)
def mean_r_f(X):
    return (np.cos((np.pi*2/(48*60))* (X - 24*60-12*60) ) +1)*1 + 0.2

# (Version 4.1 Update)
def cov_scale_f(X):
    coord_list = []
    for aw in range(11):
        x1 = np.cos((np.pi*2/(18*60))* (X - aw*24*60) )
        f1 = ((X >= (aw)*24*60)) * (X < (aw)*24*60+18*60)
        xf1 = x1 * f1
        f2 = ((X >= (aw)*24*60+18*60)) * (X < (aw+1)*24*60)
        xf2 = 1* f2
        coord_list.append(xf1+xf2)
    XS = np.stack(coord_list).sum(0)
    XS_norm = (XS +1)/2
    XS_result = (XS_norm*50)+10
    return XS_result

if visualize:
    X1 = np.arange(0, 24*60)
    mean_r_f(X1)
    cov_scale_f(X1)
    visualize_periodic(mean_r_f(X1))
    visualize_periodic(cov_scale_f(X1))

    n_points=50
    X1 = np.arange(0, 24*60, 60)
    for ei, (r, c) in enumerate(zip(mean_r_f(X1), cov_scale_f(X1))):
        means0 = circle_xy(center = [0.5,0.5], radius = r, n_points = n_points)
        cov0 = np.repeat((np.identity(2)/c)[np.newaxis,...], n_points, axis=0)
        sg0 = SyntheticGaussian()
        sg0.synthesize(means=means0, covs=cov0)
        sg0.visualize(normalize=1.5, title=ei)
        clear_output(wait=True)


# TemporalGaussian Amplitude ------------------------------------------------------------
# (Version 4.1 Update)
class TemporalGaussianAmp():
    def __init__(self, T_arr=np.arange(24*60), mean_center=[0,0], mean_r_f=None, cov_f=None,
                 centers=None, base_adj_mat=None, n_points=50, normalize=True, adjust=True, repeat=9, random_state=None):
        if T_arr.max() > 24*60:
            raise Exception("Elements in T_arr must be under 24*60.")
        self.T_arr = T_arr
        self.idx = dict(zip(T_arr, np.arange(len(T_arr))))

        self.n_points = n_points
        self.mean_center = mean_center
        self.mean_r_f = mean_r_f
        self.cov_f = cov_f
        self.centers = centers
        self.base_adj_mat = base_adj_mat
        
        self.sg_arr = np.array([])
        self.sg_dict = {}
        self.temporal_adj_mat = None
        self.temporal_adj_dict = {}

        self.normalize = normalize
        self.adjust = adjust
        self.repeat = repeat

        self.temporal_amp = None
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def default_f(self, x):
        return np.ones_like(X)

    def make_tmeporal_list(self, mean_center=None, radius_scale=1, n_points=None):
        mean_center = self.mean_center if mean_center is None else mean_center
        n_points = self.n_points if n_points is None else n_points

        mean_r_f = self.default_f if self.mean_r_f is None else self.mean_r_f
        cov_f = self.default_f if self.cov_f is None else self.cov_f

        mean_r_X = mean_r_f(self.T_arr) * radius_scale
        covs_scale_X = cov_f(self.T_arr)

        for ti, r, c in zip(self.T_arr, mean_r_X, covs_scale_X):
            sg = SyntheticGaussian()
            circle_mean = circle_xy(center=mean_center, radius=r, n_points=self.n_points)
            circle_covs = np.repeat((np.identity(2)/c)[np.newaxis,...], self.n_points, axis=0)

            sg.synthesize(means=circle_mean, covs=circle_covs)
            self.sg_arr = np.append(self.sg_arr, sg)
            self.sg_dict[ti] = sg

    def idx_f(self, t):
        return self.idx[t]

    def predict(self, t, centers=None, normalize=None):
        centers = self.centers if centers is None else centers
        normalize = self.normalize if normalize is None else normalize
        return self.sg_dict[t].predict(centers, normalize=normalize)

    def visualize(self, t, normalize=None, title=None):
        normalize = self.normalize if normalize is None else normalize

        title = f"amp distribution at {t}" if title is None else title
        self.sg_dict[t].visualize(normalize=normalize, title=title)
    
    def predict_vis(self, t, centers=None, normalize=None, title=None):
        centers = self.centers if centers is None else centers
        normalize = self.normalize if normalize is None else normalize

        title = f"node amp at {t}" if title is None else title
        self.sg_dict[t].predict_vis(centers, normalize=normalize, title=title)

    def generate_temproal_adj_mat(self, centers=None, base_adj_mat=None, normalize=None, adjust=None):
        if centers is not None:
            self.centers = centers
        else:
            centers = self.centers

        if base_adj_mat is not None:
            self.base_adj_mat = base_adj_mat
        else:
            base_adj_mat = self.base_adj_mat

        normalize = self.normalize if normalize is None else normalize
        adjust = self.adjust if adjust is None else adjust

        adj_con_mat = (base_adj_mat >0).astype(float)
        temporal_adj_con_mat = np.repeat(adj_con_mat[np.newaxis,...], len(self.T_arr), axis=0)
        temporal_adj_con_dict = {}

        for ti, (t, sg) in enumerate(self.sg_dict.items()):
            adj_con_mat_t = temporal_adj_con_mat[ti]
            pred = sg.predict(centers, normalize=normalize)

            for ni, w in enumerate(pred):
                adj_con_mat_t[:,ni] = adj_con_mat_t[:,ni]*w

            temporal_adj_con_mat[ti] = adj_con_mat_t
            temporal_adj_con_dict[t] = adj_con_mat_t

        if adjust:    
            t1_list = self.T_arr[self.T_arr <= 6*60][::-1]
            # t2_list = self.T_arr[(self.T_arr >= 6*60) * (self.T_arr <= 18*60)]
            t3_list = self.T_arr[self.T_arr >= 18*60]

            t1_decreasing_mean = temporal_adj_con_mat[self.idx_f(t1_list[0])]/len(t1_list)*3
            t1_decreasing_std = t1_decreasing_mean/10
            for ti, t1 in enumerate(t1_list):
                # print(t1_list[ti+1])
                t1_decreasing_mat = self.rng.normal(loc=t1_decreasing_mean, scale=t1_decreasing_std)

                temporal_adj_dec_mat = temporal_adj_con_mat[self.idx_f(t1_list[ti])] - t1_decreasing_mat
                temporal_adj_dec_mat[temporal_adj_dec_mat<0] = 0

                temporal_adj_con_mat[self.idx_f(t1_list[ti+1])] = temporal_adj_dec_mat
                temporal_adj_con_dict[ t1_list[ti+1] ] = temporal_adj_dec_mat

                if t1 == t1_list[-2]:
                    break
            
            t3_decreasing_mean = temporal_adj_con_mat[self.idx_f(t3_list[0])]/len(t3_list)*2
            t3_decreasing_std = t3_decreasing_mean/10
            for ti, t3 in enumerate(t3_list):
                # print(t3_list[ti+1])
                t3_decreasing_mat = self.rng.normal(loc=t3_decreasing_mean, scale=t3_decreasing_std)

                temporal_adj_dec_mat = temporal_adj_con_mat[self.idx_f(t3_list[ti])] - t3_decreasing_mat
                temporal_adj_dec_mat[temporal_adj_dec_mat<0] = 0

                temporal_adj_con_mat[self.idx_f(t3_list[ti+1])] = temporal_adj_dec_mat
                temporal_adj_con_dict[ t3_list[ti+1] ] = temporal_adj_dec_mat

                if t3 == t3_list[-2]:
                    break

        self.temporal_adj_mat = temporal_adj_con_mat
        self.temporal_adj_dict = temporal_adj_con_dict
    
    def repeat_temporal_adj_mat(self, repeat=None):
        repeat = self.repeat if repeat is None else repeat

        if self.temporal_adj_mat is not None:
            temporal_adj_mat = np.tile(self.temporal_adj_mat, (9,1,1))
            self.temporal_amp = temporal_adj_mat.transpose(1,2,0)

    def update(self, centers, base_adj_mat):
        self.generate_temproal_adj_mat(centers=centers, base_adj_mat=base_adj_mat, normalize=self.normalize, adjust=self.adjust)
        self.repeat_temporal_adj_mat(repeat=self.repeat)

    def generate_temporal_amp(self, centers=None, base_adj_mat=None, mean_center=None, repeat=None, normalize=None, adjust=None):
        centers = self.centers if centers is None else centers
        base_adj_mat = self.base_adj_mat if base_adj_mat is None else base_adj_mat
        mean_center = self.mean_center if mean_center is None else mean_center
        normalize = self.normalize if normalize is None else normalize
        adjust = self.adjust if adjust is None else adjust
        repeat = self.repeat if repeat is None else repeat
        
        if len(self.sg_arr) == 0:
            self.make_tmeporal_list(mean_center=mean_center)

        if self.temporal_adj_mat is None:
            self.generate_temproal_adj_mat(centers=centers, base_adj_mat=base_adj_mat, normalize=normalize, adjust=adjust)
        
        if self.temporal_amp is None:
            self.repeat_temporal_adj_mat(repeat=repeat)
        return self.temporal_amp

    def __call__(self, centers=None, base_adj_mat=None, mean_center=None, repeat=None, normalize=None, adjust=None):
        return self.generate_temporal_amp(centers=centers, base_adj_mat=base_adj_mat, mean_center=mean_center,
                                         repeat=repeat, normalize=normalize, adjust=adjust)
    

if example:
    n_nodes=50
    node_map = GenerateNodeMap(n_nodes, random_state)
    node_map.create_node(node_scale=50, cov_knn=3)           # create node
    node_map.create_connect(connect_scale=0)             # create connection

    visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)

if example:
    # centers = node_map.centers
    # node_map.adj_matrix

    X1 = np.arange(0, 24*60)
    # X1 = np.arange(0, 24*60,20)
    tga = TemporalGaussianAmp(T_arr = X1, mean_center=[0.5,0.5], mean_r_f=mean_r_f, cov_f=cov_scale_f,
                            centers=node_map.centers, base_adj_mat=node_map.adj_matrix,
                            normalize=1.5, adjust=True, repeat=9, random_state=random_state)
    tga.generate_temporal_amp()
    # tga.make_tmeporal_list(mean_center=[0.5,0.5])
    # tga.sg_arr
    # tga.generate_temproal_adj_mat(normalize=1.5)
    # tga.temporal_adj_mat
    # tga.repeat(9)
    tga.temporal_amp



if visualize:
    X1 = np.arange(0, 24*60, 20)
    # tga = TemporalGaussianAmp(T_arr = X1, mean_center=[0.5,0.5], mean_r_f=mean_r_f, cov_f=cov_scale_f,
    #                         centers=node_map.centers, base_adj_mat=node_map.adj_matrix,
    #                         normalize=1.5, adjust=True, repeat=9, random_state=random_state)
    # tga.make_tmeporal_list()

    tga.predict(3*60)
    tga.predict(11*60)
    tga.visualize(3*60)
    tga.visualize(11*60)
    tga.predict_vis(3*60)
    tga.predict_vis(11*60)

if example:
    # Animation of filter_map
    from IPython.display import clear_output
    for i in range(24):
        tga.visualize(i*60, normalize=True,title=i*60)
        # tga.predict_vis(i*60, normalize=True)
        
        time.sleep(0.05)
        clear_output(wait=True)




if visualize:
    # tga.generate_temproal_adj_mat()

    # Animation of adjacent_matrix
    from IPython.display import clear_output
    for t, t_mat in zip(tga.T_arr, tga.temporal_adj_mat):
        if t%60 == 0:
            plt.title(t //60)
            plt.imshow(t_mat, cmap='Blues', vmin=0, vmax=1.5)
            plt.colorbar()
            plt.show()
            clear_output(wait=True)
            time.sleep(0.1)

if visualize:
    # Animation of graph
    for t, t_mat in zip(tga.T_arr, tga.temporal_adj_mat):
        if t%60 == 0:
            visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix*(t_mat*0.5+1)
                        ,weight_vis_base_mat=node_map.adj_matrix, weight_direction='forward', vmax=1.5, title=t//60)
            clear_output(wait=True)
        # time.sleep(0.05)

# temporal_repeat
if visualize:
    tga.repeat_temporal_adj_mat()
    tga.temporal_amp


# update
if visualize:
    n_nodes=50
    node_map = GenerateNodeMap(n_nodes, random_state)
    node_map.create_node(node_scale=50, cov_knn=3)           # create node
    node_map.create_connect(connect_scale=0)             # create connection

    visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)

    tga.update(centers=node_map.centers, base_adj_mat=node_map.adj_matrix)

    # Animation of graph
    for t, t_mat in zip(tga.T_arr, tga.temporal_adj_mat):
        if t%60 == 0:
            visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix*(t_mat*0.5+1)
                        ,weight_vis_base_mat=node_map.adj_matrix, weight_direction='forward', vmax=1.5, title=t//60)
            clear_output(wait=True)

# Noise Setting -------------------------------------------------------------------------------
class RandomNoise():
    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.kwargs = kwargs

    def weighted_random_noise(self, std):
        return self.rng.normal(loc=0, scale=std) * self.kwargs['scale']

    def __call__(self, x):
        return self.weighted_random_noise(x)





################################################################################################
# Aggregate temporal graph with periodic condition #####################################################

# (Version 4.1 Update)
# TemporalGraph
class TemporalGraph():
    # (Version 4.1 Update) add node_map instace as argument
    def __init__(self, node_map, amp_class=None,  periodic_f = lambda x: 0,
        error_f= lambda x: 0, centers=None, covs=None, base_mat=None, random_state=None, **kwargs):
        """
            . set_periodic_f(periodic_f, **kwargs) : periodic_f (0~1)
            . set_amplitude(amp_mat, amp = 1, **kwargs) :

            . gen_periods(x, **kwargs) : 
            . transform_oracle(x, base_mat) : 
            . transform(x, base_mat) : oracle + error
        """
        # (Version 4.0 Update) --------------------------------------------------
        self.node_map = node_map
        self.centers = self.node_map.centers if centers is None else centers
        self.covs = self.node_map.covs if covs is None else covs
        self.base_mat = self.node_map.adj_matrix if base_mat is None else base_mat
        # -----------------------------------------------------------------------

        self.amp_class=amp_class
        self.amp_instance = None

        self.temporal_oracle = None
        self.errors = None
        self.temporal_observations = None

        self.periodic_f = periodic_f
        self.error_f = error_f
        self.periodic_kwargs = kwargs
        self.reset_error = False

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def set_periodic_f(self, periodic_f, **kwargs):
        self.periodic_f = periodic_f
        self.periodic_kwargs = kwargs

    def create_amp_instance(self, **kwargs):
        self.amp_instance = self.amp_class(random_state=self.random_state, **kwargs)

    def gen_periods(self, x, **kwargs):
        if len(kwargs) == 0:
            return self.periodic_f(x, **self.periodic_kwargs)
        else:
            return self.periodic_f(x, **kwargs)
    
    # (Version 4.1 Update)
    def make_temporal_oracle(self, X=np.arange(0, 9*24*60), base_mat=None):
        base_mat = self.base_mat if base_mat is None else base_mat

        if base_mat is not None:
            n_row, n_cols = base_mat.shape

            # (Version 4.1 Update) ------------------------------------------------------------
            amp = self.amp_instance()
            periods = self.periodic_f(X, **self.periodic_kwargs)
            
            # transformed = ( (amp * periods + 1) * base_mat.reshape(n_row, n_cols, 1)  ).transpose(2,0,1).squeeze()
            transformed = ( (amp*0.5 + 1) * (periods*0.5 + 1) * base_mat.reshape(n_row, n_cols, 1)  ).transpose(2,0,1).squeeze()
            # ----------------------------------------------------------------------------------

            # diagonal term → 0
            diag_indices = np.arange(n_row)
            if transformed.ndim == 3: 
                transformed[:, diag_indices, diag_indices] = 0
            elif transformed.ndim == 2: 
                transformed[diag_indices, diag_indices] = 0

            # transformed[transformed==0] = np.inf        # 연결되지 않는 노드는 infty로 치환
            self.temporal_oracle =  transformed
        else:
            print("missing base_mat.")

    # (Version 2.2 Update)
    def transform_oracle(self, x):
        if self.temporal_oracle is None:
            self.make_temporal_oracle()
        return self.temporal_oracle[x]

    # (Version 2.2 Update)
    def make_temporal_observations(self):
        # (Version 2.2) -----------------------------------------------
        # (Version 4.0 Update) ------------------------------------
        self.make_temporal_oracle()
        # ---------------------------------------------------------
    
        oracle = self.temporal_oracle
        errors = self.error_f(oracle)
        filter = (self.temporal_oracle.sum(0) > 0)[np.newaxis, ...]

        errors[0] += self.rng.normal(loc=0, scale=oracle[0]) * 0.05 * filter[0]
        self.temporal_errors = errors.cumsum(axis=0) * filter
        self.temporal_observations = oracle + self.temporal_errors
        # --------------------------------------------------------------

        # # (Version 2.1) ------------------------------------------------
        # if self.temporal_oracle is None:
        #     self.make_temporal_oracle()
        # oracle = self.temporal_oracle
        # self.temporal_errors = self.error_f(oracle)    # bottle_neck
        # self.temporal_observations = oracle + self.temporal_errors
        # #  -------------------------------------------------------------

    # (Version 2.2 Update)
    def transform(self, x, reset_error=None):
        if reset_error is not None:
            self.reset_error = reset_error

        if (self.reset_error is True) or (self.temporal_observations is None):
            self.make_temporal_observations()
        return self.temporal_observations[x]
    
    # (Version 4.0 Update) update node_map, temporal information to temporalGraph
    def update_graph(self, node_map=None):
        if node_map is not None:
            self.node_map = node_map

        self.centers = self.node_map.centers
        self.covs = self.node_map.covs
        self.base_mat = self.node_map.adj_matrix
        self.amp_instance.update(centers=self.node_map.centers, base_adj_mat=self.node_map.adj_matrix)
        self.make_temporal_observations()

    def __call__(self, x):
        return self.transform(x)


# ---------------------------------------------------------------------------------------


if visualize:
    # (create base graph) 
    node_map = GenerateNodeMap(n_nodes, random_state)
    node_map.create_node(node_scale=50, cov_knn=3)           # create node
    node_map.create_connect(connect_scale=0)             # create connection

    visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix, covs=node_map.covs)

if example: 
    # (periodic setting)
    periodic_f = periodic_improve    # periodic function

    # (random noise)
    # random_noise_f = RandomNoise(scale=1/30, random_state=random_state)       # (Version 2.1)
    random_noise_f = RandomNoise(scale=1/1000, random_state=random_state)      # (Version 2.2)
    # random_noise_f = RandomNoise(scale=1/300)                                 # (Version 2.2)

    # (aggregated route_graph instance)
    temporal_graph = TemporalGraph(node_map=node_map, amp_class=TemporalGaussianAmp,
                                 periodic_f=periodic_f, error_f=random_noise_f, random_state=random_state)
    # (temporal gaussian amplitude)
    temporal_graph.create_amp_instance(T_arr = X1, mean_center=[0.5,0.5], mean_r_f=mean_r_f, cov_f=cov_scale_f,
                            centers=node_map.centers, base_adj_mat=node_map.adj_matrix,
                            normalize=1.5, adjust=True, repeat=9)
    temporal_graph.make_temporal_observations()
    # ---------------------------------------------------------------------------------------

    # Example Create Graph ----------------------------------------------------------------
    # x = np.arange(0, 8*24*60)

    # temporal_graph.transform(x)
    # temporal_graph.transform(0*24*60 + 14*60)    # 월요일 오후 2시

    # temporal_graph.transform_oracle(x)
    # temporal_graph.transform_oracle(0*24*60 + 14*60)


    # Example Visualize --------------------------------------------------------------------

    # Node_map 재구성
    temporal_graph.node_map.create_node(node_scale=50, cov_knn=3)
    temporal_graph.node_map.create_connect(connect_scale=0) 
    temporal_graph.update_graph()

    # temporal_graph.reset_error = True
    # temporal_graph.reset_error = False
    # temporal_graph.make_temporal_observations()     # error term reset

    visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)


# (Version 2.2 Update)
if visualize: 
    # (temporal periodic plot)
    x = np.arange(0, 8*24*60)
    idx = [30, 24]
    # idx = [8, 11]
    # idx = [33, 38]

    y_pred = temporal_graph.transform(x)[:, idx[0], idx[1]]
    y_true = temporal_graph.transform_oracle(x)[:, idx[0], idx[1]]
    visualize_periodic(y_pred, y_true, return_plot=False)
# ----------------------------------------------------------------



# Animation Graph
if visualize:
    min_idx = np.argmin(temporal_graph.temporal_oracle.sum(axis=(1,2)))
    base_weights = temporal_graph.transform(min_idx)
    for t in range(24):
        visualize_graph(centers=node_map.centers, adjacent_matrix=temporal_graph(t*60), 
                        weight_vis_base_mat=base_weights, title=f"hour : {t}", vmax=3)
        clear_output(wait=True)
        # time.sleep(0.1)
        
# Animation Adjacent_Matrix
from IPython.display import clear_output
import time
if visualize: 
    # (temporal matrix plot)
    for t in range(24):
        plt.figure()
        plt.title(f"hour : {t}")
        plt.imshow(temporal_graph(t*60), cmap='Blues', vmin=0, vmax=node_map.adj_matrix.max()*0.8)
        # plt.imshow(route_transform.transform_oracle(t*60), cmap='coolwarm', vmin=0, vmax=base_sigma*12)
        plt.colorbar()
        plt.show()
        clear_output(wait=True)
        time.sleep(0.1)


# Animation Graph
if example:
    min_idx = np.argmin(temporal_graph.temporal_oracle.sum(axis=(1,2)))
    base_weights = temporal_graph.transform(min_idx)
    for t in range(24):
        visualize_graph(centers=node_map.centers, adjacent_matrix=temporal_graph(t*60), 
                        weight_vis_base_mat=base_weights, title=f"hour : {t}", vmax=2.5)
        clear_output(wait=True)
        # time.sleep(0.1)


################################################################################################
# Shortest Path (Dijkstra) #####################################################################

# Shortest Path
import heapq

# (Dijkstra Algorithm)
# (Version 4.0 Update)
class Dijkstra:
    def __init__(self, graph=None):
        self.graph = graph
        self.size = len(graph)

    # (Version 4.0 Update)
    def set_graph(self, graph):
        self.graph = graph

    def dijkstra(self, start, end):
        distances = [float('inf')] * self.size
        distances[start] = 0
        priority_queue = [(0, start)]
        visited = [False] * self.size
        previous_nodes = [-1] * self.size

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if visited[current_node]:
                continue
            
            visited[current_node] = True

            for neighbor, weight in enumerate(self.graph[current_node]):
                if weight > 0 and not visited[neighbor]:  # There is a neighbor and it's not visited
                    distance = current_distance + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous_nodes[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))

        path = self._reconstruct_path(previous_nodes, start, end)
        return distances[end], path

    def _reconstruct_path(self, previous_nodes, start, end):
        path = []
        current_node = end
        while current_node != -1:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path.reverse()

        if path[0] == start:
            return path
        else:
            return []  # If the path does not start with the start node, return an empty list
    
    def __call__(self, start, end):
        return self.dijkstra(start, end)
        



# Example Visualize --------------------------------------------------------------------
if example: 
    # Node_map 재구성
    temporal_graph.node_map.create_node(node_scale=50, cov_knn=3)
    temporal_graph.node_map.create_connect(connect_scale=0) 
    temporal_graph.update_graph()

    adj_mat = temporal_graph.transform(0*24*60 + 14*60)

    dijkstra = Dijkstra(adj_mat)
    start_node = 33
    end_node = 27

    shortest_distance, path = dijkstra.dijkstra(start_node, end_node)
    visualize_graph(centers=node_map.centers, adjacent_matrix=adj_mat,
                path=path, distance=shortest_distance)
    print(f"The shortest distance from node {start_node} to node {end_node} is {shortest_distance}")
    print(f"The path taken is {path}")






#####################################################################################################
# Episodes Simulation ###############################################################################
import os
from datetime import datetime
import pandas as pd
import warnings

# os.getcwd()
warnings.filterwarnings('ignore')

# (Version 4.0 Update)
# 시간을 "요일. 시:분" 형식으로 변환하는 함수
def format_time_to_str(time, return_type='str'):
    """
    return_type : 'str', 'dict'
    """
    if time is None:
        return None
    else:
        int_time = int(time)
        week_code = ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."]
        week = int_time // (24*60)
        hour = (int_time % (24*60)) // 60
        min = (int_time % (24*60)) % 60

        # (Version 4.0 Update) -----------------------------------
        if return_type == 'str':
            return f"{week_code[week % 7]} {hour:02d}:{min:02d}"
        elif return_type == 'dict':
            return {"week": week_code[week % 7], "hour": hour, "min":min}
            # return {"week": week, "hour": hour, "min":min}
        # ---------------------------------------------------------

# (Version 4.0 Update)
# "요일. 시:분" 형식을 시간형식으로 변환하는 함수
def format_str_to_time(time_str):
    if time_str is None:
        return None
    else:
        week_dict = {"Mon.":0, "Tue.":1, "Wed.":2, "Thu.":3, "Fri.":4, "Sat.":5, "Sun.":6}

        week_str, hour_min_str = time_str.split(" ")
        hour, min = hour_min_str.split(":")
        return week_dict[week_str]*24*60 + 60*int(hour) + int(min)
    

# now_date = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
# excel_filename = 'travel_times.xlsx'      # # 엑셀 파일 이름

# col_names = ['ID', 'Dep_City', 'Target_City', 'Dep_Time', 'Target_Time',
#              'Now_Time', 'Path', 'Start_add_time', 'End_add_time', 'Transfer_time', 'Total_time',
#              'Leaving_Time', 'Pred_Arrival_Time', 'Event']
# # 기존 파일이 존재하는지 확인
# if os.path.exists(excel_filename):  # 기존 파일이 존재하면 데이터를 읽어옴
#     existing_results_df = pd.read_excel(excel_filename, sheet_name='Results')
# else:   # 기존 파일이 없으면 빈 데이터프레임 생성
#     existing_results_df = pd.DataFrame(columns=col_names)




######################################################################################################
### (Agent) ##########################################################################################

# (Version 4.0 Update)
class Agent():
    # (Version 4.0 Update)
    def __init__(self, adjacent_matrix=None, departure=None, objective_path=[], start_time=None, err_sigma=0, centers=None, random_state=None):
        """
         . self.states : stop / move / hold
        """
        # if (departure is None and len(objective_path) == 0):
        #     raise print("departure or objective_path must be designated.")
        self.status = "stop"     # stop / move
        self.adjacent_matrix = adjacent_matrix
        
        self.t = 0
        self.start_time = start_time
        # (Version 4.0 Update) ----------------------------
        self.centers = centers
        if len(objective_path) > 0:
            self.cur_node = objective_path[0]
            
        else:
            self.cur_node = departure

        self.cur_point = None     # new instance variable
        if centers is not None and self.cur_node is not None:
            self.cur_point = self.cur_coordinates(centers)
        # ------------------------------------------------
        self.cur_dest = None 

        self.total_distance = 0.0
        self.target_distance = 0

        self.distance = 0.0
        self.remain_distance = 0.0
        
        self.err_sigma = err_sigma

        # (Version 4.0 Update) ----------------------------
        if len(objective_path) == 0:
            self.path = [departure]
        else:
            self.path = [objective_path[0]]
        # ------------------------------------------------
        self.path_distance = []

        self.objective_path = objective_path[1:]
        self.final_destination = None if len(objective_path) == 0 else objective_path[-1]

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.history = []

    # (Version 4.0 Update)
    def set_path(self, departure=None, objective_path=[]):
        if self.t == 0:
            if (departure is None and len(objective_path) == 0):
                raise print("departure or objective_path must be designated.")
            
            self.cur_node = departure if departure is not None else objective_path[0]
            self.path = [departure] if departure is not None else [objective_path[0]]

            self.objective_path = objective_path[1:]
            self.final_destination = None if len(objective_path) == 0 else objective_path[-1]
        else:
            print("agent is already moved.")

    def update_graph(self, adjacent_matrix):
        self.adjacent_matrix = adjacent_matrix
        
        if self.status == 'move':
            bef_target_distance = self.target_distance
            new_target_distance = self.adjacent_matrix[self.cur_node][self.cur_dest]

            self.remain_distance = new_target_distance - new_target_distance * self.distance/bef_target_distance 
            self.target_distance = self.distance + self.remain_distance

            if np.round(self.remain_distance) <= 0:
                self.stop()

    # (Version 4.0 Update)
    def hold(self):
        self.status = 'hold'
        self.cur_dest = None
        self.save_history()
        self.t += 1

    # (Version 4.0 Update)
    def move(self, destination=None, hold_prob=0.3, succ_hold_prob=0.9, verbose=0):
        if self.status == 'stop' or self.status == 'hold':
            if self.cur_node != self.final_destination:
                # (Version 4.0 Update) ----------------------------
                if self.status == 'stop':
                    next_action = self.rng.choice(['move', 'hold'], p=[1-hold_prob, hold_prob])
                elif self.status == 'hold':
                    next_action = self.rng.choice(['move', 'hold'], p=[1-succ_hold_prob, succ_hold_prob])
                
                if next_action == 'hold':
                    self.hold()
                # --------------------------------------------------
                else:
                    self.status = 'stop'
                    if destination is not None:     # if learner directly designate destination
                        self.destination_path(destination)      # designated next destination
                    elif len(self.objective_path) >0:   # if have objective path
                        cur_dest = self.objective_path[0]
                        self.destination_path(cur_dest)
                        self.objective_path = self.objective_path[1:]
                    else:
                        self.random_path()
                        self.move()
            else:
                if verbose:
                    print("arrived at destination!")
                return 'arrive'
        elif self.status == 'move':
            err = max(0, self.rng.normal(loc=0, scale=self.err_sigma))
            self.distance += 1 + err
            self.remain_distance -= (1 + err)
            self.total_distance += 1 + err
            self.path_distance[-1] = self.distance

            if np.round(self.remain_distance) <= 0:
                self.stop()
            
            self.save_history()
            self.t += 1
            # (Version 4.0 Update) ----------------------------
            if self.centers is not None:
                self.cur_point = self.cur_coordinates(self.centers)
            # ------------------------------------------------
    
    def stop(self):
        self.status = 'stop'
        self.cur_node = self.cur_dest
        self.path.append(self.cur_dest)
        self.cur_dest = None
        self.total_distance = self.total_distance - self.distance + self.target_distance
        self.path_distance[-1] = self.target_distance

    def destination_path(self, destination):
        if self.adjacent_matrix[self.cur_node][destination] > 0:
            self.cur_dest = destination
            self.distance = 0.0
            self.remain_distance = self.adjacent_matrix[self.cur_node][destination]
            self.target_distance = self.remain_distance
            self.status = 'move'
            self.path_distance.append(0)
        else:
            raise(f"destination error : not connected from {self.cur_node} to {destination}")

    # (Version 4.0 Update)
    def random_path(self):
        possible_dest = self.adjacent_matrix[self.cur_node]
        
        # (Version 4.0 Update) --------------------------------------
        random_choose = (possible_dest > 0)/(possible_dest > 0).sum()
        past_path = list(set(self.path))
        random_choose[past_path] = random_choose[past_path] * 0.2
        random_choose = random_choose/random_choose.sum()
        # -----------------------------------------------------------

        self.cur_dest = self.rng.choice(np.arange(len(possible_dest)), p=random_choose)
        self.distance = 0.0
        self.remain_distance = possible_dest[self.cur_dest]
        self.target_distance = self.remain_distance
        self.status = 'move'
        self.path_distance.append(0)

    # (Version 4.0 Update)
    def cur_coordinates(self, nodes):
        if self.status =='move' and self.target_distance > 0:
            percent = self.distance / self.target_distance
            start = nodes[self.cur_node]
            end = nodes[self.cur_dest]
            mid_x = start[0] + percent * (end[0] - start[0])
            mid_y = start[1] + percent * (end[1] - start[1])
            return (mid_x, mid_y)
        else:
            return tuple(nodes[self.cur_node])

    def print_agent_info(self):
        print(f"[step:{self.t}/dist: {self.total_distance:.1f}] status: {self.status} ({self.cur_node} -> {self.cur_dest}) / distance: {self.distance:.1f}, remain: {self.remain_distance:.1f}")

    # (Version 4.0 Update)
    def visualize(self, temporal_graph=None, cur_time=None, centers=None, cur_adj_mat=None, base_adj_mat=None, title=None, return_plot=False):
        if cur_time is None:
            if self.start_time is not None:
                cur_time = self.start_time + t

        if temporal_graph is not None:
            centers = temporal_graph.centers if centers is None else centers
            cur_adj_mat = temporal_graph.transform(cur_time) if cur_adj_mat is None else cur_adj_mat
            base_adj_mat = temporal_graph.transform(0) if base_adj_mat is None else base_adj_mat
        else:
            centers = self.centers if centers is None else centers
            base_adj_mat = cur_adj_mat if base_adj_mat is None else base_adj_mat
        cur_point = self.cur_coordinates(centers) if self.cur_point is None else self.cur_point

        if title is None:
            cur_time_str = f", {format_time_to_str(cur_time)}" if cur_time is not None else ""
            cur_agent_status = f"{self.status} : " if self.status == 'move' else self.status
            cur_dest_str = "" if self.cur_dest is None else f" toward {self.cur_dest}"
            path_str = str(self.path[-5:]).replace("[", f" {len(self.path)} paths: [{str(self.path[:3]).replace('[','').replace(']','')}, ..., ") if len(self.path) > 8 else str(self.path)
            title = f"[{self.t} round{cur_time_str}] {cur_agent_status}{cur_dest_str}\n(Dist: {self.total_distance:.1f}) {path_str}"

        fig = visualize_graph(centers=centers, adjacent_matrix=cur_adj_mat,
                path=self.path, distance=self.total_distance,
                point=cur_point, point_from=centers[self.cur_node]
                ,weight_vis_base_mat=base_adj_mat, title=title, path_distance=self.path_distance, return_plot=return_plot)
        
        if return_plot:
            return fig

    # (Version 4.1 Update)
    def save_history(self, event=None):
        history = {"t":self.t, "status":self.status, "start_time" : self.start_time, 
                "cur_node": self.cur_node, "cur_point":self.cur_point, "cur_dest":self.cur_dest,
                "total_distance" : self.total_distance, "target_distance" : self.target_distance, "distance" : self.distance, "remain_distance" : self.remain_distance, 
                "path" : self.path.copy(), "path_distance" : self.path_distance.copy(),
                "objective_path" : self.objective_path.copy(), "final_destination" : self.final_destination,
                "event": event}
        self.history.append(history)

    # (Version 4.1 Update)
    def history_visualize(self, temporal_graph=None, t=None, start_time=None, cur_time=None, centers=None, cur_adj_mat=None, base_adj_mat=None, return_plot=False):
        start_time = self.start_time if start_time is None else start_time

        if t is None:
            if cur_time is not None and start_time is not None:
                t = cur_time - start_time
            else:
                raise Exception("either t or cur_time must be needed with start_time.")
        elif cur_time is None and t is not None and start_time is not None:
            cur_time = start_time + t

        if temporal_graph is not None:
            centers = temporal_graph.centers if centers is None else centers
            cur_adj_mat = temporal_graph.transform(cur_time) if cur_adj_mat is None else cur_adj_mat
            base_adj_mat = temporal_graph.transform(0) if base_adj_mat is None else base_adj_mat
        else:
            centers = self.centers if centers is None else centers
            base_adj_mat = cur_adj_mat if base_adj_mat is None else base_adj_mat
        cur_point = self.cur_coordinates(centers) if self.cur_point is None else self.cur_point

        cur_point = self.history[t]['cur_point']

        cur_time_str = f", {format_time_to_str(cur_time)}" if cur_time is not None else ""
        cur_agent_status = f"{self.history[t]['status']} : " if self.history[t]['status'] == 'move' else self.history[t]['status']
        cur_dest_str = "" if self.history[t]['cur_dest'] is None else f" toward {self.history[t]['cur_dest']}"
        path_str = str(self.history[t]['path'][-5:]).replace("[", f" {len(self.history[t]['path'])} paths: [{str(self.history[t]['path'][:3]).replace('[','').replace(']','')}, ..., ") if len(self.history[t]['path']) > 8 else str(self.history[t]['path'])
        title = f"[{t} round{cur_time_str}] {cur_agent_status}{cur_dest_str}\n(Dist: {self.history[t]['total_distance']:.1f}) {path_str}"
        
        fig = visualize_graph(centers=centers, adjacent_matrix=cur_adj_mat,
            path=self.history[t]['path'], distance=self.history[t]['total_distance'],
            point=cur_point, point_from=centers[self.history[t]['cur_node']]
            ,weight_vis_base_mat=base_adj_mat, title=title, path_distance=self.path_distance, return_plot=return_plot)
        return fig

#------------------------------------------------------------------------------------------------------
if visualize:
    # (create base graph) 
    node_map = GenerateNodeMap(n_nodes, random_state)
    node_map.create_node(node_scale=50, cov_knn=3)           # create node
    node_map.create_connect(connect_scale=0)             # create connection


if visualize:
    visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)
    # visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix, covs=node_map.covs)

#------------------------------------------------------------------------------------------------------
if example:
    # create agent
    departure_node = rng.randint(0,n_nodes)
    print(f"departure_node: {departure_node}")

    agent = Agent(adjacent_matrix=node_map.adj_matrix, departure=departure_node, centers=node_map.centers)

# (random walk) - static time step ---------------------------------------------------------------------
# animation random walk
if example:
    from IPython.display import clear_output
    import time
    for _ in range(20):
        agent.move()
        # agent.print_agent_info()
        
        agent.visualize(cur_adj_mat=node_map.adj_matrix)
        clear_output(wait=True)
        # time.sleep(0.05)


# (random walk) - dynamic time step -------------------------------------------------------------------
if visualize:
    # (periodic setting)
    periodic_f = periodic_improve    # periodic function

    # (aggregated route_graph instance)
    temporal_graph = TemporalGraph(node_map=node_map, amp_class=TemporalGaussianAmp,
                                    periodic_f=periodic_f, error_f=random_noise_f)
    # (temporal gaussian amplitude)
    temporal_graph.create_amp_instance(T_arr = X1, mean_center=[0.5,0.5], mean_r_f=mean_r_f, cov_f=cov_scale_f,
                            centers=node_map.centers, base_adj_mat=node_map.adj_matrix,
                            normalize=1.5, adjust=True, repeat=9)
    temporal_graph.make_temporal_observations()

    # init_setting
    departure_node = rng.randint(0,n_nodes)     # start_node
    start_time = int(rng.uniform(0, 7*24*60))       # start_time
    print(f"({format_time_to_str(start_time)}) departure_node: {departure_node}")

    # create agent
    agent = Agent(adjacent_matrix=node_map.adj_matrix, departure=departure_node)

# animation random walk
if visualize:
    from IPython.display import clear_output
    import time
    min_idx = np.argmin(temporal_graph.temporal_oracle.sum(axis=(1,2)))
    base_weights = temporal_graph.transform(min_idx)

    time_scale = 60
    for i in range(50):
        cur_time = start_time + i * time_scale
        cur_adj_matrix = temporal_graph.transform(cur_time)

        agent.update_graph(cur_adj_matrix)
        agent.move()
        # agent.print_agent_info()

        if i % 1 == 0:
            agent.visualize(temporal_graph=temporal_graph, cur_time=cur_time)
            clear_output(wait=True)
            # time.sleep(0.05)

# time_scale = 30
# for i in range(300):
#     cur_adj_matrix = temporal_graph.transform(start_time + i)

#     agent.update_graph(cur_adj_matrix)
#     agent.move()
#     # agent.print_agent_info()

#     title = f"({format_time_to_str(start_time + i)}) toward {agent.cur_dest} \n(Dist: {agent.total_distance:.1f}) {agent.path}"

#     if i % time_scale == 0:
#         visualize_graph(centers=temporal_graph.centers, adjacent_matrix=cur_adj_matrix,
#                 path=agent.path, distance=agent.total_distance,
#                 point=agent.cur_coordinates(temporal_graph.centers), point_from=temporal_graph.centers[agent.cur_node]
#                 ,weight_vis_base_mat= base_weights, title=title, path_distance=agent.path_distance)
#         clear_output(wait=True)
#     # time.sleep(0.05)




# (designated walk) - dynamic time step -------------------------------------------------------------------
if visualize:
    departure_node = rng.randint(0,n_nodes)
    dest_node = rng.choice([i for i in np.arange(n_nodes) if i!=departure_node])
    start_time = int(rng.uniform(0, 7*24*60))       # start_time
    print(f"({format_time_to_str(start_time)}) from {departure_node} to {dest_node}")

    # (aggregated route_graph instance)
    temporal_graph.make_temporal_observations()         # (Version 2.2)


    dijkstra = Dijkstra(temporal_graph.transform(start_time))
    shortest_distance, shortest_path = dijkstra.dijkstra(departure_node, dest_node)
    print(f". designated_path : {shortest_path}")

if visualize:
    visualize_graph(centers=temporal_graph.centers, adjacent_matrix=temporal_graph.transform(start_time),
        path=shortest_path, distance=shortest_distance)
    print(f"The shortest distance from node {start_node} to node {end_node} is {shortest_distance}")
    print(f"The path taken is {shortest_path}")

# ------------------------------------------------------------------------


if visualize:
    # create agent
    agent = Agent(adjacent_matrix=temporal_graph.transform(start_time), objective_path=shortest_path)

# animation designated walk
if visualize:
    time_scale = 60
    for i in range(50):
        cur_time = start_time + i* time_scale
        cur_adj_matrix = temporal_graph.transform(cur_time)

        agent.update_graph(cur_adj_matrix)
        is_arrive = agent.move()
        # agent.print_agent_info()

        agent.visualize(temporal_graph=temporal_graph, cur_time=cur_time)

        if is_arrive == 'arrive':
            break
        clear_output(wait=True)
        # time.sleep(0.05)


# ------------------------------------------------------------------------
############################################################################################################
############################################################################################################



################################################################################################################
# (Case Simulation : Stationary Location) ######################################################################

# Simulator
# (Version 4.4 Update)
class TemporalMapSimulation():
    """
    To execute this class, it must Need 'Agent', 'Dijkstra' classes
    """
    id_no = 1
    id_base = None

    def __init__(self, temporal_graph=None, target_time_safe_margin=8):
        if self.id_no == 1:
            TemporalMapSimulation.id_base = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')

        self.id = f"{TemporalMapSimulation.id_base}_{TemporalMapSimulation.id_no:08d}"
        TemporalMapSimulation.id_no += 1

        self.temporal_graph = temporal_graph
        self.base_adj_mat = temporal_graph.transform(0)
        self.agent = None
        self.cur_adj_mat = None     # (Version 4.0 Update)
        self.simul_status = "init"

        self.iter = 0
        self.iteration_limit = 1500

        # loc / node
        self.start_point = None
        self.start_node = None
        self.target_point = None
        self.target_node = None

        # time
        self.start_time = None
        self.target_time = None

        self.start_add_time = 0
        self.target_add_time = 0

        self.target_time_safe_margin = target_time_safe_margin

        # true_info
        self.true_req_leaving_time = None
        self.true_leaving_info = {}

        # cur_info
        self.t = 0
        self.cur_point = None
        self.cur_node = None
        self.cur_time = None
        self.cur_path = None
        self.req_leaving_time = None

        # TimeMachine
        self.call_time_TimeMachine = None
        self.path_TimeMachine = None
        self.path_time_TimeMachine = None
        self.req_leaving_time_TimeMachine = None

        # LastAPI
        self.call_time_LastAPI = None
        self.call_node_LastAPI = None
        self.call_point_LastAPI = None 
        self.path_LastAPI = None
        self.path_time_LastAPI = None

        # full info
        self.req_leaving_time_full_info = None

        # loop
        self.loop = True
        self.loop_history = True
        self.loop_full_info = True

        # save history
        self.save_interval = 1      # (Version 4.0 Update)
        self.save_t = np.array([]).astype(bool)
        self.history = []
        self.history_api_call = np.array([]).astype(bool)  # (Version 4.0 Update)
        self.history_time_machine = []
        self.history_full_info = []
        self.history_policy = []

        # algorithm
        self.policy_f = None

    def set_start_loc(self, start_loc=None):
        if start_loc is None:
            start_point = np.random.uniform(size=2)
            _, start_node = self.temporal_graph.node_map.predict(start_point)

            while start_node == self.target_node:   # 이전에 지정된 target node와 겹치는 경우
                start_point = np.random.uniform(size=2)
                _, start_node = self.temporal_graph.node_map.predict(start_point)
    
        elif np.array(start_loc).ndim == 0:    # node no.
            start_point = self.temporal_graph.centers[start_loc]
            start_node = start_loc
        elif np.array(start_loc).ndim == 1:       # node coordinate
            start_point = start_loc
            _, start_node = self.temporal_graph.node_map.predict(start_point)
        
        # self.start_point = start_point
        self.start_point = self.temporal_graph.centers[start_node]
        self.start_node = start_node

    def set_start_time(self, start_time=None, time_interval = 3*60):
        if start_time is None:
            if self.target_time is None:
                start_time = int(np.random.uniform(0, 7*24*60))
            else:
                start_time = self.target_time - ( time_interval + int(np.random.uniform(0, 24*60-time_interval)) )
                if start_time < 0:
                    start_time = 0

        self.start_time = start_time

    def set_target_loc(self, target_loc=None):
        if target_loc is None:
            target_point = np.random.uniform(size=2)
            _, target_node = self.temporal_graph.node_map.predict(target_point)

            while target_node == self.start_node:
                target_point = np.random.uniform(size=2)
                _, target_node = node_map.predict(target_point)
        elif np.array(target_loc).ndim == 0:    # node no.
            target_point = self.temporal_graph.centers[target_loc]
            target_node = target_loc
        elif np.array(target_loc).ndim == 1:       # node coordinate
            target_point = target_loc
            _, target_node = self.temporal_graph.node_map.predict(target_point)

        # self.target_point = target_point
        self.target_point = self.temporal_graph.centers[target_node]
        self.target_node = target_node

    def set_target_time(self,target_time=None, time_interval = 3*60):
        if target_time is None:
            if self.start_time is None:
                target_time = int(np.random.uniform(time_interval, time_interval + 8*24*60))
            else:
                target_time = self.start_time + time_interval + int(np.random.uniform(0, 24*60-time_interval))
        
        self.target_time = target_time

    def set_start_info(self, start_loc=None, start_time=None, time_interval = 3*60, verbose=0):
        self.set_start_loc(start_loc)
        self.set_start_time(start_time, time_interval)

        if verbose:
            print(f"[{format_time_to_str(self.start_time)}] {self.start_node}: ({self.start_point[0]:.2f}, {self.start_point[1]:.2f})", end=" ")

    def set_target_info(self, target_loc=None, target_time=None, time_interval = 3*60, verbose=0):
        self.set_target_loc(target_loc)
        self.set_target_time(target_time, time_interval)

        if verbose:
            print(f"[{format_time_to_str(self.target_time)}] {self.target_node}: ({self.target_point[0]:.2f}, {self.target_point[1]:.2f})", end=" ")

    def set_policy(self, policy_f):
        self.policy_f = policy_f

    # (Version 4.1 Update)
    def reset_state(self, reset_all=False, save_interval=None, objective_path=[], verbose=0):
   
        if reset_all:
            self.__init__(self.temporal_graph, self.target_time_safe_margin)

        if (self.start_point is None or reset_all is True):
            self.set_start_info()

        if (self.target_point is None or reset_all is True):
            self.set_target_info()
        
        if len(objective_path) > 0:
            self.agent = Agent(adjacent_matrix = self.temporal_graph.transform(self.start_time), start_time=self.start_time, objective_path=objective_path, centers=self.temporal_graph.centers)
        else:
            self.agent = Agent(adjacent_matrix = self.temporal_graph.transform(self.start_time), start_time=self.start_time, departure=self.start_node, centers=self.temporal_graph.centers)
        
        if reset_all is False:
            self.start_add_time = 0
            self.target_add_time = 0
            self.cur_adj_mat = None     # (Version 4.0 Update)
            self.iter = 0

            # true_info
            self.true_req_leaving_time = None
            self.true_leaving_info = {}

            # cur_info
            self.t = 0
            self.cur_point = None
            self.cur_node = None
            self.cur_time = None
            self.cur_path = None
            self.req_leaving_time = None

            # TimeMachine
            self.call_time_TimeMachine = None
            self.path_TimeMachine = None
            self.path_time_TimeMachine = None
            self.req_leaving_time_TimeMachine = None

            # LastAPI
            self.call_time_LastAPI = None
            self.call_node_LastAPI = None
            self.call_point_LastAPI = None 
            self.path_LastAPI = None
            self.path_time_LastAPI = None
            # full info
            self.req_leaving_time_full_info = None

            # loop
            self.loop = True
            self.loop_history = True
            self.loop_full_info = True

            # save history
            self.save_t = np.array([]).astype(bool)   # (Version 4.1 Update)
            self.history = []
            self.history_api_call = np.array([]).astype(bool)  # (Version 4.1 Update)
            self.history_time_machine = []
            self.history_full_info = []
            self.history_policy = []

            # algorithm
            self.policy_f = None

        # (Version 4.1 Update) -----------------
        if save_interval is not None:
            self.save_interval = save_interval
        
        self.simul_status = "ready"
        # --------------------------------------
        
        if verbose:
            print(f"[{format_time_to_str(self.start_time)}] {self.start_node}: ({self.start_point[0]:.2f}, {self.start_point[1]:.2f})", end=" ")
            print("→", end=" ")
            print(f"[{format_time_to_str(self.target_time)}] {self.target_node}: ({self.target_point[0]:.2f}, {self.target_point[1]:.2f})", end=" ")
            print(f" ▶ total {self.target_time - self.start_time} rounds.")

        self.cur_node = self.agent.cur_node
        self.cur_point = self.agent.cur_coordinates(self.temporal_graph.centers)
        self.cur_time = self.start_time

    def run_time_machine(self, temporal_graph=None, start_node=None, target_node=None, target_time=None, target_time_safe_margin=None, start_add_time=None, target_add_time=None, save_instance=True):
        temporal_graph = self.temporal_graph if temporal_graph is None else temporal_graph
        start_node = self.start_node if start_node is None else start_node
        target_node = self.target_node if target_node is None else target_node
        target_time = self.target_time if target_time is None else target_time
        target_time_safe_margin = self.target_time_safe_margin if target_time_safe_margin is None else target_time_safe_margin

        start_add_time = (0 if self.start_add_time is None else self.start_add_time) if start_add_time is None else start_add_time
        target_add_time = (0 if self.target_add_time is None else self.target_add_time) if target_add_time is None else target_add_time

        if (start_node is not None and target_node is not None and\
             target_time is not None and start_add_time is not None and target_add_time is not None):
            # first predict from target_time
            pre_adj_TimeMachine = temporal_graph.transform_oracle(target_time)
            pre_dijkstra_TimeMachine = Dijkstra(pre_adj_TimeMachine)
            pre_shortest_time_TimeMachine, pre_path_TimeMachine = pre_dijkstra_TimeMachine.dijkstra(start_node, target_node)
            pre_total_time_TimeMachine = pre_shortest_time_TimeMachine + start_add_time + target_add_time

            # second predict from adjusted predict leaving time
            call_time_TimeMachine = int(np.round(target_time - target_time_safe_margin - pre_total_time_TimeMachine))
            adj_TimeMachine = temporal_graph.transform_oracle(call_time_TimeMachine)
            dijkstra_TimeMachine = Dijkstra(adj_TimeMachine)
            shortest_time_TimeMachine, path_TimeMachine = dijkstra_TimeMachine.dijkstra(start_node, target_node)

            path_time_TimeMachine = shortest_time_TimeMachine + start_add_time + target_add_time
            req_leaving_time_TimeMachine = target_time - target_time_safe_margin - path_time_TimeMachine

            # result data
            if save_instance:
                self.call_time_TimeMachine = call_time_TimeMachine
                self.path_TimeMachine = path_TimeMachine
                self.path_time_TimeMachine = path_time_TimeMachine
                self.req_leaving_time_TimeMachine = req_leaving_time_TimeMachine

                if self.req_leaving_time is None:
                    self.req_leaving_time = req_leaving_time_TimeMachine


            return {"call_time":call_time_TimeMachine,
                    "path":path_TimeMachine,
                    "path_time":path_time_TimeMachine,
                    "req_leaving_time":req_leaving_time_TimeMachine}
        else:
            print("There is no one of needed arguments (start_node, target_node, target_time, start_add_time, target_add_time)")

    # (Version 4.1 Update)
    def api_call(self, adjacent_matrix=None, cur_node=None, target_node=None, agent=None,
                start_add_time=None, target_add_time=None, save_instance=True, save_history=True):
        adjacent_matrix = self.temporal_graph.transform(self.cur_time) if adjacent_matrix is None else adjacent_matrix
        cur_node = self.cur_node if cur_node is None else cur_node
        target_node = self.target_node if target_node is None else target_node
        agent = self.agent if agent is None else agent
        start_add_time = (0 if self.start_add_time is None else self.start_add_time) if start_add_time is None else start_add_time
        target_add_time = (0 if self.target_add_time is None else self.target_add_time) if target_add_time is None else target_add_time

        dijkstra = Dijkstra(graph = adjacent_matrix)
        if agent is None or agent.status == 'stop' or agent.status == 'hold':
            pred_time, pred_path = dijkstra(cur_node, target_node)
        elif agent.status == 'move':
            pred_time_, pred_path_ = dijkstra(agent.cur_dest, target_node)
            pred_time = pred_time_ + agent.remain_distance
            pred_path = [agent.cur_node] + pred_path_

        call_time_LastAPI = self.cur_time
        call_node_LastAPI = cur_node
        call_point_LastAPI = agent.cur_point
        path_LastAPI = pred_path
        path_time_LastAPI = pred_time + start_add_time + target_add_time
        req_leaving_time = self.target_time - self.target_time_safe_margin - path_time_LastAPI
        
        if save_instance:
            self.call_time_LastAPI = call_time_LastAPI
            self.call_node_LastAPI =  call_node_LastAPI
            self.call_point_LastAPI = call_point_LastAPI
            self.path_LastAPI = path_LastAPI
            self.path_time_LastAPI = path_time_LastAPI
            self.req_leaving_time = req_leaving_time

        # (Version 4.1 Update) -------------------------------------------------
        if save_history:
            # save api_call history
            if len(self.history_api_call) < self.t + 1:
                self.history_api_call = np.append(self.history_api_call, True)
            elif len(self.history_api_call) == self.t + 1:
                self.history_api_call[-1] = True
        # ---------------------------------------------------------------------
            

        return {"call_time": call_time_LastAPI,
                "call_node": call_node_LastAPI,
                "call_point": call_point_LastAPI,
                "path": path_LastAPI,
                "path_time": path_time_LastAPI,
                "req_leaving_time" : req_leaving_time}

    def visualize(self, temporal_graph=None, cur_time=None, title=None, return_plot=False):
        temporal_graph = self.temporal_graph if temporal_graph is None else temporal_graph
        cur_time = self.cur_time if cur_time is None else cur_time
        # cur_adj_mat = self.temporal_graph.transform(cur_time)
        # base_adj_mat = self.temporal_graph.transform(0)

        fig = self.agent.visualize(temporal_graph=self.temporal_graph, cur_time = cur_time, return_plot=return_plot)
        if return_plot:
            return fig

    def history_visualize(self, t, return_plot=False):
        fig = self.agent.history_visualize(temporal_graph=self.temporal_graph, t=t, start_time=self.start_time, return_plot=return_plot)
        if return_plot:
            return fig

    # (Version 4.3 Update)
    def save_data(self, instance_name, save_data=True, save_plot=False, **kwargs):
        """
        instance_name : 'history', 'history_time_machine', 'history_full_info', 'history_policy'
        group : 'time_machine', 'algorithm', 'full'
        **kwargs : group, start_time, target_time, start_node, target_node, start_point, target_point, 
                call_time_TimeMachine, path_TimeMachine, path_time_TimeMachine, 
                cur_time, cur_node, cur_point, 
                call_time_LastAPI, call_node_LastAPI, call_point_LastAPI, path_LastAPI, path_time_LastAPI, 
                weather, event, action, reward
        """
        history = {"id" : self.id, "group" : None, "round": self.t,
                "start_time" : self.start_time, "target_time" : self.target_time,
                "start_node" : self.start_node, "target_node" : self.target_node, "start_point" : self.start_point, "target_point" : self.target_point,
                "call_time_TimeMachine" : None, "path_TimeMachine" : None, "path_time_TimeMachine" : None,
                "cur_time" : self.cur_time, "cur_node" : self.cur_node, "cur_point" : self.cur_point,
                "call_time_LastAPI" : None, "call_node_LastAPI" : None, "call_point_LastAPI" : None, "path_LastAPI" : None, "path_time_LastAPI" : None,
                "req_leaving_time": None, "weather" : None, "event" : None, "action":None, "reward":None}
        
        if self.call_time_TimeMachine is not None:
            history["call_time_TimeMachine"] = self.call_time_TimeMachine
            history["path_TimeMachine"] = self.path_TimeMachine
            history["path_time_TimeMachine"] = self.path_time_TimeMachine
        
        if self.call_time_LastAPI is not None:
            history["call_time_LastAPI"] = self.call_time_LastAPI
            history["call_node_LastAPI"] = self.call_node_LastAPI
            history["call_point_LastAPI"] = self.call_point_LastAPI
            history["path_LastAPI"] = self.path_LastAPI
            history["path_time_LastAPI"] = self.path_time_LastAPI
        
        if instance_name == 'history_time_machine':
            history["req_leaving_time"] = self.req_leaving_time_TimeMachine
        elif instance_name == 'history':
            history["req_leaving_time"] = self.req_leaving_time
        elif instance_name == 'history_full_info':
            history["req_leaving_time"] = self.req_leaving_time_full_info

        if instance_name != 'history_time_machine':
            if self.t == 0:
                history['event'] = "register_schedule"
            elif (history['req_leaving_time'] is not None) and (self.cur_time >= history['req_leaving_time']): 
                history['event'] = "leaving_required"
                

        for key, value in kwargs.items():
            history[key] = value

        if save_data:
            # if history['req_leaving_time'] is not None and self.cur_time >= history['req_leaving_time']:
            #     if instance_name == 'history':
            #         self.loop_history = False
            #     elif instance_name == 'history_full_info':
            #         self.loop_full_info = False
            
            if instance_name == 'history' :
                if len(self.save_t) < self.t + 1:
                    self.save_t = np.append(self.save_t, True)
                elif len(self.save_t) == self.t + 1:
                    self.save_t[-1] = True

        history['start_time'] = format_time_to_str(history['start_time'])
        history['target_time'] = format_time_to_str(history['target_time'])
        history['cur_time'] = format_time_to_str(history['cur_time'])
        history['call_time_TimeMachine'] = format_time_to_str(history['call_time_TimeMachine'])
        history['call_time_LastAPI'] = format_time_to_str(history['call_time_LastAPI'])
        history["req_leaving_time"] = format_time_to_str(history["req_leaving_time"])

        if save_plot:
            history["plot"] = self.visualize(return_plot=True)

        if save_data:
            if hasattr(self, instance_name) and instance_name in ['history', 'history_time_machine', 'history_full_info', 'history_policy']:
                instance = getattr(self, instance_name)
                instance.append(history)

        return history

    # (Version 4.3 Update)
    def save_full_info(self, save_plot=False):
        cur_adj_mat = self.temporal_graph.transform(self.cur_time)

        full_api_call_dict = self.api_call(cur_adj_mat, save_instance=False, save_history=False)
        self.req_leaving_time_full_info = full_api_call_dict['req_leaving_time']

        # full_event = None
        # if self.t == 0:
        #     full_event = "register_schedule"
        # elif self.cur_time > full_api_call_dict['req_leaving_time']: 
        #     full_event = "leaving_required"
        self.save_data(instance_name="history_full_info", group="full",
            call_time_LastAPI = full_api_call_dict["call_time"], 
            call_node_LastAPI = full_api_call_dict["call_node"], 
            call_point_LastAPI = full_api_call_dict["call_point"], 
            path_LastAPI = full_api_call_dict["path"], 
            path_time_LastAPI = full_api_call_dict["path_time"],
            save_plot = save_plot)

    # (Version 4.3 Update)
    def step(self, agent_move=False, save_plot=False):
        if self.simul_status == 'run':

            # agent move
            self.cur_adj_mat = self.temporal_graph.transform(self.cur_time)

            self.agent.update_graph(self.cur_adj_mat)
            if agent_move:
                self.agent.move()
            else:
                self.agent.hold()
            
            # save api_call history
            if len(self.history_api_call) < self.t + 1:
                self.history_api_call = np.append(self.history_api_call, False)

            if len(self.save_t) < self.t + 1:
                self.save_t = np.append(self.save_t, False)

            # round_step ---------------------------------------------
            self.cur_time += 1
            self.t = self.agent.t

            # current info update ----------------------------
            self.cur_node = self.agent.cur_node
            self.cur_point = self.agent.cur_point
        else:
            raise Exception("being able to run at only 'run' status.")

    # (Version 4.3 Update)
    def run(self, agent_move=False, policy=None, save_interval=None, auto_step=False,
            visualize=False, save_plot=False):
        save_interval = self.save_interval if save_interval is None else save_interval

        # (Version 4.1 Update) ---------------------------------
        # iteration limitation
        self.iter += 1
        if self.iter >= self.iteration_limit:
            self.loop = False

        # if not step after run
        if self.simul_status == 'run':
            history_return = self.step(agent_move=agent_move, save_plot=save_plot)
        # ------------------------------------------------------

        history_return = None

        if self.t == 0:
            if self.cur_time is None:
                self.reset_state()
            self.run_time_machine(save_instance=True) # time_machine
            self.save_data(instance_name="history_time_machine", group="TimeMachine", event="TimeMachine", save_plot=save_plot)

        # full info
        if  (self.t % save_interval == 0) or (self.cur_time >= self.req_leaving_time_full_info and self.loop_full_info is True):
            self.save_full_info(save_plot=save_plot)

            if self.cur_time >= self.req_leaving_time_full_info:
                self.loop_full_info = False

        # history
        if  (self.t % save_interval == 0) or (self.cur_time >= self.req_leaving_time and self.loop_history is True):
            history_return = self.save_data(instance_name='history', group="history", save_plot=save_plot)

            if self.cur_time >= self.req_leaving_time:
                if self.loop_history is True:
                    cur_adj_mat = self.temporal_graph.transform(self.cur_time)
                    api_call_dict = self.api_call(cur_adj_mat, save_instance=False, save_history=False)
                    
                    # true time
                    self.true_req_leaving_time = self.target_time - self.target_time_safe_margin - api_call_dict['path_time']
                    self.true_leaving_info = api_call_dict

                    if len(self.history_api_call) < self.t + 1:
                        self.history_api_call = np.append(self.history_api_call, False)
                self.loop_history = False
        
        # Loop
        if (self.loop_full_info is False) and (self.loop_history is False):
            self.loop = False

        # visualize
        if visualize:
            self.visualize()

        self.simul_status = 'run'   # (Version 4.1 Update)

        # progress to next step
        if auto_step:
            self.step(agent_move=agent_move, save_plot=save_plot)

        return history_return




if simulation:
    # (Initial Setting & Parameter Setting) -----------------------------------------------------------------------------------------------
    random_state = None
    rng = np.random.RandomState(random_state)      # (RandomState) ★ Set_Params
    # n_nodes = 50                        # (Node수) ★ Set_Params

    total_period = 7*24*60              # (전체 Horizon) ★ Set_Params
    # T = 24*60                           # (주기) ★ Set_Params

    # (create base graph) 
    node_map = GenerateNodeMap(n_nodes, random_state)
    node_map.create_node(node_scale=50, cov_knn=3)           # create node
    node_map.create_connect(connect_scale=0)             # create connection

    # (periodic setting)
    periodic_f = periodic_improve    # periodic function

    # (random noise)
    # random_noise_f = RandomNoise(scale=1/30, random_state=random_state)       # (Version 2.1)
    random_noise_f = RandomNoise(scale=1/1000, random_state=random_state)      # (Version 2.2)
    # random_noise_f = RandomNoise(scale=1/300)                                 # (Version 2.2)

    # (aggregated route_graph instance)
    temporal_graph = TemporalGraph(node_map=node_map, amp_class=TemporalGaussianAmp,
                                    periodic_f=periodic_f, error_f=random_noise_f)
    # (temporal gaussian amplitude)
    temporal_graph.create_amp_instance(T_arr = np.arange(24*60), mean_center=[0.5,0.5], mean_r_f=mean_r_f, cov_f=cov_scale_f,
                            centers=node_map.centers, base_adj_mat=node_map.adj_matrix,
                            normalize=1.5, adjust=True, repeat=9)
    temporal_graph.make_temporal_observations()

    visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)


if simulation:
    # Simulation ----------------------------------------------------------------
    simul = TemporalMapSimulation(temporal_graph)
    simul.reset_state(reset_all=False, save_interval=3, verbose=1)
    simul.reset_state(reset_all=True, save_interval=3, verbose=1)

    # simul.run()
    # simul.run(visualize=True)
    # simul.run(agent_move=True, visualize=True)
    # simul.run(agent_move=True)

    # pd.DataFrame(simul.history)
    # pd.DataFrame(simul.history_full_info)


    # simul.reset_state(verbose=1)
    # simul.reset_state(reset_all=True, verbose=1)
    # format_time_to_str(simul.start_time)

    # simul.run_time_machine()
    # simul.api_call()
    # simul.visualize()

    for t in range(simul.target_time - simul.start_time):
        simul.run()


    while(simul.loop):
        # if simul.t == 0:
        #     simul.api_call()
        simul.run()
        if simul.t > 0 and simul.t % 30 == 0:
            simul.api_call()

    # for _ in range(simul.target_time - simul.start_time):
    #     simul.run()
    #     if simul.t > 0 and simul.t % 30 == 0:
    #         simul.api_call()
    # simul.history_api_call
    # simul.history_visualize(t=55)

    pd.DataFrame(simul.history_time_machine).to_clipboard()
    pd.DataFrame(simul.history).to_clipboard()
    pd.DataFrame(simul.history_full_info).to_clipboard()

    # true_value
    simul.true_req_leaving_time
    format_time_to_str(simul.true_req_leaving_time)
    simul.true_leaving_info
    
    #
    from IPython.display import clear_output
    import time

    i = 0
    while (simul.loop):
        if i % 60 == 0:
            simul.run(agent_move=True)
            simul.api_call()
            simul.visualize()
            clear_output(wait=True)
        else:
            simul.run(agent_move=True)
        
        if simul.loop is False:
            simul.visualize()
        i += 1


    pd.DataFrame(simul.history_time_machine).to_clipboard()
    pd.DataFrame(simul.history).to_clipboard()
    pd.DataFrame(simul.history_full_info).to_clipboard()




# # (error analysis) --------------------------------------------------------------------------------------
# object = result_history['240730_223027_2']

# object['leaving_graph']

# leaving_path = object['leaving_path']

# leaving_oracle = np.zeros_like(object['temporal_oracle'][:,0,0])
# leaving_obs = np.zeros_like(object['temporal_observations'][:,0,0])

# for i in range(len(leaving_path)-1):
#     leaving_oracle += object['temporal_oracle'][:, leaving_path[i], leaving_path[i+1]]
#     leaving_obs += object['temporal_observations'][:, leaving_path[i], leaving_path[i+1]]
    
# visualize_periodic(leaving_obs, leaving_oracle, title=leaving_path, return_plot=False)







###############################################################################################################
# LinUCB  #####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# LinUCB Final Version
class LinUCB:
    def __init__(self, n_actions=2, feature_dim=0, 
            shared_theta=False, shared_context=True, 
            alpha=None, allow_duplicates=True, random_state=None):

        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
    
        self.n_actions = None if shared_theta and shared_context else n_actions   # Number of arms
        self.feature_dim = feature_dim  # Dimension of context vectors

        self.shared_context = shared_context
        self.shared_theta = shared_theta

        self.t = 0

        self.alpha_update_auto = True if alpha is None else False
        if self.shared_theta:
            self.A = np.identity(feature_dim)
            self.b = np.zeros(feature_dim)
            # self.theta = np.zeros(feature_dim)
            self.theta = self.rng.normal(0, 0.3, size=feature_dim)
            
            self.alpha = 1 if self.alpha_update_auto else alpha
        else:
            self.A = np.tile(np.identity(feature_dim)[np.newaxis,...],(n_actions,1,1))
            self.b = np.tile(np.zeros((1, feature_dim)),(n_actions,1))
            # self.theta = np.zeros((n_actions, feature_dim))
            self.theta = self.rng.normal(0, 0.3, size=(n_actions, feature_dim))

            self.alpha = np.ones(n_actions) if self.alpha_update_auto else alpha
        
        self.batched_contexts = np.array([])
        self.matched_action_for_batched_contexts = np.array([]).astype(int)
        self.batched_actions = np.array([]).astype(int)
        self.batched_actions_for_rewards_match = np.array([]).astype(int)
        self.batched_rewards = np.array([])

        self.history = {}
        self.history['mu'] = []
        self.history['var'] = []
        self.history['ucb'] = []
        self.history['actions'] = np.array([]).astype(int)
        self.history['rewards'] = np.array([])

        self.allow_duplicates = allow_duplicates

    def calc_mu(self, context, action=None):
        context = np.array(context).astype(float)
        if context.ndim == 1:
            context = context[np.newaxis, ...]

        mu = context @ self.theta.T

        if self.shared_theta:
            return mu.reshape(-1,1)    #(n, 1)
        else:
            if action is None:
                return mu   #(n, a)
            else:
                return mu[:,action]   # (n, 1), (n, a)
    
    def calc_var(self, context, action=None):
        context = np.array(context).astype(float)
        if context.ndim == 1:
            context = context[np.newaxis, ...]
        A_inv = np.linalg.inv(self.A)

        if self.shared_theta:
            var = np.einsum("nd,nd->n", np.einsum("nd, de->ne", context, A_inv), context)  # (n,)
            return var.reshape(-1,1)      # (n, 1)
        else:
            var = np.einsum("nad,nd->na", np.einsum("nd, ade->nae", context, A_inv), context) # (n,a)
            if action is None:
                return var  #(n, a)
            else:
                return var[:,action]   # (n, 1), (n, a)

    def predict(self, context, action=None, return_mu=True, return_var=True, return_ucb=False):
        mu = self.calc_mu(context, action)
        var = self.calc_var(context, action)
        ucb = mu + self.alpha * np.sqrt(var)

        return_elements = []
        if return_mu:
            return_elements.append(mu)
        if return_var:
            return_elements.append(var)
        if return_ucb:
            return_elements.append(ucb)
        return tuple(return_elements)

    def update_instance(self, instance_name, value, key=None, reshape_dim=None):
        if hasattr(self, instance_name):
            instance = getattr(self, instance_name)

            if key is None:
                if len(instance) == 0:
                    setattr(self, instance_name, value.reshape(-1, reshape_dim))
                else:
                    while (value.shape[-1] > instance.shape[-1]):
                        instance = getattr(self, instance_name)
                        setattr(self, instance_name, np.pad(instance, ((0,0),(0,1)), mode='constant', constant_values=np.nan))
                    setattr(self, instance_name, np.append(instance, value.reshape(-1, reshape_dim), axis=0))
            else:
                if len(instance[key]) == 0:
                    instance[key] = value.reshape(-1, reshape_dim)
                else:
                    while (value.shape[-1] > instance[key].shape[-1]):
                        instance[key] = np.pad(instance[key], ((0,0),(0,1)), mode='constant', constant_values=np.nan)
                    instance[key] = np.append(instance[key], value.reshape(-1, reshape_dim), axis=0)

    def observe_context(self, context, action=None):
        context = np.array(context).astype(float)
        if context.ndim == 1:
            context = context[np.newaxis, ...].astype(float)
        len_context = len(context)
        len_action = 1 if np.array(action).ndim == 0 else len(np.array(action))

        if (self.shared_context is False) and (action is not None) and (len_context != len_action):
            raise Exception("input action(s) must have same length with input context(s)")

        elif (self.shared_context is False) and (action is None) and (len_context != self.n_actions):
            raise Exception("context(s) must input with corresponding action(s) when sharing parameters")

        else:
            if self.shared_context:
                fill_matched_action = np.full(len_context, np.nan)
            elif (action is not None) and (len_context == len_action):
                if np.array(action).max() >= self.n_actions:
                    raise Exception(f"invalid action input (possible actions: 0~{self.n_actions-1})")
                fill_matched_action = action
            elif (action is None) and (len_context == self.n_actions):
                fill_matched_action = np.arange(self.n_actions)
            else:
                print("else condition occurs!")

            self.update_instance('batched_contexts', context, reshape_dim=self.feature_dim)
            self.matched_action_for_batched_contexts = np.append(self.matched_action_for_batched_contexts, fill_matched_action)
        return context

    def select_action(self, context=None, action=None, allow_duplicates=None, verbose=0):
        if context is not None:
            context = self.observe_context(context, action)
        allow_duplicates = self.allow_duplicates if allow_duplicates is None else allow_duplicates
        len_contexts = len(context)

        mu, var, ucb = self.predict(context, action, return_ucb=True)

        if self.shared_context:        # select action
            if self.shared_theta:
                mask = (ucb == ucb.max())
                if allow_duplicates:
                    action = np.argmax(ucb)
                else:
                    action = np.random.choice(np.flatnonzero(mask))
                self.t += 1
            else:
                mask = (ucb == np.max(ucb, axis=1, keepdims=True))
                if allow_duplicates:
                    action = np.argmax(ucb,axis=1)
                else:
                    counts = mask.sum(axis=1,keepdims=True)
                    action = np.apply_along_axis(lambda x: np.random.choice(np.flatnonzero(x)), axis=1, arr=mask)
                self.t += len_contexts
        else:       # designated action
            action = self.matched_action_for_batched_contexts
            if self.shared_theta:
                action_idx = (action,0)
            else:
                action_idx = (np.arange(len(action)), action)
            self.t += len_contexts
        
        # reset matched_action_for_batched_contexts
        self.matched_action_for_batched_contexts = np.array([]).astype(int)

        # if (self.shared_context is True) and (self.shared_theta is False):
        if self.shared_context:
            dim = len(context)  if self.shared_theta is True else self.n_actions
            self.history['mu'].append(mu)
            self.history['var'].append(var)
            self.history['ucb'].append(ucb)
        else:
            self.history['mu'].append(mu[action_idx])
            self.history['var'].append(var[action_idx])
            self.history['ucb'].append(ucb[action_idx])
        self.history['actions'] = np.append(self.history['actions'], action)

        self.batched_actions = np.append(self.batched_actions, action)
        self.batched_actions_for_rewards_match = np.append(self.batched_actions_for_rewards_match, action)

        if verbose:
            print(f"action: {action}", end='\t')
        return action

    def observe_reward(self, reward=None, reward_f=None, verbose=0):
        if len(self.batched_actions_for_rewards_match) > 0:
            reward_save = None

            if reward is not None:              # directly injected reward
                array_reward = np.array(reward)

                if array_reward.ndim == 0:      # scalar input
                    len_reward = 1
                    reward_save = array_reward.copy()
                    self.batched_rewards = np.append(self.batched_rewards, reward_save)
                    self.batched_actions_for_rewards_match = self.batched_actions_for_rewards_match[1:]

                    if (self.shared_theta is True) and (self.shared_context is True):
                        action = self.history['actions'][-1]
                        self.batched_contexts = self.batched_contexts[[action]]
                        self.history['mu'][-1] = self.history['mu'][-1][action]
                        self.history['var'][-1] = self.history['var'][-1][action]
                        self.history['ucb'][-1] = self.history['ucb'][-1][action]

                elif array_reward.ndim == 1:
                    if (len(array_reward) == self.n_actions):
                        if len(self.batched_actions_for_rewards_match) == 1:      # scalar input
                            len_reward = 1
                            reward_save = array_reward[self.batched_actions_for_rewards_match[0]]
                            self.batched_rewards = np.append(self.batched_rewards, reward_save)
                            self.batched_actions_for_rewards_match = self.batched_actions_for_rewards_match[1:]
                        else:
                            print('Confused reward argument. Transform the reward observation to (-1,1) shape.')
                    elif (self.shared_theta is True) and (self.shared_context is True):
                        len_reward = len(array_reward)

                        if (len_reward == 1) or (len_reward == len(self.batched_contexts)):
                            reward_save = array_reward.copy()

                            if len_reward == 1:
                                action = self.history['actions'][-1]
                                self.batched_contexts = self.batched_contexts[[action]]
                                self.history['mu'][-1] = self.history['mu'][-1][action]
                                self.history['var'][-1] = self.history['var'][-1][action]
                                self.history['ucb'][-1] = self.history['ucb'][-1][action]
                                self.batched_rewards = np.append(self.batched_rewards, reward_save)

                            elif len_reward == len(self.batched_contexts):
                                self.batched_actions = np.arange(len_reward)
                                self.batched_rewards = np.append(self.batched_rewards, reward_save)
                        else:
                            print('Reward must have 1 lenth or contexts length.')

                    else:
                        len_reward = len(array_reward)
                        if len_reward <= len(self.batched_actions_for_rewards_match):         # array input
                            reward_save = array_reward.copy()
                            self.batched_rewards = np.append(self.batched_rewards, reward_save)
                            self.batched_actions_for_rewards_match = self.batched_actions_for_rewards_match[len_reward:]
                        else:
                            print('Exceeds required length of reward observations.')
                            
                elif array_reward.ndim == 2:
                    len_reward = len(array_reward)
                    if (self.shared_theta is True) and (self.shared_context is True):
                        if (len_reward == 1) or (len_reward == len(self.batched_contexts)):
                            reward_save = array_reward.ravel()
                            if len_reward == 1:
                                action = self.history['actions'][-1]
                                self.batched_contexts = self.batched_contexts[[action]]
                                self.history['mu'][-1] = self.history['mu'][-1][action]
                                self.history['var'][-1] = self.history['var'][-1][action]
                                self.history['ucb'][-1] = self.history['ucb'][-1][action]
                                self.batched_rewards = np.append(self.batched_rewards, reward_save)

                            elif len_reward == len(self.batched_contexts):
                                self.batched_actions = np.arange(len_reward)
                                self.batched_rewards = np.append(self.batched_rewards, reward_save)
                        else:
                            print('Reward must have 1 lenth or contexts length.')

                    elif len_reward <= len(self.batched_actions_for_rewards_match):
                        if array_reward.shape[1] == 1:                                  # array input
                            reward_save = array_reward.ravel()
                        else:                                                           # matrix input
                            reward_save = array_reward[np.arange(len(self.batched_actions_for_rewards_match[:len_reward])), self.batched_actions_for_rewards_match[:len_reward]]

                        self.batched_rewards = np.append(self.batched_rewards, reward_save)
                        self.batched_actions_for_rewards_match = self.batched_actions_for_rewards_match[len_reward:]
                    else:
                        print('Exceeds required length of reward observations.')

            elif reward_f is not None:          # reward from functional call
                reward_list = []
                for action in self.batched_actions:
                    reward_list.append(reward_f(action))
                reward_save = np.array(reward_list)
                self.batched_rewards = np.append(self.batched_rewards, reward_save)

            self.history['rewards'] = np.append(self.history['rewards'], reward_save)
            if verbose:
                print(f"reward: {reward_save}", end='\t')
        else:
            print("Rewards corresponding to action have already been matched.")

    def update_params(self):
        if len(self.batched_contexts) == len(self.batched_actions) == len(self.batched_rewards):
            len_update_data = len(self.batched_rewards)

            if self.shared_theta:
                contexts = self.batched_contexts
                rewards = self.batched_rewards

                self.A = self.A + contexts.T @ contexts
                self.b = self.b + rewards @ contexts
                self.theta = np.linalg.inv(self.A) @ self.b

                if self.alpha_update_auto:
                    residual = np.array(self.history['rewards']) - np.array(self.history['mu'])
                    if len(residual) > 1:
                        self.alpha = residual.std()
            else:
                for action in np.unique(self.batched_actions):
                    idx_filter = (self.batched_actions == action)
                    contexts = self.batched_contexts[idx_filter]
                    rewards = self.batched_rewards[idx_filter]

                    self.A[action] = self.A[action] + contexts.T @ contexts
                    self.b[action] = self.b[action] + rewards @ contexts
                    self.theta[action] = np.linalg.inv(self.A[action]) @ self.b[action]

                    if self.alpha_update_auto:
                        action_TF = self.history['actions'] == action
                        if np.sum(action_TF) > 1:
                            if self.shared_context:
                                mu = self.history['mu'][action_TF][:, action]
                            else:
                                mu = self.history['mu'][action_TF]
                            residual = self.history['rewards'][action_TF] - mu
                            self.alpha[action] = residual.std()
            
            self.batched_contexts = np.array([])
            self.matched_action_for_batched_contexts = np.array([]).astype(int)
            self.batched_actions = np.array([]).astype(int)
            self.batched_actions_for_rewards_match = np.array([]).astype(int)
            self.batched_rewards = np.array([])

    def undo(self):
        if len(self.batched_contexts) > 0:
            len_batched = len(self.batched_contexts)
            self.t -= len_batched

            if len_batched == len(self.history['actions']):
                self.history['mu'] = self.history['mu'][:-len_batched]
                self.history['var'] = self.history['var'][:-len_batched]
                self.history['ucb'] = self.history['ucb'][:-len_batched]
                self.history['actions'] = self.history['actions'][:-len_batched]

            if len_batched == len(self.history['rewards']):
                self.history['rewards'] = self.history['rewards'][:-len_batched]

            self.batched_contexts = np.array([])
            self.matched_action_for_batched_contexts = np.array([]).astype(int)
            self.batched_actions = np.array([]).astype(int)
            self.batched_actions_for_rewards_match = np.array([]).astype(int)
            self.batched_rewards = np.array([])

    def run(self, context, reward=None, action=None, reward_f=None, update=True, allow_duplicates=None, verbose=0):
        allow_duplicates = self.allow_duplicates if allow_duplicates is None else allow_duplicates

        if verbose:
            print(f"(step {self.t}) ", end="")
        
        self.select_action(context=context, action=action, allow_duplicates=allow_duplicates, verbose=verbose)

        if (reward is not None) or (reward_f is not None):
            self.observe_reward(reward, reward_f, verbose=verbose)
        if update:
            self.update_params()
        if verbose:
            print()


if example:
    # Parameters
    alpha = 0.1  # Exploration parameter
    feature_dim = 5
    n_actions = 3

    contexts = np.random.randn(1000, feature_dim)
    true_theta = np.stack([np.random.randn(feature_dim) for _ in range(n_actions)])
    rewards = np.stack([context @ true_theta.T + + np.random.randn() for context in contexts])
    
    # one instruction pass -------------------------------------------------
    # lucb = LinUCB(n_actions, feature_dim, alpha=alpha, allow_duplicates=False)
    lucb = LinUCB(n_actions, feature_dim, shared_theta=False, shared_context=True, allow_duplicates=False)

    lucb.run(contexts[0], rewards[0], verbose=1)
    lucb.run(contexts[:10], rewards[:10], verbose=1)
    lucb.theta

    lucb.run(contexts[:10], rewards[:10], update=False, verbose=1)
    lucb.update_params()
    lucb.theta

    lucb.undo()
    lucb.t

    # separate instructions -------------------------------------------------
    lucb = LinUCB(n_actions, feature_dim, allow_duplicates=False)

    lucb.select_action(contexts[0])
    lucb.batched_contexts
    lucb.batched_actions
    lucb.batched_actions_for_rewards_match
    lucb.theta

    lucb.observe_reward(rewards[:2])
    lucb.batched_rewards
    lucb.batched_actions
    lucb.batched_actions_for_rewards_match
    lucb.theta

    lucb.update_params()
    lucb.theta


    # online learning with UCB -------------------------------------------------
    lucb = LinUCB(n_actions, feature_dim, allow_duplicates=False)

    for ei, (context ,reward) in enumerate(zip(contexts, rewards)):
        lucb.run(context, reward)
    lucb.theta

    action = 2
    plt.scatter( lucb.predict(contexts)[0][:,action], rewards[:,action], alpha=0.5)
    plt.show()

    for a in range(n_actions):
        plt.scatter( lucb.predict(contexts)[0][:,a], rewards[:,a], alpha=0.2, label=a)
    plt.legend()
    plt.show()
    # --------------------------------------------------------------------------------------------------------------------------------



def format_str_to_split(time_str, week_str=False):
    if time_str is None:
        return {'week':None, 'hour':None, 'min':None}
    else:
        week_dict = {"Mon.":0, "Tue.":1, "Wed.":2, "Thu.":3, "Fri.":4, "Sat.":5, "Sun.":6}

        week_string, hour_min_str = time_str.split(" ")
        hour, min = hour_min_str.split(":")
        
        return {'week' : week_string if week_str else week_dict[week_string],
                'hour': int(hour), 'min': int(min)}

def time_to_weekgroup(x, name=""):
    week_feature = {None: {f"{name}_Week": 0, f"{name}_Sat": 0, f"{name}_Sun": 0}}
    for i in range(7):
        if i < 5:
            week_feature[i] = {f"{name}_Week": 1, f"{name}_Sat": 0, f"{name}_Sun": 0}
        elif i == 5:
            week_feature[i] = {f"{name}_Week": 0, f"{name}_Sat": 1, f"{name}_Sun": 0}
        elif i == 6:
            week_feature[i] = {f"{name}_Week": 0, f"{name}_Sat": 0, f"{name}_Sun": 1}

    week_code = None if x is None else format_str_to_split(x)['week']
    return week_feature[week_code]

# def time_to_hourgroup(x, name=""):
#     hour_feature = {None: {f"{name}_Commute": 0, f"{name}_Lunch": 0, f"{name}_Day": 0, f"{name}_Night": 0}}
#     for i in range(24):
#         if (i>=7 and i<=9) or (i>=16 and i<19):
#             hour_feature[i] = {f"{name}_Commute": 1, f"{name}_Lunch": 0, f"{name}_Day": 0, f"{name}_Night": 0}
#         elif (i>=11 and i<=13):
#             hour_feature[i] = {f"{name}_Commute": 0, f"{name}_Lunch": 1, f"{name}_Day": 0, f"{name}_Night": 0}
#         elif (i>=23 and i<=5):
#             hour_feature[i] = {f"{name}_Commute": 0, f"{name}_Lunch": 0, f"{name}_Day": 0, f"{name}_Night": 1}
#         else:
#             hour_feature[i] = {f"{name}_Commute": 0, f"{name}_Lunch": 0, f"{name}_Day": 1, f"{name}_Night": 0}
#     hour_code = None if x is None else format_str_to_split(x)['hour']
#     return hour_feature[hour_code]

def time_to_hourgroup(x, name=""):
    hour_feature = {None: {f"{name}_Commute": 0, f"{name}_Day": 0, f"{name}_Night": 0}}
    for i in range(24):
        if (i>=7 and i<=9) or (i>=16 and i<19):
            hour_feature[i] = {f"{name}_Commute": 1, f"{name}_Day": 0, f"{name}_Night": 0}
        elif (i>=23 and i<=5):
            hour_feature[i] = {f"{name}_Commute": 0, f"{name}_Day": 0, f"{name}_Night": 1}
        else:
            hour_feature[i] = {f"{name}_Commute": 0, f"{name}_Day": 1, f"{name}_Night": 0}
    hour_code = None if x is None else format_str_to_split(x)['hour']
    return hour_feature[hour_code]


def hour_min_periodic(x):
    if x is None:
        x_h = np.inf
    else:
        x_h = format_str_to_split(x)['hour'] + format_str_to_split(x)['min']/60

    if x_h <= 1:
        t = 0
    elif x_h <= 9:
        t = (np.cos( (np.pi*2)/18 * (x_h-9) )/2)+0.5
    elif x_h <= 22:
        t = (np.cos( (np.pi*2)/9 * (x_h-9) )/2)+0.5
    else:
        t = 0
    return t


# week_feature = {0:[1,0,0], 1:[1,0,0], 2:[1,0,0], 3:[1,0,0], 4:[1,0,0], 5:[0,1,0], 6:[0,0,1], None: [0,0,0]}
# week_code = ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."]
# week_series = pd.Series(["Mon. 0:00", "Tue. 0:00", "Wed. 0:00", "Thu. 0:00", "Fri. 0:00", "Sat. 0:00", "Sun. 0:00", None])
# week_series.apply(lambda x: pd.Series(time_to_weekgroup(x)))

# T = [f"{w} {h:02d}:{m:02d}" for w, h, m in zip(np.random.choice(week_code, size=24), np.arange(24), np.random.randint(1,59, size=24))]
# T_series = pd.Series(np.append(T, None))

# T_series.apply(hour_min_periodic)
# plt.scatter(np.arange(25), T_series.apply(hour_min_periodic))


# feature_set commmon -------------------------------------------------------------------------------
def make_feature_set(dataframe):
    df = dataframe.copy()
    feature_set_common = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                        'cur_time_hour_min',
                        'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                        'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                        'target_time_hour_min',
                        'target_time_Commute', 'target_time_Day', 'target_time_Night',
                        'cur_point_lat','cur_point_lng', 'target_point_lat', 'target_point_lng']
    # df['cur_time']
    # df['cur_time_week'] = df['cur_time'].apply(lambda x: time_to_weekgroup(x, name='cur_time'))
    df = pd.concat([df, df['cur_time'].apply(lambda x: pd.Series(time_to_weekgroup(x, name='cur_time')))], axis=1)

    # df['cur_time_hour'] = df['cur_time'].apply(lambda x: time_to_hourgroup(x, name='cur_time') )
    df = pd.concat([df, df['cur_time'].apply(lambda x: pd.Series(time_to_hourgroup(x, name='cur_time')))], axis=1)
    df['cur_time_hour_min'] = df['cur_time'].apply(hour_min_periodic)

    # df['target_time']
    # df['target_time_week'] = df['target_time'].apply(lambda x: time_to_weekgroup(x, name='target_time'))
    df = pd.concat([df, df['target_time'].apply(lambda x: pd.Series(time_to_weekgroup(x, name='target_time')))], axis=1)

    # df['target_time_hour'] = df['target_time'].apply(lambda x: time_to_hourgroup(x, name='target_time') )
    df = pd.concat([df, df['target_time'].apply(lambda x: pd.Series(time_to_hourgroup(x, name='target_time')))], axis=1)
    df['target_time_hour_min'] = df['target_time'].apply(hour_min_periodic)

    # df['cur_point']
    # df['target_point']
    df['cur_point_lat'] = df['cur_point'].apply(lambda x: x[0])
    df['cur_point_lng'] = df['cur_point'].apply(lambda x: x[1])

    df['target_point_lat'] = df['target_point'].apply(lambda x: x[0])
    df['target_point_lng'] = df['target_point'].apply(lambda x: x[1])

    # df['remain_time'] = df['target_time'].apply(lambda x : format_str_to_time(x)) - df['cur_time'].apply(lambda x : format_str_to_time(x))

    # API_Call -------------------------------------------------------------------
    feature_set_LastAPI = ['call_time_LastAPI_Week', 'call_time_LastAPI_Sat', 'call_time_LastAPI_Sun',
                        'call_time_LastAPI_hour_min',
                        'call_time_LastAPI_Commute', 'call_time_LastAPI_Day', 'call_time_LastAPI_Night',
                        'call_point_LastAPI_lat', 'call_point_LastAPI_lng', 'path_time_LastAPI', 'movedist_LastAPI', 'remain_req_leaving_time_LastAPI']

    # df['call_time_LastAPI']
    # df['call_time_LastAPI_week'] = df['call_time_LastAPI'].apply(lambda x: time_to_weekgroup(x, name='call_time_LastAPI'))
    df = pd.concat([df, df['call_time_LastAPI'].apply(lambda x: pd.Series(time_to_weekgroup(x, name='call_time_LastAPI')))], axis=1)

    # df['call_time_LastAPI_hour'] = df['call_time_LastAPI'].apply(lambda x: time_to_hourgroup(x, name='call_time_LastAPI') )
    df = pd.concat([df, df['call_time_LastAPI'].apply(lambda x: pd.Series(time_to_hourgroup(x, name='call_time_LastAPI')))], axis=1)
    df['call_time_LastAPI_hour_min'] = df['call_time_LastAPI'].apply(hour_min_periodic)

    # df['call_point_LastAPI']
    # df['call_point_LastAPI_lat'] = df['call_point_LastAPI'].apply(lambda x: 0 if pd.isna(x) else x[0])
    # df['call_point_LastAPI_lng'] = df['call_point_LastAPI'].apply(lambda x: 0 if pd.isna(x) else x[1])
    df['call_point_LastAPI_lat'] = df['call_point_LastAPI'].apply(lambda x: 0 if np.array(x).ndim==0 else x[0])
    df['call_point_LastAPI_lng'] = df['call_point_LastAPI'].apply(lambda x: 0 if np.array(x).ndim==0 else x[1])


    df['path_time_LastAPI'] = df['path_time_LastAPI'].apply(lambda x: 0 if pd.isna(x) else x)
    df['movedist_LastAPI'] = ( df['cur_point'].apply(lambda x: np.array(x)) - df['call_point_LastAPI'].apply(lambda x: np.ones(2)*np.inf if x is None else np.array(x)) ).apply(lambda x: np.linalg.norm(x, ord=2)).apply(lambda x: 0 if x == np.inf else x)
    df['remain_req_leaving_time_LastAPI'] = df.apply(lambda x: 0 if pd.isna(x['path_time_LastAPI']) or  x['path_time_LastAPI'] ==0 else format_str_to_time(x['target_time']) - 8 - x['path_time_LastAPI'] - format_str_to_time(x['cur_time']) ,axis=1)

    # TimeMachine -------------------------------------------------------------------
    feature_set_TimeMachine = ['call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        'call_time_TimeMachine_hour_min',
                        'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine', 'movedist_TimeMachine', 'remain_req_leaving_time_TimeMachine']

    # df['call_time_TimeMachine']
    # df['call_time_TimeMachine_week'] = df['call_time_TimeMachine'].apply(lambda x: time_to_weekgroup(x, name='call_time_TimeMachine'))
    df = pd.concat([df, df['call_time_TimeMachine'].apply(lambda x: pd.Series(time_to_weekgroup(x, name='call_time_TimeMachine')))], axis=1)

    # df['call_time_TimeMachine_hour'] = df['call_time_TimeMachine'].apply(lambda x: time_to_hourgroup(x, name='call_time_TimeMachine') )
    df = pd.concat([df, df['call_time_TimeMachine'].apply(lambda x: pd.Series(time_to_hourgroup(x, name='call_time_TimeMachine')))], axis=1)
    df['call_time_TimeMachine_hour_min'] = df['call_time_TimeMachine'].apply(hour_min_periodic)

    # df['start_point']
    df['call_point_TimeMachine_lat'] = df['start_point'].apply(lambda x: x[0])
    df['call_point_TimeMachine_lng'] = df['start_point'].apply(lambda x: x[1])

    df['path_time_TimeMachine']
    df['movedist_TimeMachine'] = (df['cur_point'].apply(lambda x: np.array(x)) - df['start_point']).apply(lambda x: np.linalg.norm(x, ord=2))
    df['remain_req_leaving_time_TimeMachine'] = df.apply(lambda x : format_str_to_time(x['target_time'])  - 8 - x['path_time_TimeMachine'] - format_str_to_time(x['cur_time']), axis=1)

    return df



# ----------------------------------------------------------------------------------------------------

# feature_set_common : cur_time, target_time, cur_point, target_point
feature_set_common = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                        'cur_time_hour_min',
                        'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                        'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                        'target_time_hour_min',
                        'target_time_Commute', 'target_time_Day', 'target_time_Night',
                        'cur_point_lat','cur_point_lng', 'target_point_lat', 'target_point_lng']

# feature_set_LastAPI : call_time_LastAPI, call_point_LastAPI, path_time_LastAPI
feature_set_LastAPI = ['call_time_LastAPI_Week', 'call_time_LastAPI_Sat', 'call_time_LastAPI_Sun',
                        'call_time_LastAPI_hour_min',
                        'call_time_LastAPI_Commute', 'call_time_LastAPI_Day', 'call_time_LastAPI_Night',
                        'call_point_LastAPI_lat', 'call_point_LastAPI_lng', 'path_time_LastAPI', 'movedist_LastAPI', 'remain_req_leaving_time_LastAPI']

# feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
feature_set_TimeMachine = ['call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        'call_time_TimeMachine_hour_min',
                        'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine', 'movedist_TimeMachine', 'remain_req_leaving_time_TimeMachine']


if visualize:
    simul = TemporalMapSimulation(temporal_graph)
    simul.reset_state(reset_all=False, verbose=1)
    simul.reset_state(reset_all=True, verbose=1)

    simul.api_call()
    context = simul.run(save_interval=5)

    while(simul.loop):
        simul.run(save_interval=5)
        if simul.t % 30 == 0:
            simul.api_call()

    # pd.DataFrame(simul.history_time_machine).to_clipboard()
    # pd.DataFrame(simul.history).to_clipboard()
    # pd.DataFrame(simul.history_full_info).to_clipboard()

    context_df =  pd.Series(context).to_frame().T
    context_feature_df = make_feature_set(context_df)
    context_feature_df.to_clipboard()
    context_feature_df[feature_set_common].shape
    context_feature_df[feature_set_LastAPI].shape
    context_feature_df[feature_set_TimeMachine].shape

    context_feature_df[feature_set_common + feature_set_LastAPI + feature_set_TimeMachine]






############################################################################################################
import numpy as np
from scipy.optimize import fsolve

# def equations(v):
#     a, b, c, d = v
#     eq1 = a * 8**2 + b - c * (8 - d)**3     # f(8) = g(8)
#     eq2 = 2* a * 8 - 3 * c * (8 - d)**2     # f'(8) = g'(8)
#     eq3 = a - 0.5       # 선택된 추가 방정식들 (예: d의 정의를 활용)
#     eq4 = c - 3     # 이외에도 c, d의 관계에 따른 조건 설정
#     return [eq1, eq2, eq3, eq4]
# initial_guess = [0.5, 1, 3, 1]  # 초기 추정값
# solution = fsolve(equations, initial_guess) # 방정식 풀이
# a, b, c, d = solution
# print(f"a = {a}, b = {b}, c = {c}, d = {d}")

# def f0(x):
#     if x >=8:
#         return c*(x-d)**3
#     else:
#         return a*x**2+b

# def f1(x):
#     return a*x**2+b

# def f2(x):
#     return c*(x-d)**3

# x = np.linspace(5,10, 100)

# plt.plot(x, [f0(xi) for xi in x], alpha=0.5)
# plt.plot(x, f1(x), alpha=0.5)
# plt.plot(x, f2(x), alpha=0.5)


# def func(x):
#     if x >=13:
#         return 100 - (c*(x-d-5)**3 + 30)
#     else:
#         return 100 - (a*(x-5)**2+b + 30)

# x = np.linspace(-5,17, 100)
# plt.plot(x, [func(xi) for xi in x], 'o-')
# plt.axvline(0, color='red', alpha=0.3)
# plt.axvline(13, color='red', alpha=0.3)


import numpy as np
from scipy.optimize import fsolve

class RewardFunction():
    def __init__(self):
        self.a = None
        self.b = None 
        self.c = None
        self.d = None
        
    def equations(self, x):
        a, b, c, d = x
        # a * x**2 + b
        # c*(x-d)**3
        eq1 = a * 8**2 + b - c * (8 - d)**3     # f(8) = g(8)
        eq2 = 2* a * 8 - 3 * c * (8 - d)**2     # f'(8) = g'(8)
        eq3 = a + 0.8e-1       # 선택된 추가 방정식들 (예: d의 정의를 활용)
        eq4 = c + 1.3e-1     # 이외에도 c, d의 관계에 따른 조건 설정
        return [eq1, eq2, eq3, eq4]

    def generate_formula(self):
        initial_guess = [0.5, 1, 3, 1]  # 초기 추정값
        self.a, self.b, self.c, self.d = fsolve(self.equations, initial_guess)  # 방정식 풀이
    
    def run(self, x):
        if self.a is None:
            self.generate_formula()
        
        x_arr = np.array(x)

        y_1 = ( (self.a * (x_arr - 5)**2 + self.b + 30) ) * (x_arr <13)
        y_2 = ( (self.c * (x_arr - self.d - 5)**3 + 30) ) * (x_arr >=13)
        y = y_1 + y_2
        return np.ones_like(x_arr)* (-100) * (y < -100) + y * (y >= -100)

    def __call__(self, x):
        return self.run(x)


class RewardFunction():
    def __init__(self):
        self.sigma1 = 25 
        self.mu1 = -13
        self.mu2 = 0
        self.sigma2 = 7
    
    def true_f(self, x):
        gaussian1 = np.exp(-1/(self.sigma1**2)*(x-self.mu1)**2)
        gaussian2 = np.exp(-1/(self.sigma2**2)*(x-self.mu2)**2)
        return gaussian1 * (x < self.mu1) + 1* (x>=self.mu1)*(x<self.mu2) + gaussian2 * (x >= self.mu2)

    def forward(self, x):
        return self.true_f(x)

    def __call__(self, x):
        return self.forward(x)



# 1/(np.sqrt(2*np.pi)*self.sigma1)*
# 1/(np.sqrt(2*np.pi)*self.sigma2)*


if visualize:
    # reward_f = RewardFunction()
    reward_f = RewardFunction()

    x = np.linspace(-60,60, 1000)
    plt.plot(x, reward_f(x), 'o-')
    # plt.plot(x, [reward_f(xi) for xi in x], 'o-')
    plt.axvline(-37, color='red', alpha=0.3)
    plt.axvline(0, color='red', alpha=0.3)
    plt.axvline(13, color='red', alpha=0.3)
    plt.axvline(23, color='red', alpha=0.3)





############################################################################################################
# Simulation with LinUCB ###################################################################################
# (Initial Setting & Parameter Setting) -----------------------------------------------------------------------------------------------

random_state = 1
rng = np.random.RandomState(random_state)      # (RandomState) ★ Set_Params
# n_nodes = 50                        # (Node수) ★ Set_Params

total_period = 7*24*60              # (전체 Horizon) ★ Set_Params
# T = 24*60                           # (주기) ★ Set_Params

# (create base graph) 
node_map = GenerateNodeMap(n_nodes, random_state=random_state)
node_map.create_node(node_scale=50, cov_knn=3)           # create node
node_map.create_connect(connect_scale=0)             # create connection

# (periodic setting)
periodic_f = periodic_improve    # periodic function

# (random noise)
# random_noise_f = RandomNoise(scale=1/30, random_state=random_state)       # (Version 2.1)
random_noise_f = RandomNoise(scale=1/1000, random_state=random_state)      # (Version 2.2)
# random_noise_f = RandomNoise(scale=1/300)                                 # (Version 2.2)

# (aggregated route_graph instance)
temporal_graph = TemporalGraph(node_map=node_map, amp_class=TemporalGaussianAmp,
                                periodic_f=periodic_f, error_f=random_noise_f, random_state=random_state)
# (temporal gaussian amplitude)
temporal_graph.create_amp_instance(T_arr = np.arange(24*60), mean_center=[0.5,0.5], mean_r_f=mean_r_f, cov_f=cov_scale_f,
                        centers=node_map.centers, base_adj_mat=node_map.adj_matrix,
                        normalize=1.5, adjust=True, repeat=9)
temporal_graph.make_temporal_observations()

# # Node_map 재구성
# temporal_graph.node_map.create_node(node_scale=50, cov_knn=3)
# temporal_graph.node_map.create_connect(connect_scale=0) 
# temporal_graph.update_graph()

visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)



# (temporal periodic plot)
x = np.arange(0, 8*24*60)
idx = [1, 49]
# idx = [33, 38]

y_pred = temporal_graph.transform(x)[:, idx[0], idx[1]]
y_true = temporal_graph.transform_oracle(x)[:, idx[0], idx[1]]
visualize_periodic(y_pred, y_true, return_plot=False)


# # (temporal periodic plot)
# x = np.arange(0, 8*24*60)
# idx = [1, 49]
# # idx = [33, 38]

# y_pred = temporal_graph.transform(x)[:, idx[0], idx[1]]
# y_true = temporal_graph.transform_oracle(x)[:, idx[0], idx[1]]
# visualize_periodic(y_pred, y_true, return_plot=False)



# create_dataset -------------------------------------------------------------------------------------
# simul = TemporalMapSimulation(temporal_graph)

# # safe_margin = 8

# contexts_history = []
# for _ in tqdm(range(100000)):
#     # simul.reset_state(reset_all=False, save_interval=time_interval, verbose=0)
#     simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)
#     while(simul.start_time < 300):
#         simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)
#     true_api_info = simul.api_call(save_instance=False, save_history=False)

#     true_path_time = true_api_info['path_time']  #
#     new_target_time = int(np.round(simul.cur_time + true_path_time + safe_margin))  #
#     true_req_leaving_time = new_target_time - safe_margin - true_path_time  #
#     true_api_info['req_leaving_time'] = true_req_leaving_time
#     time_machine_info = simul.run_time_machine(target_time=new_target_time, save_instance=False)
#     time_machine_path_time = time_machine_info['path_time']     #
#     time_machine_req_leaving_time = new_target_time - safe_margin - time_machine_path_time  #

#     new_start_time, new_api_call_time = sorted(np.random.randint(np.max([0, new_target_time - 24*60]), int(np.floor(true_req_leaving_time)), size=2))
#     sample_adjcent_matrix = simul.temporal_graph.transform(new_api_call_time)
#     sample_api_info = simul.api_call(adjacent_matrix= sample_adjcent_matrix, save_instance=False, save_history=False)
#     sample_api_info['path_time']

#     context = simul.save_data('check', save_data=False, save_plot=False)
#     context['start_time'] = format_time_to_str(new_start_time)
#     context['target_time'] = format_time_to_str(new_target_time)
#     context['call_time_TimeMachine'] = format_time_to_str(int(time_machine_req_leaving_time))
#     context['path_TimeMachine'] = time_machine_info['path']
#     context['path_time_TimeMachine'] = time_machine_info['path_time']
#     context['cur_time'] = context['start_time']
#     context['call_time_LastAPI'] = format_time_to_str(new_api_call_time)
#     context['call_node_LastAPI'] = context['start_node']
#     context['call_point_LastAPI'] = context['start_point']
#     context['path_LastAPI'] = sample_api_info['path']
#     context['path_time_LastAPI'] = sample_api_info['path_time']
#     context['req_leaving_time'] = format_time_to_str(time_machine_req_leaving_time)
#     contexts_history.append( (context, true_api_info) )




# # save_data
# from six.moves import cPickle
# # path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\240820_UCB3_3_history.pkl'
# path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\dataset'
# # cPickle.dump(contexts_history, open(f"{path}/240829_context_dataset_rs_1.pkl", 'wb'))

# # for i in range(10):
# #     contexts_ = contexts_history[i*10000:(i+1)*10000]
# #     cPickle.dump(contexts_, open(f"{path}/240829_context_dataset{i:02d}_rs_1.pkl", 'wb'))

# # print('save_dataset')

# # contexts_history = cPickle.load(open(f"{path}/240829_context_dataset_rs_1.pkl", 'rb'))
# contexts_history = cPickle.load(open(f"{path}/240829_context_dataset01_rs_1.pkl", 'rb'))
# len(contexts_history)

# contexts = [xy[0] for xy in contexts_history]
# true_api = [xy[1] for xy in contexts_history]
# # pd.DataFrame(contexts[:1000])
# # pd.DataFrame(true_api[:1000])


# -------------------------------------------------------------------------------------




























































































































###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
# Feature-Set Version 1-1
from IPython.display import clear_output
import time
from tqdm.auto import tqdm

# feature_set_common : cur_time, target_time, cur_point, target_point
feature_set_common = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                        'cur_time_hour_min',
                        # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                        'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                        'target_time_hour_min',
                        # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
                        'cur_point_lat','cur_point_lng', 'target_point_lat', 'target_point_lng']

# feature_set_LastAPI : call_time_LastAPI, call_point_LastAPI, path_time_LastAPI
feature_set_LastAPI = ['call_time_LastAPI_Week', 'call_time_LastAPI_Sat', 'call_time_LastAPI_Sun',
                        'call_time_LastAPI_hour_min',
                        # 'call_time_LastAPI_Commute', 'call_time_LastAPI_Day', 'call_time_LastAPI_Night',
                        'call_point_LastAPI_lat', 'call_point_LastAPI_lng', 'path_time_LastAPI', 'movedist_LastAPI', 'remain_req_leaving_time_LastAPI']

# feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
feature_set_TimeMachine = ['call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        'call_time_TimeMachine_hour_min',
                        # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine', 'movedist_TimeMachine', 'remain_req_leaving_time_TimeMachine']


save_interval = 5

simul = TemporalMapSimulation(temporal_graph)

feature_cols = feature_set_common + feature_set_LastAPI + feature_set_TimeMachine
# feature_cols = feature_set_common + feature_set_LastAPI
# feature_cols = feature_set_common + feature_set_TimeMachine

n_actions = 2
feature_dim = len(feature_cols) + 3


# lucb_simul_1 = LinUCB(n_actions, feature_dim, alpha=alpha, allow_duplicates=False)
lucb_simul_1 = LinUCB(n_actions, feature_dim, allow_duplicates=False)
# reward_f = RewardFunction()
regualize_params = 1
history = []

# -------------------------------------------------------------------------------------
# run episodes and get information
for _ in range(20):
    history_t = {}

    # simul.reset_state(reset_all=False, save_interval=save_interval, verbose=1)
    simul.reset_state(reset_all=True, save_interval=save_interval, verbose=1)
    # start_time = simul.start_time
    # start_node = simul.start_node
    # target_time = simul.target_time
    # target_node = simul.target_node

    # simul.set_start_time(start_time)
    # simul.set_start_loc(start_node)
    # simul.set_target_time(target_time)
    # simul.set_target_loc(target_node)

    # observe contexts and select actions --------------------------------
    t = 0
    # pbar = tqdm()
    while(simul.loop):
        # if simul.t == 0:
        #     simul.api_call()

        context = simul.run()

        if context is not None:
            # print(t ,end =' ')
            # preprocessing of features
            context_df =  pd.Series(context).to_frame().T
            context_feature_df = make_feature_set(context_df)
            feature_df = context_feature_df[feature_cols]
            feature_df['constant'] = 1
            feature_df['square_remain_req_leaving_time_LastAPI'] = feature_df['remain_req_leaving_time_LastAPI']**2
            feature_df['square_remain_req_leaving_time_TimeMachine'] = feature_df['remain_req_leaving_time_TimeMachine']**2
            feature_arr = np.array(feature_df).squeeze()

            # select action
            action = lucb_simul_1.select_action(feature_arr)

            if action == 1:
                simul.api_call()

                # # visualize
                # history_t = np.where(simul.save_t == True)[0]
                # # history_a = lucb_simul_1.batched_actions_for_rewards_match
                # history_a = lucb_simul_1.history['actions'][-len(history_t):]
                # plt.figure(figsize=(20,3))
                # plt.plot(history_t, history_a, 'o-', alpha=0.5)
                # for ti in history_t[history_a.astype(bool)]:
                #     plt.text(ti, 1, ti, rotation=45)
                # plt.show()
                # clear_output(wait=True)
                # time.sleep(0.05)
        t += 1
        # pbar.update(1)
    # pbar.close()
    # feature_df.to_clipboard()





    # observe rewards  -------------------------------------------------------------
    # lucb_simul_1.batched_actions_for_rewards_match
    residual = simul.target_time -8 - (simul.true_leaving_info['call_time'] + simul.true_leaving_info['path_time'])    # residual

    api_call = simul.history_api_call[simul.save_t]
    sum_of_api_call = api_call.sum() 
    api_call_ratio = sum_of_api_call / simul.save_t.sum()

    total_reward = 100 - residual**2
    api_call_reward = total_reward - regualize_params * sum_of_api_call
    # api_call_reward = total_reward - regualize_params * api_call_ratio

    rewards_obs = np.ones(lucb_simul_1.batched_actions_for_rewards_match.shape) * total_reward
    rewards_obs[simul.history_api_call[simul.save_t]] = api_call_reward

    summary_of_episodes = f"api_call: {sum_of_api_call} ({sum_of_api_call/simul.save_t.sum():.3f}), residual: {residual:.1f}, \
total_reward: {total_reward:.1f}, api_call_reward: {api_call_reward:.1f}"
    print(f"  → {summary_of_episodes}")

    lucb_simul_1.observe_reward(rewards_obs)

    # update params -------------------------------------------------------------
    lucb_simul_1.update_params()
    # lucb_simul_1.undo()
    # lucb_simul_1.theta
    # lucb_simul_1.batched_actions_for_rewards_match
    # lucb_simul_1.history['actions'].shape
    # lucb_simul_1.history['rewards'].shape
    
    # save_history -------------------------------------------------------------
    history_t["start_time"] = simul.start_time
    history_t["start_node"] = simul.start_node
    history_t["target_time"] = simul.target_time
    history_t["target_node"] = simul.target_node
    history_t["end_of_round"] = simul.t
    history_t["sum_of_api_call"] = sum_of_api_call
    history_t["api_call"] = api_call
    history_t["residual"] = residual
    history_t["total_reward"] = total_reward
    history_t["api_call_reward"] = api_call_reward
    history_t["summary_of_episodes"] = summary_of_episodes
    history_t['theta_hat'] = lucb_simul_1.theta
    history_t['A'] = lucb_simul_1.A
    history_t['b'] = lucb_simul_1.b
    history_t['alpha'] = lucb_simul_1.alpha
    history.append(history_t)

    print()
# -----------------------------------------------------------------------------

# from six.moves import cPickle
# path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\240816_UCB2_history.pkl'
# cPickle.dump(history, open(path, 'wb'))
# cPickle.load(open(path, 'rb'))


# -----------------------------------------------------------------------------
# pd.DataFrame(simul.history_time_machine).to_clipboard()
# pd.DataFrame(simul.history).to_clipboard()
# pd.DataFrame(simul.history_full_info).to_clipboard()


# -----------------------------------------------------------------------------

# pd.DataFrame(history).to_clipboard()

df = pd.DataFrame(history)
df1 = df.drop(['summary_of_episodes', 'theta_hat' ,'residual', 'total_reward','api_call_reward'], axis=1)
# df1 = df1.drop(['api_call'], axis=1)
summary_df = pd.DataFrame(history)['summary_of_episodes'].apply(lambda x: pd.Series([i.split(": ")[1] for i in x.split(", ")], index=[i.split(": ")[0] for i in x.split(", ")]) )
df2 = pd.concat([df1, summary_df], axis=1)
df2.to_clipboard()

# -----------------------------------------------------------------------------

# make_feature_set(pd.DataFrame(simul.history))[feature_cols].to_clipboard()














###########################################################################################################
# Feature-Set Version 1-2
feature_set_common = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                        'cur_time_hour_min',
                        # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                        # 'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                        # 'target_time_hour_min',
                        # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
                        # 'cur_point_lat','cur_point_lng',
                        'target_point_lat', 'target_point_lng']

# feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
feature_set_TimeMachine = ['call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        'call_time_TimeMachine_hour_min',
                        # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine',
                        # 'movedist_TimeMachine',
                        'remain_req_leaving_time_TimeMachine']


from IPython.display import clear_output
import time
from tqdm.auto import tqdm

save_interval = 5
simul = TemporalMapSimulation(temporal_graph)
alpha = 1
# feature_cols = feature_set_common + feature_set_LastAPI + feature_set_TimeMachine
# feature_cols = feature_set_common + feature_set_LastAPI
feature_cols = feature_set_common + feature_set_TimeMachine

n_actions = 2
feature_dim = len(feature_cols) + 2
time_interval = 3

lucb_simul_2 = LinUCB(n_actions, feature_dim, alpha=alpha, shared_theta=True, shared_context=True, allow_duplicates=False)
history = []





# -----------------------------------------------------------------------------------------------------------
for _ in range(200):
    history_t = {}
    # simul.reset_state(reset_all=False, save_interval=time_interval, verbose=1)
    simul.reset_state(reset_all=True, save_interval=time_interval, verbose=1)

    # prediction
    context = simul.run()

    repeat = (simul.target_time - 8 - simul.cur_time) // time_interval
    context_df =  pd.Series(context).to_frame().T
    cotexts_repeat_df = pd.concat([context_df]*repeat, ignore_index=True)
    cotexts_repeat_df['cur_time'] = time_interval
    cotexts_repeat_df['cur_time'][0] = simul.cur_time
    cotexts_repeat_df['cur_time'] = (cotexts_repeat_df['cur_time'].cumsum()).apply(format_time_to_str)

    context_feature_df = make_feature_set(cotexts_repeat_df)
    feature_df = context_feature_df[feature_cols]
    feature_df['constant'] = 1
    feature_df['square_remain_req_leaving_time_TimeMachine'] = feature_df['remain_req_leaving_time_TimeMachine']**2
    feature_arr = np.array(feature_df).squeeze()

    # select action
    action = lucb_simul_2.select_action(feature_arr)        # ucb : select_action

    for _ in range(simul.target_time - simul.start_time):
        if simul.t == action:
            api_call_dict = simul.api_call()
            simul.save_data('check', save_data=False, save_plot=False)
        simul.run()

    # calcul rewards
    residual = simul.target_time -8 - (simul.true_leaving_info['call_time'] + simul.true_leaving_info['path_time'])     # residual
    reward = -residual**2
    lucb_simul_2.observe_reward(reward)     # ucb : observe_reward

    # update params
    lucb_simul_2.update_params()            # ucb : update_params

    # summary
    summary_of_episodes = f"(residual: {residual:.1f}) Start: {simul.start_time}, Call: {simul.start_time + action}, TimeMachine {simul.call_time_TimeMachine}, Target: {simul.target_time}"
    print(f"  → {summary_of_episodes}")

    # history
    history_t["start_time"] = simul.start_time
    history_t["start_node"] = simul.start_node
    history_t["target_time"] = simul.target_time
    history_t["target_node"] = simul.target_node
    history_t["end_of_round"] = int(np.ceil(simul.req_leaving_time - simul.start_time))
    history_t['action'] = action
    history_t['call_time'] = simul.start_time + action
    history_t['call_time_TimaMachine'] = simul.call_time_TimeMachine
    history_t["residual_call_time"] = history_t['call_time_TimaMachine'] - history_t['call_time']
    history_t["residual"] = residual
    history_t["reward"] = reward
    history_t["summary_of_episodes"] = summary_of_episodes
    history_t['theta'] = lucb_simul_2.theta
    history_t['A'] = lucb_simul_2.A
    history_t['b'] = lucb_simul_2.b
    history_t['alpha'] = lucb_simul_2.alpha
    history.append(history_t)

    print()

# save_data
from six.moves import cPickle
path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\240820_UCB3_2_history.pkl'
cPickle.dump(history, open(path, 'wb'))
print('save_pkl')

# load data
# history_load = cPickle.load(open(path, 'rb'))
# history = history_load.copy()
# lucb_simul_2.t = len(history_load)
# lucb_simul_2.A = history_load[-1]['A']
# lucb_simul_2.b = history_load[-1]['b']
# lucb_simul_2.alpha = history_load[-1]['alpha']
# lucb_simul_2.theta = history_load[-1]['theta']

# copy to clipboard
pd.DataFrame(history).to_clipboard()
pd.DataFrame(simul.history_time_machine).to_clipboard()
pd.DataFrame(simul.history).to_clipboard()
pd.DataFrame(simul.history_full_info).to_clipboard()



# visualize
plt.figure(figsize=(20,3))
plt.plot(np.arange(len(history)),pd.DataFrame(history)['residual'], 'o-')
plt.show()







###########################################################################################################
import torch
import torch.nn as nn
import torch.optim as optim

# Feature-Set Version 1-3
class RewardFunction():
    def __init__(self):
        self.sigma1 = 25 
        self.mu1 = -13
        self.mu2 = 0
        self.sigma2 = 7
    
    def true_f(self, x):
        gaussian1 = np.exp(-1/(self.sigma1**2)*(x-self.mu1)**2)
        gaussian2 = np.exp(-1/(self.sigma2**2)*(x-self.mu2)**2)
        return gaussian1 * (x < self.mu1) + 1* (x>=self.mu1)*(x<self.mu2) + gaussian2 * (x >= self.mu2)

    def forward(self, x):
        return self.true_f(x)

    def __call__(self, x):
        return self.forward(x)

x = np.linspace(-100,100, 100)
reward_f = RewardFunction()
plt.title("reward function")
plt.plot(x,reward_f(x), 'o-')
plt.axvline(0, color='red', alpha=0.3)
plt.axvline(-13, color='red', alpha=0.3)
plt.show()


class RegularizeFunction():
    def __init__(self):
        self.slope = 0.5

    def true_f(self, x):
        return (1/(1+np.exp(-self.slope * x))-0.5)*2

    def forward(self, x):
        return self.true_f(x)

    def __call__(self, x):
        return self.forward(x)

x = np.linspace(0, 100, 100)
reqularize_f = RegularizeFunction()
plt.title("reqularize function")
plt.plot(x,reqularize_f(x), 'o-')
plt.axvline(0, color='red', alpha=0.3)
plt.axvline(10, color='red', alpha=0.3)
plt.show()



# Feature-Set Version 1-3
from IPython.display import clear_output
import time
from tqdm.auto import tqdm

# feature_set_common : cur_time, target_time, cur_point, target_point
feature_set_common = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                        'cur_time_hour_min',
                        # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                        'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                        'target_time_hour_min',
                        # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
                        'cur_point_lat','cur_point_lng', 'target_point_lat', 'target_point_lng']

# feature_set_LastAPI : call_time_LastAPI, call_point_LastAPI, path_time_LastAPI
feature_set_LastAPI = ['call_time_LastAPI_Week', 'call_time_LastAPI_Sat', 'call_time_LastAPI_Sun',
                        'call_time_LastAPI_hour_min',
                        # 'call_time_LastAPI_Commute', 'call_time_LastAPI_Day', 'call_time_LastAPI_Night',
                        'call_point_LastAPI_lat', 'call_point_LastAPI_lng', 'path_time_LastAPI', 'movedist_LastAPI', 'remain_req_leaving_time_LastAPI']

# feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
feature_set_TimeMachine = ['call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        'call_time_TimeMachine_hour_min',
                        # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine', 'movedist_TimeMachine', 'remain_req_leaving_time_TimeMachine']

save_interval = 5
simul = TemporalMapSimulation(temporal_graph)
feature_cols = feature_set_common + feature_set_LastAPI + feature_set_TimeMachine


from IPython.display import clear_output
import time
from tqdm.auto import tqdm
from six.moves import cPickle

simul = TemporalMapSimulation(temporal_graph)
feature_cols = feature_set_common + feature_set_TimeMachine

n_actions = 2; feature_dim = len(feature_cols); time_interval = 3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
alpha = 1
input_dim = feature_dim; hidden_dim = 32; output_dim = 1
n_models = 10; n_samples = 30; n_layers = 5
learning_rate = 1e-4


model = DirectEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=n_models, n_layers=n_layers)
# model = SampleEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_samples=n_samples, n_layers=n_layers)
model.to(device)
history = []
# load_path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject'
# history = cPickle.load(open(f"{load_path}/240827_Alg1_3_DirectEnsemble_history.pkl", 'rb'))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

regualize_params = 1
reward_function = RewardFunction()
regularize_function = RegularizeFunction()
loss_gaussian = nn.GaussianNLLLoss()


num_episodes = 100
for episode in range(num_episodes):
    history_t = {}

    # simul.reset_state(reset_all=False, save_interval=save_interval, verbose=1)
    simul.reset_state(reset_all=True, save_interval=save_interval, verbose=1)

    # observe contexts and select actions --------------------------------
    t = 0
    contexts = []
    actions = []

    # pbar = tqdm()
    while(simul.loop):
        context = simul.run()
        if context is not None:
            # preprocessing of features
            context_df =  pd.Series(context).to_frame().T
            context_feature_df = make_feature_set(context_df)
            feature_df = context_feature_df[feature_cols]
            feature_arr = np.array(feature_df)
            feature_tensor = torch.tensor(feature_arr.astype(np.float32)).to(device)

            # select action
            action = None
            with torch.no_grad():
                mu, logvar = model(feature_tensor)
                ucb = mu + alpha * torch.exp(0.5*logvar)
                ucb_mean = torch.mean(ucb, dim=1, keepdims=True)
                action = int(torch.sigmoid(ucb_mean).item() >= 0.5)
                contexts.append( feature_arr.squeeze() )
                actions.append( action )

            if action == 1:
                simul.api_call()

                # # visualize
                # history_t = np.where(simul.save_t == True)[0]
                # # history_a = lucb_simul_1.batched_actions_for_rewards_match
                # history_a = lucb_simul_1.history['actions'][-len(history_t):]
                # plt.figure(figsize=(20,3))
                # plt.plot(history_t, history_a, 'o-', alpha=0.5)
                # for ti in history_t[history_a.astype(bool)]:
                #     plt.text(ti, 1, ti, rotation=45)
                # plt.show()
                # clear_output(wait=True)
                # time.sleep(0.05)
        t += 1

    # lucb_simul_1.batched_actions_for_rewards_match
    reward_x = (simul.true_leaving_info['call_time'] + simul.true_leaving_info['path_time']) - simul.target_time      # residual
    reward = reward_function(reward_x)
    sum_of_api_call = np.sum(actions)
    reward_regualize = regularize_function(sum_of_api_call)
    api_call_reward = reward - reward_regualize

    action_TF = np.array(actions).astype(bool)
    rewards_obs = np.ones(len(contexts)) * reward
    rewards_obs[action_TF] = api_call_reward


    summary_of_episodes = f"api_call: {sum_of_api_call} ({sum_of_api_call/len(actions):.3f}),\
 reward_x: {reward_x:.1f}, reward: {reward:.1f}, api_call_reward: {api_call_reward:.1f}"
    print(f"  → {summary_of_episodes}")

    history_t['contexts'] = contexts
    history_t['actions'] = actions
    history_t['reward_x'] = reward_x
    history_t['reward'] = reward
    history_t['api_call_reward'] = api_call_reward
    history_t['summary_of_episodes'] = summary_of_episodes
    history.append(history_t)
    
    # (Training Model)
    x_train = torch.tensor(np.stack(contexts).astype(np.float32)).to(device)
    y_train = torch.tensor(rewards_obs.reshape(-1,1)).to(device)

    model.train()
    optimizer.zero_grad()
    
    mu, logvar = model(x_train)
    std = torch.exp(0.5*logvar)
    logit = torch.log(y_train / (1 - y_train + 1e-10))
    loss = loss_gaussian0(mu, logit, std**2)
    loss.backward()
    optimizer.step()
    # save weights
    path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\240827_Alg1_3_DirectEnsemble_weights.pkl'
    cPickle.dump(model_action0.state_dict(), open(path, 'wb'))
    print()
print('save_weights', end=" ")
    
# save history
path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\240827_Alg1_3_DirectEnsemble_history.pkl'
cPickle.dump(history, open(path, 'wb'))
print('save_history')

 
dist_y = [np.sum(hist['actions']) for hist in history]
dist_y = [hist['reward_x'] for hist in history]
dist_y = [hist['reward'] for hist in history]
dist_y = [hist['api_call_reward'] for hist in history]

# visualize
plt.figure(figsize=(20,3))
plt.plot(np.arange(len(history)), dist_y, 'o-')
plt.axhline(0, c='black', alpha=0.3)
plt.axhline(-13, c='red', alpha=0.3)
plt.show()








###########################################################################################################
# Feature-Set Version 1-4
import torch
import torch.nn as nn
import torch.optim as optim


# ★Mu/Var Ensemble only last layer
class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(),
                batchNorm=True,  dropout=0.5):
        super().__init__()
        ff_block = [nn.Linear(input_dim, output_dim)]
        if activation:
            ff_block.append(activation)
        if batchNorm:
            ff_block.append(nn.BatchNorm1d(output_dim))
        if dropout > 0:
            ff_block.append(nn.Dropout(dropout))
        self.ff_block = nn.Sequential(*ff_block)
    
    def forward(self, x):
        return self.ff_block(x)

# ★Mu/Var Ensemble only last layer
class DirectEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, n_models=10):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim*2*n_models, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.n_models = n_models
        self.output_dim = output_dim

    # train step
    def train_forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        mu_logvar = (x)
        mu, logvar = torch.split(mu_logvar, self.n_models, dim=1)
        logvar = torch.clamp(logvar, min=-10, max=20) 
        return mu, logvar

    # eval step : 여러 번 샘플링하여 평균과 분산 계산
    def predict(self, x, idx=None):
        mu, logvar = self.train_forward(x)
        if idx is None:
            mu_mean = torch.mean(mu, dim=1, keepdims=True)
            logvar_mean = torch.mean(logvar, dim=1, keepdims=True)
        else:
            mu_mean = torch.mean(mu[:, idx], dim=1, keepdims=True)
            logvar_mean = torch.mean(logvar[:, idx], dim=1, keepdims=True)
        return  mu_mean, logvar_mean

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x)

# point estimate
class DirectEstimate(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.output_dim = output_dim

    def forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection
        logmu = x
        logmu = torch.clamp(logmu, min=-10, max=20) 
        return logmu
# ----------------------------------------------------------------------------


class RewardFunction():
    def __init__(self):
        self.sigma1 = 25 
        self.mu1 = -13
        self.mu2 = 0
        self.sigma2 = 7
    
    def true_f(self, x):
        gaussian1 = np.exp(-1/(self.sigma1**2)*(x-self.mu1)**2)
        gaussian2 = np.exp(-1/(self.sigma2**2)*(x-self.mu2)**2)
        return gaussian1 * (x < self.mu1) + 1* (x>=self.mu1)*(x<self.mu2) + gaussian2 * (x >= self.mu2)

    def forward(self, x):
        return self.true_f(x)

    def __call__(self, x):
        return self.forward(x)

x = np.linspace(-100,100, 100)
reward_f = RewardFunction()
plt.title("reward function")
plt.plot(x,reward_f(x), 'o-')
plt.axvline(0, color='red', alpha=0.3)
plt.axvline(-13, color='red', alpha=0.3)
plt.show()


class RegularizeFunction():
    def __init__(self):
        self.slope = 0.5

    def true_f(self, x):
        return 0 *(x <1) + ((1/(1+np.exp(-self.slope * (x-1)))-0.5)*2) *(x>=1)

    def forward(self, x):
        return self.true_f(x)

    def __call__(self, x):
        return self.forward(x)

x = np.linspace(0, 100, 100)
reqularize_f = RegularizeFunction()
plt.title("reqularize function")
plt.plot(x,reqularize_f(x), 'o-')
plt.axvline(0, color='red', alpha=0.3)
plt.axvline(10, color='red', alpha=0.3)
plt.show()



# Feature-Set Version 1-4

# feature_set_common : cur_time, target_time, cur_point, target_point
feature_set_common = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                        'cur_time_hour_min',
                        # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                        'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                        'target_time_hour_min',
                        # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
                        'cur_point_lat','cur_point_lng', 'target_point_lat', 'target_point_lng']

# feature_set_LastAPI : call_time_LastAPI, call_point_LastAPI, path_time_LastAPI
feature_set_LastAPI = ['call_time_LastAPI_Week', 'call_time_LastAPI_Sat', 'call_time_LastAPI_Sun',
                        'call_time_LastAPI_hour_min',
                        # 'call_time_LastAPI_Commute', 'call_time_LastAPI_Day', 'call_time_LastAPI_Night',
                        'call_point_LastAPI_lat', 'call_point_LastAPI_lng', 'path_time_LastAPI'
                        # , 'movedist_LastAPI', 'remain_req_leaving_time_LastAPI'
                        ]

# feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
feature_set_TimeMachine = ['call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        'call_time_TimeMachine_hour_min',
                        # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine',
                        #  'movedist_TimeMachine', 'remain_req_leaving_time_TimeMachine'
                         ]


feature_pathtime = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                    'cur_time_hour_min',
                    'target_point_lat', 'target_point_lng',
                    'call_time_LastAPI_Week', 'call_time_LastAPI_Sat', 'call_time_LastAPI_Sun',
                    'call_time_LastAPI_hour_min',
                    # 'call_time_LastAPI_Commute', 'call_time_LastAPI_Day', 'call_time_LastAPI_Night',
                    'call_point_LastAPI_lat', 'call_point_LastAPI_lng', 'path_time_LastAPI',
                    'call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                    'call_time_TimeMachine_hour_min',
                    'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine'
                    ]

from IPython.display import clear_output
import time
from tqdm.auto import tqdm
from six.moves import cPickle

simul = TemporalMapSimulation(temporal_graph)

def context_to_tensor(context, feature_cols=None):
    if type(context) == dict:
        context_df =  pd.Series(context).to_frame().T
    elif type(context) == list or type(context) == np.ndarray:
        context_df =  pd.DataFrame(context)
    context_feature_df = make_feature_set(context_df)
    feature_df = context_feature_df[feature_cols] if feature_cols is not None else context_feature_df
    feature_arr = np.array(feature_df)
    feature_tensor = torch.tensor(feature_arr.astype(np.float32))
    return feature_tensor





simul = TemporalMapSimulation(temporal_graph)
feature_cols = feature_set_common + feature_set_LastAPI + feature_set_TimeMachine

n_actions = 2; feature_dim = len(feature_cols); time_interval = 3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
alpha = 1
input_dim = feature_dim; input_path_dim = len(feature_pathtime)
hidden_dim = 32; output_dim = 1
n_models = 10; n_samples = 30; n_layers = 5
learning_rate = 1e-4

model_action0 = DirectEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=n_models, n_layers=n_layers)
model_action1 = DirectEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=n_models, n_layers=n_layers)
model_pathtime = DirectEstimate(input_dim=input_path_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers)

model_action0.to(device)
model_action1.to(device)
model_pathtime.to(device)


history = []
# load_path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject'
# history = cPickle.load(open(f"{load_path}/240827_Alg1_4_DirectEnsemble_history.pkl", 'rb'))

optimizer_action0 = optim.Adam(model_action0.parameters(), lr=learning_rate)
optimizer_action1 = optim.Adam(model_action1.parameters(), lr=learning_rate)
optimizer_pathtime = optim.Adam(model_pathtime.parameters(), lr=1e-5)

reward_function = RewardFunction()
regularize_function = RegularizeFunction()
loss_gaussian = nn.GaussianNLLLoss()
loss_mse = nn.MSELoss()



num_episodes = 100
for episode in tqdm(range(num_episodes)):
    history_t = {}

    # simul.reset_state(reset_all=False, save_interval=save_interval, verbose=1)
    simul.reset_state(reset_all=True, save_interval=save_interval, verbose=0)

    # observe contexts and select actions --------------------------------
    contexts = []
    actions = []
    pred_path_times = []



    # START OF EPISODES ------------------------------------------------------
    for t in range(simul.target_time - simul.start_time):
        
        pred_path_time = 0
        context = simul.run()
        if context is not None:
            # preprocessing of features
            feature_tensor = context_to_tensor(context, feature_cols).to(device)
            
            # select action
            action = None
            with torch.no_grad():
                mu0, logvar0 = model_action0.predict(feature_tensor)
                ucb0 = mu0 + alpha * torch.exp(0.5*logvar0)

                mu1, logvar1 = model_action1.predict(feature_tensor)
                ucb1 = mu1 + alpha * torch.exp(0.5*logvar1)

                action = 0 if ucb0.item() >= ucb1.item() else 1
            context
            contexts.append( context )
            actions.append( action )
            
            if action == 1:
                simul.api_call()

                # # visualize
                # history_t = np.where(simul.save_t == True)[0]
                # # history_a = lucb_simul_1.batched_actions_for_rewards_match
                # history_a = lucb_simul_1.history['actions'][-len(history_t):]
                # plt.figure(figsize=(20,3))
                # plt.plot(history_t, history_a, 'o-', alpha=0.5)
                # for ti in history_t[history_a.astype(bool)]:
                #     plt.text(ti, 1, ti, rotation=45)
                # plt.show()
                # clear_output(wait=True)
                # time.sleep(0.05)
            
            # predict path time
            context = simul.save_data('check', save_data=False, save_plot=False)
            feature_tensor = context_to_tensor(context, feature_pathtime).to(device)

            with torch.no_grad():
                # path_logmu, path_logvar = model_pathtime.predict(feature_tensor)
                # path_mu = torch.exp(path_logmu)
                # pred_path_time = path_mu.to('cpu').detach().numpy().item()    # normalize
                path_logmu = model_pathtime(feature_tensor)
                path_mu = torch.exp(path_logmu)
                pred_path_time = path_mu.to('cpu').detach().numpy().item()    # normalize
                pred_path_times.append(pred_path_time)
        
        if simul.cur_time + pred_path_time > simul.target_time -8:
            adjcent_matrix = simul.temporal_graph.transform(simul.cur_time)
            true_leaving_info = simul.api_call(adjcent_matrix, save_instance=False, save_history=False)
            break
    # context_to_tensor(context, feature_pathtime).to(device)
    # END OF EPISODES ------------------------------------------------------

    # accrate path time reward
    reward_x = (true_leaving_info['call_time'] + true_leaving_info['path_time']) - simul.target_time
    reward = reward_function(reward_x)
    
    # regualize_penalty
    rev_action = np.array(actions)[::-1]
    mask = (rev_action == 1)
    cumsum_action = np.zeros_like(rev_action)
    cumsum_action[mask] = np.cumsum(rev_action[mask])
    rev_cumsum_action = cumsum_action[::-1]
    reward_regualize = regularize_function(rev_cumsum_action)

    # total_reward
    total_reward = reward - reward_regualize
    
    # # from_last_api_call data
    # last_action_idx = np.argwhere(np.array(actions)==1)[-1].item()
    
    # action model update ----------------------------------------------------------------------
    action0_TF = (np.array(actions)==0)
    action1_TF = (np.array(actions)==1)

    if action0_TF.sum() > 0:
        x_train = context_to_tensor(list(np.stack(contexts)[action0_TF]), feature_cols).to(device)
        y_train = torch.tensor(np.stack(total_reward)[action0_TF].reshape(-1,1)).to(device)
        
        model_action0.train()
        optimizer_action0.zero_grad()
        mu, logvar = model_action0(x_train)
        std = torch.exp(0.5*logvar)
        # logit = torch.log(y_train / (1 - y_train + 1e-10))
        atanh = 0.5* torch.log((1+y_train + 1e-10) / (1 - y_train + 1e-10))
        loss = torch.nn.functional.gaussian_nll_loss(mu, atanh, std**2)
        loss.backward()
        optimizer_action0.step()
        
    if action1_TF.sum() > 0:
        x_train = context_to_tensor(list(np.stack(contexts)[action1_TF]), feature_cols).to(device)
        y_train = torch.tensor(np.stack(total_reward)[action1_TF].reshape(-1,1)).to(device)
        
        model_action1.train()
        optimizer_action1.zero_grad()
        mu, logvar = model_action1(x_train)
        std = torch.exp(0.5*logvar)
        # logit = torch.log(y_train / (1 - y_train + 1e-10))
        atanh = 0.5* torch.log((1+y_train + 1e-10) / (1 - y_train + 1e-10))
        loss = torch.nn.functional.gaussian_nll_loss(mu, atanh, std**2)
        loss.backward()
        optimizer_action1.step()

    # path_time model update ----------------------------------------------------------------------
    x_train = context_to_tensor(list(np.stack(contexts)[[-1]]), feature_pathtime).to(device)
    y_train = torch.tensor(true_leaving_info['path_time'], dtype=torch.float32).view(-1,1).to(device)
    
    model_pathtime.train()
    optimizer_pathtime.zero_grad()
    
    # path_logmu, path_logvar = model_pathtime(x_train)
    # path_mu = torch.exp(path_logmu)
    # path_std = torch.exp(0.5*path_logvar)
    # loss = torch.nn.functional.gaussian_nll_loss(path_mu, y_train, path_std**2)
    # loss.backward()

    path_logmu = model_pathtime(x_train)
    path_mu = torch.exp(path_logmu)
    loss = loss_mse(path_mu, y_train)
    loss.backward()
    optimizer_pathtime.step()

    # print result
    sum_of_api_call = np.sum(actions)
    summary_of_episodes = f"api_call: {sum_of_api_call} ({sum_of_api_call/len(actions):.3f}),\
 reward_x: {reward_x:.1f}, reward: {reward:.1f}"
    print(f"\r  → {summary_of_episodes}")

    
    history_t['contexts'] = contexts
    history_t['actions'] = actions
    with torch.no_grad():
        history_t['pred_path_time_mu'] = torch.mean(path_mu).to('cpu').detach().item()
        # history_t['pred_path_time_std'] = torch.mean(path_std).to('cpu').detach().item()
        # print(f"\r{history_t['pred_path_time_mu']}, {history_t['pred_path_time_std']}")
        print(f"\r{history_t['pred_path_time_mu']}")
    history_t['true_leaving_info'] = true_leaving_info
    history_t['reward_x'] = reward_x
    history_t['reward'] = reward
    history_t['reward_regualize'] = reward_regualize
    history_t['summary_of_episodes'] = summary_of_episodes
    history.append(history_t)
    
    # save weights
    # path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject'
    path = "/content/drive/MyDrive/SNU_OhLAB/CLabProject"
    cPickle.dump(model_action0.state_dict(), open(f"{path}/240829_Alg1_4_DirectEnsemble_action0_weights.pkl", 'wb'))
    cPickle.dump(model_action1.state_dict(), open(f"{path}/240829_Alg1_4_DirectEnsemble_action1_weights.pkl", 'wb'))
    cPickle.dump(model_pathtime.state_dict(), open(f"{path}/240829_Alg1_4_DirectEnsemble_pathtime_weights.pkl", 'wb'))
    print()
        
    # save history
    cPickle.dump(history, open(f"{path}/240829_Alg1_4_DirectEnsemble_history.pkl", 'wb'))
# print('save weights & history ')

from six.moves import cPickle
path = 'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\Model'
# path = "/content/drive/MyDrive/SNU_OhLAB/CLabProject/Model"
history = cPickle.load(open(f"{path}/240829_Alg1_4_DirectEnsemble_history.pkl", 'rb'))


dist_y = [np.sum(hist['actions']) for hist in history]
dist_y = [hist['reward_x'] for hist in history]
dist_y = [hist['reward'] for hist in history]
dist_y = [hist['true_leaving_info']['path_time'] - hist['pred_path_time_mu'] for hist in history]


dist_y = [hist['pred_path_time_mu'] for hist in history]
dist_y = [hist['true_leaving_info']['path_time'] for hist in history]

# visualize
plt.figure(figsize=(20,3))
plt.plot(np.arange(len(history)), dist_y, 'o-')
plt.axhline(0, c='black', alpha=0.3)
plt.axhline(-13, c='red', alpha=0.3)
plt.yscale("symlog")
plt.show()







































































###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
# Feature-Set Version 2
feature_set_common = [
                    # 'cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                    # 'cur_time_hour_min',
                        # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                        # 'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                        # 'target_time_hour_min',
                        # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
                        # 'cur_point_lat','cur_point_lng',
                        # 'target_point_lat', 'target_point_lng'
                        ]

# feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
feature_set_TimeMachine = [
                        # 'call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        # 'call_time_TimeMachine_hour_min',
                        # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        # 'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 
                        'path_time_TimeMachine',
                        # 'movedist_TimeMachine',
                        'remain_req_leaving_time_TimeMachine'
                        ]


from IPython.display import clear_output
import time
from tqdm.auto import tqdm

save_interval = 5
simul = TemporalMapSimulation(temporal_graph)
alpha = 1
feature_cols = feature_set_TimeMachine

n_actions = 2
feature_dim = len(feature_cols) + 2
time_interval = 3

lucb_simul_2 = LinUCB(n_actions, feature_dim, alpha=alpha, shared_theta=True, shared_context=True, allow_duplicates=False)
history = []




# -----------------------------------------------------------------------------------------------------------
for _ in range(50):
    history_t = {}
    # simul.reset_state(reset_all=False, save_interval=time_interval, verbose=1)
    simul.reset_state(reset_all=True, save_interval=time_interval, verbose=1)

    # prediction
    context = simul.run()

    repeat = (simul.target_time - 8 - simul.cur_time) // time_interval
    context_df =  pd.Series(context).to_frame().T
    cotexts_repeat_df = pd.concat([context_df]*repeat, ignore_index=True)
    cotexts_repeat_df['cur_time'] = time_interval
    cotexts_repeat_df['cur_time'][0] = simul.cur_time
    cotexts_repeat_df['cur_time'] = (cotexts_repeat_df['cur_time'].cumsum()).apply(format_time_to_str)

    context_feature_df = make_feature_set(cotexts_repeat_df)
    feature_df = context_feature_df[feature_cols]
    feature_df['constant'] = 1
    feature_df['square_remain_req_leaving_time_TimeMachine'] = feature_df['remain_req_leaving_time_TimeMachine']**2
    feature_df['path_time_by_remain_req_time'] = feature_df['path_time_TimeMachine'] * feature_df['remain_req_leaving_time_TimeMachine']
    feature_df = feature_df.drop('path_time_TimeMachine', axis=1)
    feature_arr = np.array(feature_df).squeeze()

    # select action
    action = lucb_simul_2.select_action(feature_arr)        # ucb : select_action

    for _ in range(simul.target_time - simul.start_time):
        if simul.t == action:
            api_call_dict = simul.api_call()
            simul.save_data('check', save_data=False, save_plot=False)
        simul.run()

    
    # calcul rewards
    # residual = simul.target_time -8 - (simul.true_leaving_info['call_time'] + simul.true_leaving_info['path_time'])     # residual
    residual = simul.target_time - 8 - (simul.start_time + api_call_dict['path_time'] )
    reward = - residual**2
    lucb_simul_2.observe_reward(reward)     # ucb : observe_reward

    # update params
    lucb_simul_2.update_params()            # ucb : update_params

    # summary
    summary_of_episodes = f"(residual: {residual:.1f}) Start: {simul.start_time}, Call: {simul.start_time + action}, TimeMachine {simul.call_time_TimeMachine}, Target: {simul.target_time}"
    print(f"  → {summary_of_episodes}")

    # history
    history_t["start_time"] = simul.start_time
    history_t["start_node"] = simul.start_node
    history_t["target_time"] = simul.target_time
    history_t["target_node"] = simul.target_node
    history_t["end_of_round"] = int(np.ceil(simul.req_leaving_time - simul.start_time))
    history_t['action'] = action
    history_t['call_time'] = simul.start_time + action
    history_t['call_time_TimaMachine'] = simul.call_time_TimeMachine
    history_t["residual_call_time"] = history_t['call_time_TimaMachine'] - history_t['call_time']
    history_t["residual"] = residual
    history_t["reward"] = reward
    history_t["summary_of_episodes"] = summary_of_episodes
    history_t['theta'] = lucb_simul_2.theta
    history_t['A'] = lucb_simul_2.A
    history_t['b'] = lucb_simul_2.b
    history_t['alpha'] = lucb_simul_2.alpha
    history.append(history_t)

    print()

# save_data
from six.moves import cPickle
# path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\240820_UCB3_3_history.pkl'
path = r'D:\DataScience\SNU OhLAB\CLab_Project\240820_UCB3_3_history.pkl'
cPickle.dump(history, open(path, 'wb'))
print('save_pkl')
# -----------------------------------------------------------------------------------------------------------


# load data
# history_load = cPickle.load(open(path, 'rb'))
# history = history_load.copy()
# lucb_simul_2.t = len(history_load)
# lucb_simul_2.A = history_load[-1]['A']
# lucb_simul_2.b = history_load[-1]['b']
# lucb_simul_2.alpha = history_load[-1]['alpha']
# lucb_simul_2.theta = history_load[-1]['theta']

# copy to clipboard
pd.DataFrame(history).to_clipboard()
pd.DataFrame(simul.history_time_machine).to_clipboard()
pd.DataFrame(simul.history).to_clipboard()
pd.DataFrame(simul.history_full_info).to_clipboard()


# x[np.argmax(y_hat)]

x = np.linspace(1200,-200, 100)
y_hat = lucb_simul_2.theta[2] * x **2 +  lucb_simul_2.theta[3] +  lucb_simul_2.theta[0]
plt.plot(x,y_hat)
plt.show()

# visualize
plt.figure(figsize=(20,3))
plt.plot(np.arange(len(history)),pd.DataFrame(history)['residual'], 'o-')
plt.show()














###########################################################################################################
# Feature-Set Version 2-2

# import httpimport
# remote_library_url = 'https://raw.githubusercontent.com/kimds929/'


# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_DataFrame import DF_Summary

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_Plot import ttest_each, violin_box_plot, distbox

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_MachineLearning import ScalerEncoder, DataSet

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_DeepLearning import EarlyStopping

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_Torch import TorchDataLoader, TorchModeling, AutoML

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim


class RunningNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(RunningNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)
    
    def forward(self, x):
        if self.training:
            # 현재 입력의 평균과 분산 계산
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Running mean and variance 업데이트
            self.running_mean.data = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean.data
            self.running_var.data = self.momentum * batch_var + (1 - self.momentum) * self.running_var.data
            
            # 현재 입력 데이터 정규화
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # 테스트 모드에서는 running mean과 variance 사용
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        return x_normalized

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(),
                batchNorm=True, runningNorm=False, dropout=0.5):
        super().__init__()
        ff_block = [nn.Linear(input_dim, output_dim)]
        if activation:
            ff_block.append(activation)
        if batchNorm:
            ff_block.append(nn.BatchNorm1d(output_dim))
        if runningNorm:
            ff_block.append(nn.RunningNorm(output_dim))
        if dropout > 0:
            ff_block.append(nn.Dropout(dropout))
        self.ff_block = nn.Sequential(*ff_block)
    
    def forward(self, x):
        return self.ff_block(x)

# ★Mu/Var Ensemble only last layer
class DirectEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, n_models=10):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim*2*n_models, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.n_models = n_models
        self.output_dim = output_dim

    # train step
    def train_forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        mu_logvar = (x)
        mu, logvar = torch.split(mu_logvar, self.n_models, dim=1)
        logvar = torch.clamp(logvar, min=-10, max=20) 
        return mu, logvar

    # eval step : 여러 번 샘플링하여 평균과 분산 계산
    def predict(self, x, idx=None):
        mu, logvar = self.train_forward(x)
        if idx is None:
            mu_mean = torch.mean(mu, dim=1, keepdims=True)
            logvar_mean = torch.mean(logvar, dim=1, keepdims=True)
        else:
            mu_mean = torch.mean(mu[:, idx], dim=1, keepdims=True)
            logvar_mean = torch.mean(logvar[:, idx], dim=1, keepdims=True)
        return  mu_mean, logvar_mean

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x)

# ★ Sample Ensemble only last layers Model
class SampleEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, n_samples=10):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim*n_samples, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.n_samples = n_samples
        self.output_dim = output_dim

    
    def forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        # return ensemble_outputs
        mu = torch.mean(x, dim=1, keepdims=True)
        var = torch.var(x, dim=1, keepdims=True)
        logvar = torch.log(var)
        return mu, logvar

# point estimate
class DirectEstimate(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.n_models = n_models
        self.output_dim = output_dim

    def forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection
        logmu = x
        logmu = torch.clamp(logmu, min=-10, max=20) 
        return logmu

#########################################################################################################
import torch
class RewardFunctionTorch():
    def __init__(self):
        self.sigma1 = 25 
        self.mu1 = -13
        self.mu2 = 0
        self.sigma2 = 7
    
    def true_f(self, x):
        gaussian1 = torch.exp(-1/(self.sigma1**2)*(x-self.mu1)**2)
        gaussian2 = torch.exp(-1/(self.sigma2**2)*(x-self.mu2)**2)
        return gaussian1 * (x < self.mu1) + 1 * (x >= self.mu1) * (x < self.mu2)+  gaussian2 * (x >= self.mu2)

    def forward(self, x):
        return self.true_f(x)
        # return self.true_f(x) + torch.randn_like(x)*0.01

    def __call__(self, x):
        return self.forward(x)

x = torch.linspace(-100,100, 100)
reward_f = RewardFunctionTorch()
plt.plot(x,reward_f(x), 'o-')
# plt.axvline(-37, color='red', alpha=0.3)
plt.axvline(0, color='red', alpha=0.3)
plt.axvline(-13, color='red', alpha=0.3)
# plt.axvline(23, color='red', alpha=0.3)
plt.show()


#########################################################################################################
# Feature-Set Version 2-2

feature_set_common = [
                    'cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                    'cur_time_hour_min',
                        # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                        # 'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                        # 'target_time_hour_min',
                        # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
                        'cur_point_lat','cur_point_lng',
                        'target_point_lat', 'target_point_lng'
                        ]

# feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
feature_set_TimeMachine = [
                        'call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        'call_time_TimeMachine_hour_min',
                        # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        # 'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 
                        'path_time_TimeMachine',
                        # 'movedist_TimeMachine',
                        'remain_req_leaving_time_TimeMachine'
                        ]



from IPython.display import clear_output
import time
from tqdm.auto import tqdm
from six.moves import cPickle

simul = TemporalMapSimulation(temporal_graph)
feature_cols = feature_set_common + feature_set_TimeMachine

n_actions = 2; feature_dim = len(feature_cols); time_interval = 3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
alpha = 1
input_dim = feature_dim; hidden_dim = 32; output_dim = 1
n_models = 10; n_samples = 30; n_layers = 5
learning_rate = 1e-4

model = DirectEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=n_models, n_layers=n_layers)
# model = SampleEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_samples=n_samples, n_layers=n_layers)
# load_path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject'
# load_weights = cPickle.load(open(f"{load_path}/240826_Alg2_2_DirectEnsemble_weights.pkl", 'rb'))
# model.load_state_dict(load_weights)
model.to(device)

history = []
# history = cPickle.load(open(f"{load_path}/240826_Alg2_2_DirectEnsemble_history.pkl", 'rb'))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
reward_function = RewardFunctionTorch()
loss_gaussian = nn.GaussianNLLLoss()

# ----------------------------------------------------------------------------------------------------
num_episodes = 3000
for episode in range(num_episodes):
    history_t = {}

    # simul.reset_state(reset_all=False, save_interval=time_interval, verbose=1)
    simul.reset_state(reset_all=True, save_interval=time_interval, verbose=1)
    # prediction
    context = simul.run()

    repeat = (simul.target_time - 8 - simul.cur_time) // time_interval
    context_df =  pd.Series(context).to_frame().T
    cotexts_repeat_df = pd.concat([context_df]*repeat, ignore_index=True)
    cotexts_repeat_df['cur_time'] = time_interval
    cotexts_repeat_df['cur_time'][0] = simul.cur_time
    cotexts_repeat_df['cur_time'] = (cotexts_repeat_df['cur_time'].cumsum()).apply(format_time_to_str)

    context_feature_df = make_feature_set(cotexts_repeat_df)
    feature_df = context_feature_df[feature_cols]
    feature_arr = np.array(feature_df).astype(np.float32)
    featrue_tensor = torch.tensor(feature_arr).to(device)

    with torch.no_grad():
        model.eval()
        mu, logvar = model(featrue_tensor)
        ucb = mu + alpha * torch.exp(0.5*logvar)
        ucb_mean = torch.mean(ucb, dim=1, keepdims=True)
        action = torch.argmax(ucb_mean).to('cpu').numpy()
    adjcent_matrix1 = simul.temporal_graph.transform(simul.cur_time + action*time_interval)
    api_dict1 = simul.api_call(adjcent_matrix1, save_instance=False, save_history=False)

    # adjcent_matrix2 = simul.temporal_graph.transform(np.ceil(api_dict1['req_leaving_time']).astype(int))
    # api_dict2 = simul.api_call(adjcent_matrix2, save_instance=False, save_history=False)
    select_context = feature_df.loc[[action]].to_numpy().astype(np.float32)

    # Training Model
    model.train()
    optimizer.zero_grad()
    select_context_torch = torch.tensor(select_context).to(device)
    mu, logvar = model(select_context_torch)
    std = torch.exp(0.5*logvar)

    # reward_x = api_dict2['path_time'] - api_dict1['path_time'] - 8
    reward_x = (simul.cur_time + action*time_interval + api_dict1['path_time']) - simul.target_time
    reward = reward_function(torch.tensor(reward_x)).to(device)

    loss = loss_gaussian(mu, reward, std**2)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        # summary
        summary_of_episodes = f"(reward_x: {reward_x:.1f}) Start: {simul.start_time}, action: {simul.start_time + action*time_interval}, TimeMachine {simul.call_time_TimeMachine}, Target: {simul.target_time}"
        print(f"  → {summary_of_episodes}")

        # history
        history_t['context'] = select_context.squeeze()
        history_t["start_time"] = simul.start_time
        history_t["start_node"] = simul.start_node
        history_t["target_time"] = simul.target_time
        history_t["target_node"] = simul.target_node
        history_t["end_of_round"] = int(np.ceil(api_dict1['req_leaving_time']))
        history_t['action'] = action
        history_t['call_time'] = simul.start_time + action*time_interval
        history_t['call_time_TimaMachine'] = simul.call_time_TimeMachine
        history_t["residual_call_time"] = history_t['call_time_TimaMachine'] - history_t['call_time']
        history_t["reward_x"] = reward_x
        history_t["reward"] = reward.detach().to('cpu').numpy()
        history_t["summary_of_episodes"] = summary_of_episodes
        # history_t['weights'] = model.state_dict()
        history.append(history_t)
        print()

        # save weights
        path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\240826_Alg2_2_DirectEnsemble_weights.pkl'
        cPickle.dump(model.state_dict(), open(path, 'wb'))
print('save_weights')

# save history
path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\240826_Alg2_2_DirectEnsemble_history.pkl'
cPickle.dump(history, open(path, 'wb'))
print('save_history')


# history.drop('weights', axis=1)

# copy to clipboard
pd.DataFrame(history).to_clipboard()
pd.DataFrame(simul.history_time_machine).to_clipboard()
pd.DataFrame(simul.history).to_clipboard()
pd.DataFrame(simul.history_full_info).to_clipboard()


# x[np.argmax(y_hat)]


# visualize
plt.figure(figsize=(20,3))
plt.plot(np.arange(len(history)),pd.DataFrame(history)['reward_x'], alpha=0.3)
plt.scatter(np.arange(len(history)),pd.DataFrame(history)['reward_x'], alpha=0.7, edgecolor='white')
plt.show()

# visualize
plt.figure(figsize=(20,3))
plt.plot(np.arange(len(history)),pd.DataFrame(history)['reward'], alpha=0.3)
plt.scatter(np.arange(len(history)),pd.DataFrame(history)['reward'], alpha=0.7, edgecolor='white')
plt.show()




# # load_history
# load_path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject'
# history1 = cPickle.load(open(f"{load_path}/240816_UCB1_history.pkl", 'rb'))
# history2 = cPickle.load(open(f"{load_path}/240816_UCB2_history.pkl", 'rb'))
# history3_1 = cPickle.load(open(f"{load_path}/240820_UCB3_history.pkl", 'rb'))
# history3_2 = cPickle.load(open(f"{load_path}/240820_UCB3_2_history.pkl", 'rb'))
# history3_3 = cPickle.load(open(f"{load_path}/240820_UCB3_3_history.pkl", 'rb'))
# history3 = cPickle.load(open(f"{load_path}/240826_Alg2_2_DirectEnsemble_history.pkl", 'rb'))
# print('load_history')


history = history3
history_df = pd.DataFrame(history).iloc[:10000]
history_df


plt.figure(figsize=(14,3))
plt.plot(np.arange(len(history_df)), history_df['reward_x'], alpha=0.3)
plt.scatter(np.arange(len(history_df)), history_df['reward_x'], alpha=0.7, edgecolor='white')
plt.ylabel('y - 8 - y_hat')
plt.xlabel('Episodes')
plt.axhline(8, color='red', alpha=0.3)
plt.axhline(-5, color='red', alpha=0.3)
plt.show()


plt.figure(figsize=(14,3))
plt.plot(np.arange(len(history_df)), history_df['reward'] - 100, alpha=0.3)
plt.scatter(np.arange(len(history_df)), history_df['reward'] - 100, alpha=0.7, edgecolor='white')
# plt.ylim(-2000, 10)
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.show()











###########################################################################################################

# Feature-Set Version 2-3

# import httpimport
# remote_library_url = 'https://raw.githubusercontent.com/kimds929/'


# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_DataFrame import DF_Summary

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_Plot import ttest_each, violin_box_plot, distbox

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_MachineLearning import ScalerEncoder, DataSet

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_DeepLearning import EarlyStopping

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_Torch import TorchDataLoader, TorchModeling, AutoML

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim


class RunningNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(RunningNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)
    
    def forward(self, x):
        if self.training:
            # 현재 입력의 평균과 분산 계산
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Running mean and variance 업데이트
            self.running_mean.data = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean.data
            self.running_var.data = self.momentum * batch_var + (1 - self.momentum) * self.running_var.data
            
            # 현재 입력 데이터 정규화
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # 테스트 모드에서는 running mean과 variance 사용
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        return x_normalized

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(),
                batchNorm=True, runningNorm=False, dropout=0.5):
        super().__init__()
        ff_block = [nn.Linear(input_dim, output_dim)]
        if activation:
            ff_block.append(activation)
        if batchNorm:
            ff_block.append(nn.BatchNorm1d(output_dim))
        if runningNorm:
            ff_block.append(nn.RunningNorm(output_dim))
        if dropout > 0:
            ff_block.append(nn.Dropout(dropout))
        self.ff_block = nn.Sequential(*ff_block)
    
    def forward(self, x):
        return self.ff_block(x)

# ★Mu/Var Ensemble only last layer
class DirectEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, n_models=10):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim*2*n_models, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.n_models = n_models
        self.output_dim = output_dim

    # train step
    def train_forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        mu_logvar = (x)
        mu, logvar = torch.split(mu_logvar, self.n_models, dim=1)
        logvar = torch.clamp(logvar, min=-10, max=20) 
        return mu, logvar

    # eval step : 여러 번 샘플링하여 평균과 분산 계산
    def predict(self, x, idx=None):
        mu, logvar = self.train_forward(x)
        if idx is None:
            mu_mean = torch.mean(mu, dim=1, keepdims=True)
            logvar_mean = torch.mean(logvar, dim=1, keepdims=True)
        else:
            mu_mean = torch.mean(mu[:, idx], dim=1, keepdims=True)
            logvar_mean = torch.mean(logvar[:, idx], dim=1, keepdims=True)
        return  mu_mean, logvar_mean

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x)

# ★ Sample Ensemble only last layers Model
class SampleEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, n_samples=10):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim*n_samples, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.n_samples = n_samples
        self.output_dim = output_dim

    
    def forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        # return ensemble_outputs
        mu = torch.mean(x, dim=1, keepdims=True)
        var = torch.var(x, dim=1, keepdims=True)
        logvar = torch.log(var)
        return mu, logvar

# point estimate
class DirectEstimate(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.output_dim = output_dim

    def forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection
        logmu = x
        logmu = torch.clamp(logmu, min=-20, max=7) 
        return logmu




#########################################################################################################
import torch
class RewardFunctionTorch():
    def __init__(self):
        self.sigma1 = 25 
        self.mu1 = -13
        self.mu2 = 0
        self.sigma2 = 7
    
    def true_f(self, x):
        gaussian1 = torch.exp(-1/(self.sigma1**2)*(x-self.mu1)**2)
        gaussian2 = torch.exp(-1/(self.sigma2**2)*(x-self.mu2)**2)
        return gaussian1 * (x < self.mu1) + 1 * (x >= self.mu1) * (x < self.mu2)+  gaussian2 * (x >= self.mu2)

    def forward(self, x):
        return self.true_f(x)
        # return self.true_f(x) + torch.randn_like(x)*0.01

    def __call__(self, x):
        return self.forward(x)

x = torch.linspace(-100,100, 100)
reward_f = RewardFunctionTorch()
plt.plot(x,reward_f(x), 'o-')
# plt.axvline(-37, color='red', alpha=0.3)
plt.axvline(0, color='red', alpha=0.3)
plt.axvline(-13, color='red', alpha=0.3)
# plt.axvline(23, color='red', alpha=0.3)
plt.show()


#########################################################################################################
# Feature-Set Version 2-3
feature_set_common = [
                    'cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                    'cur_time_hour_min',
                    # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                    # 'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                    # 'target_time_hour_min',
                    # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
                    'cur_point_lat','cur_point_lng',
                    'target_point_lat', 'target_point_lng'
                    ]

# feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
feature_set_TimeMachine = [
                        'call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        'call_time_TimeMachine_hour_min',
                        # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        # 'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 
                        'path_time_TimeMachine',
                        # 'movedist_TimeMachine',
                        'remain_req_leaving_time_TimeMachine'
                        ]

# feature data (x)
feature_cols = feature_set_common + feature_set_TimeMachine + ['path_time_LastAPI']
contexts_feature_df = make_feature_set(pd.DataFrame(contexts))
contexts_feature_arr = np.array(contexts_feature_df[feature_cols]).astype(np.float32)
contexts_featrue_tensor = torch.tensor(contexts_feature_arr)

# api_data (y)
true_api_df = pd.DataFrame(true_api)
true_api_arr = np.array(true_api_df['path_time']).astype(np.float32).reshape(-1,1)
true_api_tensor = torch.tensor(true_api_arr)

from torch.utils.data import DataLoader, TensorDataset
batch_size = 64
train_dataset = TensorDataset(contexts_featrue_tensor, true_api_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)




import httpimport
remote_library_url = 'https://raw.githubusercontent.com/kimds929/'

with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping

with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# loss_gaussian = nn.GaussianNLLLoss()
def mse_loss(model, x, y):
    logmu = model(x)
    mu = torch.exp(logmu)
    loss = torch.nn.functional.mse_loss(mu, y)
    return loss

hidden_dim = 64
output_dim = 1
n_layers = 5

model_pathtime = DirectEstimate(input_dim=len(feature_cols), hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers)
model_pathtime.to(device)

learning_rate = 1e-2
optimizer_pathtime = optim.Adam(model_pathtime.parameters(), lr=learning_rate)
scheduler_pathtime = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_pathtime, T_0=10, T_mult=2)

tm = TorchModeling(model=model_pathtime, device=device)

tm.compile(optimizer=optimizer_pathtime
            ,loss_function = mse_loss
            , scheduler=scheduler_pathtime
            , early_stop_loss = EarlyStopping(patience=5)
            )
tm.train_model(train_loader=train_loader, epochs=100, display_earlystop_result=True, early_stop=False)
tm.test_model(test_loader=test_loader)

# tm.recompile(optimizer = optim.Adam(model_pathtime.parameters(), lr=1e-4))




#########################################################################################################
# Feature-Set Version 2-3
feature_set_common = [
                    'cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
                    'cur_time_hour_min',
                    # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
                    # 'target_time_Week', 'target_time_Sat', 'target_time_Sun',
                    # 'target_time_hour_min',
                    # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
                    'cur_point_lat','cur_point_lng',
                    'target_point_lat', 'target_point_lng'
                    ]

# feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
feature_set_TimeMachine = [
                        'call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
                        'call_time_TimeMachine_hour_min',
                        # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
                        # 'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 
                        'path_time_TimeMachine',
                        # 'movedist_TimeMachine',
                        'remain_req_leaving_time_TimeMachine'
                        ]

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from IPython.display import clear_output
import time
from tqdm.auto import tqdm
from six.moves import cPickle

simul = TemporalMapSimulation(temporal_graph)
feature_cols = feature_set_common + feature_set_TimeMachine

n_actions = 2; feature_dim = len(feature_cols); time_interval = 3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
alpha = 1
input_dim = feature_dim; hidden_dim = 32; output_dim = 1
n_models = 10; n_samples = 30; n_layers = 5
learning_rate = 1e-4

model = DirectEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_models=n_models, n_layers=n_layers)
model_pathtime = DirectEstimate(input_dim=input_dim+1, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers)

# load_path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject'
# load_weights = cPickle.load(open(f"{load_path}/240826_Alg2_2_DirectEnsemble_weights.pkl", 'rb'))
# model.load_state_dict(load_weights)
model.to(device)
model_pathtime.to(device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer_pathtime = optim.Adam(model_pathtime.parameters(), lr=learning_rate)
reward_function = RewardFunctionTorch()
loss_gaussian = nn.GaussianNLLLoss()
loss_mse = nn.MSELoss()


history = []
# history = cPickle.load(open(f"{load_path}/240826_Alg2_2_DirectEnsemble_history.pkl", 'rb'))


# ----------------------------------------------------------------------------------------------------
num_episodes = 10
for episode in tqdm(range(num_episodes)):
    history_t = {}

    # simul.reset_state(reset_all=False, save_interval=time_interval, verbose=1)
    simul.reset_state(reset_all=True, save_interval=time_interval, verbose=1)
    # prediction
    context = simul.run()

    repeat = (simul.target_time - 8 - simul.cur_time) // time_interval
    context_df =  pd.Series(context).to_frame().T
    cotexts_repeat_df = pd.concat([context_df]*repeat, ignore_index=True)
    cotexts_repeat_df['cur_time'] = time_interval
    cotexts_repeat_df['cur_time'][0] = simul.cur_time
    cotexts_repeat_df['cur_time'] = (cotexts_repeat_df['cur_time'].cumsum()).apply(format_time_to_str)

    context_feature_df = make_feature_set(cotexts_repeat_df)
    feature_df = context_feature_df[feature_cols]
    feature_arr = np.array(feature_df).astype(np.float32)
    featrue_tensor = torch.tensor(feature_arr).to(device)

    with torch.no_grad():
        model.eval()
        mu, logvar = model(featrue_tensor)
        ucb = mu + alpha * torch.exp(0.5*logvar)
        ucb_mean = torch.mean(ucb, dim=1, keepdims=True)
        action = torch.argmax(ucb_mean).to('cpu').numpy()
    adjcent_matrix1 = simul.temporal_graph.transform(simul.cur_time + action*time_interval)
    api_dict1 = simul.api_call(adjcent_matrix1, save_instance=False, save_history=False)
    api_dict1

    select_feature_df = feature_df.loc[[action]]
    select_feature_df['path_time_LastAPI'] = api_dict1['path_time']
    select_context = select_feature_df.to_numpy().astype(np.float32)
    
    # Train pathtime model ---------------------------------------------------------------------------
    model_pathtime.train()
    optimizer_pathtime.zero_grad()
    log_pred_path_time = model_pathtime(torch.tensor(select_context).to(device))
    pred_path_time = torch.exp(log_pred_path_time)

    with torch.no_grad():
        pred_path_time
        pred_leaving_time = int(np.ceil(simul.target_time - 8 -  pred_path_time.item()))
        adjcent_matrix2 = simul.temporal_graph.transform(pred_leaving_time)
        api_dict2 = simul.api_call(adjcent_matrix2, save_instance=False, save_history=False)
        true_path_time = torch.tensor(api_dict2['path_time'], dtype=torch.float32).view(-1,1).to(device)
    loss = loss_mse(pred_path_time, true_path_time)
    loss.backward()
    optimizer_pathtime.step()
    
    # Train action select model ---------------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    select_context_torch = torch.tensor(select_context[:,:-1]).to(device)
    mu, logvar = model(select_context_torch)
    std = torch.exp(0.5*logvar)
    reward_x = (pred_leaving_time + api_dict2['path_time']) - simul.target_time
    reward = reward_function(torch.tensor(reward_x)).to(device)
    loss = loss_gaussian(mu, reward, std**2)
    loss.backward()
    optimizer.step()

    # ----------------------------------------------------------------------------------------------

    with torch.no_grad():
        # history
        history_t['context'] = select_context.squeeze()
        history_t['action'] = action
        history_t['pred_leaving_time'] = pred_leaving_time
        history_t['pred_pathtime'] = pred_path_time.item()
        history_t['true_pathtime'] = true_path_time.item()
        history_t['pred_mu'] = torch.mean(mu).item()
        history_t['pred_std'] = torch.mean(std).item()
        history_t["reward_x"] = reward_x
        history_t["reward"] = reward.detach().to('cpu').numpy()
        
        # summary
        summary_of_episodes = f"reward_x: {reward_x:.1f}, reward: {reward:.1f}, action: {action}, pathtime: {history_t['pred_pathtime']:.1f} / {history_t['true_pathtime']:.1f}, mu/std: {history_t['pred_mu']:.3f}/{history_t['pred_std']:.3f}"
        print(f"\r  → {summary_of_episodes}")

        history_t["summary_of_episodes"] = summary_of_episodes
        # history_t['weights'] = model.state_dict()
        history.append(history_t)
        print()

        # save weights
        path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject'
        cPickle.dump(model.state_dict(), open(f"{path}/240829_Alg2_3_DirectEnsemble_weights.pkl", 'wb'))
        cPickle.dump(model_pathtime.state_dict(), open(f"{path}/240829_Alg2_3_DirectEnsemble_pathtime_weights.pkl", 'wb'))
        cPickle.dump(history, open(f"{path}/240829_Alg2_3_DirectEnsemble_history.pkl", 'wb'))
print('save weights & history')


# history.drop('weights', axis=1)

# copy to clipboard
pd.DataFrame(history).to_clipboard()
pd.DataFrame(simul.history_time_machine).to_clipboard()
pd.DataFrame(simul.history).to_clipboard()
pd.DataFrame(simul.history_full_info).to_clipboard()






# # load_history --------------------------------------------------------------------------------------------------------------------------------------
from six.moves import cPickle
load_path = r'D:\DataScience\SNU_DataScience\SNU_OhLab\CLab_Project\Model'
# load_path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\Model'
history = cPickle.load(open(f"{load_path}/240903_Alg2_4_DirectEnsemble_history.pkl", 'rb'))
# print('load_history')
len(history)



# visualize --------------------------------------------------------------------------------------------------------------------------------------
dist_y = [hist['loss_pathtime'] for hist in history]
dist_y = [hist['loss'] for hist in history]
dist_y = [np.sum(hist['actions']) for hist in history]
dist_y = [hist['reward_x'] for hist in history]
dist_y = [hist['reward'] for hist in history]
dist_y = [hist['true_leaving_info']['path_time'] - hist['pred_path_time_mu'] for hist in history]


dist_y = [hist['pred_path_time_mu'] for hist in history]
dist_y = [hist['true_leaving_info']['path_time'] for hist in history]

# visualize 
plt.figure(figsize=(14,3))
plt.plot(np.arange(len(history)), dist_y, 'o-')
# plt.yscale("symlog")
plt.ylim(0,2000)
plt.axhline(0, c='black', alpha=0.3)
plt.axhline(-13, c='red', alpha=0.3)
plt.show()



# visualize --------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(14,3))
plt.plot(np.arange(len(history_df)), history_df['reward_x'], alpha=0.3)
plt.scatter(np.arange(len(history_df)), history_df['reward_x'], alpha=0.7, edgecolor='white')
plt.ylabel('y - 8 - y_hat')
plt.xlabel('Episodes')
plt.axhline(8, color='red', alpha=0.3)
plt.axhline(-5, color='red', alpha=0.3)
plt.show()


plt.figure(figsize=(14,3))
plt.plot(np.arange(len(history_df)), history_df['reward'] - 100, alpha=0.3)
plt.scatter(np.arange(len(history_df)), history_df['reward'] - 100, alpha=0.7, edgecolor='white')
# plt.ylim(-2000, 10)
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------------

























###########################################################################################################
# Version 2-4


import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'

# response_files = requests.get("https://raw.githubusercontent.com/kimds929/CodeNote/main/42_Temporal_Spatial/DL13_Temporal_12_TemporalEmbedding.py")
# exec(response_files.text)
# response_files.text

with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/42_Temporal_Spatial/"):
    from DL13_Temporal_12_TemporalEmbedding import TemporalEmbedding
    from DL13_Spatial_11_SpatialEmbedding import SpatialEmbedding

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/47_Bayesian_Neural_Network/"):
    from BNN05_DensityEstimate_Regressor_Torch import FeedForwardBlock



# -------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------------------------------------------------------------------------
class CombinedEmbedding(nn.Module):
    def __init__(self, input_dim, t_input_dim,  output_dim=None, t_emb_dim=8, t_hidden_dim=None, s_emb_dim=None, **spatial_kwargs):
        super().__init__()

        # temporal embedding layer (t_input_dim)
        self.t_input_dim = t_input_dim
        self.temporal_embedding = TemporalEmbedding(input_dim=t_input_dim, embed_dim=t_emb_dim, hidden_dim=t_hidden_dim)

        # spatial embedding layer (4)
        self.spatial_embedding = SpatialEmbedding(embed_dim=s_emb_dim, **spatial_kwargs)
        
        # other feature dimension (input_dim - t_input_dim - 4)
        self.other_feature_dim = input_dim - t_input_dim - 4

        # embed_dim
        self.output_dim = output_dim
        self.embed_dim = self.temporal_embedding.embed_dim + self.spatial_embedding.embed_dim + self.other_feature_dim

        # fc block
        if output_dim is not None:
            self.fc_layer = nn.Linear(self.embed_dim, output_dim)
            self.embed_dim = output_dim
        
    def forward(self, x):
        temporal_features = self.temporal_embedding(x[:,:self.t_input_dim])
        spatial_features = self.spatial_embedding(x[:,self.t_input_dim:self.t_input_dim+2], x[:,self.t_input_dim+2:self.t_input_dim+4])
        other_features = x[:,self.t_input_dim+4:]
        outputs = torch.cat([temporal_features, spatial_features, other_features], dim=1)
        
        if self.output_dim is not None:
            outputs = self.fc_layer(outputs)
        return outputs


class EnsembleCombinedModel(nn.Module):
    def __init__(self, input_dim, output_dim, t_input_dim, hidden_dim,  n_layers=3, n_models = 10, n_output=1,
                embed_output_dim=None, t_emb_dim=8, t_hidden_dim=None, s_emb_dim=None, **spatial_kwargs):
        super().__init__()

        # combined embedding layer
        self.combined_embedding = CombinedEmbedding(input_dim=input_dim, t_input_dim=t_input_dim,  output_dim=embed_output_dim,
                                                    t_emb_dim=t_emb_dim, t_hidden_dim=t_hidden_dim, s_emb_dim=s_emb_dim, **spatial_kwargs)
        self.embed_output_dim = self.combined_embedding.embed_dim

        # fc block
        self.fc_block = nn.ModuleDict({'in_layer':FeedForwardBlock(self.embed_output_dim, hidden_dim, batchNorm=False, dropout=0)})

        out_dim = output_dim*n_models if n_output == 1 else output_dim*n_output*n_models
        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.fc_block[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False, dropout=0)
            else:
                self.fc_block['out_layer'] = FeedForwardBlock(hidden_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        
        self.output_dim = output_dim
        self.n_output = n_output
        self.n_layers = n_layers
        self.n_models = n_models

    def train_forward(self, x):
        x = self.combined_embedding(x)

        for layer_name, layer in self.fc_block.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection
        
        if self.n_output == 1:
            return x
        else:
            return torch.split(x, self.output_dim*self.n_models, dim=1)

    def predict(self, x, idx=None):
        if self.n_output == 1:
            if idx is None:
                return self.train_forward(x).mean(dim=1, keepdims=True)
            else:
                return self.train_forward(x)[:, idx].mean(dim=1, keepdims=True)
        else:
            if idx is None:
                return tuple([output.mean(dim=1, keepdims=True) for output in self.train_forward(x)])
            else:
                return tuple([output[:, idx].mean(dim=1, keepdims=True) for output in self.train_forward(x)])

    def forward(self, x, idx=None):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x, idx)

# ce = CombinedEmbedding(input_dim=8, output_dim=32, t_input_dim=3)
# ce = CombinedEmbedding(input_dim=8, t_input_dim=3)
# ce(torch.rand(5,8)).shape

# ecm = EnsembleCombinedModel(input_dim=8, output_dim=1, t_input_dim=3, hidden_dim=64, n_output=5)
# ecm.train()
# ecm.eval()
# out = ecm(torch.rand(5,8))



# ★Multiple output Ensemble only last layer
class DirectEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_output=1, n_layers=3, n_models=10):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False, dropout=0)})

        out_dim = output_dim*n_models if n_output == 1 else output_dim*n_output*n_models

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False, dropout=0)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        
        self.output_dim = output_dim
        self.n_output = n_output
        self.n_layers = n_layers
        self.n_models = n_models

    # train step
    def train_forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        if self.n_output == 1:
            return x
        else:
            return torch.split(x, self.output_dim*self.n_models, dim=1)

    # eval step : ensemble mean
    def predict(self, x, idx=None):
        if self.n_output == 1:
            if idx is None:
                return self.train_forward(x).mean(dim=1, keepdims=True)
            else:
                return self.train_forward(x)[:, idx].mean(dim=1, keepdims=True)
        else:
            if idx is None:
                return tuple([output.mean(dim=1, keepdims=True) for output in self.train_forward(x)])
            else:
                return tuple([output[:, idx].mean(dim=1, keepdims=True) for output in self.train_forward(x)])

    def forward(self, x, idx=None):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x, idx)


# de = DirectEnsemble(input_dim=14, hidden_dim=32, output_dim=1, n_output=1, n_layers=3, n_models=10)
# de = DirectEnsemble(input_dim=14, hidden_dim=32, output_dim=1, n_output=2, n_layers=3, n_models=10)
# de.eval()
# de(torch.rand(8,14))


# -------------------------------------------------------------------------------------------
import torch
class RewardFunctionTorch():
    def __init__(self):
        self.sigma1 = 15 
        self.mu1 = -13
        self.mu2 = 0
        self.sigma2 = 4
    
    def true_f(self, x):
        gaussian1 = torch.exp(-1/(self.sigma1**2)*(x-self.mu1)**2)
        gaussian2 = torch.exp(-1/(self.sigma2**2)*(x-self.mu2)**2)
        return gaussian1 * (x < self.mu1) + 1 * (x >= self.mu1) * (x < self.mu2)+  gaussian2 * (x >= self.mu2)

    def forward(self, x):
        return self.true_f(x)
        # return self.true_f(x) + torch.randn_like(x)*0.01

    def __call__(self, x):
        return self.forward(x)

x = torch.linspace(-100,100, 100)
reward_f = RewardFunctionTorch()
plt.plot(x,reward_f(x), 'o-')
# plt.axvline(-37, color='red', alpha=0.3)
plt.axvline(0, color='red', alpha=0.3)
plt.axvline(-13, color='red', alpha=0.3)
# plt.axvline(23, color='red', alpha=0.3)
plt.show()



#########################################################################################################
# Feature-Set Version 2-4

# feature_set_common : cur_time, target_time, cur_point, target_point
# feature_set_common = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
#                         'cur_time_hour_min',
#                         # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
#                         'target_time_Week', 'target_time_Sat', 'target_time_Sun',
#                         'target_time_hour_min',
#                         # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
#                         'cur_point_lat','cur_point_lng', 'target_point_lat', 'target_point_lng']

# # feature_set_LastAPI : call_time_LastAPI, call_point_LastAPI, path_time_LastAPI
# feature_set_LastAPI = ['call_time_LastAPI_Week', 'call_time_LastAPI_Sat', 'call_time_LastAPI_Sun',
#                         'call_time_LastAPI_hour_min',
#                         # 'call_time_LastAPI_Commute', 'call_time_LastAPI_Day', 'call_time_LastAPI_Night',
#                         'call_point_LastAPI_lat', 'call_point_LastAPI_lng', 'path_time_LastAPI'
#                         # , 'movedist_LastAPI', 'remain_req_leaving_time_LastAPI'
#                         ]

# # feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
# feature_set_TimeMachine = ['call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
#                         'call_time_TimeMachine_hour_min',
#                         # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
#                         'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine',
#                         #  'movedist_TimeMachine', 'remain_req_leaving_time_TimeMachine'
#                          ]


# -------------------------------------------------------------------------------------------
def make_feature_set_embedding(context_df, temporal_cols, spatial_cols, other_cols, fillna=None):
    # temproal features preprocessing
    temproal_arr = context_df[temporal_cols].applymap(lambda x: format_str_to_time(x) if type(x) == str else x).fillna(0).to_numpy().astype(np.float32)

    # spatial features preprocessing
    spatial_cols_transform = list(np.stack([[f"{cols}_x", f"{cols}_y"] for cols in spatial_cols]).ravel())

    spatial_arr_stack = np.stack(list(context_df[spatial_cols].applymap(lambda x: np.array(x)).to_dict('list').values())).astype(np.float32)
    spatial_arr = spatial_arr_stack.transpose(1,0,2).reshape(-1,4)

    # other features
    other_arr = context_df[other_cols].to_numpy().astype(np.float32)

    # # combine and transform to dataframe
    df_columns = temporal_cols + spatial_cols_transform + other_cols
    df_transform = pd.DataFrame(np.hstack([temproal_arr, spatial_arr, other_arr]),
                             columns=df_columns, index=context_df.index)
    if fillna is not None:
        df_transform = df_transform.fillna(fillna)
    return df_transform

# # temproal feature preprocessing
# temproal_arr = context_df[temporal_feature_cols].applymap(lambda x: format_str_to_time(x)).fillna(0).to_numpy().astype(np.float32)
# # temproal_tensor = torch.tensor(temproal_arr)

# # spatial feature preprocessing
# spatial_arr_stack = np.stack(list(context_df[spatial_feature_cols].applymap(lambda x: np.array(x)).to_dict('list').values())).astype(np.float32)
# spatial_arr = spatial_arr_stack.transpose(1,0,2).reshape(-1,4)
# # spatial_tensor = torch.tensor(spatial_arr)

# # other feature
# other_arr = context_df[continuous_feature_cols].fillna(0).to_numpy().astype(np.float32)

# # combine & transform to dataframe
# spatial_cols_transform = list(np.stack([[f"{cols}_x", f"{cols}_y"] for cols in spatial_feature_cols]).ravel())
# df_columns = temporal_feature_cols + spatial_cols_transform + other_feature_cols
# df_tranform = pd.DataFrame(np.hstack([temproal_arr, spatial_arr, other_arr]), columns=df_columns, index=context_df.index)


# (use)
# context_df_transform = make_feature_set_embedding(context_df, 
#             temporal_feature_cols, spatial_feature_cols, other_feature_cols)
# context_df_transform

# -------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------

from IPython.display import clear_output
import time
from tqdm.auto import tqdm
from six.moves import cPickle


# simulaiton -------------------------------------------------------------------------------------
simul = TemporalMapSimulation(temporal_graph)
time_interval = 3

# feature -------------------------------------------------------------------------------------
# temporal_feature_cols = ['cur_time','target_time', 'call_time_TimeMachine', 'call_time_LastAPI']
temporal_feature_cols = ['cur_time','target_time', 'call_time_TimeMachine', 'remain_time_from_TimeMachine']
spatial_feature_cols = ['cur_point', 'target_point']
other_feature_cols = ['path_time_TimeMachine']
other_feature_cols_pathtime = ['path_time_TimeMachine', 'path_time_LastAPI']


# model -------------------------------------------------------------------------------------
input_dim = len(temporal_feature_cols) + len(spatial_feature_cols)*2 + len(other_feature_cols)
input_dim_pathtime = len(temporal_feature_cols) + len(spatial_feature_cols)*2 + len(other_feature_cols_pathtime)
hidden_dim = 64

model = EnsembleCombinedModel(input_dim=input_dim, output_dim=1, t_input_dim=len(temporal_feature_cols),
                hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=2,
                embed_output_dim=64, t_hidden_dim=32, s_emb_dim=32, t_emb_dim=16,
                coord_embed_dim=16, coord_hidden_dim=32, coord_depth=2, grid_size=32, periodic_embed_dim=9)

# model = EnsembleCombinedModel(input_dim=input_dim, output_dim=1, t_input_dim=len(temporal_feature_cols),
#                 hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=2)
# model(torch.rand(5,8))
sum(p.numel() for p in model.parameters() if p.requires_grad)     # 61039

model_pathtime = EnsembleCombinedModel(input_dim=input_dim_pathtime, output_dim=1, t_input_dim=len(temporal_feature_cols),
                hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=1,
                embed_output_dim=64, t_hidden_dim=32, s_emb_dim=32, t_emb_dim=16,
                coord_embed_dim=16, coord_hidden_dim=32, coord_depth=2, grid_size=32, periodic_embed_dim=9)
# model_pathtime = EnsembleCombinedModel(input_dim=input_dim_pathtime, output_dim=1, t_input_dim=len(temporal_feature_cols),
#                 hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=1)
# model_pathtime(torch.rand(5,9))
sum(p.numel() for p in model_pathtime.parameters() if p.requires_grad)     # 59813


# compile -------------------------------------------------------------------------------------
alpha = 1

optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer_pathtime = optim.Adam(model_pathtime.parameters(), lr=1e-3)
reward_function = RewardFunctionTorch()
loss_gaussian = nn.GaussianNLLLoss()
loss_mse = nn.MSELoss()

history = []
# history = cPickle.load(open(f"{load_path}/240826_Alg2_2_DirectEnsemble_history.pkl", 'rb'))


# learn -------------------------------------------------------------------------------------
num_episodes = 100
for episode in tqdm(range(num_episodes)):
    history_t = {}
    # simul.reset_state(reset_all=False, save_interval=time_interval, verbose=0)
    simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)

    context = simul.run()

    repeat = (simul.target_time - 8 - simul.cur_time) // time_interval
    context_df =  pd.Series(context).to_frame().T
    cotexts_repeat_df = pd.concat([context_df]*repeat, ignore_index=True)
    cotexts_repeat_df['cur_time'] = time_interval
    cotexts_repeat_df['cur_time'][0] = simul.cur_time
    cotexts_repeat_df['cur_time'] = (cotexts_repeat_df['cur_time'].cumsum()).apply(format_time_to_str)
    cotexts_repeat_df['remain_time_from_TimeMachine'] = cotexts_repeat_df['req_leaving_time'].apply(format_str_to_time) - cotexts_repeat_df['cur_time'].apply(format_str_to_time)

    contexts_feature_df = make_feature_set_embedding(cotexts_repeat_df, 
                temporal_feature_cols, spatial_feature_cols, other_feature_cols)
    contexts_tensor = torch.tensor(contexts_feature_df.to_numpy()).to(device)
    # contexts_tensor.shape

    # calculate UCB & select action
    with torch.no_grad():
        model.eval()
        mu, logvar = model(contexts_tensor)
        logvar = torch.clamp(logvar, min=-10, max=10)   # clamp

        ucb = mu + alpha * torch.exp(0.5*logvar)
        ucb_mean = torch.mean(ucb, dim=1, keepdims=True)
        action = torch.argmax(ucb_mean).to('cpu').item()
    action_time = simul.cur_time + action*time_interval
    adjcent_matrix1 = simul.temporal_graph.transform(action_time)
    api_dict1 = simul.api_call(adjcent_matrix1, save_instance=False, save_history=False)

    # select contexts for selected action
    select_context_df = contexts_feature_df.loc[[action]]
    select_context_df['path_time_LastAPI'] = np.array(api_dict1['path_time']).astype(np.float32)
    select_context_tensor = torch.tensor(select_context_df.to_numpy()).to(device)
    select_context_tensor.shape

    # Train pathtime model ---------------------------------------------------------------------------
    model_pathtime.train()
    optimizer_pathtime.zero_grad()
    pred_change_path = model_pathtime(select_context_tensor)
    pred_change_path = torch.clamp(pred_change_path, min=-api_dict1['path_time'], max=1440)       # clamp
    

    with torch.no_grad():
        pred_pathtime = api_dict1['path_time'] + pred_change_path.mean().item()
        pred_leaving_time = int(np.ceil(simul.target_time - 8 - (pred_pathtime)))
        adjcent_matrix2 = simul.temporal_graph.transform(pred_leaving_time)
        api_dict2 = simul.api_call(adjcent_matrix2, save_instance=False, save_history=False)
        true_change_pathtime = torch.tensor(api_dict2['path_time'] - api_dict1['path_time'], dtype=torch.float32).view(-1,1).to(device)

    loss_pathtime = loss_mse(pred_change_path, true_change_pathtime)
    loss_pathtime.backward()
    optimizer_pathtime.step()

    # Train action select model ---------------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    mu, logvar = model(select_context_tensor[:,:-1])
    logvar = torch.clamp(logvar, min=-10, max=20)   # clamp
    std = torch.exp(0.5*logvar)
    # std = torch.clamp(std, min=0, max=10000)   # clamp
    with torch.no_grad():
        reward_x = (pred_leaving_time + api_dict2['path_time']) - simul.target_time
        reward = reward_function(torch.tensor(reward_x)).reshape(-1,1)
        # logit_reward = ( 0.5* torch.log((1+ reward + 1e-10) / (1 - reward + 1e-10)) ).to(device)   # tanh logit
        logit_reward = ( torch.log( reward / ( 1- reward + 1e-10)) ).to(device)   # logit

    loss = loss_gaussian(mu, logit_reward, std**2)
    loss.backward()
    optimizer.step()


    # ----------------------------------------------------------------------------------------------

    with torch.no_grad():
        # history
        history_t['loss_pathtime'] = loss_pathtime.item()
        history_t['loss'] = loss.item()
        history_t['context'] = select_context_df.to_numpy()
        history_t['action'] = action
        history_t['pred_leaving_time'] = pred_leaving_time
        history_t['pred_pathtime'] = pred_pathtime
        history_t['true_pathtime'] = api_dict2['path_time']
        history_t['pred_mu'] = torch.mean(mu).item()
        history_t['pred_std'] = torch.mean(std).item()
        history_t["reward_x"] = reward_x
        history_t["reward"] = reward.item()

        history_t['start_time'] = simul.start_time
        history_t['target_time'] = simul.target_time
        history_t['action_time'] = action_time
        history_t['time_machine_time'] = format_str_to_time(simul.history_time_machine[0]['req_leaving_time'])
        history_t['pathtime_TimeMachine'] = simul.history_time_machine[0]['path_time_TimeMachine']
        # summary
        summary_of_episodes = f"loss_pathime: {loss_pathtime.item():.2f}, loss_action: {loss.item():.2f}, reward_x: {reward_x:.1f}, reward: {reward.item():.1f}, action: {action}, pathtime: {history_t['pred_pathtime']:.1f} / {history_t['true_pathtime']:.1f}, mu/std: {history_t['pred_mu']:.2f}/{history_t['pred_std']:.2f}"
        print(f"\r  → {summary_of_episodes}")

        history_t["summary_of_episodes"] = summary_of_episodes
        # history_t['weights'] = model.state_dict()
        history.append(history_t)
        print()

        # save weights
        path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\Model'
        # path = "/content/drive/MyDrive/SNU_OhLAB/CLabProject"
        cPickle.dump(model.state_dict(), open(f"{path}/240903_Alg2_4_DirectEnsemble_weights.pkl", 'wb'))
        cPickle.dump(model_pathtime.state_dict(), open(f"{path}/240903_Alg2_4_DirectEnsemble_pathtime_weights.pkl", 'wb'))
        cPickle.dump(history, open(f"{path}/240903_Alg2_4_DirectEnsemble_history.pkl", 'wb'))





# # load_history --------------------------------------------------------------------------------------------------------------------------------------
from six.moves import cPickle
load_path = r'D:\DataScience\SNU_DataScience\SNU_OhLab\CLab_Project\Model'
# load_path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\Model'
# history = cPickle.load(open(f"{load_path}/240903_Alg2_4_DirectEnsemble1_history.pkl", 'rb'))
history = cPickle.load(open(f"{load_path}/240903_Alg2_4_DirectEnsemble2_history.pkl", 'rb'))
# print('load_history')
len(history)




# visualize --------------------------------------------------------------------------------------------------------------------------------------
dist_y = [hist['loss_pathtime'] for hist in history]
dist_y = [hist['loss'] for hist in history]
dist_y = [np.sum(hist['action']) for hist in history]
dist_y = [hist['reward_x'] for hist in history]
dist_y = [hist['reward'] for hist in history]

dist_y = [ (hist['action_time'] - hist['start_time']) /(hist['target_time'] - hist['start_time']) for hist in history]
dist_y2 = [ (hist['target_time']-8-hist['pathtime_TimeMachine']- hist['start_time']) /(hist['target_time'] - hist['start_time']) for hist in history]
dist_y = [hist['action_time'] for hist in history]
dist_y = [hist['pred_pathtime'] - hist['true_pathtime'] for hist in history]

dist_y = [hist['pred_mu'] for hist in history]
dist_y = [hist['pred_std'] for hist in history]


# dist_y = dist_y[-1000:]
# dist_y2 = dist_y2[-1000:]

# # visualize
plt.figure(figsize=(14,3))
plt.plot(np.arange(len(dist_y)), dist_y, alpha=0.3)
plt.scatter(np.arange(len(dist_y)), dist_y, alpha=0.7, edgecolor='white')

# plt.plot(np.arange(len(dist_y)), dist_y2, alpha=0.3)
# plt.scatter(np.arange(len(dist_y)), dist_y2, alpha=0.7, edgecolor='white')

# plt.ylim(0,1000)
# plt.yscale("symlog")
# plt.ylabel('y - 8 - y_hat')
plt.xlabel('Episodes')
plt.axhline(0, c='black', alpha=0.3)
plt.axhline(-13, c='red', alpha=0.3)
plt.show()


# moving average
window_size = 100

# dist_y = np.array([hist['reward_x'] for hist in history])
# dist_y = np.array([hist['reward'] for hist in history])
dist_y = np.array([(hist['reward_x'] >= -13 and hist['reward_x'] < 0) for hist in history])
move_avg_y = np.convolve(dist_y, np.ones(window_size)/window_size, mode='valid')
# np.convolve(data, np.ones(window_size)/window_size, mode='valid')   # moving average

plt.figure(figsize=(14,3))
plt.plot(np.arange(len(move_avg_y)), move_avg_y, alpha=0.3)
# plt.scatter(np.arange(len(move_avg_y)), move_avg_y, alpha=0.7, edgecolor='white')
plt.xlabel('Episodes')
plt.axhline(1, c='black', alpha=0.3)
plt.axhline(0.95, c='red', alpha=0.3)
plt.axhline(0.90, c='coral', alpha=0.3)
plt.axhline(0.80, c='orange', alpha=0.3)
# plt.axhline(0, c='black', alpha=0.3)
# plt.axhline(-13, c='red', alpha=0.3)
plt.show()


# -----------------------------------------------------------------------------------------------------------------------------------------------
































###########################################################################################################
# Version 2-5


import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'

# response_files = requests.get("https://raw.githubusercontent.com/kimds929/CodeNote/main/42_Temporal_Spatial/DL13_Temporal_12_TemporalEmbedding.py")
# exec(response_files.text)
# response_files.text

with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/42_Temporal_Spatial/"):
    from DL13_Temporal_12_TemporalEmbedding import TemporalEmbedding
    from DL13_Spatial_11_SpatialEmbedding import SpatialEmbedding

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/47_Bayesian_Neural_Network/"):
    from BNN05_DensityEstimate_Regressor_Torch import FeedForwardBlock



# -------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------------------------------------------------------------------------
class CombinedEmbedding(nn.Module):
    def __init__(self, input_dim, t_input_dim,  output_dim=None, t_emb_dim=8, t_hidden_dim=None, s_emb_dim=None, **spatial_kwargs):
        super().__init__()

        # temporal embedding layer (t_input_dim)
        self.t_input_dim = t_input_dim
        self.temporal_embedding = TemporalEmbedding(input_dim=t_input_dim, embed_dim=t_emb_dim, hidden_dim=t_hidden_dim)

        # spatial embedding layer (4)
        self.spatial_embedding = SpatialEmbedding(embed_dim=s_emb_dim, **spatial_kwargs)
        
        # other feature dimension (input_dim - t_input_dim - 4)
        self.other_feature_dim = input_dim - t_input_dim - 4

        # embed_dim
        self.output_dim = output_dim
        self.embed_dim = self.temporal_embedding.embed_dim + self.spatial_embedding.embed_dim + self.other_feature_dim

        # fc block
        if output_dim is not None:
            self.fc_layer = nn.Linear(self.embed_dim, output_dim)
            self.embed_dim = output_dim
        
    def forward(self, x):
        temporal_features = self.temporal_embedding(x[:,:self.t_input_dim])
        spatial_features = self.spatial_embedding(x[:,self.t_input_dim:self.t_input_dim+2], x[:,self.t_input_dim+2:self.t_input_dim+4])
        other_features = x[:,self.t_input_dim+4:]
        outputs = torch.cat([temporal_features, spatial_features, other_features], dim=1)
        
        if self.output_dim is not None:
            outputs = self.fc_layer(outputs)
        return outputs


class EnsembleCombinedModel(nn.Module):
    def __init__(self, input_dim, output_dim, t_input_dim, hidden_dim,  n_layers=3, n_models = 10, n_output=1,
                embed_output_dim=None, t_emb_dim=8, t_hidden_dim=None, s_emb_dim=None, **spatial_kwargs):
        super().__init__()

        # combined embedding layer
        self.combined_embedding = CombinedEmbedding(input_dim=input_dim, t_input_dim=t_input_dim,  output_dim=embed_output_dim,
                                                    t_emb_dim=t_emb_dim, t_hidden_dim=t_hidden_dim, s_emb_dim=s_emb_dim, **spatial_kwargs)
        self.embed_output_dim = self.combined_embedding.embed_dim

        # fc block
        self.fc_block = nn.ModuleDict({'in_layer':FeedForwardBlock(self.embed_output_dim, hidden_dim, batchNorm=False, dropout=0)})

        out_dim = output_dim*n_models if n_output == 1 else output_dim*n_output*n_models
        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.fc_block[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False, dropout=0)
            else:
                self.fc_block['out_layer'] = FeedForwardBlock(hidden_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        
        self.output_dim = output_dim
        self.n_output = n_output
        self.n_layers = n_layers
        self.n_models = n_models

    def train_forward(self, x):
        x = self.combined_embedding(x)

        for layer_name, layer in self.fc_block.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection
        
        if self.n_output == 1:
            return x
        else:
            return torch.split(x, self.output_dim*self.n_models, dim=1)

    def predict(self, x, idx=None):
        if self.n_output == 1:
            if idx is None:
                return self.train_forward(x).mean(dim=1, keepdims=True)
            else:
                return self.train_forward(x)[:, idx].mean(dim=1, keepdims=True)
        else:
            if idx is None:
                return tuple([output.mean(dim=1, keepdims=True) for output in self.train_forward(x)])
            else:
                return tuple([output[:, idx].mean(dim=1, keepdims=True) for output in self.train_forward(x)])

    def forward(self, x, idx=None):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x, idx)

# ce = CombinedEmbedding(input_dim=8, output_dim=32, t_input_dim=3)
# ce = CombinedEmbedding(input_dim=8, t_input_dim=3)
# ce(torch.rand(5,8)).shape

# ecm = EnsembleCombinedModel(input_dim=8, output_dim=1, t_input_dim=3, hidden_dim=64, n_output=5)
# ecm.train()
# ecm.eval()
# out = ecm(torch.rand(5,8))



# ★Multiple output Ensemble only last layer
class DirectEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_output=1, n_layers=3, n_models=10):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False, dropout=0)})

        out_dim = output_dim*n_models if n_output == 1 else output_dim*n_output*n_models

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False, dropout=0)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        
        self.output_dim = output_dim
        self.n_output = n_output
        self.n_layers = n_layers
        self.n_models = n_models

    # train step
    def train_forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        if self.n_output == 1:
            return x
        else:
            return torch.split(x, self.output_dim*self.n_models, dim=1)

    # eval step : ensemble mean
    def predict(self, x, idx=None):
        if self.n_output == 1:
            if idx is None:
                return self.train_forward(x).mean(dim=1, keepdims=True)
            else:
                return self.train_forward(x)[:, idx].mean(dim=1, keepdims=True)
        else:
            if idx is None:
                return tuple([output.mean(dim=1, keepdims=True) for output in self.train_forward(x)])
            else:
                return tuple([output[:, idx].mean(dim=1, keepdims=True) for output in self.train_forward(x)])

    def forward(self, x, idx=None):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x, idx)


# de = DirectEnsemble(input_dim=14, hidden_dim=32, output_dim=1, n_output=1, n_layers=3, n_models=10)
# de = DirectEnsemble(input_dim=14, hidden_dim=32, output_dim=1, n_output=2, n_layers=3, n_models=10)
# de.eval()
# de(torch.rand(8,14))


# -------------------------------------------------------------------------------------------
import torch
class RewardFunctionTorch():
    def __init__(self):
        self.sigma1 = 15 
        self.mu1 = -13
        self.mu2 = 0
        self.sigma2 = 4
    
    def true_f(self, x):
        gaussian1 = torch.exp(-1/(self.sigma1**2)*(x-self.mu1)**2)
        gaussian2 = torch.exp(-1/(self.sigma2**2)*(x-self.mu2)**2)
        return gaussian1 * (x < self.mu1) + 1 * (x >= self.mu1) * (x < self.mu2)+  gaussian2 * (x >= self.mu2)

    def forward(self, x):
        return self.true_f(x)
        # return self.true_f(x) + torch.randn_like(x)*0.01

    def __call__(self, x):
        return self.forward(x)

x = torch.linspace(-100,100, 100)
reward_f = RewardFunctionTorch()
plt.plot(x,reward_f(x), 'o-')
# plt.axvline(-37, color='red', alpha=0.3)
plt.axvline(0, color='red', alpha=0.3)
plt.axvline(-13, color='red', alpha=0.3)
# plt.axvline(23, color='red', alpha=0.3)
plt.show()




class ActionPenaltyTorch():
    def __init__(self):
        self.sigma = 33 
        self.mu = 0
    
    def true_f(self, x):
        gaussian = torch.exp(-1/(self.sigma**2)*(x-self.mu)**2)
        return 1-(gaussian * (x<self.mu) + 0 * (x>=self.mu))

    def forward(self, x):
        return self.true_f(x)
        # return self.true_f(x) + torch.randn_like(x)*0.01

    def __call__(self, x):
        return self.forward(x)

# x = torch.linspace(-200,50, 100)
# action_penalty_f = ActionPenalty()
# plt.plot(x, action_penalty_f(x), 'o-')
# # plt.axvline(-37, color='red', alpha=0.3)
# plt.axvline(0, color='red', alpha=0.3)
# plt.axvline(-13, color='red', alpha=0.3)
# # plt.axvline(23, color='red', alpha=0.3)
# plt.show()







#########################################################################################################
# Feature-Set Version 2-5

# feature_set_common : cur_time, target_time, cur_point, target_point
# feature_set_common = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
#                         'cur_time_hour_min',
#                         # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
#                         'target_time_Week', 'target_time_Sat', 'target_time_Sun',
#                         'target_time_hour_min',
#                         # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
#                         'cur_point_lat','cur_point_lng', 'target_point_lat', 'target_point_lng']

# # feature_set_LastAPI : call_time_LastAPI, call_point_LastAPI, path_time_LastAPI
# feature_set_LastAPI = ['call_time_LastAPI_Week', 'call_time_LastAPI_Sat', 'call_time_LastAPI_Sun',
#                         'call_time_LastAPI_hour_min',
#                         # 'call_time_LastAPI_Commute', 'call_time_LastAPI_Day', 'call_time_LastAPI_Night',
#                         'call_point_LastAPI_lat', 'call_point_LastAPI_lng', 'path_time_LastAPI'
#                         # , 'movedist_LastAPI', 'remain_req_leaving_time_LastAPI'
#                         ]

# # feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
# feature_set_TimeMachine = ['call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
#                         'call_time_TimeMachine_hour_min',
#                         # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
#                         'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine',
#                         #  'movedist_TimeMachine', 'remain_req_leaving_time_TimeMachine'
#                          ]


# -------------------------------------------------------------------------------------------
def make_feature_set_embedding(context_df, temporal_cols, spatial_cols, other_cols, fillna=None):
    # temproal features preprocessing
    temproal_arr = context_df[temporal_cols].applymap(lambda x: format_str_to_time(x) if type(x) == str else x).fillna(0).to_numpy().astype(np.float32)

    # spatial features preprocessing
    spatial_cols_transform = list(np.stack([[f"{cols}_x", f"{cols}_y"] for cols in spatial_cols]).ravel())

    spatial_arr_stack = np.stack(list(context_df[spatial_cols].applymap(lambda x: np.array(x)).to_dict('list').values())).astype(np.float32)
    spatial_arr = spatial_arr_stack.transpose(1,0,2).reshape(-1,4)

    # other features
    other_arr = context_df[other_cols].to_numpy().astype(np.float32)

    # # combine and transform to dataframe
    df_columns = temporal_cols + spatial_cols_transform + other_cols
    df_transform = pd.DataFrame(np.hstack([temproal_arr, spatial_arr, other_arr]),
                             columns=df_columns, index=context_df.index)
    if fillna is not None:
        df_transform = df_transform.fillna(fillna)
    return df_transform


# # temproal feature preprocessing
# temproal_arr = context_df[temporal_feature_cols].applymap(lambda x: format_str_to_time(x)).fillna(0).to_numpy().astype(np.float32)
# # temproal_tensor = torch.tensor(temproal_arr)

# # spatial feature preprocessing
# spatial_arr_stack = np.stack(list(context_df[spatial_feature_cols].applymap(lambda x: np.array(x)).to_dict('list').values())).astype(np.float32)
# spatial_arr = spatial_arr_stack.transpose(1,0,2).reshape(-1,4)
# # spatial_tensor = torch.tensor(spatial_arr)

# # other feature
# other_arr = context_df[continuous_feature_cols].fillna(0).to_numpy().astype(np.float32)

# # combine & transform to dataframe
# spatial_cols_transform = list(np.stack([[f"{cols}_x", f"{cols}_y"] for cols in spatial_feature_cols]).ravel())
# df_columns = temporal_feature_cols + spatial_cols_transform + other_feature_cols
# df_tranform = pd.DataFrame(np.hstack([temproal_arr, spatial_arr, other_arr]), columns=df_columns, index=context_df.index)


# (use)
# context_df_transform = make_feature_set_embedding(context_df, 
#             temporal_feature_cols, spatial_feature_cols, other_feature_cols)
# context_df_transform

# -------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------

from IPython.display import clear_output
import time
from tqdm.auto import tqdm
from six.moves import cPickle


# simulaiton -------------------------------------------------------------------------------------
simul = TemporalMapSimulation(temporal_graph)
time_interval = 3

# feature -------------------------------------------------------------------------------------
# temporal_feature_cols = ['cur_time','target_time', 'call_time_TimeMachine', 'call_time_LastAPI']
temporal_feature_cols = ['cur_time','target_time', 'call_time_TimeMachine', 'remain_time_from_TimeMachine']
spatial_feature_cols = ['cur_point', 'target_point']
other_feature_cols = ['path_time_TimeMachine']
other_feature_cols_pathtime = ['path_time_TimeMachine', 'path_time_LastAPI']

# model -------------------------------------------------------------------------------------
input_dim = len(temporal_feature_cols) + len(spatial_feature_cols)*2 + len(other_feature_cols)
input_dim_pathtime = len(temporal_feature_cols) + len(spatial_feature_cols)*2 + len(other_feature_cols_pathtime)
hidden_dim = 64

model = EnsembleCombinedModel(input_dim=input_dim, output_dim=1, t_input_dim=len(temporal_feature_cols),
                hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=2,
                embed_output_dim=64, t_hidden_dim=32, s_emb_dim=32, t_emb_dim=16,
                coord_embed_dim=16, coord_hidden_dim=32, coord_depth=2, grid_size=32, periodic_embed_dim=9)

# model = EnsembleCombinedModel(input_dim=input_dim, output_dim=1, t_input_dim=len(temporal_feature_cols),
#                 hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=2)
# model(torch.rand(5,8))
sum(p.numel() for p in model.parameters() if p.requires_grad)     # 61039

model_pathtime = EnsembleCombinedModel(input_dim=input_dim_pathtime, output_dim=1, t_input_dim=len(temporal_feature_cols),
                hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=1,
                embed_output_dim=64, t_hidden_dim=32, s_emb_dim=32, t_emb_dim=16,
                coord_embed_dim=16, coord_hidden_dim=32, coord_depth=2, grid_size=32, periodic_embed_dim=9)
# model_pathtime = EnsembleCombinedModel(input_dim=input_dim_pathtime, output_dim=1, t_input_dim=len(temporal_feature_cols),
#                 hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=1)
# model_pathtime(torch.rand(5,9))
sum(p.numel() for p in model_pathtime.parameters() if p.requires_grad)     # 59813


# compile -------------------------------------------------------------------------------------
alpha = 1

optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer_pathtime = optim.Adam(model_pathtime.parameters(), lr=1e-3)

reward_function = RewardFunctionTorch()
action_penalty_function = ActionPenaltyTorch()
action_penalty_lambda = 0.5
loss_gaussian = nn.GaussianNLLLoss()
loss_mse = nn.MSELoss()

history = []
# history = cPickle.load(open(f"{load_path}/240826_Alg2_2_DirectEnsemble_history.pkl", 'rb'))


# learn -------------------------------------------------------------------------------------
num_episodes = 1
for episode in tqdm(range(num_episodes)):
    history_t = {}
    # simul.reset_state(reset_all=False, save_interval=time_interval, verbose=0)
    simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)

    context = simul.run()

    repeat = (simul.target_time - 8 - simul.cur_time) // time_interval
    context_df =  pd.Series(context).to_frame().T
    cotexts_repeat_df = pd.concat([context_df]*repeat, ignore_index=True)
    cotexts_repeat_df['cur_time'] = time_interval
    cotexts_repeat_df['cur_time'][0] = simul.cur_time
    cotexts_repeat_df['cur_time'] = (cotexts_repeat_df['cur_time'].cumsum()).apply(format_time_to_str)
    cotexts_repeat_df['remain_time_from_TimeMachine'] = cotexts_repeat_df['req_leaving_time'].apply(format_str_to_time) - cotexts_repeat_df['cur_time'].apply(format_str_to_time)

    contexts_feature_df = make_feature_set_embedding(cotexts_repeat_df, 
                temporal_feature_cols, spatial_feature_cols, other_feature_cols)
    contexts_tensor = torch.tensor(contexts_feature_df.to_numpy()).to(device)
    # contexts_tensor.shape

    # calculate UCB & select action
    with torch.no_grad():
        model.eval()
        mu, logvar = model(contexts_tensor)
        logvar = torch.clamp(logvar, min=-10, max=10)   # clamp

        ucb = mu + alpha * torch.exp(0.5*logvar)
        ucb_mean = torch.mean(ucb, dim=1, keepdims=True)
        action = torch.argmax(ucb_mean).to('cpu').item()
    action_time = simul.cur_time + action*time_interval
    adjcent_matrix1 = simul.temporal_graph.transform(action_time)
    api_dict1 = simul.api_call(adjcent_matrix1, save_instance=False, save_history=False)

    # select contexts for selected action
    select_context_df = contexts_feature_df.loc[[action]]
    select_context_df['path_time_LastAPI'] = np.array(api_dict1['path_time']).astype(np.float32)
    select_context_tensor = torch.tensor(select_context_df.to_numpy()).to(device)
    select_context_tensor.shape

    # Train pathtime model ---------------------------------------------------------------------------
    model_pathtime.train()
    optimizer_pathtime.zero_grad()
    pred_change_path = model_pathtime(select_context_tensor)
    pred_change_path = torch.clamp(pred_change_path, min=-api_dict1['path_time'], max=1440)       # clamp
    
    simul.start_time
    simul.call_time_TimeMachine

    # (predicted time-step에서 api_call)
    with torch.no_grad():
        pred_pathtime = api_dict1['path_time'] + pred_change_path.mean().item()
        pred_leaving_time = int(np.ceil(simul.target_time - 8 - (pred_pathtime)))
        adjcent_matrix2 = simul.temporal_graph.transform(pred_leaving_time)
        api_dict2 = simul.api_call(adjcent_matrix2, save_instance=False, save_history=False)
        true_change_pathtime = torch.tensor(api_dict2['path_time'] - api_dict1['path_time'], dtype=torch.float32).view(-1,1).to(device)

    # (time-machine 기준 출발시간에서 api_call)
    # with torch.no_grad():
    #     pred_pathtime = api_dict1['path_time'] + pred_change_path.mean().item()
    #     adjcent_matrix2 = simul.temporal_graph.transform(simul.call_time_TimeMachine)
    #     api_dict2 = simul.api_call(adjcent_matrix2, save_instance=False, save_history=False)
    #     true_change_pathtime = torch.tensor(api_dict2['path_time'] - api_dict1['path_time'], dtype=torch.float32).view(-1,1).to(device)

    loss_pathtime = loss_mse(pred_change_path, true_change_pathtime)
    loss_pathtime.backward()
    optimizer_pathtime.step()


    # Train action select model ---------------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    mu, logvar = model(select_context_tensor[:,:-1])
    logvar = torch.clamp(logvar, min=-10, max=20)   # clamp
    std = torch.exp(0.5*logvar)
    # std = torch.clamp(std, min=0, max=10000)   # clamp
    with torch.no_grad():
        reward_x = (pred_leaving_time + api_dict2['path_time']) - simul.target_time
        reward = reward_function(torch.tensor(reward_x)).reshape(-1,1)
        action_penalty = action_penalty_function( torch.tensor(pred_leaving_time - action_time) ).reshape(-1,1)

        reward_adjust = reward - action_penalty_lambda * action_penalty
        logit_reward = ( 0.5* torch.log((1 + reward_adjust + 1e-10) / (1 - reward_adjust + 1e-10)) ).to(device)   # tanh logit
        # logit_reward = ( torch.log( reward / ( 1- reward + 1e-10)) ).to(device)   # logit

    loss = loss_gaussian(mu, logit_reward, std**2)
    loss.backward()
    optimizer.step()


    # ----------------------------------------------------------------------------------------------

    with torch.no_grad():
        # history
        history_t['loss_pathtime'] = loss_pathtime.item()
        history_t['loss'] = loss.item()
        history_t['context'] = select_context_df.to_numpy()
        history_t['action'] = action
        history_t['pred_leaving_time'] = pred_leaving_time
        history_t['pred_pathtime'] = pred_pathtime
        history_t['true_pathtime'] = api_dict2['path_time']
        history_t['pred_mu'] = torch.mean(mu).item()
        history_t['pred_std'] = torch.mean(std).item()
        history_t["reward_x"] = reward_x
        history_t["reward"] = reward.item()

        history_t['start_time'] = simul.start_time
        history_t['target_time'] = simul.target_time
        history_t['action_time'] = action_time
        history_t['time_machine_time'] = format_str_to_time(simul.history_time_machine[0]['req_leaving_time'])
        history_t['pathtime_TimeMachine'] = simul.history_time_machine[0]['path_time_TimeMachine']
        # summary
        summary_of_episodes = f"loss_pathime: {loss_pathtime.item():.2f}, loss_action: {loss.item():.2f}, reward_x: {reward_x:.1f}, reward: {reward.item():.1f}, action: {action}, pathtime: {history_t['pred_pathtime']:.1f} / {history_t['true_pathtime']:.1f}, mu/std: {history_t['pred_mu']:.2f}/{history_t['pred_std']:.2f}"
        print(f"\r  → {summary_of_episodes}")

        history_t["summary_of_episodes"] = summary_of_episodes
        # history_t['weights'] = model.state_dict()
        history.append(history_t)
        print()

        # save weights
        path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\Model'
        # path = "/content/drive/MyDrive/SNU_OhLAB/CLabProject"
        cPickle.dump(model.state_dict(), open(f"{path}/240903_Alg2_4_DirectEnsemble_weights.pkl", 'wb'))
        cPickle.dump(model_pathtime.state_dict(), open(f"{path}/240903_Alg2_4_DirectEnsemble_pathtime_weights.pkl", 'wb'))
        cPickle.dump(history, open(f"{path}/240903_Alg2_4_DirectEnsemble_history.pkl", 'wb'))





# # load_history --------------------------------------------------------------------------------------------------------------------------------------
from six.moves import cPickle
load_path = r'D:\DataScience\SNU_DataScience\SNU_OhLab\CLab_Project\Model'
# load_path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\Model'
history = cPickle.load(open(f"{load_path}/240903_Alg2_4_DirectEnsemble1_history.pkl", 'rb'))
# print('load_history')
len(history)




# visualize --------------------------------------------------------------------------------------------------------------------------------------
dist_y = [hist['loss_pathtime'] for hist in history]
dist_y = [hist['loss'] for hist in history]
dist_y = [np.sum(hist['action']) for hist in history]
dist_y = [hist['reward_x'] for hist in history]
dist_y = [hist['reward'] for hist in history]

dist_y = [ (hist['action_time'] - hist['start_time']) /(hist['target_time'] - hist['start_time']) for hist in history]
dist_y2 = [ (hist['target_time']-8-hist['pathtime_TimeMachine']- hist['start_time']) /(hist['target_time'] - hist['start_time']) for hist in history]
dist_y = [hist['action_time'] for hist in history]
dist_y = [hist['pred_pathtime'] - hist['true_pathtime'] for hist in history]

dist_y = [hist['pred_mu'] for hist in history]
dist_y = [hist['pred_std'] for hist in history]


# dist_y = dist_y[-1000:]
# dist_y2 = dist_y2[-1000:]

# # visualize
# plt.figure(figsize=(14,3))
# plt.plot(np.arange(len(dist_y)), dist_y, alpha=0.3)
# plt.scatter(np.arange(len(dist_y)), dist_y, alpha=0.7, edgecolor='white')

# plt.plot(np.arange(len(dist_y)), dist_y2, alpha=0.3)
# plt.scatter(np.arange(len(dist_y)), dist_y2, alpha=0.7, edgecolor='white')

# plt.ylim(0,10)
plt.yscale("symlog")
# plt.ylabel('y - 8 - y_hat')
plt.xlabel('Episodes')
plt.axhline(0, c='black', alpha=0.3)
plt.axhline(-13, c='red', alpha=0.3)
plt.show()


# moving average
window_size = 100

dist_y = np.array([hist['reward_x'] for hist in history])
dist_y = np.array([(hist['reward_x'] >= -13 and hist['reward_x'] < 0) for hist in history])
move_avg_y = np.convolve(dist_y, np.ones(window_size)/window_size, mode='valid')
# np.convolve(data, np.ones(window_size)/window_size, mode='valid')   # moving average

plt.figure(figsize=(14,3))
plt.plot(np.arange(len(move_avg_y)), move_avg_y, alpha=0.3)
# plt.scatter(np.arange(len(move_avg_y)), move_avg_y, alpha=0.7, edgecolor='white')
plt.xlabel('Episodes')
plt.axhline(1, c='black', alpha=0.3)
plt.axhline(0.95, c='red', alpha=0.3)
plt.axhline(0.90, c='coral', alpha=0.3)
plt.axhline(0.80, c='orange', alpha=0.3)
# plt.axhline(0, c='black', alpha=0.3)
# plt.axhline(-13, c='red', alpha=0.3)
plt.show()


# -----------------------------------------------------------------------------------------------------------------------------------------------








###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################








































































































# # # create_dataset 1 -------------------------------------------------------------------------------------
# simul = TemporalMapSimulation(temporal_graph)
# time_interval = 3
# safe_margin = 8

# contexts_history = []
# for _ in tqdm(range(100)):
#     # simul.reset_state(reset_all=False, save_interval=time_interval, verbose=0)
#     simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)
#     while(simul.start_time < 300):
#         simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)
#     true_api_info = simul.api_call(save_instance=False, save_history=False)

#     true_path_time = true_api_info['path_time']  #
#     new_target_time = int(np.round(simul.cur_time + true_path_time + safe_margin))  #
#     true_req_leaving_time = new_target_time - safe_margin - true_path_time  #
#     true_api_info['req_leaving_time'] = true_req_leaving_time
#     time_machine_info = simul.run_time_machine(target_time=new_target_time, save_instance=False)
#     time_machine_path_time = time_machine_info['path_time']     #
#     time_machine_req_leaving_time = new_target_time - safe_margin - time_machine_path_time  #

#     new_start_time, new_api_call_time = sorted(np.random.randint(np.max([0, new_target_time - 24*60]), int(np.floor(true_req_leaving_time)), size=2))
#     sample_adjcent_matrix = simul.temporal_graph.transform(new_api_call_time)
#     sample_api_info = simul.api_call(adjacent_matrix= sample_adjcent_matrix, save_instance=False, save_history=False)
#     sample_api_info['path_time']

#     context = simul.save_data('check', save_data=False, save_plot=False)
#     context['start_time'] = format_time_to_str(new_start_time)
#     context['target_time'] = format_time_to_str(new_target_time)
#     context['start_point'] = simul.start_point
#     context['target_point'] = simul.target_point
#     context['call_time_TimeMachine'] = format_time_to_str(int(time_machine_req_leaving_time))
#     context['path_TimeMachine'] = time_machine_info['path']
#     context['path_time_TimeMachine'] = time_machine_info['path_time']
#     context['cur_time'] = context['start_time']
#     context['call_time_LastAPI'] = format_time_to_str(new_api_call_time)
#     context['call_node_LastAPI'] = context['start_node']
#     context['call_point_LastAPI'] = context['start_point']
#     context['path_LastAPI'] = sample_api_info['path']
#     context['path_time_LastAPI'] = sample_api_info['path_time']
#     context['req_leaving_time'] = format_time_to_str(time_machine_req_leaving_time)
#     contexts_history.append( (context, true_api_info) )



# # # create_dataset 2 -------------------------------------------------------------------------------------
# simul = TemporalMapSimulation(temporal_graph)

# time_interval = 3
# safe_margin = 8

# contexts_history = []
# for _ in tqdm(range(100000)):
#     # simul.reset_state(reset_all=False, save_interval=time_interval, verbose=0)
#     simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)
#     context = simul.run()
#     time_machine_dict = simul.run_time_machine(save_instance=False)
#     api_call_time = int(time_machine_dict['req_leaving_time'])
#     adjcent_matrix_at_TimeMachine = simul.temporal_graph.transform(api_call_time)
#     true_api_info = simul.api_call(adjacent_matrix=adjcent_matrix_at_TimeMachine, save_instance=False, save_history=False)
#     true_api_info['call_time'] = api_call_time

#     # context['call_time_LastAPI'] = context['call_time_TimeMachine']
#     # context['call_node_LastAPI'] = context['start_node']
#     # context['call_point_LastAPI'] = context['start_point']
#     # context['path_LastAPI'] = true_api_info['path']
#     # context['path_time_LastAPI'] = true_api_info['path_time']
#     # context['req_leaving_time'] = format_time_to_str(int(np.ceil(true_api_info['path_time'])))
#     context['req_leaving_time'] = None
#     context['req_leaving_time_TimeMachine'] = format_time_to_str(int(np.ceil(time_machine_dict['req_leaving_time'])))

#     contexts_history.append( (context, true_api_info) )

# # # save_data
# from six.moves import cPickle
# path = r'D:\DataScience\SNU_DataScience\SNU_OhLab\CLab_Project\dataset'
# cPickle.dump(contexts_history, open(f"{path}/240906_context_TimeMachine_dataset_100000.pkl", 'wb'))

# del contexts_history

# # for i in range(10):
# #     contexts_ = contexts_history[i*10000:(i+1)*10000]
# #     cPickle.dump(contexts_, open(f"{path}/240829_context_dataset{i:02d}_rs_1.pkl", 'wb'))

# # print('save_dataset')

contexts_history = cPickle.load(open(f"{path}/240906_context_TimeMachine_dataset_10000.pkl", 'rb'))
# contexts_history = cPickle.load(open(f"{path}/240829_context_dataset01_rs_1.pkl", 'rb'))
# len(contexts_history)

contexts = [xy[0] for xy in contexts_history]
true_api = [xy[1] for xy in contexts_history]
# # pd.DataFrame(contexts[:1000])
# # pd.DataFrame(true_api[:1000])


# -------------------------------------------------------------------------------------










###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

###########################################################################################################
# Version 3-1


import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'

# response_files = requests.get("https://raw.githubusercontent.com/kimds929/CodeNote/main/42_Temporal_Spatial/DL13_Temporal_12_TemporalEmbedding.py")
# exec(response_files.text)
# response_files.text

with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/42_Temporal_Spatial/"):
    from DL13_Temporal_12_TemporalEmbedding import TemporalEmbedding
    from DL13_Spatial_11_SpatialEmbedding import SpatialEmbedding

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/47_Bayesian_Neural_Network/"):
    from BNN05_DensityEstimate_Regressor_Torch import FeedForwardBlock



# -------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------------------------------------------------------------------------
class CombinedEmbedding(nn.Module):
    def __init__(self, input_dim, t_input_dim,  output_dim=None, t_emb_dim=8, t_hidden_dim=None, s_emb_dim=None, **spatial_kwargs):
        super().__init__()

        # temporal embedding layer (t_input_dim)
        self.t_input_dim = t_input_dim
        self.temporal_embedding = TemporalEmbedding(input_dim=t_input_dim, embed_dim=t_emb_dim, hidden_dim=t_hidden_dim)

        # spatial embedding layer (4)
        self.spatial_embedding = SpatialEmbedding(embed_dim=s_emb_dim, **spatial_kwargs)
        
        # other feature dimension (input_dim - t_input_dim - 4)
        self.other_feature_dim = input_dim - t_input_dim - 4

        # embed_dim
        self.output_dim = output_dim
        self.embed_dim = self.temporal_embedding.embed_dim + self.spatial_embedding.embed_dim + self.other_feature_dim

        # fc block
        if output_dim is not None:
            self.fc_layer = nn.Linear(self.embed_dim, output_dim)
            self.embed_dim = output_dim
        
    def forward(self, x):
        temporal_features = self.temporal_embedding(x[:,:self.t_input_dim])
        spatial_features = self.spatial_embedding(x[:,self.t_input_dim:self.t_input_dim+2], x[:,self.t_input_dim+2:self.t_input_dim+4])
        other_features = x[:,self.t_input_dim+4:]
        outputs = torch.cat([temporal_features, spatial_features, other_features], dim=1)
        
        if self.output_dim is not None:
            outputs = self.fc_layer(outputs)
        return outputs


class EnsembleCombinedModel(nn.Module):
    def __init__(self, input_dim, output_dim, t_input_dim, hidden_dim,  n_layers=3, n_models = 10, n_output=1,
                embed_output_dim=None, t_emb_dim=8, t_hidden_dim=None, s_emb_dim=None, **spatial_kwargs):
        super().__init__()

        # combined embedding layer
        self.combined_embedding = CombinedEmbedding(input_dim=input_dim, t_input_dim=t_input_dim,  output_dim=embed_output_dim,
                                                    t_emb_dim=t_emb_dim, t_hidden_dim=t_hidden_dim, s_emb_dim=s_emb_dim, **spatial_kwargs)
        self.embed_output_dim = self.combined_embedding.embed_dim

        # fc block
        self.fc_block = nn.ModuleDict({'in_layer':FeedForwardBlock(self.embed_output_dim, hidden_dim, batchNorm=False, dropout=0)})

        out_dim = output_dim*n_models if n_output == 1 else output_dim*n_output*n_models
        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.fc_block[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False, dropout=0)
            else:
                self.fc_block['out_layer'] = FeedForwardBlock(hidden_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        
        self.output_dim = output_dim
        self.n_output = n_output
        self.n_layers = n_layers
        self.n_models = n_models

    def train_forward(self, x):
        x = self.combined_embedding(x)

        for layer_name, layer in self.fc_block.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection
        
        if self.n_output == 1:
            return x
        else:
            return torch.split(x, self.output_dim*self.n_models, dim=1)

    def predict(self, x, idx=None):
        if self.n_output == 1:
            if idx is None:
                return self.train_forward(x).mean(dim=1, keepdims=True)
            else:
                return self.train_forward(x)[:, idx].mean(dim=1, keepdims=True)
        else:
            if idx is None:
                return tuple([output.mean(dim=1, keepdims=True) for output in self.train_forward(x)])
            else:
                return tuple([output[:, idx].mean(dim=1, keepdims=True) for output in self.train_forward(x)])

    def forward(self, x, idx=None):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x, idx)

# ce = CombinedEmbedding(input_dim=8, output_dim=32, t_input_dim=3)
# ce = CombinedEmbedding(input_dim=8, t_input_dim=3)
# ce(torch.rand(5,8)).shape

# ecm = EnsembleCombinedModel(input_dim=8, output_dim=1, t_input_dim=3, hidden_dim=64, n_output=5)
# ecm.train()
# ecm.eval()
# out = ecm(torch.rand(5,8))



# ★Multiple output Ensemble only last layer
class DirectEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_output=1, n_layers=3, n_models=10):
        super().__init__()
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False, dropout=0)})

        out_dim = output_dim*n_models if n_output == 1 else output_dim*n_output*n_models

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False, dropout=0)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        
        self.output_dim = output_dim
        self.n_output = n_output
        self.n_layers = n_layers
        self.n_models = n_models

    # train step
    def train_forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        if self.n_output == 1:
            return x
        else:
            return torch.split(x, self.output_dim*self.n_models, dim=1)

    # eval step : ensemble mean
    def predict(self, x, idx=None):
        if self.n_output == 1:
            if idx is None:
                return self.train_forward(x).mean(dim=1, keepdims=True)
            else:
                return self.train_forward(x)[:, idx].mean(dim=1, keepdims=True)
        else:
            if idx is None:
                return tuple([output.mean(dim=1, keepdims=True) for output in self.train_forward(x)])
            else:
                return tuple([output[:, idx].mean(dim=1, keepdims=True) for output in self.train_forward(x)])

    def forward(self, x, idx=None):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x, idx)


# de = DirectEnsemble(input_dim=14, hidden_dim=32, output_dim=1, n_output=1, n_layers=3, n_models=10)
# de = DirectEnsemble(input_dim=14, hidden_dim=32, output_dim=1, n_output=2, n_layers=3, n_models=10)
# de.eval()
# de(torch.rand(8,14))





#########################################################################################################
# Feature-Set Version 3-1

# feature_set_common : cur_time, target_time, cur_point, target_point
# feature_set_common = ['cur_time_Week', 'cur_time_Sat', 'cur_time_Sun',
#                         'cur_time_hour_min',
#                         # 'cur_time_Commute', 'cur_time_Day', 'cur_time_Night',
#                         'target_time_Week', 'target_time_Sat', 'target_time_Sun',
#                         'target_time_hour_min',
#                         # 'target_time_Commute', 'target_time_Day', 'target_time_Night',
#                         'cur_point_lat','cur_point_lng', 'target_point_lat', 'target_point_lng']

# # feature_set_LastAPI : call_time_LastAPI, call_point_LastAPI, path_time_LastAPI
# feature_set_LastAPI = ['call_time_LastAPI_Week', 'call_time_LastAPI_Sat', 'call_time_LastAPI_Sun',
#                         'call_time_LastAPI_hour_min',
#                         # 'call_time_LastAPI_Commute', 'call_time_LastAPI_Day', 'call_time_LastAPI_Night',
#                         'call_point_LastAPI_lat', 'call_point_LastAPI_lng', 'path_time_LastAPI'
#                         # , 'movedist_LastAPI', 'remain_req_leaving_time_LastAPI'
#                         ]

# # feature_set_TimeMachine : call_time_TimeMachine, call_point_TimeMachine, path_time_TimeMachine
# feature_set_TimeMachine = ['call_time_TimeMachine_Week', 'call_time_TimeMachine_Sat', 'call_time_TimeMachine_Sun',
#                         'call_time_TimeMachine_hour_min',
#                         # 'call_time_TimeMachine_Commute', 'call_time_TimeMachine_Day', 'call_time_TimeMachine_Night',
#                         'call_point_TimeMachine_lat', 'call_point_TimeMachine_lng', 'path_time_TimeMachine',
#                         #  'movedist_TimeMachine', 'remain_req_leaving_time_TimeMachine'
#                          ]


# -------------------------------------------------------------------------------------------
def make_feature_set_embedding(context_df, temporal_cols, spatial_cols, other_cols, fillna=None):
    # temproal features preprocessing
    temproal_arr = context_df[temporal_cols].applymap(lambda x: format_str_to_time(x) if type(x) == str else x).fillna(0).to_numpy().astype(np.float32)

    # spatial features preprocessing
    spatial_cols_transform = list(np.stack([[f"{cols}_x", f"{cols}_y"] for cols in spatial_cols]).ravel())

    spatial_arr_stack = np.stack(list(context_df[spatial_cols].applymap(lambda x: np.array(x)).to_dict('list').values())).astype(np.float32)
    spatial_arr = spatial_arr_stack.transpose(1,0,2).reshape(-1,4)

    # other features
    other_arr = context_df[other_cols].to_numpy().astype(np.float32)

    # # combine and transform to dataframe
    df_columns = temporal_cols + spatial_cols_transform + other_cols
    df_transform = pd.DataFrame(np.hstack([temproal_arr, spatial_arr, other_arr]),
                             columns=df_columns, index=context_df.index)
    if fillna is not None:
        df_transform = df_transform.fillna(fillna)
    return df_transform


# # temproal feature preprocessing
# temproal_arr = context_df[temporal_feature_cols].applymap(lambda x: format_str_to_time(x)).fillna(0).to_numpy().astype(np.float32)
# # temproal_tensor = torch.tensor(temproal_arr)

# # spatial feature preprocessing
# spatial_arr_stack = np.stack(list(context_df[spatial_feature_cols].applymap(lambda x: np.array(x)).to_dict('list').values())).astype(np.float32)
# spatial_arr = spatial_arr_stack.transpose(1,0,2).reshape(-1,4)
# # spatial_tensor = torch.tensor(spatial_arr)

# # other feature
# other_arr = context_df[continuous_feature_cols].fillna(0).to_numpy().astype(np.float32)

# # combine & transform to dataframe
# spatial_cols_transform = list(np.stack([[f"{cols}_x", f"{cols}_y"] for cols in spatial_feature_cols]).ravel())
# df_columns = temporal_feature_cols + spatial_cols_transform + other_feature_cols
# df_tranform = pd.DataFrame(np.hstack([temproal_arr, spatial_arr, other_arr]), columns=df_columns, index=context_df.index)


# (use)
# context_df_transform = make_feature_set_embedding(context_df, 
#             temporal_feature_cols, spatial_feature_cols, other_feature_cols)
# context_df_transform

# -------------------------------------------------------------------------------------------







# -------------------------------------------------------------------------------------------

from IPython.display import clear_output
import time
from tqdm.auto import tqdm
from six.moves import cPickle




# feature -------------------------------------------------------------------------------------
# temporal_feature_cols = ['cur_time','target_time', 'call_time_TimeMachine', 'call_time_LastAPI']
# temporal_feature_cols = ['target_time', 'call_time_TimeMachine']
temporal_feature_cols = ['target_time', 'req_leaving_time_TimeMachine']
spatial_feature_cols = ['start_point', 'target_point']
other_feature_cols = ['path_time_TimeMachine']
other_feature_cols_pathtime = ['path_time_TimeMachine', 'path_time_LastAPI']

# model -------------------------------------------------------------------------------------
input_dim = len(temporal_feature_cols) + len(spatial_feature_cols)*2 + len(other_feature_cols)
input_dim_pathtime = len(temporal_feature_cols) + len(spatial_feature_cols)*2 + len(other_feature_cols_pathtime)
hidden_dim = 128

model = EnsembleCombinedModel(input_dim=input_dim, output_dim=1, t_input_dim=len(temporal_feature_cols),
                hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=1,
                embed_output_dim=64, t_hidden_dim=32, s_emb_dim=32, t_emb_dim=16,
                coord_embed_dim=16, coord_hidden_dim=32, coord_depth=2, grid_size=32, periodic_embed_dim=9)

# model = EnsembleCombinedModel(input_dim=input_dim, output_dim=1, t_input_dim=len(temporal_feature_cols),
#                 hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=2)
# model(torch.rand(5,8))
sum(p.numel() for p in model.parameters() if p.requires_grad)     # 61039


# # -------------------------------------------------------------------------------------------
# # dataset
# contexts_sample = contexts[:10000]
# true_api_sample = true_api[:10000]

# contexts_df = pd.DataFrame(contexts_sample)
# true_api_df = pd.DataFrame(true_api_sample)

# # make X
# contexts_feature_df = make_feature_set_embedding( contexts_df, 
#                         temporal_feature_cols, spatial_feature_cols, other_feature_cols)
# train_x = torch.tensor(contexts_feature_df.to_numpy()).to(device)

# # make Y
# path_time_diff = (contexts_feature_df['path_time_TimeMachine'] - true_api_df['path_time']).to_numpy().astype(np.float32).reshape(-1,1)
# train_y = torch.tensor(path_time_diff)

# # Dataset and DataLoader
# batch_size = 64

# from torch.utils.data import DataLoader, TensorDataset
# train_dataset = TensorDataset(train_x, train_y)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # len(train_dataset)
# # model(next(iter(train_loader))[0])





# (Training API Call time-step) ------------------------------------------------------------

# (Offline Leaning) -----------------------------------------------------------------------------------------------
# loss_gaussian = nn.GaussianNLLLoss()
def std_gaussian_loss(model, x, y):
    logvar = model(x)
    # logvar = torch.clamp(logvar, min=-20, max=20)
    std = torch.exp(0.5*logvar.mean(dim=1, keepdims=True))
    mu = torch.zeros_like(std)
    loss = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
    return loss

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

tm = TorchModeling(model=model, device=device)
tm.compile(optimizer=optimizer
            ,loss_function = std_gaussian_loss
            , scheduler=scheduler
            , early_stop_loss = EarlyStopping(patience=5)
            )
tm.train_model(train_loader=train_loader, epochs=90,
             tqdm_display=False, display_earlystop_result=True, early_stop=False, save_parameters=True)
# tm.recompile(optimizer = optim.Adam(model.parameters(), lr=1e-5))
# tm.test_model(test_loader=test_loader)
# ------------------------------

# num_epochs = 10
# # Training loop
# for epoch in tqdm(range(num_epochs)):
#     model.train()
#     for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.to(device)

#         optimizer.zero_grad()
        
#         # Forward pass ----------------------------------------
#         logvar = model(batch_x)
#         # logvar = torch.clamp(logvar, min=-20, max=20)
#         std = torch.exp(0.5*logvar.mean(dim=1, keepdims=True))
#         mu = torch.zeros_like(std)
#         loss = torch.nn.functional.gaussian_nll_loss(mu, batch_y, std**2)
#         loss.backward()
#         optimizer.step()
   
#     with torch.no_grad():
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, logvar: {torch.mean(logvar).item():.2f}, std: {torch.mean(std).item():.2f}')
# # -----------------------------------------------------------------------------------------------



# (Online Learning) -------------------------------------------------------------------------------------
simul = TemporalMapSimulation(temporal_graph)
time_interval = 3

losses = []

num_epochs = 1000
for epoch in range(num_epochs):
    simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)

    context = simul.run()
    context_df = pd.DataFrame([context])
    context_df['req_leaving_time_TimeMachine'] = context_df['target_time'].apply(format_str_to_time) - 8 - context_df['path_time_TimeMachine']
    contexts_feature_df = make_feature_set_embedding(context_df, 
                            temporal_feature_cols, spatial_feature_cols, other_feature_cols)
    contexts_feature_tensor = torch.tensor(contexts_feature_df.to_numpy()).to(device)

    req_leaving_time_TimaMachine = int(np.ceil(simul.target_time - 8 - simul.path_time_TimeMachine))
    adjcent_matrix2 = simul.temporal_graph.transform(req_leaving_time_TimaMachine)
    api_dict2 = simul.api_call(adjcent_matrix2, save_instance=False, save_history=False)

    pathtime_diff = np.array(simul.path_time_TimeMachine - api_dict2['path_time']).astype(np.float32).reshape(-1,1)
    pathtime_diff_tensor = torch.tensor(pathtime_diff)

    optimizer.zero_grad()
    logvar = model(contexts_feature_tensor)
    # logvar = torch.clamp(logvar, min=-20, max=20)
    std = torch.exp(0.5*logvar.mean(dim=1, keepdims=True))
    mu = torch.zeros_like(std)
    loss = torch.nn.functional.gaussian_nll_loss(mu, pathtime_diff_tensor, std**2)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        losses.append( loss.item() )
        print(f'\rEpoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, logvar: {torch.mean(logvar).item():.2f}, std: {torch.mean(std).item():.2f}')

plt.figure()
plt.plot(range(len(losses)), losses, alpha=0.3)
plt.scatter(range(len(losses)), losses, alpha=0.5, edgecolor='white')
# plt.yscale('symlog')
# plt.ylim(0,10)
plt.show()
# -----------------------------------------------------------------------------------------------









#########################################################################################################
# simulaiton -------------------------------------------------------------------------------------
simul = TemporalMapSimulation(temporal_graph)
time_interval = 3

model_pathtime = EnsembleCombinedModel(input_dim=input_dim_pathtime, output_dim=1, t_input_dim=len(temporal_feature_cols),
                hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=1,
                embed_output_dim=64, t_hidden_dim=32, s_emb_dim=32, t_emb_dim=16,
                coord_embed_dim=16, coord_hidden_dim=32, coord_depth=2, grid_size=32, periodic_embed_dim=9)
# model_pathtime = EnsembleCombinedModel(input_dim=input_dim_pathtime, output_dim=1, t_input_dim=len(temporal_feature_cols),
#                 hidden_dim=hidden_dim,  n_layers=3, n_models=10, n_output=1)
# model_pathtime(torch.rand(5,9))
sum(p.numel() for p in model_pathtime.parameters() if p.requires_grad)     # 59813
optimizer_pathtime = optim.Adam(model_pathtime.parameters(), lr=1e-3)

loss_mse = nn.MSELoss()

safe_margin_sigma = 3

history = []
# history = cPickle.load(open(f"{load_path}/240826_Alg2_2_DirectEnsemble_history.pkl", 'rb'))


# learn -------------------------------------------------------------------------------------
num_episodes = 8000
for episode in tqdm(range(num_episodes)):
    history_t = {}

    simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)

    context = simul.run()
    context_df = pd.DataFrame([context])
    context_df['req_leaving_time_TimeMachine'] = context_df['target_time'].apply(format_str_to_time) - 8 - context_df['path_time_TimeMachine']
    contexts_feature_df = make_feature_set_embedding(context_df, 
                            temporal_feature_cols, spatial_feature_cols, other_feature_cols)
    contexts_feature_tensor = torch.tensor(contexts_feature_df.to_numpy()).to(device)

    # 【 Predict action(API Call time) 】 ---------------------------------------------------------------------------
    
    # 【 Train action model 】 ---------------------------------------------------------------------------
    req_leaving_time_TimaMachine = int(np.ceil(simul.target_time - 8 - simul.path_time_TimeMachine))
    adjcent_matrix2 = simul.temporal_graph.transform(req_leaving_time_TimaMachine)
    api_dict2 = simul.api_call(adjcent_matrix2, save_instance=False, save_history=False)

    pathtime_diff = np.array(simul.path_time_TimeMachine - api_dict2['path_time']).astype(np.float32).reshape(-1,1)
    pathtime_diff_tensor = torch.tensor(pathtime_diff)

    # training ----
    optimizer.zero_grad()
    logvar = model(contexts_feature_tensor)
    # logvar = torch.clamp(logvar, min=-20, max=20)
    std = torch.exp(0.5*logvar.mean(dim=1, keepdims=True))
    mu = torch.zeros_like(std)
    loss = torch.nn.functional.gaussian_nll_loss(mu, pathtime_diff_tensor, std**2)
    loss.backward()
    optimizer.step()

    # # No training ----
    # with torch.no_grad():
    #     logvar = model(contexts_feature_tensor)
    #     std = torch.exp(0.5*logvar).mean().item()

    # std.item()
    # -------------------------------------------------------------------------------------------------
    action_time = int(contexts_feature_df['req_leaving_time_TimeMachine'] - safe_margin_sigma * std.item())
    if action_time < simul.start_time:
        action_time = simul.start_time

    adjcent_matrix_pred_action = simul.temporal_graph.transform(action_time)
    api_info_pred_action = simul.api_call(adjacent_matrix=adjcent_matrix_pred_action, save_instance=False, save_history=False)
    context_df['path_time_LastAPI'] = float(api_info_pred_action['path_time'])

    contexts_pathtime_feature_df = make_feature_set_embedding(context_df, 
                            temporal_feature_cols, spatial_feature_cols, other_feature_cols_pathtime)
    contexts_pathtime_feature_tensor = torch.tensor(contexts_pathtime_feature_df.to_numpy()).to(device)


    # 【 Train pathtime model 】 ---------------------------------------------------------------------------
    model_pathtime.train()
    optimizer_pathtime.zero_grad()
    pred_change_path = model_pathtime(contexts_pathtime_feature_tensor)
    pred_change_path = torch.clamp(pred_change_path, min=-api_info_pred_action['path_time'], max=1440)       # clamp

    # (predicted time-step에서 api_call)
    with torch.no_grad():
        pred_pathtime = api_info_pred_action['path_time'] + pred_change_path.mean().item()
        pred_leaving_time = int(np.ceil(simul.target_time - 8 - (pred_pathtime)))
        adjcent_matrix2 = simul.temporal_graph.transform(pred_leaving_time)
        api_dict2 = simul.api_call(adjcent_matrix2, save_instance=False, save_history=False)
        true_change_pathtime = torch.tensor(api_dict2['path_time'] - api_info_pred_action['path_time'], dtype=torch.float32).view(-1,1).to(device)

    # # (time-machine 기준 출발시간에서 api_call)
    # with torch.no_grad():
    #     pred_pathtime = api_info_pred_action['path_time'] + pred_change_path.mean().item()
    #     req_leaving_time_TimaMachine = int(np.ceil(simul.target_time - 8 - simul.path_time_TimeMachine))
    #     adjcent_matrix2 = simul.temporal_graph.transform(req_leaving_time_TimaMachine)
    #     api_dict2 = simul.api_call(adjcent_matrix2, save_instance=False, save_history=False)
    #     true_change_pathtime = torch.tensor(api_dict2['path_time'] - api_info_pred_action['path_time'], dtype=torch.float32).view(-1,1).to(device)

    loss_pathtime = loss_mse(pred_change_path.mean(dim=1), true_change_pathtime)
    loss_pathtime.backward()
    optimizer_pathtime.step()
    # --------------------------------------------------------------------------------------------------
    
    reward_x = api_dict2['path_time'] - pred_pathtime 

    with torch.no_grad():
        # history
        history_t['loss_pathtime'] = loss_pathtime.item()
        history_t['loss'] = loss.item()
        history_t['context'] = contexts_pathtime_feature_df.to_numpy()
        history_t['pred_leaving_time'] = pred_leaving_time
        history_t['pred_pathtime'] = pred_pathtime
        history_t['true_pathtime'] = api_dict2['path_time']
        history_t['pred_std'] = torch.mean(std).item()
        history_t["reward_x"] = reward_x

        history_t['start_time'] = simul.start_time
        history_t['target_time'] = simul.target_time
        history_t['action_time'] = action_time
        history_t['req_leaving_time_TimeMachine'] = contexts_feature_df['req_leaving_time_TimeMachine'].item()
        history_t['pathtime_TimeMachine'] = contexts_feature_df['path_time_TimeMachine'].item()
        
        # summary
        summary_of_episodes = f"loss_pathime: {loss_pathtime.item():.2f}, loss_action: {loss.item():.2f}, reward_x: {reward_x:.1f}, pathtime: {history_t['pred_pathtime']:.1f} / {history_t['true_pathtime']:.1f}, std: {history_t['pred_std']:.2f}"
        print(f"\r  → {summary_of_episodes}")

        history_t["summary_of_episodes"] = summary_of_episodes
        # history_t['weights'] = model.state_dict()
        history.append(history_t)
        print()

        # save weights
        path = r'D:\DataScience\SNU_DataScience\SNU_OhLab\CLab_Project\Model'
        # path = "/content/drive/MyDrive/SNU_OhLAB/CLabProject"
        cPickle.dump(model.state_dict(), open(f"{path}/240906_Alg3_1_DirectEnsemble_weights.pkl", 'wb'))
        cPickle.dump(model_pathtime.state_dict(), open(f"{path}/240906_Alg3_1_DirectEnsemble_pathtime_weights.pkl", 'wb'))
        cPickle.dump(history, open(f"{path}/240906_Alg3_1_DirectEnsemble_history.pkl", 'wb'))

    #########################################################################################################




# # load_history --------------------------------------------------------------------------------------------------------------------------------------
from six.moves import cPickle
# load_path = r'D:\DataScience\SNU_DataScience\SNU_OhLab\CLab_Project\Model'
# load_path = r'D:\DataScience\SNU_DataScience\OhLAB\CLabProject\Model'
# history = cPickle.load(open(f"{load_path}/240903_Alg2_4_DirectEnsemble1_history.pkl", 'rb'))
# print('load_history')
len(history)




# visualize --------------------------------------------------------------------------------------------------------------------------------------
dist_y = [hist['loss_pathtime'] for hist in history]
dist_y = [hist['loss'] for hist in history]
dist_y = [np.sum(hist['action']) for hist in history]
dist_y = [hist['reward_x']-8 for hist in history]

dist_y = [ (hist['action_time'] - hist['start_time']) /(hist['target_time'] - hist['start_time']) for hist in history]
dist_y2 = [ (hist['target_time']-8-hist['pathtime_TimeMachine']- hist['start_time']) /(hist['target_time'] - hist['start_time']) for hist in history]
dist_y = [hist['action_time'] for hist in history]
dist_y = [hist['pred_pathtime'] - hist['true_pathtime'] for hist in history]

dist_y = [hist['pred_mu'] for hist in history]
dist_y = [hist['pred_std'] for hist in history]


# dist_y = dist_y[-1000:]
# dist_y2 = dist_y2[-1000:]

# # visualize
plt.figure(figsize=(14,3))
plt.plot(np.arange(len(dist_y)), dist_y, alpha=0.3)
plt.scatter(np.arange(len(dist_y)), dist_y, alpha=0.7, edgecolor='white')

# plt.plot(np.arange(len(dist_y)), dist_y2, alpha=0.3)
# plt.scatter(np.arange(len(dist_y)), dist_y2, alpha=0.7, edgecolor='white')

# plt.ylim(0,10)
# plt.yscale("symlog")
# plt.ylabel('y - 8 - y_hat')
plt.xlabel('Episodes')
plt.axhline(0, c='black', alpha=0.3)
plt.axhline(-13, c='red', alpha=0.3)
plt.show()


# moving average -------------------------------------------------------------------------------------------
window_size = 100


dist_y = np.array([np.sqrt(hist['loss_pathtime']) for hist in history])
dist_y = np.array([hist['loss'] for hist in history])
# dist_y = np.array([hist['reward_x'] for hist in history])
dist_y = np.array([(hist['reward_x'] >= -5 and hist['reward_x'] < 8) for hist in history])


move_avg_y = np.convolve(dist_y, np.ones(window_size)/window_size, mode='valid')
# np.convolve(data, np.ones(window_size)/window_size, mode='valid')   # moving average

plt.figure(figsize=(14,3))
plt.plot(np.arange(len(move_avg_y)), move_avg_y, alpha=0.3)
# plt.scatter(np.arange(len(move_avg_y)), move_avg_y, alpha=0.7, edgecolor='white')
plt.xlabel('Episodes')
plt.axhline(1, c='black', alpha=0.3)
plt.axhline(0.95, c='red', alpha=0.3)
plt.axhline(0.90, c='coral', alpha=0.3)
plt.axhline(0.80, c='orange', alpha=0.3)
# plt.axhline(0, c='black', alpha=0.3)
# plt.axhline(-13, c='red', alpha=0.3)
plt.show()



##############################################################################################