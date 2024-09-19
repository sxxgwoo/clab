import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from IPython.display import clear_output
import time
import warnings
from tqdm.auto import tqdm
from scipy.stats import multivariate_normal
from IPython.display import clear_output
import heapq
from datetime import datetime
from scipy.optimize import fsolve


warnings.filterwarnings('ignore')




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
    
    def apply_function_to_matrix(self, matrix, func, *args, **kwargs):
        vectorized_func = np.vectorize(lambda x: func(x, *args, **kwargs))
        return vectorized_func(matrix)

    def matrix_rank(self, matrix, ascending=True, axis=0):
        """
        ascending: True (low 1 -> high ...), False (high 1 -> low ...)
        axis: 0 (row-wise), 1 (column-wise)
        """
        if ascending:
            return (np.argsort(np.argsort(matrix, axis=axis), axis=axis) + 1)
        else:
            return (np.argsort(np.argsort(-matrix, axis=axis), axis=axis) + 1)

    # (Version 3.3)
    def verify_closeness(self, adj_dist_mat, n_nodes, criteria_scale=0.25):
        adj_dist_mat_copy = adj_dist_mat.copy()
        np.fill_diagonal(adj_dist_mat_copy, np.inf)
        d = np.sqrt(2) / np.sqrt(n_nodes)

        near_points = (adj_dist_mat_copy < d*criteria_scale).sum(1).astype(bool)
        # print(near_points.sum())
        return near_points

    # (Version 3.3)
    def create_node(self, node_scale, cov_knn=3):
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
        adj_dist_mat_rank = self.matrix_rank(adj_dist_mat_copy, axis=1)

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

                adj_sym_rank_rev_mat = self.apply_function_to_matrix(adj_sym_rank_mat, connect_function, connect_scale=connect_scale)

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

# (Create Periodic Setting) #####################################################

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
    
    def circle_xy(self, center=[0,0], radius=1, n_points=100):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x_points = radius * np.cos(angles) + center[0]
        y_points = radius * np.sin(angles) + center[1]
        return np.stack((x_points, y_points)).T


    def make_tmeporal_list(self, mean_center=None, radius_scale=1, n_points=None):
        mean_center = self.mean_center if mean_center is None else mean_center
        n_points = self.n_points if n_points is None else n_points

        mean_r_f = self.default_f if self.mean_r_f is None else self.mean_r_f
        cov_f = self.default_f if self.cov_f is None else self.cov_f

        mean_r_X = mean_r_f(self.T_arr) * radius_scale
        covs_scale_X = cov_f(self.T_arr)

        for ti, r, c in zip(self.T_arr, mean_r_X, covs_scale_X):
            sg = SyntheticGaussian()
            circle_mean = self.circle_xy(center=mean_center, radius=r, n_points=self.n_points)
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

# Shortest Path


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
                _, target_node = self.temporal_graph.node_map.predict(target_point)
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

    def run_time_machine(self, temporal_graph=None, start_node=None, target_node=None, target_time=None, target_time_safe_margin=8, start_add_time=None, target_add_time=None, save_instance=True):
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
            target_time = target_time - target_time_safe_margin
            pre_adj_TimeMachine = temporal_graph.transform_oracle(target_time)
            pre_dijkstra_TimeMachine = Dijkstra(pre_adj_TimeMachine)
            pre_shortest_time_TimeMachine, pre_path_TimeMachine = pre_dijkstra_TimeMachine.dijkstra(start_node, target_node)
            pre_total_time_TimeMachine = pre_shortest_time_TimeMachine + start_add_time + target_add_time

            # second predict from adjusted predict leaving time
            call_time_TimeMachine = int(np.round(target_time - pre_total_time_TimeMachine))
            adj_TimeMachine = temporal_graph.transform_oracle(call_time_TimeMachine)
            dijkstra_TimeMachine = Dijkstra(adj_TimeMachine)
            shortest_time_TimeMachine, path_TimeMachine = dijkstra_TimeMachine.dijkstra(start_node, target_node)

            path_time_TimeMachine = shortest_time_TimeMachine + start_add_time + target_add_time
            req_leaving_time_TimeMachine = target_time - path_time_TimeMachine

            # result data
            if save_instance:
                self.call_time_TimeMachine = call_time_TimeMachine
                self.path_TimeMachine = path_TimeMachine
                self.path_time_TimeMachine = path_time_TimeMachine
                self.req_leaving_time_TimeMachine = req_leaving_time_TimeMachine

                if self.req_leaving_time is None:
                    self.req_leaving_time = req_leaving_time_TimeMachine


            return {"call_time":call_time_TimeMachine,
                    "target_time":target_time,
                    "path":path_TimeMachine,
                    "path_time":path_time_TimeMachine,
                    "req_leaving_time":req_leaving_time_TimeMachine}
        else:
            print("There is no one of needed arguments (start_node, target_node, target_time, start_add_time, target_add_time)")

    # (Version 4.1 Update)
    def api_call(self, adjacent_matrix=None, cur_node=None, target_node=None, agent=None,
                start_add_time=None, target_add_time=None, save_instance=True, save_history=True, api_time=None):
        api_time= self.cur_time if api_time is None else api_time
        adjacent_matrix = self.temporal_graph.transform(api_time) if adjacent_matrix is None else adjacent_matrix
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

        call_time_LastAPI = api_time
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



