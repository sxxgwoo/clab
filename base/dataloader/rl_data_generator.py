from simul_env import *
import numpy as np
import pandas as pd


############################################################################################################
# Simulation with LinUCB ###################################################################################
# (Initial Setting & Parameter Setting) -----------------------------------------------------------------------------------------------

random_state = 1
rng = np.random.RandomState(random_state)      # (RandomState) ★ Set_Params
n_nodes = 50                        # (Node수) ★ Set_Params

total_period = 7*24*60              # (전체 Horizon) ★ Set_Params
# T = 24*60                           # (주기) ★ Set_Params

# (create base graph) 
node_map = GenerateNodeMap(n_nodes, random_state=random_state)
node_map.create_node(node_scale=n_nodes, cov_knn=3)           # create node
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

#direct로연결된 노드 다시 확인후 봐야함 ex. s1 - 47로 직접 연결된경우임
# idx = [8, 25] #100
idx = [4,6] #50
# idx = [33, 38]

y_pred = temporal_graph.transform(x)[:, idx[0], idx[1]]
y_true = temporal_graph.transform_oracle(x)[:, idx[0], idx[1]]
visualize_periodic(y_pred, y_true, return_plot=False)


# ###################################################################################
# simul = TemporalMapSimulation(temporal_graph, random_state)
# # simul.reset_state(reset_all=False, save_interval=3, verbose=1) #save_interval: 분단위 쪼개기 verbose: 정보 보이기/
# simul.reset_state(reset_all=True, save_interval=3, verbose=1)

# simul.t #라운드
# simul.cur_time #시간 숫자버전
# format_time_to_str(simul.cur_time) #실제시간
# # format_str_to_time(format_time_to_str(simul.cur_time))
# # format_str_to_split(format_time_to_str(simul.cur_time))
# simul.step() #한 step씩 보고 싶을때

# context = simul.run()
# context
# df=pd.Series(context)
# # format_str_to_time(df['target_time'])
# simul.target_time #target 시간


# api_context = simul.api_call()

# #임의의 apicall을 한순간을 출발예정시간의 optimal로 생각할경우 target time 수정
# refined_target_time = api_context['call_time']+ api_context['path_time']
# refined_target_time #optimal로 부터 나온 target time


# for t in range(simul.target_time - simul.start_time):
#     simul.run()
    
# simul.loop
# while(simul.loop):
#     simul.run()

#     if simul.t %30 == 0:
#         simul.api_call()
#         #simul.api_call(save_instance=False, save_history=False) 기록하고 싶지 않을때
#         #simul.api_call? 정보 궁금하면
    
# #바로 딕셔너리로 feature받고싶으면
# # simul.save_data('', save_data=False)


# pd.DataFrame(simul.history_time_machine).to_clipboard() #처음 알람등록시 타임머신 
# pd.DataFrame(simul.history).to_clipboard() #내가 api 친 모든 정보
# pd.DataFrame(simul.history_full_info).to_clipboard() #전체정보를 다 알숟있는 api를 다 쳐서 오라클알수있음