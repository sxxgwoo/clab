from simul_env import *
import numpy as np
import pandas as pd
import math
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(sys.path)
############################################################################################################
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

# visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)



# # (temporal periodic plot)
# x = np.arange(0, 8*24*60)

# #direct로연결된 노드 다시 확인후 봐야함 ex. s1 - 47로 직접 연결된경우임
# # idx = [8, 25] #100
# idx = [4,6] #50
# # idx = [33, 38]

# y_pred = temporal_graph.transform(x)[:, idx[0], idx[1]]
# y_true = temporal_graph.transform_oracle(x)[:, idx[0], idx[1]]
# visualize_periodic(y_pred, y_true, return_plot=False)


# ###################################################################################
simul = TemporalMapSimulation(temporal_graph)
# simul.reset_state(reset_all=False, save_interval=3, verbose=1) #save_interval: 분단위 쪼개기 verbose: 정보 보이기/
time_interval=1
episode =0
df_final = pd.DataFrame()
# test={'refinedTimemachine': [], 'oracle': [], 'TimeMachineResult':[],'target_8':[]}
for i in range(20):
    simul.reset_state(reset_all=True, save_interval=1, verbose=1)

    # simul.t #라운드
    # simul.cur_time #시간 숫자버전
    # format_time_to_str(simul.cur_time) #실제시간
    # # format_str_to_time(format_time_to_str(simul.cur_time))
    # # format_str_to_split(format_time_to_str(simul.cur_time))
    # simul.step() #한 step씩 보고 싶을때

    context = simul.run()
    # context
    # format_str_to_time(context['target_time'])
    # format_str_to_time(context['req_leaving_time'])

    # df=pd.Series(context).to_frame().T
    # # format_str_to_time(df['target_time'])
    # simul.target_time #target 시간
    # df

    api_context = simul.api_call()


    # api_context
    #임의의 apicall을 한순간을 출발예정시간의 optimal로 생각할경우 target time 수정
    refined_target_time = api_context['call_time']+ api_context['path_time'] +8
    # refined_target_time #optimal로 부터 나온 target time
    # api_context

    #수정된 타겟 타임
    # format_time_to_str(refined_target_time)
    #새로운 타임머신
    refined_timeMachine=simul.run_time_machine(target_time=int(refined_target_time))
    # refined_timeMachine

    # test['refinedTimemachine'].append(refined_timeMachine['req_leaving_time'])
    refined_timeMachine_result=simul.api_call(api_time=int(refined_timeMachine['req_leaving_time']), target_time=refined_target_time)
    # test['TimeMachineResult'].append(refined_timeMachine['req_leaving_time']+refined_timeMachine_result['path_time'])
    # test['oracle'].append(simul.cur_time)
    # test['target_8'].append(refined_target_time-8)
    # refined_timeMachine_result


    
    # timeMachine으로 부터 60분전 부터 algorithm 실행 simul.cur_time이 oracle임
    if simul.cur_time > refined_timeMachine['req_leaving_time']:
        repeat = 61 + math.ceil((simul.cur_time - refined_timeMachine['req_leaving_time'])/time_interval)
    else:
        repeat = 61
    # repeat

    # refined_timeMachine
    # format_time_to_str(refined_timeMachine['req_leaving_time'])
    #context refine
    context['start_time'] = format_time_to_str(refined_timeMachine['req_leaving_time'] - 60)
    context['target_time'] = format_time_to_str(refined_target_time - 8)
    context['call_time_TimeMachine']=format_time_to_str(refined_timeMachine['call_time'])
    context['path_TimeMachine'] = refined_timeMachine['path']
    context['path_time_TimeMachine']= refined_timeMachine['path_time']
    context['cur_time']= format_time_to_str(refined_timeMachine['req_leaving_time'] - 60)
    context['req_leaving_time'] = format_time_to_str(refined_timeMachine['req_leaving_time'])
    # context

    context_df =  pd.Series(context).to_frame().T
    # context_df

    contexts_repeat_df = pd.concat([context_df]*int(repeat), ignore_index=True)
    # contexts_repeat_df

    contexts_repeat_df['cur_time'] = time_interval
    contexts_repeat_df['cur_time'][0] = refined_timeMachine['req_leaving_time'] - 60
    contexts_repeat_df['cur_time'] = (contexts_repeat_df['cur_time'].cumsum()).apply(format_time_to_str)
    contexts_repeat_df['remain_time_from_TimeMachine'] = contexts_repeat_df['req_leaving_time'].apply(format_str_to_time) - contexts_repeat_df['cur_time'].apply(format_str_to_time)
    contexts_repeat_df['oracle']= simul.cur_time

    contexts_repeat_df['round'] = 1
    contexts_repeat_df['round'] = contexts_repeat_df['round'].cumsum() -1
    contexts_repeat_df['episode'] = episode
    episode +=1


    df_final = pd.concat([df_final ,contexts_repeat_df], ignore_index=True)

# df = pd.DataFrame(test)
save_path='../../data/traffic/output_test2.csv'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_final.to_csv(save_path, index=False)

print("file saved successfully")
##############################################

# simul.reset_state(reset_all=True, save_interval=time_interval, verbose=0)

# context = simul.run()
# context

# repeat = (simul.target_time - 8 - simul.cur_time) // time_interval
# context_df =  pd.Series(context).to_frame().T
# context_df

# cotexts_repeat_df = pd.concat([context_df]*repeat, ignore_index=True)
# # cotexts_repeat_df=cotexts_repeat_df.drop(['call_time_LastAPI', 'call_point_LastAPI','path_LastAPI','call_node_LastAPI','path_time_LastAPI'], axis=1)

# cotexts_repeat_df['cur_time'] = time_interval

# cotexts_repeat_df['cur_time'][0] = simul.cur_time
# cotexts_repeat_df['cur_time'] = (cotexts_repeat_df['cur_time'].cumsum()).apply(format_time_to_str)
# cotexts_repeat_df['remain_time_from_TimeMachine'] = cotexts_repeat_df['req_leaving_time'].apply(format_str_to_time) - cotexts_repeat_df['cur_time'].apply(format_str_to_time)

# cotexts_repeat_df

# temporal_feature_cols = ['cur_time','target_time', 'call_time_TimeMachine', 'remain_time_from_TimeMachine']
# spatial_feature_cols = ['cur_point', 'target_point']
# other_feature_cols = ['path_time_TimeMachine']
# other_feature_cols_pathtime = ['path_time_TimeMachine', 'path_time_LastAPI']

# contexts_feature_df = make_feature_set_embedding(cotexts_repeat_df, 
#             temporal_feature_cols, spatial_feature_cols, other_feature_cols)

# contexts_feature_df

