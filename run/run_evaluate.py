import numpy as np
import pandas as pd
import torch
import os
import ast

import math
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
from base.dataloader.simul_env import *

class evaluate():
    def __init__(self):
        # file_name = os.path.dirname(os.path.realpath(__file__))
        # dir_name = os.path.dirname(file_name)
        # dir_name = os.path.dirname(dir_name)
        model_path = '/home/sxxgwoo/clab/base/dataloader/saved_model/BCtest/bc_model.pth'
        
        self.model = torch.jit.load(model_path)
        
    def action(self, test_state, action):
        
        test_state = torch.tensor(test_state, dtype=torch.float)
        pred_action = self.model(test_state)
        
        return action, pred_action.detach().numpy()  # 예측된 행동을 반환 (Tensor에서 NumPy로 변환)
    
def generate_rl_data(df):
    """
    Generate RL data from raw data.
    Process each row individually and reorganize the data, grouping by the same id.
    """
    def format_str_to_time(time_str):
        if time_str is None:
            return None
        else:
            week_dict = {"Mon.":0, "Tue.":1, "Wed.":2, "Thu.":3, "Fri.":4, "Sat.":5, "Sun.":6}

            week_str, hour_min_str = time_str.split(" ")
            hour, min = hour_min_str.split(":")
            return week_dict[week_str]*24*60 + 60*int(hour) + int(min)
        
    # def extract_coordinates(point):
    #     try:
    #         x, y = point.strip('[]').split(' ')
    #         return float(x), float(y)
    #     except:
    #         print(point)
    #         return np.nan, np.nan
    def extract_coordinates(point):
        try:
            if isinstance(point, str):  # point가 문자열일 경우
                # 문자열에서 쉼표와 공백을 모두 제거한 후 값을 나누기
                x, y = point.strip('[]').split(',')
                return float(x), float(y)
            elif isinstance(point, np.ndarray):  # point가 numpy 배열일 경우
                # 배열의 첫 번째와 두 번째 요소를 반환
                return float(point[0]), float(point[1])
            else:
                # 다른 경우에는 NaN 반환
                return np.nan, np.nan
        except:
            # 에러 발생 시 NaN 반환
            return np.nan, np.nan
        
    training_data_rows = []
    df[['start_x', 'start_y']] = df['start_point'].apply(lambda x: pd.Series(extract_coordinates(x)))
    df[['target_x', 'target_y']] = df['target_point'].apply(lambda x: pd.Series(extract_coordinates(x)))
    grouped = df.groupby('id')
    
    for id_value, group in grouped:

        group = group.sort_values(by='round').reset_index(drop=True)
        
        actions = []
        for index, row in group.iterrows():

            remain_time_from_TimeMachine = format_str_to_time(row['req_leaving_time'])- format_str_to_time(row['cur_time'])
            
        
            eu_distance = np.sqrt((row['start_x'] - row['target_x'])**2 + 
                                (row['start_y'] - row['target_y'])**2)
            
            var = abs(format_str_to_time(row['req_leaving_time']) - format_str_to_time(row['cur_time'])) - abs(format_str_to_time(row['req_leaving_time']) - row['oracle'])
            
            if 0<=var<2:
                action = 1
            else:
                action = 0
            
            actions.append(action)

            if index == 0:
                avg_action_last_3 = 0
            elif index == 1:
                avg_action_last_3 = np.mean(actions[-2:-1])
            elif index == 2:
                avg_action_last_3 = np.mean(actions[-3:-1])
            else:
                avg_action_last_3 = np.mean(actions[-4:-1])

            state = (
                format_str_to_time(row['cur_time']),
                format_str_to_time(row['target_time']),
                remain_time_from_TimeMachine,
                row['start_x'],
                row['start_y'],
                row['target_x'],
                row['target_y'],
                eu_distance,
                row['path_time_TimeMachine'],
                avg_action_last_3
            )

            training_data_rows.append({
                'id': row['id'],
                'group': row['group'],
                'round': row['round'],
                'req_leaving_time':format_str_to_time(row['req_leaving_time']),
                'oracle': row['oracle'],
                'cur_time': format_str_to_time(row['cur_time']),
                'start_time': row['start_time'],
                'state': state,
                'action': action
            })

    training_data = pd.DataFrame(training_data_rows)
    
    return training_data

def safe_literal_eval(val):
    if pd.isna(val):
        return val  # NaN인 경우 그대로 반환
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        print(ValueError)
        return val  # 파싱 실패 시 원래 값을 반환
            
if __name__ == '__main__':
    # file_path = '/home/sxxgwoo/clab/data/traffic/training_data_rlData_folder/output_test-rlData.csv'
    # df = pd.read_csv(file_path)
    
    # eval = evaluate()
    
    # def safe_literal_eval(val):
    #     if pd.isna(val):
    #         return val  # NaN인 경우 그대로 반환
    #     try:
    #         return ast.literal_eval(val)
    #     except (ValueError, SyntaxError):
    #         print(ValueError)
    #         return val  # 파싱 실패 시 원래 값을 반환
    
    # # state 열을 파싱
    # df['state'] = df['state'].apply(safe_literal_eval)
     
    # # 결과 저장을 위한 리스트 초기화
    # results = []

    # # 각 행에 대해 예측된 행동과 실제 행동을 비교
    # for index, row in df.iterrows():
    #     actual_action, predicted_action = eval.action(test_state=row['state'], action=row['action'])
        
    #     # 결과를 리스트에 저장
    #     results.append({'index': index, 'actual_action': actual_action, 'predicted_action': predicted_action, 'req_leaving_time':row['req_leaving_time'],'oracle':row['oracle'],'cur_time':row['cur_time']})

    # # 리스트를 DataFrame으로 변환
    # results_df = pd.DataFrame(results)
    
    # # 결과를 CSV 파일로 저장
    # output_file_path = '/home/sxxgwoo/clab/result.csv'
    # results_df.to_csv(output_file_path, index=False)

    # print(f'Results saved to {output_file_path}')
    
    

##############################################[ online data test]######################################
    random_state = 1
    rng = np.random.RandomState(random_state)  
    n_nodes = 50               
    total_period = 7*24*60           
    time_interval=1

    # (create base graph) 
    node_map = GenerateNodeMap(n_nodes, random_state=random_state)
    node_map.create_node(node_scale=n_nodes, cov_knn=3)           # create node
    node_map.create_connect(connect_scale=0)             # create connection

    # (periodic setting)
    periodic_f = periodic_improve    # periodic function

    # (random noise)
    random_noise_f = RandomNoise(scale=1/1000, random_state=random_state)

    # (aggregated route_graph instance)
    temporal_graph = TemporalGraph(node_map=node_map, amp_class=TemporalGaussianAmp,
                                    periodic_f=periodic_f, error_f=random_noise_f, random_state=random_state)
    # (temporal gaussian amplitude)
    temporal_graph.create_amp_instance(T_arr = np.arange(24*60), mean_center=[0.5,0.5], mean_r_f=mean_r_f, cov_f=cov_scale_f,
                            centers=node_map.centers, base_adj_mat=node_map.adj_matrix,
                            normalize=1.5, adjust=True, repeat=9)
    temporal_graph.make_temporal_observations()

    # simulation
    simul = TemporalMapSimulation(temporal_graph)

    episode =0
    
    
    eval = evaluate()
    
    results = []
    for i in range(1000):
        simul.reset_state(reset_all=True, save_interval=1, verbose=1)
        df_final = pd.DataFrame()
        
        context = simul.run()
        api_context = simul.api_call()
        
        # api_context
        #임의의 apicall을 한순간을 출발예정시간의 optimal로 생각할경우 target time 수정
        refined_target_time = api_context['call_time']+ api_context['path_time'] +8

        #새로운 타임머신
        refined_timeMachine=simul.run_time_machine(target_time=int(refined_target_time))
        # refined_timeMachine

        # test['refinedTimemachine'].append(refined_timeMachine['req_leaving_time'])
        refined_timeMachine_result=simul.api_call(api_time=int(refined_timeMachine['req_leaving_time']), target_time=refined_target_time)
        
        # timeMachine으로 부터 60분전 부터 algorithm 실행 simul.cur_time이 oracle임
        if simul.cur_time > refined_timeMachine['req_leaving_time']:
            repeat = 61 + math.ceil((simul.cur_time - refined_timeMachine['req_leaving_time'])/time_interval)
        else:
            repeat = 61
        context['start_time'] = format_time_to_str(refined_timeMachine['req_leaving_time'] - 60)
        context['target_time'] = format_time_to_str(refined_target_time - 8)
        context['call_time_TimeMachine']=format_time_to_str(refined_timeMachine['call_time'])
        context['path_TimeMachine'] = refined_timeMachine['path']
        context['path_time_TimeMachine']= refined_timeMachine['path_time']
        context['cur_time']= format_time_to_str(refined_timeMachine['req_leaving_time'] - 60)
        context['req_leaving_time'] = format_time_to_str(refined_timeMachine['req_leaving_time'])

        
        context_df =  pd.Series(context).to_frame().T

        contexts_repeat_df = pd.concat([context_df]*int(repeat), ignore_index=True)

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
   
      
        #rl data generate
        df_rl=generate_rl_data(df_final)
        
        # state 열을 파싱
        # df_rl['state'] = df_rl['state'].apply(safe_literal_eval)
        # 결과 저장을 위한 리스트 초기화
        
        # print(df_rl['state'])
        # 각 행에 대해 예측된 행동과 실제 행동을 비교
        for index, row in df_rl.iterrows():
            
            actual_action, predicted_action = eval.action(test_state=row['state'], action=row['action'])
            
            # 결과를 리스트에 저장
            # results.append({'episode':i,'index': index, 'actual_action': actual_action, 'predicted_action': predicted_action, 'req_leaving_time':row['req_leaving_time'],'oracle':row['oracle'],'cur_time':row['cur_time']})
            
            if predicted_action[0] > 0.1:
                #첫번째 api call
                print(predicted_action)
                pred_1=simul.api_call(api_time=row['cur_time'], target_time=refined_target_time)
                if row['cur_time'] < row['req_leaving_time']:
                    dis=row['req_leaving_time']-row['cur_time']
                    #두번째 api call
                    refined_cur_time =row['cur_time']+dis
                    pred_2=simul.api_call(api_time=refined_cur_time, target_time=refined_target_time)
                    # print(max(refined_cur_time,pred_2['req_leaving_time']))
                    test_2=simul.api_call(api_time=max(int(refined_cur_time),int(pred_2['req_leaving_time'])), target_time=refined_target_time)
                    eval_2 = test_2['path_time']+ max(refined_cur_time,pred_2['req_leaving_time'])
                
                else:
                    eval_2=0
                    pass

                test_1=simul.api_call(api_time=max(int(row['cur_time']), int(pred_1['req_leaving_time'])), target_time=refined_target_time)
                eval_1 = test_1['path_time']+ max(row['cur_time'], pred_1['req_leaving_time'])
                    
                if (refined_target_time-13 <= eval_1 <=refined_target_time) or (refined_target_time-13 <= eval_2 <=refined_target_time):
                    test_result=1
                    results.append({'episode':i,'index': index,'result':test_result , 'actual_action': actual_action, 'predicted_action': predicted_action, 'req_leaving_time':row['req_leaving_time'],'oracle':row['oracle'],'cur_time':row['cur_time'],'eval':eval_1,'target_time':refined_target_time})
                    if row['cur_time'] < row['req_leaving_time']:
                        results.append({'episode':i,'index': index+dis,'result':test_result , 'actual_action': actual_action, 'predicted_action': predicted_action, 'req_leaving_time':row['req_leaving_time'],'oracle':row['oracle'],'cur_time':row['cur_time']+dis,'eval':eval_2,'target_time':refined_target_time})
                        
                else:
                    test_result=0
                    results.append({'episode':i,'index': index,'result':test_result , 'actual_action': actual_action, 'predicted_action': predicted_action, 'req_leaving_time':row['req_leaving_time'],'oracle':row['oracle'],'cur_time':row['cur_time'],'eval':eval_1,'target_time':refined_target_time})
                break
            else:
                pass
            
                   
            
    # 리스트를 DataFrame으로 변환
    results_df = pd.DataFrame(results)
    
    # 결과를 CSV 파일로 저장
    output_file_path = '/home/sxxgwoo/clab/result_online.csv'
    results_df.to_csv(output_file_path, index=False)

    print(f'Results saved to {output_file_path}')
            
            