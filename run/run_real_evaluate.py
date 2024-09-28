import numpy as np
import pandas as pd
import torch
import ast
class evaluate():
    def __init__(self):
        # file_name = os.path.dirname(os.path.realpath(__file__))
        # dir_name = os.path.dirname(file_name)
        # dir_name = os.path.dirname(dir_name)
        model_path = '/home/sxxgwoo/clab/run/saved_model/BCrealtest/bc_model.pth'
        
        self.model = torch.jit.load(model_path)
        
    def action(self, test_state, action):
        
        test_state = torch.tensor(test_state, dtype=torch.float)
        pred_action = self.model(test_state)
        
        return action, pred_action.detach().numpy()  # 예측된 행동을 반환 (Tensor에서 NumPy로 변환)
    

# if __name__ == '__main__':
    
#     path = '/home/sxxgwoo/clab/data/real/filtered/training_data_rlData_folder/test_set_offline-rlData.csv'
#     df = pd.read_csv(path)
#     eval = evaluate()
#     df= df.head(130)
#     results = []
#     def safe_literal_eval(val):
#         if pd.isna(val):
#             return val  # If it is NaN, return NaN
#         try:
#             return ast.literal_eval(val)
#         except (ValueError, SyntaxError):
#             print(ValueError)
#             return val  # If parsing fails, return the original

#     # Using the apply method to apply the above function
#     df["state"] = df["state"].apply(safe_literal_eval)
    
#     for index, row in df.iterrows():
        
#         actual_action, predicted_action = eval.action(test_state=row['state'], action=row['action'])
        
#         # 결과를 리스트에 저장
#         results.append({'episode':i,'index': index, 'actual_action': actual_action, 'predicted_action': predicted_action, 'req_leaving_time':row['req_leaving_time'],'oracle':row['oracle'],'cur_time':row['cur_time']})
        
#         if predicted_action[0] > 0.1:
#             #첫번째 api call
#             print(row['state'])
#             print(predicted_action)
#             print(actual_action)
#         #     pred_1=simul.api_call(api_time=row['cur_time'], target_time=refined_target_time)
#         #     if row['cur_time'] < row['req_leaving_time']:
#         #         dis=row['req_leaving_time']-row['cur_time']
#         #         #두번째 api call
#         #         refined_cur_time =row['cur_time']+dis
#         #         pred_2=simul.api_call(api_time=refined_cur_time, target_time=refined_target_time)
#         #         # print(max(refined_cur_time,pred_2['req_leaving_time']))
#         #         test_2=simul.api_call(api_time=max(int(refined_cur_time),int(pred_2['req_leaving_time'])), target_time=refined_target_time)
#         #         eval_2 = test_2['path_time']+ max(refined_cur_time,pred_2['req_leaving_time'])
            
#         #     else:
#         #         eval_2=0
#         #         pass

#         #     test_1=simul.api_call(api_time=max(int(row['cur_time']), int(pred_1['req_leaving_time'])), target_time=refined_target_time)
#         #     eval_1 = test_1['path_time']+ max(row['cur_time'], pred_1['req_leaving_time'])
                
#         #     if (refined_target_time-13 <= eval_1 <=refined_target_time) or (refined_target_time-13 <= eval_2 <=refined_target_time):
#         #         test_result=1
#         #         results.append({'episode':i,'index': index,'result':test_result , 'actual_action': actual_action, 'predicted_action': predicted_action, 'req_leaving_time':row['req_leaving_time'],'oracle':row['oracle'],'cur_time':row['cur_time'],'eval':eval_1,'target_time':refined_target_time})
#         #         if row['cur_time'] < row['req_leaving_time']:
#         #             results.append({'episode':i,'index': index+dis,'result':test_result , 'actual_action': actual_action, 'predicted_action': predicted_action, 'req_leaving_time':row['req_leaving_time'],'oracle':row['oracle'],'cur_time':row['cur_time']+dis,'eval':eval_2,'target_time':refined_target_time})
                    
#         #     else:
#         #         test_result=0
#         #         results.append({'episode':i,'index': index,'result':test_result , 'actual_action': actual_action, 'predicted_action': predicted_action, 'req_leaving_time':row['req_leaving_time'],'oracle':row['oracle'],'cur_time':row['cur_time'],'eval':eval_1,'target_time':refined_target_time})
#         #     break
#         # else:
#         #     pass
            
                   
            
#     # # 리스트를 DataFrame으로 변환
#     # results_df = pd.DataFrame(results)
    
#     # # 결과를 CSV 파일로 저장
#     output_file_path = '/home/sxxgwoo/clab/result_real_online.csv'
#     results_df.to_csv(output_file_path, index=False)

#     print(f'Results saved to {output_file_path}')
            
        
        
if __name__ == '__main__':
    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # If it is NaN, return NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val  # If parsing fails, return the original
    path = '/home/sxxgwoo/clab/data/real/filtered/training_data_rlData_folder/test_set_offline-rlData.csv'
    df = pd.read_csv(path)
    eval = evaluate()
    # df = df.head(130)  # Subset for testing purposes
    results = []

    # Apply the literal eval to 'state' column
    df["state"] = df["state"].apply(safe_literal_eval)
    
    # Add a 'test_result' column initialized to 0
    df['test_result'] = 0

    # Collect results with actions and predicted actions
    for index, row in df.iterrows():
        actual_action, predicted_action = eval.action(test_state=row['state'], action=row['action'])
        
        # Store results in a dictionary
        results.append({
            'id': row['id'],  # Group by id
            'index': index, 
            'actual_action': actual_action, 
            'predicted_action': predicted_action[0],  # Taking first value for comparison
            'req_leaving_time': row['req_leaving_time_timemachine'],
            'oracle': row['oracle'],
            'cur_time': row['cur_time'],
        })

        # if predicted_action[0] > 0.1:
        #     # 첫 번째 api call 시 출력
        #     print(row['state'])
        #     print(predicted_action)
        #     print(actual_action)
    
    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)

    # Find the maximum predicted action per 'episode'
    max_predicted_idx = results_df.groupby('id')['predicted_action'].idxmax()

    # Set 'test_result' to 1 for rows with the maximum predicted action
    df.loc[max_predicted_idx, 'test_result'] = 1

    # Save results to CSV
    output_file_path = '/home/sxxgwoo/clab/result_real_online2.csv'
    df.to_csv(output_file_path, index=False)

    print(f'Results saved to {output_file_path}')