import os
import pandas as pd
import warnings
import glob
import numpy as np

warnings.filterwarnings('ignore')


class RlDataGenerator:
    """
    RL Data Generator for RL models.
    Reads raw data and constructs training data suitable for reinforcement learning.
    """

    def __init__(self, file_folder_path="./data/traffic"):

        self.file_folder_path = file_folder_path
        self.training_data_path = self.file_folder_path + "/" + "training_data_rlData_folder"

    def batch_generate_rl_data(self):
        os.makedirs(self.training_data_path, exist_ok=True)
        # csv_files = glob.glob(os.path.join(self.file_folder_path, '*.csv'))
        # print(csv_files)

        csv_path= self.file_folder_path + '/output_test2.csv'
        print(csv_path)
        # for csv_path in csv_files:
        print("Start processing：", csv_path)
        df = pd.read_csv(csv_path)
        df_processed = self._generate_rl_data(df)
        csv_filename = os.path.basename(csv_path)
        trainData_filename = csv_filename.replace('.csv', '-rlData.csv')
        trainData_path = os.path.join(self.training_data_path, trainData_filename)
        df_processed.to_csv(trainData_path, index=False)
        del df, df_processed
        print("File processed successfully：", csv_path)


    def _generate_rl_data(self, df):
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
            
        def extract_coordinates(point):
            try:
                x, y = point.strip('[]').split()
                return float(x), float(y)
            except:
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


def generate_rl_data():
    file_folder_path = "/home/sxxgwoo/clab/data/traffic"
    data_loader = RlDataGenerator(file_folder_path=file_folder_path)
    data_loader.batch_generate_rl_data()


if __name__ == '__main__':
    generate_rl_data()
