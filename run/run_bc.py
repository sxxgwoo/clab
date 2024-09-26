import numpy as np
import torch
import pandas as pd
import ast
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

# from base.common.utils import normalize_state, normalize_reward, save_normalize_dict
from algorithms.bc.replay_buffer import ReplayBuffer
from algorithms.bc.behavior_clone import BC
import logging

np.set_printoptions(suppress=True, precision=4)

# Log configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_bc():
    """
    Run bc model training and evaluation.
    """
    train_model()
    # load_model()


def train_model():
    """
    train BC model
    """

    train_data_path = "/home/sxxgwoo/clab/data/traffic/training_data_rlData_folder/output_1-rlData.csv"
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # If it is NaN, return NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val  # If parsing fails, return the original

    # Using the apply method to apply the above function
    training_data["state"] = training_data["state"].apply(safe_literal_eval)

    state_dim = 10
    
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data)
    print(len(replay_buffer.memory))

    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")

    model = BC(dim_obs=state_dim)
    step_num = 50000
    batch_size = 100
    for i in range(step_num):
        states, actions = replay_buffer.sample(batch_size)
        a_loss = model.step(states, actions)
        logger.info(f"Step: {i} Action loss: {np.mean(a_loss)}")

    # model.save_net("saved_model/BCtest")
    model.save_jit("saved_model/BCtest")
    test_trained_model(model, replay_buffer)


def load_model():
    """
    load model
    """
    model = BC(dim_obs=10)
    model.load_net("saved_model/BCtest")
    test_state = np.ones(10, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


def add_to_replay_buffer(replay_buffer, training_data):
    for row in training_data.itertuples():
        state, action = row.state, row.action
        
        replay_buffer.push(np.array(state), np.array([action]))



def test_trained_model(model, replay_buffer):
    states, actions = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred_action:", tem)


if __name__ == "__main__":
    run_bc()
# run_bc() -> train_model() -> 