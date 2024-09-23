import numpy as np
import torch
import pandas as pd
from base.common.utils import normalize_state, normalize_reward, save_normalize_dict
from base.algorithms.bc.replay_buffer import ReplayBuffer
from base.algorithms.bc.behavior_clone import BC
import logging
import ast

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

    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
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
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)

    state_dim = 16
    normalize_indices = [13, 14, 15]
    is_normalize = True

    normalize_dic = normalize_state(training_data, state_dim, normalize_indices) #return값이 normalize하려는 column의 stat임, training_data에 normalize값 추가되어있음
    normalize_reward(training_data, "reward_continuous") #필요없을듯
    save_normalize_dict(normalize_dic, "saved_model/BCtest") #normalize한 column의 stat을 pkl로 저장

    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")

    model = BC(dim_obs=state_dim)
    step_num = 20000
    batch_size = 100
    for i in range(step_num):
        states, actions, _, _, _ = replay_buffer.sample(batch_size)
        a_loss = model.step(states, actions)
        logger.info(f"Step: {i} Action loss: {np.mean(a_loss)}")

    # model.save_net("saved_model/BCtest")
    model.save_jit("saved_model/BCtest")
    test_trained_model(model, replay_buffer)


def load_model():
    """
    load model
    """
    model = BC(dim_obs=16)
    model.load_net("saved_model/BCtest")
    test_state = np.ones(16, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward \
                if not is_normalize else row.normalize_reward, row.next_state \
                    if not is_normalize else row.normalize_nextstate, row.done
        # Removed all data where done == 1
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred_action:", tem)


if __name__ == "__main__":
    run_bc()
# run_bc() -> train_model() -> 