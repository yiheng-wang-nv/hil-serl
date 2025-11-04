import copy
import datetime
import os
import pickle as pkl

import numpy as np
from absl import app, flags
from pynput import keyboard
from tqdm import tqdm

from unitree_examples.mappings import CONFIG_MAPPING
from unitree_examples.utils import observation_to_teleop, vector_to_teleop_action

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "unitree_assemble", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


success_key = False
def on_press(key):
    global success_key
    try:
        if str(key) == 'Key.space':
            success_key = True
    except AttributeError:
        pass

def main(_):
    global success_key
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, info = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        action_vector = np.zeros(env.action_space.shape, dtype=np.float32)
        if isinstance(info, dict) and "intervene_action_vector" in info:
            action_vector = np.asarray(info["intervene_action_vector"], dtype=np.float32)

        next_obs, rew, done, truncated, info = env.step(action_vector)

        transition = copy.deepcopy(
            dict(
                observations=observation_to_teleop(obs),
                actions=info.get("intervene_action", vector_to_teleop_action(action_vector)),
                next_observations=observation_to_teleop(next_obs),
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        if success_key:
            successes.append(transition)
            pbar.update(1)
            success_key = False
        else:
            failures.append(transition)

        if done or truncated:
            obs, info = env.reset()

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
        
if __name__ == "__main__":
    app.run(main)
