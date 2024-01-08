import argparse
import json
import numpy as np
import torch
import os
import shutil

from core import load_params, run_session, get_env_and_graph


def run(dir_path):
    # load parameters from json files
    run_params, graph_params, varied_hps = load_params(dir_path)

    # seed numpy and pytorch
    np.random.seed(run_params["seed"])
    torch.manual_seed(run_params["seed"])

    env, graph = get_env_and_graph(run_params, graph_params)

    # run session
    sess_props = run_session(dir_path, graph, env, run_params, 0)

    # delete old log directory in state directory
    log_state_dir = os.path.join(dir_path, "state", "log")
    if os.path.isdir(log_state_dir):
        shutil.rmtree(log_state_dir)

    if sess_props["timed_out"]:
        # save the state of the graph (replay buffer, parameters...) in
        # order to be able to continue training
        print("Saving state of graph.")
        state_dir = os.path.join(dir_path, "state")
        os.makedirs(state_dir, exist_ok=True)
        graph.save_state(os.path.join(state_dir))
        # save a copy of the log directory in the state directory
        shutil.copytree(os.path.join(dir_path, "log"), log_state_dir)
    else:
        # delete state if present
        state_path = os.path.join(dir_path, "state")
        if os.path.isdir(state_path):
            shutil.rmtree(state_path)

    os.makedirs(os.path.join(dir_path, "state"), exist_ok=True)
    with open(os.path.join(dir_path, "state", "step.json"), "w") as json_file:
        json.dump({"step": sess_props["total_step"]}, json_file, indent=4)


if __name__ == "__main__":
    source_folder = './data/AntFourRooms/hits_trained'
    foldername = "ant1"
    path = os.path.join("./train_data/" + foldername)

    # check if destination folder exists
    new_destination_folder = path
    # if os.path.exists(path):
    #     # create a new destination folder with a new name
    #         new_destination_folder = path + "_copy"
    #         count = 1
    #         while os.path.exists(new_destination_folder):
    #             new_destination_folder = path + "_copy" + str(count)
    #             count += 1
    #         shutil.copytree(source_folder, new_destination_folder)
    # else:
    #     shutil.copytree(source_folder, path)

    run(new_destination_folder)
