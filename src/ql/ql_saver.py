# helpfile for checkpoints while training ql

import pickle
import bz2
import os

PATH_TO_OUTPUTS = os.getcwd() + "/ql/checkpoints/"

model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + ".pbz2"

def save_qtable(training_name, q_table, i_episode, global_step):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    checkpoint_path = model_name(training_name)
    print("Saving {}".format(checkpoint_path))
    saving_dict = {}
    saving_dict["q_table"] = q_table
    saving_dict["i_episode"] = i_episode
    saving_dict["global_step"] = global_step
    # create bz2 file
    qfile = bz2.BZ2File(checkpoint_path,'w')
    pickle.dump(saving_dict, qfile)
    qfile.close()


# check checks if model with given name exists,
# and loads it
def load_qtable(training_name):
    if not os.path.exists(PATH_TO_OUTPUTS):
        print("{} does not exist".format(PATH_TO_OUTPUTS))
        return None, None, None
    checkpoint_path = model_name(training_name)
    if not os.path.isfile(checkpoint_path):
        print("{} does not exist".format(checkpoint_path))
        return None, None, None
    # load bz2 file
    qfile = bz2.BZ2File(checkpoint_path,'r')
    loading_dict = pickle.load(qfile)
    qfile.close()
    return loading_dict["q_table"], loading_dict["i_episode"], loading_dict["global_step"]



