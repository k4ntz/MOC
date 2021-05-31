# helpfile for checkpoints while training dqn

import torch
import os

PATH_TO_OUTPUTS = os.getcwd() + "/dqn/checkpoints/"
print(PATH_TO_OUTPUTS)

model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + "_model.pth"

def save_models(training_name, policy_model, target_model, optimizer, memory, episode, global_step, total_max_q, total_loss):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    model_path = model_name(training_name)
    print("Saving {}".format(model_path))
    torch.save({
            'policy_model_state_dict': policy_model.state_dict(),
            'target_model_state_dict': target_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'memory': memory,
            'episode': episode,
            'global_step': global_step,
            'total_max_q': total_max_q,
            'total_loss': total_loss
            }, model_path)


# chec# checks if model with given name exists,
def check_loading_model(training_name):
    if not os.path.exists(PATH_TO_OUTPUTS):
        print("{} does not exist".format(PATH_TO_OUTPUTS))
        return False
    model_path = model_name(training_name)
    if not os.path.isfile(model_path):
        print("{} does not exist".format(model_path))
        return False
    return True

