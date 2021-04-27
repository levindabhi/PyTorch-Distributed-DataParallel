import torch

from networks import CustomModel
from utils.saving_utils import save_checkpoint

net = CustomModel()
save_checkpoint(net, os.path.join('previous_checkpoints', 'custom_net.pth'))

trained_net_pth =  # contains trained weights
custom_net_pth = os.path.join('previous_checkpoints', 'custom_net.pth') # contains random weights

net_state_dict = torch.load(trained_net_pth)
count = 0
for k, v in net_state_dict.items():
    count += 1
print('Total number of layers in trained model are: {}'.format(count))

custom_state_dict = torch.load(custom_net_pth)
count = 0
for k, v in custom_state_dict.items():
    count += 1
print('Total number of layers in trained model are: {}'.format(count))

total_count = 0 
update_count = 0
for k, v in net_state_dict.items():
    total_count += 1
    if custom_state_dict[k].shape == v.shape:
        update_count += 1
        custom_state_dict[k] = v

print('Out of {} layers in custom network, {} layers weights are recovered from trained model'.format(total_count, update_count))

torch.save_dict(custom_state_dict, os.path.join('previous_checkpoints', 'transfer_learn_net.pth'))

###
# Saves transfer weight dict as 'transfer_learn_net.pth'
###