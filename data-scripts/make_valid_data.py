import os
import random

train_input = ""
train_label = ""

valid_input = ""
valid_label = ""
os.makedirs(valid_input, exist_ok=True)
os.makedirs(valid_label, exist_ok=True)

valid_list = os.listdir(train_input)
random.shuffle(valid_list)
valid_list = valid_list[:500]

for input_name in valid_list:
    os.rename(
        os.path.join(train_input, input_name), os.path.join(valid_input, input_name)
    )
    os.rename(
        os.path.join(train_label, input_name), os.path.join(valid_label, input_name)
    )
    print(input_name)
