# Check if gpu is available to use. True - Yes | False - No

import torch
print(torch.cuda.is_available())