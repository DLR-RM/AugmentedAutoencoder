import torch
import torch.nn as nn


class DepthShader(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        image = fragments.zbuf
        flattened = torch.flatten(image, start_dim=1)
        max_val,_ = torch.max(flattened, 1)
        #print(max_val)
        #mask = image == -1.0
        
        #print(mask[:1].shape)

        for i in range(12):
            mask = image[i] == -1.0
            image[i][mask] = 6.0 #float(torch.max(max_val).cpu().detach().numpy()) #max_val[i]
            #print(max_val[i])

        #image[mask] = float(torch.max(max_val).cpu().detach().numpy())

        return image
