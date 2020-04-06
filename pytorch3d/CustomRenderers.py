import torch
import torch.nn as nn
from pytorch3d.renderer.blending import softmax_rgb_blend

class DepthShader(nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        image = fragments.zbuf
            
        flattened = torch.flatten(image, start_dim=1)
        max_val,_ = torch.max(flattened, 1)
        #print(max_val)
        #mask = image == -1.0
        
        #print(mask[:1].shape)

        #for i in range(12):
        #    mask = image[i] == -1.0
        #    image[i][mask] = 6.0 #float(torch.max(max_val).cpu().detach().numpy()) #max_val[i]x

        #image[mask] = float(torch.max(max_val).cpu().detach().numpy())

        #colors = torch.stack([fragments.zbuf, fragments.zbuf, fragments.zbuf], dim=4)
        #images = softmax_rgb_blend(colors, fragments, self.blend_params)

        
        
        return image
