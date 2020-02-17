import torch
import numpy as np

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, Textures, list_to_padded

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SilhouetteShader, PhongShader, PointLights, DirectionalLights
)

class BatchRender:
    def __init__(self, obj_path, device, batch_size=12):
        self.batch_indeces = np.arange(batch_size)
        self.obj_path = obj_path
        self.device = device
    
        # Initialize the renderer
        self.renderer = self.initRender()

        # Setup batch of meshes
        self.batch_mesh = self.initMeshes()

    def renderBatch(self, Rs, ts):
        if(type(Rs) is list):
            batch_R = torch.tensor(np.stack(Rs), device=self.device, dtype=torch.float32).permute(0,2,1) # Bx3x3
        else:
            batch_R = Rs
        if(type(ts) is list):
            batch_T = torch.tensor(np.stack(ts), device=self.device, dtype=torch.float32) # Bx3
        else:
            batch_T = ts

        # Reshape meshes and ts to fit length of Rs
        batch_mesh = self.batch_mesh[:batch_R.shape[0]]
        batch_T = batch_T[:batch_R.shape[0]]
            
        images = self.renderer(meshes_world=batch_mesh, R=batch_R, T=batch_T)
        return images

    def initMeshes(self):
        # Load the obj and ignore the textures and materials.
        verts, faces_idx, _ = load_obj(self.obj_path)
        faces = faces_idx.verts_idx
        
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)  # (V, 3)

        batch_verts_rgb = list_to_padded([verts_rgb for k in self.batch_indeces])  # B, Vmax, 3
        batch_textures = Textures(verts_rgb=batch_verts_rgb.to(self.device))
        batch_mesh = Meshes(
            verts=[verts.to(self.device) for k in self.batch_indeces],
            faces=[faces.to(self.device) for k in self.batch_indeces],
            textures=batch_textures,
        )
        return batch_mesh


    def initRender(self):
        cameras = OpenGLPerspectiveCameras(device=self.device)
        
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        raster_settings = RasterizationSettings(
            image_size=256, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
            bin_size=0
        )

        # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SilhouetteShader(blend_params=blend_params)
        )
        return renderer
    
