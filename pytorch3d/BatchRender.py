import torch
import numpy as np
import torch.nn as nn

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

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        #return images, fragments.zbuf
        return fragments.zbuf

class BatchRender:
    def __init__(self, obj_path, device, batch_size=12,
                 render_method="silhouette", image_size=256):
        self.batch_size = batch_size
        self.batch_indeces = np.arange(self.batch_size)
        self.obj_path = obj_path
        self.device = device

        # Setup batch of meshes
        self.batch_mesh = self.initMeshes()

        # Initialize the renderer
        self.renderer = self.initRender(image_size=image_size, method=render_method)

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
        #batch_mesh = self.batch_mesh[:batch_R.shape[0]]
        #batch_T = batch_T[:batch_R.shape[0]]         
        #images = self.renderer(meshes_world=batch_mesh, R=batch_R, T=batch_T)
        
        images = self.renderer(meshes_world=self.batch_mesh, R=batch_R, T=batch_T)
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
            textures=batch_textures
        )
        return batch_mesh


    def initRender(self, method, image_size):      
        cameras = OpenGLPerspectiveCameras(device=self.device)

        if(method=="silhouette"):
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

            raster_settings = RasterizationSettings(
                image_size=image_size, 
                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
                faces_per_pixel=100, 
                bin_size=0
            )
            
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=SilhouetteShader(blend_params=blend_params)
            )
        elif(method=="depth"):
            blend_params = BlendParams(sigma=5e-5, gamma=5e-5)

            raster_settings = RasterizationSettings(
                image_size=image_size, 
                blur_radius=np.log(1. / 5e-5 - 1.) * blend_params.sigma, 
                faces_per_pixel=5, 
                bin_size=0
            )
            
            renderer = MeshRendererWithDepth(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=SilhouetteShader(blend_params=blend_params)
            )
        elif(method=="phong"):
            raster_settings = RasterizationSettings(
                image_size=image_size, 
                blur_radius=0, 
                faces_per_pixel=1, 
                bin_size=0
            )
            lights = DirectionalLights(device=self.device,
                                       ambient_color=[[0.4, 0.4, 0.4]],
                                       diffuse_color=[[0.8, 0.8, 0.8]],
                                       specular_color=[[0.3, 0.3, 0.3]],
                                       direction=[[-1.0, -1.0, 1.0]])
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=PhongShader(device=self.device, lights=lights)
            )
        else:
            print("Unknown render method!")
            return None
        return renderer
    
