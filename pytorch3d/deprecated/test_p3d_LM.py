# test pytorch 3d renderer
# TODO: make this work
# render multi objects in batch, one in one image
import errno
import os
import os.path as osp
import sys
import time
import struct
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2quat, mat2euler, quat2euler
from transforms3d.quaternions import axangle2quat, mat2quat, qinverse, qmult

# io utils
# from pytorch3d.io import load_obj, load_ply
# rendering components
from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer,
                                OpenGLPerspectiveCameras, PhongShader,
                                PointLights, RasterizationSettings,
                                SilhouetteShader, look_at_rotation,
                                look_at_view_transform)
# from pytorch3d.renderer.cameras import SfMPerspectiveCameras
from pytorch3d.renderer.cameras_real import OpenGLRealPerspectiveCameras
# datastructures
from pytorch3d.structures import Meshes, Textures, list_to_padded
# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(cur_dir, '../'))

data_dir = osp.join(cur_dir, '../datasets/')
output_directory = osp.join(cur_dir, '../output/results')

output_directory_ren = osp.join(output_directory, 'p3d')
os.makedirs(output_directory_ren, exist_ok=True)


ply_model_root = osp.join(data_dir, "BOP_DATASETS/lm/models")

HEIGHT = 480
WIDTH = 640
IMG_SIZE = 640
ZNEAR = 0.01
ZFAR = 10.0
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
objects = ["ape", "benchvise", "bowl", "camera", "can", "cat",
    "cup", "driller", "duck", "eggbox", "glue", "holepuncher", "iron", "lamp", "phone"]
id2obj = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}


def load_ply(path, vertex_scale=1.0):
    # https://github.com/thodan/sixd_toolkit/blob/master/pysixd/inout.py
    # bop_toolkit
    """Loads a 3D mesh model from a PLY file.

    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    -' pts' (nx3 ndarray),
    - 'normals' (nx3 ndarray), optional
    - 'colors' (nx3 ndarray), optional
    - 'faces' (mx3 ndarray), optional.
    - 'texture_uv' (nx2 ndarray), optional
    - 'texture_uv_face' (mx6 ndarray), optional
    - 'texture_file' (string), optional
    """
    f = open(path, "r")

    # Only triangular faces are supported.
    face_n_corners = 3

    n_pts = 0
    n_faces = 0
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False
    texture_file = None

    # Read the header.
    while True:

        # Strip the newline character(s)
        line = f.readline()
        if isinstance(line, str):
            line = line.rstrip("\n").rstrip("\r")
        else:
            line = str(line, 'utf-8').rstrip("\n").rstrip("\r")

        if line.startswith('comment TextureFile'):
            texture_file = line.split()[-1]
        elif line.startswith("element vertex"):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith("element"):  # Some other element.
            header_vertex_section = False
            header_face_section = False
        elif line.startswith("property") and header_vertex_section:
            # (name of the property, data type)
            prop_name = line.split()[-1]
            if prop_name == "s":
                prop_name = "texture_u"
            if prop_name == "t":
                prop_name = "texture_v"
            prop_type = line.split()[-2]
            pt_props.append((prop_name, prop_type))
        elif line.startswith("property list") and header_face_section:
            elems = line.split()
            if elems[-1] == "vertex_indices" or elems[-1] == 'vertex_index':
                # (name of the property, data type)
                face_props.append(("n_corners", elems[2]))
                for i in range(face_n_corners):
                    face_props.append(("ind_" + str(i), elems[3]))
            elif elems[-1] == 'texcoord':
                # (name of the property, data type)
                face_props.append(('texcoord', elems[2]))
                for i in range(face_n_corners * 2):
                    face_props.append(('texcoord_ind_' + str(i), elems[3]))
            else:
                print("Warning: Not supported face property: " + elems[-1])
        elif line.startswith("format"):
            if "binary" in line:
                is_binary = True
        elif line.startswith("end_header"):
            break

    # Prepare data structures.
    model = {}
    if texture_file is not None:
        model['texture_file'] = texture_file
    model["pts"] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model["faces"] = np.zeros((n_faces, face_n_corners), np.float)

    # print(pt_props)
    pt_props_names = [p[0] for p in pt_props]
    face_props_names = [p[0] for p in face_props]
    # print(pt_props_names)

    is_normal = False
    if {"nx", "ny", "nz"}.issubset(set(pt_props_names)):
        is_normal = True
        model["normals"] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {"red", "green", "blue"}.issubset(set(pt_props_names)):
        is_color = True
        model["colors"] = np.zeros((n_pts, 3), np.float)

    is_texture_pt = False
    if {"texture_u", "texture_v"}.issubset(set(pt_props_names)):
        is_texture_pt = True
        model["texture_uv"] = np.zeros((n_pts, 2), np.float)

    is_texture_face = False
    if {'texcoord'}.issubset(set(face_props_names)):
        is_texture_face = True
        model['texture_uv_face'] = np.zeros((n_faces, 6), np.float)

    # Formats for the binary case.
    formats = {
        "float": ("f", 4),
        "double": ("d", 8),
        "int": ("i", 4),
        "uchar": ("B", 1),
    }

    # Load vertices.
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ["x", "y", "z", "nx", "ny", "nz",
                      "red", "green", "blue", "texture_u", "texture_v"]
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                read_data = f.read(format[1])
                val = struct.unpack(format[0], read_data)[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model["pts"][pt_id, 0] = float(prop_vals["x"])
        model["pts"][pt_id, 1] = float(prop_vals["y"])
        model["pts"][pt_id, 2] = float(prop_vals["z"])

        if is_normal:
            model["normals"][pt_id, 0] = float(prop_vals["nx"])
            model["normals"][pt_id, 1] = float(prop_vals["ny"])
            model["normals"][pt_id, 2] = float(prop_vals["nz"])

        if is_color:
            model["colors"][pt_id, 0] = float(prop_vals["red"])
            model["colors"][pt_id, 1] = float(prop_vals["green"])
            model["colors"][pt_id, 2] = float(prop_vals["blue"])

        if is_texture_pt:
            model["texture_uv"][pt_id, 0] = float(prop_vals["texture_u"])
            model["texture_uv"][pt_id, 1] = float(prop_vals["texture_v"])

    # Load faces.
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == "n_corners":
                    if val != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                        # print("Number of face corners: " + str(val))
                        # exit(-1)
                elif prop[0] == 'texcoord':
                    if val != face_n_corners * 2:
                        raise ValueError('Wrong number of UV face coordinates.')
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == "n_corners":
                    if int(elems[prop_id]) != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                elif prop[0] == 'texcoord':
                    if int(elems[prop_id]) != face_n_corners * 2:
                        raise ValueError('Wrong number of UV face coordinates.')
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model["faces"][face_id, 0] = int(prop_vals["ind_0"])
        model["faces"][face_id, 1] = int(prop_vals["ind_1"])
        model["faces"][face_id, 2] = int(prop_vals["ind_2"])

        if is_texture_face:
            for i in range(6):
                model['texture_uv_face'][face_id, i] = float(
                    prop_vals['texcoord_ind_{}'.format(i)])

    f.close()
    model['pts'] *= vertex_scale

    return model


def grid_show(ims, titles=None, row=1, col=3, dpi=200, save_path=None, title_fontsize=5, show=True):
    if row * col < len(ims):
        print('_____________row*col < len(ims)___________')
        col = int(np.ceil(len(ims) / row))
    fig = plt.figure(dpi=dpi, figsize=plt.figaspect(row / float(col)))
    k = 0
    for i in range(row):
        for j in range(col):
            plt.subplot(row, col, k + 1)
            plt.axis('off')
            plt.imshow(ims[k])
            if titles is not None:
                # plt.title(titles[k], size=title_fontsize)
                plt.text(0.5, 1.08, titles[k],
                        horizontalalignment='center',
                        fontsize=title_fontsize,
                        transform=plt.gca().transAxes)
            k += 1
            if k == len(ims):
                break
    # plt.tight_layout()
    if show:
        plt.show()
    else:
        if save_path is not None:
            mkdir_p(osp.dirname(save_path))
            plt.savefig(save_path)
    return fig


def mkdir_p(dirname):
    """Like "mkdir -p", make a dir recursively, but do nothing if the dir
    exists.

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == "" or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def print_stat(data, name=""):
    print(name, "min", data.min(), "max", data.max(),
                "mean", data.mean(), "std", data.std(),
                "sum", data.sum(), "shape", data.shape)
###################################################################################################

def load_ply_models(model_paths, device='cuda', dtype=torch.float32, vertex_scale=0.001):
    ply_models = [load_ply(ply_path, vertex_scale=vertex_scale) for ply_path in model_paths]
    verts = [torch.tensor(m['pts'], device=device, dtype=dtype) for m in ply_models]
    faces = [torch.tensor(m['faces'], device=device, dtype=torch.int64) for m in ply_models]
    for m in ply_models:
        if m['colors'].max() > 1.1:
            m['colors'] /= 255.0
    verts_rgb_list = [torch.tensor(m['colors'], device=device, dtype=dtype) for m in ply_models]  # [V,3]
    res_models = []
    for i in range(len(ply_models)):
        model = {}
        model['verts'] = verts[i]
        model['faces'] = faces[i]
        model['verts_rgb'] = verts_rgb_list[i]
        res_models.append(model)
    return res_models


def main():
    # Set the cuda device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    ###########################
    # load objects
    ###########################
    objs = objects
    np.array([[-5.87785252e-01,  8.09016994e-01,  0.00000000e+00], [-4.95380036e-17, -3.59914664e-17, -1.00000000e+00], [-8.09016994e-01, -5.87785252e-01,  6.12323400e-17]])
    # obj_paths = [osp.join(model_root, '{}/textured.obj'.format(cls_name)) for cls_name in objs]
    # texture_paths = [osp.join(model_root, '{}/texture_map.png'.format(cls_name)) for cls_name in objs]

    ply_paths = [osp.join(ply_model_root, "obj_{:06d}.ply".format(obj2id[cls_name]))
            for cls_name in objs]

    models = load_ply_models(ply_paths, vertex_scale=0.001)

    cameras = OpenGLRealPerspectiveCameras(
        focal_length=((K[0,0], K[1,1]),),  # Nx2
        principal_point=((K[0,2], K[1,2]),),  # Nx2
        x0=0,
        y0=0,
        w=WIDTH,
        h=WIDTH, #HEIGHT,
        znear=ZNEAR,
        zfar=ZFAR,
        device=device,
    )

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 640x640. To form the blended image we use 100 faces for each pixel. Refer to rasterize_meshes.py
    # for an explanation of this parameter.
    silhouette_raster_settings = RasterizationSettings(
        image_size=IMG_SIZE,  # longer side or scaled longer side
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=100,  # the nearest faces_per_pixel points along the z-axis.
        bin_size=0
    )
    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=silhouette_raster_settings
        ),
        shader=SilhouetteShader(blend_params=blend_params)
    )
    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    phong_raster_settings = RasterizationSettings(
        image_size=IMG_SIZE,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=phong_raster_settings
        ),
        shader=PhongShader(device=device, lights=lights)
    )

    # pose =============================================
    R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
    R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
    R = np.dot(R1, R2)
    print("R det", torch.det(torch.tensor(R)))
    quat = mat2quat(R)
    t = np.array([-0.1, 0.1, 0.7], dtype=np.float32)
    t2 = np.array([0.1, 0.1, 0.7], dtype=np.float32)
    t3 = np.array([-0.1, -0.1, 0.7], dtype=np.float32)
    t4 = np.array([0.1, -0.1, 0.7], dtype=np.float32)
    t5 = np.array([0, 0.1, 0.7], dtype=np.float32)

    batch_size = 3
    Rs    = [R,    R.copy(),    R.copy(),    R.copy(),    R.copy()][:batch_size]
    print("R", R)
    quats = [quat, quat.copy(), quat.copy(), quat.copy(), quat.copy()][:batch_size]
    ts    = [t,    t2,          t3,          t4,          t5][:batch_size]

    runs = 100
    t_render = 0
    for i in tqdm(range(runs)):
        t_render_start = time.perf_counter()
        obj_ids = np.random.randint(0, len(objs), size=len(quats))

        # Render the objs providing the values of R and T.
        batch_verts_rgb = list_to_padded([models[obj_id]['verts_rgb'] for obj_id in obj_ids])  # B, Vmax, 3
        batch_textures = Textures(verts_rgb=batch_verts_rgb.to(device))
        batch_mesh = Meshes(
            verts=[models[obj_id]['verts'] for obj_id in obj_ids],
            faces=[models[obj_id]['faces'] for obj_id in obj_ids],
            textures=batch_textures,
        )
        batch_R = torch.tensor(np.stack(Rs), device=device, dtype=torch.float32).permute(0,2,1) # Bx3x3
        batch_T = torch.tensor(np.stack(ts), device=device, dtype=torch.float32) # Bx3

        silhouete = silhouette_renderer(meshes_world=batch_mesh, R=batch_R, T=batch_T)
        image_ref = phong_renderer(meshes_world=batch_mesh, R=batch_R, T=batch_T)
        # crop results
        silhouete = silhouete[:, :HEIGHT, :WIDTH, :].cpu().numpy()
        image_ref = image_ref[:, :HEIGHT, :WIDTH, :3].cpu().numpy()

        t_render += time.perf_counter() - t_render_start
        if True:
            pred_images = image_ref
            for i in range(pred_images.shape[0]):
                pred_mask = silhouete[i, :, :, 3].astype('float32')

                print("num rendered images", pred_images.shape[0])
                image = pred_images[i]
                print('image', image.shape)

                print('dr mask area: ', pred_mask.sum())

                print_stat(pred_mask, "pred_mask")
                show_ims = [image, pred_mask]
                show_titles = ['image', 'mask']
                grid_show(show_ims, show_titles, row=1, col=2)

    print("runs: {}, {:.3f}fps, {:.3f}ms/im".format(runs, runs / t_render, t_render / runs * 1000))


if __name__ == '__main__':
    main()
