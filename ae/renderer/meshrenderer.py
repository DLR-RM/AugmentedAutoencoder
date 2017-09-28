# -*- coding: utf-8 -*-
import os
import numpy as np

from OpenGL.GL import *

import gl_utils as gu

class Renderer(object):

    MAX_FBO_WIDTH = 2000
    MAX_FBO_HEIGHT = 2000

    def __init__(self, models_cad_files, samples=1, vertex_tmp_store_folder='.', vertex_scale=1.):
        self._samples = samples
        self._context = gu.OffscreenContext()

        # FBO
        W, H = Renderer.MAX_FBO_WIDTH, Renderer.MAX_FBO_HEIGHT
        self._fbo = gu.Framebuffer( { GL_COLOR_ATTACHMENT0: gu.Texture(GL_TEXTURE_2D, 1, GL_RGB8, W, H),
                                      GL_COLOR_ATTACHMENT1: gu.Texture(GL_TEXTURE_2D, 1, GL_R32F, W, H),
                                      GL_DEPTH_STENCIL_ATTACHMENT: gu.Renderbuffer(GL_DEPTH32F_STENCIL8, W, H) } )
        self._fbo_depth = gu.Framebuffer( { GL_COLOR_ATTACHMENT0: gu.Texture(GL_TEXTURE_2D, 1, GL_RGB8, W, H),
                                      GL_COLOR_ATTACHMENT1: gu.Texture(GL_TEXTURE_2D, 1, GL_R32F, W, H),
                                      GL_DEPTH_STENCIL_ATTACHMENT: gu.Renderbuffer(GL_DEPTH32F_STENCIL8, W, H) } )
        glNamedFramebufferDrawBuffers(self._fbo.id, 2, np.array( (GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1),dtype=np.uint32 ) )
        glNamedFramebufferDrawBuffers(self._fbo_depth.id, 2, np.array( (GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1),dtype=np.uint32 ) )

        if self._samples > 1:
            self._render_fbo = gu.Framebuffer( { GL_COLOR_ATTACHMENT0: gu.TextureMultisample(self._samples, GL_RGB8, W, H, True),
                                                 GL_COLOR_ATTACHMENT1: gu.TextureMultisample(self._samples, GL_R32F, W, H, True),
                                                 GL_DEPTH_STENCIL_ATTACHMENT: gu.RenderbufferMultisample(self._samples, GL_DEPTH32F_STENCIL8, W, H) } )
            glNamedFramebufferDrawBuffers(self._render_fbo.id, 2, np.array( (GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1),dtype=np.uint32 ) )

        self._fbo.bind()

        # VAO
        vert_norms = gu.geo.load_meshes(models_cad_files, vertex_tmp_store_folder, recalculate_normals=True)

        vertices = np.empty(0, dtype=np.float32)
        for vert_norm in vert_norms:
            _verts = vert_norm[0] * vertex_scale
            vertices = np.hstack((vertices, np.hstack((_verts, vert_norm[1])).reshape(-1)))


        vao = gu.VAO({(gu.Vertexbuffer(vertices), 0, 6*4):
                        [   (0, 3, GL_FLOAT, GL_FALSE, 0*4),
                            (1, 3, GL_FLOAT, GL_FALSE, 3*4)]})
        vao.bind()

        sizes = [vert[0].shape[0] for vert in vert_norms]
        offsets = [sum(sizes[:i]) for i in xrange(len(sizes))]

        ibo = gu.IBO(sizes, np.ones(len(vert_norms)), offsets, np.zeros(len(vert_norms)))
        ibo.bind()

        gu.Shader.shader_folder = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'shader')
        shader = gu.Shader('cad_shader.vs', 'cad_shader.frag')
        shader.compile_and_use()

        self._scene_buffer = gu.ShaderStorage(0, gu.Camera().data , True)
        self._scene_buffer.bind()

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def light_direction(self, a):
        glUniform3f(0, a[0], a[1], a[2])

    def render(self, obj_id, W, H, K, R, t, near, far, randomLight=False):
        assert W <= Renderer.MAX_FBO_WIDTH and H <= Renderer.MAX_FBO_HEIGHT

        if self._samples > 1:
            self._render_fbo.bind()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |  GL_STENCIL_BUFFER_BIT)
        glViewport(0, 0, W, H)

        camera = gu.Camera()
        camera.realCamera(W, H, K, R, t, near, far)

        if randomLight:
            self.light_direction( 1000.*np.random.random(3) )
        else:
            self.light_direction( np.array([100., 100., 100]) )

        self._scene_buffer.update(camera.data)
        glDrawArraysIndirect(GL_TRIANGLES, ctypes.c_void_p(obj_id*16))

        if self._samples > 1:
            for i in xrange(2):
                glNamedFramebufferReadBuffer(self._render_fbo.id, GL_COLOR_ATTACHMENT0 + i)
                glNamedFramebufferDrawBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0 + i)
                glBlitNamedFramebuffer(self._render_fbo.id, self._fbo.id, 0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST)
            self._fbo.bind()

        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0)
        rgb_flipped = np.frombuffer( glReadPixels(0, 0, W, H, GL_BGR, GL_UNSIGNED_BYTE), dtype=np.uint8 ).reshape(H,W,3)
        rgb = np.flipud(rgb_flipped).copy()

        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT1)
        depth_flipped = glReadPixels(0, 0, W, H, GL_RED, GL_FLOAT).reshape(H,W)
        depth = np.flipud(depth_flipped).copy()

        return rgb, depth

    def close(self):
        self._context.close()
