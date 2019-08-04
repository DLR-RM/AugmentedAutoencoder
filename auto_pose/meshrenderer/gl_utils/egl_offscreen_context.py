# -*- coding: utf-8 -*-
# flake8: noqa
import os
from ctypes import pointer

if not os.environ.get( 'PYOPENGL_PLATFORM' ):
    os.environ['PYOPENGL_PLATFORM'] = 'egl'


from OpenGL.GL import *
from OpenGL.EGL import (
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_BLUE_SIZE,
            EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_DEPTH_SIZE,
            EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_CONFORMANT,
            EGL_NONE, EGL_DEFAULT_DISPLAY, EGL_NO_CONTEXT,
            EGL_OPENGL_API, EGL_CONTEXT_MAJOR_VERSION,
            EGL_CONTEXT_MINOR_VERSION,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            eglGetDisplay, eglInitialize, eglChooseConfig,
            eglBindAPI, eglCreateContext, EGLConfig
        )
from OpenGL import arrays
from OpenGL.GL.NV.bindless_texture import *


class OffscreenContext(object):
    def __init__(self):
        config_attributes = arrays.GLintArray.asArray([
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_BLUE_SIZE, 8,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_DEPTH_SIZE, 24,
            EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_CONFORMANT, EGL_OPENGL_BIT,
            EGL_NONE
        ])

        context_attributes = arrays.GLintArray.asArray([
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 1,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE
        ])
        major, minor = ctypes.c_long(), ctypes.c_long()
        num_configs = ctypes.c_long()
        configs = (EGLConfig * 1)()

        # Cache DISPLAY if necessary and get an off-screen EGL display
        orig_dpy = None
        if 'DISPLAY' in os.environ:
            orig_dpy = os.environ['DISPLAY']
            del os.environ['DISPLAY']
        self._egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if orig_dpy is not None:
            os.environ['DISPLAY'] = orig_dpy

        # Initialize EGL
        assert eglInitialize(self._egl_display, major, minor)
        assert eglChooseConfig(
            self._egl_display, config_attributes, pointer(configs), 1, pointer(num_configs)
        )

        # Bind EGL to the OpenGL API
        assert eglBindAPI(EGL_OPENGL_API)

        # Create an EGL context
        self._egl_context = eglCreateContext(
            self._egl_display, configs[0],
            EGL_NO_CONTEXT, context_attributes
        )
        if self._egl_context == EGL_NO_CONTEXT:
            raise RuntimeError('Unable to create context')

        # Make it current
        self.make_current()

        if not glInitBindlessTextureNV():
            raise RuntimeError("Bindless Textures not supported")
        self.__display = self._egl_display

    def make_current(self):
        from OpenGL.EGL import eglMakeCurrent, EGL_NO_SURFACE
        assert eglMakeCurrent(
            self._egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
            self._egl_context
        )

    def close(self):
        self.delete_context()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def delete_context(self):
        from OpenGL.EGL import eglDestroyContext, eglTerminate
        if self._egl_display is not None:
            if self._egl_context is not None:
                eglDestroyContext(self._egl_display, self._egl_context)
                self._egl_context = None
            eglTerminate(self._egl_display)
            self._egl_display = None

    def supports_framebuffers(self):
        return True


if __name__=='__main__':
    conntext = OffscreenContext()
