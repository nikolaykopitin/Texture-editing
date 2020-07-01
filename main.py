import glfw
import numpy
import pyrr
from PIL import Image
from Loaders import *

w_width, w_height = 800, 600


def main():
    if not glfw.init():
        print("Ошибка иннициализации")
        return

    window = glfw.create_window(w_width, w_height, "Window", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)
    obj = ObjLoader()
    obj.load_model("res/face.obj")
    texture_offset = len(obj.vertex_index) * 12
    normal_offset = (texture_offset + len(obj.texture_index) * 8)
    shader = compile_shader("mvs.vs", "mvs.fs")
    glBindBuffer(GL_ARRAY_BUFFER, glGenBuffers(1))
    glBufferData(GL_ARRAY_BUFFER, obj.model.itemsize * len(obj.model), obj.model, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, obj.model.itemsize * 3, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, obj.model.itemsize * 2, ctypes.c_void_p(texture_offset))
    glEnableVertexAttribArray(1)
    # normals
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, obj.model.itemsize * 3, ctypes.c_void_p(normal_offset))
    glEnableVertexAttribArray(2)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load image
    image = Image.open("res/african_head_diffuse.tga")
    flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = numpy.array(list(flipped_image.getdata()), numpy.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glEnable(GL_TEXTURE_2D)

    glUseProgram(shader)

    glClearColor(0.2, 0.3, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)
    view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -3.0]))
    projection = pyrr.matrix44.create_perspective_projection_matrix(65.0, w_width / w_height, 0.1, 100.0)
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))

    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    position = None
    position1 = 0
    position2 = 0
    clef = -0.05

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if glfw.get_mouse_button(window, 1) == 1:
            if position is None:
                if position2 is not None:
                    position = clef * glfw.get_cursor_pos(window)[0] - position2
                else:
                    position = clef * glfw.get_cursor_pos(window)[0]
                position1 = position
            rot_y = pyrr.Matrix44.from_y_rotation(position1 - position)
            position1 = clef * glfw.get_cursor_pos(window)[0]
            position2 = None
        else:
            if position2 is None:
                position2 = position1 - position
            rot_y = pyrr.Matrix44.from_y_rotation(position2)
            position = None

        transform_loc = glGetUniformLocation(shader, "transform")
        light_loc = glGetUniformLocation(shader, "light")

        glUniformMatrix4fv(transform_loc, 1, GL_FALSE, rot_y)
        glUniformMatrix4fv(light_loc, 1, GL_FALSE, rot_y)

        glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))

        glfw.swap_buffers(window)

    glfw.terminate()


main()
