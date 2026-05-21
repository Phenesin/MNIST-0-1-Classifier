from PIL import Image, ImageDraw

def create_drawing_surface():
    image = Image.new(
        "L",
        (280, 280),
        color = 0
    )

    draw_object = ImageDraw.Draw(image)
    return image, draw_object

last_x = None
last_y = None

def draw(event, canvas, draw_object):
    global last_x
    global last_y

    x = event.x
    y = event.y

    if last_x is not None and last_y  is not None:
        canvas.create_line(
            last_x,
            last_y,
            x,
            y,
            fill = "white",
            width = 10,
            capstyle = "round",
            smooth = True
        )
        draw_object.line((
            last_x,
            last_y,
            x,
            y
        ),
        fill = 255,
        width = 10
        )
    last_x = x
    last_y = y

def reset_brush():
    global last_x
    global last_y

    last_x = None
    last_y = None




def clear_canvas(canvas, image, draw_object):
    canvas.delete("all")
    draw_object.rectangle(
        (0,0,280,280),
        fill = 0
    )