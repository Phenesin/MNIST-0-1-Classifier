import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
from .canvas_utils import (draw, clear_canvas, create_drawing_surface, reset_brush)
from .inference import (predict, DEVICE)
from .preprocess import (preprocess_image)



root = tk.Tk()
root.title("MNIST 0 vs 1 Classifier")

root.geometry("500x800")
root.configure(bg = "#1e1e1e")


canvas = tk.Canvas(
    root, 
    width = 280,
    height = 280,
    bg = "black",
    highlightthickness = 2,
    highlightbackground = "#444"
)
canvas.pack(pady = 20)

processed_preview = tk.Label(
    root,
    bg = "#1e1e1e"
)
processed_preview.pack(pady = 10)

image, draw_object = create_drawing_surface()

def update_prediction():
    tensor, processed_image  = preprocess_image(
        image,
        DEVICE
    )
    probability_0, probability_1 = predict(tensor)
    predicted_digit = (
        "0"
        if probability_0 > probability_1
        else "1"
    )
    confidence = max(probability_0, probability_1) * 100
    zero_bar["value"] = probability_0 * 100
    one_bar["value"] = probability_1 * 100

    preview_image = processed_image.resize((140, 140))
    preview_photo = ImageTk.PhotoImage(preview_image)
    processed_preview.config(
        image = preview_photo
    )
    processed_preview.image = preview_photo


    prediction_label.config(
        text = (
            f"Prediction : {predicted_digit}\n"
            f"Confidence : {confidence:.1f}%"
        )
    )

def reset_prediction():
    prediction_label.config(
        text = "Draw 0 or 1"
    )
    zero_bar["value"] = 0
    one_bar["value"] = 0

canvas.bind("<B1-Motion>", lambda event :(draw(event, canvas, draw_object), update_prediction()))
canvas.bind("<ButtonRelease-1>", lambda event : reset_brush())

prediction_label = tk.Label(
    root, 
    text = "Draw 0 or 1",
    font = ("Helvetica", 20, "bold"),
    fg = "white",
    bg = "#1e1e1e",
    justify = "center"
)
prediction_label.pack(pady = 10)

zero_text = tk.Label(
    root,
    text = "0 Confidence",
    fg = "white",
    bg = "#1e1e1e"
)
zero_text.pack()

zero_bar = ttk.Progressbar(
    root,
    length = 300,
    maximum = 100
)
zero_bar.pack(pady = 5)


one_text = tk.Label(
    root,
    text = "1 Confidence",
    fg = "white",
    bg = "#1e1e1e"
)
one_text.pack()

one_bar = ttk.Progressbar(
    root,
    length = 300,
    maximum = 100
)
one_bar.pack(pady = 5)

clear_button = tk.Button(
    root,
    text = "Clear",
    font = ("Helvetica", 14, "bold"),
    bg = "#333",
    fg = "white",
    activebackground = "#555",
    activeforeground = "white",
    relief = "flat",
    padx = 20,
    pady = 5,
    command = lambda :(clear_canvas(canvas, image, draw_object), reset_prediction()) 
)

clear_button.pack(pady = 10)
root.mainloop()

