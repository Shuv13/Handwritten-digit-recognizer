import tkinter as tk
from PIL import ImageGrab, ImageOps, Image
import numpy as np
from keras.models import load_model

MODEL_PATH = "mnist.h5"
CANVAS_SIZE = 300
BRUSH_RADIUS = 8

model = load_model(MODEL_PATH)

def prepare(img: Image.Image) -> np.ndarray:
    img = img.resize((28,28)).convert("L")
    img = ImageOps.invert(img)
    arr = np.array(img).reshape(1,28,28,1) / 255.0
    return arr

class DigitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.label = tk.Label(self, text="Draw", font=("Helvetica", 48))
        btn_recog = tk.Button(self, text="Recognise", command=self.classify)
        btn_clear = tk.Button(self, text="Clear", command=self.clear)
        self.canvas.grid(row=0, column=0, rowspan=2)
        self.label.grid(row=0, column=1)
        btn_recog.grid(row=1, column=1, sticky="n")
        btn_clear.grid(row=2, column=0, sticky="s")
        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")

    def clear(self):
        self.canvas.delete("all")
        self.label.config(text="Draw")

    def classify(self):
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + CANVAS_SIZE
        y1 = y + CANVAS_SIZE
        img = ImageGrab.grab(bbox=(x,y,x1,y1))
        pred = model.predict(prepare(img))[0]
        digit = np.argmax(pred)
        conf = np.max(pred) * 100
        self.label.config(text=f"{digit}, {conf:.0f}%")

if __name__ == "__main__":
    DigitApp().mainloop()
