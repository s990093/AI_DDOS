import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os



def load_images_from_directory(directory):
    """Load all images from a given directory."""
    image_files = []
    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    for file in os.listdir(directory):
        if file.lower().endswith(supported_formats):
            image_files.append(os.path.join(directory, file))
    return image_files


class DraggableImage(tk.Canvas):
    def __init__(self, parent, img, original_img, **kwargs):
        super().__init__(parent, width=img.width(), height=img.height(), **kwargs)
        self.parent = parent
        self.img = img
        self.original_img = original_img
        self.current_scale = 1.0  # Track current scale factor
        self.current_angle = 0  # Track current rotation angle
        self.image_id = self.create_image(0, 0, anchor="nw", image=self.img)
        self.bind("<Button-1>", self.start_drag)

    def start_drag(self, event):
        self.parent.selected_image = self  # Mark this image as selected
        self._drag_data = {"x": event.x, "y": event.y}
        self.bind("<B1-Motion>", self.do_drag)

    def do_drag(self, event):
        x = self.winfo_x() - self._drag_data["x"] + event.x
        y = self.winfo_y() - self._drag_data["y"] + event.y
        self.place(x=x, y=y)

    def resize_image(self, scale_factor):
        """Resize the image dynamically based on scale factor."""
        self.current_scale *= scale_factor
        new_width = int(self.original_img.width * self.current_scale)
        new_height = int(self.original_img.height * self.current_scale)
        resized_img = self.original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.img = ImageTk.PhotoImage(resized_img)
        self.configure(width=new_width, height=new_height)
        self.itemconfig(self.image_id, image=self.img)

    def rotate_image(self, angle):
        """Rotate the image by a given angle."""
        self.current_angle += angle
        rotated_img = self.original_img.rotate(self.current_angle, resample=Image.Resampling.BICUBIC, expand=True)
        resized_img = rotated_img.resize(
            (int(rotated_img.width * self.current_scale), int(rotated_img.height * self.current_scale)),
            Image.Resampling.LANCZOS
        )
        self.img = ImageTk.PhotoImage(resized_img)
        self.configure(width=resized_img.width, height=resized_img.height)
        self.itemconfig(self.image_id, image=self.img)

    def mirror_image(self):
        """Mirror the image horizontally."""
        mirrored_img = self.original_img.transpose(Image.FLIP_LEFT_RIGHT)
        resized_img = mirrored_img.resize(
            (int(mirrored_img.width * self.current_scale), int(mirrored_img.height * self.current_scale)),
            Image.Resampling.LANCZOS
        )
        self.img = ImageTk.PhotoImage(resized_img)
        self.configure(width=resized_img.width, height=resized_img.height)
        self.itemconfig(self.image_id, image=self.img)


def main():
    root = tk.Tk()
    root.title("Image Loader and Viewer")
    root.geometry("800x600")
    root.configure(bg="black")
    root.selected_image = None  # Track currently selected image

    def load_directory():
        # directory = "/Users/hungwei/Downloads/untitled folder 2"  # You can use filedialog.askdirectory() for selection
        directory =  filedialog.askdirectory() # You can use filedialog.askdirectory() for selection
        if directory:
            images = load_images_from_directory(directory)
            for img_path in images:
                try:
                    img = Image.open(img_path)
                    img = img.convert("RGBA") 
                    original_img = img.copy()  
                    img.thumbnail((300, 300)) 

                    tk_img = ImageTk.PhotoImage(img)
                    draggable_image = DraggableImage(root, tk_img, original_img, bg="black", highlightthickness=0)
                    draggable_image.place(x=50, y=50)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    def resize_selected_image(event):
        """Resize or rotate the currently selected image using arrow keys."""
        if root.selected_image:
            if event.keysym == "Up":  # Increase size
                root.selected_image.resize_image(1.1)
            elif event.keysym == "Down":  # Decrease size
                root.selected_image.resize_image(0.9)
            elif event.keysym == "Left":  # Rotate counterclockwise
                root.selected_image.rotate_image(-10)
            elif event.keysym == "Right":  # Rotate clockwise
                root.selected_image.rotate_image(10)
            elif event.keysym == "m":  # Mirror the image
                root.selected_image.mirror_image()

    load_button = tk.Button(root, text="Load Images", command=load_directory)
    load_button.pack(pady=10)

    root.bind("<Key>", resize_selected_image)

    root.mainloop()


if __name__ == "__main__":
    main()