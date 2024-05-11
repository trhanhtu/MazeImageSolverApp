import io
import customtkinter
from PIL import Image

class PictureFrame(customtkinter.CTkFrame):
    def __init__(self, master, titleImage: str, **kwargs):
        super().__init__(master, fg_color="#FFEFD5", **kwargs)
        
        customtkinter.CTkLabel(self,
            fg_color="#483434",
            text_color="#FFF3E4",
            text=titleImage,
            font=("Consolas", 15, 'bold'),
            corner_radius=10
        ).pack()

        self.image_label = customtkinter.CTkLabel(self, fg_color="#B5C18E",text="")
        self.image_label.pack()


    def changeImage(self, byte_buffer):
        image = customtkinter.CTkImage(
            light_image=Image.open(io.BytesIO(byte_buffer)),
            size=(200, 290)
        )
        # Update the image displayed in the label
        self.image_label.configure(image=image)
        