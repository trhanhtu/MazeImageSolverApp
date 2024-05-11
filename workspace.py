import customtkinter

from picturespace import PictureSpace

class Workspace(customtkinter.CTkFrame):
    def __init__(self, master,labelText:str,active=True,**kwargs):
        super().__init__(master,fg_color="#00FA9A",**kwargs)
        self.activate = active
        self.picturespace:PictureSpace = PictureSpace(self,labelText=labelText)
        self.picturespace.pack(side='left')
        self.logspace: customtkinter.CTkTextbox = customtkinter.CTkTextbox(self,
            width=300,
            height=380,
            border_width=2,
            fg_color="#FFEFD5",
            font=("Consolas",15),
            state="normal",
            wrap='word',
            text_color="black"
        )  
        self.logspace.pack(side='right',padx=30)

    def changePictureAt(self,index:int,imgArray):
        self.picturespace.changePictureAt(index=index,imgArray=imgArray)
    def clearAllPicture(self):
        self.picturespace.clearAllPicture()
        
    def logging(self,message:str):
        # Ensure the textbox is enabled before inserting text
        self.logspace.configure(state="normal")
        # self.logspace.insert('end', message + '\n')
        self.logspace.insert('0.0', message + '\n')
        self.logspace.configure(state="disabled")  # Disable the textbox after inserting text