import customtkinter

from pictureframe import PictureFrame

class PictureSpace(customtkinter.CTkScrollableFrame):
    def __init__(self, master,labelText:str ,**kwargs):
        super().__init__(master,
            border_width=2,
            width= 750,
            height= 300,
            label_text= labelText,
            label_anchor="w",
            label_font=("Consolas",15,'bold'),
            fg_color="#FFEFD5",
            orientation="horizontal",
            **kwargs)
        
        titles: list[str] = [ "ban đầu","grayscale","lọc nhiễu","tìm góc","vẽ contours","cắt ảnh","lời giải","góc nhìn","kết quả"]
        self.imageList: list[PictureFrame] = [ ]
        
        for i in range(9):
            pf = PictureFrame(self,titleImage=titles[i])
            # pf.pack()
        # add widgets onto the frame...
            self.imageList.append( pf )
    
    def changePictureAt(self,index:int,imgArray):
        self.imageList[index].changeImage(imgArray)
        self.imageList[index].pack(side='left',padx=50)
    def clearAllPicture(self):
        for i in range(9):
            self.imageList[i].pack_forget()

    
        