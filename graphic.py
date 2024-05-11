import customtkinter

def createGraphic(serverIp : str) -> customtkinter.CTk:
    root_tk = customtkinter.CTk()
    root_tk.geometry("1200x910")
    root_tk.title("sever maze solver graphic mode")
    root_tk.configure(fg_color="#00FA9A")

    serverIpLabel: customtkinter.CTkLabel = customtkinter.CTkLabel(
        master= root_tk,
        padx=10,
        pady=2,
        fg_color="#87CEEB",
        text= serverIp,
        font=("Consolas",30,'bold')
    )
    memberLabel: customtkinter.CTkLabel = customtkinter.CTkLabel(
        master= root_tk,
        fg_color="#DB7093",
        padx=4,
        pady=5,
        corner_radius=10,
        font=("Consolas",15),
        justify="left",
        text= "21110861 - Trần Hoàng Anh Tú\n21110198 - Trần Vĩnh Hùng\n21110131 - Ngô Hoàng Ân"
    )

    serverIpLabel.place(relx=0.5, y=0)
    memberLabel.place(x=8,y=8)
    return root_tk

def createWorkSpace(clientIp:str):
    pass