import customtkinter

from workspace import Workspace

class ClientArea(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master,fg_color="#00FA9A", **kwargs)
        self.workspaces:list[Workspace] = []
        self.place(x=4,y=75)
    def addWorkspace(self,clientIP:str) -> Workspace:
        for ws in self.workspaces:
            if( ws.activate == False):
                ws.destroy()
                self.workspaces.remove(ws)
                break

        newWorkspace:Workspace = Workspace(self,
            labelText   = clientIP
        )
        newWorkspace.pack(pady= 20)
        newWorkspace.logging("[+] Client joined server")
        self.workspaces.append( newWorkspace )
        return newWorkspace
       