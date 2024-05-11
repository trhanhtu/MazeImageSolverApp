from http import client
import socket
import customtkinter
from clientArea import ClientArea
from graphic import createGraphic
from imgsocket import OpenConnectToServer, get_ip_address
import threading


def main():
    
    # Get the IP address of the server machine
    server_ip = get_ip_address()
    print(u"Server IP address:", server_ip)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, 27015))
    server_socket.listen(2)

    graphicApp: customtkinter.CTk = createGraphic(server_ip)
    clientArea:ClientArea = ClientArea(graphicApp)
    threading.Thread(target=OpenConnectToServer,args=(server_socket,clientArea,)).start()
    graphicApp.mainloop()
    
    
if __name__ == "__main__":
    main()