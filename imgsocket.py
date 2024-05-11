import socket
import sys
import threading
import traceback
import numpy as np
import cv2

from clientArea import ClientArea
from imageprocess import GetImageForEachStep
from workspace import Workspace

def get_ip_address():
    """Get the IP address of the current machine."""
    try:
        # Create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to any address and port (doesn't need to be reachable)
        s.connect(("8.8.8.8", 80))
        # Get the IP address of the socket (which is the IP address of the machine)
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except socket.error:
        return "Unable to retrieve IP address"

def ProcessImage(image_data,workspace: Workspace):
    """Process the received image data."""
    decoded_image = None
    try:
        # Convert image data to numpy array
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        # Decode the image (assuming it's in a valid format)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Check if the decoding was successful
        if decoded_image is None:
            workspace.logging("[?] Failed to decode the image.")
    except Exception as e:
       workspace.logging(f"[!] Error while decoding image: {e}")
    return decoded_image


def HandleClient(client_socket, workspace: Workspace):
    try:
        while True:
            size_bytes = client_socket.recv(4)
            if not size_bytes:
                break
            image_size = int.from_bytes(size_bytes, byteorder='big')
            image_data = ReceiveImageData(client_socket, image_size)
            if image_data is None:
                continue
            processed_images = ProcessImageData(image_data, workspace)
            if processed_images is None:
                continue
            SendProcessedImages(client_socket, processed_images, workspace)
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        CloseClientSocket(client_socket, workspace)

def ReceiveImageData(client_socket, image_size):
    image_data = b''
    while len(image_data) < image_size:
        chunk = client_socket.recv(4096)
        if not chunk:
            break
        image_data += chunk
    return image_data

def ProcessImageData(image_data, workspace):
    processed_image = ProcessImage(image_data, workspace)
    workspace.clearAllPicture()
    if processed_image is None:
        return None
    return GetImageForEachStep(processed_image, workspace)

def SendProcessedImages(client_socket, processed_images, workspace):
    for index, processed_image in enumerate(processed_images):
        processed_image_data = None
        ret, processed_image_data = cv2.imencode('.jpg', processed_image)
        if not ret:
            workspace.logging("[?] Failed to encode processed image.")
            continue
        workspace.changePictureAt(index, processed_image_data)
        SendImageData(client_socket, processed_image_data)
    SendEndSignal(client_socket)

def SendImageData(client_socket, image_data):
    size_bytes = len(image_data).to_bytes(4, byteorder='big', signed=True)
    client_socket.send(size_bytes)
    client_socket.sendall(image_data)

def SendEndSignal(client_socket):
    client_socket.send(int(0).to_bytes(4, byteorder='big', signed=True))

def CloseClientSocket(client_socket, workspace):
    client_socket.close()
    workspace.logging('[?] Client disconnected')
    workspace.activate = False
        

def OpenConnectToServer(server_socket:socket.socket,clientArea:ClientArea):
    try:
        while True:
            client_socket, addr = server_socket.accept()
            clientWorkspace: Workspace = clientArea.addWorkspace(str(addr))
            # Handle each client connection in a separate thread
            client_thread = threading.Thread(target=HandleClient, args=(client_socket, clientWorkspace,))
            client_thread.start()
    except KeyboardInterrupt:
        print(u"Ctrl+C pressed. Exiting...")

