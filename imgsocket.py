import socket
import numpy as np
import cv2

from imageprocess import GetImageForEachStep

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

def process_image(image_data):
    """Process the received image data."""
    decoded_image = None
    try:
        # Convert image data to numpy array
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        # Decode the image (assuming it's in a valid format)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Check if the decoding was successful
        if decoded_image is None:
            print("Failed to decode the image.")
    except Exception as e:
        print(f"Error while decoding image: {e}")
    return decoded_image


def main():
    try:
        # Get the IP address of the server machine
        server_ip = get_ip_address()
        print("Server IP address:", server_ip)
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((server_ip, 27015))
        server_socket.listen(1)
        client_socket, addr = server_socket.accept()
        print("Connection from", addr)
        while True:
            # Receive image data size from client
            size_bytes = client_socket.recv(4)
            image_size = int.from_bytes(size_bytes, byteorder='big')
            # Receive image data from client
            image_data = b''
            while len(image_data) < image_size:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                image_data += chunk
            # Process received image data
            processed_image = process_image(image_data)
            processed_images = GetImageForEachStep(processed_image)
            # Send processed images back to client
            for processed_image in processed_images:
                processed_image_data = None
                # Convert processed image (numpy array) to JPEG bytes
                ret, processed_image_data = cv2.imencode('.jpg', processed_image)
                if not ret:
                    print("Failed to encode processed image.")
                    continue
                # Send the size of the image first
                size_bytes = len(processed_image_data).to_bytes(4, byteorder='big', signed=True)
                print('size byte', len(processed_image_data))
                client_socket.send(size_bytes)
                # Send the image data
                client_socket.sendall(processed_image_data)
            client_socket.send(int(0).to_bytes(4, byteorder='big', signed=True))
    except KeyboardInterrupt:
        print("Server stopped by user.")
        client_socket.close()
        server_socket.close()

if __name__ == "__main__":
    main()
