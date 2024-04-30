import cv2
import socket
import struct
import numpy as np
import threading

# Set up multicast socket
MCAST_GRP = '224.0.0.1'
MCAST_PORT = 5007
BUFFER_SIZE = 65507  # Maximum UDP packet size

# Function to decode and display frames
def receive_frames():
    global sock
    global desired_id

    try:
        while True:
            data, _ = sock.recvfrom(65535)
            # Split the received data into ID and frame data
            parts = data.split(b'_', 1)
            if len(parts) == 2 and parts[0].decode() == desired_id:
                # Decode JPEG image data
                frame_gray = cv2.imdecode(np.frombuffer(parts[1], dtype=np.uint8), cv2.IMREAD_COLOR)
                # Convert grayscale frame to RGB
                frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stream stopped.")
    finally:
        sock.close()
        cv2.destroyAllWindows()

# Create the socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', MCAST_PORT))
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

# ID to filter frames
desired_id = "12345"  # Example ID to filter for

# Start receiving frames in a separate thread
receive_thread = threading.Thread(target=receive_frames)
receive_thread.start()
