import cv2
import socket
import struct

# Set up multicast socket
MCAST_GRP = '224.0.0.1'
MCAST_PORT = 5007
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
ttl = struct.pack('b', 1)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

# Video capture setup
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# ID to be sent along with the frames
sender_id = "12345"  # Example ID, you can change this to whatever ID you want

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Downscale the frame
            resized_frame = cv2.resize(gray_frame, (320,320))
            _, buffer = cv2.imencode('.jpg', gray_frame)
            # Concatenate sender ID and frame data
            data_to_send = sender_id.encode() + b'_' + buffer.tobytes()
            sock.sendto(data_to_send, (MCAST_GRP, MCAST_PORT))
        else:
            break
except KeyboardInterrupt:
    print("Stream stopped.")
finally:
    cap.release()
    sock.close()
