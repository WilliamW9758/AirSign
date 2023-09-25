import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import imageio.v3 as iio
import imageio
from PIL import Image

import cv2
import numpy
import time

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

model_path = 'signatureDetector.task'

#live stream option
# limit to one hand

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def smoothing(points, num=3):
    if len(points) <= num:
        return points

    newPoints = []
    for i in range(num//2, len(points) - num//2):
        newPoints.append([(np.sum([x[0] for x in points[i-num//2:i+num//2]]))//num,
                          (np.sum([y[1] for y in points[i-num//2:i+num//2]]))//num])

    return newPoints

img_buffer = []
sig_buffer = []
past_states = []
is_writing = False

def saveGIF(img_height, img_width):
    image_frames = []
    smoothed = []
    global sig_buffer
    for line in sig_buffer:
        line = smoothing(line, 5)
        smoothed.append(line)
    for i in range(len(smoothed)):
        if len(smoothed[i]) > 1:
            for j in range(len(smoothed[i])):
                img = np.ones([img_height, img_width, 3]) * 255
                for k in range(i):
                    pts = np.array(smoothed[k]).reshape((-1, 1, 2))
                    img = cv2.polylines(img, [pts], False, (0, 0, 0), 10)
                pts = np.array(smoothed[i][:j]).reshape((-1, 1, 2))
                img = cv2.polylines(img, [pts], False, (0, 0, 0), 10)
                img = cv2.resize(img, dsize=(img_width // 4, img_height // 4), interpolation=cv2.INTER_CUBIC)
                image_frames.append(Image.fromarray(img.astype(np.uint8)))
    # print("start saving")
    image_frames[0].save("signature.gif", append_images=image_frames[1:],
                         save_all=True, optimize=True, duration=40, loop=1)
    # print("finished saving")

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global is_writing
    global sig_buffer
    if result and result.gestures:
        # print('gesture recognition result: {}'.format(result.gestures[0][0].category_name))
        if result.gestures[0][0].category_name == 'sign':
            past_states.append('sign')
            if not is_writing:
                is_writing = True
                sig_buffer.append([])
            sig_buffer[-1].append([int(result.hand_landmarks[0][8].x * output_image.numpy_view().shape[1]), \
                int(result.hand_landmarks[0][8].y * output_image.numpy_view().shape[0])])

        elif result.gestures[0][0].category_name == 'reset':
            past_states.append('reset')
            is_writing = False
            if len(past_states) > 20 and all(state == 'reset' for state in past_states[-20:]):
                sig_buffer = []

        elif result.gestures[0][0].category_name == 'save':
            past_states.append('save')
            # print(output_image.width)
            is_writing = False
            if len(past_states) > 20 and all(state == 'save' for state in past_states[-20:]):
                cv2.destroyAllWindows()
                img = np.ones([output_image.height, output_image.width, 3]) * 255
                for line in sig_buffer:
                    line = smoothing(line, 5)
                    if len(line) > 1:
                        pts = np.array(line).reshape((-1, 1, 2))
                        img = cv2.polylines(img, [pts], False, (0, 0, 0), 10)
                cv2.imwrite("signature.png", img)
                saveGIF(output_image.height, output_image.width)
                exit(0)

        else:
            past_states.append('none')
            is_writing = False
                
        img_buffer.append(draw_landmarks_on_image(output_image.numpy_view(), result))

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  gesture_list = detection_result.gestures
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]
    gesture = gesture_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{gesture[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with GestureRecognizer.create_from_options(options) as recognizer:
  # The detector is initialized. Use it here.
  # ...

    #capture video from camera
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    start_time = time.time()
    # write a milisecond counter that increments frame_timestamp_ms to reflect how long the video has been running

    while(True):
        #capture frame by frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognition_result = recognizer.recognize_async(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 1
        
        if len(img_buffer) > 0:
            frame = img_buffer.pop(0)
    
        for line in sig_buffer:
            if len(line) > 1:
                pts = np.array(line).reshape((-1,1,2))
                frame = cv2.polylines(frame, [pts], False,(0, 0, 255), 10)
            
        cv2.imshow('Sign', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #release capture
    cap.release()
    cv2.destroyAllWindows()