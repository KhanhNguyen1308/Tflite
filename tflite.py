import cv2
import time
from PIL import Image
import tensorflow as tf
import numpy as np
import tflite_runtime.interpreter as tflite
INPUT_IMAGE_URL = "6_class/face2.jpg"
DETECTION_THRESHOLD = 0.5

# Load the TFLite model
interpreter = tflite.Interpreter("model.tflite")
interpreter.allocate_tensors()
cap = cv2.VideoCapture("aespa.mp4")
def preprocess_image(img, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  original_image = img
  resized_img = cv2.resize(img, input_size)
  print(resized_img)
  resized_img = resized_img[tf.newaxis, :]
  print(resized_img)
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results
def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path, 
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute 
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])
    # Draw the bounding box and label on the image
    color = (255, 0, 255)
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format('person', obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    original_image_np=cv2.resize(original_image_np, dsize=None, fx=0.7, fy=0.7)
  return(original_image_np)

# Run inference and draw detection result on the local copy of the original file
while True:
    ret, img = cap.read()
    start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detection_result_image = run_odt_and_draw_results(img, interpreter, threshold=DETECTION_THRESHOLD)
    end = time.time()
    fps = int(1/(end - start))
    cv2.putText(detection_result_image, "FPS: "+str(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    img = cv2.cvtColor(detection_result_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("result", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


