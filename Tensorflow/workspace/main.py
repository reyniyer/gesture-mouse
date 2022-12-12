import sys
import cv2 
import numpy as np
import os
import tensorflow as tf
import collections
import six
import pyautogui as pygui
import time
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


WORKSPACE_PATH = '.'
if hasattr(sys, '_MEIPASS'):
    WORKSPACE_PATH = sys._MEIPASS

ANNOTATION_PATH = os.path.join(WORKSPACE_PATH, 'annotations')
IMAGE_PATH = os.path.join(WORKSPACE_PATH, 'images')
MODEL_PATH = os.path.join(WORKSPACE_PATH, 'models')
PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH, 'pre-trained-models')
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
CONFIG_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'pipeline.config')
CHECKPOINT_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)
LABEL_PATH = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')


configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-8')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

category_index = label_map_util.create_category_index_from_labelmap(LABEL_PATH)


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]



def override_visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4):
    
    
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
  
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    box = ()
    class_name = ''
    
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
                        
            display_str = ''
            if not agnostic_mode:
                if classes[i] in six.viewkeys(category_index):
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
            display_str = '{}: {}%'.format(str(class_name), round(100*scores[i]))
            box_to_display_str_map[box].append(display_str)
            
            if agnostic_mode:
                box_to_color_map[box] = 'DarkOrange'
            else:
                box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]

        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            
            viz_utils.draw_bounding_box_on_image_array(
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates
            )
        
        return box, class_name


prev_time = time.time()
duraion = 0
UI_THRESHOLD = 0.2
is_ui_wait = False
last_action = None
while True: 
    current_time = time.time()
    delta_time = current_time - prev_time
    duraion += delta_time
    if duraion > UI_THRESHOLD:
        duration = 0
        is_ui_wait = False


    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    box, class_name = override_visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.5,
        agnostic_mode=False,
    )

    im_width, im_height = list(pygui.size())

    if len(box) != 0:
        ymin, xmin, ymax, xmax = box
        print(ymin, xmin, ymax, xmax)
        left, right, top, bottom = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)


        if not is_ui_wait:
            if(class_name == 'single' and last_action != 'left'):
                pygui.click(button='left')
                last_action = 'left'

            if(class_name == 'double' and last_action != 'right'):
                pygui.click(button='right')
                last_action = 'right'

            if(class_name == 'palm'):
                pygui.moveTo(round(right), round(top))

            is_ui_wait = True


    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

    prev_time = current_time
