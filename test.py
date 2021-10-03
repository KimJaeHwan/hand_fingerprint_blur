# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:27:12 2021

@author: rfkjh
"""

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def comp_int(num1, num2):
    if (num1 < num2):
        return int(num1), int(num2)
    else :
        return int(num2), int(num1)
    
# For static images:
IMAGE_FILES = ["./hand2_img.jpg"]
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(type(mp_hands.HandLandmark.INDEX_FINGER_TIP))
      print(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      print(
          f'index finger dip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height})'
      )
      
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height})'
      )
      print(
          f'index finger dip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height})'
      )
      img_test = image.copy()
      idx_ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
      idy_ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
      itx_ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
      ity_ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
      color = ( 255,0,0)
      '''
      if(idx_ < itx_):
          x1 = int(idx_)
          x2 = int(itx_)
      else:
          x1 = int(itx_)
          x2 = int(idx_)
      
      if(idy_ < ity_):
          y1 = int(idy_)
          y2 = int(ity_)
      else:
          y1 = int(ity_)
          y2 = int(idy_)
      '''
      
      
      img_test = cv2.line(img_test,(int(idx_),int(idy_)),(int(itx_),int(ity_)),color,5)
     
      
      cv2.imshow('test',img_test)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      
      
    cv2.imwrite(
        './' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    cv2.imwrite(
        './' + str(idx)+'_' + '.png', cv2.flip(img_test, 1))
    
    