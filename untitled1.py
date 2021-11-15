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
    image_o = image.copy()
    img_test = image.copy()

    hand_cnt = 0
    for hand_landmarks in results.multi_hand_landmarks:

        # detecting finter_tips
        finger_tip_coord = []
        finger_dip_coord = []
        for i in finger_dip:
            itx_ = hand_landmarks.landmark[i + 1].x * image_width
            ity_ = hand_landmarks.landmark[i + 1].y * image_height
            idx_ = hand_landmarks.landmark[i].x * image_width
            idy_ = hand_landmarks.landmark[i].y * image_height
            '''
            color = ( 255,0,0)
            img_test = cv2.line(img_test,(int(idx_),int(idy_)),(int(itx_),int(ity_)),color,5)
            '''
            # detection finger_fold
            finger_tip_coord.append([itx_, ity_])
            finger_dip_coord.append([idx_, idy_])

        # detectng palm
        palm_point_coordinate = [];
        for i in palm_point:
            temp = []
            temp.append(hand_landmarks.landmark[i].x * image_width)
            temp.append(hand_landmarks.landmark[i].y * image_height)
            palm_point_coordinate.append(temp)

        hand_ = MessageToDict(results.multi_handedness[hand_cnt])['classification'][0]['index']

        hand_cnt += 1
        print(palm_point_coordinate)

        front = isPalm(np.array(palm_point_coordinate), hand_)
        '''
        차후 손바닥 여부를 판단하여 해당 손을 blur처리 할 것인지 결정한다. 
        또한 손가락의 접힘 정도를 판단하여 blur처리를 하지 않아야할 손가락 또한 구별할 것이다. 
        '''
        if (front == 1):  # 손바닥 앞면 구분
            color = (255, 102, 165)  # 손바닥인경우
            finger_folded = foldedFinger(np.array(finger_tip_coord), np.array(finger_dip_coord),
                                         np.array(palm_point_coordinate[0]))

            finger_folded[0] = thumb_folded(np.array(palm_point_coordinate), hand_, np.array(finger_tip_coord[0]),
                                            np.array(finger_dip_coord[0]))
            print(finger_folded)

            # 테스트용 이미지 그림
            '''
            img_test = cv2.line(img_test, (int(palm_point_coordinate[0][0]), int(palm_point_coordinate[0][1])),(int(palm_point_coordinate[1][0]), int(palm_point_coordinate[1][1])), color, 5)
            img_test = cv2.line(img_test, (int(palm_point_coordinate[2][0]), int(palm_point_coordinate[2][1])),(int(palm_point_coordinate[1][0]), int(palm_point_coordinate[1][1])), color, 5)
            img_test = cv2.line(img_test, (int(palm_point_coordinate[3][0]), int(palm_point_coordinate[3][1])),(int(palm_point_coordinate[0][0]), int(palm_point_coordinate[0][1])), color, 5)
            img_test = cv2.line(img_test, (int(palm_point_coordinate[3][0]), int(palm_point_coordinate[3][1])),(int(palm_point_coordinate[2][0]), int(palm_point_coordinate[2][1])), color, 5)
            '''
            temp_tip = np.array(finger_tip_coord, dtype=int)
            temp_dip = np.array(finger_dip_coord, dtype=int)
            # 손가락 굽힘 확인

            for i, ff in enumerate(finger_folded):
                if (ff):  # 손가락 굽힘
                    color = (0, 153, 0)
                else:  # 안굽힘
                    color = (0, 0, 0)
                    img = draw_ellipse(img_test, (temp_tip[i][0], temp_tip[i][1]), (temp_dip[i][0], temp_dip[i][1]),
                                       color)

            final_img = finger_print_blur(image_o, img)

            # color = (0,0,0)

            # img_test = cv2.line(img_test, (temp_tip[i][0], temp_tip[i][1]), (temp_dip[i][0], temp_dip[i][1]), color, 5)

        '''
        cv2.imshow('test',img_test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        '''
    '''
    cv2.imwrite(
        './' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    '''
    cv2.imwrite(
        './img_res/' + str(idx) + '_final' + '.png', cv2.flip(final_img, 1))
    cv2.imwrite(
        './img_res/' + str(idx) + '_ellipes' + '.png', cv2.flip(img, 1))