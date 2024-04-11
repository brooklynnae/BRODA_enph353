#! /usr/bin/env python3

import cv2
import numpy as np


def cropToBlue(img):
    """!
    @brief      Crops image of robot perspective to white clue sign

    @param      img: image to cropped

    @return     cropped_img: cropped and perspective transformed image
    """

    height, width = img.shape[:2]
    lower_hsv = (5,20,0)
    upper_hsv = (150,255,255)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sign_mask1 = cv2.inRange(gray_img, 95, 105)
    sign_mask2 = cv2.inRange(gray_img, 195, 205)
    sign_mask3 = cv2.inRange(gray_img, 115, 125)
    sign_mask = cv2.bitwise_or(sign_mask1, sign_mask2)
    sign_mask = cv2.bitwise_or(sign_mask, sign_mask3)

    mask_not = cv2.bitwise_not(mask)
    combined_mask = cv2.bitwise_and(mask_not, sign_mask)

    contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    epsilon = 0.03 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    corners = [point[0] for point in approx_polygon]
    midpoint = int(len(corners)/2)
    sorted_corner_points = sorted(corners, key=lambda point: point[0])
    left = sorted(sorted_corner_points[:midpoint], key=lambda point: point[1])
    right = sorted(sorted_corner_points[midpoint:], key=lambda point: point[1], reverse=True)

    upperLeft = max((pt for pt in left), key=lambda p: p[1])
    lowerLeft = min((pt for pt in left), key=lambda p: p[1])
    upperRight = max((pt for pt in right), key=lambda p: p[1])
    lowerRight = min((pt for pt in right), key=lambda p: p[1])

    src_pts = np.array([lowerLeft, upperLeft, lowerRight, upperRight], dtype=np.float32)
    dst_pts = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped_img = cv2.warpPerspective(img, M, (width, height))

    return cropped_img

def cropToWord(img):
  """!
    @brief      Crops image of clue sign to individual top and bottom words of sign

    @param      img: image to cropped

    @return     croppedWords: top and bottom words of sign
    """
  Hstart, Wstart = img.shape[:2]
  words = []
  buff = int(0.01*Wstart)
  words.append(img[buff:int(Hstart/2), buff:Wstart-buff])
  words.append(img[int(Hstart/2):Hstart-buff, buff:Wstart-buff])
  croppedWords = []
  for word in words:
    h_, w_ = word.shape[:2]
    lower_hsv = (5,20,0)
    upper_hsv = (150,255,255)
    hsv_img = cv2.cvtColor(word, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    lim = 0
    v[v > lim] = 255
    v[v <= lim] += 255
    final_hsv = cv2.merge((h, s, v))
    mask = cv2.inRange(final_hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    startX = w_
    startY = h_
    endX = 0
    endY = 0
    for cnt in contours:
      box = cv2.minAreaRect(cnt)
      points = cv2.boxPoints(box)
      for p in points:
        if p[0] <= startX:
          startX = p[0]
        if p[0] >= endX:
          endX = p[0]
        if p[1] <= startY:
          startY = p[1]
        if p[1] >= endY:
          endY = p[1]
      '''
      if x <= startX:
        startX = x
      if w+x >= endX:
        endX = w+x
      if y <= startY:
        startY = y
      if h+y >= endY:
        endY = h+y
        '''    
    cropped = word[int(round(startY)):int(round(endY)), int(round(startX)):int(round(endX))]
    h, w = cropped.shape[:2]
    ratio = w/h
    newY = 90
    newX = int(newY*ratio)
    cropped = cv2.resize(cropped, (newX,newY),  interpolation= cv2.INTER_LINEAR)
    croppedWords.append(cropped)
  return croppedWords

def wordToLetters(word):
  """!
    @brief      Crops individual words into each character in the word

    @param      word: image of word to cropped

    @return     letters: cropped and scaled images of letters
    """
  cv2.imshow("word", word)
  cv2.waitKey(1)
  letters = []
  h_, w_ = word.shape[:2]
  gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
  blur = cv2.blur(gray, (9,9))
  _, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
  kernel = np.ones((5, 5), np.uint8)
  dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
  erosion = cv2.erode(dilation, kernel, iterations = 1)
  cv2.imshow("mask", erosion)
  cv2.waitKey(1)
  contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  threshArea = 500
  possibleLetters = []
  letters = []
  wAvg = 0
  hAvg = 0
  wSafe = h_*8/9
  nums = 0
  for c in contours:
    if cv2.contourArea(c) > threshArea:
      x, y, w, h = cv2.boundingRect(c)
      letter = word[y:y+h, x:x+w]
      if abs(w - wSafe) < 30 and abs(h - h_) < 25:
        nums += 1
        wAvg += w
      if abs(h - h_) < 25:
        possibleLetters.append((letter, w, x))

  if nums != 0:
    wAvg = 1.1*wAvg/(nums)
  possibleLetters = sorted(possibleLetters, key=lambda a: a[2])
  if wAvg == 0:
    wAvg = wSafe
  
  for l in possibleLetters:
    h0, w0 = l[0].shape[:2]
    newW = round(w0/wAvg)
    for i in range(newW):
      letter = l[0][0:h0, i*int(wAvg):(i+1)*int(wAvg)]
      letter = cv2.resize(letter, (60, 90),  interpolation= cv2.INTER_LINEAR)
      gray = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
      _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
      letter = cv2.dilate(thresh1, rect_kernel, iterations = 1)
      erosion = cv2.erode(dilation, kernel, iterations = 1)
      letters.append(letter)
  '''for i in range(newW):
    letter = word[0:h_, i*int(wAvg):(i+1)*int(wAvg)]
    letter = cv2.resize(letter, (60, 90),  interpolation= cv2.INTER_LINEAR)
    gray = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    letter = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    letters.append(letter)'''
  

  return letters

'''height, width = cropped.shape[:2]
    src_pts = np.array([lowerLeft, upperLeft, lowerRight, upperRight], dtype=np.float32)
    dst_pts = np.array([[0, 0], [0, h_], [w_, 0], [w_, h_]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped_img = cv2.warpPerspective(word, M, (w_, h_))'''

def signToLetters(sign):
  """!
    @brief      Crops sign into each character in the words present on the sign

    @param      sign: image of sign to cropped

    @return     category: cropped and scaled images of letters in category
                clue: cropped and scaled images of letters in clue
    """
  img = cv2.resize(sign, (600,400))

  yRange = (250, 340)
  startX = 30
  endX = 75
  inc = 45
  numChar = 12

  #clue = []

  
  words = cropToWord(sign)
  category = wordToLetters(words[0])
  clue = wordToLetters(words[1])
  cv2.imshow("clue", np.concatenate(clue, axis=0))
  cv2.waitKey(1)
  return np.array(clue)
'''
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  kernel = np.ones((5, 5), np.uint8)

  for i in range(numChar):
    letter = img[yRange[0]:yRange[1], startX:endX]
    letter = cv2.resize(letter, (60, 90),  interpolation= cv2.INTER_LINEAR)
    gray = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    clue.append(erosion)
    startX += inc
    endX += inc
  cv2.imshow("clue", np.concatenate(clue, axis=0))
  cv2.waitKey(1)
  return np.array(clue)

  '''
  