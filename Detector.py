import cv2 
import numpy as np
import time 
import os 
import cards
import LiveStream 

## Camera Settings 
im_width = 1280 
im_height = 720
frame_rate = 10 

frame_rate_calc = 1 
freq = cv2.getTickFrequency()

font = cv2.FONT_HERSHEY_SIMPLEX

livestream = LiveStream.LiveStream(im_width, im_height, frame_rate)
time.sleep(1)

path = os.path.dirname(os.path.abspath(__file__))
train_ranks = cards.load_ranks( path + '/Card_Imgs/')
train_suits = cards.load_suits( path + '/Card_Imgs/')

cam_quit = 0

while cam_quit == 0:
     image = livestream.get_frame()
     t1 = cv2.getTickCount()
     pre_proc = cards.preprocess_image(image)

     conts_sort, cont_is_card = cards.find_cards(pre_proc)

     if len(conts_sort) != 0:
          cards = []
          k = 0

          for i in range(len(conts_sort)):
               cards.append(cards.preprocess_card(conts_sort[i], image))

               cards[k].best_rank, cards[k].best_suit, cards[k].rank_diff, cards[k].suit_diff = cards.match_card(cards[k], train_ranks, train_suits)

               image = cards.draw_results(image, cards[k])
               k = k + 1

          if (len(cards) != 0):
               temp_conts = []
               for i in range(len(cards)):
                    temp_conts.append(cards[i].contour)
               cv2.drawContours(image, temp_conts, -1, (255, 0, 0), 2)

cv2.putText(image,"FPS: "+str(int(frame_rate_calc)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)


cv2.imshow("Card Detector",image)


t2 = cv2.getTickCount()
time1 = (t2-t1)/freq
frame_rate_calc = 1/time1
    
  
key = cv2.waitKey(1) & 0xFF
if key == ord("q"):
  cam_quit = 1
        

cv2.destroyAllWindows()
LiveStream.stop()
          
              
            