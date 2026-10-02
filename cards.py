import cv2
import numpy as np 

#Background threshold for card contour detection
#Subtract background from image and threshold to find card contours
bg_threst = 60
card_thresh = 30

# Card corner and suit/rank image sizes
corner_width = 32
corner_height = 84

# Card rank and suit image sizes
rank_width = 70
rank_height = 125
suit_width = 70
suit_height = 100

# max pixel difference for rank and suit images to be considered a match
rank_diff_max = 2000
suit_diff_max = 700

card_max_area = 120000
card_min_area = 25000

card_width, card_height = 200, 300

rank_names = ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']
suit_names = ['Diamonds', 'Hearts', 'Spades', 'Clubs']

font = cv2.FONT_HERSHEY_SIMPLEX

# Data Containers 

class QueryCard: 
    def __init__(self):
        self.contour = []
        self.width, self.height = 0, 0
        self.corner_pts = []
        self.center = []
        self.warp = []
        self.rank_img = []
        self.suit_img = []
        self.best_rank = "Unknown"
        self.best_suit = "Unknown"
        self.rank_diff = 0
        self.suit_diff = 0

class TrainCard:
    def __init__(self, img, name):
        self.img = img
        self.name = name

# Loading Templates 

def load_ranks(filepath):
    train_ranks = []
    for name in suit_names: 
        img = cv2.imread(f"{filepath}{name}.jpg", cv2.IMREAD_GRAYSCALE)
        if img is None: 
            raise FileNotFoundError(f"Rank image for {name} not found at {filepath}{name}.jpg")
        train_ranks.append(TrainCard(img, name))
    return train_ranks

def load_suits(filepath):
    train_suits = []
    for name in suit_names: 
        img = cv2.imread(f"{filepath}{name}.jpg", cv2.IMREAD_GRAYSCALE)
        if img is None: 
            raise FileNotFoundError(f"Suit image for {name} not found at {filepath}{name}.jpg")
        train_suits.append(TrainCard(img, name))
    return train_suits


# Frame Processing 


# Add grayscale and blur to the frame so cards are white blobs. The thresholds will adapt to lighting by 
# sampling the background color in the top left corner of the image. Make sure to keep that spot free of cards. 
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    img_h, img_w = image.shape[:2]
    bg_level = int(gray[img_h // 100, img_w // 100])
    thresh_level = bg_level + bg_threst
    _, thresh = cv2.theshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
    return thresh 

# Returns card_countours sorted from largest to smallest and a list of booleans indicating if the contour is a card or not.
# The flag list is kept so that detector.py's two value unpacking of the return value still works.
def find_cards(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [], []
    order = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)

    card_conts = []
    for i in order: 
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        no_parent = hierarchy[0][i][3] == -1
        if card_min_area < area < card_max_area and len(approx) == 4 and no_parent:
            card_conts.append(cnt)

    return card_conts, [1] * len(card_conts)
        

   def flattener(image, pts, w, h):
    
        