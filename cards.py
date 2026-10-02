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
    pts = np.float32(pts).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    t1, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, b1 = pts[np.argmin(diff)], pts[np.argmax(diff)]

    if w <= 0.8 * h: 
        rect = [t1, tr, br, b1]
    elif w >= 1.2 * h: 
        rect = [b1, br, tr, t1]
    else:
        if pts[1][1] <= pts[3][1]:
            rect = [pts[1], pts[0], pts[3], pts[2]]
        else:
            rect = [pts[0], pts[1], pts[2], pts[3]]

    dst = np.float32([[0, 0], [card_width - 1, 0], [card_width - 1, card_height - 1], [0, card_height - 1]])
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (card_width, card_height))
    return cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

def _largest_symbol(binary_img, out_w, out_h):
    
    conts, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(conts) == 0:
        return []
    conts = sorted(conts, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(conts[0])
    return cv2.resize(binary_img[y:y + h, x:x + w], (out_w, out_h), 0, 0)
 
 
def preprocess_card(contour, image):
    
    card = QueryCard()
    card.contour = contour
 
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    card.corner_pts = np.float32(approx)
 
    x, y, w, h = cv2.boundingRect(contour)
    card.width, card.height = w, h
 
    center = np.sum(card.corner_pts, axis=0)[0] / len(card.corner_pts)
    card.center = [int(center[0]), int(center[1])]
 
    card.warp = flattener(image, card.corner_pts, w, h)
 
    # Zoom in on the top-left corner where rank and suit live
    corner = card.warp[0:corner_height, 0:corner_width]
    corner_zoom = cv2.resize(corner, (0, 0), fx=4, fy=4)
 
    white_level = int(corner_zoom[15, (corner_width * 4) // 2])
    thresh_level = max(white_level - card_thresh, 1)
    _, corner_thresh = cv2.threshold(corner_zoom, thresh_level, 255,
                                     cv2.THRESH_BINARY_INV)
 
    rank_roi = corner_thresh[20:185, 0:128]
    suit_roi = corner_thresh[186:336, 0:128]
 
    card.rank_img = _largest_symbol(rank_roi, rank_width, rank_height)
    card.suit_img = _largest_symbol(suit_roi, suit_width, suit_height)
    return card
 
 
# ---------- Matching ----------
def match_card(card, train_ranks, train_suits):
  
    best_rank, best_suit = "Unknown", "Unknown"
    best_rank_diff, best_suit_diff = 10000, 10000
 
    if len(card.rank_img) != 0 and len(card.suit_img) != 0:
        for t in train_ranks:
            diff = int(np.sum(cv2.absdiff(card.rank_img, t.img)) / 255)
            if diff < best_rank_diff:
                best_rank_diff, best_rank = diff, t.name
 
        for t in train_suits:
            diff = int(np.sum(cv2.absdiff(card.suit_img, t.img)) / 255)
            if diff < best_suit_diff:
                best_suit_diff, best_suit = diff, t.name
 
    if best_rank_diff >= rank_diff_max:
        best_rank = "Unknown"
    if best_suit_diff >= suit_diff_max:
        best_suit = "Unknown"
 
    return best_rank, best_suit, best_rank_diff, best_suit_diff
 
 
# Drawing 
def draw_results(image, card):
    x, y = card.center
    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
 
    for offset, text in ((-20, card.best_rank), (25, card.best_suit)):
        cv2.putText(image, text, (x - 60, y + offset), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x - 60, y + offset), font, 1, (50, 200, 200), 2, cv2.LINE_AA)
    return image


        