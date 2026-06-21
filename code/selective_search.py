'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2hsv
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###
    # Ensure image is 3-channel for segmentation
    if im_orig.ndim == 2:
        #Makes the grayscale image look like RGB by copying it into 3 channels
        im_for_seg = np.dstack([im_orig, im_orig, im_orig])
    else:
        im_for_seg = im_orig

    # Felzenszwalb expects float; normalize if image looks like uint8 [0..255]
    im_float = im_for_seg.astype(np.float32) #Convert image values to float numbers
    if im_float.max() > 1.0: #Checks if pixel values are likely in 0–255 range.
        im_float /= 255.0 #Scales pixel values down to 0–1 range.

    # 1.1 Generate initial segmentation mask (labels) 
    mask = felzenszwalb(im_float, scale=scale, sigma=sigma, min_size=min_size)

    # 1.2 Append mask as 4th channel [R, G, B, label]
    im_orig = np.dstack((im_for_seg, mask.astype(np.int32)))

    return im_orig

def sim_colour(r1, r2):#It measures the similarity of the colors between two regions r1 and r2 are.
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###
    inter_sum = 0.0
    for a, b in zip(r1["hist_c"], r2["hist_c"]): #bin by bin (same index bins together).
        inter_sum += min(a, b) # Add the smaller value of the two bins to the score
    return inter_sum

    # return 0


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###
    inter_sum = 0.0
    for a, b in zip(r1["hist_t"], r2["hist_t"]):
        inter_sum += min(a, b)
    return inter_sum

    # return 0


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###
    comb_size = r1["size"] + r2["size"]
    similar_size = 1.0 - (comb_size / imsize)

    # clamp for stability: If it goes outside the valid range, push it back into [0, 1]
    if similar_size < 0.0: #If the score becomes negative (shouldn’t usually), force it to 0.
        similar_size = 0.0
    elif similar_size > 1.0: #If score becomes above 1, force it to 1.
        similar_size = 1.0

    return similar_size

    # return 0


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###
    #Find the smallest bounding box that can cover both regions together.
    min_x = min(r1["min_x"], r2["min_x"])
    min_y = min(r1["min_y"], r2["min_y"])
    max_x = max(r1["max_x"], r2["max_x"])
    max_y = max(r1["max_y"], r2["max_y"])

    #Compute the area (in pixels) of that combined bounding box.
    # +1 because coordinates are typically inclusive pixel indices
    bb_width = (max_x - min_x + 1) #how many pixels wide the big rectangle is
    bb_height = (max_y - min_y + 1) #how many pixels tall the big rectangle is
    bb_size = bb_width * bb_height  #total pixels inside the rectangle (width × height)

    comb_size = r1["size"] + r2["size"] #Total pixels in the two regions (their combined region area).
    
    #bb_size - comb_size = empty space inside the bounding box
    #Divide by imsize to normalize by image size.
    similar_fill = 1.0 - (bb_size - comb_size) / imsize

    # clamp for stability keep the score safely between 0 and 1.
    if similar_fill < 0.0:
        similar_fill = 0.0
    elif similar_fill > 1.0:
        similar_fill = 1.0

    return similar_fill

    # return 0

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    ### YOUR CODE HERE ###
    # Ensure we only use RGB channels if a 4th channel (mask/label) exists
    if img is None or img.size == 0: #Handle empty input
        return np.zeros(BINS * 3, dtype=np.float32)

    if img.ndim == 2: #Make sure the image is RGB (3 channels)
        img_rgb = np.dstack([img, img, img])
    else:
        img_rgb = img[:, :, :3] #f it has 4 channels, take only the first 3 (RGB)

    # Convert to float in [0,1] for HSV conversion
    img_rgb = img_rgb.astype(np.float32)
    if img_rgb.max() > 1.0:
        img_rgb /= 255.0  #Convert pixel values to float 0–1

    hsv = rgb2hsv(img_rgb) #Convert RGB to HSV

    # 25-bin histogram per HSV channel, concatenated
    hists = []
    for ch in range(3):
        channel = hsv[:, :, ch].ravel() #flatten  into a 1D list of values
        #Count how many values fall into each of the BINS ranges between 0 and 1
        h, _ = np.histogram(channel, bins=BINS, range=(0.0, 1.0))
        hists.append(h.astype(np.float32)) #Save this channel’s histogram into list

    hist = np.concatenate(hists, axis=0) # join 3 all at the end

    # L1 normalize (important for histogram intersection)
    s = hist.sum()
    if s > 0:
        hist /= s


    return hist

def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###
    P = 8 #Use 8 neighbouring pixels around each pixel
    R = 1 #Look 1 pixel away (radius 1)
    METHOD = "uniform" #best for histogram

    # Compute per-channel LBP on the first 3 channels (ignore possible 4th label channel)
    n_ch = min(3, img.shape[2])
    for c in range(n_ch):
        channel = img[:, :, c]

        # Convert to uint8 for stable LBP ( better texture hist)
        if channel.dtype != np.uint8:
            ch = channel.astype(np.float32)
            if ch.max() <= 1.0:
                ch = ch * 255.0 #scale them up to 0–255.
            channel = np.clip(ch, 0, 255).astype(np.uint8) #Keep values within 0–255 and convert to uint8.

        lbp = local_binary_pattern(channel, P, R, METHOD)
        ret[:, :, c] = lbp


        

    return ret

def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###
     # Handle empty / None
    if img is None or img.size == 0:
        return np.zeros(BINS * 3, dtype=np.float32)

    # Use first 3 channels (ignore possible 4th label channel)
    if img.ndim == 2:
        img_use = np.dstack([img, img, img])
    else:
        img_use = img[:, :, :3]

    hists = []
    #Compute histogram per channel
    for c in range(3):
        channel = img_use[:, :, c].ravel() #flatten all values into a 1D list

        # For uniform LBP with P=8,R=1, values are in [0..9] -> 10 bins
        h, _ = np.histogram(channel, bins=BINS, range=(0, BINS)) #count how many values fall into bins 0..9
        hists.append(h.astype(np.float32))

    hist = np.concatenate(hists, axis=0)

    # L1 normalize (important for histogram intersection)
    s = hist.sum()
    if s > 0:
        hist /= s


    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    ### YOUR CODE HERE ###
    # Expect the 4th channel to be the segmentation label mask
    img_rgb = img[:, :, :3] #take only color channels
    img_mask = img[:, :, 3].astype(np.int32) #take region id of each pixel

    # Convert to float in [0, 1] for HSV conversion / LBP stability
    img_rgb_f = img_rgb.astype(np.float32)
    if img_rgb_f.max() > 1.0:
        img_rgb_f /= 255.0

    # Convert image to HSV color map (used for colour histograms)
    hsv = rgb2hsv(img_rgb_f)

    # Calculate texture gradient for the whole image (LBP per channel)
    tex_grad = calc_texture_gradient(img_rgb_f)

    # Count pixel positions & build region properties
    for k in np.unique(img_mask):
        mask = (img_mask == k) #True where pixel is in region k
        ys, xs = np.where(mask) #coordinates of region pixels
        if ys.size == 0: #f region is empty, skip it
            continue

        #tells the region possition and size
        R[k] = {}
        R[k]["min_x"] = int(xs.min())
        R[k]["max_x"] = int(xs.max())
        R[k]["min_y"] = int(ys.min())
        R[k]["max_y"] = int(ys.max())
        R[k]["size"] = int(ys.size)

        # Keep track of original segment labels (needed later for merges)
        R[k]["labels"] = [k]

        # --- Colour histogram on HSV (25 bins per channel => 75 dims) ---
        BINS_C = 25
        hc = []
        for c in range(3):
            vals = hsv[:, :, c][mask]
            h, _ = np.histogram(vals, bins=BINS_C, range=(0.0, 1.0))
            hc.append(h.astype(np.float32))
        hist_c = np.concatenate(hc, axis=0)
        s = hist_c.sum()
        if s > 0:
            hist_c /= s
        R[k]["hist_c"] = hist_c

        # --- Texture histogram on LBP output (10 bins per channel => 30 dims) ---
        BINS_T = 10
        ht = []
        for c in range(3):
            vals = tex_grad[:, :, c][mask]
            h, _ = np.histogram(vals, bins=BINS_T, range=(0, BINS_T))
            ht.append(h.astype(np.float32))
        hist_t = np.concatenate(ht, axis=0)
        s = hist_t.sum()
        if s > 0:
            hist_t /= s
        R[k]["hist_t"] = hist_t

    return R

def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    # Sort regions by min_x to reduce unnecessary pair checks (more efficient than O(n^2) brute force)
    items = list(regions.items()) #Convert dictionary into a list
    items.sort(key=lambda x: x[1]["min_x"]) #Sort regions by their left boundary (min_x) so we can skip many comparisons faster.

    n = len(items)
    for i in range(n):
        a_id, a = items[i] #region id and its data
        a_min_y, a_max_y = a["min_y"], a["max_y"] #vertical range of box a
        a_max_x = a["max_x"] #right boundary of box a
 
        # Only compare with boxes that start before a ends in x
        for j in range(i + 1, n):
            b_id, b = items[j]

            # Since items are sorted by min_x, once b starts after a ends, we can break
            if b["min_x"] > a_max_x:
                break

            # Quick reject on y-range overlap
            if b["min_y"] > a_max_y or b["max_y"] < a_min_y:
                continue

            #  check both directions 
            if intersect(a, b) or intersect(b, a):
                neighbours.append((a_id, b_id))


    return neighbours

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"] #combines two regions into one new region
    rt = {}
    ### YOUR CODE HERE
    rt["min_x"] = min(r1["min_x"], r2["min_x"]) #Left edge of new box = the smaller left edge.
    rt["min_y"] = min(r1["min_y"], r2["min_y"])#Top edge of new box = the smaller top edge.
    rt["max_x"] = max(r1["max_x"], r2["max_x"])#Right edge of new box = the bigger right edge.
    rt["max_y"] = max(r1["max_y"], r2["max_y"]) #Bottom edge of new box = the bigger bottom edge.
    rt["size"] = new_size #Store the merged pixel count.

    # Weighted average of histograms by region size (as in Selective Search)
    rt["hist_c"] = (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / float(new_size) #bigger region contributes more
    rt["hist_t"] = (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / float(new_size)

    # Keep list of original labels for the merged region
    rt["labels"] = r1["labels"] + r2["labels"]

    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50): #default scale=1.0, sigma=0.8, min_size=50
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1] #total number of pixels (used to normalize similarity scores).

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size) #Runs Felzenszwalb segmentation and creates a 4-channel image: [R,G,B,label]
    
    #If segmentation failed, stop.
    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # will store similarities between neighbour pairs
    S = {} 
    # for (ai, ar), (bi, br) in neighbours:
    #     S[(ai, bi)] = calc_sim(ar, br, imsize)
    for n in neighbours:
        # Support both formats:
        # 1) ((ai, ar), (bi, br))
        # 2) (ai, bi)

        #If neighbour list contains full region objects,
        if isinstance(n[0], tuple) and len(n[0]) == 2:
            (ai, ar), (bi, br) = n
        else:                           #Otherwise (your case), fetch region data from R
            ai, bi = n
            ar, br = R[ai], R[bi]

        if ai == bi: #ignore same regions
            continue

        # store similarities with a canonical undirected key
        key = (ai, bi) if ai < bi else (bi, ai)
        if key not in S:  # avoid duplicates like (2,5) and (5,2))
            S[key] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1 # keep as int (avoid 1.0 float key collisions)
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###
        keys_to_delete = []
        new_neighs = set()
        for (a, b) in S.keys():
            if a == i or b == i or a == j or b == j:
                keys_to_delete.append((a, b))
                # collect the "other" region as a neighbor candidate for the new region
                if a == i or a == j:
                    new_neighs.add(b)
                if b == i or b == j:
                    new_neighs.add(a)

        # Do not consider the merged-away regions as neighbours of the new region
        if i in new_neighs:
            new_neighs.remove(i)
        if j in new_neighs:
            new_neighs.remove(j)


        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for k in keys_to_delete:
            if k in S:
                del S[k]


        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###
        for k in new_neighs:
            if k not in R:
                continue
            key = (k, t) if k < t else (t, k)
            S[key] = calc_sim(R[k], R[t], imsize)


    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    for k, r in R.items():
        x = int(r["min_x"])
        y = int(r["min_y"])
        w = int(r["max_x"] - r["min_x"] + 1)
        h = int(r["max_y"] - r["min_y"] + 1)

        regions.append({
            "rect": (x, y, w, h),
            "labels": r["labels"],
            "size": r["size"]
        })


    return image, regions


