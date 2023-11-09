import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from google.colab.patches import cv2_imshow

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#import custom

# Root directory of the project
ROOT_DIR = "/content/drive/MyDrive/MaskRCNN"

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


WEIGHTS_PATH = "/content/drive/MyDrive/MaskRCNN/logs/mmodel20210814T0715/mask_rcnn_mmodel_0010.h5"

source="mydataset"

class ModelConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Mmodel"
 
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1 
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3 # Background,
    # typically after labeled, class can be set from Dataset class
    # if you want to test your model, better set it corectly based on your trainning dataset
 
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
 
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.75

config = ModelConfig()
CUSTOM_DIR = os.path.join(ROOT_DIR, "/Dataset/")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.75

config = InferenceConfig()
config.display()

class LabelmeDataset(utils.Dataset):
    # Load annotations
    # Labelme Image Annotator v = 3.16.7 
    # different version may have different structures
    # besides, labelme's annotation setup one file for 
    # one picture not for all pictures
    # and this annotations  are all dicts after Python json load 
    # {
    #   "version": "3.16.7",
    #   "flags": {},
    #   "shapes": [
    #     {
    #       "label": "balloon",
    #       "line_color": null,
    #       "fill_color": null,
    #       "points": [[428.41666666666674,  875.3333333333334 ], ...],
    #       "shape_type": "polygon",
    #       "flags": {}
    #     },
    #     {
    #       "label": "balloon",
    #       "line_color": null,
    #       "fill_color": null,
    #       "points": [... ],
    #       "shape_type": "polygon",
    #       "flags": {}
    #     },
    #   ],
    #   "lineColor": [(4 number)],
    #   "fillColor": [(4 number)],
    #   "imagePath": "10464445726_6f1e3bbe6a_k.jpg",
    #   "imageData": null,
    #   "imageHeight": 2019,
    #   "imageWidth": 2048
    # }
    # We mostly care about the x and y coordinates of each region
    def load_labelme(self, dataset_dir, subset):
        """
        Load a subset of the dataset.
        source: coustomed source id, exp: load data from coco, than set it "coco",
                it is useful when you ues different dataset for one trainning.(TODO)
                see the prepare func in utils model for details
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
 
        filenames = os.listdir(dataset_dir)
        jsonfiles,annotations=[],[]
        for filename in filenames:
            if filename.endswith(".json"):
                jsonfiles.append(filename)
                annotation = json.load(open(os.path.join(dataset_dir,filename)))
                # Insure this picture is in this dataset
                imagename = annotation['imagePath']
                if not os.path.isfile(os.path.join(dataset_dir,imagename)):
                    continue
                if len(annotation["shapes"]) == 0:
                    continue
                # you can filter what you don't want to load
                annotations.append(annotation)
                
        print("In {source} {subset} dataset we have {number:d} annotation files."
            .format(source=source, subset=subset,number=len(jsonfiles)))
        print("In {source} {subset} dataset we have {number:d} valid annotations."
            .format(source=source, subset=subset,number=len(annotations)))
 
        # Add images and get all classes in annotation files
        # typically, after labelme's annotation, all same class item have a same name
        # this need us to annotate like all "ball" in picture named "ball"
        # not "ball_1" "ball_2" ...
        # we also can figure out which "ball" it is refer to.
        labelslist = []
        for annotation in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            shapes = [] 
            classids = []
 
            for shape in annotation["shapes"]:
                # first we get the shape classid
                label = shape["label"]
                if labelslist.count(label) == 0:
                    labelslist.append(label)
                classids.append(labelslist.index(label)+1)
                shapes.append(shape["points"])
            
            # load_mask() needs the image size to convert polygons to masks.
            width = annotation["imageWidth"]
            height = annotation["imageHeight"]
            self.add_image(
                source,
                image_id=annotation["imagePath"],  # use file name as a unique image id
                path=os.path.join(dataset_dir,annotation["imagePath"]),
                width=width, height=height,
                shapes=shapes, classids=classids)
 
        print("In {source} {subset} dataset we have {number:d} class item"
            .format(source=source, subset=subset,number=len(labelslist)))
 
        for labelid, labelname in enumerate(labelslist):
            self.add_class(source,labelid,labelname)
 
    def load_mask(self,image_id):
        """
        Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not the source dataset you want, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != source:
            return super(self.__class__, self).load_mask(image_id)
 
        # Convert shapes to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["shapes"])], dtype=np.uint8)
        #printsx,printsy=zip(*points)
        for idx, points in enumerate(info["shapes"]):
            # Get indexes of pixels inside the polygon and set them to 1
            pointsy,pointsx = zip(*points)
            rr, cc = skimage.draw.polygon(pointsx, pointsy)
            mask[rr, cc, idx] = 1
        masks_np = mask.astype(np.bool)
        classids_np = np.array(image_info["classids"]).astype(np.int32)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return masks_np, classids_np
 
    def image_reference(self,image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == source:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

TEST_MODE = "inference"
ROOT_DIR = "/content/drive/MyDrive/MaskRCNN/Dataset"

def get_ax(rows=1, cols=1, size=16):
  """Return a Matplotlib Axes array to be used in all visualizations in the notebook.  Provide a central point to control graph sizes. Adjust the size attribute to control how big to render images"""
  _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
  return ax

# Load validation dataset
# Must call before using the dataset
CUSTOM_DIR = "/content/drive/MyDrive/MaskRCNN/Dataset"
dataset = LabelmeDataset()
dataset.load_labelme(CUSTOM_DIR, "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

config = ModelConfig()
#LOAD MODEL. Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load COCO weights Or, load the last model you trained
weights_path = WEIGHTS_PATH
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

#Now, we are ready for testing our model on any image.

#RUN DETECTION
image_id = random.choice(dataset.image_ids)
print(image_id)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
  modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,dataset.image_reference(image_id)))
# Run object detection
results = model.detect([image], verbose=1)
# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
# This is for predicting images which are not present in dataset
path_to_new_image = '/content/drive/MyDrive/MaskRCNN/image(56).jpg'
image1 = mpimg.imread(path_to_new_image)
# Run object detection
print(len([image1]))
results1 = model.detect([image1], verbose=1)
# Display results
ax = get_ax(1)
r1 = results1[0]
visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'], dataset.class_names, r1['scores'], ax=ax, title="Predictions1")