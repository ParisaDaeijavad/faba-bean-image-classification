
################################################################################################################################
###Image classification pipeline using amg SAM, inverted mask, label, feature extraction and classification

    
    ## for loop for pipeline
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import skimage
from skimage import filters, io, color
from skimage.io import imread
from skimage import filters
from skimage import measure
#from pyclesperanto_prototype import imshow
from skimage.color import rgb2gray
from glob import glob
import os as os
import warnings
from skimage.measure import label, regionprops, regionprops_table
# Ignore all warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import supervision as sv
from skimage.util import crop
from pycocotools import mask as mask_utils
from typing import Any, Dict, List, Optional, Tuple
import sys
sys.path.append("..")





def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


#Dependencies for SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"    

        
#Pics = ['SAM/Faba-Seed-CC_Vf1-1-2.JPG','SAM/Faba-Seed-CC_Vf80-1-1.JPG','SAM/Faba-Seed-CC_Vf66-1-1.JPG','SAM/Faba-Seed-CC_Vf112-1-2.JPG','SAM/Faba-Seed-CC_Vf125-1-3.JPG','SAM/Faba-Seed-CC_Vf140-1-1.JPG','SAM/Faba-Seed-CC_Vf216-1-1.JPG','SAM/Faba-Seed-CC_Vf238-1-1.JPG','SAM/Faba-Seed-CC_Vf427-1-1.JPG','SAM/Faba-Seed-CC_Vf462-3-1.JPG' ]
Pics = ['SAM/Faba-Seed-CC_Vf125-1-3.JPG','SAM/Faba-Seed-CC_Vf80-1-1.JPG', 'SAM/Faba-Seed-CC_Vf62-1-1.JPG']

df_total = []

#Step1:Read an image
for x in Pics:
	image = cv2.imread(x)   # read an image using openCV
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # convert into RGB format
	image = crop(image, ((2000, 800), (50, 50), (0,0)), copy=False)   #crop the rectangular part  having color card 


#Step 2: Image Segmenation using SegmentAnything MetaAI

	sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) #SAM model and checkpoint
	sam.to(device=device) #device
	mask_generator = SamAutomaticMaskGenerator(sam, box_nms_thresh = 0.1, output_mode = "binary_mask") #Specify the parameters
	masks_SAM = mask_generator.generate(image)  #generates masks

	print("The total length of all masks is: ", len(masks_SAM))
	print(masks_SAM[0].keys())

# Metafile for SAM masks
  
	df1=[]
	header1=['id','area','bbox_x0','bbox_y0','bbox_w','bbox_h','point_input_x','point_input_y','predicted_iou','stability_score','crop_box_x0','crop_box_y0','crop_box_w','crop_box_h']
	metadata = []
	for i, mask_data in enumerate(masks_SAM):
		mask = mask_data["segmentation"]
		#filename = f"binarymask-{i}.png"
		#cv2.imwrite(filename, mask * 255)
		mask_metadata = [
            	str(i),
            	str(mask_data["area"]),
            	*[str(x) for x in mask_data["bbox"]],
           	*[str(x) for x in mask_data["point_coords"][0]],
            	str(mask_data["predicted_iou"]),
            	str(mask_data["stability_score"]),
            	*[str(x) for x in mask_data["crop_box"]],
        	]    #create a list of metadata with header1
		df=pd.DataFrame(mask_metadata, index=header1) #create the dataframe with header1 as index
		df1.append(df) #append all the dfs for all the masks for an image
	df2=pd.concat(df1,axis=1)  # join all the dataframes formed for all images
	df_metadata=df2.T.apply(pd.to_numeric).set_index('id')  #change the datatype to numerical and set the index by 'id'for analysis

	# Filtering the masks of beans and eliminatingg the masks of labels and coin
	df_metadata = df_metadata.drop(df_metadata[(df_metadata['bbox_x0'] <= 1600) & (df_metadata['bbox_y0'] >= 2600)].index) #eliminating the masks for label
	df_metadata = df_metadata.drop(df_metadata[(df_metadata['bbox_x0'] >= 3300) & (df_metadata['bbox_y0'] >= 2500)].index)  #eliminating the masks for coin
	df_metadata = df_metadata.drop(df_metadata[(df_metadata['bbox_h'] >= 800)].index) #eliminating the masks that exceed a certain height of the bounding boxes 
	df_metadata=df_metadata.loc[(df_metadata['area'] >= 5000) & (df_metadata['area'] <= 290000)] #filtering the masks according to the minimum and maximum area of beans
	print (df_metadata)
	#df_metdata.to_excel("SAM/metadata.xlsx")
	Total_number_of_seeds=len(df_metadata.index)
	print("Total number of seeds is ", Total_number_of_seeds)

#print ("Filtering of mask is done")

	Masks_list=df_metadata.index.tolist()
	masks_SAM = [masks_SAM[x] for x in Masks_list]
	#print(len(masks_SAM))
	plt.figure()
	plt.imshow(image)
	show_anns(masks_SAM)
	plt.axis('on')
	plt.savefig("SAM/Mask_{x}.jpg")
#plt.savefig("SAM/df_allmask{i}.png".format(i=i))  
	print("After filtering of the masks, now the length of all masks required for feature extraction is: ", len(masks_SAM))
##### IMAGE ANNOTATION --  Supervision detections
	detections = sv.Detections.from_sam(sam_result=masks_SAM)
	polygon_annotator = sv.PolygonAnnotator(color=sv.ColorPalette.ROBOFLOW, thickness=36, color_lookup=sv.ColorLookup.INDEX)
	annotated_frame = polygon_annotator.annotate(scene=image.copy(),	detections=detections)
	cv2.imwrite("SAM/annotated_frame{x}.jpg", annotated_frame)


############################################################################################################################################
############ FEATURE EXTRACTION

	df_list=[]
	for i, mask_data in enumerate(masks_SAM):
		mask = mask_data["segmentation"]
 	#label masks
		label_image = measure.label(mask)

	#analyse masks
		properties = measure.regionprops(label_image, intensity_image=image)
		regions=regionprops(label_image)
# extract relevant properties
		statistics = {
		'Area':       [p.area for p in properties],
      		'Perimeter':       [p.perimeter     for p in properties],
	    	'MajorAxisLength': [p.major_axis_length  for p in properties],
		'MinorAxisLength': [p.minor_axis_length for p in properties],
		'ConvexArea': [p.area_convex  for p in properties],
		'EquivDiameter': [p.equivalent_diameter_area for p in properties],
		'Extent': [p.extent for p in properties],
		'Solidity': [p.solidity for p in properties],
		'Eccentricity': [p.eccentricity for p in properties]}

  # convert to dataframe
		df_FE = pd.DataFrame(statistics)

		df_FE["AspectRation"] = df_FE["MajorAxisLength"]/df_FE['MinorAxisLength']
		#df_FE["Roundness"]=df_FE[(4*3.14*(df_FE["Area"]))/((df_FE["Perimeter"])*(df_FE["Perimeter"]))]
		df_FE["Compactness"] = df_FE["EquivDiameter"]/df_FE["MajorAxisLength"]
		#Shape features
		df_FE["Shapefactor1"] = df_FE["MajorAxisLength"]/df_FE["Area"]
		df_FE["Shapefactor2"] = df_FE["MinorAxisLength"]/df_FE["Area"]
 		#df_FE["Shapefactor3"] = df_FE["axis_minor_length"]/df_FE["area"]
 		#df_FE["Shapefactor4"] = df_FE["axis_minor_length"]/df_FE["area"]
		class_in_image=(x).split('.JPG')[0]
		df_FE["class"]= (x).split('.JPG')[0]
		df_list.append(df_FE)
	df_FE2 = pd.concat(df_list)
	df_total.append(df_FE2)
df_image=pd.concat(df_total)
print(df_image)
	
df_image.to_excel("SAM/Feature_Extraction.xlsx")   
print ("FE done!") 
 

      



     


    
