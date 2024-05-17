#__author__="harpreet kaur bargota"
#__email__="harpreet.bargota@agr.gc.ca"
#__Project__="WGRF - Image Classification pipeline for Faba beans"

#References: 
#SegmentAnthing (MetaAI): https://github.com/facebookresearch/segment-anything 
# Reference paper: @article{kirillov2023segany, title={Segment Anything}, author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},journal={arXiv:2304.02643},year={2023}}

#Feature extraction: 
#scikit-image library for image processing: Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart, Tony Yu and the scikit-image contributors. scikit-image: Image processing in Python. PeerJ 2:e453 (2014) https://doi.org/10.7717/peerj.453
#https://scikit-image.org/docs/stable/api/skimage.measure.html






###Image classification pipeline using amg SAM, inverted mask, label, feature extraction and classification

    
    ## for loop for pipeline
import torch
import argparse
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




#Dependencies for SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"    





## defining parameters for masks
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

# defining the function for processing of images


def process_images(input_folder, output_folder):
    # Check if the input folder exists
	if not os.path.exists(input_folder):
		print(f"Error: Input folder '{input_folder}' does not exist.")
		return

    # Create the output folder if it doesn't exist
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
		print(f"Output folder '{output_folder}' created.")

	for root, dirs, files in os.walk(input_folder):
		df_total = []
		
		for idx, file in enumerate(files):
#Step1:        # Read an image
			image_path=os.path.join(root,file)
			image_path = os.path.join(root,file)
			image=cv2.imread(image_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
			#image = crop(image, ((2000, 800), (50, 50), (0,0)), copy=False)   
			image = crop(image, ((2200, 800), (50, 50), (0,0)), copy=False)
#Step2: 	#Apply SegmentAnything model 
 
			sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) 
			sam.to(device=device) 
			mask_generator = SamAutomaticMaskGenerator(sam, box_nms_thresh = 0.1, output_mode = "binary_mask") 
			masks_SAM = mask_generator.generate(image)  
			print("The total length of all masks is: ", len(masks_SAM))
			print(masks_SAM[0].keys())
			
            # Metadata
			
			df1=[]
			header1=['id','area','bbox_x0','bbox_y0','bbox_w','bbox_h','point_input_x','point_input_y','predicted_iou','stability_score','crop_box_x0','crop_box_y0','crop_box_w','crop_box_h']
			metadata = []
			for i, mask_data in enumerate(masks_SAM):
				mask = mask_data["segmentation"]
				mask_metadata = [
					str(i),
					str(mask_data["area"]),
					*[str(x) for x in mask_data["bbox"]],
					*[str(x) for x in mask_data["point_coords"][0]],
					str(mask_data["predicted_iou"]),
					str(mask_data["stability_score"]),
					*[str(x) for x in mask_data["crop_box"]],
					]    #create a list of metadata with header1
				df=pd.DataFrame(mask_metadata, index=header1)
				df1.append(df)
			df2=pd.concat(df1,axis=1)
			df_metadata=df2.T.apply(pd.to_numeric).set_index('id')
			
 #Step3:           
            # Filtering the masks of beans and eliminating the masks of labels and coin
			df_metadata = df_metadata.drop(df_metadata[(df_metadata['bbox_x0'] <= 1600) & (df_metadata['bbox_y0'] >= 2400)].index) 
			df_metadata = df_metadata.drop(df_metadata[(df_metadata['bbox_x0'] >= 3300) & (df_metadata['bbox_y0'] >= 2000)].index)
			df_metadata = df_metadata.drop(df_metadata[(df_metadata['bbox_h'] >= 600)].index) #800
			df_metadata = df_metadata.drop(df_metadata[(df_metadata['bbox_w'] >= 600)].index) #1000
			df_metadata=df_metadata.loc[(df_metadata['area'] >= 5000) & (df_metadata['area'] <= 290000)]
			print (df_metadata)
			
             
	
			image_filename = os.path.basename(image_path)
			output_filename = f"Metadata_{image_filename}.xlsx"
			output_path= os.path.join(output_folder, output_filename)
			df_metadata.to_excel(output_path)
			Total_number_of_seeds=len(df_metadata.index)
			print("Total number of seeds is ", Total_number_of_seeds)
			Masks_list=df_metadata.index.tolist()
			masks_SAM=[masks_SAM[x] for x in Masks_list]
		
            

				
			# Save fava bean masks in output_folder
			for i, mask_data in enumerate(masks_SAM):
				mask = mask_data["segmentation"]
				image_filename = os.path.basename(image_path)
				output_filename = f"{image_filename}_mask_{idx}_{i}.png"
				output_path= os.path.join(output_folder, output_filename)
				cv2.imwrite(output_path, mask*255)
			
			print("After filtering of the masks, now the length of all masks required for feature extraction is: ", len(masks_SAM))
			
#Step4:            ##### IMAGE ANNOTATION --  Supervision detections
			detections = sv.Detections.from_sam(sam_result=masks_SAM)
			polygon_annotator = sv.PolygonAnnotator(color=sv.ColorPalette.ROBOFLOW, thickness=36, color_lookup=sv.ColorLookup.INDEX)
			annotated_frame = polygon_annotator.annotate(scene=image.copy(),	detections=detections)
			image_filename = os.path.basename(image_path)
			output_filename = f"Annotated_image_{image_filename}.png"
			output_path= os.path.join(output_folder, output_filename)
			cv2.imwrite(output_path, annotated_frame)
#Step5:			
# FEATURE EXTRACTION
			df_list=[]
			for i, mask_data in enumerate(masks_SAM):
				mask = mask_data["segmentation"]

				#Label the mask
				#Reference: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_label.html
				label_image = measure.label(mask)

				#analyse masks
				#Reference: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
				props = regionprops_table(label_image,properties=('centroid_local', 'orientation', 'area','eccentricity', 'equivalent_diameter_area','perimeter','solidity', 'area_convex', 'extent','axis_major_length', 'axis_minor_length'),) #properties = measure.regionprops(label_image, intensity_image=image)
				
				# convert to dataframe
				df_FE = pd.DataFrame(props)
				
				#For calculating the other parameters, mathematical operations have been used as advised in paper(Multiclass classification of dry beans using computer vision and machine learning techniques)
				df_FE["Aspect_Ratio"] = df_FE["axis_major_length"]/df_FE['axis_minor_length']
				#df_FE["Roundness"]=df_FE[(4*3.14*(df_FE["area"]))/((df_FE["perimeter"])*(df_FE["perimeter"]))]
				df_FE["Compactness"] = df_FE["equivalent_diameter_area"]/df_FE["axis_major_length"]
				#Shape features
				df_FE["Shapefactor1"] = df_FE["axis_major_length"]/df_FE["area"]
				df_FE["Shapefactor2"] = df_FE["axis_minor_length"]/df_FE["area"]
				#df_FE["Shapefactor3"] = df_FE["MinorAxisLength"]/df_FE["Area"]
				#df_FE["Shapefactor4"] = df_FE["axis_minor_length"]/df_FE["area"]

				# The name of class has been extracted from the image file name 
				class_in_image=(image_path).split('.JPG')[0]
				df_FE["class"]= (image_path).split('.JPG')[0]
				df_list.append(df_FE)
			df_FE2 = pd.concat(df_list)
			df_total.append(df_FE2)
		df_image=pd.concat(df_total)
		print(df_image)
		output_filename = f"Fava_bean_Features_extraction.xlsx"
		output_path= os.path.join(output_folder, output_filename)
		df_image.to_csv(output_path)
		print ("Feature extraction from fava bean images is completed..!")





if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Process files from input folder to output folder.")

    # Add arguments for input and output folders
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("output_folder", help="Path to the output folder")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the process_folders function with provided input and output folders
    process_images(args.input_folder, args.output_folder)

#for x in args.input_folder:
#	image = cv2.imread(r'x')   # read an image using openCV


