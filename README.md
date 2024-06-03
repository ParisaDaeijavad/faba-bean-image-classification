[]([url](url))# WGRF Faba beans Image Classification Pipeline

- Repository to store code and files related to image work

**Introduction:**
The current work involves constructing of image classification pipeline to use faba bean images as the input file, to extract the features from them and make accurate class predictions based on deep learning models. This work involves the use of different machine learning and deep learning models such as SegmentAnything (SAM), OpenCV, Sci-kit, supervision and CNN classification. 
The novelty of this work lies in the utilization of SegmentAnything for image segmentation. While researchers have traditionally relied on OpenCV and scikit-image libraries for segmentation tasks, this study leverages SegmentAnything to produce high-quality masks. 

**SegmentAnything:**
The Segment Anything Model (SAM) excels at generating high-quality object masks from input prompts like points or boxes and can produce masks for all objects within an image. Trained on a vast dataset of 11 million images and 1.1 billion masks, SAM demonstrates robust zero-shot performance across various segmentation tasks.
This innovative approach ensures more precise and detailed segmentation, marking a significant advancement over conventional methods.

![Feature_Extraction Flowchart](https://001gc-my.sharepoint.com/:i:/r/personal/harpreet_bargota_agr_gc_ca/Documents/Pictures/Picture1.jpg?csf=1&web=1&e=a115Gl?raw=true "Title")

**Current Status:**
Currently, 254 images of fava beans (Vicia faba) provided by Dr. Nicholas Larkan have been used/tested for development of an image analysis pipeline for extracting 14 Dimensional features (Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter, Extent, Solidity, Eccentricity, Aspect ratio, Compactness, Roundness) and 4 shape factors (Shapefactor1-4) from images. For using SegmentAnything for this pipeline, initially some research work was done for comparing the segmenatation stratesies of SAM and other conventional libraries as OpenCV and Sci-kit. Furthermore, to test the effeciacy of different segmentation approaches in SAM, some research tests were done to conclude the utilization of automatic mask generation method of SAM to be used for segmentation purposes. 

**Installation:**
This work is being done at Lethbridge Superdome and AWS Phenomics sandbox.
Lethbridge superdome
(Section 1) For using SegmentAnything, GPUs with cuda are highly recommended. For using Lethbridge superdome, follow the instructions as:
1.	Connect to the AAFC network - either directly or by GCSRA/VPN
2.	Open the VS Code (Visual Studio Code) pre-installed in your localApps folder (no IT support required)
3.	For logging into the Lethbridge superdome server, “connect to the host” and use username and password to log in
4.	For installing the conda environment, follow the steps below:
Step 1: Install Conda (if not already installed)
If you don't have Conda installed, you can install Miniconda for a lightweight installation. Download Miniconda Installer:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
1.	Run the Installer:
```
bash Miniconda3-latest-Linux-x86_64.sh
```
2.	Follow the Installation Prompts:
o	Review the license agreement by pressing Enter.
o	Accept the license terms by typing yes.
o	Choose the installation location (default is usually fine).
o	Allow the installer to initialize Miniconda by typing yes.
3.	Initialize Conda:
```
source miniconda3/bin/activate
```
Step 2: Create a Conda Environment
1.	Create a New Conda Environment:
```
conda create --name myenv
```
Replace myenv with your desired environment name.
2.	Specify Python Version (optional):
```
conda create --name myenv python=3.10
```
This command creates an environment named myenv with Python version 3.10
Step 3: Activate the Conda Environment
```
conda activate myenv
```
Step 4: Install Packages in the Environment
```
conda install numpy pandas matplotlib
conda install git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip3 install opencv-python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip install -U scikit-image
pip install supervision
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
pip3 install --upgrade opencv-python
pip install pandas
conda install seaborn
pip install openpyxl
```

Step 5: Type 'nano <Human Readable>.py' in the Command line. Here we are using the nano editor, any other editor could also be used to save the data.
Step 6: Copy the python code for the specific analysis and paste it in the <Human Readable>.py file 
Step 7: The press 'ctrl+o' followed by 'enter'. 
Step 8: The press 'ctrl+x'
Step 9: Type python3 <Human Readable>.py.

There are different python files for different analysis.  For extracting the features, use feature_extraction.py. For using command line, paste the code as: 
```
python3 feature_extraction.py image/S3-input output_image
```
(Note: image/S3-input is the input folder containing all the images to be processed, and output_image is the name of the output folder having the resultant files of .csv features extracted, masks, annotated images and metadata files for each image)
For comparison studies of SAM, OpenCV and Sci-kit, copy and paste the code (from compare1.py) in the  Type 'nano <Human Readable>.py'. Change the path and name of the image in the code (e.g. image=cv2.read(‘path_to_image’)).
Step 10: After running the python files, download the output files.
AWS Phenomics Sandbox
(Section 2): Currently, for using AWS Phenomics sandbox, CIS Amazon Linux 2 Benchmark - Level 1 EC2 instance and m5.xlarge (4 CPU, 16GB Ram 0.214$/Hour with 50 GB storage is being used in AWS CLOUD Phenomics sandbox for comparison of SAM (point, box, amg) methods.

SUMMARY for feature extraction using SegmentAnything (SAM) and Sci-kit image toolboxes in AWS CLOUD Phenomics sandbox:

1.	Log in to AWS Phenomics sandbox 
2.	Create an input and output s3 bucket according to Terms of Cloud Naming conventions 
3.	Secure the buckets by applying Permissions and bucket policy
4.	Upload the images into S3 input bucket
5.	Create a custom AMI: IAM Role (with permissions) and IAM Policies
6.	Create a CIS Amazon Linux 2 Benchmark - Level 1 EC2 instance and m5.xlarge (4 CPU, 16GB Ram 0.214$/Hour with 50 GB storage and apply the security groups.
7.	Once, the instance is created, start the instance and use Session Manager to connect to the EC2 instance
8.	Once connected, you will have a linux terminal where you can create, edit and run python files.
9.	If you logged in successfully, the sh-4.2$ command displays.
10.	Switch from sh to bash.
11.	Install Conda, segment_anything (using the installation instructions in the repository), required libraries and packages for data extraction and analysis 
12.	Make a new directory “SAM” and copy your image “Faba-Seed-CC_Vf1-1-2.JPG” to this current folder.




```
/bin/bash
cd /home/ssm-user
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
mkdir /home/ssm-user/tmpconda; TMPDIR=/home/ssm-user/tmpconda bash Miniconda3-latest-Linux-x86_64.sh
source miniconda3/bin/activate
conda install git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip3 install opencv-python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U scikit-image
pip install supervision
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
pip3 install --upgrade opencv-python
pip install pandas
conda install seaborn
pip install openpyxl
mkdir images
aws s3 cp s3://agsg-hkkb-test-input-s3/Faba-Seed-CC_Vf1-1-2.JPG SAM
```
13.	Type 'nano <Human Readable>.py' in the Command line. Here we are using the nano editor, any other editor could also be used to save the data.
14.	Copy the code from below for the specific analysis and paste it in the <Human Readable>.py file.
15.	The press 'ctrl+o' followed by 'enter'.
16.	The press 'ctrl+x'
17.	Type python3 <Human Readable>.py.
18.	After the completion of the run of python3 <Human Readable>.py, use aws sync to transfer result files to output s3 bucket

```
aws s3 sync SAM s3://agsg-hkkb-test-output-s3
```
19.	Terminate the instance and stop the instance after completion of work.




