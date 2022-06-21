## Data Preparation

### Download Images 

**Download COCO image dataset (for VQA-X)**

~~~bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
~~~

### Download Annotations

COCO Caption for evaluation

~~~bash
git clone https://github.com/tylin/coco-caption.git
mv coco-caption cococaption
cd cococaption
./get_stanford_models.sh
pip install bert_score==0.3.7
~~~

Download VQA-X annotations

~~~bash
gdown https://drive.google.com/drive/folders/16sJjeEQE2o23G-GGUi870ubXzJjdRDua --folder
mv VQA-X nle_anno/VQA-X
~~~

Download processed annotations

~~~bash
gdown https://drive.google.com/drive/folders/1b8kUPbgtEduiz8A_VbUg0W_vca7PyXsZ --folder
mv "cococaption annot"/* cococaption/annotations/ 
~~~

### Download Captioning dataset


Conceptual Captions
<training split, validation split, and image labels>
~~~bash
wget https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250
wget https://storage.cloud.google.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv?_ga=2.141047602.-1896153081.1529438250
wget https://storage.cloud.google.com/conceptual-captions-v1-1-labels/Image_Labels_Subset_Train_GCC-Labels-training.tsv?_ga=2.234395421.-20118413.1607637118
~~~
VizWiz visual question answering dataset 
<training set, validation set, test set, and annotations>
~~~bash
wget https://ivc.ischool.utexas.edu/VizWiz_final/images/train.zip
wget https://ivc.ischool.utexas.edu/VizWiz_final/images/val.zip
wget https://ivc.ischool.utexas.edu/VizWiz_final/images/test.zip
wget https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations.zip
~~~
ReferIT dataset
<train dataset, test dataset>
~~~bash
wget http://tamaraberg.com/referitgame/ReferitData.zip
wget http://tamaraberg.com/referitgame/test_set_ground_truth.zip_ga=2.234395421.-20118413.1607637118
wget http://www-i6.informatik.rwth-aachen.de/imageclef/resources/saiaprtc12/saiaprtc12ok.part1.rar
wget http://www-i6.informatik.rwth-aachen.de/imageclef/resources/saiaprtc12/saiaprtc12ok.part2.rar
wget http://www-i6.informatik.rwth-aachen.de/imageclef/resources/saiaprtc12/saiaprtc12ok.part3.rar
~~~
nocaps
<validation set, test set>
~~~bash
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
wget https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json
~~~
Localized Narratives
<train image set(10), validation image set , test image set, train caption set, validation caption set, test caption set >
image data from open-images-dataset\
https://github.com/cvdfoundation/open-images-dataset.git

~~~bash
wget https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl
wget https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_captions.jsonl
wget https://storage.googleapis.com/localized-narratives/annotations/open_images_test_captions.jsonl
~~~

