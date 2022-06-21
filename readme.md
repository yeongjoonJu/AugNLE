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

