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

Install NLG evaluation package for filtering

~~~bash
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
~~~

### Detect objects in images

~~~bash
pip install timm
~~~

### Download image captioning datasets

~~~bash
cd captioning_data
bash download.sh # please check download.sh file for the conceptual captions dataset
~~~
