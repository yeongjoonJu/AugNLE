wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
mkdir images
mv train2014.zip images/
mv val2014.zip images/
cd images
unzip train2014.zip
unzip val2014.zip
cd ..
git clone https://github.com/tylin/coco-caption.git
mv coco-caption cococaption
cd cococaption
./get_stanford_models.sh
pip install bert_score==0.3.7
cd ..
mkdir nle_anno
gdown https://drive.google.com/drive/folders/16sJjeEQE2o23G-GGUi870ubXzJjdRDua --folder
mv VQA-X nle_anno/VQA-X
gdown https://drive.google.com/drive/folders/1b8kUPbgtEduiz8A_VbUg0W_vca7PyXsZ --folder
mv 'cococaption annot'/* cococaption/annotations/