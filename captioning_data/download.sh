# Conceptual Captions <training split, validation split, and image labels>
# mkdir conceptual
# cd conceptual
# Download https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250
# Download https://storage.cloud.google.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv?_ga=2.141047602.-1896153081.1529438250
# Download https://storage.cloud.google.com/conceptual-captions-v1-1-labels/Image_Labels_Subset_Train_GCC-Labels-training.tsv?_ga=2.234395421.-20118413.1607637118
# cd ..

# VizWiz visual question answering dataset <training set, validation set, test set, and annotations>
# mkdir vizwiz
# cd vizwiz
# wget https://ivc.ischool.utexas.edu/VizWiz_final/images/train.zip
# wget https://ivc.ischool.utexas.edu/VizWiz_final/images/val.zip
# wget https://ivc.ischool.utexas.edu/VizWiz_final/images/test.zip
# wget https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations.zip
# cd ..

# ReferIT dataset <train dataset, test dataset>
# mkdir referit
# cd referit
# wget http://tamaraberg.com/referitgame/ReferitData.zip
# wget http://tamaraberg.com/referitgame/test_set_ground_truth.zip_ga=2.234395421.-20118413.1607637118
# cd ..

# nocaps <validation set, test set>
mkdir nocaps
cd nocaps
mkdir annotations
mkdir image
mkdir obj_detection
cd annotations
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
cd ../..

# Localized Narratives <train image set(10), validation image set , test image set, train caption set, validation caption set, test caption set >
mkdir narratives
cd narratives
mkdir annotations
mkdir image
mkdir obj_detection
cd annotationss
gdown https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl
gdown https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_captions.jsonl
gdown https://storage.googleapis.com/localized-narratives/annotations/open_images_test_captions.jsonl
cd ../..

# cococaption
mkdir cococaption
cd cococaption
mkdir image
mkdir obj_detection
git clone https://github.com/tylin/coco-caption.git
cd coco-caption
cp -r annotations ../
cd ../..

# Flickr30k
mkdir flickr30k
cd flickr30k
mkdir annotations
mkdir image
mkdir obj_detection
cd ..