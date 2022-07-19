# nocaps <validation set, test set>
mkdir nocaps
cd nocaps
mkdir annotations
mkdir image
mkdir obj_labels
cd annotations
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
cd ../..

# Localized Narratives <train image set(10), validation image set , test image set, train caption set, validation caption set, test caption set >
mkdir narratives
cd narratives
mkdir annotations
mkdir image
mkdir obj_labels
cd annotations
gdown https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl
gdown https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_captions.jsonl
gdown https://storage.googleapis.com/localized-narratives/annotations/open_images_test_captions.jsonl
cd ../..

# cococaption
mkdir coco
cd coco
mkdir annotations
mkdir obj_labels
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
mv annotations_trainval2014/annotations/captions_train2014.json coco/annotations
mv annotations_trainval2014/annotations/captions_val2014.json coco/annotations
cd ..

# Flickr30k
mkdir flickr30k
cd flickr30k
mkdir annotations
mkdir image
mkdir obj_labels
cd ..