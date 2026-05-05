Unzip the Label Studio "COCO with Images" export here.

Expected layout:
    dataset/
        result.json
        images/
            0001.jpg
            0002.jpg
            ...

Then run from the repo root:
    python -m vision.train_seg
