This folder contains some code specific to our data.

Kaggle RSNA Pulmonary Embolism dataset has around 1,790,000 512x512 DICOM files.

These files are big and expensive to process. This code takes DICOM files as input and outputs 256x256 JPEG files.

I created replicated the code across four files, with different ranges to make it run faster.

To submit on SCC:

```
qsub batch1.qsub
qsub batch2.qsub
qsub batch3.qsub
qsub batch4.qsub
```

This splits the data across 4 CPU batch jobs.
