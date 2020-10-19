This folder contains training code and batch job script for submission to SCC.

A few things of note:
* Copying millions of small image files across nodes takes a very long time, must combine the images into one file and copy bulk.
* Copy images to local node /scratch directory for quick access.
* One training epoch (one pass through data) took just under three hours on V100 GPU.

I trained the network on SCC for two epochs and achieved 95% accuracy on validation data. This was a failure, because only 95% of images have PE label,
the network was just outputting zero for all the images.

Need to use focal loss to train network so that the network does not bias toward PE-negative.
