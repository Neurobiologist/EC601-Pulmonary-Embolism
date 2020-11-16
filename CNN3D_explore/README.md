To explore 3D CNN model training on exam level target

Target: 
image level: 'pe_present_on_image'
exam level: 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe', 'rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate'

3D CNN model: Using 3D densenet121

Still exploring to solve some errors

Preprocessed Data reference: https://www.kaggle.com/vaillant/rsna-str-pe-detection-jpeg-256  
3D Densenet121 reference:https://www.kaggle.com/boliu0/monai-3d-cnn-training/data

Future plan:  
Combine 2D CNN feature and 3D CNN feature with RNN  
2D CNN:  
3D CNN:  
Use several densenet121 model to generate features for different exam level targets
(1)negative_exam_for_pe
(2)rv_lv_ratio_gte_1
(3)leftsided_pe, rightsided_pe, central_pe
RNN:  
