This folder contains the following models:

* model-resnet.pth: RexNeXt-50 model, input 224x224 slice, output 1 value.
* model-efficientb0.pth: Efficient Net B0 model, input 224x224 slice, output 1 value.
* model-combined-lstm1.pth: ResNeXt-EfficientNet-LSTM image level model. Input combined slice-level feature vector, output 1 value per slice.
* model-combined-lstm2.pth: ResNeXt-EfficientNet-LSTM study level model. Input combined slice-level feature vector, output 9 values per study.
* model-resnet-lstm1.pth: ResNeXt-LSTM image level model. Input combined slice-level feature vector, output 1 value per slice.
* model-resnet-lstm2.pth: ResNeXt-LSTM study level model. Input combined slice-level feature vector, output 9 values per study.
* model-efficientb0-lstm1.pth: EfficientNet-LSTM image level model. Input combined slice-level feature vector, output 1 value per slice.
* model-efficientb0-lstm2.pth: EfficientNet-LSTM study level model. Input combined slice-level feature vector, output 9 values per study.

This folder contains the following study-level test results:

* pe_label.pkl: ground truth label vector.
* pred.pkl: prediction (probability of PE).

Use the above two pickled lists to generate confusion matrices or ROC curve.
