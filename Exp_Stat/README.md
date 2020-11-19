This folder contains the statistics for reproducing all the figures in our paper:

- Calibration statistics (NLL, ECE, Brier score) are saved in the folder **full_image_calibration_stat** and **region_calibration_stat**, each file is named as *Vx_Method_x_Stage_x_Version_x.obj*
  - Vx means experiment index (multiple experiments for per method are carried out to get confidence interval)
  - Method_x_Stage_x: B, C and D correspond to methods VarRatio (stage 1), Entropy (stage 2) and BALD (stage 3)
  - Version_x: experiment version
- Uncertainty distribution of the pixels in the acquired images and regions are saved in the folder **acquired_full_image_uncertainty** and **acquired_region_uncertainty**, the files are named with the same fashion as previous ones. Each file contains the predicted probability of all the pixels in the acquired images/regions at each acquisition step. 
- The statistics for creating the expected calibration error histogram using full image acquisition strategy are saved in the folder **ece_histogram**
- Glas.xlsx file contains the segmentation accuracy (F1 score, Dice Index) for the models at each acquisition step

