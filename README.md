This is the code for the CVPR'19 paper ["Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos".](https://openaccess.thecvf.com/content_CVPR_2019/html/Morais_Learning_Regularity_in_Skeleton_Trajectories_for_Anomaly_Detection_in_Videos_CVPR_2019_paper.html)

# Environment Setup
First please create an appropriate environment using conda: 

> conda env create -f environment.yml

> conda activate tbad

# Download Data
Due to space constraints in the Github repository, please download the data from the following link and place the `data` folder in this directory.

Link: [trajectories](https://bit.ly/2TWCxFY)

# Test Pre-Trained Models
To evaluate pre-trained models run the evaluate.py script.
Some examples:

#### Evaluate MPED-RNN on all cameras individually and all cameras combined on HR-ShanghaiTech.
> python evaluate.py --gpu_ids 0 --gpu_memory 0.2 combined_models ./pretrained/CVPR19/ShanghaiTech/combined_model/\_mp_Grobust_Lrobust_Orobust_concatdown\_
 ./data/HR-ShanghaiTech/testing/trajectories ./data/HR-ShanghaiTech/testing/frame_level_masks --video_resolution 856x480 --overlapping_trajectories

#### Evaluate MPED-RNN on HR-Avenue.
> python evaluate.py --gpu_ids 0 --gpu_memory 0.2 combined_model ./pretrained/CVPR19/Avenue/combined_model/\_mp_Grobust_Lrobust_Orobust_concatdown\_/01_2018_11_13_06_36_20 
./data/HR-Avenue/testing/trajectories/01 ./data/HR-Avenue/testing/frame_level_masks/01 --video_resolution 640x360 --overlapping_trajectories

# Train Models from Scratch
To train a model from scratch you should look up the model's configuration options using the option --help on the training.py script. Here is one example:

#### Train MPED-RNN on the ShanghaiTech data set.
> python train.py --gpu_ids 0 --gpu_memory 0.1 combined_model ./data/HR-ShanghaiTech/training/trajectories/00 
--video_resolution 856x480 --message_passing --reconstruct_original_data --multiple_outputs --multiple_outputs_before_concatenation --input_length 12
 --rec_length 12 --pred_length 6 --reconstruct_reverse --cell gru --global_hidden 8 --local_hidden 16 --output_activation linear 
 --optimiser adam --learning_rate 0.001 --loss mse --epochs 5 --batch_size 256 --global_normalisation robust --local_normalisation robust 
 --out_normalisation robust