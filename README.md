# OpenSeeFace OV Fork
OpenVINO Implementation of [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace) taking advantage of Intel iGPU and NPU.  
**Please note: This is a personal project for re-implementing inference backends.**


## Usage
This fork is focused solely on transmitting face tracking data for applications like VTubeStudio. Additional features like Unity examples have been removed in this fork.

Run the python script with `--help` to learn about the possible options you can set.
```bash
python facetracker.py --help
```
Visualize face tracking with a camera input.
```bash
# Replace -c ${CamIDX} with your camera input, found with --list-cameras 1
python facetracker.py --visualize 3 --pnp-points 1 --max-threads 4 -c ${CamIDX}
```

# Dependencies (Python 3.7+)

* openvino
* opencv-python
* pillow
* numpy
* psutil

The required libraries can be installed using pip:
```bash
pip install openvino opencv-python pillow numpy psutil
```
## References

## Training dataset

The model was trained on a 66 point version of the [LS3D-W](https://www.adrianbulat.com/face-alignment) dataset.

    @inproceedings{bulat2017far,
      title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
      author={Bulat, Adrian and Tzimiropoulos, Georgios},
      booktitle={International Conference on Computer Vision},
      year={2017}
    }

Additional training has been done on the WFLW dataset after reducing it to 66 points and replacing the contour points and tip of the nose with points predicted by the model trained up to this point. This additional training is done to improve fitting to eyes and eyebrows.

    @inproceedings{wayne2018lab,
      author = {Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
      title = {Look at Boundary: A Boundary-Aware Face Alignment Algorithm},
      booktitle = {CVPR},
      month = June,
      year = {2018}
    }

For the training the gaze and blink detection model, the [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/) dataset was used. Additionally, around 125000 synthetic eyes generated with [UnityEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) were used during training.

It should be noted that additional custom data was also used during the training process and that the reference landmarks from the original datasets have been modified in certain ways to address various issues. It is likely not possible to reproduce these models with just the original LS3D-W and WFLW datasets, however the additional data is not redistributable.

The heatmap regression based face detection model was trained on random 224x224 crops from the WIDER FACE dataset.

	@inproceedings{yang2016wider,
	  Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	  Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  Title = {WIDER FACE: A Face Detection Benchmark},
	  Year = {2016}
    }

## Algorithm

### Model Layout

The OpenVINO IR files shipped with this fork now live in dedicated subdirectories under `ov_models/`:

* `ov_models/landmark-mobilenet-v3/landmark-mobilenet-v3.{xml,bin}` – 66‑point landmark head
* `ov_models/headmap-detector-mobilenet-v3/headmap-detector-mobilenet-v3.{xml,bin}` – coarse heatmap detector
* `ov_models/gaze-estimator-mobilenet-v3/gaze-estimator-mobilenet-v3.{xml,bin}` – per-eye gaze estimator

You can replace these with your own fine‑tuned models as long as the paths and tensor shapes stay compatible.

The algorithm is inspired by:

* [Designing Neural Network Architectures for Different Applications: From Facial Landmark Tracking to Lane Departure Warning System](https://www.synopsys.com/designware-ip/technical-bulletin/ulsee-designing-neural-network.html) by YiTa Wu, Vice President of Engineering, ULSee
* [Real-time Human Pose Estimation in the Browser with TensorFlow.js](https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) by Olaf Ronneberger, Philipp Fischer, Thomas Brox
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
* [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) by Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam

The MobileNetV3 code was taken from [here](https://github.com/rwightman/gen-efficientnet-pytorch).

For all training a modified version of [Adaptive Wing Loss](https://github.com/tankrant/Adaptive-Wing-Loss) was used.

* [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399) by Xinyao Wang, Liefeng Bo, Li Fuxin

For expression detection, [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) is used.

Face detection is done using a custom heatmap regression based face detection model.

# Thanks!

Many thanks to everyone who helped me test things!

* [@Virtual_Deat](https://twitter.com/Virtual_Deat), who also inspired me to start working on this.
* [@ENiwatori](https://twitter.com/eniwatori) and family.
* [@ArgamaWitch](https://twitter.com/ArgamaWitch)
* [@AngelVayuu](https://twitter.com/AngelVayuu)
* [@DapperlyYours](https://twitter.com/DapperlyYours)
* [@comdost_art](https://twitter.com/comdost_art)
* [@Ponoki_Chan](https://twitter.com/Ponoki_Chan)

# License

The code and models are distributed under the BSD 2-clause license. 

You can find licenses of third party libraries used for binary builds in the `Licenses` folder.
