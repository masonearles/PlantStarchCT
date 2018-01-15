# PlantStarchCT
### In-vivo quantification of starch reserves in plant stems using a Random Forest machine learning algorithm

This is the GitHub repository associated with the paper:

Earles, J.M.*, Knipfer, T.K.*, Tixier, A., Orozco, J., Reyes, C., Zwieniecki, M.A., Brodersen, C.R., and McElrone, A.J. (accepted). In-vivo quantification of starch reserves in plants using X-ray microCT imaging and machine learning. *Authors contributed equally

#### Initial observations of depleted regions in stem parenchymal tissue of grapevine plants

We began this project after observing apparently depleted regions in the stems of parenchymal tissue of grapevine plants. Here's a panel of images demonstrating what we saw that piqued our curiosity.

![Alt text](imgs/Fig_1.jpg?raw=true "Fig. 1")

#### Machine learning framework for in-vivo quantification of starch in plant stems

We suspected that these depleted regions correpsond with starch depletion in ray and axial parenchyma (RAP) cells. If so, this would be very exciting, as no techniques currently exist for quantifying and spatially mapping plant starch reserves *in-vivo*. <br> <br> To help us quantify the presence/absence of these full versus empty RAP regions, we developed the following machine learning framework:

![Alt text](imgs/Fig_4.png?raw=true "Fig. 4")

First, X-ray microCT images are collected, resulting in 32-bit images of the stem cross section for each plant. Visually empty/full parenchymal regions are manually labeled as full or empty of starch (see image below). Manually labeled images are split equally into test and training image datasets. MicroCT images are preprocessed (i.e. cropped, denoised, and contrast stretched) to normalize images across plant samples and to facilitate learning by the training algorithm. Feature layers are generated by convolving the preprocessed images with various types of kernels (e.g. Gaussian, variance, lines, and patches) that corresponded with spatial patterns in X-ray absorption of starch, parenchymal cells, and cell wall tissue. A random forest algorithm is used to train a model to predict the labeled training images based on available feature layers. The trained model is used to predict empty/full RAP regions in test images that are not used for training and the model’s performance is evaluated. <br>

![Alt text](imgs/Fig_3.jpg?raw=true "Fig. 3")

#### The image datasets used in *Earles et al. (accepted)* are several Gb, so we aren't hosting them on GitHub. Please send us a message if you would like a direct link to download them.
