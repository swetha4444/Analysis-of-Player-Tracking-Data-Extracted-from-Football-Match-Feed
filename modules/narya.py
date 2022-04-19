import numpy as np
import cv2
from narya.models.keras_models import DeepHomoModel
import tensorflow as tf
from narya.utils.homography import *
import kornia as K
from matplotlib import pyplot as plt
from narya.utils.vizualization import visualize
from narya.utils.vizualization import merge_template

def get_perspective_transform_torch(src, dst):
    """Get the homography matrix between src and dst

    Arguments:
        src: Tensor of shape (B,4,2), the four original points per image
        dst: Tensor of shape (B,4,2), the four corresponding points per image
    Returns:
        A tensor of shape (B,3,3), each homography per image
    Raises:

    """
    return K.geometry.get_perspective_transform(src, dst)

def compute_homography(batch_corners_pred):
    """Compute the homography from the predictions of DeepHomoModel

    Arguments:
        batch_corners_pred: np.array of shape (B,8) with the predictions
    Returns:
        np.array of shape (B,3,3) with the homographies
    Raises:
        
    """
    batch_size = batch_corners_pred.shape[0]
    corners = get_corners_from_nn(batch_corners_pred)
    orig_corners = get_default_corners(batch_size)
    homography = get_perspective_transform_torch(
        to_torch(orig_corners), to_torch(corners)
    )
    return to_numpy(homography)


direct_homography_model = DeepHomoModel()
WEIGHTS_PATH = (
"https://storage.googleapis.com/narya-bucket-1/models/deep_homo_model.h5"
)
WEIGHTS_NAME = "deep_homo_model.h5"
WEIGHTS_TOTAR = False
checkpoints = tf.keras.utils.get_file(
                WEIGHTS_NAME, WEIGHTS_PATH, WEIGHTS_TOTAR,
            )
direct_homography_model.load_weights(checkpoints)

def getHomogrpahyMatrix(templatePath,image):
    corners = direct_homography_model(image)
    template = cv2.imread(templatePath)
    template = cv2.resize(template,(320, 320))
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    gt_h, gt_w, _ = template.shape
    pred_homo = compute_homography(corners)[0]
    print("Predicted homography: {}".format(pred_homo))  
    return pred_homo,gt_h, gt_w

def visualise_homography(image,templatePath,pred_homo):
    template = cv2.imread(templatePath)
    pred_warp = warp_image(np_img_to_torch_img(template),to_torch(pred_homo),method='torch')
    pred_warp = torch_img_to_np_img(pred_warp[0])
    image = cv2.resize(image,(320, 320))
    test = merge_template(image/255.,cv2.resize(pred_warp,(320, 320)))
    plt.imshow(test)
    plt.savefig('img.png')
    plt.show()
    