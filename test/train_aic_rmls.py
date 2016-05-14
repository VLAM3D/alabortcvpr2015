from pathlib import Path
import menpo.io as mio
from menpofast.utils import convert_from_menpo
from menpo.landmark import labeller, ibug_face_66
from menpofast.feature import no_op, fast_dsift, fast_daisy
from alabortcvpr2015.aam import PartsAAMBuilder
from alabortcvpr2015.unified import GlobalUnifiedBuilder
from alabortcvpr2015.clm.classifier import MCF
from alabortcvpr2015.aam import PartsAAMFitter, AIC, PIC
from alabortcvpr2015.unified import GlobalUnifiedFitter 
import argparse
import pickle
import numpy as np

def load_test_data(testset):
    test_images = []
    for i in mio.import_images(Path(testset), verbose=True, max_images=16):    
        # convert the image from menpo Image to menpofast Image (channels at front)
        i = convert_from_menpo(i)    
        i.crop_to_landmarks_proportion_inplace(0.5)
        labeller(i, 'PTS', ibug_face_66)
        if i.n_channels == 3:
            i = i.as_greyscale(mode='average')
        test_images.append(i)

    return test_images

def train_aic_rmls(trainset, output):
    training_images = []
    # load landmarked images
    for i in mio.import_images(Path(trainset) / '*', verbose=True, max_images = None):
        # crop image
        i = convert_from_menpo(i)
        i.rescale_landmarks_to_diagonal_range(200)
        i.crop_to_landmarks_proportion_inplace(0.5)
        labeller(i, 'PTS', ibug_face_66)
        # convert it to greyscale if needed
        if i.n_channels == 3:
            i = i.as_greyscale(mode='average')
        # append it to the list
        training_images.append(i)

    offsets = np.meshgrid(range(-0, 1, 1), range(-0, 1, 1))
    offsets = np.asarray([offsets[0].flatten(), offsets[1].flatten()]).T 

    builder = GlobalUnifiedBuilder(parts_shape=(17, 17), features=fast_dsift, diagonal=100, 
                                   classifier=MCF, offsets=offsets, normalize_parts=False, 
                                   covariance=2, scale_shapes=False, scales=(1, .5),  max_appearance_components = 50)

    unified = builder.build(training_images, group='ibug_face_66', verbose=True)
    fitter = GlobalUnifiedFitter(unified, n_shape=[3, 12], n_appearance=[25, 50])

    return fitter

def test_fitter(fitter, test_images):
    np.random.seed(seed=1)
    fitter_results = []
    for j, i in enumerate(test_images[:]):    
        gt_s = i.landmarks['ibug_face_66'].lms
        s = fitter.perturb_shape(gt_s, noise_std=0.04)    
        fr = fitter.fit(i, s, gt_shape=gt_s, max_iters=50, prior=True)
        fr.downscale = 0.5    
        fitter_results.append(fr)    
        print(('Image: ', j))
        print(fr)

    return fitter_results

if __name__ == "__main__" :
    # Example:
    # C:\face_databases\lfpw C:\face_databases\lfpw_unified.pickle
    parser = argparse.ArgumentParser(description='Train basic AAM model.')
    parser.add_argument('trainset', type=str, help='Path to training images folder')
    parser.add_argument('testset', type=str, help='Path to testing images folder')
    parser.add_argument('output', type=str, help='File path where write the AAM')

    args = parser.parse_args()
    fitter = train_aic_rmls(args.trainset, args.output)
    test_images = load_test_data(args.testset)
    test_fitter(fitter, test_images)