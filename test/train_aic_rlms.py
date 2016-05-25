import sys
from pathlib import Path
import menpo.io as mio
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_66_trimesh
from menpo.feature import no_op, dsift
from menpo.transform import Similarity, AlignmentSimilarity
from menpofit.unified import UnifiedAAMCLM, UnifiedAAMCLMFitter, AICRLMS
from menpofit.aam import HolisticAAM
from menpofit.aam import LucasKanadeAAMFitter
import argparse
import pickle
import numpy as np
import csv
from functools import partial

test_group='face_ibug_66_trimesh'

fast_dsift = partial(dsift, fast=True, cell_size_vertical=3,
                     cell_size_horizontal=3, num_bins_horizontal=1,
                     num_bins_vertical=1, num_or_bins=8)

# This function was carried over from an old version to reproduce 
# always the same noise pattern for regression tests
def noisy_align(source, target, noise_std=0.04, rotation=False):
    r"""
    Constructs and perturbs the optimal similarity transform between source
    to the target by adding white noise to its weights.

    Parameters
    ----------
    source: :class:`menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment
    target: :class:`menpo.shape.PointCloud`
        The target pointcloud instance used in the alignment
    noise_std: float
        The standard deviation of the white noise

        Default: 0.04
    rotation: boolean
        If False the second parameter of the Similarity,
        which captures captures inplane rotations, is set to 0.

        Default:False

    Returns
    -------
    noisy_transform : :class: `menpo.transform.Similarity`
        The noisy Similarity Transform
    """
    transform = AlignmentSimilarity(source, target, rotation=rotation)
    parameters = transform.as_vector()
    parameter_range = np.hstack((parameters[:2], target.range()))
    noise = (parameter_range * noise_std *
             np.random.randn(transform.n_parameters))
    return Similarity.init_identity(source.n_dims).from_vector(parameters + noise)

def load_test_data(testset, n_test_imgs=None):
    test_images = []
    for i in mio.import_images(Path(testset), verbose=True, max_images=n_test_imgs):    
        i = i.crop_to_landmarks_proportion(0.5)
        labeller(i, 'PTS', face_ibug_68_to_face_ibug_66_trimesh)
        if i.n_channels == 3:
            i = i.as_greyscale(mode='average')
        test_images.append(i)

    return test_images

def train_aic_rlms(trainset, output, n_train_imgs=None):
    training_images = []
    # load landmarked images
    for i in mio.import_images(Path(trainset) / '*', verbose=True, max_images=n_train_imgs):
        # crop image
        i = i.crop_to_landmarks_proportion(0.5)
        labeller(i, 'PTS', face_ibug_68_to_face_ibug_66_trimesh)
        # convert it to greyscale if needed
        if i.n_channels == 3:
            i = i.as_greyscale(mode='average')
        # append it to the list
        training_images.append(i)

    offsets = np.meshgrid(range(-0, 1, 1), range(-0, 1, 1))
    offsets = np.asarray([offsets[0].flatten(), offsets[1].flatten()]).T 

    np.seterr(divide ='ignore')
    np.seterr(invalid ='ignore')    
    
    unified = UnifiedAAMCLM(training_images, 
                            parts_shape=(17, 17),
                            offsets=offsets,
                            group = test_group, 
                            holistic_features=fast_dsift, 
                            diagonal=100, 
                            scales=(1, .5), 
                            max_appearance_components = min(50,int(n_train_imgs/2)),
                            verbose=True) 

    n_appearance=[min(25,int(n_train_imgs/2)), min(50,int(n_train_imgs/2))]
    fitter = UnifiedAAMCLMFitter(unified, algorithm_cls=AICRLMS, n_shape=[3, 12], n_appearance=n_appearance)
    return fitter

def test_fitter(fitter, test_images):
    np.random.seed(seed=1)
    fitter_results = []
    for j, i in enumerate(test_images[:]):    
        gt_s = i.landmarks[test_group].lms
        s = noisy_align(fitter.reference_shape, gt_s, noise_std=0.04).apply(fitter.reference_shape)
        fr = fitter.fit_from_shape(i, s, gt_shape=gt_s, max_iters=50, prior=True)
        fitter_results.append(fr)    
        print(('Image: ', j))
        print(fr)

    return fitter_results

if __name__ == "__main__" :
    # Example:
    # C:\face_databases\lfpw\trainset  C:\face_databases\afw C:\face_databases\lfpw_unified.pickle
    parser = argparse.ArgumentParser(description='Train basic AAM model.')
    parser.add_argument('trainset', type=str, help='Path to training images folder')
    parser.add_argument('testset', type=str, help='Path to testing images folder')
    parser.add_argument('output', type=str, help='File path where to write the fitter object')
    parser.add_argument('--csv_results', type=str, help='File path where to write the results in CSV format')

    args = parser.parse_args()
    fitter = train_aic_rlms(args.trainset, args.output, 100)
    
    try:
        with open(args.output, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(fitter, f, pickle.HIGHEST_PROTOCOL)
    except:
        # don't skip the test because of an I/O error while pickling
        e = sys.exc_info()[0]
        print("Exception while saving the fitter",e)

    test_images = load_test_data(args.testset, 16)
    results = test_fitter(fitter, test_images)

    if args.csv_results is not None:
        with open(args.csv_results, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Initial Error', 'Final Error'])
            csv_writer.writerows([(f.initial_error(), f.final_error()) for f in results])
