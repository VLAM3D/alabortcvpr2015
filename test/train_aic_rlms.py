import sys
from pathlib import Path
import menpo.io as mio
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_66
from menpo.feature import no_op, dsift
from alabortcvpr2015.aam import PartsAAMBuilder
from alabortcvpr2015.unified import GlobalUnifiedBuilder
from alabortcvpr2015.clm.classifier import MCF, LinearSVMLR
from alabortcvpr2015.aam import PartsAAMFitter, AIC, PIC
from alabortcvpr2015.unified import GlobalUnifiedFitter 
import argparse
import pickle
import numpy as np
import csv
from functools import partial

fast_dsift = partial(dsift, fast=True, cell_size_vertical=5,
                     cell_size_horizontal=5, num_bins_horizontal=1,
                     num_bins_vertical=1, num_or_bins=8)

def load_test_data(testset, n_test_imgs=None):
    test_images = []
    for i in mio.import_images(Path(testset), verbose=True, max_images=n_test_imgs):    
        i = i.crop_to_landmarks_proportion(0.5)
        labeller(i, 'PTS', face_ibug_68_to_face_ibug_66)
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
        labeller(i, 'PTS', face_ibug_68_to_face_ibug_66)
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

    unified = builder.build(training_images, group='face_ibug_66', verbose=True)
    fitter = GlobalUnifiedFitter(unified, n_shape=[3, 12], n_appearance=[25, 50])

    return fitter

def test_fitter(fitter, test_images):
    np.random.seed(seed=1)
    fitter_results = []
    for j, i in enumerate(test_images[:]):    
        gt_s = i.landmarks['face_ibug_66'].lms
        s = fitter.perturb_shape(gt_s, noise_std=0.04)    
        fr = fitter.fit(i, s, gt_shape=gt_s, max_iters=50, prior=True)
        fr.downscale = 0.5    
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
