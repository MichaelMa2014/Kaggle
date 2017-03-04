# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import time
import platform

import dicom
import matplotlib
import numpy as np  # linear algebra
import scipy.ndimage

if platform.system().find('Darwin') != -1:
    matplotlib.use('macosx')
else:
    matplotlib.use('agg')

import matplotlib.pyplot as plt

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from util import INPUT_PATH, OUTPUT_PATH, listdir_no_hidden


def load_slices(path):
    """
    Load the scans in given folder, which contains all scans from a patient
    :param path:
    :return: np array of HU values of the patient
    """
    slices = []
    for file_name in listdir_no_hidden(path):
        if not file_name.endswith('.npy'):
            slices.append(dicom.read_file(path + '/' + file_name))
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    # This works because of DICOM definition
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except KeyError:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    # Slick thickness is important in later resampling
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def slices_to_pixel(slices):
    """
    Convert scan slices of a patient to np array of HU values
    :param slices:
    :return: np array
    """
    # np.array() should also work
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    # TODO: Understand what `intercept` and `slope` is
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def load_pixel(path):
    """
    Load the pixel (np array) saved for this patient, if not found, try to construct the np array and save it
    :param path: Load the scans in given folder, which contains all scans from a patient
    :return:
    """
    patient_id = path.split('/')[-1]
    if not os.path.exists(path + '/' + patient_id + '.npy'):
        pixel = slices_to_pixel(load_slices(path))
        np.save(path + '/' + patient_id, pixel)
    else:
        pixel = np.load(path + '/' + patient_id + '.npy')
    return pixel


def plot_3d(pixel, threshold=-300):
    """
    Visualize all scan slices from a patient as a 3D image
    :param pixel: np array of HU values of the patient
    :param threshold: threshold for marching_cubes
    :return: No return. A plot was shown
    """
    print("plotting")

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = pixel.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    path = OUTPUT_PATH + '/' + str(int(time.time() * 100) % 100) + str(pixel.shape) + 'threshold=' + str(threshold)
    fig.savefig(path)
    print('figure saved to ' + path)


def resample(pixel, slices, new_spacing=(1, 1, 1)):
    """
    Use spline interpolation to resample the pixels so that spacing on 3 directions is equal
    :param pixel: np array of HU values of the patient
    :param slices: array of scan slices from the patient
    :param new_spacing:
    :return: resampled np array of HU values of the patient and actual new spacing
    """
    spacing = np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    actual_resize_factor = np.round(pixel.shape * resize_factor) / pixel.shape

    new_pixel = scipy.ndimage.interpolation.zoom(pixel, actual_resize_factor, mode='nearest')
    actual_new_spacing = spacing / actual_resize_factor
    return new_pixel, actual_new_spacing


def dcm_to_npy(patients):
    for patient in patients:
        path = INPUT_PATH + '/' + patient
        if not os.path.exists(path + '/' + patient + '.npy'):
            pixel = slices_to_pixel(load_slices(path))
            np.save(path + '/' + patient, pixel)
            print('%s scan shape' % patient, pixel.shape)
