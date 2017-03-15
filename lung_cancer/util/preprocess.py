# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import sys
import time

import dicom
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import scipy.ndimage

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


def slices_to_pixels(slices):
    """
    Convert scan slices of a patient to np array of HU values
    :param slices:
    :return: np array
    """
    image = np.array([s.pixel_array for s in slices])
    # Convert to int16 (from uint16),
    image = image.astype(np.int16)

    # Convert to Hounsfield units (HU)
    for i in range(len(slices)):
        slope = slices[i].RescaleSlope
        intercept = slices[i].RescaleIntercept
        image[i] = slope * image[i].astype(np.float64) + intercept
    return image


def load_pixels_by_patient(patient):
    """
    Load the pixels (np array) saved for this patient, if not found, try to construct the np array and save it
    :param patient: patient id
    :return:
    """
    path = INPUT_PATH + '/' + patient
    if not os.path.exists(path + '/' + patient + '.npy'):
        pixels = slices_to_pixels(load_slices(path))
        np.save(path + '/' + patient, pixels)
    else:
        pixels = np.load(path + '/' + patient + '.npy')
    return pixels


def plot_2d(pixel, patient="Unknown", i=0):
    """
    Draw one pixel and save to OUTPUT_PATH
    :param pixel: np array
    :param patient: patient id for output
    :param i: pixel id for output
    :return:
    """
    fig, ax = plt.subplots(1, 1)

    ax.axis('off')
    ax.set_title('Original')
    cax = ax.imshow(pixel)
    fig.colorbar(cax, ax=ax)

    fig.savefig(OUTPUT_PATH + '/%s_%s' % (patient, i), dpi=400)


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

    vertex, faces = measure.marching_cubes(p, threshold)

    # Fancy indexing: `vertex[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(vertex[faces], alpha=0.1)
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

    new_pixel = scipy.ndimage.interpolation.zoom(pixel, actual_resize_factor, mode='nearest'.encode('utf-8'))
    actual_new_spacing = spacing / actual_resize_factor
    return new_pixel, actual_new_spacing


def dcm_to_npy(patients):
    for patient in patients:
        path = INPUT_PATH + '/' + patient
        if not os.path.exists(path + '/' + patient + '.npy'):
            pixel = slices_to_pixels(load_slices(path))
            np.save(path + '/' + patient, pixel)
            print('%s scan shape' % patient, pixel.shape)
            sys.stdout.flush()
