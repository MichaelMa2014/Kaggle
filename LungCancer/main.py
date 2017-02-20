import os

import preprocess as pre


def main():
    input_folder = './Data/sample_images/'
    patients = os.listdir(input_folder)
    patients.sort()
    first_patient_slices = pre.load_scan(input_folder + patients[1])
    first_patient_pixel = pre.slices_to_pixel(first_patient_slices)
    pre.plot_3d(first_patient_pixel, 400)
    first_patient_pixel_resampled, new_spacing = pre.resample(first_patient_pixel, first_patient_slices)
    pre.plot_3d(first_patient_pixel_resampled, 400)


if __name__ == '__main__':
    main()
