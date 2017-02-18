import os

import preprocess as pre


def main():
    input_folder = './Data/sample_images/'
    patients = os.listdir(input_folder)
    patients.sort()
    first_patient_scan = pre.load_scan(input_folder + patients[1])
    pre.plot_3d(first_patient_scan, 400)


if __name__ == '__main__':
    main()
