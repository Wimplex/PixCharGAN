import os
import shutil
# from google.colab import files


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


# def google_colab_zip_and_dwnload(dir_path, output_path):
#    shutil.make_archive(output_path, 'zip', dir_path)
#     files.download(output_path)


def int_to_grayscale_hex(value):
    return '#%02x%02x%02x' % (value, value, value)