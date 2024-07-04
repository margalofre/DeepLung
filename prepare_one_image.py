import numpy as np
import os
import SimpleITK as sitk
from scipy.ndimage import zoom
import pydicom

def load_scan(path):
    """
    Load DICOM files from a given folder path and sort them by ImagePositionPatient.
    
    Args:
        path (str): Directory path containing DICOM files.
        
    Returns:
        list: Sorted list of DICOM files.
    """
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        for s in slices:
            s.SliceThickness = slice_thickness
    except AttributeError:
        print("Error: Missing ImagePositionPatient data. Sorting may be incorrect.")
        return None
    except IndexError:
        print("Error: Less than two slices found in the directory.")
        return None
    return slices

def get_pixels_hu(slices):
    """
    Convert pixel values to Hounsfield Units (HU).
    
    Args:
        slices (list): List of DICOM slices.
        
    Returns:
        numpy.ndarray: 3D array of pixel values in HU.
    """
    image = np.stack([s.pixel_array for s in slices])
    return image.astype(np.int16)

def lum_trans(img):
    """
    Apply lung window transformation.
    
    Args:
        img (numpy.ndarray): 3D image array.
        
    Returns:
        numpy.ndarray: 3D image array with window transformation applied.
    """
    lungwin = np.array([-1200., 600.])
    img = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    img[img < 0] = 0
    img[img > 1] = 1
    return (img * 255).astype('uint8')

def load_itk_image(filename):
    """
    Load an ITK image and return its numpy representation, origin, and spacing.
    
    Args:
        filename (str): Path to the ITK image file.
        
    Returns:
        tuple: numpy representation of the image, origin, and spacing.
    """
    image = sitk.ReadImage(filename)
    numpy_image = sitk.GetArrayFromImage(image)
    spacing = np.array(list(image.GetSpacing()))
    origin = np.array(list(image.GetOrigin()))
    return numpy_image, origin, spacing

def resample_dicom(image, spacing, new_spacing=[1., 1., 1.], order=3):
    """
    Resample the 3D image to new spacing.
    
    Args:
        image (numpy.ndarray): 3D image array.
        spacing (numpy.ndarray): Current spacing.
        new_spacing (list): Desired spacing.
        order (int): The order of the spline interpolation.
        
    Returns:
        tuple: Resampled image and new spacing.
    """
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = zoom(image, real_resize_factor, order=order, mode='nearest')
    return image, new_spacing

def preprocess_single_dicom(dicom_path, savepath, resolution=[1, 1, 1]):
    """
    Preprocess a single DICOM dataset by resampling and saving the transformed data.
    
    Args:
        dicom_path (str): Directory path containing DICOM files.
        savepath (str): Directory path to save the processed data.
        resolution (list): Desired resolution for resampling.
    """
    dicom_patient = load_scan(dicom_path)
    if dicom_patient is None:
        print(f"Error: Could not load DICOM files from {dicom_path}")
        return
    
    spacing_ini = np.array([float(dicom_patient[0].SliceThickness),
                            float(dicom_patient[0].PixelSpacing[0]),
                            float(dicom_patient[0].PixelSpacing[1])])
    
    patient_hu = get_pixels_hu(dicom_patient)
    print(f"Original image shape: {patient_hu.shape}")
    
    patient_hu = lum_trans(patient_hu)
    
    patient_hu, spacing = resample_dicom(patient_hu, spacing_ini, resolution, order=1)
    print(f"Resampled image shape: {patient_hu.shape}")

    os.makedirs(savepath, exist_ok=True)
    np.save(os.path.join(savepath, 'test_VH_processed.npy'), patient_hu)
    print(f"Processed and saved DICOM data to {savepath}")

# Example usage
dicom_path = "/mnt/md1/VH_test/T1"
savepath = "/mnt/md1/VH_test/"
preprocess_single_dicom(dicom_path, savepath)
