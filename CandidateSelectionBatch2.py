import SimpleITK as sitk
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import csv
import glob
import os
import numpy as np
import shutil 
import time

def LoadImage(path):
	reader = sitk.ImageFileReader()
	reader.SetFileName(path)
	img = reader.Execute()
	# print 'Finish loading image'
	return img 

def SaveImage(img,path):
	writer = sitk.ImageFileWriter()
	writer.SetFileName(path)
	writer.Execute(img)
	# print 'Finish saving image'

if __name__ == '__main__':
	original_data_folder = '../data/original/lung_2016_subset0/'
	save_data_folder = '../data/candidates/lung_2016_subset0/'
	annotation_path = '../data/annotations.csv'

	output_spacing = [0.75,0.75,0.75]

	check_point = 1

	# load patient name
	files = glob.glob(original_data_folder + '*.mhd')
	patient_list = []

	for f in files:
		patient = os.path.basename(f)[0:-4]
		patient_list.append(patient)

	with open(annotation_path, 'rb') as f:
		reader = csv.reader(f)
		annotation = list(reader)

	ground_truth_total = 0
	candidate_total = 0
	true_positive_total = 0

	for patient in patient_list[2:3]:
		print 'Processing',patient

		save_dir = save_data_folder + '/'+patient
		img = LoadImage(save_dir + '/lung_enhanced.nii')
		label = LoadImage(save_dir + '/lung_label.nii')
		print 'Finish loading data'

		for i in xrange(6):
			t0 = time.time()

			# perform gaussian blurring
			print 'Laplacian of Gaussian blurring with kernel size',i+1
			LoGFilter = sitk.LaplacianRecursiveGaussianImageFilter()
			LoGFilter.SetSigma(i+1)
			LoG = LoGFilter.Execute(img)

			# print 'Computing gradient magnitude'
			# gradientMagnitudeFilter = sitk.GradientMagnitudeImageFilter()
			# gradMag = gradientMagnitudeFilter.Execute(img)
			
			# print 'Computing Laplacian'
			# laplacianFilter = sitk.LaplacianImageFilter()
			# laplacian = laplacianFilter.Execute(gaussianImg)

			# print 'Computing DNG'
			# LoG_np = sitk.GetArrayFromImage(LoG)
			# gradMag_np = sitk.GetArrayFromImage(gradMag)
			# img_np = sitk.GetArrayFromImage(img)
			# laplacian_np = sitk.GetArrayFromImage(laplacian)

			# DNG_np = LoG_np/gradMag_np
			# DNG = sitk.GetImageFromArray(DNG_np)
			# DNG.CopyInformation(img)

			print 'Computing LoG minimum...'
			if i ==0:
				LoG_min = LoG
			else:
				minFilter = sitk.MinimumImageFilter()
				LoG_min = minFilter.Execute(LoG_min,LoG)

			# # castFilter = sitk.CastImageFilter()
			# castFilter.SetOutputPixelType(1)
			# gradMag = castFilter.Execute(gradMag)
			# gradientFilter = sitk.GradientImageFilter()
			# grad = gradientFilter.Execute(img)

			# edgePotentialFilter = sitk.EdgePotentialImageFilter()
			# edgePotential = edgePotentialFilter.Execute(grad)

			# whiteTopHatFilter = sitk.WhiteTopHatImageFilter()
			# whiteTopHatFilter.SetKernelRadius(2)
			# whiteTopHatFilter.SetKernelType(1)
	 	# 	whiteTopHatFilter.SetSafeBorder(True)
			# whiteTopHat = whiteTopHatFilter.Execute(img)

			# blackTopHatFilter = sitk.BlackTopHatImageFilter()
			# blackTopHatFilter.SetKernelRadius(2)
			# blackTopHatFilter.SetKernelType(1)
	 	# 	blackTopHatFilter.SetSafeBorder(True)
			# blackTopHat = blackTopHatFilter.Execute(img)

			

			# print gradMag
			# print laplacian
			# divideFilter = sitk.DivideImageFilter()
			# img_norm = divideFilter.Execute(laplacian,gradMag)

			# thresholdFilter = sitk.BinaryThresholdImageFilter()
			# thresholdFilter.SetLowerThreshold(-500)
			# thresholdFilter.SetInsideValue(1)
			# thresholdFilter.SetOutsideValue(0)
			# thresholdFilter.SetUpperThreshold(4.5)
			# labelSitk = thresholdFilter.Execute(laplacian)

	 		t1 = time.time()

	 		print t1-t0


	 		# print 'Saving image'
	 		# SaveImage(gradMag,save_dir + '/lung_gradient_magnitude.nii')
	 		# SaveImage(gradMagNorm,save_dir + '/lung_gradient_magnitude_norm.nii')
	 		# SaveImage(whiteTopHat,save_dir + '/lung_white_top_hat.nii')
	 		# SaveImage(blackTopHat,save_dir + '/lung_blackTopHat.nii')
	 		# SaveImage(laplacian,save_dir + '/lung_laplacian.nii')

	 	# extract regional minimal within lung
	 	minimaFilter = sitk.RegionalMinimaImageFilter()
	 	DNG_min = minimaFilter.Execute(LoG_min)

		maskFilter = sitk.MaskImageFilter()
		DNG_min = maskFilter.Execute(LoG_min,label)

	 	SaveImage(DNG_min,save_dir + '/lung_DNG_min.nii')