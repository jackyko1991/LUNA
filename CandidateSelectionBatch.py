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

'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return int(voxel_coordinates)

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def CropImage(img,center,extend):
	roiFilter = sitk.RegionOfInterestImageFilter()
	roiFilter.SetSize([extend*2+1,extend*2+1,extend*2+1])
	roiFilter.SetIndex([int(center[0]-extend),int(center[1]-extend),int(center[2]-extend)])
	imgCrop = roiFilter.Execute(img)
	return imgCrop

def ExtractNodule(img):
	addFilter = sitk.AddImageFilter()
	img = addFilter.Execute(img,1000)

	# denoising
	print 'Denoising...'
	blurFilter = sitk.CurvatureFlowImageFilter()
	blurFilter.SetNumberOfIterations(5)
	blurFilter.SetTimeStep(0.125)
	img = blurFilter.Execute(img)

	# # apply Otsu on the lung
	# thresholdFilter = sitk.OtsuMultipleThresholdsImageFilter()
	# thresholdFilter.SetNumberOfThresholds(2)
	# labelSitk = thresholdFilter.Execute(img)

	# # lung label
	# thresholdFilter = sitk.BinaryThresholdImageFilter()
	# thresholdFilter.SetInsideValue(1)
	# thresholdFilter.SetOutsideValue(0)
	# thresholdFilter.SetLowerThreshold(0.5)
	# thresholdFilter.SetUpperThreshold(1.5) 
	# labelSitk = thresholdFilter.Execute(labelSitk)

	# lung label
	thresholdFilter = sitk.BinaryThresholdImageFilter()
	thresholdFilter.SetInsideValue(1)
	thresholdFilter.SetOutsideValue(0)
	thresholdFilter.SetLowerThreshold(0)
	thresholdFilter.SetUpperThreshold(300) 
	labelSitk = thresholdFilter.Execute(img)

	# # VotingBinaryHoleFillingImageFilter to extend the segmentation
	# fillingFilter = sitk.VotingBinaryHoleFillingImageFilter()
	# fillingFilter.SetRadius(5)
	# fillingFilter.SetBackgroundValue(0)
	# fillingFilter.SetForegroundValue(1)
	# labelSitk = fillingFilter.Execute(labelSitk)

	labelNP = sitk.GetArrayFromImage(labelSitk)

	for i in xrange(labelNP.shape[0]):
		if i % 25 == 0:
			print 'Processing Layer:', i,'/',str(labelNP.shape[0])
		# Remove the blobs connected to the border of the image.
		labelNP[i,] = clear_border(labelNP[i,])

		# # Label the image.
		# labelNP[i,] = label(labelNP[i,])
	
	labelSitk = sitk.GetImageFromArray(labelNP)	
	labelSitk.CopyInformation(img)

	# Closure operation with a sphere of radius 10. This operation is to keep nodules attached to the lung wall.
	print 'Image closing'
	closeFilter = sitk.BinaryMorphologicalClosingImageFilter()
	closeFilter.SetKernelRadius(10)
	closeFilter.SetForegroundValue(1)
	closeFilter.SetKernelType(1)
	labelSitk = closeFilter.Execute(labelSitk)

	# # remove small isolated volumes
	# openFilter = sitk.BinaryMorphologicalOpeningImageFilter()
	# openFilter.SetKernelRadius(3)
	# openFilter.SetBackgroundValue(0)
	# openFilter.SetForegroundValue(1)
	# openFilter.SetKernelType(1)
	# labelSitk = openFilter.Execute(labelSitk)

	# # isloate the lungs, using image opening with erosion and dialtion with different kernel size
	# print 'Image opening'
	# erodeFilter = sitk.BinaryErodeImageFilter()
	# erodeFilter.SetKernelRadius(3)
	# erodeFilter.SetBackgroundValue(0)
	# erodeFilter.SetForegroundValue(1)
	# erodeFilter.SetKernelType(1)
	# labelSitk = erodeFilter.Execute(labelSitk)

	# dilateFilter = sitk.BinaryDilateImageFilter()
	# dilateFilter.SetKernelRadius(4)
	# dilateFilter.SetBackgroundValue(0)
	# dilateFilter.SetForegroundValue(1)
	# dilateFilter.SetKernelType(1)
	# labelSitk = dilateFilter.Execute(labelSitk)



	# Fill in the small holes inside the binary mask of lungs.
	print 'Filling holes'
	fillHoleFilter = sitk.BinaryFillholeImageFilter()
	labelSitk = fillHoleFilter.Execute(labelSitk)

	# extract lung
	maskFilter = sitk.MaskImageFilter()
	lung = maskFilter.Execute(img,labelSitk)

	# # apply Otsu on the lung
	# thresholdFilter = sitk.OtsuMultipleThresholdsImageFilter()
	# thresholdFilter.SetNumberOfThresholds(4)
	# nodule_label = thresholdFilter.Execute(lung)

	# thresholdFilter = sitk.BinaryThresholdImageFilter()
	# thresholdFilter.SetLowerThreshold(3.5)
	# thresholdFilter.SetInsideValue(1)
	# thresholdFilter.SetOutsideValue(0)
	# thresholdFilter.SetUpperThreshold(4.5)
	# nodule_label = thresholdFilter.Execute(nodule_label)

	# apply thresholding on lung
	thresholdFilter = sitk.BinaryThresholdImageFilter()
	thresholdFilter.SetLowerThreshold(600)
	thresholdFilter.SetInsideValue(1)
	thresholdFilter.SetOutsideValue(0)
	thresholdFilter.SetUpperThreshold(3000) # threshold value by experiment
	nodule_label = thresholdFilter.Execute(lung)

	# # remove small isolated volumes, not recommened if there is are small blobs
	# openFilter = sitk.BinaryMorphologicalOpeningImageFilter()
	# openFilter.SetKernelRadius(1)
	# openFilter.SetBackgroundValue(0)
	# openFilter.SetForegroundValue(1)
	# openFilter.SetKernelType(1)
	# nodule_label = openFilter.Execute(nodule_label)

	# return lung,nodule_label

	# statFilter = sitk.LabelStatisticsImageFilter()
	# statFilter.Execute(lung, nodule_label)
	# print statFilter.GetBoundingBox(1)
	# print lung.GetSize()

	# # buff = 2
	# # crop image to reduce size
	# size = [int(statFilter.GetBoundingBox(1)[1]-statFilter.GetBoundingBox(1)[0]),\
	# 	int(statFilter.GetBoundingBox(1)[3]-statFilter.GetBoundingBox(1)[2]),\
	# 	int(statFilter.GetBoundingBox(1)[5]-statFilter.GetBoundingBox(1)[4])]
	# # print size
	# roiFilter = sitk.RegionOfInterestImageFilter()
	# roiFilter.SetIndex([statFilter.GetBoundingBox(1)[0],statFilter.GetBoundingBox(1)[2],statFilter.GetBoundingBox(1)[4]])
	# roiFilter.SetSize(size)
	# lung = roiFilter.Execute(lung)
	# nodule_label = roiFilter.Execute(nodule_label)
	# labelSitk = roiFilter.Execute(labelSitk)
	# img_crop = roiFilter.Execute(addFilter.Execute(img,-1000))

	return lung,nodule_label,labelSitk,img

def ConnectedComponent(labelImage):
	ccFilter = sitk.ConnectedComponentImageFilter()
	label = ccFilter.Execute(labelImage)

	labelShapeStatFilter = sitk.LabelShapeStatisticsImageFilter()
	labelShapeStatFilter.Execute(label)

	return label,labelShapeStatFilter

def RemoveVessels(label,kernel_size):
	# image opening to remove small isolated volumes
	print 'Processing image opening with kernel size:', kernel_size
	openFilter = sitk.BinaryMorphologicalOpeningImageFilter()
	openFilter.SetKernelRadius(kernel_size)
	openFilter.SetBackgroundValue(0)
	openFilter.SetForegroundValue(1)
	openFilter.SetKernelType(1)
	blobs = openFilter.Execute(label)

	return blobs

def GetLabelCentroid(label,labelShapeFilter,label_num):
	centroid = labelShapeFilter.GetCentroid(label_num)
	origin = label.GetOrigin()
	spacing = label.GetSpacing()

	voxel_coord = []
	for i in xrange(3):
		voxel_coord.append(world_2_voxel(centroid[i],origin[i],spacing[i]))
	return voxel_coord

def MutualInformation(hgram):
	""" Mutual information for joint histogram
	"""
	# Convert bins counts to probability values
	pxy = hgram / float(np.sum(hgram))
	px = np.sum(pxy, axis=1) # marginal for x over y
	py = np.sum(pxy, axis=0) # marginal for y over x
	px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
	# Now we can do the calculation using the pxy, px_py 2D arrays
	nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
	return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def Similarity(img1,img2):
	img1_np = sitk.GetArrayFromImage(img1)
	img2_np = sitk.GetArrayFromImage(img2)

	corrCoef = np.corrcoef(img1_np.ravel(), img2_np.ravel())[0, 1]
	# hist_2d, x_edges, y_edges = np.histogram2d(img1_np.ravel(),img2_np.ravel(),bins=20)
	# mutInfo = MutualInformation(hist_2d)

	# return corrCoef,mutInfo
	return corrCoef

def dist(p1, p2):
	return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def fuse(points, d):
	ret = []
	n = len(points)
	taken = [False] * n
 	for i in range(n):
	    if not taken[i]:
	        count = 1
	        point = [points[i][0], points[i][1], points[i][2]]
	        taken[i] = True
	        for j in range(i+1, n):
	            if dist(points[i], points[j]) <=d:
	                point[0] += points[j][0]
	                point[1] += points[j][1]
	                point[2] += points[j][2]
	                count+=1
	                taken[j] = True
	        point[0] /= count
	        point[1] /= count
	        point[2] /= count
	        ret.append((point[0], point[1],point[2]))
	return ret

if __name__ == '__main__':
	original_data_folder = '../data/original/mixed/'
	save_data_folder = '../data/candidates/mixed/'
	annotation_path = '../data/annotations.csv'

	output_spacing = [1,1,1]

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

	for patient in patient_list[0:]:
		save_dir = save_data_folder + '/'+patient

		print 'Processing patient', patient

		# check point 1 start
		check_point_1_end = False
		if check_point < 2:
 			#load image
			print original_data_folder + patient + '.mhd'
			original_img = LoadImage(original_data_folder + patient + '.mhd')

			if os.path.isdir(save_data_folder + patient):
				# empty output folders
				for root, dirs, files in os.walk(save_data_folder + patient, topdown=False):
					for name in files:
						os.remove(os.path.join(root, name))
					for name in dirs:
						os.rmdir(os.path.join(root, name))
			else:
				os.mkdir(save_data_folder + patient)

			resampler = sitk.ResampleImageFilter()
			resampler.SetOutputSpacing(output_spacing)
			resampler.SetOutputOrigin(original_img.GetOrigin())
			resampler.SetOutputDirection(original_img.GetDirection())
			resampler.SetInterpolator(sitk.sitkLinear)  # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results. 

			original_size = original_img.GetSize()

			# calulate the new image size
			dx = int(original_size[0]*original_img.GetSpacing()[0]/output_spacing[0])
			dy = int(original_size[1]*original_img.GetSpacing()[1]/output_spacing[1])
			dz = int((original_size[2]-1)*original_img.GetSpacing()[2]/output_spacing[2])
			resampler.SetSize([dx,dy,dz])

			print 'Resampling original image to isotropic...'
			resampled_img = resampler.Execute(original_img)
			# lung_enhanced = resampler.Execute(lung_enhanced)
			# nodule = resampler.Execute(nodule)
			# lung_label = resampler.Execute(lung_label)

			# extract nodule label
			print 'Extracting lung...'
			lung,nodule, lung_label,resampled_img = ExtractNodule(resampled_img)

			print 'Saving isotrpoic image and labels...'
			SaveImage(resampled_img, save_dir+'/resampled.nii')
			SaveImage(lung, save_dir+'/lung.nii')
			SaveImage(lung_label, save_dir+'/lung_label.nii')
			SaveImage(nodule, save_dir+'/nodule_label.nii')

			check_point_1_end = True
			# check point 1 end

		check_point_2_end = False
		if check_point < 3:
			# check point 2 start
			print 'Extracting ground truth...'
			# extract ground truth
			if check_point_1_end == False:
				resampled_img = LoadImage(save_dir+'/resampled.nii')
				lung = LoadImage(save_dir+'/lung.nii')

			ground_truth_count = 0

			if os.path.isdir(save_dir + '/ground_truth'):
				# empty output folders
				files = glob.glob(save_dir + '/ground_truth/*')
				for f in files:
					os.remove(f)
			else:
				os.mkdir(save_dir + '/ground_truth')

			for i in xrange(len(annotation)):
				if annotation[i][0] == patient:
					ground_truth_count = ground_truth_count + 1 
					# print i

					ground_truth_center_world = [0,0,0]
					for j in xrange(3):
						ground_truth_center_world[j] = np.float(annotation[i][j+1])
	
					ground_truth_center_voxel = [0,0,0]
					for j in xrange(3):
						ground_truth_center_voxel[j] = world_2_voxel(ground_truth_center_world[j], resampled_img.GetOrigin()[j], resampled_img.GetSpacing()[j])

					# crop the ground truth
					lungCrop = CropImage(lung, ground_truth_center_voxel, 25)
					SaveImage(lungCrop, save_dir+'/ground_truth/'+ str(ground_truth_count) +'.nii')
			
			check_point_2_end = True
			# check point 2 end

		check_point_3_end = False
		if check_point < 4:
			# check point 3 start
			if check_point_1_end == False:
				# resampled_img = LoadImage(save_dir+'/resampled.nii')
				lung = LoadImage(save_dir+'/lung.nii')
				lung_label = LoadImage(save_dir+'/lung_label.nii')
				nodule = LoadImage(save_dir+'/nodule_label.nii')

			# Candidate selection by Laplacian of Guassian blur
			for i in xrange(6):
				# perform gaussian blurring
				print 'Laplacian of Gaussian blurring with kernel size',i+1
				LoGFilter = sitk.LaplacianRecursiveGaussianImageFilter()
				LoGFilter.SetSigma(i+1)
				LoG = LoGFilter.Execute(lung)

				if i ==0:
					LoG_min = LoG
				else:
					minFilter = sitk.MinimumImageFilter()
					LoG_min = minFilter.Execute(LoG_min,LoG)

			# save LoG image
			SaveImage(LoG_min, save_dir+'/LoG_minimum.nii')

	 		# extract regional minimal within lung
	 		minimaFilter = sitk.RegionalMinimaImageFilter()
	 		minimaFilter.SetFlatIsMinima(False)
	 		LoG_min = minimaFilter.Execute(LoG_min)

	 		# # extract regional minimal within lung
	 		minimaFilter = sitk.HMinimaImageFilter()
	 		# minimaFilter.SetHeight(0)
	 		# LoG_min = minimaFilter.Execute(LoG_min)

			maskFilter = sitk.MaskImageFilter()
			LoG_min = maskFilter.Execute(LoG_min,nodule)

			# # shrink the lung label to remove candidates at edge
			# erodeFilter = sitk.BinaryErodeImageFilter()
			# erodeFilter.SetKernelRadius(1)
			# erodeFilter.SetBackgroundValue(0)
			# erodeFilter.SetForegroundValue(1)
			# erodeFilter.SetKernelType(1)
			# lung_label_shrink = erodeFilter.Execute(lung_label)

			# LoG_min = maskFilter.Execute(LoG_min,lung_label_shrink)

			SaveImage(LoG_min, save_dir+'/LoG_candidate.nii')

			check_point_3_end = True
			# # remove vessels
			# # create folder to save the connect compoenent labels
			# if os.path.isdir(save_data_folder + patient+'/nodule_label_CC'):
			# 	# empty output folders
			# 	for root, dirs, files in os.walk(save_data_folder + patient+'/nodule_label_CC', topdown=False):
			# 		for name in files:
			# 			os.remove(os.path.join(root, name))
			# 		for name in dirs:
			# 			os.rmdir(os.path.join(root, name))
			# else:
			# 	os.mkdir(save_data_folder + patient+'/nodule_label_CC')

			# # create folder to save the cropped candidates
			# if os.path.isdir(save_data_folder + patient+'/cropped_candidates'):
			# 	# empty output folders
			# 	for root, dirs, files in os.walk(save_data_folder + patient+'/cropped_candidates', topdown=False):
			# 		for name in files:
			# 			os.remove(os.path.join(root, name))
			# 		for name in dirs:
			# 			os.rmdir(os.path.join(root, name))
			# else:
			# 	os.mkdir(save_data_folder + patient+'/cropped_candidates')

			# for i in xrange(6):
			# 	if i in xrange(1):
			# 		blobs = nodule
			# 		continue
			# 	else:
			# 		kernel_size = i
			# 		blobs = RemoveVessels(nodule, kernel_size)

			# 	# save the blob label after remove vessel
			# 	SaveImage(blobs, save_dir+'/nodule_label_CC/' + str(i) +'.nii')

			check_point_3_end = True
			# check point 3 end

		check_point_4_end = False
		if check_point < 5:
			# check point 4 start
			if check_point_3_end == False:
				resampled_img = LoadImage(save_dir+'/resampled.nii')
				lung = LoadImage(save_dir+'/lung.nii')
				nodule = LoadImage(save_dir+'/nodule_label.nii')
				LoG_min = LoadImage(save_dir+'/LoG_candidate.nii')

			candidate = []
			print 'Extracting candidates...'
			lung = LoadImage(save_dir+'/lung.nii')
			
			blobs,blobsShapeFilter = ConnectedComponent(LoG_min)
			origin = blobs.GetOrigin()
			spacing = blobs.GetSpacing()

			for i in xrange(blobsShapeFilter.GetNumberOfLabels()+1):
				if i == 0:
					continue

				centroid = blobsShapeFilter.GetCentroid(i)
				world_coord = centroid

				voxel_coord = []
				for j in xrange(3):
					voxel_coord.append(world_2_voxel(centroid[j],origin[j],spacing[j]))

				# check if the centroid is too close to edge
				voxel_coord_min = np.asarray(voxel_coord)-25
				voxel_coord_max = np.asarray(voxel_coord)+25

				if voxel_coord_min[0] < 0 or  voxel_coord_min[1] < 0 or voxel_coord_min[2] < 0:
					continue
				if voxel_coord_max[0] >= lung.GetSize()[0] or  voxel_coord_max[1] >= lung.GetSize()[1] or voxel_coord_max[2] >= lung.GetSize()[2]:
					continue

				candidate.append(world_coord)

			# candidate fuse
			print 'Number of candidates before fuse =', len(candidate)
			candidate = fuse(candidate, 2.5)
			print 'Number of candidates after fuse =', len(candidate)

			# check similarity with ground truth
			ground_truth_files = glob.glob(save_data_folder + patient+'/ground_truth/' + '*.nii')
			ground_truth_list = []

			for ground_truth_filename in ground_truth_files:
				name = os.path.basename(ground_truth_filename)[0:-4]
				ground_truth_list.append(name)

			candidate_total = candidate_total + len(candidate)

			true_positive_list = []
			if len(ground_truth_list) == 0:
				ground_truth_total = ground_truth_total
				true_positive_total = true_positive_total
			else:
				for i in xrange(len(ground_truth_list)):
					ground_truth_total = ground_truth_total + 1
					ground_truth_img = LoadImage(save_data_folder + patient+'/ground_truth/' + ground_truth_list[i] + '.nii')
					for j in xrange(len(candidate)):
						voxel_coord = []
						for k in xrange(3):
							voxel_coord.append(world_2_voxel(candidate[j][k],origin[k],spacing[k]))
						candidate_img = CropImage(lung,voxel_coord,25)

						# corrCoef, mutInfo = Similarity(ground_truth_img,candidate_img)
						# print 'Similarity check:',i,'    ',j,'/',len(candidate)
						corrCoef = Similarity(ground_truth_img,candidate_img)
						if corrCoef > 0.6: # similarity with ground truth higher than 0.6 is considered to be true positive
							true_positive_list.append(candidate[j])
							# print ground_truth_list[i],candidate[j]
							# continue

			# remove repeated candidates
			discard_list = []
			print 'Removing repeated candidates...'
			for i in xrange(len(true_positive_list)):
				if i == 0:
					continue

				voxel_coord_1 = []
				for k in xrange(3):
					voxel_coord_1.append(world_2_voxel(true_positive_list[i-1][k],origin[k],spacing[k]))
				img1 = CropImage(lung,voxel_coord_1,25)
				for j in xrange(len(true_positive_list)-i):
					# print cropped_list[i+j]
					voxel_coord_2 = []
					for k in xrange(3):
						voxel_coord_2.append(world_2_voxel(true_positive_list[i+j][k],origin[k],spacing[k]))
					img2 = CropImage(lung,voxel_coord_2,25)

					# corrCoef, mutInfo = Similarity(img1,img2)
					corrCoef= Similarity(img1,img2)
					if corrCoef > 0.6:
						discard_list.append(i)
						continue

			true_positive_list_clean = []
			for i in xrange(len(true_positive_list)):
				print true_positive_list[i]
				voxel_coord = []
				for j in xrange(3):
					voxel_coord.append(world_2_voxel(true_positive_list[i][j],origin[j],spacing[j]))
				print voxel_coord
				if not (i in discard_list):
					true_positive_list_clean.append(true_positive_list[i])

			true_positive_total = true_positive_total + len(true_positive_list_clean)

			print 'True positive found:',len(true_positive_list_clean),'/',len(ground_truth_list)
			print 'Number of candidates:',len(candidate)

			check_point_4_end = True
			# check point 4 end

			# for i in xrange(len(candidate)):
			# 	print candidate[i]


			# print true_positive_list_clean


			# for i in xrange(6):
			# 	if i in xrange(1):
			# 		continue
		# 		else:
		# 			blobs = []
		# 			blobs = LoadImage(save_dir+'/nodule_label_CC/' + str(i) +'.nii')
		# 			origin = blobs.GetOrigin()
		# 			spacing = blobs.GetSpacing()
		# 			blobs,blobsShapeFilter = ConnectedComponent(blobs)

		# 			labelStatFilter = sitk.LabelStatisticsImageFilter()
		# 			# print lung_enhanced
		# 			# print blobs
		# 			labelStatFilter.Execute(lung_enhanced,blobs)
		# 			for j in xrange(blobsShapeFilter.GetNumberOfLabels()+1):
		# 				if j == 0:
		# 					continue

		# 				centroid = blobsShapeFilter.GetCentroid(j)
		# 				world_coord = centroid

		# 				voxel_coord = []
		# 				for k in xrange(3):
		# 					voxel_coord.append(world_2_voxel(centroid[k],origin[k],spacing[k]))


		# 				# check if the centroid is too close to edge
		# 				voxel_coord_min = np.asarray(voxel_coord)-25
		# 				voxel_coord_max = np.asarray(voxel_coord)+25

		# 				if voxel_coord_min[0] < 0 or  voxel_coord_min[1] < 0 or voxel_coord_min[2] < 0:
		# 					continue
		# 				if voxel_coord_max[0] >= lung_enhanced.GetSize()[0] or  voxel_coord_max[1] >= lung_enhanced.GetSize()[1] or voxel_coord_max[2] >= lung_enhanced.GetSize()[2]:
		# 					continue

		# 				# volume thresholding, it is known that the volume of a nodule would not exceed 35000mm^3, nodule radius <= 20mm, this condition could be lossen
		# 				volume = labelStatFilter.GetCount(j)*output_spacing[0]*output_spacing[1]*output_spacing[2]
		# 				if volume > 35000:
		# 					continue

		# 				candidate.append(world_coord)


		# 	# check similarity with ground truth
		# 	ground_truth_files = glob.glob(save_data_folder + patient+'/ground_truth/' + '*.nii')
		# 	ground_truth_list = []

		# 	for ground_truth_filename in ground_truth_files:
		# 		name = os.path.basename(ground_truth_filename)[0:-4]
		# 		ground_truth_list.append(name)

		# 	candidate_total = candidate_total + len(candidate)

		# 	true_positive_list = []
		# 	if len(ground_truth_list) == 0:
		# 		ground_truth_total = ground_truth_total
		# 		true_positive_total = true_positive_total
		# 	else:
		# 		for i in xrange(len(ground_truth_list)):
		# 			ground_truth_total = ground_truth_total + 1
		# 			ground_truth_img = LoadImage(save_data_folder + patient+'/ground_truth/' + ground_truth_list[i] + '.nii')
		# 			for j in xrange(len(candidate)):
		# 				voxel_coord = []
		# 				for k in xrange(3):
		# 					voxel_coord.append(world_2_voxel(candidate[j][k],origin[k],spacing[k]))
		# 				candidate_img = CropImage(lung_enhanced,voxel_coord,25)

		# 				# corrCoef, mutInfo = Similarity(ground_truth_img,candidate_img)
		# 				# print 'Similarity check:',i,'    ',j,'/',len(candidate)
		# 				corrCoef = Similarity(ground_truth_img,candidate_img)
		# 				if corrCoef > 0.65: # similarity with ground truth higher than 0.6 is considered to be true positive
		# 					true_positive_list.append(candidate[j])
		# 					# print ground_truth_list[i],candidate[j]
		# 					# continue

		# 	# remove repeated candidates
		# 	discard_list = []
		# 	print 'Removing repeated candidates...'
		# 	for i in xrange(len(true_positive_list)):
		# 		if i == 0:
		# 			continue

		# 		voxel_coord_1 = []
		# 		for k in xrange(3):
		# 			voxel_coord_1.append(world_2_voxel(true_positive_list[i-1][k],origin[k],spacing[k]))
		# 		img1 = CropImage(lung_enhanced,voxel_coord_1,25)
		# 		for j in xrange(len(true_positive_list)-i):
		# 			# print cropped_list[i+j]
		# 			voxel_coord_2 = []
		# 			for k in xrange(3):
		# 				voxel_coord_2.append(world_2_voxel(true_positive_list[i+j][k],origin[k],spacing[k]))
		# 			img2 = CropImage(lung_enhanced,voxel_coord_2,25)

		# 			# corrCoef, mutInfo = Similarity(img1,img2)
		# 			corrCoef= Similarity(img1,img2)
		# 			if corrCoef > 0.6:
		# 				discard_list.append(i)
		# 				continue

		# 	true_positive_list_clean = []
		# 	for i in xrange(len(true_positive_list)):
		# 		# print true_positive_list[i]
		# 		if not (i in discard_list):
		# 			true_positive_list_clean.append(true_positive_list[i])

		# 	true_positive_total = true_positive_total + len(true_positive_list_clean)

		# 	print 'True positive found:',len(true_positive_list_clean),'/',len(ground_truth_list)
		# 	print 'Number of candidates:',len(candidate)

		# 	check_point_4_end = True
			# check point 4 end

			# for i in xrange(len(candidate)):
			# 	print candidate[i]


			# print true_positive_list_clean

	# 	check_point_4_end = False
	# 	if check_point < 5:
	# 		# check point 4 start
	# 		# check similarity among all cropped candidateds
	# 		cropped_files = glob.glob(save_data_folder + patient+'/cropped_candidates/' + '*.nii')
	# 		cropped_list = []

	# 		for cropped_filename in cropped_files:
	# 			name = os.path.basename(cropped_filename)[0:-4]
	# 			cropped_list.append(name)

	# 		# create folder to save the similarity thresholded candidates
	# 		if os.path.isdir(save_data_folder + patient+'/cropped_candidates_similarity_threshold'):
	# 			# empty output folders
	# 			for root, dirs, files in os.walk(save_data_folder + patient+'/cropped_candidates_similarity_threshold', topdown=False):
	# 				for name in files:
	# 					os.remove(os.path.join(root, name))
	# 				for name in dirs:
	# 					os.rmdir(os.path.join(root, name))
	# 		else:
	# 			os.mkdir(save_data_folder + patient+'/cropped_candidates_similarity_threshold')
			

	# 		discard_list = []
	# 		print 'Checking similarity...'
	# 		for i in xrange(len(cropped_list)):
	# 			if i % 20 == 0:
	# 				print 'Progress:',i+1,'/',len(cropped_list)
	# 			if i == 0:
	# 				continue
	# 			# print 'img1',cropped_list[i-1]
	# 			img1 = LoadImage(save_data_folder + patient+'/cropped_candidates/'+ cropped_list[i-1] +'.nii')
	# 			for j in xrange(len(cropped_list)-i):
	# 				# print cropped_list[i+j]
	# 				img2 = LoadImage(save_data_folder + patient+'/cropped_candidates/'+ cropped_list[i+j] +'.nii')

	# 				# corrCoef, mutInfo = Similarity(img1,img2)
	# 				corrCoef= Similarity(img1,img2)
	# 				if corrCoef > 0.6:
	# 					discard_list.append(cropped_list[i-1])
	# 					continue

	# 		for i in xrange(len(cropped_list)):
	# 			if cropped_list[i] in discard_list:
	# 				continue
	# 			shutil.copyfile(save_data_folder + patient+'/cropped_candidates/'+ cropped_list[i] +'.nii',\
	# 				save_data_folder + patient+'/cropped_candidates_similarity_threshold/'+ cropped_list[i]+'.nii')

	# 		check_point_4_end = True
	# 		# check point 4 end

	# 	check_point_5_end = False
	# 	if check_point < 6:
	# 		# check point 5 start
	# 		print 'Checking candidates with ground truth...'
	# 		# candidate_files = glob.glob(save_data_folder + patient+'/cropped_candidates/' + '*.nii')
	# 		candidate_files = glob.glob(save_data_folder + patient+'/cropped_candidates_similarity_threshold/' + '*.nii')
	# 		candidate_list = []

	# 		for candidate_filename in candidate_files:
	# 			name = os.path.basename(candidate_filename)[0:-4]
	# 			candidate_list.append(name)

	# 		ground_truth_files = glob.glob(save_data_folder + patient+'/ground_truth/' + '*.nii')
	# 		ground_truth_list = []

	# 		for ground_truth_filename in ground_truth_files:
	# 			name = os.path.basename(ground_truth_filename)[0:-4]
	# 			ground_truth_list.append(name)

	# 		label_list = []
	# 		candidate_total = candidate_total + len(candidate_list)
	# 		if len(ground_truth_list) ==0:
	# 			ground_truth_total = ground_truth_total
	# 			true_positive_total = true_positive_total
				
	# 		else:
	# 			# check similarity among ground truth and candidate
	# 			for i in xrange(len(ground_truth_list)):
	# 				ground_truth_total = ground_truth_total + 1
	# 				ground_truth_img = LoadImage(save_data_folder + patient+'/ground_truth/' + ground_truth_list[i] + '.nii')
	# 				for j in xrange(len(candidate_list)):
	# 					# candidate_img = LoadImage(save_data_folder + patient+'/cropped_candidates/' + candidate_list[j] + '.nii')
	# 					candidate_img = LoadImage(save_data_folder + patient+'/cropped_candidates_similarity_threshold/' + candidate_list[j] + '.nii')

	# 					# corrCoef, mutInfo = Similarity(ground_truth_img,candidate_img)
	# 					corrCoef = Similarity(ground_truth_img,candidate_img)
	# 					if corrCoef > 0.7:
	# 						true_positive_total = true_positive_total + 1
	# 						label_list.append([ground_truth_list[i],candidate_list[j]])
	# 						print ground_truth_list[i],candidate_list[j]
	# 						continue

	# 		result_file = open(save_data_folder + patient + '/result.txt','w') 
	# 		result_file.write(str(label_list))
	# 		result_file.close() 				

	# 		

			
	# 		check_point_5_end = True
	# 		# check point 5 end

	false_positive_total = candidate_total - true_positive_total
	print 'Total number of ground truth =',ground_truth_total
	print 'Total number of candidates =',candidate_total
	print 'Total number of true positive =',true_positive_total
	print 'Total number of false positive =',false_positive_total
	print 'Correction rate =',true_positive_total*1./ground_truth_total*100,'%'