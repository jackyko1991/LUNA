import SimpleITK as sitk

reader = sitk.ImageFileReader()
reader.SetFileName('../data/candidates/lung_2016_subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260/resampled.nii')
img = reader.Execute()

# denoising
print 'Denoising...'
blurFilter = sitk.CurvatureFlowImageFilter()
blurFilter.SetNumberOfIterations(5)
blurFilter.SetTimeStep(0.125)
img = blurFilter.Execute(img)

# apply thresholding to get body mask
thresholdFilter = sitk.BinaryThresholdImageFilter()
thresholdFilter.SetLowerThreshold(0)
thresholdFilter.SetInsideValue(1)
thresholdFilter.SetOutsideValue(0)
thresholdFilter.SetUpperThreshold(300) # threshold value by experiment
mask = thresholdFilter.Execute(img)

# addFilter = sitk.AddImageFilter()
# img = addFilter.Execute(img,1000)

# # Fill in the small holes inside the binary mask of lungs.
# print 'Filling holes'
# fillHoleFilter = sitk.BinaryFillholeImageFilter()
# mask = fillHoleFilter.Execute(mask)

# # extract body
# maskFilter = sitk.MaskImageFilter()
# img = maskFilter.Execute(img,mask)

# # apply Otsu on the body
# thresholdFilter = sitk.OtsuMultipleThresholdsImageFilter()
# thresholdFilter.SetNumberOfThresholds(2)
# labelSitk = thresholdFilter.Execute(img)


writer = sitk.ImageFileWriter()
writer.SetFileName('../data/candidates/lung_2016_subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260/resampled_label_body.nii')
writer.Execute(mask)

# invertFilter = sitk.InvertIntensityImageFilter()
# img = invertFilter.Execute(img)



# # apply Otsu on the lung
# thresholdFilter = sitk.OtsuMultipleThresholdsImageFilter()
# thresholdFilter.SetNumberOfThresholds(2)
# labelSitk = thresholdFilter.Execute(img)

# writer = sitk.ImageFileWriter()
# writer.SetFileName('../data/candidates/lung_2016_subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.129055977637338639741695800950/resampled_label.nii')
# writer.Execute(labelSitk)

# writer.SetFileName('../data/candidates/lung_2016_subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.129055977637338639741695800950/invert.nii')
# writer.Execute(img)