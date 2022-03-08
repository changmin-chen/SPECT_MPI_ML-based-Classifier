import SimpleITK as sitk
import preproc.registration_gui as rgui
import numpy as np


def registration_estimate(fixed_nda, moving_nda):
    # Transform image from ndarray to sitk image
    fixed_image = sitk.Cast(sitk.GetImageFromArray(fixed_nda), sitk.sitkFloat32)
    moving_image = sitk.Cast(sitk.GetImageFromArray(moving_nda), sitk.sitkFloat32)
    
    # Setup the initial transform and registration parameters
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                                minStep=1e-5,
                                                                relaxationFactor=0.5,
                                                                gradientMagnitudeTolerance=1e-4,
                                                                numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift() 
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Connect all of the observers so that we can perform plotting during registration
    registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method))
    
    # Execute the registration-estimation
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    # Reample the moving image
    moving_image_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    moving_image_resampled = sitk.GetArrayFromImage(moving_image_resampled).astype(np.uint8)

    return moving_image_resampled, final_transform, registration_method.GetMetricValue()


def imregister(fixed_nda, moving_nda, final_transform):
    fixed_image = sitk.Cast(sitk.GetImageFromArray(fixed_nda), sitk.sitkFloat32)
    moving_image = sitk.Cast(sitk.GetImageFromArray(moving_nda), sitk.sitkFloat32)
    
    # reample the moving image
    moving_image_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    moving_image_resampled = sitk.GetArrayFromImage(moving_image_resampled).astype(np.uint8)

    return moving_image_resampled