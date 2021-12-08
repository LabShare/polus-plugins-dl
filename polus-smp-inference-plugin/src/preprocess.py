import numpy as np
import typing
import torchvision

class LocalNorm(object):
    def __init__(self,
                 window: int = 129,
                 clip: typing.Union[int,float] = 6):
        assert window % 2 == 1, 'window_size must be an odd integer'
        self.window = window
        self.clip = clip
        self.pad = torchvision.transforms.Pad(window//2+1,
                                              padding_mode='reflect')
        # Mode can be test, train, and eval
        self.mode = 'eval'


    def __call__(self,x):
        image = self.pad(x)
        response = self.local_response(image)
        response[response < -self.clip] = -self.clip
        response[response > self.clip] = self.clip
        return response


    def imboxfilt(self,image):
        """imboxfilt Use a box filter on a stack of images
        This method applies a box filter to an image. The input is assumed to be a
        4D array, and should be pre-padded. The output will be smaller by
        window_size-1 pixels in both width and height since this filter does not pad
        the input to account for filtering.
        """
        # Generate an integral image
        image_ii = image.cumsum(-1).cumsum(-2)
        # Create the output
        output = image_ii[...,0:-self.window-1,0:-self.window-1] + \
                 image_ii[...,self.window:-1,self.window:-1] - \
                 image_ii[...,self.window:-1,0:-self.window-1] - \
                 image_ii[...,0:-self.window-1,self.window:-1]
        return output

        
    def local_response(self,image):
        """local_response Regional normalization
        This method normalizes each pixel using the mean and standard deviation of
        all pixels within the window_size. The window_size parameter should be
        2*radius+1 of the desired region of pixels to normalize by. The image should
        be padded by window_size//2 on each side.
        """
        local_mean = self.imboxfilt(image)/(self.window ** 2)
        local_mean_square = self.imboxfilt(image.pow(2))/(self.window ** 2)
        # Use absolute difference because sometimes error causes negative values
        local_std = (local_mean_square - local_mean.pow(2)).abs().sqrt()
        local_std[local_std<10**-3] = 10**-3
        response = (image[...,
                          self.window//2:-self.window//2-1,
                          self.window//2:-self.window//2-1] - local_mean)/local_std
        return response