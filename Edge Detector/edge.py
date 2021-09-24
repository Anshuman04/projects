# AUTHOR: Anshuman Gaharwar (UCF ID: 5321571)
# Usage:
#     python .\edge.py --input img5.jpg --output img5_result.jpg --lowThresh 25
#                      --highThresh 45 --sigma 10


import math
import numpy as np
import cv2
import os
import sys
import logging
import argparse
import time

class ImageUtils(object):
    """
    Class comprising of logical funcationalities which are not part of edge detector
    """

    def gaussianKernel1D(self, sigma, n=3):
        """
        This function uses 1D gaussian formula to generate gaussian kernel for length 'n' and with sigma 'sigma'
        """

        logging.info("Creating gaussian kernel with length {} and sigma {}".format(n, sigma))
        kernel = []
        # As gaussian distribution is symmetric around 0, range goes from -n/2 to n/2 +1
        for nVal in range(-1 * int(n/2), int(n/2) + 1):
            # Formula fo 1D Gaussian
            val = (math.exp(float(-1 * (nVal) ** 2) / float(2 * sigma ** 2))) / math.sqrt(2 * 22 * sigma * sigma / 7)
            kernel.append(val)
        # Converting to np.ndarray as this will be treated as kernel and normalizing
        kernel = np.asarray(kernel) / sum(kernel)
        return kernel
    
    def loadImage(self, imgPath):
        """
        This function loads the input image provided through argument.
        As, the expected input image is grayscale, this function will convert any color image passed into grayscale

        Raise:
            Exception: If file not present or file corrupted.
        """

        assert os.path.exists(imgPath), "Inavlid image path passed: [{}]".format(imgPath)
        logging.info("Reading image: {}".format(imgPath))
        img = cv2.imread(imgPath)
        if img is None:
            raise Exception("Error while reading the image")
        if len(img.shape) >= 3:         # Length of shape is 3 for color images and images with alpha channel
            logging.warning("Colored image passed for edge detection")
            logging.warning("Converting the input image into grayscale")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def getSquareKernel(self, kernel):
        """
        This converts 1D kernel into 2D kernel by padding zeros. This will detect and pad row or col based on
        info that 1D kernel passed was row matrix or col matrix.
        """

        dim = max(kernel.shape)
        newKernel = np.zeros((dim,dim))
        if kernel.shape[0] == dim:
            newKernel[:,int(dim/2)] = kernel
        else:
            newKernel[int(dim/2),:] = kernel
        return newKernel

    def getDerivativeKernelY(self, kernel):
        """
        This function takes in smoothing kernel and generates a derivative kernel
        This function works on forward derivative #TODO
        """
        negKernel = -1 * kernel
        derivativeKernel = np.zeros((max(kernel.shape), 3))
        derivativeKernel[0, :] = kernel
        derivativeKernel[2, :] = negKernel
        return derivativeKernel

    def convolute(self, image, kernel):
        """
        This method canvolves kernel over the image and returns the output.
        This method also includes zero padding in order to ensure there is no reduction in resolution
        """
        rows, cols = image.shape
        # ZERO PADDING STARTS
        paddingReq = int(max(kernel.shape) / 2)
        temp = np.zeros((rows + 2*paddingReq, cols + 2*paddingReq))
        output = np.zeros((rows, cols))
        temp[paddingReq:-paddingReq, paddingReq:-paddingReq] = image
        # ZERO PADDING ENDS
        logging.info("Convolution starts: {}".format(time.time()))
        for i in range(paddingReq, rows+paddingReq):
            for j in range(paddingReq, cols+paddingReq):
                val = 0
                for k in range(kernel.shape[0]):
                    for l in range(kernel.shape[1]):
                        # logging.info("Image: [{},{}] => Kernel: [{}, {}]".format(k,l,i-paddingReq+k,j-paddingReq+l))
                        val +=  kernel[k, l] * temp[i-paddingReq+k, j-paddingReq+l]
                output[i-paddingReq, j-paddingReq] = int(val)
        logging.info("Convolution ends: {}".format(time.time()))

        return output

class EdgeDetector(object):
    """
        Class having methods related to edge detector
    """
    def __init__(self, imgPath, lowThresh, highThresh, gaussianSigma=3):
        self.utilsObj = ImageUtils()
        self.inputImage = imgPath
        self.lowThresh = lowThresh
        self.highThresh = highThresh
        self.sigma = gaussianSigma
        

    def combineEdges(self, edgeX, edgeY):
        """
            Parameters:
                edgeX: Vertical edges found using smooth+derivative kernel
                edgeY: Horizontal edges found using smooth+derivative kernel
            This method performs following functions:
                1. Combines horizontal and vertical edges by calculating magnitude
                2. Calculates orientation of the gradients using arctan formula
        """
        output = np.zeros(edgeX.shape)
        thetaMat = np.zeros(edgeX.shape)
        for i in range(edgeX.shape[0]):
            for j in range(edgeX.shape[1]):
                # Magnitude formula to combine X and Y parts
                mag = math.sqrt(edgeX[i,j] ** 2 + edgeY[i,j] ** 2)

                # Theta formula to calculate orientation of gradients
                theta = math.degrees(math.atan(edgeY[i,j] / edgeX[i,j]))
                output[i, j] = mag
                thetaMat[i, j] = theta
        return output, thetaMat

    def edgeThinner(self, edges, thetaMat):
        """
            This method implements non maximum supression algorithm
            Parameters:
                edges: All edges, having both X and Y counterparts
                thetaMat: Matrix with size of image(edges) where each pixel has theta value
                          i.e Orientation of gradient at each pixel location

            Theory:
                - Suppose there was a vertical edge with 1 pixel width: [0, 0, 200, 0, 0]
                - When we run smoothing filter (to remove noise), the edge gets blurred too like [0, 100, 160, 100, 0]
                - Above values is just to give the idea of effect not the actual values.
                - Hence our sharp edge is now 3 differnt edges
                - We use this edgeThinner to convert back edges to width of 1 pixel

            Implementation Details:
                1. Bucket dictionary initialized to club the gradients together
                    - The grouping is done to get the formula for neighbouring elements
                    - Ex: if gradient has theta = 0, This means, we have vertical edge
                    - Assuming this edge is blurred, alternate edges will also lie vertically on both sides of this edge
                    - Hence we will check left and right neighbour magnitude to decide whether we should keep present pixel or not
                    - Bucket dictionary holds all the relative positions of neighbours based on the degrees of orientation
                2. Comparing present pixels with its respective neighbours (coming from bucketDict)
                3. Make present pixel as 0, if its magnitude is not maximum in its vicinity
        """
        bucket = {
            "0<={deg}<22.5 or 157.5<={deg}<202.5 or 337.5<={deg}<360": [0, [[1,0], [-1, 0]]],
            "22.5<={deg}<67.5 or 202.5<={deg}<247.5": [45, [[-1, 1], [1, -1]]],
            "67.5<={deg}<112.5 or 247.5<={deg}<292.5": [90, [[0, -1], [0, 1]]],
            "112.5<={deg}<157.5 or 292.5<={deg}<337.5": [135, [[-1,-1], [1, 1]]], 
        }
        # Zero Padding starts
        canvas = np.zeros((thetaMat.shape[0]+2, thetaMat.shape[1]+2))
        canvas[1:-1, 1:-1] = edges
        # Zero Padding ends

        for i in range(thetaMat.shape[0]):
            for j in range(thetaMat.shape[1]):
                deg = thetaMat[1,j]
        
                # Convert negative degrees into postive by adding 360
                if deg < 0:
                    deg = 360 + deg

                for key in bucket.keys():
                    if np.isnan(deg):
                        """
                            math.atan function returns NaN (Not a Number) when X component is 0 (resulting in divide by 0)
                            Thus for exact horizontal edges, there is no counterpart in edgeY. Hence specifically assigning
                            its neighbours which are just above and below elements of center pixel
                        """
                        [a, b] = [0, -1]    # Relative position of neighbour above center pixel
                        [c, d] =  [0, 1]    # Relative position of neighbour below center pixel
                        maxCandidates = [canvas[i+1+a, j+1+b], canvas[i+1, j+1], canvas[i+1+c, j+1+d]]
                        if max(maxCandidates) != canvas[i+1, j+1]:
                            canvas[i+1, j+1] = 0    # Making pixel 0, when pixel was not max gradient in vicinity
                        break
                    elif eval(key.format(deg=deg)):
                        [a, b], [c, d] = bucket[key][1]     # Neighbours coming from bucket for valid degrees
                        maxCandidates = [canvas[i+1+a, j+1+b], canvas[i+1, j+1], canvas[i+1+c, j+1+d]]
                        if max(maxCandidates) != canvas[i+1, j+1]:
                            canvas[i+1, j+1] = 0    # Making pixel 0, when pixel was not max gradient in vicinity
                        break
        return canvas[1:-1,1:-1]    # Remove padding and return the results
    
    def hysteresisThresholder(self, img):
        """
        This method implements hysteresis thresholding
        Theory:
            1. Given 2 thresholds: low and high, thresholding works like below
            2. Everything below low is not an edge
            3. Everything above high is an edge
            4. Everything between high and low is an edge only if it is CONNECTED to some high/strong edge
            5. Connectedness is implemented using DFS algorithm like graphs in method "_explore"

        Implementation Details:
            1. Visited matrix to keep track of all the pixel location already visited.
            2. Whenever non visited edge point is found, it is passed to explore method to get all connected pixels
            3. explore method also returns the maximum value of all the connected components
            4. if maximum value of the connected pixels is above high threshold, all pixels are made strong edge(255)
        """
        visited = np.zeros(img.shape)       # Visit tracking matrix

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] > 0 and visited[i,j] == 0:      # Non visited and pixel has some edge
                    img, visited, retPixels, maxVal = self._explore(img, visited, i, j, img.shape[0], img.shape[1])
                    if maxVal > self.highThresh:
                        pigment = 255
                    else:
                        pigment = 0

                    # Convert all connected pixels into an edge(255) or black(0)
                    for x, y in retPixels:
                        img[x, y] = pigment
        return img
    
    def _explore(self, img, visited, i, j, rowLimit, colLimit):
        """
            This method returns all location of connected edges, given any point on edge.
            Implemented using graphs' DFS approach
        """
        stack = [[i,j]]
        retPixels = []
        maxVal = img[i,j]
        while stack:
            nodeI, nodeJ = stack.pop()
            visited[nodeI, nodeJ] = 1       # Mark pixel visited

            # Keep track of maximum value across all connected pixels
            if img[nodeI, nodeJ] > maxVal:
                maxVal = img[nodeI, nodeJ]

            # Remove the edge if less than lowThresh
            if img[nodeI, nodeJ] < self.lowThresh:
                img[nodeI, nodeJ] = 0
                continue

            retPixels.append([nodeI, nodeJ]) # Add present pixel in list of connected edges

            neighBours = []
            for x, y in [[nodeI+1, nodeJ], [nodeI-1, nodeJ], [nodeI+1, nodeJ+1], [nodeI-1, nodeJ-1],
                         [nodeI, nodeJ+1], [nodeI, nodeJ-1], [nodeI-1,nodeJ+1], [nodeI+1, nodeJ-1]]:
                if (0 <= x < rowLimit) and (0 <= y < colLimit) and visited[x, y] == 0:
                    stack.append([x, y])
            
        return img, visited, retPixels, maxVal

    def detectEdges(self):
        """
            This method is runner for edge detector
        """
        logging.info("EDGE DETECTION STARTS !!!")
        logging.info("Loading image")
        image = self.utilsObj.loadImage(self.inputImage)
        logging.info("Image loaded")

        logging.info("Create 1D gaussian kernel for smoothning")
        gauss_1D = self.utilsObj.gaussianKernel1D(self.sigma)
        logging.info("1D gaussian kernel created")

        logging.info("Creating derivative kernel for Y")
        derivativeY = self.utilsObj.getDerivativeKernelY(gauss_1D)
        logging.info("Derivative kernel for Y created")

        logging.info("Creating derivative kernel for X by transposing Y")
        derivativeX = np.transpose(derivativeY)
        logging.info("Derivative kernel for X created")

        logging.info("Finding edges using Y derivative kernel. It may take some time")
        edgeY = self.utilsObj.convolute(image, derivativeY)
        logging.info("Y edges calculated")

        logging.info("Finding edges using X derivative kernel. It may take some time")
        edgeX = self.utilsObj.convolute(image, derivativeX)
        logging.info("X edges calculated")

        logging.info("Calculating final gradients and their orientation")
        edges, thetaMat = self.combineEdges(edgeX, edgeY)
        logging.info("Gradients and their orientation calculated")

        logging.info("Performing edge thinning using non-maximum supression algorithm. It may take some time")
        thinEdge = self.edgeThinner(edges, thetaMat)
        logging.info("Edge thinning completed")

        logging.info("Performing hystersis thresholding with thresholds: L={}; R={}".format(self.lowThresh, self.highThresh))
        output = self.hysteresisThresholder(thinEdge)
        logging.info("Hysteresis thresholding completed")
        logging.info("EDGE DETECTION COMPLETE !!!")
        return output


def getArguments():
    """
        Argument parser. Read help for details
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", dest="sigma", default=3, type=int, help="Sigma to be used for gaussian kernel")
    parser.add_argument("--lowThresh", dest="lowThresh", required=True, type=int, help="Low threshold for hysteresis")
    parser.add_argument("--highThresh", dest="highThresh", required=True, type=int, help="High threshold for hysteresis")
    parser.add_argument("--input", dest="inpImg", required=True, help="Input image for edge detector")
    parser.add_argument("--output", dest="opImg", default="result.jpg", help="Filename to dump the results")
    allArgs = parser.parse_args()
    return allArgs

def setupLogger():
    """
        Logger setup
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", '%m-%d-%Y %H:%M:%S')
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.DEBUG)
    stdout.setFormatter(formatter)
    logger.addHandler(stdout)
    logging.debug("Setting up logger completed")

if __name__ == "__main__":
    """
        Usage: python .\edge.py --input img5.jpg --output img5_result.jpg --lowThresh 25 --highThresh 45 --sigma 10
    """
    print ("MAIN INITIATED")
    allArgs = getArguments()
    setupLogger()
    detectorObj = EdgeDetector(allArgs.inpImg, allArgs.lowThresh, allArgs.highThresh, gaussianSigma=allArgs.sigma)
    output = detectorObj.detectEdges()
    logging.info("Dumping results of edge detection in file: {}".format(allArgs.opImg))
    cv2.imwrite(allArgs.opImg, output)
    print ("MAIN TERMINATED")

