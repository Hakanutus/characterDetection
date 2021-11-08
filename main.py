# IMPORTED MODULES
#-----------------------------------------------------------------------------------
from PIL import Image, ImageDraw, ImageOps, ImageTk   # Python Imaging Library (PIL) modules
import numpy as np   # fundamental Python module for scientific computing
import math   # math module that provides mathematical functions
import os   # os module can be used for file and directory operations
import cv2
import os
import array
import PySimpleGUI as sg
import io
file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]



# MAIN FUNCTION OF THE PROGRAM
#-----------------------------------------------------------------------------------
# Main function where this python script starts execution
def main():
   layout = [
      [sg.Image(key="-IMAGE-")],
      [
         sg.Text("Image File"),
         sg.Input(size=(25, 1), key="-FILE-"),
         sg.FileBrowse(file_types=file_types),
         sg.Button("Load Image"),
      ],
   ]
   window = sg.Window("Image Viewer", layout)
   while True:
      event, values = window.read()
      if event == "Exit" or event == sg.WIN_CLOSED:
         break
      if event == "Load Image":
         filename = values["-FILE-"]
         if os.path.exists(filename):
            image = Image.open(values["-FILE-"])
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-IMAGE-"].update(data=bio.getvalue())
   window.close()

   # read color image from file
   #--------------------------------------------------------------------------------
   # path of the current directory where this program file is placed
   curr_dir = os.path.dirname(os.path.realpath(__file__))
   #img_file = curr_dir +  '/thinABC123.jpg'
   img_file = filename
   img_color = Image.open(img_file)
   img_color.show() # display the color image
   txtProject = open("C:/Users/Hakan/Desktop/output.txt", "w+")
   txtProject.write("Full path and the name of the input image is " + filename + "\n For level 1's extraction first of all, the 8-blob algorithm provided by you used.\n Simply we threshold the image and label each pixel value.\n To classificate we cropped the detected characters and count after we label them again, we counted the holes for classification.")
   txtProject.write("\nFor level 3 we need to distinguish c's from 1's because they both have no holes. In order to do that we locate the center pixel\nwithin the binary crropped images. After that we check whether the center pixel is 1 or zero.\nIf zero than the character is 1 because the center pixel is filled in 1. Else character is C")
   # convert the color image to a grayscale image
   #--------------------------------------------------------------------------------
   img_gray = img_color.convert('L')
   img_gray.show() # display the grayscale image
   # create a binary image by thresholding the grayscale image
   #--------------------------------------------------------------------------------
   # convert the grayscale PIL Image to a numpy array
   arr_gray = np.asarray(img_gray)
   # values below the threshold are considered as ONE and values above the threshold
   # are considered as ZERO (assuming that the image has dark objects (e.g., letters
   # ABC or digits 123) on a light background)
   THRESH, ZERO, ONE = 155, 0, 255
   # the threshold function defined below returns the binary image as a numpy array
   arr_bin = threshold(arr_gray, THRESH, ONE, ZERO)
   # you can uncomment the line below to work on a 100x100 artificial binary image
   # that contains 3 lines, a square and a circle instead of an input image file
   # arr_bin = artificial_binary_image(ONE)
   # convert the numpy array of the binary image to a PIL Image
   img_bin = Image.fromarray(arr_bin)
   # display the binary image
   # component (object) labeling based on 4-connected components
   #--------------------------------------------------------------------------------
   # blob_coloring_4_connected function returns a numpy array that contains labels
   # for the pixels in the input image, the number of different labels and the numpy
   # array of the image with colored blobs
   arr_labeled_img, num_labels, arr_blobs = blob_coloring_8_connected(arr_bin, ONE)
   # print the number of objects as the number of different labels
   print("There are " + str(num_labels) + " objects in the input image.")
   # write the values in the labeled image to a file
   labeled_img_file = curr_dir + '/labeled_img.txt'
   np.savetxt(labeled_img_file, arr_labeled_img, fmt='%d', delimiter=',')
   # convert the numpy array of the colored components (blobs) to a PIL Image
   img_blobs = Image.fromarray(arr_blobs)
   # display the colored components (blobs)
   min_x = np.full(num_labels, np.inf)
   max_x = np.zeros(num_labels)
   min_y = np.full(num_labels, np.inf)
   max_y = np.zeros(num_labels)
   height, width = arr_labeled_img.shape

   for y in range(height):
      for x in range(width):
         label_value = arr_labeled_img[y, x]
         if label_value != 0:
            if y < min_y[label_value - 1]:
               min_y[label_value - 1] = y
            if y > max_y[label_value - 1]:
               max_y[label_value - 1] = y
            if x < min_x[label_value - 1]:
               min_x[label_value - 1] = x
            if x > max_x[label_value - 1]:
               max_x[label_value - 1] = x
   img_bounded = ImageDraw.Draw(img_color)
   for i in range(num_labels):
      coordinates = [(min_x[i], min_y[i]), (max_x[i], max_y[i])]
      img_bounded.rectangle(coordinates, outline ="red")
   img_final = ImageDraw.Draw(img_color)
   a_count = 0
   b_count = 0
   c_count = 0
   for i in range(num_labels):
      cropCoordinates = [(min_x[i], min_y[i]), (max_x[i], max_y[i])]
      cropped = img_bin.crop((min_x[i] - 3, min_y[i] - 3, max_x[i] + 3, max_y[i] + 3))
      cropped_invert = ImageOps.invert(cropped.convert('RGB'))
      cropped_invert = cropped_invert.convert('L')
      cropped_arr_invert = np.asarray(cropped_invert)
      labeled_crop, detectedCrop, arr_crops = blob_coloring_8_connected(cropped_arr_invert, ONE)
      if detectedCrop == 1:
         img_final.text((min_x[i], min_y[i] - 10), "C", fill=(0, 0, 0))
         txtProject.write("\n Coordinates of C ")
         str2 = ' '.join(map(str, cropCoordinates))
         txtProject.write(str2)
         c_count+=1
      if detectedCrop == 2:
         img_final.text((min_x[i], min_y[i]-10), "A", fill=(0, 0, 0))
         txtProject.write("\n Coordinates of A ")
         str3 = ' '.join(map(str, cropCoordinates))
         txtProject.write(str3)
         a_count+=1
      if detectedCrop == 3:
         txtProject.write("\n Coordinates of B ")
         str4 = ' '.join(map(str, cropCoordinates))
         txtProject.write(str4)
         img_final.text((min_x[i], min_y[i]-10), "B", fill=(0, 0, 0))
         b_count+=1
   txtProject.write("\nA count is " + str(a_count) + "\nB count is " + str(b_count) + "\nC count is " +str(c_count))
   txtProject.write("\n Total number of detected characters is " + str(num_labels))
   img_color.show()
# GENERATING AN ARTIFICIAL BINARY IMAGE
#-----------------------------------------------------------------------------------
# Function that creates and returns a 100x100 artificial binary image with 3 lines,
# a square and a circle (background pixels = 0 and shape pixels = HIGH)
# The returned image can be used for comparing 4-connected and 8-connected labeling
def artificial_binary_image(HIGH):
   # the generated image has the size 100 x 100
   n_rows = n_cols = 100
   # y and x are 2D arrays that store row and column indices for each pixel of the
   # artificial binary image
   y, x = np.indices((n_rows, n_cols))
   # code part that is used to generate the 3 lines on the artificial binary image
   #--------------------------------------------------------------------------------
   mask_lines = np.zeros(shape = (n_rows, n_cols))
   for i in range (50, 70):
      # code part that generates the mask for the thick \ shaped line on the right
      mask_lines[i][i] = 1
      mask_lines[i][i + 1] = 1
      mask_lines[i][i + 2] = 1
      mask_lines[i][i + 3] = 1
      # code part that generates the mask for the thin \ shaped line on the right
      # (this line can not be labeled correctly by using 4-connected labeling thus
      # it requires using 8-connected labeling)
      mask_lines[i][i + 6] = 1
      # code part that generates the mask for the thick / shaped line on the left
      mask_lines[i - 20][90 - i + 1] = 1
      mask_lines[i - 20][90 - i + 2] = 1
      mask_lines[i - 20][90 - i + 3] = 1
   # code part that is used to generate the masks for creating a square and a circle
   # on the artificial binary image
   #--------------------------------------------------------------------------------
   x0, y0, r0 = 30, 30, 5
   x1, y1, r1 = 70, 30, 5
   mask_square = np.fmax(np.absolute(x - x1), np.absolute(y - y1)) <= r1
   # the created circle can not be labeled correctly by using 4-connected labeling
   # thus it requires using 8-connected labeling
   mask_circle = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
   # an artificial binary image is created by applying the masks
   #--------------------------------------------------------------------------------
   mask_square_and_circle = np.logical_or(mask_square, mask_circle)
   a_bin_image = np.logical_or(mask_lines, mask_square_and_circle) * HIGH
   # the resulting artificial binary image is returned
   return a_bin_image

# BINARIZATION
#-----------------------------------------------------------------------------------
# Function for creating and returning a binary image as a numpy array by thresholding
# the given array of the grayscale image
def threshold(arr_gray_in, T, LOW, HIGH):
   # get the numbers of rows and columns in the array of the grayscale image
   n_rows, n_cols = arr_gray_in.shape
   # initialize the output (binary) array by using the same size as the input array
   # and filling with zeros
   arr_bin_out = np.zeros(shape = arr_gray_in.shape)
   # for each value in the given array of the grayscale image
   for i in range(n_rows):
      for j in range(n_cols):
         # if the value is smaller than the given threshold T
         if abs(arr_gray_in[i][j]) < T:
            # the corresponding value in the output (binary) array becomes LOW
            arr_bin_out[i][j] = LOW
         # if the value is greter than or equal to the given threshold T
         else:
            # the corresponding value in the output (binary) array becomes HIGH
            arr_bin_out[i][j] = HIGH
   # return the resulting output (binary) array
   return arr_bin_out

# CONNECTED COMPONENT LABELING AND BLOB COLORING
#-----------------------------------------------------------------------------------
# Function for labeling objects as 4-connected components in a binary image whose
# numpy array is given as an input argument and creating an image with randomly
# colored components (blobs)
def blob_coloring_8_connected(arr_bin, ONE):
   # get the numbers of rows and columns in the array of the binary image
   n_rows, n_cols = arr_bin.shape
   # max possible label value is set as 10000
   max_label = 10000
   # initially all the pixels in the image are labeled as max_label
   arr_labeled_img = np.zeros(shape = (n_rows, n_cols), dtype = int)
   for i in range(n_rows):
      for j in range(n_cols):
         arr_labeled_img[i][j] = max_label
   # keep track of equivalent labels in an array
   # initially this array contains values from 0 to max_label - 1
   equivalent_labels = np.arange(max_label, dtype = int)
   # labeling starts with k = 1
   k = 1
   # first pass to assign initial labels and update equivalent labels from conflicts
   # for each pixel in the binary image
   #--------------------------------------------------------------------------------
   for i in range(1, n_rows - 1):
      for j in range(1, n_cols - 1):
         c = arr_bin[i][j] # value of the current (center) pixel
         l = arr_bin[i][j - 1] # value of the left pixel
         label_l = arr_labeled_img[i][j - 1] # label of the left pixel
         u = arr_bin[i - 1][j] # value of the upper pixel
         label_u = arr_labeled_img[i - 1][j] # label of the upper pixel
         d = arr_bin[i - 1][j - 1]
         label_d = arr_labeled_img[i - 1][j - 1]
         r = arr_bin[i - 1][j + 1]
         label_r = arr_labeled_img[i - 1][j + 1]
         # only the non-background pixels are labeled
         if c == ONE:
            # get the minimum of the labels of the upper and left pixels
            min_label = min(label_u, label_l, label_d, label_r)
            # if both upper and left pixels are background pixels
            if min_label == max_label:
               # label the current (center) pixel with k and increase k by 1
               arr_labeled_img[i][j] = k
               k += 1
            # if at least one of upper and left pixels is not a background pixel
            else:
               # label the current (center) pixel with min_label
               arr_labeled_img[i][j] = min_label
               # if upper pixel has a bigger label and it is not a background pixel
               if min_label != label_u and label_u != max_label:
                  # update the array of equivalent labels for label_u
                  update_array(equivalent_labels, min_label, label_u)
               # if left pixel has a bigger label and it is not a background pixel
               if min_label != label_l and label_l != max_label:
                  # update the array of equivalent labels for label_l
                  update_array(equivalent_labels, min_label, label_l)
               if min_label != label_r and label_r != max_label:
                  update_array(equivalent_labels, min_label, label_r)
               if min_label != label_d and label_d != max_label:
                  update_array(equivalent_labels, min_label, label_d)

   # final reduction in the array of equivalent labels to obtain the min. equivalent
   # label for each used label (values from 1 to k - 1) in the first pass of labeling
   #--------------------------------------------------------------------------------
   for i in range(1, k):
      index = i
      while equivalent_labels[index] != index:
         index = equivalent_labels[index]
      equivalent_labels[i] = equivalent_labels[index]
   # rearrange equivalent labels so they all have consecutive values starting from 1
   # using the rearrange_array function which also returns the number of different
   # values of the labels used to label the image
   num_different_labels = rearrange_array(equivalent_labels, k)
   # create a color map for randomly coloring connected components (blobs)
   #--------------------------------------------------------------------------------
   color_map = np.zeros(shape = (k, 3), dtype = np.uint8)
   np.random.seed(0)
   for i in range(k):
      color_map[i][0] = np.random.randint(0, 255, 1, dtype = np.uint8)
      color_map[i][1] = np.random.randint(0, 255, 1, dtype = np.uint8)
      color_map[i][2] = np.random.randint(0, 255, 1, dtype = np.uint8)
   # create an array for the image to store randomly colored blobs
   arr_color_img = np.zeros(shape = (n_rows, n_cols, 3), dtype = np.uint8)
   # second pass to resolve labels by assigning the minimum equivalent label for each
   # label in arr_labeled_img and color connected components (blobs) randomly
   #--------------------------------------------------------------------------------
   for i in range(n_rows):
      for j in range(n_cols):
         # only the non-background pixels are taken into account and the pixels
         # on the boundaries of the image are always labeled as 0
         if arr_bin[i][j] == ONE and arr_labeled_img[i][j] != max_label:
            arr_labeled_img[i][j] = equivalent_labels[arr_labeled_img[i][j]]
            arr_color_img[i][j][0] = color_map[arr_labeled_img[i][j], 0]
            arr_color_img[i][j][1] = color_map[arr_labeled_img[i][j], 1]
            arr_color_img[i][j][2] = color_map[arr_labeled_img[i][j], 2]
         # change the label values of background pixels from max_label to 0
         else:
            arr_labeled_img[i][j] = 0
   # return the labeled image as a numpy array, number of different labels and the
   # image with colored blobs (components) as a numpy array
   return arr_labeled_img, num_different_labels, arr_color_img

# Function for updating the equivalent labels array by merging label1 and label2
# that are determined to be equivalent
def update_array(equ_labels, label1, label2) :
   # determine the small and large labels between label1 and label2
   if label1 < label2:
      lab_small = label1
      lab_large = label2
   else:
      lab_small = label2
      lab_large = label1
   # starting index is the large label
   index = lab_large
   # using an infinite while loop
   while True:
      # update the label of the currently indexed array element with lab_small when
      # it is bigger than lab_small
      if equ_labels[index] > lab_small:
         lab_large = equ_labels[index]
         equ_labels[index] = lab_small
         # continue the update operation from the newly encountered lab_large
         index = lab_large
      # update lab_small when a smaller label value is encountered
      elif equ_labels[index] < lab_small:
         lab_large = lab_small # lab_small becomes the new lab_large
         lab_small = equ_labels[index] # smaller value becomes the new lab_small
         # continue the update operation from the new value of lab_large
         index = lab_large
      # end the loop when the currently indexed array element is equal to lab_small
      else: # equ_labels[index] == lab_small
         break

# Function for rearranging min equivalent labels so they all have consecutive values
# starting from 1
def rearrange_array(equ_labels, final_k_value):
   # find different values of equivalent labels and sort them in increasing order
   different_labels = set(equ_labels[1:final_k_value])
   different_labels_sorted = sorted(different_labels)
   # compute the number of different values of the labels used to label the image
   num_different_labels = len(different_labels)
   # create an array for storing new (consecutive) values for equivalent labels
   new_labels = np.zeros(final_k_value, dtype = int)
   count = 1 # first label value to assign
   # for each different label value (sorted in increasing order)
   for l in different_labels_sorted:
      # determine the new label
      new_labels[l] = count
      count += 1 # increase count by 1 so that new label values are consecutive
   # assign new values of each equivalent label
   for ind in range(1, final_k_value):
      old_label = equ_labels[ind]
      new_label = new_labels[old_label]
      equ_labels[ind] = new_label
   # return the number of different values of the labels used to label the image
   return num_different_labels


# main() function is specified as the entry point where the program starts running
if __name__=='__main__':
   main()