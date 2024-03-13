import imutils
import pandas
import imageio
import glob

final_dataset = pandas.DataFrame()

for i in range(10):
  print(i)
  result = []
  result_extended = []
  for file_path in glob.glob("dataset_raw/" + str(i) + "/*.jpg"):
    imimage = imageio.v2.imread(file_path)
    imimage_rotate_positive_5 = imutils.rotate(imimage, angle=5)
    imimage_rotate_negative_5 = imutils.rotate(imimage, angle=-5)
    imimage_enlarge = imutils.resize(imimage, width=30)
    imimage_enlarge = imimage_enlarge[1:29, 1:29]
    
    imimage = imimage.flatten()
    imimage_rotate_positive_5 = imimage_rotate_positive_5.flatten()
    imimage_rotate_negative_5 = imimage_rotate_negative_5.flatten()
    imimage_enlarge = imimage_enlarge.flatten()
    
    result.append(imimage)
    result_extended.append(imimage_rotate_positive_5)
    result_extended.append(imimage_rotate_negative_5)
    result_extended.append(imimage_enlarge)
    
  result_dataframe = pandas.DataFrame(result)
  result_dataframe = result_dataframe.add_prefix('pixel_')
  result_dataframe['digit'] = i
  result_dataframe['extended'] = 0
  result_extended_dataframe = pandas.DataFrame(result_extended)
  result_extended_dataframe = result_extended_dataframe.add_prefix('pixel_')
  result_extended_dataframe['digit'] = i
  result_extended_dataframe['extended'] = 1
  
  
  final_dataset = pandas.concat([result_dataframe, final_dataset], axis=0)
  
final_dataset.to_csv("dataset.csv", index=False)

    
    
    
    