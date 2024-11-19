import SimpleITK as sitk

# A file name that belongs to the series we want to read
file_name = 'testing/input/combatvt/Echo_AP4CH.dcm'
data_directory = '.'

# Read the file's meta-information without reading bulk pixel data
file_reader = sitk.ImageFileReader()
file_reader.SetFileName(file_name)
file_reader.ReadImageInformation()

# Get the sorted file names, opens all files in the directory and reads the meta-information
# without reading the bulk pixel data
series_ID = file_reader.GetMetaData('0020|000e')
sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_ID)

# Read the bulk pixel data
img = sitk.ReadImage(sorted_file_names)