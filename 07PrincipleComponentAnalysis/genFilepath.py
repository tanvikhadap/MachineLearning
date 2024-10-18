import os
current_dir = os.getcwd()
dataFolder = 'DataSets_Required'
imageFile = 'IMAGE2'
suffix = 'jepg'
sFilePath =os.path.join(current_dir, dataFolder,imageFile + '.' + suffix)
print(sFilePath)
sFilePath=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'DataSets_Required'))
print(sFilePath)