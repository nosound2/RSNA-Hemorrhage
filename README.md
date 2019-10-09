RSNA Intracranial Hemorrhage Detection

metadata location
```
gs://rsna-hemorrhage
```

metadata columns to use:
```
cols_cat = ['BitsStored','PixelRepresentation','RescaleIntercept','WindowCenter_1_NAN']
cols_float = ['ImageOrientationPatient_0', 'ImageOrientationPatient_1',
       'ImageOrientationPatient_2', 'ImageOrientationPatient_3',
       'ImageOrientationPatient_4', 'ImageOrientationPatient_5',
       'ImagePositionPatient_0', 'ImagePositionPatient_1','ImagePositionPatient_2',
        'PixelSpacing_0', 'PixelSpacing_1', 'WindowCenter_0', 'WindowCenter_1']
```

and subset of the above columns which are significant:
```
significant_cols = ['PixelRepresentation', 'ImageOrientationPatient_4',
       'ImagePositionPatient_0', 'ImagePositionPatient_1',
       'ImagePositionPatient_2', 'PixelSpacing_0', 'PixelSpacing_1',
       'WindowCenter_0']
```