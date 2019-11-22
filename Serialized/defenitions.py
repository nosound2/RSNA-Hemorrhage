data_dir = '/media/nvme/data/RSNA/'
train_images_dir=data_dir+'stage_1_train_images/'
test_images_dir=data_dir+'stage_1_test_images/'  #data_dir+'stage_2_test_images/'
models_dir = '/media/hd/notebooks/data/RSNA/models/'
outputs_dir = '/media/hd/notebooks/data/RSNA/outputs/'
models_format='model_{}_version_{}_split_{}.pth'
outputs_format='model_{}_version_{}_type_{}_split_{}.pkl'
hemorrhage_types=['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']
bad_images = ['e20bf3f8a','8da38f2e4','6431af929','470e639ae','0e21abf7a','d91d52bdc']
my_gmail='********'
my_pass='*********'
to_email='*********'
parameters={ 'se_resnet101_3': {
                            'model_name':'se_resnet101',
                            'SEED':8153,
                            'n_splits':3,
                            'Pre_version':None,
                            'focal':False,
                            'version':'classifier_splits',
                            'train_prediction':'predictions_train_tta',
                            'train_features':'features_train_tta',
                            'test_prediction':'predictions_test',
                            'test_features':'features_test',
                            'num_epochs' : 5,
                            'num_pool':8
                            },
         'se_resnet101_5': {
                            'model_name':'se_resnet101',
                            'SEED':432,
                            'n_splits':5,
                            'Pre_version':None,
                            'focal':False,
                            'version':'new_splits',
                            'train_prediction':'predictions_train_tta',
                            'train_features':'features_train_tta',
                            'test_prediction':'predictions_test',
                            'test_features':'features_test',
                            'num_epochs' : 5,
                            'num_pool':8
                            },
         'se_resnet101_focal': {
                            'model_name':'se_resnet101',
                            'SEED':432,
                            'n_splits':5,
                            'Pre_version':'new_splits',
                            'focal':True,
                            'version':'new_splits_focal',
                            'train_prediction':'predictions_train_tta',
                            'train_features':'features_train_tta',
                            'test_prediction':'predictions_test',
                            'test_features':'features_test',
                            'num_epochs' : 5,
                            'num_pool':8
                            },
         'se_resnext101_32x4d_3': {
                            'model_name':'se_resnext101_32x4d',
                            'SEED':8153,
                            'n_splits':3,
                            'Pre_version':None,
                            'focal':False,
                            'version':'classifier_splits',
                            'train_prediction':'predictions_train_tta',
                            'train_features':'features_train_tta',
                            'test_prediction':'predictions_test_tta',
                            'test_features':'features_test_tta',
                            'num_epochs' : 5,
                            'num_pool':8
                            },
         'se_resnext101_32x4d_5': {
                            'model_name':'se_resnext101_32x4d',
                            'SEED':432,
                            'n_splits':5,
                            'Pre_version':None,
                            'focal':False,
                            'version':'new_splits',
                            'train_prediction':'predictions_train_tta',
                            'train_features':'features_train_tta',
                            'test_prediction':'predictions_test',
                            'test_features':'features_test',
                            'num_epochs' : 5,
                            'num_pool':8
                            },
         'Densenet161_3': {
                            'model_name':'Densenet161_3',
                            'SEED':8153,
                            'n_splits':3,
                            'Pre_version':None,
                            'focal':False,
                            'version':'classifier_splits',
                            'train_prediction':'predictions_train_tta2',
                            'train_features':'features_train_tta2',
                            'test_prediction':'predictions_test',
                            'test_features':'features_test',
                            'num_epochs' : 5,
                            'num_pool':4
                            },
         'Densenet169_3': {
                            'model_name':'Densenet169_3',
                            'SEED':8153,
                            'n_splits':3,
                            'Pre_version':None,
                            'focal':False,
                            'version':'classifier_splits',
                            'train_prediction':'predictions_train_tta2',
                            'train_features':'features_train_tta2',
                            'test_prediction':'predictions_test',
                            'test_features':'features_test',
                            'num_epochs' : 5,
                            'num_pool':8
                            }
        
       }

