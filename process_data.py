import pandas as pd
import glob

path_train = 'train'
path_test = 'test'

def read_data(data_dir):
    filenames_train = glob.glob(data_dir + "/*.csv")
    data_files_train = []
    df = pd.DataFrame()
    for filename in filenames_train:
        df = pd.concat([df, pd.read_csv(filename)])
        print(f'{filename}: was concated')
    df = df.drop(['feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_7', 'feature_10', 'feature_13', 'feature_15', 'feature_23', 'feature_24', 'feature_28', 'feature_30', 'feature_32', 'feature_34', 'feature_37', 'feature_39', 'feature_40', 'feature_41', 'feature_42', 'feature_45', 'feature_49', 'feature_52', 'feature_53', 'feature_54', 'feature_56', 'feature_57', 'feature_58', 'feature_63', 'feature_64', 'feature_65', 'feature_67', 'feature_68', 'feature_69', 'feature_70', 'feature_73', 'feature_74', 'feature_80', 'feature_81', 'feature_82', 'feature_83', 'feature_85', 'feature_86', 'feature_88', 'feature_89', 'feature_92', 'feature_96', 'feature_97', 'feature_99', 'feature_101', 'feature_102', 'feature_104', 'feature_106', 'feature_107', 'feature_108', 'feature_109', 'feature_110', 'feature_113', 'feature_114', 'feature_115', 'feature_116', 'feature_117', 'feature_119', 'feature_120', 'feature_121', 'feature_122', 'feature_123', 'feature_125', 'feature_128', 'feature_129', 'feature_130', 'feature_131', 'feature_132', 'feature_135', 'feature_136', 'feature_137', 'feature_138', 'feature_139', 'feature_140', 'feature_144', 'feature_148', 'feature_149', 'feature_150', 'feature_151', 'feature_154', 'feature_155', 'feature_158', 'feature_159', 'feature_160', 'feature_162', 'feature_164', 'feature_165', 'feature_166', 'feature_170', 'feature_171', 'feature_172', 'feature_173', 'feature_174', 'feature_175', 'feature_176', 'feature_178', 'feature_180', 'feature_181', 'feature_182', 'feature_184', 'feature_185', 'feature_186'], axis=1)
    return df


train_df = read_data(path_train)
print('train is saving')
train_df.to_csv('all_train.csv', index=False)
print(f'all_train was saved with shape {train_df.shape}')

test_df = read_data(path_test)
print('test is saving')
test_df.to_csv('all_test.csv', index=False)
print(f'all_test was saved with shape {train_df.shape}')
