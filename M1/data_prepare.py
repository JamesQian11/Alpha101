import torch
from scipy.stats import rankdata
from dateutil import parser
import numpy as np
import numpy.linalg as la
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from single_factor import *
import glob
import os


def file_preprocess(filename):
    input_csv = pd.read_csv(filename)
    input_csv.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    input_csv.index = input_csv['time']
    input_csv = input_csv.drop(columns='time')
    return input_csv.replace(0, np.nan)


def get_file_names_list(path):
    file_names = glob.glob(os.path.join(path, '*.csv'))
    data_list = []
    for i in file_names:
        data = i.split('/')[-1].split('.')[0]
        data_list.append(data)
    data = sorted(data_list)
    return data


def returns():
    file_returns = '/Volumes/James/01_MyProjectTrees/00_GAM/dataset/分钟数据_20211114/returns'
    file_close = '/Volumes/James/01_MyProjectTrees/00_GAM/dataset/分钟数据_20211114/close'
    for d in get_file_names_list(file_close):
        print(d)
        close_files_name = f'{file_close}/{d}.csv'
        close_files_data = file_preprocess(close_files_name)

        returns_files_data = close_files_data.pct_change(periods=30)
        returns_files_name = '/returns_' + d.split('_')[-1] + '.csv'
        returns_files_path = file_returns + returns_files_name
        print(returns_files_path)
        returns_files_data.to_csv(returns_files_path, header=True)


def data_prepare(file, d):
    amount_path = file + '/amount'
    close_path = file + '/close'
    high_path = file + '/high'
    low_path = file + '/low'
    open_path = file + '/open'
    returns_path = file + '/returns'
    vol_path = file + '/vol'
    vwap_path = file + '/vwap'

    amount_name = 'amount_' + d
    close_name = 'close_' + d
    high_name = 'high_' + d
    low_name = 'low_' + d
    open_name = 'open_' + d
    returns_name = 'returns_' + d
    vol_name = 'vol_' + d
    vwap_name = 'vwap_' + d

    amount_names = f'{amount_path}/{amount_name}.csv'
    close_names = f'{close_path}/{close_name}.csv'
    high_names = f'{high_path}/{high_name}.csv'
    low_names = f'{low_path}/{low_name}.csv'
    open_names = f'{open_path}/{open_name}.csv'
    returns_names = f'{returns_path}/{returns_name}.csv'
    vol_names = f'{vol_path}/{vol_name}.csv'
    vwap_names = f'{vwap_path}/{vwap_name}.csv'

    amount_names_data = file_preprocess(amount_names)
    close_names_data = file_preprocess(close_names)
    high_names_data = file_preprocess(high_names)
    low_names_data = file_preprocess(low_names)
    open_names_data = file_preprocess(open_names)
    returns_names_data = file_preprocess(returns_names)
    vol_names_data = file_preprocess(vol_names)
    vwap_names_data = file_preprocess(vwap_names)

    return amount_names_data, close_names_data, high_names_data, low_names_data, open_names_data, returns_names_data, vol_names_data, vwap_names_data


def calc_vwap():
    day_list = get_time(20210104, 20210630)
    print(day_list)
    file = '/Volumes/James/01_MyProjectTrees/00_GAM/dataset/分钟数据_20211114_all'
    vol_path = file + '/vol'
    amount_path = file + '/amount'
    for d in day_list:
        d = str(d)
        print(d)
        amount_name = 'amount_' + d
        vol_name = 'vol_' + d
        amount_names = f'{amount_path}/{amount_name}.csv'
        vol_names = f'{vol_path}/{vol_name}.csv'
        amount_names_data = file_preprocess(amount_names)
        vol_names_data = file_preprocess(vol_names)

        vwap = amount_names_data / vol_names_data
        vwap_name = vol_path + '/vwap_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        print('vwap_name', vwap_name)
        vwap.to_csv(vwap_name, header=True)


def get_time(start, end):
    time = np.load('time.npy', allow_pickle=True).astype(int)
    return sorted([str(x) for x in time if (start <= x) & (x <= end)])


def cal_factor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = '/home/fudan/fudan2020/00_Datasets/分钟数据_20211114'
    alpha_path = '/home/fudan/fudan2020/00_Datasets/分钟数据_20211114//alpha'
    start_time = 20210104
    end_time = 20210630
    day_list = get_time(start_time, end_time)
    for day_time in day_list:
        print(day_time)
        amount, close, high, low, Open, returns, volume, vwap = data_prepare(file_path, day_time)

        # amount.to(device)
        # close.to(device)
        # high.to(device)
        # # low.to(device)
        # Open.to(device)
        # returns.to(device)
        # volume.to(device)
        # vwap.to(device)

        alpha_1 = alpha1(close, returns)
        alpha_2 = alpha2(Open, close, volume)
        alpha_3 = alpha3(Open, volume)
        alpha_4 = alpha4(low)
        alpha_5 = alpha5(Open, vwap, close)
        alpha_6 = alpha6(Open, volume)
        alpha_7 = alpha7(volume, close)
        alpha_8 = alpha8(Open, returns)
        alpha_9 = alpha9(close)
        alpha_10 = alpha10(close)
        alpha_11 = alpha11(vwap, close, volume)
        alpha_12 = alpha12(volume, close)
        alpha_13 = alpha13(volume, close)
        alpha_14 = alpha14(Open, volume, returns)
        alpha_15 = alpha15(high, volume)
        alpha_16 = alpha16(high, volume)
        alpha_17 = alpha17(volume, close)
        alpha_18 = alpha18(close, Open)
        alpha_19 = alpha19(close, returns)
        alpha_20 = alpha20(Open, high, close, low)
        alpha_21 = alpha21(volume, close)
        alpha_22 = alpha22(high, volume, close)
        alpha_23 = alpha23(high, close)
        alpha_24 = alpha24(close)
        alpha_25 = alpha25(volume, returns, vwap, high, close)
        alpha_26 = alpha26(volume, high)
        alpha_27 = alpha27(volume, vwap)
        alpha_28 = alpha28(volume, high, low, close)
        alpha_29 = alpha29(close, returns)
        alpha_30 = alpha30(close, volume)
        alpha_31 = alpha31(close, low, volume)
        alpha_32 = alpha32(close, vwap)
        alpha_33 = alpha33(Open, close)
        alpha_34 = alpha34(close, returns)
        alpha_35 = alpha35(volume, close, high, low, returns)
        alpha_36 = alpha36(Open, close, volume, returns, vwap)
        alpha_37 = alpha37(Open, close)
        alpha_38 = alpha38(close, Open)
        alpha_39 = alpha39(volume, close, returns)
        alpha_40 = alpha40(high, volume)
        alpha_41 = alpha41(high, low, vwap)
        alpha_42 = alpha42(vwap, close)
        alpha_43 = alpha43(volume, close)
        alpha_44 = alpha44(high, volume)
        alpha_45 = alpha45(close, volume)
        alpha_46 = alpha46(close)
        alpha_47 = alpha47(volume, close, high, vwap)
        alpha_49 = alpha49(close)
        alpha_50 = alpha50(volume, vwap)
        alpha_51 = alpha51(close)
        alpha_52 = alpha52(returns, volume, low)
        alpha_53 = alpha53(close, high, low)
        alpha_54 = alpha54(Open, close, high, low)
        alpha_55 = alpha55(high, low, close, volume)
        # alpha_56 = alpha56(returns, cap)
        alpha_57 = alpha57(close, vwap)
        alpha_60 = alpha60(close, high, low, volume)
        alpha_61 = alpha61(volume, vwap)
        alpha_62 = alpha62(volume, high, low, Open, vwap)
        # alpha_64 = alpha64(high, low, Open, volume, vwap)
        alpha_65 = alpha65(volume, vwap, Open)
        alpha_66 = alpha66(vwap, low, Open, high)
        # alpha_68 = alpha68(volume, high, close, low) # miss low
        alpha_71 = alpha71(volume, close, low, Open, vwap)
        alpha_72 = alpha72(volume, high, low, vwap)
        alpha_73 = alpha73(vwap, Open, low)
        alpha_74 = alpha74(volume, close, high, vwap)
        alpha_75 = alpha75(volume, vwap, low)
        alpha_77 = alpha77(volume, high, low, vwap)
        alpha_78 = alpha78(volume, low, vwap)
        alpha_81 = alpha81(volume, vwap)
        alpha_83 = alpha83(high, low, close, volume, vwap)
        alpha_84 = alpha84(vwap, close)
        alpha_85 = alpha85(volume, high, close, low)
        # alpha_86 = alpha86(high, low, vwap)
        # alpha_88 = alpha88(volume, Open, low, high, close)
        # alpha_92 = alpha92(volume, high, low, close, Open)
        # alpha_94 = alpha94(volume, vwap)
        # alpha_95 = alpha95(volume, high, low, Open)
        # alpha_96 = alpha96(volume, vwap, close)
        # alpha_98 = alpha98(volume, Open, vwap)
        # alpha_99 = alpha99(volume, high, low)
        # alpha_100 = alpha100(high, low, vwap)
        # alpha_101 = alpha101(close, Open, high, low)

        alpha001_name = alpha_path + '/alpha001_' + day_time + '.csv'
        alpha002_name = alpha_path + '/alpha002_' + day_time + '.csv'
        alpha003_name = alpha_path + '/alpha003_' + day_time + '.csv'
        alpha004_name = alpha_path + '/alpha004_' + day_time + '.csv'
        alpha005_name = alpha_path + '/alpha005_' + day_time + '.csv'
        alpha006_name = alpha_path + '/alpha006_' + day_time + '.csv'
        alpha007_name = alpha_path + '/alpha007_' + day_time + '.csv'
        alpha008_name = alpha_path + '/alpha008_' + day_time + '.csv'
        alpha009_name = alpha_path + '/alpha009_' + day_time + '.csv'
        alpha010_name = alpha_path + '/alpha010_' + day_time + '.csv'
        alpha011_name = alpha_path + '/alpha011_' + day_time + '.csv'
        alpha012_name = alpha_path + '/alpha012_' + day_time + '.csv'
        alpha013_name = alpha_path + '/alpha013_' + day_time + '.csv'
        alpha014_name = alpha_path + '/alpha014_' + day_time + '.csv'
        alpha015_name = alpha_path + '/alpha015_' + day_time + '.csv'
        alpha016_name = alpha_path + '/alpha016_' + day_time + '.csv'
        alpha017_name = alpha_path + '/alpha017_' + day_time + '.csv'
        alpha018_name = alpha_path + '/alpha018_' + day_time + '.csv'
        alpha019_name = alpha_path + '/alpha019_' + day_time + '.csv'
        alpha020_name = alpha_path + '/alpha020_' + day_time + '.csv'
        alpha021_name = alpha_path + '/alpha021_' + day_time + '.csv'
        alpha022_name = alpha_path + '/alpha022_' + day_time + '.csv'
        alpha023_name = alpha_path + '/alpha023_' + day_time + '.csv'
        alpha024_name = alpha_path + '/alpha024_' + day_time + '.csv'
        alpha025_name = alpha_path + '/alpha025_' + day_time + '.csv'
        alpha026_name = alpha_path + '/alpha026_' + day_time + '.csv'
        alpha027_name = alpha_path + '/alpha027_' + day_time + '.csv'
        alpha028_name = alpha_path + '/alpha028_' + day_time + '.csv'
        alpha029_name = alpha_path + '/alpha029_' + day_time + '.csv'
        alpha030_name = alpha_path + '/alpha030_' + day_time + '.csv'
        alpha031_name = alpha_path + '/alpha031_' + day_time + '.csv'
        alpha032_name = alpha_path + '/alpha032_' + day_time + '.csv'
        alpha033_name = alpha_path + '/alpha033_' + day_time + '.csv'
        alpha034_name = alpha_path + '/alpha034_' + day_time + '.csv'
        alpha035_name = alpha_path + '/alpha035_' + day_time + '.csv'
        alpha036_name = alpha_path + '/alpha036_' + day_time + '.csv'
        alpha037_name = alpha_path + '/alpha037_' + day_time + '.csv'
        alpha038_name = alpha_path + '/alpha038_' + day_time + '.csv'
        alpha039_name = alpha_path + '/alpha039_' + day_time + '.csv'
        alpha040_name = alpha_path + '/alpha040_' + day_time + '.csv'
        alpha041_name = alpha_path + '/alpha041_' + day_time + '.csv'
        alpha042_name = alpha_path + '/alpha042_' + day_time + '.csv'
        alpha043_name = alpha_path + '/alpha043_' + day_time + '.csv'
        alpha044_name = alpha_path + '/alpha044_' + day_time + '.csv'
        alpha045_name = alpha_path + '/alpha045_' + day_time + '.csv'
        alpha046_name = alpha_path + '/alpha046_' + day_time + '.csv'
        alpha047_name = alpha_path + '/alpha047_' + day_time + '.csv'
        alpha048_name = alpha_path + '/alpha048_' + day_time + '.csv'
        alpha049_name = alpha_path + '/alpha049_' + day_time + '.csv'
        alpha050_name = alpha_path + '/alpha050_' + day_time + '.csv'
        alpha051_name = alpha_path + '/alpha051_' + day_time + '.csv'
        alpha052_name = alpha_path + '/alpha052_' + day_time + '.csv'
        alpha053_name = alpha_path + '/alpha053_' + day_time + '.csv'
        alpha054_name = alpha_path + '/alpha054_' + day_time + '.csv'
        alpha055_name = alpha_path + '/alpha055_' + day_time + '.csv'
        alpha056_name = alpha_path + '/alpha056_' + day_time + '.csv'
        alpha057_name = alpha_path + '/alpha057_' + day_time + '.csv'
        alpha058_name = alpha_path + '/alpha058_' + day_time + '.csv'
        alpha059_name = alpha_path + '/alpha059_' + day_time + '.csv'
        alpha060_name = alpha_path + '/alpha060_' + day_time + '.csv'
        alpha061_name = alpha_path + '/alpha061_' + day_time + '.csv'
        alpha062_name = alpha_path + '/alpha062_' + day_time + '.csv'
        alpha063_name = alpha_path + '/alpha063_' + day_time + '.csv'
        alpha064_name = alpha_path + '/alpha064_' + day_time + '.csv'
        alpha065_name = alpha_path + '/alpha065_' + day_time + '.csv'
        alpha066_name = alpha_path + '/alpha066_' + day_time + '.csv'
        alpha067_name = alpha_path + '/alpha067_' + day_time + '.csv'
        alpha068_name = alpha_path + '/alpha068_' + day_time + '.csv'
        alpha069_name = alpha_path + '/alpha069_' + day_time + '.csv'
        alpha070_name = alpha_path + '/alpha070_' + day_time + '.csv'
        alpha071_name = alpha_path + '/alpha071_' + day_time + '.csv'
        alpha072_name = alpha_path + '/alpha072_' + day_time + '.csv'
        alpha073_name = alpha_path + '/alpha073_' + day_time + '.csv'
        alpha074_name = alpha_path + '/alpha074_' + day_time + '.csv'
        alpha075_name = alpha_path + '/alpha075_' + day_time + '.csv'
        alpha076_name = alpha_path + '/alpha076_' + day_time + '.csv'
        alpha077_name = alpha_path + '/alpha077_' + day_time + '.csv'
        alpha078_name = alpha_path + '/alpha078_' + day_time + '.csv'
        alpha079_name = alpha_path + '/alpha079_' + day_time + '.csv'
        alpha080_name = alpha_path + '/alpha080_' + day_time + '.csv'
        alpha081_name = alpha_path + '/alpha081_' + day_time + '.csv'
        alpha082_name = alpha_path + '/alpha082_' + day_time + '.csv'
        alpha083_name = alpha_path + '/alpha083_' + day_time + '.csv'
        alpha084_name = alpha_path + '/alpha084_' + day_time + '.csv'
        alpha085_name = alpha_path + '/alpha085_' + day_time + '.csv'
        alpha086_name = alpha_path + '/alpha086_' + day_time + '.csv'
        alpha087_name = alpha_path + '/alpha087_' + day_time + '.csv'
        alpha088_name = alpha_path + '/alpha088_' + day_time + '.csv'
        alpha089_name = alpha_path + '/alpha089_' + day_time + '.csv'
        alpha090_name = alpha_path + '/alpha090_' + day_time + '.csv'
        alpha091_name = alpha_path + '/alpha091_' + day_time + '.csv'
        alpha092_name = alpha_path + '/alpha092_' + day_time + '.csv'
        alpha093_name = alpha_path + '/alpha093_' + day_time + '.csv'
        alpha094_name = alpha_path + '/alpha094_' + day_time + '.csv'
        alpha095_name = alpha_path + '/alpha095_' + day_time + '.csv'
        alpha096_name = alpha_path + '/alpha096_' + day_time + '.csv'
        alpha097_name = alpha_path + '/alpha097_' + day_time + '.csv'
        alpha098_name = alpha_path + '/alpha098_' + day_time + '.csv'
        alpha099_name = alpha_path + '/alpha099_' + day_time + '.csv'
        alpha100_name = alpha_path + '/alpha100_' + day_time + '.csv'
        alpha101_name = alpha_path + '/alpha101_' + day_time + '.csv'

        alpha_1.to_csv(alpha001_name, header=True)
        alpha_2.to_csv(alpha002_name, header=True)
        alpha_3.to_csv(alpha003_name, header=True)
        alpha_4.to_csv(alpha004_name, header=True)
        alpha_5.to_csv(alpha005_name, header=True)
        alpha_6.to_csv(alpha006_name, header=True)
        alpha_7.to_csv(alpha007_name, header=True)
        alpha_8.to_csv(alpha008_name, header=True)
        alpha_9.to_csv(alpha009_name, header=True)
        alpha_10.to_csv(alpha010_name, header=True)
        alpha_11.to_csv(alpha011_name, header=True)
        alpha_12.to_csv(alpha012_name, header=True)
        alpha_13.to_csv(alpha013_name, header=True)
        alpha_14.to_csv(alpha014_name, header=True)
        alpha_15.to_csv(alpha015_name, header=True)
        alpha_16.to_csv(alpha016_name, header=True)
        alpha_17.to_csv(alpha017_name, header=True)
        alpha_18.to_csv(alpha018_name, header=True)
        alpha_19.to_csv(alpha019_name, header=True)
        alpha_20.to_csv(alpha020_name, header=True)
        alpha_21.to_csv(alpha021_name, header=True)
        alpha_22.to_csv(alpha022_name, header=True)
        alpha_23.to_csv(alpha023_name, header=True)
        alpha_24.to_csv(alpha024_name, header=True)
        alpha_25.to_csv(alpha025_name, header=True)
        alpha_26.to_csv(alpha026_name, header=True)
        alpha_27.to_csv(alpha027_name, header=True)
        alpha_28.to_csv(alpha028_name, header=True)
        alpha_29.to_csv(alpha029_name, header=True)
        alpha_30.to_csv(alpha030_name, header=True)
        alpha_31.to_csv(alpha031_name, header=True)
        alpha_32.to_csv(alpha032_name, header=True)
        alpha_33.to_csv(alpha033_name, header=True)
        alpha_34.to_csv(alpha034_name, header=True)
        alpha_35.to_csv(alpha035_name, header=True)
        alpha_36.to_csv(alpha036_name, header=True)
        alpha_37.to_csv(alpha037_name, header=True)
        alpha_38.to_csv(alpha038_name, header=True)
        alpha_39.to_csv(alpha039_name, header=True)
        alpha_40.to_csv(alpha040_name, header=True)
        alpha_41.to_csv(alpha041_name, header=True)
        alpha_42.to_csv(alpha042_name, header=True)
        alpha_43.to_csv(alpha043_name, header=True)
        alpha_44.to_csv(alpha044_name, header=True)
        alpha_45.to_csv(alpha045_name, header=True)
        alpha_46.to_csv(alpha046_name, header=True)
        alpha_47.to_csv(alpha047_name, header=True)
        # alpha_48.to_csv(alpha048_name, header=True)
        alpha_49.to_csv(alpha049_name, header=True)
        alpha_50.to_csv(alpha050_name, header=True)
        alpha_51.to_csv(alpha051_name, header=True)
        alpha_52.to_csv(alpha052_name, header=True)
        alpha_53.to_csv(alpha053_name, header=True)
        alpha_54.to_csv(alpha054_name, header=True)
        alpha_55.to_csv(alpha055_name, header=True)
        # alpha_56.to_csv(alpha056_name, header=True)
        alpha_57.to_csv(alpha057_name, header=True)
        # alpha_58.to_csv(alpha058_name, header=True)
        # alpha_59.to_csv(alpha059_name, header=True)
        alpha_60.to_csv(alpha060_name, header=True)
        alpha_61.to_csv(alpha061_name, header=True)
        alpha_62.to_csv(alpha062_name, header=True)
        # alpha_63.to_csv(alpha063_name, header=True)
        # alpha_64.to_csv(alpha064_name, header=True)
        alpha_65.to_csv(alpha065_name, header=True)
        alpha_66.to_csv(alpha066_name, header=True)
        # alpha_67.to_csv(alpha067_name, header=True)
        # alpha_68.to_csv(alpha068_name, header=True)
        # alpha_69.to_csv(alpha069_name, header=True)
        # alpha_70.to_csv(alpha070_name, header=True)
        alpha_71.to_csv(alpha071_name, header=True)
        alpha_72.to_csv(alpha072_name, header=True)
        alpha_73.to_csv(alpha073_name, header=True)
        alpha_74.to_csv(alpha074_name, header=True)
        alpha_75.to_csv(alpha075_name, header=True)
        # alpha_76.to_csv(alpha076_name, header=True)
        alpha_77.to_csv(alpha077_name, header=True)
        alpha_78.to_csv(alpha078_name, header=True)
        # alpha_79.to_csv(alpha079_name, header=True)
        # alpha_80.to_csv(alpha080_name, header=True)
        alpha_81.to_csv(alpha081_name, header=True)
        # alpha_82.to_csv(alpha082_name, header=True)
        # alpha_83.to_csv(alpha083_name, header=True)
        alpha_84.to_csv(alpha084_name, header=True)
        alpha_85.to_csv(alpha085_name, header=True)
        # alpha_86.to_csv(alpha086_name, header=True)
        # # alpha_87.to_csv(alpha087_name, header=True)
        # alpha_88.to_csv(alpha088_name, header=True)
        # # alpha_89.to_csv(alpha089_name, header=True)
        # # alpha_90.to_csv(alpha090_name, header=True)
        # # alpha_91.to_csv(alpha091_name, header=True)
        # alpha_92.to_csv(alpha092_name, header=True)
        # # alpha_93.to_csv(alpha093_name, header=True)
        # alpha_94.to_csv(alpha094_name, header=True)
        # alpha_95.to_csv(alpha095_name, header=True)
        # alpha_96.to_csv(alpha096_name, header=True)
        # # alpha_97.to_csv(alpha097_name, header=True)
        # alpha_98.to_csv(alpha098_name, header=True)
        # alpha_99.to_csv(alpha099_name, header=True)
        # # alpha_100.to_csv(alpha100_name, header=True)
        # alpha_101.to_csv(alpha101_name, header=True)
        print("____________________finish_______________")


def get_factor_name():
    return ['alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008',
            'alpha009', 'alpha010',
            'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016',
            'alpha017', 'alpha018',
            'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026',
            'alpha027', 'alpha028',
            'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036',
            'alpha037', 'alpha038',
            'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046',
            'alpha047', 'alpha048',
            'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha056',
            'alpha057', 'alpha058',
            'alpha059', 'alpha060', 'alpha061', 'alpha062', 'alpha063', 'alpha064', 'alpha065', 'alpha066',
            'alpha067', 'alpha068',
            'alpha069', 'alpha070', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha076',
            'alpha077', 'alpha078',
            'alpha079', 'alpha080', 'alpha081', 'alpha082', 'alpha083', 'alpha084', 'alpha085', 'alpha086',
            'alpha087', 'alpha088',
            'alpha089', 'alpha090', 'alpha091', 'alpha092', 'alpha093', 'alpha094', 'alpha095', 'alpha096',
            'alpha097', 'alpha098',
            'alpha099', 'alpha100', 'alpha101']


# 中位数去极值
def extreme_process_MAD(sample):  # 输入的sample为时间截面的股票因子df数据
    factor_name = list(sample.columns)
    for name in factor_name:
        x = sample[name]
        median = x.median()
        MAD = abs(x - median).median()
        x[x > (median + 3 * 1.4826 * MAD)] = median + 3 * 1.4826 * MAD
        x[x < (median - 3 * 1.4826 * MAD)] = median - 3 * 1.4826 * MAD
        sample[name] = x
    return sample


# 标准化
def standardize(sample):
    factor_name = list(sample.columns)
    for name in factor_name:
        x = sample[name]
        sample[name] = (x - np.mean(x)) / (np.std(x))
    return sample


if __name__ == '__main__':
    print("*********************************************************************************")
    alpha_path = '/home/fudan/fudan2020/00_Datasets/分钟数据_20211114/alpha'
    returns_path = '/home/fudan/fudan2020/00_Datasets/分钟数据_20211114/returns'
    start_time = 20210104
    end_time = 20210631
    for factor_name in get_factor_name():
        day_list = get_time(start_time, end_time)
        IC = pd.DataFrame()
        for day_time in day_list:
            alpha_name = factor_name + '_' + day_time
            alpha_path_name = f'{alpha_path}/{alpha_name}.csv'
            returns_name = 'returns_' + day_time
            returns_path_name = f'{returns_path}/{returns_name}.csv'

            if os.path.exists(alpha_path_name):
                print(alpha_path_name)
                single_alpha = file_preprocess(alpha_path_name)
                single_alpha = extreme_process_MAD(single_alpha)
                single_alpha = standardize(single_alpha)

                returns = file_preprocess(returns_path_name)
                ic_s = []
                d = list(single_alpha.index)
                for i in range(len(d)):
                    if i < 210:
                        alpha = single_alpha.loc[d[i]]
                        next_returns = returns.loc[d[i + 30]]
                        try:
                            ic_s.append(next_returns.corr(alpha, method='spearman'))
                        except:
                            ic_s.append(0)
                times = alpha_path_name.split('/')[-1].split('.')[0]
                print(times)
                IC[times] = pd.Series(ic_s, index=single_alpha.index[30:])
                print(IC)
        IC_names = factor_name + '_IC.csv'
        print(IC_names)

        IC.to_csv(IC_names, header=True)
