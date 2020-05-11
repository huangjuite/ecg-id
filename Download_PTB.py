# download PTB file( .mat, .info, .html)
# coding=utf-8
import os
import urllib.request

# -----defined 'cbk' function----- #


def cbk(a, b, c):
    '''''回调函数
    @a:已经下载的数据块
    @b:数据块的大小
    @c:远程文件的大小
    '''
    per = 100.0 * a * b / c
    if per > 100:
        per = 100
    print
    '%.2f%%' % per
# --------------end--------------- #


# -----read RECORDS.txt(store all data name)----- #
# change directory & Get RECORDS.txt
# os.chdir("data")  # save directory

try:
    os.mkdir("data")
except:
    pass

dir = os.path.abspath('./data')
url = 'https://archive.physionet.org/physiobank/database/ptbdb/RECORDS'
work_path = os.path.join(dir, 'RECORDS.txt')
urllib.request.urlretrieve(url, work_path, cbk)


# read RECORDS.txt
f = open("RECORDS.txt", "r")
Data0 = []
Data1 = []
for lines in f.readlines():
    tempData = ""
    tempData = lines.rstrip()  # delete"\n"
    tempData = tempData.split("/")
    Data0.append(tempData[0])  # Patient Num
    Data1.append(tempData[1])  # Data Num => 549 files
f.close()
# print(Data1)


def auto_down(url_func, path):
    try:
        urllib.request.urlretrieve(url_func, path)
    except urllib.error.ContentTooShortError:  # urllib.error.URLError as e:
        print('Network conditions is not good.Reloading.')
        auto_down(url_func, path)  # Try again !!!!

# ----------------------end---------------------- #


# download ".hea file"
url_seg1 = 'https://archive.physionet.org/atm/ptbdb/'
url_seg2 = '/0/10/export/matlab/'
url_seg3 = 'm.hea'
url_seg4 = 'm.info'
url_seg5 = 'm.mat'
MIList = ['antero-lateral', 'anterior', 'antero-septal', 'antero-septo-lateral', 'inferior',
          'infero-lateral', 'infero-posterior', 'infero-postero-lateral', 'lateral',
          'posterior', 'postero-lateral', 'Normal']
# for i in range(len(Data0)):
# for i in [25]:
for i in range(len(Data0)):
    # create directory
    dir = os.path.abspath('./data')
    work_path = os.path.join(dir, Data0[i])
    if not os.path.exists(work_path):
        os.makedirs(work_path)
    # print("Create dir:" + work_path)

    # produce url
    patientURL = url_seg1 + Data0[i] + '/' + \
        Data1[i] + url_seg2 + Data1[i] + url_seg3
    # 'https://archive.physionet.org/atm/ptbdb/patient001/s0010_re/0/10/export/matlab/s0010_rem.hea'
    url_Info = url_seg1 + Data0[i] + '/' + \
        Data1[i] + url_seg2 + Data1[i] + url_seg4
    # 'https://archive.physionet.org/atm/ptbdb/patient001/s0010_re/0/10/export/matlab/s0010_rem.info'
    url_Mat = url_seg1 + Data0[i] + '/' + \
        Data1[i] + url_seg2 + Data1[i] + url_seg5
    # 'https://archive.physionet.org/atm/ptbdb/patient001/s0010_re/0/10/export/matlab/s0010_rem.mat'

    # print connection state (e.g. 200, 404, 501...)
    try:
        conn = urllib.request.urlopen(url_Info)
    except urllib.error.HTTPError as e:
        # Return code error (e.g. 404, 501, ...)
        # ...
        print(url_Info)
        print('HTTPError: {}'.format(e.code))
        continue
    except urllib.error.URLError as e:
        # Not an HTTP-specific error (e.g. connection refused)
        # ...
        continue
        print(url_Info)
        print('URLError: {}'.format(e.reason))
    else:
        # 200
        # ...
        # print('good connection for .info')
        pass
    try:
        conn = urllib.request.urlopen(url_Mat)
    except urllib.error.HTTPError as e:
        # Return code error (e.g. 404, 501, ...)
        # ...
        print(url_Mat)
        print('HTTPError: {}'.format(e.code))
        continue
    except urllib.error.URLError as e:
        # Not an HTTP-specific error (e.g. connection refused)
        # ...
        print(url_Mat)
        print('URLError: {}'.format(e.reason))
        continue
    else:
        # 200
        # ...
        # print('good connection for .mat')

        # e.g. url = 'https://archive.physionet.org/atm/ptbdb/patient021/s0073lre/0/10/export/matlab/s0073lrem.hea'

        # print(patientURL)
        pass

    dir = os.path.abspath('./data')
    HeaName = Data1[i] + '.txt'
    work_path = os.path.join(dir, Data0[i])
    work_path = os.path.join(work_path, HeaName)
    urllib.request.urlretrieve(
        patientURL, work_path, cbk)  # download .hea file
    # print('Get .hea_' + str(i))
    # read patient.txt [e.g. s0073lrem.txt]
    file = open(work_path, "r")
    temp = file.readlines()
    file.close()

    # -----download MI .info file to certain directory-----#
    url_Info = url_seg1 + Data0[i] + '/' + \
        Data1[i] + url_seg2 + Data1[i] + url_seg4
    # print(url_Info)
    # e.g. url = 'https://archive.physionet.org/atm/ptbdb/patient021/s0073lre/0/10/export/matlab/s0073lrem.info'
    dir = os.path.abspath('./data')
    InfoFile = Data1[i] + url_seg4  # filename of .info檔
    work_path = os.path.join(dir, Data0[i])
    work_path = os.path.join(work_path, InfoFile)
    # print(work_path)
    urllib.request.urlretrieve(url_Info, work_path, cbk)
    # print('Get .info_' + str(i))
    # -----download Patient ".mat file" to certain directory-----#
    url_Mat = url_seg1 + Data0[i] + '/' + \
        Data1[i] + url_seg2 + Data1[i] + url_seg5
    # print(url_Mat)
    # e.g. url = 'https://archive.physionet.org/atm/ptbdb/patient021/s0073lre/0/10/export/matlab/s0073lrem.mat'
    dir = os.path.abspath('./data')
    MatFile = Data1[i] + url_seg5  # filename of .info檔
    work_path = os.path.join(dir, Data0[i])
    work_path = os.path.join(work_path, MatFile)
    # print(work_path)
    auto_down(url_Mat, work_path)
    # urllib.request.urlretrieve(url_Mat, work_path, cbk)
    # print('Get .mat_' + str(i))

    """
    MIclass = 0
    if temp[20].find('Healthy control') != -1: #check 是否為Normal
        MIclass = 12
    elif temp[20].find('Myocardial infarction') != -1: #check 是否為MI
        # find acute location of MI
        if temp[21].find('antero-lateral') != -1:
            MIclass = 1
        elif temp[21].find('anterior') != -1:
            MIclass = 2
        elif temp[21].find('antero-septal') != -1:
            MIclass = 3
        elif temp[21].find('antero-septo-lateral') != -1:
            MIclass = 4
        elif temp[21].find('inferior') != -1:
            MIclass = 5
        elif (temp[21].find('infero-latera') != -1) or (temp[21].find('infero-lateral') != -1):
            MIclass = 6
        elif temp[21].find('infero-posterior') != -1:
            MIclass = 7
        elif (temp[21].find('infero-postero-lateral') != -1) or (temp[21].find('infero-poster-lateral') != -1):
            MIclass = 8
        elif temp[21].find('postero-lateral') != -1:
            MIclass = 11
        elif temp[21].find('lateral') != -1:
            MIclass = 9
        elif temp[21].find('posterior') != -1:
            MIclass = 10
        elif (temp[21].find('no') != -1) and (len(temp[22].split(" ")) == 5): # No acute MI & only 1 location
            former = temp[22].split(" ")
            print('No Acute')
            print('MI: '+former[4])
            if former[4].find('antero-lateral') != -1:
                MIclass = 1
            elif former[4].find('anterior') != -1:
                MIclass = 2
            elif former[4].find('antero-septal') != -1:
                MIclass = 3
            elif former[4].find('antero-septo-lateral') != -1:
                MIclass = 4
            elif former[4].find('inferior') != -1:
                MIclass = 5
            elif (former[4].find('infero-latera') != -1) or (former[4].find('infero-lateral') != -1):
                MIclass = 6
            elif former[4].find('infero-posterior') != -1:
                MIclass = 7
            elif (former[4].find('infero-postero-lateral') != -1) or (former[4].find('infero-poster-lateral') != -1):
                MIclass = 8
            elif former[4].find('postero-lateral') != -1:
                MIclass = 11
            elif former[4].find('lateral') != -1:
                MIclass = 9
            elif former[4].find('posterior') != -1:
                MIclass = 10
    """

    """
    if MIclass != 0: # 可歸類之MI patient
        #-----download MI .info file to certain directory-----#
        url_Info = url_seg1 + Data0[i] + '/' + Data1[i] + url_seg2 + Data1[i] + url_seg4
        print(url_Info)
        # e.g. url = 'https://archive.physionet.org/atm/ptbdb/patient021/s0073lre/0/10/export/matlab/s0073lrem.info'
        dir = os.path.abspath('.')
        MIPath = MIList[MIclass-1]
        InfoFile = Data1[i] + url_seg4 # filename of .info檔
        work_path = os.path.join(dir, MIPath)
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        work_path = os.path.join(work_path, InfoFile)
        print(work_path)
        urllib.request.urlretrieve(url_Info, work_path, cbk)
        print('Get .info_' + str(i))
        #-----download MI .mat file to certain directory-----#
        url_Mat = url_seg1 + Data0[i] + '/' + Data1[i] + url_seg2 + Data1[i] + url_seg5
        print(url_Mat)
        # e.g. url = 'https://archive.physionet.org/atm/ptbdb/patient021/s0073lre/0/10/export/matlab/s0073lrem.mat'
        dir = os.path.abspath('.')
        MIPath = MIList[MIclass-1]
        MatFile = Data1[i] + url_seg5 # filename of .info檔
        work_path = os.path.join(dir, MIPath)
        work_path = os.path.join(work_path, MatFile)
        print(work_path)
        auto_down(url_Mat, work_path)
        #urllib.request.urlretrieve(url_Mat, work_path, cbk)
        print('Get .mat_' + str(i))

    """
