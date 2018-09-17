"""
http://etlcdb.db.aist.go.jp/?page_id=1181
"""
import os
import logging
import struct

from PIL import Image, ImageEnhance

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

jps_chars = {
    7: ['ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク'],
    8: ['ケ', 'コ', 'サ', 'シ', 'ス', 'セ', 'ソ', 'タ'],
    9: ['チ', 'ツ', 'テ', 'ト', 'ナ', 'ニ', 'ヌ', 'ネ'],
    10: ['ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ', 'マ', 'ミ'],
    11: ['ム', 'メ', 'モ', 'ヤ', 'イ', 'ユ', 'エ', 'ヨ'],
    12: ['ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ヰ', 'ウ'],
    13: ['ヱ', 'ヲ', 'ン']
}


def remove_dirs(base_dir, indexes=None):
    import shutil  # noqa
    for index in indexes:
        for value in jps_chars[index]:
            shutil.rmtree(os.path.join(base_dir, value))


def check_dir_not_exist(base_dir):
    for key, value in jps_chars.items():
        for char in value:
            if not os.path.exists(os.path.join(base_dir, char)):
                logging.info(key, char)


def read_record_make_dir(f, f_no_records, path_data, index):
    count, jps_index = 0, 0
    for i in range(f_no_records):
        f.seek(i * 2052)
        s = f.read(2052)
        # Return a tuple containing values unpacked according to the
        # format string fmt. Requires len(buffer) == struct.calcsize(fmt)
        # Ex: struct.calcsize('>H2sH6BI4H4B4x2016s4x') = 2052
        try:
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
        except Exception as e:
            logging.error(e)
            break

        if index < 6:
            label = r[1].decode("utf-8").strip()
        elif index == 6:
            label = r[1].decode("utf-8").strip()
            count += 1
            if count <= 1445:
                label = 'slash'
            if 4 * 1445 < count <= 5 * 1445:
                label = 'dot'
            if 6 * 1445 < count <= 7 * 1445:
                label = '␣'
        else:
            label = jps_chars[index][jps_index].strip()
            count += 1
            if (index == 9 and jps_index == 4) or \
                    (index == 12 and jps_index == 1):
                # Read more: http://etlcdb.db.aist.go.jp/?page_id=1181
                # 11287 note: ナ(NA) on Sheet 2672 is missing
                # 11287 note: リ(RI) on Sheet 2708 is missing
                if count == 1410:
                    jps_index += 1
                    count = 0
            else:
                if count == 1411:
                    jps_index += 1
                    count = 0

        label_dir = os.path.join(path_data, label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        # <class 'PIL.Image.Image'>
        img = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
        # convert float to int: F -> P
        img = img.convert('P')
        fn = os.path.join(label_dir, "{:1d}-{:4d}-{:1d}-{:2x}.png".format(
            r[0], r[2], r[3], r[4]))

        # iP.save(fn, 'PNG', bits=4)
        enhancer = ImageEnhance.Brightness(img)
        eimg = enhancer.enhance(16)
        try:
            eimg.save(fn, 'PNG')
        except OSError as e:
            # PermissionError
            logging.error(e)
    logging.info("Number of sample: %d", i + 1)


def read_no_records(path_etl):
    with open(os.path.join(path_etl, 'ETL1INFO')) as f:
        data = f.read().split('\n')
        data = data[:13]

    no_records = []
    for sample in data:
        sample = sample.split(' ')
        record = sample[1].split('=')[-1].split('record')[0]
        no_records.append(int(record))

    return no_records


def make_datasets():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    root_datasets = os.path.join(root_dir, 'datasets')
    if not os.path.exists(root_datasets):
        os.mkdir(root_datasets)

    path_etl = os.path.join(root_datasets, 'ETL1')
    path_data = os.path.join(root_datasets, 'ETL1C_data')
    if not os.path.exists(path_data):
        os.mkdir(path_data)

    no_records = read_no_records(path_etl)
    fmt_etl = 'ETL1C_{:02d}'
    for i in range(1, 14):
        # 13 datasets
        logging.info('>>> Process %s', str(i))
        filename = os.path.join(path_etl, fmt_etl.format(i))
        if not os.path.exists(filename):
            logging.info('File does not exists')
            break

        with open(filename, 'rb') as f:
            read_record_make_dir(f, no_records[i - 1], path_data, i)


if __name__ == '__main__':
    make_datasets()
