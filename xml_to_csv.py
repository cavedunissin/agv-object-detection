#!/usr/bin/env python3

import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse


def xml_to_csv(path):

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            )
            xml_list.append(value)

    xml_df = pd.DataFrame(
        xml_list,
        columns=[
            'filename',
            'width',
            'height',
            'class',
            'xmin',
            'ymin',
            'xmax',
            'ymax'
            ]
    )
    return xml_df


parser = argparse.ArgumentParser()
parser.add_argument(
    '--annotations-dir',
    default='./data/annotations/',
    help='annotations directory'
)
parser.add_argument(
    '--csv-file',
    default='./labels/labels.csv',
    help='label file(csv)'
)
args = parser.parse_args()


xml_df = xml_to_csv(args.annotations_dir)
xml_df.to_csv(args.csv_file, index=None)
print('Converted .xml files into', args.csv_file)
