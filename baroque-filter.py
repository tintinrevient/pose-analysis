import pandas as pd
import os
from shutil import copyfile


# csv file
infile = os.path.join('dataset', 'painter-by-numbers', 'all_data_info.csv')

# dataset indir and outdir
outdir = os.path.join('dataset', 'baroque-painters')
indir = os.path.join('dataset', 'painter-by-numbers')


def extract_date(df, col):

    # extract 4 digits anywhere: e.g., c.1982, November 1982
    df[col] = df[col].str.extract(r'(\d{4})')

    # convert to numeric
    df[col] = pd.to_numeric(df[col])

    return df


def read_csv(infile):

    df = pd.read_csv(infile)

    # index
    df.set_index('date')

    # filter columns
    df_baroque = df[['artist', 'date', 'genre', 'style', 'artist_group', 'new_filename']]

    # convert 'date' column to numeric
    df_baroque = extract_date(df_baroque, 'date')

    # filter conditions
    is_baroque_start = df_baroque['date'] > 1580
    is_baroque_end = df_baroque['date'] < 1750

    # filter rows
    df_baroque = df_baroque[is_baroque_start]
    df_baroque = df_baroque[is_baroque_end]

    # filter caravaggio
    # is_caravaggio = df_baroque['artist'] == 'Caravaggio'
    # df_caravaggio = df_baroque[is_caravaggio]

    return df_baroque


def create_dataset(df, indir, outdir):

    # make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for index, row in df[['artist_group', 'new_filename']].iterrows():

        group = row['artist_group']
        fn = row['new_filename']

        print(group, fn)

        if group.find('train') != -1:
            # train images are from the 'train' folder; but sometimes it is put in the 'test' folder, given 'artist_group' == 'train_and_test'.
            src = os.path.join(indir, 'train', fn)
            dst = os.path.join(outdir, fn)

            try:
                copyfile(src, dst)
            except:
                src = os.path.join(indir, 'test', fn)
                copyfile(src, dst)
        else:
            # test images are from the 'test' folder; images are not duplicate in 'train' and 'test' folders.
            src = os.path.join(indir, 'test', fn)
            dst = os.path.join(outdir, fn)

            try:
                copyfile(src, dst)
            except:
                src = os.path.join(indir, 'train', fn)
                copyfile(src, dst)


if __name__ == '__main__':

    # filter the painters during the Baroque period
    df = read_csv(infile)

    # create the dataset
    create_dataset(df, indir=indir, outdir=outdir)