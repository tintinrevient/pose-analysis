import pandas as pd
import os
from shutil import copyfile, move
import numpy as np
import matplotlib.pyplot as plt


# dataset indir and outdir
indir = os.path.join('dataset', 'painter-by-numbers')
outdir_baroque = os.path.join('dataset', 'painter-of-baroque')
outdir_nude = os.path.join('dataset', 'painter-of-nude')

# csv file
infile = os.path.join(indir, 'all_data_info.csv')

# pix file
outfile_timeline_style = os.path.join('pix', 'timeline-top-21-styles.png')
outfile_timeline_nu_artist = os.path.join('pix', 'timeline-top-26-nu-artists.png')

outfile_pie_style = os.path.join('pix', 'pie-top-16-styles-of-nu.png')
outfile_vstack_style = os.path.join('pix', 'vstack-top-16-styles-of-nu.png')


def convert_date_to_numeric(df):

    # extract 4 digits anywhere: e.g., c.1982, November 1982
    df['date'] = df['date'].str.extract(r'(\d{4})')

    # convert to numeric
    df['date'] = pd.to_numeric(df['date'])

    # df.loc[pd.isna(df['date']), 'date'] = 0

    return df


def group_by_colname(df, is_grouped_by_artist=False, is_grouped_by_style=False, is_grouped_by_genre=False, show_all=False):

    df = df[['artist', 'style', 'genre', 'new_filename']]

    if is_grouped_by_artist and is_grouped_by_style and is_grouped_by_genre:
        df = df.groupby(['artist', 'style', 'genre']).agg({'new_filename': lambda x: x.unique().tolist()}).reset_index()[['artist', 'style', 'genre', 'new_filename']]
        df['counts'] = [len(row) for row in df['new_filename'].tolist()]
    elif is_grouped_by_artist and is_grouped_by_style:
        df = df.groupby(['artist', 'style']).agg({'new_filename': lambda x: x.unique().tolist()}).reset_index()[['artist', 'style', 'new_filename']]
        df['counts'] = [len(row) for row in df['new_filename'].tolist()]
    elif is_grouped_by_artist and is_grouped_by_genre:
        df = df.groupby(['artist', 'genre']).agg({'new_filename': lambda x: x.unique().tolist()}).reset_index()[['artist', 'genre', 'new_filename']]
        df['counts'] = [len(row) for row in df['new_filename'].tolist()]
    elif is_grouped_by_style and is_grouped_by_genre:
        df = df.groupby(['style', 'genre']).agg({'new_filename': lambda x: x.unique().tolist()}).reset_index()[['style', 'genre', 'new_filename']]
        df['counts'] = [len(row) for row in df['new_filename'].tolist()]
    elif is_grouped_by_artist:
        # aggregate for 'artist' by the total count of the paintings
        df = df.groupby(['artist']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    elif is_grouped_by_style:
        # aggregate for 'style' by the total count of the paintings
        df = df.groupby(['style']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    elif is_grouped_by_genre:
        # aggregate for 'genre' by the total count of the paintings
        df = df.groupby(['genre']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    else:
        pass

    # print all the rows
    pd.set_option('display.max_rows', df.shape[0] + 1)
    # print all columns
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    if show_all:
        pd.set_option('display.max_colwidth', None)

    # display the dataframe
    print(df)

    # return the dataframe
    return df


def combine_same_styles(df):

    # Impressionism = Impressionism + Post-Impressionism
    df_impressionism = df[df['style'].str.contains('Impressionism', case=False)]
    for style in df_impressionism['style'].tolist():
        df.drop(df[df['style'] == style].index, inplace=True)
    df.loc[df.index.max() + 1] = ['Impressionism'] + list([df_impressionism.sum().loc['counts']])
    print(df_impressionism)

    # Renaissance = Early Renaissance + Northern Renaissance + High Renaissance + Mannerism (Late Renaissance)
    df_renaissance = df[df['style'].str.contains('Renaissance', case=False)]
    for style in df_renaissance['style'].tolist():
        df.drop(df[df['style'] == style].index, inplace=True)
    df.loc[df.index.max() + 1] = ['Renaissance'] + list([df_renaissance.sum().loc['counts']])
    print(df_renaissance)

    return df


def filter_style_genre_artist(infile, artist_list, style, genre, is_grouped_by_artist=False, is_grouped_by_style=False, is_grouped_by_genre=False, show_all=False):

    df = pd.read_csv(infile)

    # filter columns
    df = df[['artist', 'date', 'genre', 'style', 'artist_group', 'new_filename']]

    if style and genre and artist_list:
        df = df[df['style'] == style]
        df = df[df['genre'] == genre]
        df = df[df['artist'].isin(artist_list)]
    elif style and genre:
        df = df[df['style'] == style]
        df = df[df['genre'] == genre]
    elif style and artist_list:
        df = df[df['style'] == style]
        df = df[df['artist'].isin(artist_list)]
    elif genre and artist_list:
        df = df[df['genre'] == genre]
        df = df[df['artist'].isin(artist_list)]
    elif style:
        df = df[df['style'] == style]
    elif genre:
        df = df[df['genre'] == genre]
    elif artist_list:
        df = df[df['artist'].isin(artist_list)]
    else:
        pass

    df_group = group_by_colname(df, is_grouped_by_artist, is_grouped_by_style, is_grouped_by_genre, show_all)

    return df, df_group


def filter_top_style_ordered_by_date(infile):

    df = pd.read_csv(infile)

    # convert date to numeric
    df = convert_date_to_numeric(df)

    # aggregate 'df' -> 'df_date': mean of date -> date
    df_date = df.groupby(['style']).agg({'date': 'mean'}).reset_index()[['style', 'date']].sort_values(by='date',
                                                                                                  ascending=True)
    # 'df': select columns
    df = df[['style', 'new_filename']]

    # filter 'df' -> 'df_style_count': style + count
    df_style_count = df.groupby(['style']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    # filter 'df_style_count' -> 'df_top_style_count': counts > 1000
    df_top_style_count = df_style_count[df_style_count['counts'] > 1000]

    # filter 'df_date': style in 'df_top_style_count'
    is_style = df_date['style'].apply(lambda x: True if x in df_top_style_count['style'].unique().tolist() else False)
    df_date = df_date[is_style]

    # add 'counts' in 'df': from 'df_top_style_count'
    df_date['counts'] = [df_top_style_count[df_top_style_count['style']==style]['counts'].values[0] for style in df_date['style'].tolist()]

    pd.set_option('display.max_rows', df_date.shape[0] + 1)
    print(df_date)

    return df_date


def filter_top_artist_of_nu_ordered_by_date(infile):

    df = pd.read_csv(infile)

    # convert date to numeric
    df = convert_date_to_numeric(df)

    # aggregate 'df' -> 'df_date': mean of date -> date
    df_date = df.groupby(['artist']).agg({'date': 'mean'}).reset_index()[['artist', 'date']].sort_values(by='date', ascending=True)

    # 'df': select artists with nude paintings
    df = df[df['genre'] == 'nude painting (nu)'][['artist', 'new_filename']]

    # filter 'df' -> 'df_artist_count': artist + count
    df_artist_count = df.groupby(['artist']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    # filter 'df_artist_count' -> 'df_top_artist_count': counts > 15
    df_top_artist_count = df_artist_count[df_artist_count['counts'] > 15]

    # filter 'df_date': artist in 'df_top_artist_count'
    is_artist = df_date['artist'].apply(lambda x: True if x in df_top_artist_count['artist'].unique().tolist() else False)
    df_date = df_date[is_artist]

    # add 'counts' in 'df': from 'df_top_artist_count'
    df_date['counts'] = [df_top_artist_count[df_top_artist_count['artist']==artist]['counts'].values[0] for artist in df_date['artist'].tolist()]

    pd.set_option('display.max_rows', df_date.shape[0] + 1)
    print(df_date)

    return df_date


def filter_combined_styles(df):

    df = combine_same_styles(df)

    # filter by: counts > 30
    df = df[df['counts'] > 30]
    df = df.sort_values(by='counts', ascending=False)

    return df


def plot_stacked_bar_chart(df, colname, outfile, is_hstack=True):

    style_names = df[colname].tolist()
    results = {colname: df['counts'].tolist()}

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    style_colors = plt.get_cmap('rainbow')(np.linspace(0.15, 0.85, data.shape[1])) # cmap = RdYlGn or rainbow

    if is_hstack:
        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())
    else:
        fig, ax = plt.subplots(figsize=(5, 9.2))
        ax.yaxis.set_visible(False)
        ax.set_ylim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(style_names, style_colors)):
        if is_hstack:
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            ax.barh(labels, width=widths, left=starts, height=0.5, label=colname, color=color)
            xcenters = starts + widths / 2
        else:
            heights = data[:, i]
            starts = data_cum[:, i] - heights
            ax.bar(labels, height=heights, bottom=starts, width=0.2, label=colname, color=color)
            xcenters = starts + heights / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        if is_hstack:
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                ax.text(x, y, str(int(c)), ha='center', va='center', color=text_color, rotation=90)
        else:
            for x, (y, c) in enumerate(zip(xcenters, heights)):
                ax.text(x, y, str(int(c)), ha='center', va='center', color=text_color)

    if is_hstack:
        ax.legend(ncol=3, loc='upper right', fontsize='small')
    else:
        ax.legend(ncol=1, loc='upper right', fontsize='small')

    # save the plot
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)

    # show the plot
    # plt.show()


def plot_pie_chart(df, outfile):

    labels = df['style'].tolist()
    sizes = df['counts'].tolist()

    colors = plt.get_cmap('rainbow')(np.linspace(0.85, 0.15, len(labels)))

    fig, ax = plt.subplots()
    _, _, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    for autotext in autotexts:
        autotext.set_color('white')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # save the plot
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)

    # show the plot
    # plt.show()


def analyze_style_and_genre(infile, styles, genre, by_date=False):

    df = pd.read_csv(infile)

    # filter by style and genre
    df = df[df['style'].isin(styles)][['artist', 'date', 'genre', 'style', 'new_filename']]
    df = df[df['genre'] == genre][['artist', 'date', 'genre', 'style', 'new_filename']]

    if by_date:
        df = convert_date_to_numeric(df)
        df = df.groupby(['style']).agg({'date': 'mean'}).reset_index()[['style', 'date']].sort_values(by='date', ascending=True)
    else:
        df = df.groupby(['style', 'genre', 'artist']).size().reset_index(name='counts').sort_values(by='style', ascending=False)
        df = df[df['counts'] >= 10][['style', 'artist', 'counts']]

    pd.set_option('display.max_rows', df.shape[0] + 1)
    print(df)


def analyze_artist(infile, artist, genre):

    df = pd.read_csv(infile)

    # filter artist
    df = df[df['artist'] == artist][['artist', 'date', 'genre', 'style', 'new_filename']]
    df = df[df['genre'] == genre][['artist', 'date', 'genre', 'style', 'new_filename']]

    df = df.groupby(['genre', 'style']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    pd.set_option('display.max_rows', df.shape[0] + 1)
    print(df)


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


def create_painter_dir(df, indir):

    for index, row in df.iterrows():

        artist = row['artist']
        fn = row['new_filename']

        painter_dir = os.path.join(indir, artist)

        if not os.path.exists(painter_dir):
            os.makedirs(painter_dir)

        src = os.path.join(indir, fn)
        dst = os.path.join(painter_dir, fn)
        move(src, dst)


if __name__ == '__main__':

    # analyze statistics of style and genre
    # analyze_style_and_genre(infile, styles=['Impressionism', 'Post-Impressionism'], genre='nude painting (nu)', by_date=True)
    # analyze_style_and_genre(infile,
    #                         styles = ['Neoclassicism', 'Romanticism', 'Realism', 'Baroque',
    #                                   'Expressionism', 'Impressionism', 'Post-Impressionism', 'High Renaissance',
    #                                   'Northern Renaissance', 'Mannerism (Late Renaissance)', 'Early Renaissance',
    #                                   'Cubism', 'Art Nouveau (Modern)', 'Surrealism', 'Symbolism', 'Art Deco',
    #                                   'Ukiyo-e', 'Academicisim', 'Magic Realism', 'Fauvism'],
    #                         genre='nude painting (nu)')

    # analyze statistics of artist
    # analyze_artist(infile, artist='Pablo Picasso', genre='nude painting (nu)')

    # filter style, genre and artist
    classical_artist_list = ['El Greco', 'Titian', 'Michelangelo', 'Caravaggio', 'Pierre-Auguste Renoir',
                             'Edgar Degas', 'Pierre-Paul Prud\'hon', 'Artemisia Gentileschi']

    modern_artist_list = ['Henri Matisse', 'Paul Gauguin', 'Paul Jacoulet', 'Kathe Kollwitz', 'Tamara de Lempicka',
                          'Amedeo Modigliani', 'Paul Delvaux', 'Felix Vallotton ']

    df, df_group = filter_style_genre_artist(infile=infile, artist_list=modern_artist_list, style=None, genre=None,
                                             is_grouped_by_artist=True, is_grouped_by_style=True, is_grouped_by_genre=False, show_all=False)
    # create_dataset(df, indir=indir, outdir=outdir_nude)
    # create_painter_dir(df, indir=outdir_nude)

    # plot the stacked bar chart for styles: (counts of styles > 1000) ordered by date
    # df = filter_top_style_ordered_by_date(infile)
    # plot_stacked_bar_chart(df, colname='style', outfile = outfile_timeline_style)

    # plot the stacked bar chart for artists of nude paintings: (counts of artists > 15) ordered by date
    # df = filter_top_artist_of_nu_ordered_by_date(infile)
    # plot_stacked_bar_chart(df, colname='artist', outfile = outfile_timeline_nu_artist)

    # plot the pie chart for the nude paintings grouped by styles
    # _, df = filter_style_genre_artist(infile=infile, artist_list=None, style=None, genre='nude painting (nu)',
    #                                   is_grouped_by_artist=False, is_grouped_by_style=True)
    # df = filter_combined_styles(df)
    # plot_stacked_bar_chart(df, colname='style', outfile=outfile_vstack_style, is_hstack=False)
    # plot_pie_chart(df, outfile=outfile_pie_style)