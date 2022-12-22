# coding: utf-8

import streamlit as st
import streamlit.components.v1 as components
from shillelagh.backends.apsw.db import connect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="音声配信者の機材サポート",
    menu_items={
        'Get Help': 'https://twitter.com/deiko_cs',
        'Report a bug': "https://twitter.com/deiko_cs",
        'About': """
         # 機材をレコメンドするWebアプリ
         このWebアプリは音声配信者のデータを元に、機材をレコメンドするアプリです。
         """
    })

HOP = 1000
GRAPH_WIDTH = 1200
GRAPH_HEIGHT = 300

connection = connect(":memory:")
cursor = connection.cursor()

sheet_url = st.secrets["private_gsheets_url"]
query = f'SELECT * FROM "{sheet_url}"'


@st.cache
def draw_graph(df, kizai):
    count = pd.crosstab(df['Main'], df[kizai])
    count = count.div(count.sum(axis=1), axis=0)
    count.loc['全体'] = count.sum() / len(count)

    n_rows, n_cols = count.shape
    positions = np.arange(n_rows)
    offsets = np.zeros(n_rows, dtype=count.values.dtype)
    colors = plt.get_cmap("tab20_r")(np.linspace(0, 1, n_cols))

    fig, ax = plt.subplots()
    ax.set_yticks(positions)
    ax.set_yticklabels(count.index)

    for i in range(len(count.columns)):
        bar = ax.barh(positions, count.iloc[:, i], left=offsets, color=colors[i], label=count.columns[i]
                      )
        offsets += count.iloc[:, i]

        for rect, value in zip(bar, count.iloc[:, i]):
            if (value >= 0.1):
                cx = rect.get_x() + rect.get_width() / 2
                cy = rect.get_y() + rect.get_height() / 2
                ax.text(cx, cy, f"{value*100:.1f}%",
                        color="k", ha="center", va="center")

    ax.legend(ncol=2, bbox_to_anchor=(0, -0.1),
              loc='upper left', borderaxespad=0)

    plt.tick_params(length=0)
    plt.xticks(color="None")
    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.3, top=0.99)
    fig.canvas.draw()
    data = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    c = len(data) // (w * h)
    img = Image.frombytes("RGB", (w, h), data, "raw")
    return img


@st.cache
def recommend1(option, option2):
    row_list = []
    for row in cursor.execute(query):
        row_list.append(row)

    dfr = pd.DataFrame(row_list)
    dfr.columns = ['Main', 'Mic', 'MicOthers',
                   'Audio', 'AudioOthers']
    for i in range(len(dfr.index)):
        if dfr['Mic'][i] == 'その他':
            dfr['Mic'][i] = dfr['MicOthers'][i]

        if dfr['Audio'][i] == 'その他':
            dfr['Audio'][i] = dfr['AudioOthers'][i]

    dfr = dfr.drop(['MicOthers', 'AudioOthers'], axis=1)

    countMain = dfr.groupby('Main').size().sort_values(ascending=False)
    countMic = dfr.groupby('Mic').size().sort_values(ascending=False)
    countAudio = dfr.groupby('Audio').size().sort_values(ascending=False)

    test_df = pd.DataFrame(data={'Mic': [option],  'Audio': [option2]})
    sort1 = sorted(countMic.index)
    sort2 = sorted(countAudio.index)
    sort3 = sorted(countMain.index)
    test_df['Mic'] = sort1.index(option)
    test_df['Audio'] = sort2.index(option2)

    category = ['Main', 'Mic', 'Audio']
    for c in category:
        le = LabelEncoder()
        dfr[c] = le.fit_transform(dfr[c])

    testset = test_df.loc[:, ['Mic', 'Audio']].values
    dataset = dfr.loc[:, ['Mic', 'Audio']].values

    data_M = dfr.loc[:, ['Main']].values

    model = RandomForestClassifier()
    test = model.fit(dataset, data_M.ravel()).predict(testset)
    answer = sort3[test[0]]
    return answer


@st.cache
def recommend2(option, option2):
    row_list = []
    for row in cursor.execute(query):
        row_list.append(row)

    dfr = pd.DataFrame(row_list)
    dfr.columns = ['Main', 'Mic', 'MicOthers',
                   'Audio', 'AudioOthers']
    for i in range(len(dfr.index)):
        if dfr['Mic'][i] == 'その他':
            dfr['Mic'][i] = dfr['MicOthers'][i]

        if dfr['Audio'][i] == 'その他':
            dfr['Audio'][i] = dfr['AudioOthers'][i]

    dfr = dfr.drop(['MicOthers', 'AudioOthers'], axis=1)

    countMain = dfr.groupby('Main').size().sort_values(ascending=False)
    countMic = dfr.groupby('Mic').size().sort_values(ascending=False)
    countAudio = dfr.groupby('Audio').size().sort_values(ascending=False)

    if option2 in countMic:
        c2 = countMic
        c3 = countAudio
        count2 = 'Mic'
    else:
        c2 = countAudio
        c3 = countMic
        count2 = 'Audio'

    test_df = pd.DataFrame(data={'Main': [option], count2: [option2]})
    sort1 = sorted(countMain.index)
    sort2 = sorted(c2.index)
    sort3 = sorted(c3.index)
    test_df['Main'] = sort1.index(option)
    test_df[count2] = sort2.index(option2)

    category = ['Main', 'Mic', 'Audio']
    for c in category:
        le = LabelEncoder()
        dfr[c] = le.fit_transform(dfr[c].values)

    testset = test_df.loc[:, ['Main', count2]].values
    dataset = dfr.loc[:, ['Main', count2]].values

    if option2 in countMic:
        category.remove('Mic')
    else:
        category.remove('Audio')

    category.remove('Main')

    data_M = dfr.loc[:, [category[0]]].values

    model = RandomForestClassifier()
    test = model.fit(dataset, data_M.ravel()).predict(testset)
    answer = sort3[test[0]]

    return answer


def _set_block_container_style(max_width: int = GRAPH_WIDTH + 100, max_width_100_percent: bool = False, padding_top: int = 5, padding_right: int = 1, padding_left: int = 1, padding_bottom: int = 10):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"

    st.markdown(f"""<style>
    .reportview-container .main .block-container{{{max_width_str}
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
        }}
        </style>""", unsafe_allow_html=True)


def main():
    st.title('音声配信の機材ラボ')
    st.write('create by Deiko')
    st.markdown("---")

    href = f'<a href="https://forms.gle/tYUKZXwvVPRMKNH17">Googleフォーム</a>'
    st.markdown(
        f'<span style="font-size:16px">あなたの使用機材を登録する ▶︎ {href}</span>', unsafe_allow_html=True)
    st.write('登録すると有料の「音声EQのハンドブック」（PDF）がもらえます！')
    st.markdown("---")

    row_list = []
    for row in cursor.execute(query):
        row_list.append(row)

    df = pd.DataFrame(row_list)

    df.columns = ['Main', 'Mic', 'MicOthers',
                  'Audio', 'AudioOthers']

    for i in range(len(df.index)):
        if df['Mic'][i] == 'その他':
            df['Mic'][i] = df['MicOthers'][i]

        if df['Audio'][i] == 'その他':
            df['Audio'][i] = df['AudioOthers'][i]

    df = df.drop(['MicOthers', 'AudioOthers'], axis=1)

    tab1, tab2, tab3 = st.tabs(
        ["💻 適職診断", "👪 先人の知恵", "🎤 機材の比率"])

    col1, col2 = st.columns(2)
    with tab1:
        st.markdown("---")
        st.header('【音声配信の適職診断】')
        st.write('① マイクとオーディオインターフェイスを選択')
        st.write('② あなたに向いてる音声配信を判定！')
        st.write('')
        col1, col2 = st.columns(2)
        countMic = df.groupby('Mic').size().sort_values(ascending=False)
        cMic = countMic.index.union([''])
        option1 = col1.selectbox('マイクを入力', cMic)
        countAudio = df.groupby('Audio').size().sort_values(ascending=False)
        cAudio = countAudio.index.union([''])
        option2 = col2.selectbox('オーディオインターフェイスを入力', cAudio)

        if option1 != '' and option2 != '':
            answer = recommend1(option1, option2)
            st.write('あなたに向いている音声配信は、【' + answer + '】です！')
            twitter1 = """
                <a href="http://twitter.com/intent/tweet" class="twitter-share-button"
                data-text="あなたに向いてる音声配信は、【""" + answer + """】です！ #音声配信の機材ラボ"
                data-url="https://deiko0-voice-analysis-app-m0fgp5.streamlit.app"
                Tweet
                </a>
                <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                """
            components.html(twitter1)

        st.markdown("---")

    with tab2:
        st.markdown("---")
        st.header('【先人配信者たちの知恵】')
        st.write('① メイン活動と機材を選択')
        st.write('② 機材の組み合わせを提案！')
        st.write('')
        col3, col4 = st.columns(2)
        option3 = col3.selectbox(
            'メイン活動を入力', ['', 'ASMR', 'ボイスドラマ', 'ライブ配信', 'ラジオ', '朗読', '音楽'])
        MicAudio = countAudio.index.union(cMic)
        option4 = col4.selectbox('マイク or オーディオインターフェイスを入力', MicAudio)

        if option3 != '' and option4 != '':
            answer = recommend2(option3, option4)
            st.write('【' + option1 + '】におすすめの機材の組み合わせは、【' + option2 + '】×【' + answer + '】です！')
            twitter2 = """
                <a href="http://twitter.com/intent/tweet" class="twitter-share-button"
                data-text=" """ + option1+ """におすすめの機材の組み合わせは、【""" + option2 + """ × """ + answer + """】です！ #音声配信の機材ラボ"
                data-url="https://deiko0-voice-analysis-app-m0fgp5.streamlit.app"
                Tweet
                </a>
                <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                """
            components.html(twitter2)

        st.markdown("---")

    with tab3:
        img = draw_graph(df, 'Mic')
        st.markdown("---")
        st.header("使用マイクの構成比")
        st.image(img)
        st.markdown("---")
        img = draw_graph(df, 'Audio')
        st.header("使用オーディオインターフェイスの構成比")
        st.image(img)
        st.markdown("---")


if __name__ == "__main__":
    _set_block_container_style()
    main()
