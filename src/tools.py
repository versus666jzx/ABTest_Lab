from uuid import uuid4
from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.stats import norm
import altair as alt
import plotly.express as px
import streamlit as st


def conversion_rate(conversions, visitors):
    return (conversions / visitors) * 100


def lift(cra, crb):
    return ((crb - cra) / cra) * 100


def std_err(cr, visitors):
    return np.sqrt((cr / 100 * (1 - cr / 100)) / visitors)


def std_err_diff(sea, seb):
    return np.sqrt(sea ** 2 + seb ** 2)


def z_score(cra, crb, error):
    return ((crb - cra) / error) / 100


def p_value(z, hypothesis):
    if hypothesis == "One-sided" and z < 0:
        return 1 - norm().sf(z)
    elif hypothesis == "One-sided" and z >= 0:
        return norm().sf(z) / 2
    else:
        return norm().sf(z)


def significance(alpha, p):
    return "YES" if p < alpha else "NO"


def plot_chart(df):
    chart = (
        alt.Chart(df)
        .mark_bar(color="#61b33b")
        .encode(
            x=alt.X("Group:O", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Conversion:Q", title="Conversion rate (%)"),
            opacity="Group:O",
        )
        .properties(width=500, height=500)
    )

    chart_text = chart.mark_text(
        align="center", baseline="middle", dy=-10, color="black"
    ).encode(text=alt.Text("Conversion:Q", format=",.3g"))

    return st.altair_chart((chart + chart_text).interactive())


def style_negative(v, props=""):
    return props if v < 0 else None


def style_p_value(v, props=""):
    return np.where(v < st.session_state.alpha, "color:green;", props)


def calculate_significance(
    conversions_a, conversions_b, visitors_a, visitors_b
):
    st.session_state.cra = conversion_rate(int(conversions_a), int(visitors_a))
    st.session_state.crb = conversion_rate(int(conversions_b), int(visitors_b))
    st.session_state.uplift = lift(st.session_state.cra, st.session_state.crb)
    st.session_state.sea = std_err(st.session_state.cra, float(visitors_a))
    st.session_state.seb = std_err(st.session_state.crb, float(visitors_b))
    st.session_state.sed = std_err_diff(st.session_state.sea, st.session_state.seb)
    st.session_state.z = z_score(
        st.session_state.cra, st.session_state.crb, st.session_state.sed
    )
    st.session_state.p = p_value(st.session_state.z, st.session_state.hypothesis)
    st.session_state.significant = significance(
        st.session_state.alpha, st.session_state.p
    )


def get_dataset(size, days) -> pd.DataFrame:

    end = datetime.today()
    start = end - timedelta(days=days)

    data = pd.DataFrame(data={
        'user_id': [str(uuid4()) for _ in range(size)],
        'group':   np.random.choice(['old_version', 'new_version'], size=size),
        'timestamp': pd.date_range(start=start, end=end, periods=size)
    })

    old_version_index = data[data['group'] == 'old_version'].index
    new_version_index = data[data['group'] == 'new_version'].index

    data.loc[old_version_index, 'converted'] = np.random.choice(
                                                        [0, 1],
                                                        size=(len(old_version_index), 1),
                                                        p=[0.8, 0.2]
                                                    )

    data.loc[new_version_index, 'converted'] = np.random.choice(
                                                        [0, 1],
                                                        size=(len(new_version_index), 1),
                                                        p=[0.75, 0.25]
                                                    )

    data['converted'] = data['converted'].astype('int')

    data.loc[old_version_index, 'avg_check'] = np.random.normal(
                                                        size=len(old_version_index),
                                                        loc=15,
                                                        scale=7
                                                    )

    data.loc[new_version_index, 'avg_check'] = np.random.normal(
                                                        size=len(new_version_index),
                                                        loc=17,
                                                        scale=6.4
                                                    )

    return data


def get_plotly_converted_hist(data: pd.DataFrame):

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            dict(
                x=data[data['group'] == 'old_version']['converted'].map({1: 'Да', 0: 'Нет'}),
                name='old_version'
            )
        )
    )

    fig.add_trace(
        go.Histogram(
            dict(
                x=data[data['group'] == 'new_version']['converted'].map({1: 'Да', 0: 'Нет'}),
                name='new_version'
            )
        )
    )

    fig.update_traces(hovertemplate="Сконвертирован: %{x}<br>"
                                    "Количество: %{y}")

    fig.update_layout(
        title='Распределение конверсий в новой и старой версии сайта'
    )

    fig.update_xaxes(
        title='Сконвертирован'
    )

    fig.update_yaxes(
        title='Количество'
    )

    return fig


def get_fig(df: pd.DataFrame):

    p = []
    x = []
    with st.spinner('Строю график статзначимости...'):
        for i in range(50, df.shape[0]):
            visitors_a = df.loc[:i][df['group'] == 'old_version'].shape[0]
            visitors_b = df.loc[:i][df['group'] == 'new_version'].shape[0]

            conversions_a = df.loc[:i].groupby(['group', 'converted']).agg('count')['user_id'][3]
            conversions_b = df.loc[:i].groupby(['group', 'converted']).agg('count')['user_id'][1]

            calculate_significance(
                conversions_a,
                conversions_b,
                visitors_a,
                visitors_b
            )
            p.append(np.round(p_value(st.session_state.z, st.session_state.hypothesis) * 100, 2))
            x.append(df['timestamp'].iloc[i])

    fig = px.line(
        x=x,
        y=p,
        title='Зависимость статзначимости от времени проведения эксперимента')

    fig.update_xaxes(
        title='Количество пользователей'
    )

    fig.update_yaxes(
        title='p-value'
    )

    fig.update_layout(
        showlegend=False
    )

    fig.add_hline(
        y=st.session_state.alpha * 100,
        line_color='green',
        line_dash='dash'
    )

    fig.update_traces(hovertemplate="Время А/B теста: %{x}<br>"
                                    "Достигнутая статзначимость: %{y}%")

    return fig


def get_interval(data):
    return t.interval(
        alpha=st.session_state.alpha,
        df=2,
        loc=data['avg_check'].mean(),
        scale=data['avg_check'].sem()
    )