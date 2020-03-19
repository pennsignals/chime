from functools import reduce
from typing import Tuple, Dict, Any
import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
import i18n

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
i18n.set('filename_format', '{locale}.{format}')
i18n.set('locale', 'en')
i18n.set('fallback', 'en')
i18n.load_path.append('./locales')

delaware = 564696
chester = 519293
montgomery = 826075
bucks = 628341
philly = 1581000
S_default = delaware + chester + montgomery + bucks + philly
known_infections = 91 # update daily
known_cases = 4 # update daily

# Widgets
current_hosp = st.sidebar.number_input(
    i18n.t("Currently Hospitalized COVID-19 Patients"), value=known_cases, step=1, format="%i"
)
doubling_time = st.sidebar.number_input(
    i18n.t("Doubling time before social distancing (days)"), value=6, step=1, format="%i"
)
relative_contact_rate = st.sidebar.number_input(
    i18n.t("Social distancing (% reduction in social contact)"), 0, 100, value=0, step=5, format="%i"
)/100.0

hosp_rate = (
    st.sidebar.number_input(i18n.t("Hospitalization %(total infections)"), 0.0, 100.0, value=5.0, step=1.0, format="%f")
    / 100.0
)
icu_rate = (
    st.sidebar.number_input(i18n.t("ICU %(total infections)"), 0.0, 100.0, value=2.0, step=1.0, format="%f") / 100.0
)
vent_rate = (
    st.sidebar.number_input(i18n.t("Ventilated %(total infections)"), 0.0, 100.0, value=1.0, step=1.0, format="%f")
    / 100.0
)
hosp_los = st.sidebar.number_input(i18n.t("Hospital Length of Stay"), value=7, step=1, format="%i")
icu_los = st.sidebar.number_input(i18n.t("ICU Length of Stay"), value=9, step=1, format="%i")
vent_los = st.sidebar.number_input(i18n.t("Vent Length of Stay"), value=10, step=1, format="%i")
Penn_market_share = (
    st.sidebar.number_input(
        i18n.t("Hospital Market Share (%)"), 0.0, 100.0, value=15.0, step=1.0, format="%f"
    )
    / 100.0
)
S = st.sidebar.number_input(
    i18n.t("Regional Population"), value=S_default, step=100000, format="%i"
)

initial_infections = st.sidebar.number_input(
    i18n.t("Currently Known Regional Infections (only used to compute detection rate - does not change projections)"), value=known_infections, step=10, format="%i"
)

total_infections = current_hosp / Penn_market_share / hosp_rate
detection_prob = initial_infections / total_infections

S, I, R = S, initial_infections / detection_prob, 0

intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1

recovery_days = 14.0
# mean recovery rate, gamma, (in 1/days).
gamma = 1 / recovery_days

# Contact rate, beta
beta = (
    intrinsic_growth_rate + gamma
) / S * (1-relative_contact_rate) # {rate based on doubling time} / {initial S}

r_t = beta / gamma * S # r_t is r_0 after distancing
r_naught = r_t / (1-relative_contact_rate)
doubling_time_t = 1/np.log2(beta*S - gamma +1) # doubling time after distancing

def head():
    st.markdown(i18n.t("Penn Medicine - COVID-19 Hospital Impact Model for Epidemics"), unsafe_allow_html=True)
    st.markdown(i18n.t("This tool was developed by..."))

    st.markdown(
    i18n.t("The estimated number of currently infected...").format(
        total_infections=total_infections,
        initial_infections=initial_infections,
        detection_prob=detection_prob,
        current_hosp=current_hosp,
        hosp_rate=hosp_rate,
        S=S,
        Penn_market_share=Penn_market_share,
        recovery_days=recovery_days,
        r_naught=r_naught,
        doubling_time=doubling_time,
        relative_contact_rate=relative_contact_rate,
        r_t=r_t,
        doubling_time_t=doubling_time_t
    )
    )

    return None

head()

def show_more_info_about_this_tool():
    """a lot of streamlit writing to screen."""

    st.subheader(
        i18n.t("Discrete-time SIR modeling")
    )
    st.markdown(
        i18n.t("The model consists of individuals who are either...")
    )
    st.markdown(i18n.t("The dynamics are given by the following 3 equations."))

    st.latex("S_{t+1} = (-\\beta S_t I_t) + S_t")
    st.latex("I_{t+1} = (\\beta S_t I_t - \\gamma I_t) + I_t")
    st.latex("R_{t+1} = (\\gamma I_t) + R_t")

    st.markdown(
       i18n.t("To project the expected impact to Penn Medicine...")
    )
    st.latex("\\beta = \\tau \\times c")

    st.markdown(i18n.t("which is the transmissibility multiplied...").format(recovery_days=int(recovery_days)    , c='c'))
    st.latex("R_0 = \\beta /\\gamma")
    st.markdown(i18n.t("$R_0$ gets bigger when...").format(doubling_time=doubling_time,
           recovery_days=recovery_days,
           r_naught=r_naught,
           relative_contact_rate=relative_contact_rate,
           doubling_time_t=doubling_time_t,
           r_t=r_t)
    )
    st.latex("g = 2^{1/T_d} - 1")

    st.markdown(
        i18n.t("Since the rate of new infections in the SIR model...").format(
            delaware=delaware,
            chester=chester,
            montgomery=montgomery,
            bucks=bucks,
            philly=philly,
        )
    )
    return None

if st.checkbox(i18n.t("Show more info about this tool")):
    show_more_info_about_this_tool()

# The SIR model, one time step
def sir(y, beta, gamma, N):
    S, I, R = y
    Sn = (-beta * S * I) + S
    In = (beta * S * I - gamma * I) + I
    Rn = gamma * I + R
    if Sn < 0:
        Sn = 0
    if In < 0:
        In = 0
    if Rn < 0:
        Rn = 0

    scale = N / (Sn + In + Rn)
    return Sn * scale, In * scale, Rn * scale


# Run the SIR model forward in time
def sim_sir(S, I, R, beta, gamma, n_days, beta_decay=None):
    N = S + I + R
    s, i, r = [S], [I], [R]
    for day in range(n_days):
        y = S, I, R
        S, I, R = sir(y, beta, gamma, N)
        if beta_decay:
            beta = beta * (1 - beta_decay)
        s.append(S)
        i.append(I)
        r.append(R)

    s, i, r = np.array(s), np.array(i), np.array(r)
    return s, i, r


n_days = st.slider(i18n.t("Number of days to project"), 30, 200, 60, 1, "%i")

beta_decay = 0.0
s, i, r = sim_sir(S, I, R, beta, gamma, n_days, beta_decay=beta_decay)


hosp = i * hosp_rate * Penn_market_share
icu = i * icu_rate * Penn_market_share
vent = i * vent_rate * Penn_market_share

days = np.array(range(0, n_days + 1))
data_list = [days, hosp, icu, vent]
data_dict = dict(zip(["day", "hosp", "icu", "vent"], data_list))

projection = pd.DataFrame.from_dict(data_dict)

st.subheader(i18n.t("New Admissions"))
st.markdown(i18n.t("Projected number of **daily** COVID-19 admissions at Penn hospitals"))

# New cases
projection_admits = projection.iloc[:-1, :] - projection.shift(1)
projection_admits[projection_admits < 0] = 0

plot_projection_days = n_days - 10
projection_admits["day"] = range(projection_admits.shape[0])


def new_admissions_chart(projection_admits: pd.DataFrame, plot_projection_days: int) -> alt.Chart:
    """docstring"""
    projection_admits = projection_admits.rename(columns={"hosp": i18n.t("Hospitalized"), "icu": i18n.t("ICU"), "vent": i18n.t("Ventilated")})
    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=[i18n.t("Hospitalized"), i18n.t("ICU"), i18n.t("Ventilated")])
        .mark_line(point=True)
        .encode(
            x=alt.X("day", title=i18n.t("Days from today")),
            y=alt.Y("value:Q", title=i18n.t("Daily admissions")),
            color="key:N",
            tooltip=["day", "key:N"]
        )
        .interactive()
    )

st.altair_chart(new_admissions_chart(projection_admits, plot_projection_days), use_container_width=True)



if st.checkbox(i18n.t("Show Projected Admissions in tabular form")):
    admits_table = projection_admits[np.mod(projection_admits.index, 7) == 0].copy()
    admits_table["day"] = admits_table.index
    admits_table.index = range(admits_table.shape[0])
    admits_table = admits_table.fillna(0).astype(int)
    st.dataframe(admits_table)

st.subheader(i18n.t("Admitted Patients (Census)"))
st.markdown(
    i18n.t("Projected **census** of COVID-19 patients, accounting for arrivals and discharges at Penn hospitals")
)

def _census_table(projection_admits, hosp_los, icu_los, vent_los) -> pd.DataFrame:
    """ALOS for each category of COVID-19 case (total guesses)"""

    los_dict = {
        "hosp": hosp_los,
        "icu": icu_los,
        "vent": vent_los,
    }

    census_dict = dict()
    for k, los in los_dict.items():
        census = (
            projection_admits.cumsum().iloc[:-los, :]
            - projection_admits.cumsum().shift(los).fillna(0)
        ).apply(np.ceil)
        census_dict[k] = census[k]


    census_df = pd.DataFrame(census_dict)
    census_df["day"] = census_df.index
    census_df = census_df[["day", "hosp", "icu", "vent"]]

    census_table = census_df[np.mod(census_df.index, 7) == 0].copy()
    census_table.index = range(census_table.shape[0])
    census_table.loc[0, :] = 0
    census_table = census_table.dropna().astype(int)

    return census_table

census_table = _census_table(projection_admits, hosp_los, icu_los, vent_los)

def admitted_patients_chart(census: pd.DataFrame) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": i18n.t("Hospital Census"), "icu": i18n.t("ICU Census"), "vent": i18n.t("Ventilated Census")})

    return (
        alt
        .Chart(census)
        .transform_fold(fold=[i18n.t("Hospital Census"), i18n.t("ICU Census"), i18n.t("Ventilated Census")])
        .mark_line(point=True)
        .encode(
            x=alt.X("day", title=i18n.t("Days from today")),
            y=alt.Y("value:Q", title=i18n.t("Census")),
            color="key:N",
            tooltip=["day", "key:N"]
        )
        .interactive()
    )

st.altair_chart(admitted_patients_chart(census_table), use_container_width=True)

if st.checkbox(i18n.t("Show Projected Census in tabular form")):
    st.dataframe(census_table)

def additional_projections_chart(i: np.ndarray, r: np.ndarray) -> alt.Chart:
    dat = pd.DataFrame({i18n.t("Infected"): i, i18n.t("Recovered"): r})

    return (
        alt
        .Chart(dat.reset_index())
        .transform_fold(fold=[i18n.t("Infected"), i18n.t("Recovered")])
        .mark_line()
        .encode(
            x=alt.X("index", title=i18n.t("Days from today")),
            y=alt.Y("value:Q", title=i18n.t("Case Volume")),
            tooltip=["key:N", "value:Q"],
            color="key:N"
        )
        .interactive()
    )

st.markdown(
    i18n.t("**Click the checkbox below to view additional data generated by this simulation**")
)

def show_additional_projections():
    st.subheader(
        i18n.t("The number of infected and recovered individuals in the hospital catchment region at any given moment")
    )

    st.altair_chart(additional_projections_chart(i, r), use_container_width=True)

    if st.checkbox(i18n.t("Show Raw SIR Similation Data")):
        # Show data
        days = np.array(range(0, n_days + 1))
        data_list = [days, s, i, r]
        data_dict = dict(zip(["day", "susceptible", "infections", "recovered"], data_list))
        projection_area = pd.DataFrame.from_dict(data_dict)
        infect_table = (projection_area.iloc[::7, :]).apply(np.floor)
        infect_table.index = range(infect_table.shape[0])

        st.dataframe(infect_table)

if st.checkbox(i18n.t("Show Additional Projections")):
    show_additional_projections()


# Definitions and footer

st.subheader(i18n.t("Guidance on Selecting Inputs"))
st.markdown(
    i18n.t("**Hospitalized COVID-19 Patients:**...")
)


st.subheader(i18n.t("References & Acknowledgements"))
st.markdown(
    """* AHA Webinar, Feb 26, James Lawler, MD, an associate professor University of Nebraska Medical Center, What Healthcare Leaders Need To Know: Preparing for the COVID-19
* We would like to recognize the valuable assistance in consultation and review of model assumptions by Michael Z. Levy, PhD, Associate Professor of Epidemiology, Department of Biostatistics, Epidemiology and Informatics at the Perelman School of Medicine
    """
)
st.markdown("© 2020, The Trustees of the University of Pennsylvania")
