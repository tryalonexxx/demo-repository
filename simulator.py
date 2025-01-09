from datetime import datetime, timedelta

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots


def run_simulation(params):
    # Unpack parameters
    current_wtu = params["current_wtu"]
    initial_w1 = params["initial_w1"]
    initial_plateau = params["initial_plateau"]
    weeks_to_plateau = params["weeks_to_plateau"]
    final_w1 = params["final_w1"]
    final_plateau = params["final_plateau"]
    weeks_to_increase = params["weeks_to_increase"]
    cac = params["cac"]
    marketing_percentage = params["marketing_percentage"]
    current_weekly_arpu = params["current_weekly_arpu"]

    weeks_to_simulate = params["weeks_to_simulate"]

    final_weekly_arpu = params["final_weekly_arpu"]
    weeks_to_increase_arpu = params["weeks_to_increase_arpu"]
    # cac_multiplier = params['cac_multiplier']
    # cac_multiplier_limit = params['cac_multiplier_limit']

    total_acquired_users = 0
    original_cac = cac
    current_cac = cac

    # 플래토 계산
    w1_increase_per_week = (final_w1 - initial_w1) / weeks_to_increase
    plateau_increase_per_week = (final_plateau - initial_plateau) / weeks_to_increase
    w1_values = [
        min(initial_w1 + i * w1_increase_per_week, final_w1, key=abs)
        for i in range(weeks_to_simulate)
    ]
    plateau_values = [
        min(initial_plateau + i * plateau_increase_per_week, final_plateau, key=abs)
        for i in range(weeks_to_simulate)
    ]
    retention_values = []

    # arpu 계산
    w1_arpu_increase_per_week = (
        final_weekly_arpu - current_weekly_arpu
    ) / weeks_to_increase_arpu
    arpu_values = [
        min(
            current_weekly_arpu + i * w1_arpu_increase_per_week,
            final_weekly_arpu,
            key=abs,
        )
        for i in range(weeks_to_simulate)
    ]
    # arpu_values = []
    # for week in range(weeks_to_simulate):
    #     cohort_arpu = [current_weekly_arpu]
    #     for i in range(weeks_to_simulate):
    #         if i < weekly_arpu_growth_weeks:
    #             new_arpu = cohort_arpu[-1] * (1 + weekly_arpu_growth_rate)
    #         else:
    #             new_arpu = cohort_arpu[-1]
    #         cohort_arpu.append(new_arpu)
    #     arpu_values.append(cohort_arpu)

    arpu_curves = []
    for i in range(weeks_to_simulate):
        arpu_curves.append(arpu_values)

    for week in range(weeks_to_simulate):
        w1 = w1_values[week]
        plateau = plateau_values[week]
        retention_curve = [
            (
                max(w1 - (w1 - plateau) * (i / weeks_to_plateau), plateau)
                if i < weeks_to_plateau
                else plateau
            )
            for i in range(weeks_to_simulate)
        ]
        retention_values.append(retention_curve)

    # calculate wtu, gmv per week per cohort based on arpu, retention, and new wtu

    weekly_wtu = [current_wtu]
    weekly_revenue = [current_wtu * current_weekly_arpu]
    weekly_new_wtu = [0]
    weekly_retained_wtu = [0]
    user_per_cohort_matrix = []
    weekly_gmv = [current_wtu * current_weekly_arpu]
    marketing_cost = [0]
    cohort_ltv = [0]
    cac_list = []

    for week in range(1, weeks_to_simulate):
        # Check if we've exceeded maximum marketing cost
        total_marketing_spent = sum(marketing_cost)

        new_wtu = np.floor((weekly_gmv[-1] * marketing_percentage) / current_cac)
        cac_list.append(current_cac)
        marketing_cost.append(weekly_gmv[-1] * marketing_percentage)
        # Update total acquired users and check CAC multiplier
        total_acquired_users += new_wtu
        # multiplier = total_acquired_users // 10000000000000  # Number of times we've hit 10M

        # if multiplier > 0:
        #     current_cac = original_cac * (1.25 ** multiplier)

        # if total_marketing_spent >= maximum_marketing_cost:
        #     # If exceeded, append zeros for new users and marketing cost
        #     new_wtu = 0
        #     marketing_cost.append(0)
        # else:
        #     # Calculate new users normally
        #     new_wtu = np.floor(
        #         (weekly_gmv[-1] * marketing_percentage) / current_cac)
        #     cac_list.append(current_cac)
        #     marketing_cost.append(weekly_gmv[-1] * marketing_percentage)

        #     # Update total acquired users and check CAC multiplier
        #     total_acquired_users += new_wtu
        #     multiplier = total_acquired_users // 100000000  # Number of times we've hit 10M

        #     if multiplier > 0:
        #         current_cac = original_cac * (1.25 ** multiplier)

        retained_wtu = 0
        retained_gmv = 0
        user_per_cohort_list = []
        retained_gmv_per_cohort_list = []

        for i in range(week):
            user_per_cohort = (
                np.floor(weekly_new_wtu[i]) * retention_values[week - i - 1][i]
            )
            arpu_per_cohort = arpu_curves[week - i - 1][i]
            gmv_per_cohort = user_per_cohort * arpu_per_cohort
            user_per_cohort_list.append(user_per_cohort)
            retained_gmv_per_cohort_list.append(gmv_per_cohort)
            retained_wtu += np.floor(user_per_cohort)
            retained_gmv += gmv_per_cohort

        current_week_wtu = current_wtu + new_wtu + retained_wtu
        current_week_gmv = (
            new_wtu * current_weekly_arpu
            + current_wtu * arpu_curves[0][week]
            + retained_gmv
        )
        user_per_cohort_matrix.append(user_per_cohort_list)
        weekly_wtu.append(current_week_wtu)
        weekly_new_wtu.append(new_wtu)
        weekly_retained_wtu.append(retained_wtu)
        weekly_gmv.append(current_week_gmv)

        weeks_in_year = 52

        if week + weeks_in_year > weeks_to_simulate:
            continue

        # Get retention curve and ARPU for this cohort
        retention_curve = retention_values[week][:weeks_in_year]
        cohort_arpu = arpu_curves[week][:weeks_in_year]

        # Calculate weekly retained WTU and revenue
        weekly_retained_users = [
            np.floor(weekly_new_wtu[week] * retention_curve[w])
            for w in range(weeks_in_year)
        ]
        weekly_revenue = [
            np.floor(weekly_retained_users[w] * cohort_arpu[w])
            for w in range(weeks_in_year)
        ]

        # Sum up to get 1-year LTV for the cohort
        ltv = (sum(weekly_revenue) + current_wtu * arpu_curves[0][week]) / (
            weekly_new_wtu[week] + current_wtu
        )
        print()
        cohort_ltv.append(ltv)

    def calculate_monthly_gmv(dates, weekly_revenue):
        monthly_gmv = {}
        for i, date in enumerate(dates):
            week_revenue = weekly_revenue[i]
            week_start = date
            week_end = week_start + timedelta(days=6)  # End of the week

            if week_start.month == week_end.month:
                # If the week is contained within a single month
                month_key = week_start.strftime("%Y-%m")
                monthly_gmv[month_key] = monthly_gmv.get(month_key, 0) + week_revenue
            else:
                # For weeks that span two months
                # Calculate days in each month
                days_in_first_month = (
                    week_start.replace(day=1) + timedelta(days=32)
                ).replace(day=1) - week_start
                days_in_first_month = days_in_first_month.days
                days_in_second_month = 7 - days_in_first_month

                # Split revenue proportionally
                first_month_revenue = week_revenue * (days_in_first_month / 7)
                second_month_revenue = week_revenue * (days_in_second_month / 7)

                # Add to respective months
                first_month_key = week_start.strftime("%Y-%m")
                second_month_key = week_end.strftime("%Y-%m")

                monthly_gmv[first_month_key] = (
                    monthly_gmv.get(first_month_key, 0) + first_month_revenue
                )
                monthly_gmv[second_month_key] = (
                    monthly_gmv.get(second_month_key, 0) + second_month_revenue
                )

        return monthly_gmv

    start_date = datetime.now()
    dates = [start_date + timedelta(weeks=i) for i in range(weeks_to_simulate)]

    monthly_gmv = calculate_monthly_gmv(dates, weekly_gmv)

    monthly_gmv_df = pd.DataFrame(list(monthly_gmv.items()), columns=["Month", "GMV"])
    monthly_gmv_df.sort_values("Month", inplace=True)

    average_arpu = [weekly_gmv[i] / weekly_wtu[i] for i in range(weeks_to_simulate)]

    return {
        "dates": dates,
        "weekly_wtu": weekly_wtu,
        "weekly_new_wtu": weekly_new_wtu,
        "weekly_retained_wtu": weekly_retained_wtu,
        "weekly_gmv": weekly_gmv,
        "average_arpu": average_arpu,
        "marketing_cost": marketing_cost,
        "cohort_ltv": cohort_ltv,
        "cac_list": cac_list,
        "monthly_gmv_df": monthly_gmv_df,
        "arpu_values": arpu_values,
        "retention_values": retention_values,
        "arpu_curves": arpu_curves,
    }


# Streamlit UI
st.title("Business Metrics Simulator")

# Initial parameters
# Initial parameters
st.sidebar.header("Current State")
params = {
    "current_wtu": st.sidebar.number_input("Current WTU", value=200000),
}

# Retention parameters
st.sidebar.header("Retention Parameters")
params.update(
    {
        "initial_w1": st.sidebar.slider("Current WPR W1", 0.0, 1.0, 0.30),
        "initial_plateau": st.sidebar.slider("Current WPR Plateau", 0.0, 1.0, 0.10),
        "weeks_to_plateau": st.sidebar.number_input("Weeks to Plateau", value=10),
        "final_w1": st.sidebar.slider("To-be WPR W1", 0.0, 1.0, 0.40),
        "final_plateau": st.sidebar.slider("To-be WPR Plateau", 0.0, 1.0, 0.20),
        "weeks_to_increase": st.sidebar.number_input(
            "Weeks to To-Be Scenario", value=48
        ),
    }
)


# ARPU parameters
st.sidebar.header("ARPPU Parameters")
params.update(
    {
        "current_weekly_arpu": st.sidebar.number_input("Initial W1 ARPPU", value=28000),
        "final_weekly_arpu": st.sidebar.number_input("Final W1 ARPPU", value=28000),
        "weeks_to_increase_arpu": st.sidebar.number_input(
            "Weeks to To-Be Scenario (ARPPU)", value=48
        ),
    }
)

# Marketing parameters
st.sidebar.header("Marketing Parameters")
params.update(
    {
        "cac": st.sidebar.number_input("CAC", value=20000),
        "marketing_percentage": st.sidebar.number_input(
            "Marketing Percentage (GMV)", value=0.025, step=0.001, format="%.3f"
        ),
    }
)

# Simulation parameters
st.sidebar.header("Simulation Parameters")
params.update(
    {
        "weeks_to_simulate": st.sidebar.selectbox("Years to Simulate", [1, 2, 3, 4, 5])
        * 52,
    }
)
# Run simulation
results = run_simulation(params)

# Display plots in tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "WTU",
        "GMV",
        "Monthly GMV",
        "ARPU",
        "Marketing Cost",
        "CAC/LTV",
        "Retention Heatmap",
        "ARPU Heatmap",
    ]
)
with tab1:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results["dates"],
            y=results["weekly_wtu"],
            mode="lines",
            name="Total Weekly WTU",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results["dates"],
            y=results["weekly_new_wtu"],
            mode="lines",
            line=dict(dash="dash"),
            name="New Weekly WTU",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results["dates"],
            y=results["weekly_retained_wtu"],
            mode="lines",
            line=dict(dash="dot"),
            name="Retained Weekly WTU",
        )
    )
    fig.update_layout(
        title="WTU", xaxis_title="Date", yaxis_title="Users", hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    target_date = results["dates"][params["weeks_to_increase"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=results["dates"], y=results["weekly_gmv"], name="Weekly GMV")
    )
    fig.add_shape(
        type="line",
        x0=target_date,
        x1=target_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dash", color="red"),
    )
    fig.add_annotation(
        x=target_date, y=1, text="To-be Date", showarrow=False, yref="paper"
    )
    fig.update_layout(
        title="Weekly GMV", xaxis_title="Date", yaxis_title="GMV", hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    target_month = target_date.strftime("%Y-%m")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=results["monthly_gmv_df"]["Month"],
            y=results["monthly_gmv_df"]["GMV"],
            name="Monthly GMV",
        )
    )
    fig.add_shape(
        type="line",
        x0=target_month,
        x1=target_month,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dash", color="red"),
    )
    fig.add_annotation(
        x=target_month, y=1, text="To-be Date", showarrow=False, yref="paper"
    )
    fig.update_layout(
        title="Monthly GMV",
        xaxis_title="Month",
        yaxis_title="GMV",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
with tab4:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results["dates"],
            y=results["average_arpu"],
            mode="lines",
            name="Average ARPU",
        )
    )
    fig.update_layout(
        title="Average ARPU",
        xaxis_title="Date",
        yaxis_title="ARPU",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


with tab5:
    # Existing marketing cost plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results["dates"],
            y=results["marketing_cost"],
            mode="lines",
            name="Marketing Cost",
        )
    )
    fig.update_layout(
        title="Marketing Cost",
        xaxis_title="Date",
        yaxis_title="Cost",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calculate monthly marketing costs
    monthly_marketing_cost = {}
    for date, cost in zip(results["dates"], results["marketing_cost"]):
        month_key = date.strftime("%Y-%m")  # Format as 'YYYY-MM'
        monthly_marketing_cost[month_key] = (
            monthly_marketing_cost.get(month_key, 0) + cost
        )

    # Create and display the table
    monthly_cost_df = pd.DataFrame(
        [
            {"Year-Month": month, "Marketing Cost (원)": f"{cost:,.0f}"}
            for month, cost in monthly_marketing_cost.items()
        ]
    )
    monthly_cost_df = monthly_cost_df.sort_values("Year-Month")  # Sort by date

    st.subheader("Monthly Marketing Cost")
    st.dataframe(
        monthly_cost_df,
        hide_index=True,
        column_config={
            "Year-Month": st.column_config.TextColumn(
                "연도-월",
                width="large",
            ),
            "Marketing Cost (원)": st.column_config.TextColumn(
                "마케팅 비용",
                width="large",
            ),
        },
        use_container_width=True,
    )
# In the CAC/LTV plot section
with tab6:
    fig = go.Figure()

    # Plot Cohort LTV
    fig.add_trace(
        go.Scatter(
            x=results["dates"][: len(results["cohort_ltv"])],
            y=results["cohort_ltv"],
            mode="lines",
            name="Cohort LTV",
        )
    )

    # Calculate and plot CAC changes
    total_users = 0
    cac_values = []
    original_cac = params["cac"]
    current_cac = original_cac
    cac_list = results["cac_list"]

    # Add CAC trace
    fig.add_trace(
        go.Scatter(
            x=results["dates"],
            y=cac_list,
            mode="lines",
            name="CAC",
            line=dict(dash="dash", color="red"),
        )
    )

    fig.update_layout(
        title="CAC/LTV", xaxis_title="Date", yaxis_title="Value", hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


with tab7:

    def plot_retention_heatmap(retention_values, weekly_new_wtu, dates):
        fig, ax = plt.subplots(
            1,
            3,
            figsize=(20, 10),
            sharey=True,
            gridspec_kw={"width_ratios": [0.5, 1, 11]},
        )

        # dates
        white_cmap = mcolors.ListedColormap(["white"])
        dates_df = pd.DataFrame({"dates": np.zeros(len(dates))}, index=dates)
        formatted_dates = [[d.strftime("%Y-%m-%d")] for d in dates_df.index]
        sns.heatmap(
            dates_df[:50],
            annot=formatted_dates[:50],
            cbar=False,
            fmt="",
            cmap=white_cmap,
            ax=ax[0],
            annot_kws={"size": 8},
        )
        ax[0].set_ylabel("")

        # cohort sizes
        cohort_size_df = pd.DataFrame({"cohort_size": weekly_new_wtu}, index=dates)
        sns.heatmap(
            cohort_size_df[:50],
            annot=True,
            cbar=False,
            fmt=".0f",
            cmap=white_cmap,
            ax=ax[1],
            annot_kws={"size": 8},
        )
        ax[1].set_ylabel("")

        # retention matrix
        retention_df = pd.DataFrame(retention_values, index=dates)
        sns.heatmap(
            retention_df.iloc[:50, :50],
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            ax=ax[2],
            annot_kws={"size": 8},
        )

        plt.suptitle("Modeled User Retention Weekly Cohorts", fontsize=16)
        ax[2].set(xlabel="# of periods", ylabel="")
        fig.text(0.08, 0.5, "Cohort", va="center", rotation="vertical")

        return fig

    retention_values = results["retention_values"]
    retention_fig = plot_retention_heatmap(
        retention_values, results["weekly_new_wtu"], results["dates"]
    )
    st.pyplot(retention_fig)

with tab8:

    def plot_arpu_heatmap(arpu_values, weekly_new_wtu, dates):
        fig, ax = plt.subplots(
            1,
            3,
            figsize=(20, 10),
            sharey=True,
            gridspec_kw={"width_ratios": [0.5, 1, 11]},
        )

        # dates
        white_cmap = mcolors.ListedColormap(["white"])
        dates_df = pd.DataFrame({"dates": np.zeros(len(dates))}, index=dates)
        formatted_dates = [[d.strftime("%Y-%m-%d")] for d in dates_df.index]
        sns.heatmap(
            dates_df[:50],
            annot=formatted_dates[:50],
            cbar=False,
            fmt="",
            cmap=white_cmap,
            ax=ax[0],
            annot_kws={"size": 8},
        )
        ax[0].set_ylabel("")

        # cohort sizes
        cohort_size_df = pd.DataFrame({"cohort_size": weekly_new_wtu}, index=dates)
        sns.heatmap(
            cohort_size_df[:50],
            annot=True,
            cbar=False,
            fmt=".0f",
            cmap=white_cmap,
            ax=ax[1],
            annot_kws={"size": 8},
        )
        ax[1].set_ylabel("")

        # ARPU matrix
        arpu_df = pd.DataFrame(arpu_values, index=dates)
        sns.heatmap(
            arpu_df.iloc[:50, :50],
            annot=True,
            fmt=".0f",
            cmap="RdYlGn",
            ax=ax[2],
            annot_kws={"size": 8},
        )

        plt.suptitle("Modeled User ARPU Weekly Cohorts", fontsize=16)
        ax[2].set(xlabel="# of periods", ylabel="")
        fig.text(0.08, 0.5, "Cohort", va="center", rotation="vertical")

        return fig

    arpu_values = results["arpu_values"]
    arpu_curves = results["arpu_curves"]
    arpu_fig = plot_arpu_heatmap(
        arpu_curves, results["weekly_new_wtu"], results["dates"]
    )
    st.pyplot(arpu_fig)
# Add metrics
col1 = st.columns(1)
st.metric(
    "Total Acquired Transacting Users", f"{sum(results['weekly_new_wtu']):,.0f}명"
)
st.metric("Total Marketing Cost", f"{sum(results['marketing_cost']):,.0f}원")

# with col1:
