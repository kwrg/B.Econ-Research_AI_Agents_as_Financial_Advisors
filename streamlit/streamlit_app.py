import time
import re
from crewai_tools import tool
import pandas as pd
from IPython.display import Markdown
import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai import LLM


load_dotenv('')
gpt = LLM(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.0,
    top_p=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# FA tool


class CSVFilterTool:
    def __init__(self, csv_file_path="csv"):
        self.csv_file_path = csv_file_path

    def filter_by_month_and_year(self, month: int, year: int):
        df = pd.read_csv(self.csv_file_path, parse_dates=['date'])

        filtered_df = df[(df['date'].dt.year == year) &
                         (df['date'].dt.month == month)]

        filtered_data = filtered_df[['news_headline', 'stock_symbol']]

        return filtered_data


def filter_2(filtered_data):
    """
    After filtering the news headlines for a specific month and year, organize the data to be stored in a structured format.

    Args:
        filtered_data (pd.DataFrame): A Data containing the filtered news headlines and stock symbols.

    Returns:
        str: A Data containing the filtered news headlines and stock symbols for a specific month and year in a structured format. 
    """
    results = {}
    for stock_symbol, group in filtered_data.groupby('stock_symbol'):
        num_headlines = len(group)
        titles_combined = ",".join(group['news_headline'].tolist())
        results[stock_symbol] = {
            'count': num_headlines,
            'titles': titles_combined
        }
    result2 = ""
    for stock_symbol, result in results.items():
        result2 += f"Stock: {stock_symbol}\nNumber of News Headlines: {result['count']}\nNews Headlines: {result['titles']}\n\n"
    return result2


@tool
def load_data_for_month(month: int, year: int) -> str:
    """
    Load a filtered news headlines in a structured format that associated stock symbols from a CSV file for a specific month and year.

    Args:
        month (int): The month to filter by (1-12).
        year (int): The year to filter by.

    Returns:
        str: A string representation of the filtered news headlines in a structured format for a specific month and year.
    """

    csv_tool = CSVFilterTool(
        "csv")

    filtered_data = csv_tool.filter_by_month_and_year(month, year)

    filter_result = filter_2(filtered_data)

    return filter_result


# PM tool
@tool
def filter_and_average_stocks(stocks: str, month: int, year: int) -> str:
    """
    Filters a stock dataset by specified stocks, month, and year, and calculates 
    the average values (close price, PE, dividend) for each stock.

    Args:
        stocks (str): Comma-separated stock names to filter, e.g. 'stock a', 'stock b', 'stock c'.
        month (int): Month to filter by (1-12).
        year (int): Year to filter by.

    Returns:
        str: A summary of average values for each stock in text format.
    """
    try:
        stock_list = [stock.strip() for stock in stocks.split(",")]

        dataframe = pd.read_csv(
            "csv"
        )
        dataframe['date'] = pd.to_datetime(
            dataframe['date'], format='%m/%d/%Y')

        if month:
            dataframe = dataframe[dataframe['date'].dt.month == month]
        if year:
            dataframe = dataframe[dataframe['date'].dt.year == year]

        if dataframe.empty:
            return "No data found for the specified filters."

        result = []

        for stock in stock_list:
            stock_pattern = rf"^{stock}_"
            stock_columns = [
                col for col in dataframe.columns if re.match(stock_pattern, col)
            ]

            if not stock_columns:
                result.append(f"Stock {stock} not found in the dataset.")
                continue

            stock_df = dataframe[['date'] + stock_columns]

            averages = {}
            for col in stock_columns:
                averages[col] = stock_df[col].mean()

            result.append(f"Stock:  {stock}")
            for col, avg in averages.items():
                avg_value = round(avg, 2) if not pd.isna(avg) else "No data"

                metric_name = col[len(stock) + 1:]
                result.append(f"Average {metric_name}: {avg_value}")
            result.append("")

        return "\n".join(result)

    except Exception as e:
        return f"Error filtering stocks and calculating averages: {str(e)}"


# FA agent

financial_analyst_agent = Agent(
    role='Financial Analyst',
    goal="""
    - Using extensive experience in the Thai financial market, analyze news headlines about Thai stocks in the Thai financial market from the data gathered using the tool: [load_data_for_month]. The goal is to identify investment opportunities based on the most positive sentiment expressed in these headlines. Identify the top 10 stocks to consider for investment next month, as this month is {month}.
    """,
    verbose=True,
    backstory="""
    - Specialist with extensive experience in the Thai financial market.
    - Expert with a deep understanding of financial knowledge and analytical skills in the Thai financial market.
    - Experienced in analyzing stock news headlines to assess sentiment and recommend top-performing stocks.
    - Expert in the Thai language, enabling accurate interpretation and analysis of news headlines.
""",
    tools=[load_data_for_month],
    llm=gpt, memory=False
)


# FA task
financial_analyst_task = Task(
    description="""
    - Analyze news headlines to identify investment opportunities by selecting the top 10 stocks with the most positive sentiment in their headlines from the data gathered using the tool: [load_data_for_month]. This data will provide stock news headlines for the month of {month} and the year {year}. Utilize your financial knowledge and analytical skills in the Thai financial market to gain insights into these investment opportunities.
    - Example of data details:
        Stock: Stock A → Name of the stock.
        Number of News Headlines: n → The number of news headlines related to the stock mentioned above.
        News Headlines: → A list of news headlines related to the stock mentioned above (from latest to oldest within the following month and year).
    """,

    expected_output="""
    - A report on the top 10 stocks to consider for investment next month, based on an analysis of stock news headlines from the data gathered using the tool: [load_data_for_month]. The selection is focused on identifying the top stocks with the most positive sentiment in their headlines. 
    - The report must include 3 things:
        1. Reason for Selection: An explanation of why each of the top 10 stocks was chosen and provide the insights derived from the news headlines for each selected stock, emphasizing the positive sentiment.
        2. Conclusion: A list of the selected top 10 stocks, formatted exactly as 'Stock 1', 'Stock 2'.
        - Report format:
            1. Stock 1
                Reason for Selection: [Explanation]
            2. Stock 2
                Reason for Selection: [Explanation]
            Conclusion: 'Stock 1', 'Stock 2'

""",
    agent=financial_analyst_agent
)


# PM agent
portfolio_manager_agent = Agent(
    role='Portfolio Manager',
    goal="""
    - Utilize extensive financial market experience and statistical skills to analyze historical stock data of the stock list provided by the financial_analyst_agent and gather the data by using the tool: [filter_and_average_stocks]. Analyze the numeric metris like average stock price, average stock P/E ratio, and average stock dividend yield from the data of stocks in the specified month: {month} and year: {year}, as gathered by the tool, to categorize the 5 stocks into the following investment styles: {investment_styles}.
    """,
    verbose=True,
    backstory="""
    - Specialist with extensive experience in the Thai financial market.
    - Expert with a deep understanding of financial knowledge and analytical skills in the Thai financial market.
    - Specialist in statistics, especially in finance.
""",
    llm=gpt, memory=False,
    tools=[filter_and_average_stocks]


)


# PM task
portfolio_manager_task = Task(
    description="""
    - Analyze the provided historical stock data (average stock price, average stock P/E ratio, and average stock dividend yield) in the specified month: {month} and year: {year} and use your expertise to determine the best methods for analysis.
    - Your goal is to categorize 10 stocks from financial_analyst_agent into 5 stocks in the specified investment styles based on insights derived from the data.
    - Clearly explain your approach, findings, and rationale for categorization.
    - Think carefully about what aspects of the data are most relevant to categorizing stocks into investment styles. 
    - You need to use the tool [filter_and_average_stocks] to retrieve data and analyze it.
    - Example of data details:
        Stock: Stock A → Name of the stock.
        Average close → The average closing price of the stock mentioned above for the month of {month} and year {year}.
        Average pe → The average Price-to-Earnings P(/E) ratio of the stock mentioned above for the month of {month} and year {year}.
        Average dividendYield → The average dividend yield of the stock mentioned above for the month of {month} and year {year}.
    """,

    expected_output="""
    - A report on the categorization of 5 stocks into investment styles: {investment_styles}, selected from the 10 stock list provided by the financial_analyst_agent, categorized the stock into investment styles based on your analysis of historical stock data gathered by using the tool: [filter_and_average_stocks].

    - The report must include 3 things:
        1. Investment Styles Explanation
        2. Reason for Selection: An explanation of your analysis, detailing why each of the 5 stocks was categorized into the investment styles: {investment_styles}. This explanation should include an analysis of the stock data gathered by the tool: [filter_and_average_stocks], with numeric metrics such as average stock price, average stock P/E ratio, and average stock dividend yield, to support the rationale for each categorization. The explanation should provide specific statistical quantities to justify the categorization.
        
        3. Conclusion in format of 'Stock 1', 'Stock 2'

        - Report format:
            1.  Investment Styles Explanation
            2.  Stock 1
                - Reason for Selection
            3. Conclusion: 'Stock 1', 'Stock 2'
""",
    agent=portfolio_manager_agent
)


# Crew_inv
crew_inv_style = Crew(
    agents=[financial_analyst_agent, portfolio_manager_agent],
    tasks=[financial_analyst_task, portfolio_manager_task],
    process=Process.sequential
)


def run_crew_inv_style(year, month, inv_style):
    result = crew_inv_style.kickoff(
        inputs={'year': year, 'month': month, 'investment_styles': inv_style})
    st.markdown(result)


# RA tool

@tool
def gather_stock_details(stocks: str) -> str:
    """
    Gather stock details (Stocks Names, Annual Return and Annual Standard Deviation) for the given stock names.

    Args:
    stocks (str): Comma-separated stock names as input, e.g. 'stock a', 'stock b', 'stock c'

    Returns:
    str: The details (Stocks Names, Annual Return and Annual Standard Deviation) of the stocks.
    """
    data = pd.read_csv(
        "csv")
    stock_list = [stock.strip() for stock in stocks.split(',')]
    print(stock_list)

    filtered_data = data[data['Stocks Names'].isin(stock_list)]

    if filtered_data.empty:
        return "No data found for the provided stock names."

    result = ""
    for index, row in filtered_data.iterrows():
        result += (
            f"{row['Stocks Names']}, "
            f"Annual Return: {row['Annual Return']}, "
            f"Annual Standard Deviation: {row['Annual Standard Deviation']}\n"
        )

    return result


# RA agent
risk_advisor_agent = Agent(
    role='Risk Advisor',
    goal="""
    - Utilizing extensive financial market experience and statistical skills to analyze historical stock data from the stock list provided by the financial_analyst_agent using the tool: [gather_stock_details]. Leverage the Average Annual Return and Annual Standard Deviation of stocks, as gathered by the tool, to identify the top 5 stocks with the {level} risk level.
    """,
    verbose=True,
    backstory="""
    - Specialist with extensive experience in the Thai financial market.
    - Expert with a deep understanding of financial knowledge and analytical skills in the Thai financial market.
    - Specialist in statistics, especially in finance.
""",
    llm=gpt, memory=False,
    tools=[gather_stock_details]


)
# RA task
risk_advisor_task = Task(
    description="""
    - Analyze the historical stock data of the stock list provided by the financial_analyst_agent. Using the provided Average Annual Return and Annual Standard Deviation and your experience in the financial market and statistical skills, assess the risk of the stock list provided by the financial_analyst_agent, and identify the top 5 stocks with the most {level} of risk for investment. 
    - Example of data details:
        1. The "Stock Names" contains the names of each stock.
        2. The "Average Annual Return" contains the average annual return for each stock based on its historical stock prices from 3/3/2023 to 9/29/2023.
        3. The "Annual Standard Deviation" contains the annual standard deviation for each stock based on its historical stock prices from 3/3/2023 to 9/29/2023.
        - Some stock names might be similar to each other, so it is important to be cautious and ensure that you query or use the correct information for each stock.
    - Utilize your financial knowledge, analytical skills, and statistical expertise in the financial market to analyze the historical stock data of the stock list provided by the financial_analyst_agent. Identify the top 5 stocks with the most {level} risk.
    """,

    expected_output="""
    - A report on the top 5 stocks selecting from the stock list provided by the financial_analyst_agent to consider as having the most {level} risk, based on an analysis of historical stock data from the data gathered using the tool.
    - The report must include 2 things:
        1. Reason for Selection: An in-depth explanation detailing why each of the top 5 stocks was chosen as having the most {level} risk, using data from the data gathered using the tool to like provided Average Annual Return and Annual Standard Deviation to support your explanation. 
        2. Conclusion: A list of the selected top 5 stocks with the most {level} risk, formatted exactly as 'Stock 1', 'Stock 2'.
        - Report format:
            1. Stock 1
                Reason for Selection: [An in-depth explanation]
            2. Stock 2
                Reason for Selection: [An in-depth explanation]
            Conclusion: 'Stock 1', 'Stock 2'
""",
    agent=risk_advisor_agent
)

# Risk crew
crew_risk = Crew(
    agents=[financial_analyst_agent, risk_advisor_agent],
    tasks=[financial_analyst_task, risk_advisor_task],
    process=Process.sequential
)


def run_crew_risk(year, month, risk_level):
    result = crew_risk.kickoff(
        inputs={'year': year, 'month': month, 'level': risk_level})
    st.markdown(result)


st.title("Financial Advisor AI Agents")
st.image("png",
         caption="Image generated by DALL-E 3")

st.subheader("Please fill out the following information:")
st.write("**Disclaimer**: News can be used starting from October 2023 and ending in October 2024, which means stock recommendations will cover the period from November 2023 to November 2024.")

st.divider()
st.markdown(
    "**1. Enter the month and year to receive stock recommendations for that period.**")
st.markdown(
    "*(This trial of monthly stock selection can only be used for the period from 11/2023 to 11/2024)*")


col1, col2 = st.columns(2)
with col1:
    month = st.number_input(
        "Insert a month", placeholder="Type a number...", step=1, min_value=1, max_value=12, value=11,
    )


with col2:
    year = st.number_input(
        "Insert a year", placeholder="Type a number...", step=1, min_value=2023, max_value=2025, value=2023,
    )

st.write("- Your selected period: ", month, '/', year)
if month == 1:
    month = 12
    year = year - 1
else:
    month = month - 1


st.write('- The period of the news reading: ', month, '/', year)
st.divider()

st.markdown(
    "**2. Select how you would like to customize your investment portfolio**")

option = st.selectbox(
    "How would you like to customize your investment portfolio?",
    ("Risk Tolerance", "Investment Style"),
    index=None,
    placeholder="Select customize method..."
)

st.write("- Your preferred customize method:", option)


if option == "Investment Style":
    inv_style = st.selectbox(
        "Select your preferred investment style:",
        ("Growth Investing Style", "Income Investing Style"),
        index=None,
        placeholder="Select preferred investment style..."
    )
    if inv_style:
        st.write('- Your preferred investment style:', inv_style)


elif option == "Risk Tolerance":
    risk_level = st.selectbox(
        "Select your preferred risk level:",
        ("High Risk", "Low Risk"),
        index=None,
        placeholder="Select preferred risk level..."
    )
    if risk_level:
        st.write("- Your preferred risk level: ", risk_level)


if st.button("Enter", use_container_width=True):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

    if option == "Investment Style":
        st.divider()
        st.write("### The result of your customized investment style: ", inv_style)

        result = run_crew_inv_style(year, month, inv_style)

    else:
        st.divider()
        st.write("### The result of your customized risk level: ", risk_level)
        result = run_crew_risk(year, month, risk_level)
