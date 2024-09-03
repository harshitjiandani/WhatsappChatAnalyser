# WhatsappChatAnalyser Tool

This project provides a tool for analyzing chat logs. It extracts key insights, including conversation dynamics, activity patterns, and response times, and visualizes the data in a comprehensive report.

## Features

- **Load Chat Data**: Read chat logs from a text file.
- **Process Chat Data**: Parse and structure chat messages into a DataFrame.
- **Conversation Dynamics**: Evaluate message frequency, average message length, and identify the most active participants.
- **Activity Analysis**: Visualize chat activity over days, hours, and weekdays.
- **Response Time Analysis**: Calculate and compare average response times between participants.
- **Visualization**: Generate plots for message activity, response times, and participant comparisons.

## Prerequisites

- Python 3.x
- Required Python libraries: `pandas`, `matplotlib`, `seaborn`, `textblob`, `fpdf`, `argparse`

Install the required libraries using pip:

```bash
pip install pandas matplotlib seaborn textblob fpdf
```
##Usage
Prepare the Chat File: Ensure your chat file is formatted correctly. The expected format is:

```
01/01/2024, 10:00 am - Sender: Message text
```

Run the Script: Execute the script from the command line with the path to your chat file.
```
python chat_analysis.py path/to/your/chatfile.txt

```

