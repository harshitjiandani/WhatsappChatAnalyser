import re
from collections import Counter
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

def load_chat(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        chat_data = file.readlines()
    return chat_data

def process_chat(chat_data):
    message_pattern = r'(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2})[\u202f]?([ap]m) - (.*?): (.*)'
    dates, times, senders, messages = [], [], [], []

    for line in chat_data:
        match = re.match(message_pattern, line)
        if match:
            date_str, time_str, period, sender, message = match.groups()
            time_str = f"{time_str} {period}" 
            dates.append(datetime.strptime(date_str, "%d/%m/%Y"))
            times.append(datetime.strptime(time_str, "%I:%M %p").time())
            senders.append(sender)
            messages.append(message)

    df = pd.DataFrame({
        "Date": dates,
        "Time": times,
        "Sender": senders,
        "Message": messages
    })
    return df

def sentiment_analysis(df):
    df['Polarity'] = df['Message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['Sentiment'] = df['Polarity'].apply(lambda p: 'Positive' if p > 0 else ('Negative' if p < 0 else 'Neutral'))
    return df

def analyze_tone(df):
    tone_count = df.groupby('Sender')['Sentiment'].value_counts().unstack().fillna(0)
    predominant_tones = tone_count.idxmax(axis=1)
    return tone_count, predominant_tones

def conversation_dynamics(df):
    senders_count = Counter(df['Sender'])
    total_messages = df.shape[0]
    most_active_sender = senders_count.most_common(1)[0]
    average_message_length = df.groupby('Sender')['Message'].apply(lambda x: x.str.len().mean())
    return senders_count, most_active_sender, average_message_length

def dry_replies(df, average_message_length):
    driest_sender = average_message_length.idxmin()
    return driest_sender

def activity_analysis(df):
    activity_by_day = df.groupby(df['Date'].dt.date).size()

    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    activity_by_hour = df.groupby('Hour').size()

    df['Weekday'] = df['Date'].dt.day_name()
    activity_by_weekday = df.groupby('Weekday').size().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    return activity_by_day, activity_by_hour, activity_by_weekday

def calculate_average_response_time(df):
    """Calculate the average response time between messages from different senders."""
    df['Datetime'] = df.apply(lambda row: datetime.combine(row['Date'], row['Time']), axis=1)
    df = df.sort_values(by='Datetime')
    
    response_times = []
    prev_sender = None
    prev_time = None

    for _, row in df.iterrows():
        if prev_sender and prev_sender != row['Sender']:
            response_time = row['Datetime'] - prev_time
            response_times.append((prev_sender, row['Sender'], response_time))

        prev_sender = row['Sender']
        prev_time = row['Datetime']

    response_df = pd.DataFrame(response_times, columns=['From_Sender', 'To_Sender', 'Response_Time'])
    avg_response_time = response_df.groupby(['From_Sender', 'To_Sender'])['Response_Time'].mean().reset_index()
    
    return avg_response_time

def plot_activity(activity_by_day, activity_by_hour, activity_by_weekday):
    sns.set(style="whitegrid")

    # Plotting the activity by day
    plt.figure(figsize=(14, 6))
    activity_by_day.plot(kind='line', color='blue')
    plt.title('Messages per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig('activity_by_day.png', bbox_inches='tight')
    plt.close()

    # Plotting the activity by hour
    plt.figure(figsize=(10, 6))
    sns.barplot(x=activity_by_hour.index, y=activity_by_hour.values, palette="viridis")
    plt.title('Messages per Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=0)
    plt.savefig('activity_by_hour.png', bbox_inches='tight')
    plt.close()

    # Plotting the activity by weekday
    plt.figure(figsize=(10, 6))
    sns.barplot(x=activity_by_weekday.index, y=activity_by_weekday.values, palette="magma")
    plt.title('Messages per Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.savefig('activity_by_weekday.png', bbox_inches='tight')
    plt.close()

def export_to_pdf(file_path, results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Chat Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Activity Analysis Visuals
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Activity Analysis:", ln=True)
    pdf.set_font("Arial", size=12)
    
    pdf.image('activity_by_hour.png', x=10, y=None, w=190)
    pdf.ln(10)
    pdf.image('activity_by_weekday.png', x=10, y=None, w=190)
    pdf.ln(10)
    pdf.image('activity_by_day.png', x=10, y=None, w=190)
    pdf.ln(10)

    # Conversation Insights
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Conversation Insights:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, results['conversation_dynamics'])
    pdf.ln(10)

    # Average Response Time
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Average Response Time Between Different Senders:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, results['average_response_time'])
    pdf.ln(10)

    # Save PDF
    pdf.output(file_path)

def main():
    chat_data = load_chat(r"C:\Users\DELL\Desktop\chat2.txt")
    df = process_chat(chat_data)
    
    # Sentiment Analysis
    df = sentiment_analysis(df)
    
    # Tone Analysis
    tone_count, predominant_tones = analyze_tone(df)
    
    # Conversation Dynamics
    senders_count, most_active_sender, avg_msg_length = conversation_dynamics(df)
    conversation_dynamics_result = (
        f"Total Messages Sent by Each User:\n{senders_count}\n\n"
        f"Average Message Lengths by User:\n{avg_msg_length}\n\n"
        f"Most Active Sender: {most_active_sender[0]} with {most_active_sender[1]} messages"
    )
    
    # Activity Analysis
    activity_by_day, activity_by_hour, activity_by_weekday = activity_analysis(df)
    plot_activity(activity_by_day, activity_by_hour, activity_by_weekday)

    # Average Response Time
    avg_response_time_df = calculate_average_response_time(df)
    average_response_time_result = avg_response_time_df.to_string(index=False)

    # Export to PDF
    pdf_file_path = "chat_analysis_report.pdf"
    results = {
        'conversation_dynamics': conversation_dynamics_result,
        'average_response_time': average_response_time_result
    }
    export_to_pdf(pdf_file_path, results)

    print(f"Report saved as {pdf_file_path}")

if __name__ == "__main__":
    main()