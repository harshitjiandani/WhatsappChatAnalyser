import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
from collections import Counter
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


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


def activity_analysis(df):
    # Group by 'Sender' and 'Date' for messages per day
    activity_by_day = df.groupby(['Sender', df['Date'].dt.date]).size().unstack(fill_value=0)

    # Group by 'Sender' and 'Hour' for messages per hour of the day
    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    activity_by_hour = df.groupby(['Sender', 'Hour']).size().unstack(fill_value=0)

    # Group by 'Sender' and 'Weekday' for messages per day of the week
    df['Weekday'] = df['Date'].dt.day_name()
    activity_by_weekday = df.groupby(['Sender', 'Weekday']).size().unstack(fill_value=0).reindex(
        columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fill_value=0)

    return activity_by_day, activity_by_hour, activity_by_weekday



def calculate_average_response_time(df):
    """Calculate the average response time between messages from different senders, attributed to the responder."""
    # Combine Date and Time into a single datetime column and sort the DataFrame
    df['Datetime'] = df.apply(lambda row: datetime.combine(row['Date'], row['Time']), axis=1)
    df = df.sort_values(by='Datetime')
    
    response_times = []
    prev_sender = None
    prev_time = None

    # Iterate through each message to calculate response times between different senders
    for _, row in df.iterrows():
        if prev_sender and prev_sender != row['Sender']:
            response_time = row['Datetime'] - prev_time
            response_times.append((prev_sender, row['Sender'], prev_time, row['Datetime'], response_time))

        prev_sender = row['Sender']
        prev_time = row['Datetime']

    # Create a DataFrame with detailed response times
    response_df = pd.DataFrame(response_times, columns=['From_Sender', 'To_Sender', 'From_Time', 'To_Time', 'Response_Time'])
    
    # Calculate average response times attributed to the responder
    avg_response_time = response_df.groupby(['To_Sender', 'From_Sender'])['Response_Time'].mean().reset_index()
    avg_response_time.rename(columns={'Response_Time': 'Average_Time'}, inplace=True)

    # Add columns for response time in minutes
    avg_response_time['Average_Time_Minutes'] = avg_response_time['Average_Time'].dt.total_seconds() / 60  # Convert to minutes

    return avg_response_time, response_df

def visualize_monthly_response_times(detailed_responses):
    """Generate a bar plot comparing monthly average response times between users."""
    # Add a month column for grouping
    detailed_responses['Month'] = detailed_responses['To_Time'].dt.to_period('M')

    # Calculate monthly average response times
    monthly_avg = detailed_responses.groupby(['To_Sender', 'From_Sender', 'Month'])['Response_Time'].mean().reset_index()
    monthly_avg['Response_Time_Minutes'] = monthly_avg['Response_Time'].dt.total_seconds() / 60  # Convert to minutes

    # Plot monthly comparison
    plt.figure(figsize=(12, 6))
    for sender in monthly_avg['To_Sender'].unique():
        sender_data = monthly_avg[monthly_avg['To_Sender'] == sender]
        plt.plot(sender_data['Month'].astype(str), sender_data['Response_Time_Minutes'], marker='o', label=f'Responder: {sender}')
    
    plt.title('Monthly Average Response Times Between Users')
    plt.xlabel('Month')
    plt.ylabel('Average Response Time (Minutes)')
    plt.xticks(rotation=45)
    plt.legend(title='Responder')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def print_average_response_summary(avg_response_time):
    """Print a summary of average response times between users."""
    for _, row in avg_response_time.iterrows():
        print(f"On average, {row['To_Sender']} takes {row['Average_Time_Minutes']:.2f} minutes to respond to {row['From_Sender']}.")



def plot_user_comparison(avg_response_time):
    """Plot a bar chart comparing the average response times between users."""
    # Summarize average response times for each responder
    avg_times = avg_response_time.groupby('To_Sender')['Average_Time_Minutes'].mean().reset_index()

    # Plot bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(avg_times['To_Sender'], avg_times['Average_Time_Minutes'], color=['#4CAF50', '#2196F3'])
    plt.title('Comparison of Average Response Times Between Users')
    plt.xlabel('User')
    plt.ylabel('Average Response Time (Minutes)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # Plot overall histogram

def plot_activity(activity_by_day, activity_by_hour, activity_by_weekday):
    sns.set(style="whitegrid", palette="muted")

    # Plotting activity by day for each user (Messages per day)
    # Plotting activity by day for each user
    plt.figure(figsize=(14, 6))
    for user in activity_by_day.index:
        plt.plot(activity_by_day.columns, activity_by_day.loc[user], label=user)
    plt.title('Messages per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.legend(title='User')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting activity by hour of the day for each user (Messages per hour of the day)
    plt.figure(figsize=(14, 6))
    for user in activity_by_hour.index:
        plt.plot(activity_by_hour.columns, activity_by_hour.loc[user], marker='o', label=user)
    plt.title('Messages per Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Messages')
    plt.xticks(ticks=range(0, 24), labels=[f"{hour}:00" for hour in range(24)])
    plt.legend(title='User')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Plotting activity by weekday for each user (Messages per day of the week)
    plt.figure(figsize=(14, 6))
    for user in activity_by_weekday.index:
        plt.plot(activity_by_weekday.columns, activity_by_weekday.loc[user], marker='o', label=user)
    plt.title('Messages per Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='User')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def main():
    """Run the main program."""
    # Load chat data
    parser = argparse.ArgumentParser(description='Chat Analysis Tool')
    parser.add_argument('file_path',default='./chat2.txt' , type=str, help='Path to the chat file' )
    args = parser.parse_args()
    chat_data = load_chat(args.file_path)
    df = process_chat(chat_data)




    # Conversation Dynamics
    senders_count, most_active_sender, avg_msg_length = conversation_dynamics(df)




    avg_response_time, detailed_responses = calculate_average_response_time(df)


    print(f"Chat Analysis:\n"
        f"{len(senders_count)} Participants in the Chat\n")


    person1 , person2 = senders_count.keys()

    numb_mssgs1 , numb_mssgs2 = senders_count.values()

    total_messages = sum(senders_count.values())

    avg_msg_length1 , avg_msg_length2 = avg_msg_length.values

    avg_response_time1 , avg_response_time2 = avg_response_time.values[0][-1] , avg_response_time.values[1][-1]

    ppl = {
        person1: (numb_mssgs1, avg_msg_length1, avg_response_time1),
        person2: (numb_mssgs2, avg_msg_length2, avg_response_time2)
    }

    print(f"Total of {sum(senders_count.values())} messages")
    for i, j in ppl.items():
        print(f" {i} : \n Messages: {j[0]}  \n Avg mssg length  {int(j[1])} \n Avg Response Time:{j[2]} \n ({round(j[0]/total_messages*100, 2)}% of total messages) \n")
        

    activity_by_day, activity_by_hour, activity_by_weekday  =activity_analysis(df)
    plot_activity(activity_by_day ,activity_by_hour , activity_by_weekday)
    plot_user_comparison(avg_response_time)
    visualize_monthly_response_times(detailed_responses)

if __name__ == '__main__':
    main()
